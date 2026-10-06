import Foundation
import Darwin
import Metal
import os

/// Cross-thread one-shot "please stop" flag for the replay loop. The SIGINT
/// `DispatchSource` handler (running on a global queue) flips it; the training
/// loop (running on the detached replay task) reads it once per step. An
/// `OSAllocatedUnfairLock` — the project standard — guards the single Bool;
/// this is not on any hot path (one read per GPU step), so the lock cost is
/// irrelevant. Internal (not private) so the resume-equivalence harness can
/// run the real replay loop in-process without a signal source.
final class ReplayAbortFlag: @unchecked Sendable {
    private let state = OSAllocatedUnfairLock(initialState: false)
    func request() { state.withLock { $0 = true } }
    var isRequested: Bool { state.withLock { $0 } }
}

/// Plain, `Sendable` snapshot of the training parameters an offline run
/// (corpus replay or train-vs-UCI) needs. Captured on the main actor (from
/// `TrainingParameters.shared`) before the run starts, then passed into the
/// off-actor GPU work — so the detached task never touches the `@MainActor`
/// singleton (which would deadlock against the `syncWait` semaphore held on
/// the main thread).
///
/// Every trainer-level parameter travels as one `TrainerHyperparameters`,
/// resolved and applied exactly as the GUI session does it. This struct used
/// to carry its own hand-picked copy of those fields, which is how the CLI
/// runners came to train without the LR/momentum cycle, the stats interval
/// and the KL-probe interval. Only the run-level knobs that are not trainer
/// state are listed separately.
///
/// **One source: `parameters`.** Every other field is derived from it in
/// `init` and is immutable, so the parameters a run trains under and the
/// snapshot its lineage records can never disagree. (When the fields were
/// independently mutable, a caller could change `trainingBatchSize` without
/// changing `lineageParameters`, and the files said one batch size while the
/// trainer stepped at another.) A changed value means a new snapshot and a
/// new `ReplayParams`; `adoptingSchedule(_:)` is the one such change a run
/// makes.
struct ReplayParams: Sendable {
    /// The snapshot every field below is derived from — what the run's
    /// lineage records and a train-vs-UCI session's `session.json` records
    /// its settings from.
    let parameters: TrainingParametersSnapshot
    let trainer: TrainerHyperparameters
    let trainingBatchSize: Int
    let replayBufferCapacity: Int
    let replayRatioTarget: Double
    let replayBufferMinPositionsBeforeTraining: Int
    /// The batch-composition constraints the replay buffer samples under,
    /// built by the same rule the GUI's self-play buffer uses.
    let samplingConstraints: ReplayBuffer.SamplingConstraints
    /// The complete parameter set, as every lineage record of the run
    /// carries it.
    let lineageParameters: LineageRecord.Parameters

    init(_ parameters: TrainingParametersSnapshot) throws {
        self.parameters = parameters
        trainer = TrainerHyperparameters(parameters)
        lineageParameters = try LineageRecord.Parameters(values: parameters.rawValueMap())
        trainingBatchSize = parameters.trainingBatchSize
        replayBufferCapacity = parameters.replayBufferCapacity
        replayRatioTarget = parameters.replayRatioTarget
        replayBufferMinPositionsBeforeTraining = parameters.replayBufferMinPositionsBeforeTraining
        samplingConstraints = ReplayBuffer.SamplingConstraints(parameters)
    }

    /// These parameters with a resumed checkpoint's schedule in force: the
    /// warmup length and LR/momentum cycle it trained under, whatever the
    /// run was configured with. Rebuilt from one adopted snapshot, so the
    /// trainer is configured with — and every lineage record of the run
    /// records — the schedule the file's flat `trainer_*` keys are written
    /// from. A CLI trainer cannot change its schedule during a run, so this
    /// one adoption at the start describes every save.
    func adoptingSchedule(_ schedule: TrainerScheduleState) throws -> ReplayParams {
        try ReplayParams(parameters.adoptingSchedule(schedule))
    }
}

/// Configuration for one offline corpus-replay run.
struct CorpusReplayConfig: Sendable {
    var corpusDirectories: [URL]
    var stepLimit: Int?
    var epochs: Int?
    var startModelPath: String?
    /// Optional built-in preset name (`NetworkArchitecture.Preset` rawValue, e.g.
    /// `v3_8block_3x3`) for a fresh-init run. Used only when `startModelPath` is
    /// nil; selects that architecture instead of `NetworkArchitecture.current`.
    var presetName: String?
    /// Resume the corpus stream at shard sequence `startShard` (0-based — the
    /// `NNNNN` in `shard-NNNNN.dcmgames`): skip shards `0…startShard-1` on the
    /// first pass, full coverage after the epoch wrap. For warm-start runs that
    /// should pick up near where a prior run left off. Mutually exclusive with
    /// `startGameIndex`; out-of-range is a hard error.
    var startShard: Int?
    /// Resume at a global within-epoch game index (0-based, counts skipped
    /// games — matches the `games=`/`nextGame=` log counters). Resolved against
    /// the per-shard game counts into a `(shard, within-shard offset)` start.
    /// Mutually exclusive with `startShard`.
    var startGameIndex: Int?
    /// Exact resume: continue training exactly as if the `--start-model`'s run
    /// had never stopped. Restores the checkpoint's complete trainer state —
    /// fp32 masters, optimizer velocity, the completed-step clock that drives
    /// warmup, the cycle phase and the decay envelope, and the warmup length
    /// and cycle themselves (see `TrainerScheduleState`) — so warmup does not
    /// re-run; and reconstructs the replay buffer to its contents at the saved
    /// `next_game_index` (feed the preceding capacity-worth of games so the
    /// ring self-trims to the exact last-C plies) before continuing from there.
    /// Requires a `--start-model` carrying that trainer state and a corpus
    /// position for this same corpus (its lineage record, or the `replay_*`
    /// keys of a file written before lineage); mutually exclusive with `startShard` /
    /// `startGameIndex`. Without it, `--start-model` starts a new branch.
    var resumeExact: Bool = false
    /// Resume gaps `--resume-exact` may proceed without (`--accept-inexact`,
    /// determinism plan D-7): each named gap starts fresh and the segment is
    /// recorded as not exact. A changed corpus is never acceptable.
    var acceptInexact: Set<ResumeGap>
    /// Explicit destination for the rolling trainer-model file. When nil the
    /// runner derives a path next to `--start-model` (or, without one, in the
    /// app's Models directory, named after the first corpus). The same file is
    /// overwritten by the periodic autosave and by the final save on
    /// exit/abort — but a file already there before the run is replaced only
    /// under `TrainerOutputFileGuard.checkRollingOutput`'s rule (or with
    /// `overwriteOutModel`).
    var outModelPath: String?
    /// `--overwrite-out-model`: replace a regular file already at the rolling
    /// output path even when it is not the rolling file of the state this
    /// run continues (an earlier run of the same command, another run's
    /// output, an earlier checkpoint of the same line), and accept an output
    /// path named like a step-enumerated checkpoint. It never permits
    /// replacing the `--start-model` itself or anything that is not a regular
    /// file.
    var overwriteOutModel: Bool
    /// When true, every trainer-model save also writes a step-enumerated copy
    /// (`<stem>-replay-step<N>.safetensors`) next to the rolling `-replay-latest`
    /// file, so no checkpoint is ever lost to the overwrite. The rolling latest
    /// is still written (warm-start / probe / trackers key off it); the enumerated
    /// files accumulate — mind disk. Off by default. (`--enumerate-checkpoints`.)
    /// Enumerated files are never written over a file this run did not write:
    /// the run refuses to start when its stem already has step files it could
    /// reach, and a collision at write time halts it.
    var enumerateCheckpoints: Bool = false
    /// Freshly-minted `ModelID` for this run's saved model, minted on the main
    /// actor in the pre-flight handler (the `ModelIDMinter` is main-actor
    /// isolated and the replay loop runs off-actor, so it can't mint there).
    var runModelID: String
    /// Destination for the run's `results.json` (`--output`, checked before
    /// the run by `CliResultsOutput.preflight`), or nil for no JSON.
    var output: CliResultsOutput?

    /// One training step to capture as an Xcode GPU trace (`.gputrace`) for
    /// per-kernel profiling (`--gpu-capture-step` / `--gpu-capture-out`), or
    /// nil. Diagnostics only: the capture slows that one step, and nothing
    /// else about the run changes.
    struct GPUCapture: Sendable {
        /// 1-based number of the training step to capture (this run's steps).
        var step: Int
        var outputURL: URL
    }
    var gpuCapture: GPUCapture? = nil

    /// Where the trainer's policy head leaves the compute dtype for fp32
    /// (`--policy-tail-precision`). An A/B knob for the head-numerics cost;
    /// see `ChessNetwork.PolicyTailPrecision`.
    var policyTailPrecision: ChessNetwork.PolicyTailPrecision = .process

    /// The run's master seed (`RunRandomSeed.resolve`, at launch): the
    /// replay buffer's `sampler` stream and, for a run without
    /// `--start-model`, the fresh model's init seed derive from it.
    var runRandomSeed: RunRandomSeed
}

/// Consecutive failures of one kind of trainer save (the rolling file, the
/// session folder, or the step-enumerated copies), shared by corpus replay
/// and train-vs-UCI. One failure is tolerated — a transient error on an
/// external volume should not end a long run — but the next attempt failing
/// too halts the run: from then on it would train with nothing being kept. A
/// success resets the count.
///
/// The run's last save (the final save, or the abort save on Ctrl-C) has no
/// next attempt to recover it, and it is the only record of the state the
/// run ends in. After it, `requireLastSaveSucceeded` fails the run when it
/// did not succeed, so a run never exits as if its end state had been saved.
struct TrainerSaveFailureStreak {
    /// What is being saved, for the halt message.
    let what: String
    /// Consecutive failures after which the run halts.
    static let haltThreshold = 2
    private(set) var consecutiveFailures = 0

    init(what: String) {
        self.what = what
    }

    mutating func recordSuccess() {
        consecutiveFailures = 0
    }

    /// Count a failure at `step`; throws once the streak reaches the threshold.
    mutating func recordFailure(step: Int) throws {
        consecutiveFailures += 1
        if consecutiveFailures >= Self.haltThreshold {
            throw CorpusReplayError.repeatedSaveFailures(count: consecutiveFailures, lastStep: step, what: what)
        }
    }

    /// The run's last save of this kind failed; its end state is not on disk.
    struct LastSaveFailedError: LocalizedError, Equatable {
        let what: String
        let reason: String
        let step: Int

        var errorDescription: String? {
            "the \(reason) \(what) at step \(step) failed (see the log above); the run's end state was not saved"
        }
    }

    /// Call right after the run's last save of this kind (`reason`: the
    /// final or abort save). Every save records a success or a failure, so
    /// a non-zero streak here means that save failed.
    func requireLastSaveSucceeded(step: Int, reason: String) throws {
        guard consecutiveFailures == 0 else {
            throw LastSaveFailedError(what: what, reason: reason, step: step)
        }
    }
}

enum CorpusReplayError: LocalizedError {
    case noGames
    case startModelTooSmall(have: Int, need: Int)
    case diskFullDuringSave(step: Int, what: String)
    case gpuCaptureUnavailable
    case gpuCaptureFailed(String)
    /// The requested capture step is past the run's step limit.
    case gpuCaptureStepUnreachable(step: Int, stepLimit: Int)
    /// The trace's folder is missing, not a folder, or not writable.
    case gpuCaptureFolderUnusable(path: String, reason: String)
    /// The same kind of save failed on consecutive attempts; see
    /// `TrainerSaveFailureStreak`.
    case repeatedSaveFailures(count: Int, lastStep: Int, what: String)
    /// A shard could not be read during an exact resume, so the fed stream
    /// can no longer be the saved run's.
    case exactResumeShardUnreadable(shard: String, detail: String)
    /// `--resume-exact` of a checkpoint whose run has already completed the
    /// epoch budget this run is given: there is nothing left to train.
    case exactResumeEpochBudgetSpent(savedEpoch: Int, epochLimit: Int)
    /// The checkpoint's corpus position is not a position in this corpus.
    case exactResumePositionInvalid(epoch: Int, nextGameIndex: Int, totalGames: Int)
    /// The reconstruction refeed ran out of games before reaching the saved
    /// position.
    case exactResumeRefeedEndedEarly(reachedEpoch: Int, reachedGame: Int, targetEpoch: Int, targetGame: Int)
    /// The rebuilt replay buffer does not hold as many positions as the
    /// checkpoint's did, so its contents are not the saved run's.
    case exactResumeBufferMismatch(savedPositions: Int, rebuiltPositions: Int, capacity: Int)
    var errorDescription: String? {
        switch self {
        case let .exactResumeEpochBudgetSpent(savedEpoch, epochLimit):
            return "--resume-exact: the checkpoint's run has completed \(savedEpoch) pass(es) over the corpus and "
                + "--epochs is \(epochLimit) (default 1, counted from the start of the run's lineage), so there is "
                + "nothing left to train; pass --epochs greater than \(savedEpoch), or --training-step-limit"
        case let .exactResumePositionInvalid(epoch, nextGameIndex, totalGames):
            return "--resume-exact: the checkpoint's corpus position (epoch \(epoch), next game \(nextGameIndex)) is not "
                + "a position in this corpus of \(totalGames) games (epoch 0 or later, next game 0…\(totalGames - 1))"
        case let .exactResumeRefeedEndedEarly(reachedEpoch, reachedGame, targetEpoch, targetGame):
            return "--resume-exact: rebuilding the replay buffer ran out of games at epoch \(reachedEpoch) game "
                + "\(reachedGame), before the saved position (epoch \(targetEpoch) game \(targetGame))"
        case let .exactResumeBufferMismatch(savedPositions, rebuiltPositions, capacity):
            return "--resume-exact: the rebuilt replay buffer holds \(rebuiltPositions) positions but the checkpoint's "
                + "held \(savedPositions) (capacity \(capacity)), so it does not hold the games the saved run's did — "
                + "most likely the saved run started part-way into the corpus (--start-game-index or --start-shard) "
                + "and had not yet filled its buffer; continue it as a new branch without --resume-exact"
        case let .exactResumeShardUnreadable(shard, detail):
            return "--resume-exact: shard \(shard) could not be read (\(detail)); an exact resume feeds every game the checkpoint's run fed, so it stops instead of skipping the shard"
        case let .gpuCaptureStepUnreachable(step, stepLimit):
            return "--gpu-capture-step \(step) is past the run's step limit \(stepLimit); the capture could never happen"
        case let .gpuCaptureFolderUnusable(path, reason):
            return "GPU trace folder \(path) \(reason)"
        case let .repeatedSaveFailures(count, lastStep, what):
            return "\(what) failed \(count) times in a row (last at step \(lastStep)) — halting rather than training on with nothing saved"
        case .gpuCaptureUnavailable:
            return "GPU trace capture is not available in this process — launch with MTL_CAPTURE_ENABLED=1 in the environment"
        case let .gpuCaptureFailed(detail):
            return "GPU trace capture failed to start: \(detail)"
        case .noGames:
            return "No sealed shards found in the provided corpus path(s)"
        case let .startModelTooSmall(have, need):
            return "--start-model has \(have) weight tensors but the network needs at least \(need) (trainables + BN running stats)"
        case let .diskFullDuringSave(step, what):
            return "Disk full while writing \(what) at step \(step) — training halted so it can resume from the last checkpoint once space is freed"
        }
    }
}

// MARK: - Trainer-model output safety (shared by corpus replay and train-vs-UCI)

/// Which model a trainer-model file holds, as recorded in its safetensors
/// `__metadata__`: the `model_id` plus the segment-local `training_step`.
/// Checkpoints are identified by these, never by filename (see CLAUDE.md,
/// "Identify checkpoints by safetensors `__metadata__`").
struct TrainerModelFileIdentity: Equatable, Sendable, CustomStringConvertible {
    let modelID: String
    /// Nil when the file records no step (a fresh build, a champion export).
    let trainingStep: Int?

    var description: String {
        "model \(modelID) at step \(trainingStep.map(String.init) ?? "(none recorded)")"
    }

    /// Read from the header only — the weights are never decoded.
    static func read(from url: URL) throws -> TrainerModelFileIdentity {
        let metadata = try ModelFileCatalog.headerMetadata(at: url)
        guard let modelID = metadata[SafetensorsModelIO.Key.modelID], !modelID.isEmpty else {
            throw ModelFileCatalogError.missingModelID(file: url.lastPathComponent)
        }
        var trainingStep: Int? = nil
        if let text = metadata[SafetensorsModelIO.Key.trainingStep] {
            guard let step = Int(text) else {
                throw ModelFileCatalogError.notSafetensors(
                    file: url.lastPathComponent, detail: "training_step \"\(text)\" is not an integer")
            }
            trainingStep = step
        }
        return TrainerModelFileIdentity(modelID: modelID, trainingStep: trainingStep)
    }
}

/// Refusals from the trainer-model output checks. Each message names the
/// file involved and says what to do instead; none is ever downgraded to a
/// warning or worked around with a different path.
enum TrainerOutputFileError: LocalizedError, Equatable {
    case outModelIsStartModel(path: String)
    case outModelNotARegularFile(path: String, kind: FileSafety.ItemKind)
    case outModelUnreadable(path: String, detail: String)
    case outModelBelongsToAnotherRun(path: String, reason: String)
    case enumeratedStepsAlreadyPresent(firstPath: String, count: Int, steps: String, reachable: String, suggestion: String)
    case enumeratedCheckpointExists(path: String, step: Int)
    case outModelNamedLikeAnEnumeratedCheckpoint(path: String, step: Int)

    var errorDescription: String? {
        switch self {
        case let .outModelIsStartModel(path):
            return "refusing to start: --out-model \(path) is the --start-model itself. The rolling output "
                + "file is overwritten every save, so this would destroy the model the run started from. "
                + "Pass a new --out-model path."
        case let .outModelNotARegularFile(path, kind):
            return "refusing to start: --out-model \(path) is a \(kind), not a regular file. A trainer-model "
                + "save only ever replaces a regular file. Pass a different --out-model path."
        case let .outModelUnreadable(path, detail):
            return "refusing to start: --out-model \(path) already exists and its model metadata cannot be read "
                + "(\(detail)), so there is no way to tell whether this run may overwrite it. Pass a new "
                + "--out-model path, or --overwrite-out-model to replace it anyway."
        case let .outModelBelongsToAnotherRun(path, reason):
            return "refusing to start: --out-model \(path) already exists and is not this run's to overwrite: "
                + "\(reason). Pass a new --out-model path; or, to continue that run, pass that file as "
                + "--start-model (with --resume-exact) and a new --out-model; or pass --overwrite-out-model "
                + "to replace it anyway."
        case let .enumeratedStepsAlreadyPresent(firstPath, count, steps, reachable, suggestion):
            return "refusing to start: --enumerate-checkpoints would write step files this run can reach "
                + "(\(reachable)), but \(count) step file(s) of this --out-model stem already exist there "
                + "(steps \(steps); first: \(firstPath)). Step numbers restart in every segment and step "
                + "names carry the segment's lineage index, so these files belong to another run that used "
                + "this stem at the same segment index. Give this run its own --out-model stem, e.g. "
                + "\(suggestion)."
        case let .outModelNamedLikeAnEnumeratedCheckpoint(path, step):
            return "refusing to start: --out-model \(path) is named like the step-\(step) checkpoint that "
                + "--enumerate-checkpoints writes. The rolling output is rewritten on every save, so under that "
                + "name it would overwrite that checkpoint (if it exists) or later be mistaken for it. Pass a "
                + "rolling-file name, e.g. <name>-replay-latest.safetensors or <name>-vsuci-latest.safetensors; "
                + "or pass --overwrite-out-model to use this name anyway (it then replaces the regular file "
                + "there, whatever it holds)."
        case let .enumeratedCheckpointExists(path, step):
            return "enumerated checkpoint for step \(step) not written: \(path) already exists and this run "
                + "did not write it. Halting rather than overwrite another run's checkpoint; restart with "
                + "a new --out-model stem."
        }
    }
}

/// How the run will treat its rolling `--out-model` file, decided before any
/// training and carried into every save.
struct RollingOutputPlan: Equatable, Sendable {
    enum Disposition: Equatable, Sendable {
        /// Nothing is there; the first save publishes a new file.
        case createNew
        /// The file holds the model line this run continues, at the start
        /// model's own step, so replacing it loses nothing.
        case continueLineage(existing: TrainerModelFileIdentity)
        /// The operator passed `--overwrite-out-model`.
        case overwriteAuthorized(existing: String)
    }

    let disposition: Disposition
    /// The existing regular file's identity when one is to be replaced; the
    /// first save replaces only that very file.
    let existingFileIdentity: FileSafety.FileIdentity?

    var logDescription: String {
        switch disposition {
        case .createNew:
            return "new file"
        case let .continueLineage(existing):
            return "replacing the existing file, which holds \(existing) — the model line this run continues"
        case let .overwriteAuthorized(existing):
            return "replacing the existing file (\(existing)) as --overwrite-out-model allows"
        }
    }
}

/// The name of a step-enumerated checkpoint (`--enumerate-checkpoints`),
/// derived from the rolling output file's stem: `<base>-<tag>-latest` becomes
/// `<base>-<tag>-step<N>`, any other stem gains `-step<N>`. The one place the
/// name is built and the only parser of it, so the pre-flight scan and the
/// writes cannot disagree about which files are a stem's step files.
///
/// Step numbers restart in every segment, so a resumed segment's step files
/// carry the segment's lineage index: segment `k > 0` writes
/// `<base>-<tag>-seg<k>-step<N>` (`<stem>-seg<k>-step<N>` without a tag
/// marker). Segment 0 — every run's first segment — carries no marker, so
/// its names are the ones runs have always written. A resumed segment can
/// therefore keep its stem without its step files ever colliding with, or
/// being mistaken for, an earlier segment's; the index comes from
/// `LineageTracker.segmentIndex(exactResumeOf:)`, the same rule the
/// segment's lineage records use.
struct EnumeratedCheckpointNaming: Equatable, Sendable {
    /// Corpus replay's run tag (`--replay-corpus`).
    static let corpusReplayRunTag = "replay"
    /// Train-vs-UCI's run tag (`--train-vs-uci`).
    static let trainVsUciRunTag = "vsuci"
    /// Every run kind that writes step-enumerated checkpoints.
    static let allRunTags = [
        EnumeratedCheckpointNaming.corpusReplayRunTag,
        EnumeratedCheckpointNaming.trainVsUciRunTag,
    ]

    /// What precedes the step number in every enumerated name.
    private static let stepMarker = "-step"
    /// What precedes the segment index in a later segment's names.
    private static let segmentMarker = "-seg"
    private static let fileExtension = "safetensors"

    let rollingOutputURL: URL
    /// `corpusReplayRunTag` or `trainVsUciRunTag`: the run kind in the
    /// rolling file's `-<tag>-latest` marker.
    let runTag: String
    /// The writing segment's lineage index (`LineageTracker.segmentIndex(exactResumeOf:)`).
    let segmentIndex: Int

    init(rollingOutputURL: URL, runTag: String, segmentIndex: Int) {
        precondition(segmentIndex >= 0, "a lineage segment index is never negative (got \(segmentIndex))")
        self.rollingOutputURL = rollingOutputURL
        self.runTag = runTag
        self.segmentIndex = segmentIndex
    }

    private var rollingStem: String { rollingOutputURL.deletingPathExtension().lastPathComponent }
    private var rollingMarker: String { "-\(runTag)-latest" }
    /// `-seg<k>` for a later segment; empty for segment 0.
    private var segmentPart: String { segmentIndex == 0 ? "" : "\(Self.segmentMarker)\(segmentIndex)" }
    var directory: URL { rollingOutputURL.deletingLastPathComponent() }

    func fileName(step: Int) -> String {
        let stem = rollingStem
        let enumeratedStem = stem.contains(rollingMarker)
            ? stem.replacingOccurrences(of: rollingMarker, with: "-\(runTag)\(segmentPart)\(Self.stepMarker)\(step)")
            : "\(stem)\(segmentPart)\(Self.stepMarker)\(step)"
        return "\(enumeratedStem).\(Self.fileExtension)"
    }

    func url(step: Int) -> URL {
        directory.appendingPathComponent(fileName(step: step))
    }

    /// The step `name` is the enumerated checkpoint for, or nil when it is
    /// not exactly one of this stem's step files.
    func step(ofFileName name: String) -> Int? {
        let stem = rollingStem
        let prefix: String
        if let marker = stem.range(of: rollingMarker) {
            prefix = String(stem[..<marker.lowerBound]) + "-\(runTag)\(segmentPart)\(Self.stepMarker)"
        } else {
            prefix = "\(stem)\(segmentPart)\(Self.stepMarker)"
        }
        guard name.hasPrefix(prefix) else { return nil }
        let digits = name.dropFirst(prefix.count).prefix { $0.isASCII && $0.isNumber }
        guard !digits.isEmpty, let step = Int(digits), fileName(step: step) == name else { return nil }
        return step
    }

    /// The step `name` is the enumerated checkpoint for under *some* rolling
    /// stem of either run kind, or nil when no rolling file this naming
    /// knows of would enumerate to it — e.g. `x-replay-step29000.safetensors`
    /// (stem `x-replay-latest`), `x-vsuci-step7.safetensors`, or
    /// `x-step3000.safetensors` (stem `x`). Every candidate is confirmed by
    /// rebuilding the name with `fileName(step:)` / `step(ofFileName:)`, so
    /// a name is only ever claimed when this naming would really produce it.
    /// Used to keep a rolling `--out-model` off an enumerated checkpoint's
    /// name.
    static func step(ofEnumeratedFileNameUnderAnyStem name: String) -> Int? {
        let suffix = ".\(fileExtension)"
        guard name.hasSuffix(suffix) else { return nil }
        let stem = String(name.dropLast(suffix.count))
        var searchEnd = stem.endIndex
        while let markerRange = stem.range(of: stepMarker, options: .backwards, range: stem.startIndex..<searchEnd) {
            searchEnd = markerRange.lowerBound
            let digits = stem[markerRange.upperBound...].prefix { $0.isASCII && $0.isNumber }
            guard !digits.isEmpty, let step = Int(digits) else { continue }
            let beforeMarker = String(stem[..<markerRange.lowerBound])
            // Two readings of what precedes the step marker: the whole of it as
            // a segment-0 base, and — when it ends in `-seg<k>` — the part
            // before that as a later segment's base. Each candidate is
            // confirmed by rebuilding the name, so only a real reading counts.
            var readings: [(base: String, segmentIndex: Int, segmentPart: String)] = [(beforeMarker, 0, "")]
            if let segmentRange = beforeMarker.range(of: segmentMarker, options: .backwards) {
                let segmentDigits = beforeMarker[segmentRange.upperBound...]
                if !segmentDigits.isEmpty, segmentDigits.allSatisfy({ $0.isASCII && $0.isNumber }),
                   let parsed = Int(segmentDigits) {
                    readings.append((String(beforeMarker[..<segmentRange.lowerBound]), parsed,
                                     String(beforeMarker[segmentRange.lowerBound...])))
                }
            }
            for reading in readings {
                var candidateRollingStems: [(stem: String, runTag: String)] = []
                // A stem without a `-<tag>-latest` marker: the step marker ends the name.
                if !reading.base.isEmpty, digits.endIndex == stem.endIndex {
                    candidateRollingStems += allRunTags.map { (stem: reading.base, runTag: $0) }
                }
                // A stem whose `-<tag>-latest` marker(s) became `-<tag>[-seg<k>]-step<N>`.
                for runTag in allRunTags where reading.base.hasSuffix("-\(runTag)") {
                    let rollingStem = stem.replacingOccurrences(
                        of: "-\(runTag)\(reading.segmentPart)\(stepMarker)\(step)", with: "-\(runTag)-latest")
                    candidateRollingStems.append((stem: rollingStem, runTag: runTag))
                }
                for candidate in candidateRollingStems {
                    let naming = EnumeratedCheckpointNaming(
                        rollingOutputURL: URL(fileURLWithPath: "/").appendingPathComponent("\(candidate.stem)\(suffix)"),
                        runTag: candidate.runTag, segmentIndex: reading.segmentIndex)
                    if naming.step(ofFileName: name) == step { return step }
                }
            }
        }
        return nil
    }

    /// The `--out-model` to suggest when this stem is taken: the same name
    /// with a `-resumeN` segment marker.
    var newStemSuggestion: String {
        let stem = rollingStem
        let suggested = stem.contains(rollingMarker)
            ? stem.replacingOccurrences(of: rollingMarker, with: "-resumeN\(rollingMarker)")
            : "\(stem)-resumeN"
        return directory.appendingPathComponent("\(suggested).safetensors").path
    }
}

/// One existing enumerated checkpoint of a stem.
struct EnumeratedCheckpointFile: Equatable, Sendable {
    let step: Int
    let url: URL
}

/// Writes the rolling trainer-model file, replacing only the file this run
/// owns: the pre-existing file the plan adopted (checked by identity), then
/// each file this writer itself wrote.
final class RollingTrainerModelWriter {
    let url: URL
    private var ownedIdentity: FileSafety.FileIdentity?

    init(url: URL, plan: RollingOutputPlan) {
        self.url = url
        self.ownedIdentity = plan.existingFileIdentity
    }

    func write(_ data: Data) throws {
        if let ownedIdentity {
            self.ownedIdentity = try FileSafety.replaceRegularFile(data, at: url, expectedIdentity: ownedIdentity)
        } else {
            self.ownedIdentity = try FileSafety.publishNewFile(data, to: url)
        }
    }
}

/// Writes step-enumerated checkpoints, never over a file this run did not
/// write. A step this run already wrote (the final save landing on the same
/// step as the last autosave) replaces this run's own file — checked by
/// identity — so the enumerated copy matches the rolling file's final state.
final class EnumeratedCheckpointWriter {
    enum Outcome: Equatable {
        case created
        case replacedThisRunsEarlierSave
    }

    let naming: EnumeratedCheckpointNaming
    private var writtenByThisRun: [Int: FileSafety.FileIdentity] = [:]

    init(naming: EnumeratedCheckpointNaming) {
        self.naming = naming
    }

    func write(_ data: Data, step: Int) throws -> (url: URL, outcome: Outcome) {
        let url = naming.url(step: step)
        if let owned = writtenByThisRun[step] {
            writtenByThisRun[step] = try FileSafety.replaceRegularFile(data, at: url, expectedIdentity: owned)
            return (url, .replacedThisRunsEarlierSave)
        }
        do {
            writtenByThisRun[step] = try FileSafety.publishNewFile(data, to: url)
        } catch FileSafetyError.alreadyExists(path: let path, kind: _) {
            throw TrainerOutputFileError.enumeratedCheckpointExists(path: path, step: step)
        }
        return (url, .created)
    }
}

/// The before-training checks on a trainer run's output files. An operation
/// only ever touches the exact files it writes, replaces only regular files,
/// and never silently overwrites a file it does not own; anything unexpected
/// refuses the run before any GPU work is spent on it.
enum TrainerOutputFileGuard {

    /// Whether this run may replace the regular file already at its rolling
    /// output path.
    enum RollingOverwriteVerdict: Equatable, Sendable {
        case continuesLineage
        case refused(reason: String)
    }

    /// The ownership rule for an existing rolling output file, from the two
    /// headers alone.
    ///
    /// Every corpus-replay / train-vs-UCI launch mints a brand-new `model_id`
    /// for what it saves (with `parent_model_id` = the start model's), so the
    /// file at the output path can never carry this run's own ID. The file is
    /// this run's to replace exactly when it holds the very state the run
    /// starts from — the `--start-model`'s `model_id` at the `--start-model`'s
    /// `training_step` — so replacing it discards nothing that exists only
    /// there. That covers continuing a crashed segment into its rolling file
    /// from that segment's latest checkpoint (which, saved at the same step,
    /// holds the same state). Everything else is refused: a fresh run (it
    /// continues no line); a file from a different model line, which includes
    /// an earlier run of the very same command — that run minted its own ID,
    /// and its file may hold hours of training; a file ahead of the start
    /// model, which holds training that exists only there; and a file
    /// *behind* it. Same line at an earlier step is not a stale rolling file
    /// in practice but a fixed checkpoint of that line — an enumerated step
    /// file, or a hand-made copy of one — reached by a mistyped
    /// `--out-model`, and the run would overwrite it with later training.
    static func rollingOverwriteVerdict(existing: TrainerModelFileIdentity,
                                        startModel: TrainerModelFileIdentity?) -> RollingOverwriteVerdict {
        guard let startModel else {
            return .refused(reason: "it holds \(existing), and this run starts a fresh network, so it "
                + "continues no model line")
        }
        guard existing.modelID == startModel.modelID else {
            return .refused(reason: "it holds \(existing), a different model line from the --start-model's "
                + "\(startModel.modelID) — another run's output (every run, including an earlier run of "
                + "this same command, saves under its own new model ID)")
        }
        guard let existingStep = existing.trainingStep, let startStep = startModel.trainingStep else {
            return .refused(reason: "it holds \(existing) and the --start-model is \(startModel); without "
                + "both training steps there is no telling whether replacing it loses training")
        }
        // No run writes a negative step, so one marks a damaged or hand-edited
        // header. Refusing it here also keeps the step differences below in
        // range: two non-negative Ints cannot overflow when subtracted.
        guard existingStep >= 0, startStep >= 0 else {
            return .refused(reason: "it holds \(existing) and the --start-model is \(startModel); a negative "
                + "training step is never written by a run, so one of the two headers is damaged")
        }
        guard existingStep <= startStep else {
            return .refused(reason: "it holds \(existing), \(existingStep - startStep) steps ahead of the "
                + "--start-model (step \(startStep)); replacing it would discard training that exists only "
                + "in that file")
        }
        guard existingStep == startStep else {
            return .refused(reason: "it holds \(existing), \(startStep - existingStep) steps behind the "
                + "--start-model (step \(startStep)), so it is not the rolling file of the state this run "
                + "starts from but an earlier checkpoint of that line (an enumerated step file, or a copy of "
                + "one); replacing it would overwrite that checkpoint with later training")
        }
        return .continuesLineage
    }

    /// Decide how the run treats its rolling output file, or refuse:
    /// (a) never the start model itself; (b) never anything but a regular
    /// file; (c) never a name shaped like a step-enumerated checkpoint
    /// (`EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem:)`),
    /// whether or not a file is there yet, unless `overwriteAuthorized`;
    /// (d) an existing regular file only under `rollingOverwriteVerdict`, or
    /// with `overwriteAuthorized` (`--overwrite-out-model`). (a) and (b) hold
    /// even with the flag.
    ///
    /// (c) is checked by name because the header check cannot see it: an
    /// enumerated checkpoint of the very line being continued, at the start
    /// model's own step, passes (d), and a rolling file under an enumerated
    /// name — rewritten every save — would later be taken for a fixed
    /// checkpoint by anything that finds checkpoints by their names.
    static func checkRollingOutput(outModelURL: URL,
                                   startModelURL: URL?,
                                   startModel: TrainerModelFileIdentity?,
                                   overwriteAuthorized: Bool) throws -> RollingOutputPlan {
        // Every save stages first; a name the staging copy cannot have would
        // fail every save, hours into the run. Applies even with the flag.
        try FileSafety.requireStageableDestination(outModelURL)
        if let startModelURL, try FileSafety.mayNameTheSameFile(outModelURL, startModelURL) {
            throw TrainerOutputFileError.outModelIsStartModel(path: outModelURL.path)
        }
        if !overwriteAuthorized,
           let step = EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem: outModelURL.lastPathComponent) {
            throw TrainerOutputFileError.outModelNamedLikeAnEnumeratedCheckpoint(path: outModelURL.path, step: step)
        }
        guard let entry = try FileSafety.existingItem(at: outModelURL) else {
            return RollingOutputPlan(disposition: .createNew, existingFileIdentity: nil)
        }
        guard entry.kind == .regularFile else {
            throw TrainerOutputFileError.outModelNotARegularFile(path: outModelURL.path, kind: entry.kind)
        }
        let existing: TrainerModelFileIdentity
        do {
            existing = try TrainerModelFileIdentity.read(from: outModelURL)
        } catch {
            if overwriteAuthorized {
                return RollingOutputPlan(
                    disposition: .overwriteAuthorized(existing: "model metadata unreadable: \(error.localizedDescription)"),
                    existingFileIdentity: entry.identity)
            }
            throw TrainerOutputFileError.outModelUnreadable(path: outModelURL.path, detail: error.localizedDescription)
        }
        if overwriteAuthorized {
            return RollingOutputPlan(disposition: .overwriteAuthorized(existing: existing.description),
                                     existingFileIdentity: entry.identity)
        }
        switch rollingOverwriteVerdict(existing: existing, startModel: startModel) {
        case .continuesLineage:
            return RollingOutputPlan(disposition: .continueLineage(existing: existing),
                                     existingFileIdentity: entry.identity)
        case let .refused(reason):
            throw TrainerOutputFileError.outModelBelongsToAnotherRun(path: outModelURL.path, reason: reason)
        }
    }

    /// The stem's existing step files this run could write over: every step
    /// file when the run has no step limit, else those at steps
    /// `0...stepLimit` (the final save can land on any step up to the limit —
    /// an abort, the end of the corpus, a time limit). Sorted by step. An
    /// absent output directory has none (it is created at the first save).
    static func reachableEnumeratedCheckpoints(naming: EnumeratedCheckpointNaming,
                                               stepLimit: Int?) throws -> [EnumeratedCheckpointFile] {
        guard try FileSafety.existingItem(at: naming.directory) != nil else { return [] }
        let names = try FileManager.default.contentsOfDirectory(atPath: naming.directory.path)
        var found: [EnumeratedCheckpointFile] = []
        for name in names {
            guard let step = naming.step(ofFileName: name) else { continue }
            if let stepLimit, step > stepLimit { continue }
            found.append(EnumeratedCheckpointFile(step: step, url: naming.directory.appendingPathComponent(name)))
        }
        return found.sorted { $0.step < $1.step }
    }

    /// Refuse the run when its stem already has step files it could reach, or
    /// when a step file it could write has a name too long to stage. Every
    /// save step is at most the step limit, so the limit's step name is the
    /// longest; with no limit, any step an `Int` can hold is possible.
    static func requireNoReachableEnumeratedCheckpoints(naming: EnumeratedCheckpointNaming,
                                                        stepLimit: Int?) throws {
        try FileSafety.requireStageableDestination(naming.url(step: stepLimit.map { max($0, 0) } ?? Int.max))
        let collisions = try reachableEnumeratedCheckpoints(naming: naming, stepLimit: stepLimit)
        guard let first = collisions.first, let last = collisions.last else { return }
        throw TrainerOutputFileError.enumeratedStepsAlreadyPresent(
            firstPath: first.url.path,
            count: collisions.count,
            steps: first.step == last.step ? "\(first.step)" : "\(first.step)…\(last.step)",
            reachable: stepLimit.map { "steps 0…\($0)" } ?? "any step: the run has no step limit",
            suggestion: "--out-model \(naming.newStemSuggestion)")
    }
}

/// Headless offline trainer: builds a network + trainer, fills the
/// `ReplayBuffer` from a fixed game corpus (no self-play, no arena, no
/// promotion), runs a step-locked SGD loop to a budget, and exits. Invoked
/// from the `--replay-corpus` CLI pre-flight handler.
enum CorpusReplayRunner {

    struct Result: Sendable {
        var steps: Int
        var positionsFed: Int
        /// Games consumed from the corpus, whatever happened to each.
        var gamesFed: Int
        /// Consumed games whose move list could not be replayed.
        var gamesRejected: Int
        /// Consumed games skipped as empty or FEN-setup.
        var gamesSkipped: Int
        var epochs: Int
    }

    /// Check a GPU capture request against everything that can be known
    /// before training: its step is within the run's step limit (when there is
    /// one — an epoch- or corpus-bound run is checked at its end instead),
    /// nothing exists at the trace path (a dangling symbolic link counts), and
    /// the trace's folder exists, is a folder (following links: `/tmp` is one)
    /// and is writable. Capture availability in this process is a separate
    /// check (`MTLCaptureManager.supportsDestination`). The one place these
    /// rules live: the pre-flight and the capture start both call it.
    static func validateGPUCaptureRequest(_ capture: CorpusReplayConfig.GPUCapture, stepLimit: Int?) throws {
        if let stepLimit, capture.step > stepLimit {
            throw CorpusReplayError.gpuCaptureStepUnreachable(step: capture.step, stepLimit: stepLimit)
        }
        // The folder first: under a missing folder or a file, inspecting the
        // trace path itself fails with a system error instead of the reason.
        let folder = capture.outputURL.deletingLastPathComponent()
        var isFolder: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isFolder) else {
            throw CorpusReplayError.gpuCaptureFolderUnusable(path: folder.path, reason: "does not exist")
        }
        guard isFolder.boolValue else {
            throw CorpusReplayError.gpuCaptureFolderUnusable(path: folder.path, reason: "is not a folder")
        }
        guard FileManager.default.isWritableFile(atPath: folder.path) else {
            throw CorpusReplayError.gpuCaptureFolderUnusable(path: folder.path, reason: "is not writable")
        }
        if try FileSafety.existingItem(at: capture.outputURL) != nil {
            throw CorpusReplayError.gpuCaptureFailed("\(capture.outputURL.path) already exists")
        }
    }

    /// Start capturing every command buffer `device` runs into an Xcode GPU
    /// trace document at `capture.outputURL`. Throws instead of training on
    /// without the capture the operator asked for.
    private static func beginGPUCapture(_ capture: CorpusReplayConfig.GPUCapture, device: MTLDevice) throws {
        let manager = MTLCaptureManager.shared()
        guard manager.supportsDestination(.gpuTraceDocument) else {
            throw CorpusReplayError.gpuCaptureUnavailable
        }
        // Rechecked now: the folder or path may have changed since the
        // pre-flight. The step is this one, so its limit is not in question.
        try validateGPUCaptureRequest(capture, stepLimit: nil)
        let descriptor = MTLCaptureDescriptor()
        descriptor.captureObject = device
        descriptor.destination = .gpuTraceDocument
        descriptor.outputURL = capture.outputURL
        do {
            try manager.startCapture(with: descriptor)
        } catch {
            throw CorpusReplayError.gpuCaptureFailed(error.localizedDescription)
        }
        emit("[REPLAY] GPU trace capture started for step \(capture.step) -> \(capture.outputURL.path)")
    }

    /// Write a line to BOTH the session log file and stdout. `SessionLogger.log`
    /// targets only the per-launch log file, so a headless replay watched in a
    /// terminal would otherwise see only the per-step lines — routing the
    /// runner's own status/banner lines through here makes the whole run
    /// visible there too. (Errors/warnings keep going to stderr separately.)
    private static func emit(_ message: String) {
        SessionLogger.shared.log(message)
        print(message)
    }

    /// True when `error` means the filesystem is out of space (ENOSPC). Covers
    /// the Cocoa `.fileWriteOutOfSpace` that `Data.write(to:options:)` throws, a
    /// raw POSIX ENOSPC surfaced directly or as an underlying error, and a
    /// `FileSafety` system call that failed with ENOSPC (the trainer-model
    /// writers stage, sync and rename through `FileSafety`). Pure and total, so
    /// the save-failure policy is unit-testable without a real full disk.
    static func isOutOfSpace(_ error: Error) -> Bool {
        if case .systemCallFailed(_, _, let errnoValue)? = error as? FileSafetyError, errnoValue == ENOSPC {
            return true
        }
        // A session save (train-vs-UCI) wraps the file-system error it hit.
        if let checkpointError = error as? CheckpointManagerError {
            switch checkpointError {
            case .directoryCreationFailed(_, let underlying),
                 .writeFailed(_, let underlying),
                 .fsyncFailed(_, let underlying),
                 .chartFileWriteFailed(_, let underlying):
                return isOutOfSpace(underlying)
            default:
                return false
            }
        }
        if case .writeFailed(let underlying)? = error as? ReplayBuffer.PersistenceError {
            return isOutOfSpace(underlying)
        }
        let ns = error as NSError
        if ns.domain == NSCocoaErrorDomain, ns.code == CocoaError.Code.fileWriteOutOfSpace.rawValue {
            return true
        }
        if ns.domain == NSPOSIXErrorDomain, ns.code == Int(ENOSPC) {
            return true
        }
        if let underlying = ns.userInfo[NSUnderlyingErrorKey] as? NSError,
           underlying.domain == NSPOSIXErrorDomain, underlying.code == Int(ENOSPC) {
            return true
        }
        return false
    }

    /// Handle a checkpoint-save write failure.
    ///
    /// A disk-full (ENOSPC) failure is FATAL: it emits a loud `[ALARM]` to stderr
    /// AND the session log, then throws so the run halts. Training on through a
    /// full disk is the one failure we must never tolerate — with no free space,
    /// neither the enumerated checkpoints NOR the (tracking-critical) session log
    /// can be written, so the run silently accrues hours of work that leaves no
    /// probe data and punches an unrecoverable hole in the by-time/pElo charts
    /// (exactly what happened to nt8y on 2026-07-06). Halting stops at the last
    /// good checkpoint, so once space is freed the run resumes cleanly from there
    /// losing only the steps since the last autosave.
    ///
    /// Any OTHER write failure (e.g. a transient error on an external volume)
    /// is logged here as a WARNING and the caller continues — but only once in
    /// a row: callers also record each failure on a `TrainerSaveFailureStreak`,
    /// which halts the run when the same kind of save fails again at the next
    /// attempt. A run that cannot save at all (a read-only `--out-model`
    /// volume) would otherwise train for hours and keep nothing.
    static func reportSaveFailure(_ error: Error, step: Int, what: String) throws {
        if isOutOfSpace(error) {
            let msg = "[ALARM] [REPLAY] DISK FULL writing \(what) at step \(step): "
                + "\(error.localizedDescription). Halting — free disk space, then resume from the last "
                + "checkpoint. Training through a full disk loses checkpoints AND log lines, corrupting "
                + "probe data and tracking."
            FileHandle.standardError.write(Data((msg + "\n").utf8))
            SessionLogger.shared.log(msg)
            throw CorpusReplayError.diskFullDuringSave(step: step, what: what)
        }
        let msg = "[REPLAY] WARNING: \(what) failed at step \(step): \(error.localizedDescription)"
        FileHandle.standardError.write(Data((msg + "\n").utf8))
        SessionLogger.shared.log(msg)
    }

    /// Run the replay to completion and exit the process. Never returns.
    /// Exit status 0 on success, 2 when the run is refused before training
    /// (`CLIRunRefusal`), 33 on any other failure.
    static func runAndExit(config: CorpusReplayConfig, params: ReplayParams) -> Never {
        SessionLogger.shared.start()
        emit("[REPLAY] starting offline corpus replay over \(config.corpusDirectories.count) corpus path(s)")

        // Ctrl-C handling. Install BEFORE the run so an early interrupt is
        // honored. We ignore the default SIGINT disposition (which would kill
        // the process immediately, losing the final save) and instead route
        // the signal to a DispatchSource handler on a background queue, where
        // it's safe to do real work (signal handlers proper are
        // async-signal-unsafe). First press → request a clean abort; the loop
        // breaks after the in-flight step and the final save runs. Second
        // press → restore the default disposition and re-raise, so an
        // impatient or wedged run can still be force-killed.
        let abort = ReplayAbortFlag()
        signal(SIGINT, SIG_IGN)
        let sigSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        sigSource.setEventHandler {
            if abort.isRequested {
                signal(SIGINT, SIG_DFL)
                raise(SIGINT)
                return
            }
            abort.request()
            emit("[REPLAY] SIGINT received — finishing current step, saving, then exiting (Ctrl-C again to force-quit)")
        }
        sigSource.resume()

        // Hold a strong reference to the dispatch source across the blocking
        // run. `sigSource` is otherwise unused after `resume()`, and in a
        // Release build ARC may shorten its lifetime to that last use and
        // deallocate it — a released signal source stops delivering, silently
        // breaking Ctrl-C while we're parked in `syncWait`.
        let result: Result
        do {
            result = try withExtendedLifetime(sigSource) {
                try syncWait {
                    try await runReplay(config: config, params: params, abort: abort)
                }
            }
        } catch let refusal as CLIRunRefusal {
            // Refused before training: a usage problem, status 2. The log is
            // drained before the exit, so the refusal (and any `[RESUME]`
            // verdict before it) is in it.
            FileHandle.standardError.write(Data("error: \(refusal.message)\n".utf8))
            SessionLogger.shared.log("[REPLAY] refused: \(refusal.message)")
            SessionLogger.shared.shutdown()
            Darwin.exit(2)
        } catch {
            FileHandle.standardError.write(Data("replay: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[REPLAY] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(33)
        }
        let summary = "[REPLAY] done: steps=\(result.steps) positionsFed=\(result.positionsFed) gamesFed=\(result.gamesFed) rejected=\(result.gamesRejected) skipped=\(result.gamesSkipped) epochs=\(result.epochs)"
        emit(summary)
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    // MARK: - The run

    /// The whole replay run. Internal (not private) only so
    /// `ResumeEquivalenceTests` can run the real loop in-process — a full run,
    /// then the same run split by a save and a `--resume-exact` — and compare
    /// the files they end with. Production enters through `runAndExit`.
    static func runReplay(config: CorpusReplayConfig, params configuredParams: ReplayParams, abort: ReplayAbortFlag) async throws -> Result {
        // `--output` support. Only allocated when a destination was given, so a
        // run without `--output` carries no per-step recording cost at all.
        let recorder: CliTrainingRecorder? = config.output == nil ? nil : {
            let r = CliTrainingRecorder()
            r.setSessionID(config.runModelID)
            r.setRunKind(.corpusReplay)
            return r
        }()
        let runStart = CFAbsoluteTimeGetCurrent()
        // --start-model: load a saved model and continue training from it. The
        // file embeds its own architecture, which then drives both the trainer
        // and the feeder net — a start model of a different shape than the
        // current default preset trains correctly. nil → fresh random net at
        // the current preset.
        let arch: NetworkArchitecture
        let startModelFile: ModelCheckpointFile?
        let parentModelID: String
        // Set only for `--resume-exact`: the start model's full trainer state.
        // Without the flag a `--start-model` launch is a new branch — its
        // weights seed a trainer whose clock, warmup and velocity start fresh.
        var resumeSnapshot: TrainerResumeSnapshot? = nil
        if let sm = config.startModelPath {
            let url = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
            let file = try CheckpointManager.loadModelFile(at: url)
            startModelFile = file
            parentModelID = file.modelID
            arch = file.architecture
            emit("[REPLAY] start-model: \(url.lastPathComponent) modelID=\(file.modelID) encoding=\(arch.inputEncoding.rawValue)")
            if config.resumeExact {
                do {
                    resumeSnapshot = try TrainerResumeSnapshot(checkpoint: file, fileName: url.lastPathComponent)
                } catch {
                    throw CLIRunRefusal(message: "--resume-exact: \(error.localizedDescription)")
                }
                emit(PolicyTailPrecisionResume.exactResumeLogLine(
                    saved: file.metadata.trainerPolicyTailPrecision, running: config.policyTailPrecision))
            }
        } else {
            startModelFile = nil
            parentModelID = ""
            if let pn = config.presetName {
                guard let preset = NetworkArchitecture.Preset(rawValue: pn) else {
                    let names = NetworkArchitecture.Preset.allCases.map(\.rawValue).joined(separator: ", ")
                    throw CLIRunRefusal(message: "unknown --preset '\(pn)'. Available: \(names)")
                }
                arch = NetworkArchitecture.preset(preset)
                emit("[REPLAY] fresh net from preset: \(pn)")
            } else {
                arch = NetworkArchitecture.current
            }
        }

        // Startup banner: make the network type and the training hyperparameters
        // explicit in the log so a replay run is self-documenting (otherwise the
        // only clue was the input encoding). `architectureSummary` is the
        // fully-explicit form — version, encoding, block groups, heads, compute
        // dtype, and parameter count, with no silent defaults.
        let archSource = startModelFile == nil ? "default preset" : "start-model"
        emit("[REPLAY-ARCH] (\(archSource)) \(arch.architectureSummary)")
        // Numeric knobs via String(format:) (%ld for Int, %g for Double); the
        // two on/off flags are interpolated rather than passed through %@ (Swift
        // String + %@ relies on NSString bridging — avoid it).
        //
        // An exact resume trains under the checkpoint's own schedule (warmup
        // length and LR/momentum cycle), whatever `--parameters` says; each
        // field that differs is logged, and the banner below shows the
        // schedule actually in force.
        let p: ReplayParams
        if let resumeSnapshot {
            for line in configuredParams.trainer.scheduleDifferences(from: resumeSnapshot.schedule) {
                emit("[REPLAY-RESUME] WARNING \(line)")
            }
            p = try configuredParams.adoptingSchedule(resumeSnapshot.schedule)
        } else {
            p = configuredParams
        }
        let hp = p.trainer
        let hparamsLine = String(
            format: "[REPLAY-HPARAMS] lr=%.6g batch=%ld wd=%.4g momentum=%.3g gradClip=%.3g entropyBonus=%.4g drawPenalty=%.4g policyW=%.3g valueW=%.3g illegalW=%.4g ",
            Double(hp.learningRate), p.trainingBatchSize, Double(hp.weightDecayC), Double(hp.momentumCoeff), Double(hp.gradClipMaxNorm),
            Double(hp.entropyRegularizationCoeff), Double(hp.drawPenalty), Double(hp.policyLossWeight), Double(hp.valueLossWeight), Double(hp.illegalMassPenaltyWeight)
        )
            + hp.policyLabelSmoothingLogFields
            + String(
                format: " vLabelSmooth=%.4g dropout=%.4g lrWarmup=%ld bufCap=%ld replayRatio=%.3g minPrefill=%ld",
                Double(hp.valueLabelSmoothingEpsilon), Double(hp.dropoutRate),
                hp.lrWarmupSteps, p.replayBufferCapacity, p.replayRatioTarget,
                p.replayBufferMinPositionsBeforeTraining
            )
            + " complementCE=\(hp.useSignedAdvantageComplementCE ? "on" : "off")"
            + " sqrtBatchLR=\(hp.sqrtBatchScalingForLR ? "on" : "off")"
            + " batchStats=\(hp.batchStatsInterval) klProbe=\(hp.klProbeInterval)"
            + p.samplingConstraints.logFields(batchSize: p.trainingBatchSize)
        emit(hparamsLine)
        // Rolling trainer-model output file. The same file is overwritten by
        // the periodic autosave and by the final save on exit/abort, so it
        // always holds the latest weights. Destination precedence: explicit
        // --out-model; else next to --start-model; else the app's Models
        // directory named after the corpus. Overwriting during the run is
        // deliberate (the CheckpointManager never-overwrite history rule is for
        // the curated Models/Sessions store) — this is a single "latest"
        // convenience file. A file already there BEFORE the run is another
        // matter: it may be another run's only copy of hours of training, so
        // it is checked here, before the corpus scan and the network build,
        // and replaced only when it is the rolling file of the model line this
        // run continues (or with --overwrite-out-model).
        let outModelURL: URL = {
            if let explicit = config.outModelPath {
                // Always land on a `.safetensors` extension. The file is
                // safetensors-encoded, and the loaders that consume it
                // (--probe-model, --start-model) key off the extension — a
                // bare name like `corp1model` would be written verbatim and
                // then rejected as "no .safetensors found". Append it when the
                // caller didn't supply it (a supplied `.safetensors` is kept).
                let url = URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)
                return url.pathExtension.lowercased() == "safetensors"
                    ? url
                    : url.appendingPathExtension("safetensors")
            }
            if let sm = config.startModelPath {
                let smURL = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
                let stem = smURL.deletingPathExtension().lastPathComponent
                return smURL.deletingLastPathComponent().appendingPathComponent("\(stem)-replay-latest.safetensors")
            }
            // No --start-model: default into the app's Models directory — always
            // writable (even when the corpus is a read-only mounted volume),
            // keeps the corpus data dir pristine, and lands where --probe-model
            // and the GUI already look. Named after the corpus so runs over
            // different corpora don't collide on one file.
            let corpusName = config.corpusDirectories[0].lastPathComponent
            return CheckpointPaths.modelsDir.appendingPathComponent("\(corpusName)-replay-latest.safetensors")
        }()
        // Every save lands on a multiple of this (plus the final save).
        let autosaveEvery = 1000
        let rollingPlan = try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: outModelURL,
            startModelURL: config.startModelPath.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) },
            startModel: startModelFile.map {
                TrainerModelFileIdentity(modelID: $0.modelID, trainingStep: $0.metadata.trainingStep)
            },
            overwriteAuthorized: config.overwriteOutModel)
        let rollingWriter = RollingTrainerModelWriter(url: outModelURL, plan: rollingPlan)
        emit("[REPLAY] trainer-model output: \(outModelURL.path) (\(rollingPlan.logDescription))")
        // Step numbers restart in every run, so a stem reused across segments
        // would collide with the earlier segment's step files. Refuse now,
        // before any GPU work, rather than halt at the first colliding save.
        let enumeratedWriter: EnumeratedCheckpointWriter?
        if config.enumerateCheckpoints {
            let naming = EnumeratedCheckpointNaming(
                rollingOutputURL: outModelURL, runTag: EnumeratedCheckpointNaming.corpusReplayRunTag,
                segmentIndex: LineageTracker.segmentIndex(
                    exactResumeOf: resumeSnapshot != nil ? startModelFile?.lineageParent : nil))
            try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(naming: naming, stepLimit: config.stepLimit)
            enumeratedWriter = EnumeratedCheckpointWriter(naming: naming)
            emit("[REPLAY] enumerated checkpoints: \(naming.url(step: autosaveEvery).path) and siblings (never overwritten)")
        } else {
            enumeratedWriter = nil
        }

        // Resolve the corpus + resume start BEFORE building the (expensive)
        // network/trainer, so a bad --start-shard / --start-game-index (or an
        // empty corpus) fails in milliseconds instead of after a multi-second
        // MPSGraph build.
        //
        // Gather sealed shard URLs across all corpora, in stable order. Capture
        // the first corpus's id + path for the resume metadata (the common
        // single-corpus case; resume matches on corpus id, treats path as hint).
        var shardURLs: [URL] = []
        var resumeCorpusID = ""
        var resumeCorpusPath = ""
        // Read-only: replay never modifies a corpus. `GameCorpus.open` would
        // run crash recovery on any `.open` shard — truncating, sealing,
        // renaming or deleting it — which corrupts the live shard of a
        // recording or import still writing into this corpus. `.open` shards
        // are skipped and named, so the games missing from the replay are
        // visible in the log; `--validate-corpus --fix` is the explicit,
        // operator-invoked recovery for a crash leftover.
        for (di, dir) in config.corpusDirectories.enumerated() {
            let corpus = try GameCorpus.openReadOnly(directory: dir)
            let urls = corpus.sealedShardURLs
            shardURLs.append(contentsOf: urls)
            if di == 0 { resumeCorpusID = corpus.corpusID; resumeCorpusPath = dir.path }
            emit("[REPLAY] corpus \(corpus.corpusID): \(urls.count) sealed shard(s)")
            if !corpus.ignoredOpenShardURLs.isEmpty {
                let names = corpus.ignoredOpenShardURLs.map(\.lastPathComponent).joined(separator: ", ")
                let warning = "[REPLAY] WARNING corpus \(corpus.corpusID): ignoring \(corpus.ignoredOpenShardURLs.count) "
                    + "unsealed .open shard(s) (\(names)) — their games are NOT replayed. Each is either the live "
                    + "shard of a recording/import still writing into this corpus, or a crash leftover; replay "
                    + "reads sealed shards only and never modifies a corpus, so it neither reads nor recovers them. "
                    + "`--validate-corpus <dir> --fix` recovers a crash leftover and leaves a live writer's shard alone."
                emit(warning)
                FileHandle.standardError.write(Data((warning + "\n").utf8))
            }
        }
        guard !shardURLs.isEmpty else { throw CorpusReplayError.noGames }

        // Per-shard game counts via cheap trailer reads (no full-shard decode) —
        // for --start-game-index resolution, the resume `nextGame=` logging, and
        // the saved `next_game_index`. cumGames[i] = games in shards 0..<i, so
        // cumGames[i] is the global index of shard i's first game and
        // cumGames.last! the corpus total.
        let shardCounts = try shardURLs.map { try GameCorpusShardIO.readSealedTrailer(at: $0) }
        let shardSHA256 = shardCounts.map(\.sealSHA256)
        let shardGameCounts = shardCounts.map { $0.gameCount }
        let totalPlies = shardCounts.reduce(0) { $0 + $1.plyCount }
        var cumGames: [Int] = [0]
        for c in shardGameCounts { cumGames.append(cumGames.last! + c) }
        let totalCorpusGames = cumGames.last ?? 0

        // Global within-epoch game index -> (shard, within-shard offset).
        func locate(_ gi: Int) -> (shard: Int, offset: Int) {
            var s = 0
            while s + 1 < cumGames.count && cumGames[s + 1] <= gi { s += 1 }
            return (s, gi - cumGames[s])
        }

        // Resolve the resume start. Out-of-range is a HARD error (loud, not a
        // silent wrong start). All run before the network build, so a typo'd
        // resume arg never pays for it. --resume-exact (Phase 2) reconstructs the
        // buffer; --start-shard / --start-game-index (Phase 1) are the
        // approximate cold-refill resumes. Mutual exclusion is enforced at parse
        // time in DrewsChessMachineApp; the guard here is a defensive backstop.
        if config.startShard != nil && config.startGameIndex != nil {
            throw CLIRunRefusal(message: "--start-shard and --start-game-index are mutually exclusive")
        }
        var startShardCursor = 0
        var startWithinShardSkip = 0
        var reconstructUntil: Int? = nil   // resume-exact: refeed up to here, then train
        var startEpoch = 0
        // Epoch the reconstruction refeed starts in: the saved epoch, or the
        // one before it when the buffer's oldest residents were fed before
        // the saved epoch began.
        var reconstructStartEpoch = 0
        // Positions fed per trainer step: K = batchSize / R, so each position
        // is sampled ~R times before eviction. Fixed for the run.
        let perStepFeed = max(1, Int((Double(max(1, p.trainingBatchSize)) / p.replayRatioTarget).rounded()))
        // Budget: explicit step limit wins; otherwise bound by epochs (default
        // a single pass when neither is given). The epoch budget counts passes
        // from the start of the run's lineage, not of this segment: an exact
        // resume continues the saved run's epoch count, so it needs `--epochs`
        // above the epoch it was saved in (checked below).
        let stepLimit = config.stepLimit
        let epochLimit: Int? = config.epochs ?? (stepLimit == nil ? 1 : nil)
        // An exact resume's buffer fill as saved, which the rebuilt buffer
        // must match; nil when it cannot be checked (logged where decided).
        var expectedRebuiltBufferPositions: Int? = nil
        // An exact resume's saved feed phase (positions fed ahead of the
        // next step's target); nil when the checkpoint does not carry it.
        var resumedFeedAheadPositions: Int? = nil
        // The run's master seed: the configured or drawn one, or — for an
        // exact resume of a checkpoint that records the run's streams — the
        // run's own, with every stream position it saved.
        var runSeed = config.runRandomSeed
        var resumedStreams: LineageRecord.RunStreams? = nil
        var resumeGaps: [ResumeGap] = []
        if config.resumeExact {
            // Read the saved resume metadata from the --start-model header (no
            // tensor decode) and validate it's for THIS corpus.
            guard let smPath = config.startModelPath else {
                throw CLIRunRefusal(message: "--resume-exact requires --start-model")
            }
            let smURL = URL(fileURLWithPath: (smPath as NSString).expandingTildeInPath)
            let rm: SafetensorsModelIO.ReplayResumeMetadata
            do {
                rm = try SafetensorsModelIO.replayResumePoint(at: smURL)
            } catch {
                throw CLIRunRefusal(message: "--resume-exact: \(smURL.lastPathComponent): \(error)")
            }
            guard rm.corpusID == resumeCorpusID else {
                throw CLIRunRefusal(message: "--resume-exact: checkpoint corpus_id '\(rm.corpusID)' != this corpus '\(resumeCorpusID)'")
            }
            guard let startFile = startModelFile, let snapshot = resumeSnapshot else {
                preconditionFailure("--resume-exact loaded its start model and trainer snapshot above")
            }
            let parentRecord = startFile.lineageParent.lineage.record
            resumeGaps += ResumeGap.dropoutGaps(restoring: snapshot.dropoutRNG)
            resumeGaps += PolicyTailPrecisionResume.gaps(
                saved: startFile.metadata.trainerPolicyTailPrecision, running: config.policyTailPrecision)
            if let parentRecord, let corpus = parentRecord.fed.corpus {
                // The corpus must be the one the checkpoint fed: same shards,
                // same content. A changed corpus is not a gap anyone can
                // accept — the refeed and every later game would differ.
                guard corpus.shardSHA256 == shardSHA256 else {
                    let changed = zip(corpus.shardSHA256, shardSHA256).enumerated()
                        .filter { $0.element.0 != $0.element.1 }.map { shardURLs[$0.offset].lastPathComponent }
                    let detail = corpus.shardSHA256.count != shardSHA256.count
                        ? "the checkpoint fed \(corpus.shardSHA256.count) sealed shard(s), this corpus has \(shardSHA256.count)"
                        : "changed shard(s): \(changed.joined(separator: ", "))"
                    throw CLIRunRefusal(message: "--resume-exact: corpus \(resumeCorpusID) is not the content the checkpoint fed (\(detail)); a changed corpus cannot be resumed exactly, even with --accept-inexact")
                }
                if corpus.feedPerStep == perStepFeed {
                    resumedFeedAheadPositions = corpus.feedAheadPositions
                } else {
                    emit("[RESUME] feed per step: checkpoint \(corpus.feedPerStep), this run \(perStepFeed) (batch size or replay ratio changed) — the feed phase cannot continue")
                    resumeGaps += [.params, .feedCarry]
                }
                if let streams = parentRecord.rng.streams {
                    do {
                        runSeed = try RunRandomSeed.inherited(
                            from: streams,
                            configuredSeed: config.runRandomSeed.configuredSeed,
                            commandLineSeed: config.runRandomSeed.origin == .commandLine ? config.runRandomSeed.masterSeed : nil)
                    } catch {
                        throw CLIRunRefusal(message: "--resume-exact: \(error.localizedDescription)")
                    }
                    resumedStreams = streams
                } else {
                    resumeGaps.append(.rngSampler)
                }
                if let parentParameters = parentRecord.parameters {
                    for line in try ParameterDifference.exactResumeLogLines(parent: parentParameters, inForce: p.parameters) {
                        emit(line)
                    }
                } else {
                    resumeGaps.append(.params)
                }
                let environment = ResumeGap.environmentGaps(
                    writtenBy: parentRecord, runningBuild: .current, runningDevice: .current,
                    runningFingerprint: try await BehaviorFingerprint.compute(
                        for: .init(arch: arch, policyTailPrecision: config.policyTailPrecision)))
                for line in environment.logLines { emit(line) }
                resumeGaps += environment.gaps
            } else {
                // Written before lineage: no streams, feed phase, shard
                // hashes or parameter snapshot to continue from.
                resumeGaps += [.rngSampler, .feedCarry, .params]
                if let g = rm.builtByGit, g != BuildInfo.gitHash {
                    emit("[RESUME] WARNING checkpoint built by git \(g) but running \(BuildInfo.gitHash)")
                }
            }
            // The rebuilt buffer must hold what the saved one held. A refeed
            // that ends where the saved run's feed ended holds the same last
            // positions whenever it holds as many, so the saved fill is the
            // check. A different capacity cannot hold what the saved buffer
            // held at all: a parameter gap, not something to check.
            if let savedCapacity = rm.capacity, savedCapacity != p.replayBufferCapacity {
                emit("[RESUME] replay buffer capacity: checkpoint \(savedCapacity), this run \(p.replayBufferCapacity) "
                    + "— the rebuilt buffer cannot hold what the saved one held")
                resumeGaps.append(.params)
            } else if let savedPositions = rm.populatedPlies, rm.capacity != nil {
                expectedRebuiltBufferPositions = savedPositions
            } else {
                emit("[RESUME] buffer fill not checked: the checkpoint does not record its replay buffer's fill "
                    + "and capacity, so the rebuilt buffer cannot be compared with the saved one")
            }
            // Refeed enough games before `until` to overflow the ring, which then
            // self-trims to the exact last-capacity plies (so the precise refeed
            // start doesn't matter as long as it covers >= capacity FED plies).
            // Walk back by capacity/avgPly games with a 1.5x margin so skips and
            // local short games can't under-fill. avgPly is the corpus's actual
            // mean (trailer plies / games).
            //
            // The saved position must be one in this corpus: a save always
            // records a next game inside its epoch (`resumePoint()` folds the
            // end of an epoch forward to game 0 of the next), so anything else
            // is a corrupt or hand-edited file — or one written before that
            // normalization, at next game == the corpus total — and is
            // refused rather than pulled into range.
            guard rm.epoch >= 0, (0..<totalCorpusGames).contains(rm.nextGameIndex) else {
                throw CorpusReplayError.exactResumePositionInvalid(
                    epoch: rm.epoch, nextGameIndex: rm.nextGameIndex, totalGames: totalCorpusGames)
            }
            let until = rm.nextGameIndex
            reconstructUntil = until
            startEpoch = rm.epoch
            // A run whose epoch budget the checkpoint's run already spent has
            // nothing to train. Refused here, before any GPU work: run on, it
            // would end at its first step and save a corpus position behind
            // its parent's (the refeed's own wrap would have reset it), so a
            // later resume would train those games again. With the budget
            // ahead of the saved epoch, the refeed — which ends in the saved
            // epoch — can never reach it.
            if let el = epochLimit, startEpoch >= el {
                throw CorpusReplayError.exactResumeEpochBudgetSpent(savedEpoch: startEpoch, epochLimit: el)
            }
            let cap = max(1, p.replayBufferCapacity)
            let avgPly = max(1.0, Double(totalPlies) / Double(totalCorpusGames))
            let gamesBack = Int((1.5 * Double(cap) / avgPly).rounded(.up))
            // The refeed window: the games before `until` that cover the
            // ring. When the saved epoch had not yet fed that many, the window
            // wraps back into the previous epoch's tail — corpus order is
            // sequential, so those are exactly the games the run fed then
            // (C1 #6). Epoch 0 has no previous epoch: games [0, until) are the
            // whole history, full or legitimately partial.
            let reconstructStart: Int
            if startEpoch > 0 && until < gamesBack {
                let fromPreviousEpoch = gamesBack - until
                guard fromPreviousEpoch <= totalCorpusGames else {
                    throw CLIRunRefusal(message: "--resume-exact: the buffer holds more than one whole epoch of this corpus (\(gamesBack)-game window, \(totalCorpusGames) games); reconstructing it would span several epochs, which is not supported")
                }
                reconstructStartEpoch = startEpoch - 1
                reconstructStart = totalCorpusGames - fromPreviousEpoch
            } else {
                reconstructStartEpoch = startEpoch
                reconstructStart = max(0, until - gamesBack)
            }
            let (s, off) = locate(reconstructStart)
            startShardCursor = s
            startWithinShardSkip = off
            SessionLogger.shared.log("[REPLAY] --resume-exact: nextGame=\(until) epoch=\(startEpoch) cap=\(cap) savedPlies=\(rm.populatedPlies.map(String.init) ?? "unrecorded") -> refeed from epoch \(reconstructStartEpoch) game \(reconstructStart) to epoch \(startEpoch) game \(until) (window \(gamesBack) games ≈ \(Int(Double(gamesBack) * avgPly)) plies) from \(shardURLs[s].lastPathComponent) offset \(off)")
        } else if let ss = config.startShard {
            guard ss >= 0 && ss < shardURLs.count else {
                throw CLIRunRefusal(message: "--start-shard \(ss) out of range; valid 0…\(shardURLs.count - 1)")
            }
            startShardCursor = ss
            let skipDesc = ss == 0 ? "no shards skipped" : "skipping shards 0…\(ss - 1), \(cumGames[ss]) games"
            emit("[REPLAY] --start-shard \(ss) -> \(shardURLs[ss].lastPathComponent) (\(skipDesc))")
        } else if let gi = config.startGameIndex {
            guard gi >= 0 && gi < totalCorpusGames else {
                throw CLIRunRefusal(message: "--start-game-index \(gi) out of range; valid 0…\(totalCorpusGames - 1)")
            }
            let (s, off) = locate(gi)
            startShardCursor = s
            startWithinShardSkip = off
            emit("[REPLAY] --start-game-index \(gi) -> \(shardURLs[s].lastPathComponent) offset \(off) (skipping \(gi) games on the first pass)")
        }
        // Global within-epoch index of the resume start (first game fed). For
        // --resume-exact this is the refeed start; the run continues from
        // `reconstructUntil` once the buffer is rebuilt.
        let startGlobalIndex = cumGames[startShardCursor] + startWithinShardSkip

        // One exactness decision for the resume (plan C3): logged once,
        // recorded in the segment's lineage, and refused unless
        // `--accept-inexact` names every gap (D-7).
        let resumeExactness = startModelFile.map { ResumeExactness.resume(of: $0.lineageParent, gaps: resumeGaps) }
        if config.resumeExact, let resumeExactness {
            emit(resumeExactness.logLine)
            if let refusal = resumeExactness.refusal(accepting: config.acceptInexact) {
                throw CLIRunRefusal(message: refusal)
            }
        }
        for line in runSeed.parameterNotes { emit(line) }
        recorder?.setRunRandomSeed(runSeed)
        recorder?.setSamplingConstraints(p.samplingConstraints, batchSize: p.trainingBatchSize)

        emit("[REPLAY] building network + trainer (encoding=\(arch.inputEncoding.rawValue))")
        // With --start-model the file's weights and batch-norm statistics
        // replace both networks' right below, so they draw nothing; otherwise
        // the run builds a fresh model whose init seed derives from the run
        // seed, so `--seed` reproduces its initialization.
        //
        // The segment's lineage follows from the same choice: a fresh run
        // drawn under that init seed, a new branch from the start model, or
        // the start model's run continued. Every save carries the record, so
        // totals (trainer steps, games and positions fed, measured step time)
        // continue across resumes without hand-entered bases.
        let netInitMode: NetworkInitMode
        let trainerInitialization: WeightInitialization
        let lineageStart: LineageTracker.Start
        if let file = startModelFile {
            netInitMode = .overwrittenByLoad
            trainerInitialization = .overwrittenByLoad
            lineageStart = resumeSnapshot != nil
                ? .resume(parent: file.lineageParent, gaps: resumeGaps, legacyTotals: nil)
                : .branch(parent: file.lineageParent)
        } else {
            let initSeed = runSeed.streams.freshModelInitSeed
            emit("[REPLAY] fresh model init_seed=\(initSeed) init_scheme=\(WeightInitScheme.current) "
                + "(from run seed \(runSeed.masterSeed))")
            netInitMode = .randomWeights(initSeed: initSeed)
            trainerInitialization = .seeded(initSeed: initSeed)
            lineageStart = .fresh(initialization: ModelInitRecord(initSeed: initSeed, scheme: WeightInitScheme.current))
        }
        let net = try ChessMPSNetwork(netInitMode, arch: arch)
        // Configured through `TrainerHyperparameters` — the same path the GUI
        // session uses — so this trainer gets every trainer-level parameter,
        // including the LR/momentum cycle and its decay envelope, dropout, and
        // the stats / KL-probe intervals. With both cycle flags off the cycle
        // is inert and the static LR and momentum apply, exactly as in the GUI.
        let trainer = try ChessTrainer(
            dropoutStream: runSeed.streams.generator(.dropout),
            hyperparameters: hp, arch: arch, initialization: trainerInitialization,
            policyTailPrecision: config.policyTailPrecision)
        emit(ChessNetwork.PolicyTailPrecision.processLogLine)
        // A requested GPU capture must be possible before any buffer fill or
        // training is spent on the run (the capture itself starts at its step).
        if let capture = config.gpuCapture {
            try validateGPUCaptureRequest(capture, stepLimit: config.stepLimit)
            guard MTLCaptureManager.shared().supportsDestination(.gpuTraceDocument) else {
                throw CorpusReplayError.gpuCaptureUnavailable
            }
        }
        let buffer = ReplayBuffer(
            capacity: p.replayBufferCapacity,
            inputEncoding: net.inputEncoding,
            sampler: runSeed.streams.generator(.sampler))
        // Replay samples under the same constraints the parameters give a
        // self-play run; a replay run takes one snapshot of them at start.
        buffer.setSamplingConstraints(p.samplingConstraints)
        if let resumedStreams {
            // The sampler continues from where the saved run's next batch
            // would have drawn; the refeed below only appends, never draws.
            buffer.restoreSamplerState(resumedStreams.samplerState)
            emit("[RESUME] rng: sampler=restored")
        }
        let feeder = CorpusReplayFeeder(network: net, buffer: buffer)

        // Seed the feeder net (computes the value baseline while feeding) from
        // the start model's own tensors — trainables + BN running stats, the
        // prefix of a trainer-state file that also carries optimizer velocity.
        // The feeder is a plain inference network (no masters/velocity), so a
        // direct `loadWeights` is correct there.
        //
        // The trainer, by launch kind:
        // - `--resume-exact`: `restoreExactly` — fp32 masters, working copy,
        //   optimizer velocity, the completed-step clock, warmup length and
        //   cycle, all from the checkpoint. Training continues exactly as if
        //   the previous segment had never stopped; warmup does not re-run.
        // - `--start-model` alone: a new branch. `loadBaseWeightsResetVelocity`
        //   seeds the working copy and the fp32 masters from the file (a bare
        //   `network.loadWeights` would leave the masters at random init and the
        //   first SGD step would overwrite the loaded weights with them) and
        //   zeros velocity; the clock stays at zero, so warmup and the cycle
        //   start from their beginnings.
        if let file = startModelFile {
            let baseCount = net.network.trainableVariables.count + net.network.bnRunningStatsVariables.count
            guard file.weights.count >= baseCount else {
                throw CorpusReplayError.startModelTooSmall(have: file.weights.count, need: baseCount)
            }
            try await net.network.loadWeights(file.networkWeights)
            if let resumeSnapshot {
                try await trainer.restoreExactly(from: resumeSnapshot)
                if let resumedStreams {
                    try await trainer.restoreDropoutStreamState(resumedStreams.dropoutStreamState)
                    emit("[RESUME] rng: dropout stream=restored")
                }
                emit("[REPLAY] start-model trainer state restored exactly (fp32 masters, velocity, trainerStep=\(trainer.completedTrainSteps)) + feeder net (base tensors=\(baseCount))")
            } else {
                try await trainer.loadBaseWeightsResetVelocity(file.networkWeights)
                emit("[REPLAY] start-model weights loaded into trainer (working+masters, velocity zeroed; new branch) + feeder net (base tensors=\(baseCount))")
            }
        }
        // The resolved LR/momentum schedule, once, with the trainer step its
        // phase and decay continue from. `lr=off` / `mom=off` means that
        // channel trains at the static `lr=` / `momentum=` in the banner;
        // anything else overrides them, so a misconfigured schedule is visible
        // here rather than only in the loss curve.
        let launch: TrainerLaunchKind = resumeSnapshot != nil
            ? .exactResume(ofModelID: parentModelID)
            : (startModelFile != nil ? .newBranch(fromModelID: parentModelID) : .fresh)
        emit("[REPLAY-CYCLE] \(LRMomentumCycleLogFormat.cycleDescription(trainer.lrMomentumCycle)) "
            + LRMomentumCycleLogFormat.scheduleOrigin(of: trainer, launch: launch))

        let lineageTracker = try LineageTracker(
            start: lineageStart, pathKind: .replay, argv: CommandLine.arguments,
            startedAt: Date(), segmentStartTrainerStep: trainer.completedTrainSteps)
        emit(RunProvenanceLine.line(
            record: try lineageTracker.startRecord(at: Date(), trainerCompletedSteps: trainer.completedTrainSteps,
                                                   parameters: p.lineageParameters),
            seed: runSeed))

        // Export the trainer's complete state and overwrite the rolling
        // output file. Failure handling splits on cause (see reportSaveFailure):
        // a disk-full (ENOSPC) failure is FATAL — it alarms and throws so the run
        // halts rather than training on into a window where nothing persists; any
        // other failure is a WARNING the first time, and halts the run when the
        // same kind of save fails again at its next attempt
        // (`TrainerSaveFailureStreak`) — a run that cannot save at all must not
        // train on keeping nothing.
        // Resume info is passed in (not captured): the corpus index / stream
        // cursor and the feed counters are resolved AFTER this nested func, so
        // the call sites — which run inside the SGD loop where those are in
        // scope — supply them. They land in the file's lineage record, whose
        // corpus position the exact-reconstruction resume reads back and whose
        // build stamp lets a resume warn about a different encoder/feeder.
        var rollingSaveFailures = TrainerSaveFailureStreak(what: "trainer-model save")
        var enumeratedSaveFailures = TrainerSaveFailureStreak(what: "enumerated checkpoint save")
        func saveTrainerModel(step: Int, reason: String,
                              nextGameIndex: Int, shard: Int, epoch: Int, populatedPlies: Int,
                              corpusID: String, corpusPath: String,
                              segmentGames: Int, segmentPositions: Int,
                              feedAheadPositions: Int) async throws {
            // Rolling save (overwrites the output file). `encoded` is reused by the
            // enumerated copy below, so it outlives this do/catch. A disk-full
            // failure re-throws (fatal, halts the run); any other failure is a
            // non-fatal WARNING and we skip the enumerated copy (it would fail too).
            let encoded: Data
            do {
                // The complete trainer state — fp32 masters, optimizer velocity
                // and the schedule clock — so any of these files can be
                // continued exactly with `--resume-exact`. `training_step`
                // stays segment-local (the replay tracker adds each segment's
                // `cumstep_base` to it); the cumulative clock is
                // `trainer_completed_steps`. The loop is sequential, so no SGD
                // step is in flight during the export.
                let snapshot = try await trainer.exportResumeSnapshot()
                // The run's stream positions, in the same cut as the trainer
                // state: the loop is sequential, so no batch is drawn and no
                // step runs between these reads.
                let streams = runSeed.runStreams(
                    samplerState: buffer.samplerState(),
                    dropoutStreamState: try await trainer.dropoutStreamState(),
                    nextGameSerial: nil, arenasStarted: nil, opponentGameIndices: nil)
                let weights = snapshot.trainerWeights
                let metadata = ModelCheckpointMetadata.trainerFile(
                    creator: "replay",
                    trainingStep: step,
                    parentModelID: parentModelID,
                    notes: "corpus replay \(reason) @ step \(step)",
                    schedule: snapshot.schedule,
                    policyTailPrecision: trainer.policyTailPrecision
                )
                // The corpus position (what `--resume-exact` resumes from)
                // and the build that wrote it travel in the lineage record.
                let saveDate = Date()
                let lineage = try lineageTracker.record(
                    at: saveDate,
                    trainerCompletedSteps: snapshot.schedule.completedTrainSteps,
                    segmentLocalStep: step,
                    segmentGames: segmentGames,
                    segmentPositions: segmentPositions,
                    corpus: LineageRecord.CorpusPosition(
                        corpusID: corpusID, corpusPath: corpusPath, epoch: epoch,
                        nextGameIndex: nextGameIndex, shard: shard,
                        populatedPlies: populatedPlies, bufferCapacity: p.replayBufferCapacity,
                        feedAheadPositions: feedAheadPositions, feedPerStep: perStepFeed,
                        shardSHA256: shardSHA256),
                    parameters: p.lineageParameters,
                    rng: LineageRecord.RNG(
                        dropoutPhiloxState: snapshot.dropoutRNG.philoxState, streams: streams,
                        behaviorFingerprint: try await BehaviorFingerprint.compute(
                            for: .init(arch: arch, policyTailPrecision: trainer.policyTailPrecision))))
                encoded = try SafetensorsModelIO.encode(
                    modelID: config.runModelID,
                    createdAtUnix: Int64(saveDate.timeIntervalSince1970),
                    metadata: metadata,
                    weights: weights,
                    architecture: arch,
                    includesVelocity: true,
                    lineage: lineage
                )
                try FileManager.default.createDirectory(
                    at: outModelURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try rollingWriter.write(encoded)
                rollingSaveFailures.recordSuccess()
                recorder?.recordSave(of: lineage, savedAt: outModelURL, log: emit)
                emit("[REPLAY] saved trainer model (\(reason)) step=\(step) trainerStep=\(snapshot.schedule.completedTrainSteps) nextGame=\(nextGameIndex) shard=\(shard) epoch=\(epoch) -> \(outModelURL.lastPathComponent)")
                // Full layer health of exactly the state just written, from the
                // tensors already exported for it (no extra GPU read). Never
                // throws: a failed pass logs a [LAYER-HEALTH] failure line and
                // the save stands.
                let health = await LayerHealthLog.checkpoint(
                    arch: arch, trainerWeights: weights, context: "replay-\(reason)",
                    step: step, trainerStep: snapshot.schedule.completedTrainSteps)
                for line in health.lines { emit(line) }
                if let summary = health.summary {
                    recorder?.appendLayerHealth(CliTrainingRecorder.LayerHealthRecord(
                        step: step, trainerStep: snapshot.schedule.completedTrainSteps,
                        context: "replay-\(reason)", summary: summary))
                }
            } catch let ownershipRefusal as FileSafetyError where ownershipRefusal.isOwnershipRefusal {
                // The rolling path no longer holds the file this run owns
                // (another file or a folder is there now). Halt: writing on
                // would overwrite something this run did not write.
                throw ownershipRefusal
            } catch {
                // Throws on disk-full (halt), and on the second failure in a
                // row; otherwise returns (non-fatal).
                try Self.reportSaveFailure(error, step: step, what: "trainer-model save (\(reason))")
                try rollingSaveFailures.recordFailure(step: step)
                return
            }

            // Optional: also drop a step-enumerated copy so no checkpoint is lost
            // to the rolling overwrite. Reuses `encoded` (no re-export). A separate
            // do/catch so a disk-full throw here propagates OUT — it must not be
            // caught by the rolling-save catch above (which would misclassify our
            // own halt error as "some other failure" and swallow it). A file this
            // run did not write at the step's name is a hard error, never an
            // overwrite.
            if let enumeratedWriter {
                do {
                    let written = try enumeratedWriter.write(encoded, step: step)
                    enumeratedSaveFailures.recordSuccess()
                    let note = written.outcome == .replacedThisRunsEarlierSave
                        ? " (replaced this run's own earlier save of step \(step))"
                        : ""
                    emit("[REPLAY] enumerated checkpoint -> \(written.url.lastPathComponent)\(note)")
                } catch let collision as TrainerOutputFileError {
                    throw collision
                } catch let ownershipRefusal as FileSafetyError where ownershipRefusal.isOwnershipRefusal {
                    throw ownershipRefusal
                } catch {
                    try Self.reportSaveFailure(error, step: step, what: "enumerated checkpoint")
                    try enumeratedSaveFailures.recordFailure(step: step)
                }
            }
        }

        let batchSize = max(1, p.trainingBatchSize)
        let reuse = p.replayRatioTarget
        let minPrefill = max(batchSize, p.replayBufferMinPositionsBeforeTraining)

        emit("[REPLAY] batchSize=\(batchSize) reuse=\(String(format: "%.2f", reuse)) K=\(perStepFeed) minPrefill=\(minPrefill) stepLimit=\(stepLimit.map(String.init) ?? "none") epochLimit=\(epochLimit.map(String.init) ?? "none")")

        // Streaming game source, cycling the shard list for epochs.
        var shardCursor = startShardCursor
        var currentGames: [GameRecord] = []
        var gameCursor = 0
        // Resume-exact starts in the refeed's first epoch (the saved one, or
        // the one before it); otherwise 0.
        var epochsCompleted = reconstructStartEpoch
        // One-shot within-shard skip applied to the FIRST loaded shard
        // (--start-game-index); cleared after that shard and on any epoch wrap.
        var firstShardSkip = startWithinShardSkip
        // Global within-epoch index of the NEXT game nextGame() will return:
        // starts at the resume point, +1 per returned game, resets to 0 on the
        // epoch wrap. Logged as `nextGame=` and saved as `next_game_index`.
        var nextGameWithinEpoch = startGlobalIndex
        // An exact resume reproduces the fed stream only if every shard reads
        // as the content its hash names; a shard that fails to read ends such
        // a run with this error instead of being skipped.
        var exactResumeShardFailure: Error? = nil

        func nextGame() -> GameRecord? {
            while gameCursor >= currentGames.count {
                if shardCursor >= shardURLs.count {
                    // Wrap to a fresh epoch. Reset the cursors BEFORE the
                    // epoch-limit return, so a run that completes its budget
                    // leaves a consistent (nextGame=0, shard=0, epoch incremented)
                    // resume point rather than (nextGame=totalGames, shard=count)
                    // — the latter is one past the end, and `--resume-exact`
                    // refuses it as outside the corpus.
                    epochsCompleted += 1
                    shardCursor = 0
                    nextGameWithinEpoch = 0   // fresh epoch starts at game 0…
                    firstShardSkip = 0        // …and the one-shot skip is spent
                    if let el = epochLimit, epochsCompleted >= el { return nil }
                }
                let url = shardURLs[shardCursor]
                shardCursor += 1
                do {
                    currentGames = try GameCorpusShardIO.readSealed(at: url).games
                } catch {
                    if config.resumeExact {
                        exactResumeShardFailure = CorpusReplayError.exactResumeShardUnreadable(
                            shard: url.lastPathComponent, detail: error.localizedDescription)
                        return nil
                    }
                    emit("[REPLAY] skipping unreadable shard \(url.lastPathComponent): \(error.localizedDescription)")
                    currentGames = []
                }
                gameCursor = 0
                if firstShardSkip > 0 {
                    // offset < this shard's game count by construction (locate),
                    // so this never lands past the end.
                    gameCursor = min(firstShardSkip, currentGames.count)
                    firstShardSkip = 0
                }
                // Re-anchor the resume counter to the TRUE global index of the
                // next game in the shard just loaded (cumGames[loaded] +
                // gameCursor). The +1-per-game advance below only counts games
                // actually returned, but cumGames counts every shard — so if an
                // unreadable shard was skipped above (currentGames=[]), the
                // running counter would drift low by that shard's game count.
                // Deriving from cumGames here keeps it exact across skips with no
                // dependence on the skipped shard's size. (shardCursor was already
                // incremented past the loaded shard, so loaded == shardCursor-1.)
                nextGameWithinEpoch = cumGames[shardCursor - 1] + gameCursor
            }
            let g = currentGames[gameCursor]
            gameCursor += 1
            nextGameWithinEpoch += 1
            return g
        }

        // Normalize the streaming cursor into a valid (nextGame, shard, epoch)
        // tuple for a save. After the final game of an epoch is consumed,
        // nextGameWithinEpoch sits at totalCorpusGames — one past the end — until
        // the NEXT nextGame() call performs the wrap. A save taken in that window
        // (a step-limit or abort breaking the loop right at an epoch boundary)
        // would otherwise record next_game_index == totalCorpusGames and
        // locate(...).shard == shardURLs.count, both outside the positions
        // `--resume-exact` accepts. Fold the boundary state forward to the
        // start of the next epoch — (game 0, shard 0, epoch + 1) — exactly as
        // nextGame()'s wrap does, so the saved resume point is always consistent.
        func resumePoint() -> (nextGame: Int, shard: Int, epoch: Int) {
            if nextGameWithinEpoch >= totalCorpusGames {
                return (0, 0, epochsCompleted + 1)
            }
            return (nextGameWithinEpoch, locate(nextGameWithinEpoch).shard, epochsCompleted)
        }

        var feedTally = CorpusReplayFeedTally()
        var corpusExhausted = false

        // Feed one game and count it. A game whose moves cannot be replayed is
        // reported on its own line — never dropped silently — with its
        // within-epoch index (the game nextGame() just returned).
        func feedAndCount(_ game: GameRecord) {
            if let rejection = feedTally.record(feeder.feed(game)) {
                emit("[REPLAY-ERR] epoch=\(epochsCompleted) game=\(nextGameWithinEpoch - 1) \(rejection)")
            }
        }

        // Pre-fill — or, for --resume-exact, RECONSTRUCT: refeed games up to the
        // saved next_game_index so the fixed-capacity ring ends holding exactly
        // the last-capacity plies the original run had there (the surplus is
        // overwritten). Training then continues from next_game_index. The refeed
        // starts in the saved epoch, or in the previous one when its window
        // reaches back across the wrap; it ends in the saved epoch, which is
        // below the epoch budget (checked at launch), so its own wrap never
        // ends the feed.
        if let until = reconstructUntil {
            // Positions in the corpus's epoch-linear order: a game index
            // counted across epochs, where the end of epoch e is the start of
            // epoch e + 1. Refeed until the saved resume point.
            let target = startEpoch * totalCorpusGames + until
            while epochsCompleted * totalCorpusGames + nextGameWithinEpoch < target {
                // Running out of games before the saved point leaves a buffer
                // the saved run never had: an error, never a short resume.
                guard let g = nextGame() else {
                    if let exactResumeShardFailure { throw exactResumeShardFailure }
                    throw CorpusReplayError.exactResumeRefeedEndedEarly(
                        reachedEpoch: epochsCompleted, reachedGame: nextGameWithinEpoch,
                        targetEpoch: startEpoch, targetGame: until)
                }
                feedAndCount(g)
            }
            SessionLogger.shared.log("[REPLAY] --resume-exact: buffer reconstructed bufCount=\(buffer.count)/\(p.replayBufferCapacity) (refed \(feedTally.games) games / \(feedTally.positions) plies;\(feedTally.countsSuffix)); resuming at game \(nextGameWithinEpoch) epoch \(epochsCompleted)")
            // The refeed covers the last capacity's worth of positions before
            // the saved point, but not where the saved run began: a run
            // started part-way into the corpus (or one whose window held
            // locally short games) can rebuild a buffer with more, or fewer,
            // positions than it saved. Both feeds end at the same game, so
            // equal counts mean equal contents.
            if let expected = expectedRebuiltBufferPositions, buffer.count != expected {
                throw CorpusReplayError.exactResumeBufferMismatch(
                    savedPositions: expected, rebuiltPositions: buffer.count, capacity: p.replayBufferCapacity)
            }
        } else {
            while buffer.count < minPrefill {
                guard let g = nextGame() else { corpusExhausted = true; break }
                feedAndCount(g)
            }
        }
        // The feed target of step j is `base + j × K`. A fresh run's base is
        // its prefill; an exact resume's continues the saved phase, so the
        // games fed before every later step are the uninterrupted run's.
        let feedPhase: CorpusFeedPhase
        if let resumedFeedAheadPositions {
            feedPhase = .continuing(reconstructedFedPositions: feedTally.positions,
                                    savedFeedAheadPositions: resumedFeedAheadPositions, perStep: perStepFeed)
        } else {
            feedPhase = .starting(fedPositions: feedTally.positions, perStep: perStepFeed)
        }
        func feedAheadPositions(atStep step: Int) -> Int {
            feedPhase.feedAhead(fedPositions: feedTally.positions, step: step)
        }
        // A resume's reconstruction refeeds games its parent already fed;
        // the segment's own feed starts after them.
        let reconstructionFed = reconstructUntil != nil
            ? (games: feedTally.games, positions: feedTally.positions)
            : (games: 0, positions: 0)
        emit("[REPLAY] pre-filled: bufCount=\(buffer.count) positionsFed=\(feedTally.positions) gamesFed=\(feedTally.games)\(feedTally.countsSuffix)")

        // Format a possibly-not-measured diagnostic. The trainer only computes
        // the diagnostic bundle (entropy, value W/D/L, played-move prob, illegal
        // mass) on its diagnostic-cadence steps, leaving the field NaN otherwise
        // (e.g. the first logged step). Render those as "--" rather than "nan"
        // so the line stays readable.
        func dg(_ v: Float, _ digits: Int) -> String {
            v.isFinite ? String(format: "%.\(digits)f", v) : "--"
        }

        // Step-locked SGD loop.
        var step = 0
        let logEvery = 50
        var aborted = false
        // `--gpu-capture-step` bookkeeping: whether the capture started, and
        // the error if it could not.
        var gpuCaptureStarted = false
        var gpuCaptureFailure: Error? = nil
        while true {
            // Ctrl-C: stop cleanly before starting another step so the
            // post-loop save captures a complete, non-mid-step state.
            if abort.isRequested {
                aborted = true
                emit("[REPLAY] abort requested — stopping at step \(step)")
                break
            }
            if let sl = stepLimit, step >= sl { break }
            let targetFed = feedPhase.target(step: step)
            while feedTally.positions < targetFed && !corpusExhausted {
                guard let g = nextGame() else { corpusExhausted = true; break }
                feedAndCount(g)
            }
            if let exactResumeShardFailure { throw exactResumeShardFailure }
            if corpusExhausted && epochLimit != nil { break }

            // Optional one-step GPU trace (`--gpu-capture-step`). Started right
            // before the step and stopped right after it; trainStep returns
            // only once the step's GPU work has completed, so the trace holds
            // the whole step. Stopped on every exit path.
            let captureThisStep = config.gpuCapture.flatMap { $0.step == step + 1 ? $0 : nil }
            if let captureThisStep {
                do {
                    try beginGPUCapture(captureThisStep, device: trainer.network.commandQueue.device)
                    gpuCaptureStarted = true
                } catch {
                    // The capture fails before this step trains, so the
                    // trainer holds a complete state: stop here, let the final
                    // save below keep everything since the last autosave, and
                    // fail the run after it.
                    gpuCaptureFailure = error
                    let msg = "[ALARM] [REPLAY] GPU trace capture for step \(captureThisStep.step) could not start: "
                        + "\(error.localizedDescription). Stopping before that step; the final save follows, then the run fails."
                    FileHandle.standardError.write(Data((msg + "\n").utf8))
                    SessionLogger.shared.log(msg)
                    break
                }
            }
            let stepTiming: TrainStepTiming?
            do {
                stepTiming = try await trainer.trainStep(replayBuffer: buffer, batchSize: batchSize)
            } catch {
                if captureThisStep != nil {
                    MTLCaptureManager.shared().stopCapture()
                }
                throw error
            }
            if let captureThisStep {
                MTLCaptureManager.shared().stopCapture()
                emit("[REPLAY] GPU trace of step \(captureThisStep.step) written to \(captureThisStep.outputURL.path)")
            }
            guard let timing = stepTiming else {
                emit("[REPLAY] trainStep returned nil (bufCount=\(buffer.count)); stopping")
                break
            }
            lineageTracker.recordTrainingStep(totalMs: timing.totalMs)
            step += 1
            if step == 1 || step % logEvery == 0 {
                // Live, warmup-adjusted LR read from the trainer (single source
                // of truth — don't re-derive the warmup formula here).
                // Pin LR, momentum and the cycle values to one step-count
                // observation so the three agree with each other.
                let observedSteps = trainer.completedTrainSteps
                let liveLR = trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: observedSteps)
                let liveMomentum = trainer.effectiveMomentum(completedSteps: observedSteps)
                let cycleValues = trainer.lrMomentumCycleValues(completedSteps: observedSteps)
                let line = "[REPLAY] step=\(step)"
                    + String(format: " loss=%.4f pLoss=%.4f vLoss=%.4f", timing.loss, timing.policyLoss, timing.valueLoss)
                    + " pEnt=\(dg(timing.policyEntropy, 3)) pIllM=\(dg(timing.illegalMassPenalty, 4))"
                    + " playedP=\(dg(timing.playedMoveProb, 3))"
                    + " pW=\(dg(timing.valueProbWin, 2)) pD=\(dg(timing.valueProbDraw, 2)) pL=\(dg(timing.valueProbLoss, 2)) vAbs=\(dg(timing.valueAbsMean, 3))"
                    + " pLogitMean=\(dg(timing.policyLogitMean, 4)) vLogitMean=\(dg(timing.valueLogitMean, 4))"
                    + String(format: " gNorm=%.3f lr=%.3g ms=%.1f", timing.gradGlobalNorm, liveLR, timing.totalMs)
                    + " buf=\(buffer.count) plies=\(feedTally.positions) games=\(feedTally.games)\(feedTally.countsSuffix) epoch=\(epochsCompleted)"
                    + String(format: " mom=%.4f", liveMomentum)
                    + (cycleValues.learningRate != nil ? " lrCyc" + LRMomentumCycleLogFormat.envelopeBounds(cycleValues) : "")
                    + " trainerStep=\(observedSteps)"
                emit(line)
                // Live layer health at the same cadence: BN state + ReZero α,
                // read on the trainer's queue between steps.
                for healthLine in await LayerHealthLog.liveLines(trainer: trainer) {
                    emit(healthLine)
                }
                // Same cadence as the log line, so results.json and the log
                // describe the same ticks.
                recorder?.appendStats(CliTrainingRecorder.StatsLine(
                    elapsedSec: CFAbsoluteTimeGetCurrent() - runStart,
                    steps: step,
                    positionsFed: feedTally.positions,
                    bufferCount: buffer.count,
                    bufferCapacity: p.replayBufferCapacity,
                    policyLoss: Double(timing.policyLoss),
                    valueLoss: Double(timing.valueLoss),
                    policyEntropy: timing.hasDiagnostics ? Double(timing.policyEntropy) : nil,
                    policyIllegalMassPenalty: Double(timing.illegalMassPenalty),
                    gradGlobalNorm: Double(timing.gradGlobalNorm),
                    playedMoveProb: timing.hasDiagnostics ? Double(timing.playedMoveProb) : nil,
                    valueMean: timing.hasDiagnostics ? Double(timing.valueMean) : nil,
                    valueAbsMean: timing.hasDiagnostics ? Double(timing.valueAbsMean) : nil,
                    valueProbWin: timing.hasDiagnostics ? Double(timing.valueProbWin) : nil,
                    valueProbDraw: timing.hasDiagnostics ? Double(timing.valueProbDraw) : nil,
                    valueProbLoss: timing.hasDiagnostics ? Double(timing.valueProbLoss) : nil,
                    // Diagnostic-step fields (these, and the entropy, played-move,
                    // value and W/D/L fields above): nil (not measured) on a step that
                    // skipped the diagnostic reductions.
                    policyLogitMean: timing.hasDiagnostics ? Double(timing.policyLogitMean) : nil,
                    valueLogitMean: timing.hasDiagnostics ? Double(timing.valueLogitMean) : nil,
                    batchSize: batchSize,
                    // `learning_rate` is the STATIC configured base (matching
                    // the self-play path); `lr_effective_base` / momentum /
                    // cycle fields come from the cycle evaluated at this step.
                    // `liveLR` (warmup- and sqrt-batch-scaled) stays in the
                    // [REPLAY] log line, where `lr=` is the honest label for it.
                    trainerHyperparameters: hp,
                    cycleValues: cycleValues,
                    buildNumber: BuildInfo.buildNumber,
                    trainerID: config.runModelID,
                    // Corpus replay feeds every ply it reads, so produced == fed.
                    positionsProduced: feedTally.positions,
                    gamesPlayed: nil,
                    pliesCapDropped: nil,
                    maxPliesPerGame: nil,
                    // A real knob here: `perStepFeed = batchSize / target`.
                    replayRatioTarget: p.replayRatioTarget,
                    lineageTotals: lineageTracker.totals(
                        trainerCompletedSteps: observedSteps,
                        segmentGames: feedTally.games - reconstructionFed.games)
                ))
            }
            // Periodic autosave (overwrites the rolling output file). A disk-full
            // save throws here, halting the run (propagates out of runReplay) so it
            // can resume cleanly from the last checkpoint after space is freed.
            if step % autosaveEvery == 0 {
                let rp = resumePoint()
                try await saveTrainerModel(step: step, reason: "autosave",
                    nextGameIndex: rp.nextGame, shard: rp.shard,
                    epoch: rp.epoch, populatedPlies: buffer.count,
                    corpusID: resumeCorpusID, corpusPath: resumeCorpusPath,
                    segmentGames: feedTally.games - reconstructionFed.games,
                    segmentPositions: feedTally.positions - reconstructionFed.positions,
                    feedAheadPositions: feedAheadPositions(atStep: step))
            }
        }

        // A requested capture that the run ended before reaching (an abort,
        // the corpus or epoch budget, the trainer stopping) is reported; it is
        // not an error, since ending early is often deliberate.
        if let capture = config.gpuCapture, !gpuCaptureStarted, gpuCaptureFailure == nil {
            let msg = "[REPLAY] WARNING: GPU trace capture step \(capture.step) was never reached — the run ended "
                + "at step \(step); no trace was written to \(capture.outputURL.path)"
            FileHandle.standardError.write(Data((msg + "\n").utf8))
            SessionLogger.shared.log(msg)
        }

        // Final save on any clean exit path — step/epoch limit, corpus
        // exhaustion, Ctrl-C abort, or a GPU capture that could not start
        // (the trainer is untouched by that failure). A thrown error skips
        // this (it propagates out of runReplay before we get here): the
        // network state after a hard failure isn't worth persisting over the
        // last good autosave.
        let finalResume = resumePoint()
        let finalReason = gpuCaptureFailure != nil ? "capture-failed" : (aborted ? "abort" : "final")
        try await saveTrainerModel(step: step, reason: finalReason,
            nextGameIndex: finalResume.nextGame, shard: finalResume.shard,
            epoch: finalResume.epoch, populatedPlies: buffer.count,
            corpusID: resumeCorpusID, corpusPath: resumeCorpusPath,
            segmentGames: feedTally.games - reconstructionFed.games,
            segmentPositions: feedTally.positions - reconstructionFed.positions,
            feedAheadPositions: feedAheadPositions(atStep: step))
        // The rolling file is the end state; a step-enumerated copy is not
        // required for it.
        try rollingSaveFailures.requireLastSaveSucceeded(step: step, reason: finalReason)
        // The run asked for a capture it did not get: fail it, after the save.
        // No results.json — a failed run does not claim a clean record.
        if let gpuCaptureFailure {
            throw gpuCaptureFailure
        }

        // `results.json` last, after the final model save — a run that dies
        // saving weights should not also claim a clean results record.
        if let recorder, let output = config.output {
            // Ctrl-C maps to `manualStop`; every clean exit here (step limit,
            // epoch limit, corpus exhaustion) reports `stepLimitReached` — the
            // enum has no case distinguishing the latter two, and inventing one
            // would change the results.json schema for the self-play path too.
            recorder.setTerminationReason(aborted ? .manualStop : .stepLimitReached)
            let counts = recorder.countsSnapshot()
            // Logged, not thrown — matching the self-play path. The trainer
            // model is already safely on disk by this point, so a failed
            // results write (bad --output path, full volume) must not turn a
            // completed multi-hour run into a nonzero exit.
            do {
                let written = try recorder.write(
                    to: output,
                    totalTrainingSeconds: CFAbsoluteTimeGetCurrent() - runStart
                )
                emit("[REPLAY] wrote results: \(written.path) (stats=\(counts.stats))")
            } catch {
                emit("[REPLAY] results write FAILED for \(output.url.path): \(error.localizedDescription)")
            }
        }

        return Result(steps: step, positionsFed: feedTally.positions, gamesFed: feedTally.games,
                      gamesRejected: feedTally.rejected, gamesSkipped: feedTally.skipped,
                      epochs: epochsCompleted)
    }

    // MARK: - async→sync bridge (mirrors SweepCLI.syncWait)

    private final class SyncBoxRef<T>: @unchecked Sendable {
        var success: T?
        var failure: Error?
    }

    private static func syncWait<T>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = SyncBoxRef<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.success = try await work() }
            catch { box.failure = error }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure { throw error }
        guard let success = box.success else {
            preconditionFailure("CorpusReplayRunner.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}
