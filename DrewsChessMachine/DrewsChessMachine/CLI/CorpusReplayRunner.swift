import Foundation
import Darwin
import os

/// Cross-thread one-shot "please stop" flag for the replay loop. The SIGINT
/// `DispatchSource` handler (running on a global queue) flips it; the training
/// loop (running on the detached replay task) reads it once per step. An
/// `OSAllocatedUnfairLock` — the project standard — guards the single Bool;
/// this is not on any hot path (one read per GPU step), so the lock cost is
/// irrelevant.
private final class ReplayAbortFlag: @unchecked Sendable {
    private let state = OSAllocatedUnfairLock(initialState: false)
    func request() { state.withLock { $0 = true } }
    var isRequested: Bool { state.withLock { $0 } }
}

/// Plain, `Sendable` snapshot of the training parameters an offline replay run
/// needs. Captured on the main actor (from `TrainingParameters.shared`) before
/// the run starts, then passed into the off-actor GPU work — so the detached
/// replay task never touches the `@MainActor` singleton (which would deadlock
/// against the `syncWait` semaphore held on the main thread).
struct ReplayParams: Sendable {
    var learningRate: Double
    var entropyBonus: Double
    var drawPenalty: Double
    var weightDecay: Double
    var gradClipMaxNorm: Double
    var policyLossWeight: Double
    var valueLossWeight: Double
    var illegalMassWeight: Double
    var policyLabelSmoothingEpsilon: Double
    var valueLabelSmoothingEpsilon: Double
    var momentumCoeff: Double
    var signedAdvantageComplementCE: Bool
    var sqrtBatchScalingLR: Bool
    var lrWarmupSteps: Int
    var trainingBatchSize: Int
    var replayBufferCapacity: Int
    var replayRatioTarget: Double
    var replayBufferMinPositionsBeforeTraining: Int
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
    /// Exact resume (Phase 2): reconstruct the replay buffer to its contents at
    /// the `--start-model`'s saved `next_game_index` (feed the preceding
    /// capacity-worth of games so the ring self-trims to the exact last-C plies),
    /// continue from there, and use a minimal momentum-refill warm-up instead of
    /// the cold-start one. Requires a `--start-model` carrying `replay_*`
    /// metadata for this same corpus; mutually exclusive with `startShard` /
    /// `startGameIndex`.
    var resumeExact: Bool = false
    /// Explicit destination for the rolling trainer-model file. When nil the
    /// runner derives a path next to `--start-model` (or inside the first
    /// corpus directory). The same file is overwritten by the periodic
    /// autosave and by the final save on exit/abort.
    var outModelPath: String?
    /// When true, every trainer-model save also writes a step-enumerated copy
    /// (`<stem>-replay-step<N>.safetensors`) next to the rolling `-replay-latest`
    /// file, so no checkpoint is ever lost to the overwrite. The rolling latest
    /// is still written (warm-start / probe / trackers key off it); the enumerated
    /// files accumulate — mind disk. Off by default. (`--enumerate-checkpoints`.)
    var enumerateCheckpoints: Bool = false
    /// Freshly-minted `ModelID` for this run's saved model, minted on the main
    /// actor in the pre-flight handler (the `ModelIDMinter` is main-actor
    /// isolated and the replay loop runs off-actor, so it can't mint there).
    var runModelID: String
}

enum CorpusReplayError: LocalizedError {
    case noGames
    case startModelTooSmall(have: Int, need: Int)
    case diskFullDuringSave(step: Int, what: String)
    var errorDescription: String? {
        switch self {
        case .noGames:
            return "No sealed shards found in the provided corpus path(s)"
        case let .startModelTooSmall(have, need):
            return "--start-model has \(have) weight tensors but the network needs at least \(need) (trainables + BN running stats)"
        case let .diskFullDuringSave(step, what):
            return "Disk full while writing \(what) at step \(step) — training halted so it can resume from the last checkpoint once space is freed"
        }
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
        var gamesFed: Int
        var epochs: Int
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
    /// the Cocoa `.fileWriteOutOfSpace` that `Data.write(to:options:)` throws, and
    /// a raw POSIX ENOSPC surfaced directly or as an underlying error. Pure and
    /// total, so the save-failure policy is unit-testable without a real full disk.
    static func isOutOfSpace(_ error: Error) -> Bool {
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
    /// Any OTHER write failure (e.g. a read-only `--out-model` volume) stays
    /// non-fatal: it is logged as a WARNING and the caller continues, preserving
    /// the original contract that a convenience autosave which simply cannot write
    /// to its derived path must not tear down an otherwise-healthy run.
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
        } catch {
            FileHandle.standardError.write(Data("replay: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[REPLAY] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(33)
        }
        let summary = "[REPLAY] done: steps=\(result.steps) positionsFed=\(result.positionsFed) gamesFed=\(result.gamesFed) epochs=\(result.epochs)"
        emit(summary)
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    // MARK: - The run

    private static func runReplay(config: CorpusReplayConfig, params p: ReplayParams, abort: ReplayAbortFlag) async throws -> Result {
        // --start-model: load a saved model and continue training from it. The
        // file embeds its own architecture, which then drives both the trainer
        // and the feeder net — a start model of a different shape than the
        // current default preset trains correctly. nil → fresh random net at
        // the current preset.
        let arch: NetworkArchitecture
        let startModelFile: ModelCheckpointFile?
        let parentModelID: String
        if let sm = config.startModelPath {
            let url = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
            let file = try CheckpointManager.loadModelFile(at: url)
            startModelFile = file
            parentModelID = file.modelID
            arch = file.architecture
            emit("[REPLAY] start-model: \(url.lastPathComponent) modelID=\(file.modelID) encoding=\(arch.inputEncoding.rawValue)")
        } else {
            startModelFile = nil
            parentModelID = ""
            if let pn = config.presetName {
                guard let preset = NetworkArchitecture.Preset(rawValue: pn) else {
                    let names = NetworkArchitecture.Preset.allCases.map(\.rawValue).joined(separator: ", ")
                    FileHandle.standardError.write(Data("error: unknown --preset '\(pn)'. Available: \(names)\n".utf8))
                    Darwin.exit(2)
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
        let hparamsLine = String(
            format: "[REPLAY-HPARAMS] lr=%.6g batch=%ld wd=%.4g momentum=%.3g gradClip=%.3g entropyBonus=%.4g drawPenalty=%.4g policyW=%.3g valueW=%.3g illegalW=%.4g pLabelSmooth=%.4g vLabelSmooth=%.4g lrWarmup=%ld bufCap=%ld replayRatio=%.3g minPrefill=%ld",
            p.learningRate, p.trainingBatchSize, p.weightDecay, p.momentumCoeff, p.gradClipMaxNorm,
            p.entropyBonus, p.drawPenalty, p.policyLossWeight, p.valueLossWeight, p.illegalMassWeight,
            p.policyLabelSmoothingEpsilon, p.valueLabelSmoothingEpsilon,
            p.lrWarmupSteps, p.replayBufferCapacity, p.replayRatioTarget,
            p.replayBufferMinPositionsBeforeTraining
        )
            + " complementCE=\(p.signedAdvantageComplementCE ? "on" : "off")"
            + " sqrtBatchLR=\(p.sqrtBatchScalingLR ? "on" : "off")"
        emit(hparamsLine)
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
        for (di, dir) in config.corpusDirectories.enumerated() {
            let corpus = try GameCorpus.open(directory: dir)
            let urls = try corpus.sealedShardURLs()
            shardURLs.append(contentsOf: urls)
            if di == 0 { resumeCorpusID = corpus.corpusID; resumeCorpusPath = dir.path }
            emit("[REPLAY] corpus \(corpus.corpusID): \(urls.count) sealed shard(s)")
        }
        guard !shardURLs.isEmpty else { throw CorpusReplayError.noGames }

        // Per-shard game counts via cheap trailer reads (no full-shard decode) —
        // for --start-game-index resolution, the resume `nextGame=` logging, and
        // the saved `next_game_index`. cumGames[i] = games in shards 0..<i, so
        // cumGames[i] is the global index of shard i's first game and
        // cumGames.last! the corpus total.
        let shardCounts = try shardURLs.map { try GameCorpusShardIO.readSealedCounts(at: $0) }
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
            FileHandle.standardError.write(Data("error: --start-shard and --start-game-index are mutually exclusive\n".utf8))
            Darwin.exit(2)
        }
        var startShardCursor = 0
        var startWithinShardSkip = 0
        var reconstructUntil: Int? = nil   // resume-exact: refeed up to here, then train
        var startEpoch = 0
        if config.resumeExact {
            // Read the saved resume metadata from the --start-model header (no
            // tensor decode) and validate it's for THIS corpus.
            guard let smPath = config.startModelPath else {
                FileHandle.standardError.write(Data("error: --resume-exact requires --start-model\n".utf8))
                Darwin.exit(2)
            }
            let smURL = URL(fileURLWithPath: (smPath as NSString).expandingTildeInPath)
            guard let rm = SafetensorsModelIO.readResumeMetadata(at: smURL) else {
                FileHandle.standardError.write(Data("error: --resume-exact: \(smURL.lastPathComponent) carries no replay_* resume metadata (not a corpus-replay checkpoint)\n".utf8))
                Darwin.exit(2)
            }
            guard rm.corpusID == resumeCorpusID else {
                FileHandle.standardError.write(Data("error: --resume-exact: checkpoint corpus_id '\(rm.corpusID)' != this corpus '\(resumeCorpusID)'\n".utf8))
                Darwin.exit(2)
            }
            if let g = rm.builtByGit, g != BuildInfo.gitHash {
                SessionLogger.shared.log("[REPLAY] WARNING --resume-exact: checkpoint built by git \(g) but running \(BuildInfo.gitHash) — if the encoder/feeder changed, the reconstructed buffer may differ from the original.")
            }
            // Refeed enough games before `until` to overflow the ring, which then
            // self-trims to the exact last-capacity plies (so the precise refeed
            // start doesn't matter as long as it covers >= capacity FED plies).
            // Walk back by capacity/avgPly games with a 1.5x margin so skips and
            // local short games can't under-fill. avgPly is the corpus's actual
            // mean (trailer plies / games).
            let until = max(0, min(rm.nextGameIndex, totalCorpusGames))
            reconstructUntil = until
            startEpoch = max(0, rm.epoch)
            let cap = max(1, p.replayBufferCapacity)
            let avgPly = totalCorpusGames > 0 ? max(1.0, Double(totalPlies) / Double(totalCorpusGames)) : 66.0
            let gamesBack = Int((1.5 * Double(cap) / avgPly).rounded(.up))
            // Cross-epoch reconstruction isn't supported yet: a checkpoint saved
            // at epoch ≥ 1 with nextGame < the refeed window held PRIOR-epoch
            // tail games we'd have to wrap backward to refeed, which also
            // collides with the epoch-budget stop inside nextGame(). The
            // epoch-completion checkpoint Phase 1 normalizes to (nextGame=0,
            // epoch=N) is exactly this case. Fail loud with an actionable
            // alternative rather than silently reconstruct a short/empty buffer.
            // (epoch 0 is always fine — no prior epoch, so [0, until) IS the
            // exact buffer contents, full or legitimately partial.)
            if startEpoch > 0 && until < gamesBack {
                FileHandle.standardError.write(Data("error: --resume-exact from an early-epoch checkpoint (epoch \(startEpoch), nextGame \(until) < \(gamesBack)-game reconstruction window) needs cross-epoch buffer reconstruction, not yet supported. Resume from a mid-epoch checkpoint, or use --start-game-index \(until) for an approximate cold-refill resume.\n".utf8))
                Darwin.exit(2)
            }
            let reconstructStart = max(0, until - gamesBack)
            let (s, off) = locate(reconstructStart)
            startShardCursor = s
            startWithinShardSkip = off
            SessionLogger.shared.log("[REPLAY] --resume-exact: nextGame=\(until) epoch=\(startEpoch) cap=\(cap) savedPlies=\(rm.populatedPlies) -> refeed games [\(reconstructStart), \(until)) (\(until - reconstructStart) games ≈ \(Int(Double(until - reconstructStart) * avgPly)) plies) from \(shardURLs[s].lastPathComponent) offset \(off)")
        } else if let ss = config.startShard {
            guard ss >= 0 && ss < shardURLs.count else {
                FileHandle.standardError.write(Data("error: --start-shard \(ss) out of range; valid 0…\(shardURLs.count - 1)\n".utf8))
                Darwin.exit(2)
            }
            startShardCursor = ss
            let skipDesc = ss == 0 ? "no shards skipped" : "skipping shards 0…\(ss - 1), \(cumGames[ss]) games"
            emit("[REPLAY] --start-shard \(ss) -> \(shardURLs[ss].lastPathComponent) (\(skipDesc))")
        } else if let gi = config.startGameIndex {
            guard gi >= 0 && gi < totalCorpusGames else {
                FileHandle.standardError.write(Data("error: --start-game-index \(gi) out of range; valid 0…\(totalCorpusGames - 1)\n".utf8))
                Darwin.exit(2)
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

        // Warm-up: on an exact resume the weights AND the (reconstructed) buffer
        // are both warm — only SGD momentum is cold — so use a short
        // momentum-refill ramp (~50 steps at momentum 0.9, ~5 time constants)
        // rather than the cold-start warm-up. Never exceed the configured value.
        let effectiveWarmupSteps = config.resumeExact ? min(50, p.lrWarmupSteps) : p.lrWarmupSteps
        if config.resumeExact {
            SessionLogger.shared.log("[REPLAY] --resume-exact: lrWarmupSteps \(p.lrWarmupSteps) -> \(effectiveWarmupSteps) (momentum-refill)")
        }

        emit("[REPLAY] building network + trainer (encoding=\(arch.inputEncoding.rawValue))")
        let net = try ChessMPSNetwork(.randomWeights, arch: arch)
        let trainer = try ChessTrainer(
            learningRate: Float(p.learningRate),
            entropyRegularizationCoeff: Float(p.entropyBonus),
            drawPenalty: Float(p.drawPenalty),
            weightDecayC: Float(p.weightDecay),
            gradClipMaxNorm: Float(p.gradClipMaxNorm),
            policyLossWeight: Float(p.policyLossWeight),
            valueLossWeight: Float(p.valueLossWeight),
            illegalMassPenaltyWeight: Float(p.illegalMassWeight),
            policyLabelSmoothingEpsilon: Float(p.policyLabelSmoothingEpsilon),
            valueLabelSmoothingEpsilon: Float(p.valueLabelSmoothingEpsilon),
            momentumCoeff: Float(p.momentumCoeff),
            useSignedAdvantageComplementCE: p.signedAdvantageComplementCE,
            sqrtBatchScalingForLR: p.sqrtBatchScalingLR,
            lrWarmupSteps: effectiveWarmupSteps,
            arch: arch
        )
        let buffer = ReplayBuffer(capacity: p.replayBufferCapacity, inputEncoding: net.inputEncoding)
        let feeder = CorpusReplayFeeder(network: net, buffer: buffer)

        // Seed both the trainer (the network that actually learns) and the
        // feeder net (computes the value baseline while feeding) from the start
        // model's base weights — exactly trainables + BN running stats. A
        // trainer source file carries optimizer velocity after that block, so
        // take the leading base prefix (same rule as ProbeModelCLI /
        // UCIModelLoader).
        //
        // The trainer must be seeded via `loadBaseWeightsResetVelocity`, NOT a
        // bare `network.loadWeights`: under the canonical bf16 mixed-precision
        // path the optimizer steps the fp32 *master* weights and re-derives the
        // bf16 working copy from them each step. Writing only the working copy
        // would leave the masters at random init, and the first SGD step would
        // overwrite our loaded weights with that random surface. This call
        // writes the working copy, seeds the fp32 masters from the same values,
        // and zeros optimizer velocity (a fresh fork — momentum re-accumulates).
        // The feeder net is a plain inference network (no masters/velocity), so
        // a direct `loadWeights` is correct there.
        if let file = startModelFile {
            let baseCount = net.network.trainableVariables.count + net.network.bnRunningStatsVariables.count
            guard file.weights.count >= baseCount else {
                throw CorpusReplayError.startModelTooSmall(have: file.weights.count, need: baseCount)
            }
            let base = Array(file.weights.prefix(baseCount))
            try await net.network.loadWeights(base)
            try await trainer.loadBaseWeightsResetVelocity(base)
            emit("[REPLAY] start-model weights loaded into trainer (working+masters, velocity zeroed) + feeder net (base tensors=\(baseCount))")
        }

        // Rolling trainer-model output file. The same file is overwritten by
        // the periodic autosave and by the final save on exit/abort, so it
        // always holds the latest weights. Destination precedence: explicit
        // --out-model; else next to --start-model; else the app's Models
        // directory named after the corpus. Overwrite is deliberate here (the
        // CheckpointManager never-overwrite history rule is for the curated
        // Models/Sessions store) — this is a single "latest" convenience file.
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
        emit("[REPLAY] trainer-model output: \(outModelURL.path)")

        // Export the trainer's current base weights and overwrite the rolling
        // output file. Failure handling splits on cause (see reportSaveFailure):
        // a disk-full (ENOSPC) failure is FATAL — it alarms and throws so the run
        // halts rather than training on into a window where nothing persists; any
        // other failure (e.g. a read-only --out-model volume) stays a non-fatal
        // WARNING so a convenience autosave that simply cannot write to its
        // derived path does not tear down an otherwise-healthy run. Pass
        // --out-model to a writable location if the derived path can't be written.
        // Resume info is passed in (not captured): the corpus index / stream
        // cursor are resolved AFTER this nested func, so the call sites — which
        // run inside the SGD loop where those are in scope — supply them. The
        // `replay_*` keys land in the safetensors `__metadata__` (write-only in
        // Phase 1; the exact-reconstruction resume reads them later). `built_by_*`
        // pins which encoder/feeder build wrote them, so a byte-exact resume can
        // refuse a build whose encoding may differ.
        func saveTrainerModel(step: Int, reason: String,
                              nextGameIndex: Int, shard: Int, epoch: Int, populatedPlies: Int,
                              corpusID: String, corpusPath: String) async throws {
            // Rolling save (overwrites the output file). `encoded` is reused by the
            // enumerated copy below, so it outlives this do/catch. A disk-full
            // failure re-throws (fatal, halts the run); any other failure is a
            // non-fatal WARNING and we skip the enumerated copy (it would fail too).
            let encoded: Data
            do {
                let weights = try await trainer.network.exportWeights()
                let metadata = ModelCheckpointMetadata(
                    creator: "replay",
                    trainingStep: step,
                    parentModelID: parentModelID,
                    notes: "corpus replay \(reason) @ step \(step)"
                )
                let resumeMeta: [String: String] = [
                    "replay_corpus_id": corpusID,
                    "replay_corpus_path": corpusPath,
                    "replay_next_game_index": String(nextGameIndex),
                    "replay_epoch": String(epoch),
                    "replay_populated_plies": String(populatedPlies),
                    "replay_capacity": String(p.replayBufferCapacity),
                    "built_by_build": String(BuildInfo.buildNumber),
                    "built_by_git": BuildInfo.gitHash,
                ]
                encoded = try SafetensorsModelIO.encode(
                    modelID: config.runModelID,
                    createdAtUnix: Int64(Date().timeIntervalSince1970),
                    metadata: metadata,
                    weights: weights,
                    architecture: arch,
                    includesVelocity: false,
                    resumeMetadata: resumeMeta
                )
                try FileManager.default.createDirectory(
                    at: outModelURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try encoded.write(to: outModelURL, options: [.atomic])
                emit("[REPLAY] saved trainer model (\(reason)) step=\(step) nextGame=\(nextGameIndex) shard=\(shard) epoch=\(epoch) -> \(outModelURL.lastPathComponent)")
            } catch {
                // Throws on disk-full (halt); returns on any other failure (non-fatal).
                try Self.reportSaveFailure(error, step: step, what: "trainer-model save (\(reason))")
                return
            }

            // Optional: also drop a step-enumerated copy so no checkpoint is lost
            // to the rolling overwrite. Reuses `encoded` (no re-export). A separate
            // do/catch so a disk-full throw here propagates OUT — it must not be
            // caught by the rolling-save catch above (which would misclassify our
            // own halt error as "some other failure" and swallow it).
            if config.enumerateCheckpoints {
                let stem = outModelURL.deletingPathExtension().lastPathComponent
                let enumName = stem.contains("-replay-latest")
                    ? stem.replacingOccurrences(of: "-replay-latest", with: "-replay-step\(step)")
                    : "\(stem)-step\(step)"
                let enumURL = outModelURL.deletingLastPathComponent()
                    .appendingPathComponent(enumName)
                    .appendingPathExtension("safetensors")
                do {
                    try encoded.write(to: enumURL, options: [.atomic])
                    emit("[REPLAY] enumerated checkpoint -> \(enumURL.lastPathComponent)")
                } catch {
                    try Self.reportSaveFailure(error, step: step, what: "enumerated checkpoint")
                }
            }
        }

        let batchSize = max(1, p.trainingBatchSize)
        let reuse = max(0.01, p.replayRatioTarget)
        // Positions to feed per step so each is sampled ~`reuse` times before
        // eviction: K = batchSize / R.
        let perStepFeed = max(1, Int((Double(batchSize) / reuse).rounded()))
        let minPrefill = max(batchSize, p.replayBufferMinPositionsBeforeTraining)

        // Budget: explicit step limit wins; otherwise bound by epochs (default
        // a single pass when neither is given).
        let stepLimit = config.stepLimit
        let epochLimit: Int? = config.epochs ?? (stepLimit == nil ? 1 : nil)

        emit("[REPLAY] batchSize=\(batchSize) reuse=\(String(format: "%.2f", reuse)) K=\(perStepFeed) minPrefill=\(minPrefill) stepLimit=\(stepLimit.map(String.init) ?? "none") epochLimit=\(epochLimit.map(String.init) ?? "none")")

        // Streaming game source, cycling the shard list for epochs.
        var shardCursor = startShardCursor
        var currentGames: [GameRecord] = []
        var gameCursor = 0
        var epochsCompleted = startEpoch   // resume-exact carries the saved epoch; else 0
        // One-shot within-shard skip applied to the FIRST loaded shard
        // (--start-game-index); cleared after that shard and on any epoch wrap.
        var firstShardSkip = startWithinShardSkip
        // Global within-epoch index of the NEXT game nextGame() will return:
        // starts at the resume point, +1 per returned game, resets to 0 on the
        // epoch wrap. Logged as `nextGame=` and saved as `next_game_index`.
        var nextGameWithinEpoch = startGlobalIndex

        func nextGame() -> GameRecord? {
            while gameCursor >= currentGames.count {
                if shardCursor >= shardURLs.count {
                    // Wrap to a fresh epoch. Reset the cursors BEFORE the
                    // epoch-limit return, so a run that completes its budget
                    // leaves a consistent (nextGame=0, shard=0, epoch incremented)
                    // resume point rather than (nextGame=totalGames, shard=count)
                    // — the latter is one past the end and outside the resume
                    // bounds this same file enforces on read.
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
        // locate(...).shard == shardURLs.count, both outside the resume bounds
        // this same file enforces on read. Fold the boundary state forward to the
        // start of the next epoch — (game 0, shard 0, epoch + 1) — exactly as
        // nextGame()'s wrap does, so the saved resume point is always consistent.
        func resumePoint() -> (nextGame: Int, shard: Int, epoch: Int) {
            if nextGameWithinEpoch >= totalCorpusGames {
                return (0, 0, epochsCompleted + 1)
            }
            return (nextGameWithinEpoch, locate(nextGameWithinEpoch).shard, epochsCompleted)
        }

        var positionsFed = 0
        var gamesFed = 0
        var corpusExhausted = false

        // Pre-fill — or, for --resume-exact, RECONSTRUCT: refeed games up to the
        // saved next_game_index so the fixed-capacity ring ends holding exactly
        // the last-capacity plies the original run had there (the surplus is
        // overwritten). Training then continues from next_game_index. The refeed
        // stays within one epoch (reconstructStart..until are both in [0,
        // totalCorpusGames)), so no wrap fires mid-reconstruction.
        if let until = reconstructUntil {
            while nextGameWithinEpoch < until {
                guard let g = nextGame() else { corpusExhausted = true; break }
                positionsFed += feeder.feed(g)
                gamesFed += 1
            }
            SessionLogger.shared.log("[REPLAY] --resume-exact: buffer reconstructed bufCount=\(buffer.count)/\(p.replayBufferCapacity) (refed \(gamesFed) games / \(positionsFed) plies); resuming at game \(nextGameWithinEpoch) epoch \(epochsCompleted)")
        } else {
            while buffer.count < minPrefill {
                guard let g = nextGame() else { corpusExhausted = true; break }
                positionsFed += feeder.feed(g)
                gamesFed += 1
            }
        }
        let prefillPositions = positionsFed
        emit("[REPLAY] pre-filled: bufCount=\(buffer.count) positionsFed=\(positionsFed) gamesFed=\(gamesFed)")

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
        let autosaveEvery = 1000
        var aborted = false
        while true {
            // Ctrl-C: stop cleanly before starting another step so the
            // post-loop save captures a complete, non-mid-step state.
            if abort.isRequested {
                aborted = true
                emit("[REPLAY] abort requested — stopping at step \(step)")
                break
            }
            if let sl = stepLimit, step >= sl { break }
            let targetFed = prefillPositions + step * perStepFeed
            while positionsFed < targetFed && !corpusExhausted {
                guard let g = nextGame() else { corpusExhausted = true; break }
                positionsFed += feeder.feed(g)
                gamesFed += 1
            }
            if corpusExhausted && epochLimit != nil { break }

            guard let timing = try await trainer.trainStep(replayBuffer: buffer, batchSize: batchSize) else {
                emit("[REPLAY] trainStep returned nil (bufCount=\(buffer.count)); stopping")
                break
            }
            step += 1
            if step == 1 || step % logEvery == 0 {
                // Live, warmup-adjusted LR read from the trainer (single source
                // of truth — don't re-derive the warmup formula here).
                let liveLR = trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: nil)
                let line = "[REPLAY] step=\(step)"
                    + String(format: " loss=%.4f pLoss=%.4f vLoss=%.4f", timing.loss, timing.policyLoss, timing.valueLoss)
                    + " pEnt=\(dg(timing.policyEntropy, 3)) pIllM=\(dg(timing.illegalMassPenalty, 4))"
                    + " playedP=\(dg(timing.playedMoveProb, 3))"
                    + " pW=\(dg(timing.valueProbWin, 2)) pD=\(dg(timing.valueProbDraw, 2)) pL=\(dg(timing.valueProbLoss, 2)) vAbs=\(dg(timing.valueAbsMean, 3))"
                    + String(format: " gNorm=%.3f lr=%.3g ms=%.1f", timing.gradGlobalNorm, liveLR, timing.totalMs)
                    + " buf=\(buffer.count) plies=\(positionsFed) games=\(gamesFed) epoch=\(epochsCompleted)"
                emit(line)
            }
            // Periodic autosave (overwrites the rolling output file). A disk-full
            // save throws here, halting the run (propagates out of runReplay) so it
            // can resume cleanly from the last checkpoint after space is freed.
            if step % autosaveEvery == 0 {
                let rp = resumePoint()
                try await saveTrainerModel(step: step, reason: "autosave",
                    nextGameIndex: rp.nextGame, shard: rp.shard,
                    epoch: rp.epoch, populatedPlies: buffer.count,
                    corpusID: resumeCorpusID, corpusPath: resumeCorpusPath)
            }
        }

        // Final save on any clean exit path — step/epoch limit, corpus
        // exhaustion, or Ctrl-C abort. A thrown error skips this (it propagates
        // out of runReplay before we get here): the network state after a hard
        // failure isn't worth persisting over the last good autosave.
        let finalResume = resumePoint()
        try await saveTrainerModel(step: step, reason: aborted ? "abort" : "final",
            nextGameIndex: finalResume.nextGame, shard: finalResume.shard,
            epoch: finalResume.epoch, populatedPlies: buffer.count,
            corpusID: resumeCorpusID, corpusPath: resumeCorpusPath)

        return Result(steps: step, positionsFed: positionsFed, gamesFed: gamesFed, epochs: epochsCompleted)
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
