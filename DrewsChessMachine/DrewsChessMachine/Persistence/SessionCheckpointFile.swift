import Foundation

// MARK: - Errors

enum SessionCheckpointError: LocalizedError {
    case missingChampionFile
    case missingTrainerFile
    case missingSessionJSON
    case invalidJSON(Error, detail: String = "")
    case unsupportedVersion(Int)
    case targetDirectoryExists(URL)

    var errorDescription: String? {
        switch self {
        case .missingChampionFile:
            return "Session directory is missing champion.dcmmodel"
        case .missingTrainerFile:
            return "Session directory is missing trainer.dcmmodel"
        case .missingSessionJSON:
            return "Session directory is missing session.json"
        case .invalidJSON(let err, let detail):
            let base = "session.json could not be decoded: \(err)"
            return detail.isEmpty ? base : "\(base)\nFirst 2000 bytes of file:\n\(detail)"
        case .unsupportedVersion(let v):
            return "Unsupported session.json format version \(v)"
        case .targetDirectoryExists(let url):
            return "Refusing to overwrite existing session at \(url.lastPathComponent)"
        }
    }
}

// MARK: - Serializable shapes for non-Codable project types

/// Codable mirror of `SamplingSchedule`. The original struct is
/// project-owned but not Codable (and making it Codable would
/// drag an import into MPSChessPlayer.swift); the checkpoint path
/// builds this wrapper on save and turns it back into the live
/// struct on load.
struct TauConfigCodable: Codable, Equatable {
    let startTau: Float
    let decayPerPly: Float
    let floorTau: Float

    init(_ schedule: SamplingSchedule) {
        self.startTau = schedule.startTau
        self.decayPerPly = schedule.decayPerPly
        self.floorTau = schedule.floorTau
    }

    var asSamplingSchedule: SamplingSchedule {
        SamplingSchedule(
            startTau: startTau,
            decayPerPly: decayPerPly,
            floorTau: floorTau
        )
    }
}

/// Codable mirror of `TournamentRecord` — saved in the session's
/// arena history for context on resume. Only the audit fields
/// (counts, score, promoted flag, duration) are persisted; live UI
/// state like `isCurrent` is not.
///
/// `gamesPlayed` and `promotionKind` are Optional for backward
/// compatibility with session files written before those fields
/// existed. A missing `gamesPlayed` is reconstructed at load time
/// as `candidateWins + championWins + draws` (same identity the
/// tournament driver uses). A missing `promotionKind` is treated
/// as `.automatic` on load when `promoted == true`, which matches
/// the only way promotions could happen before the manual Promote
/// button existed.
struct ArenaHistoryEntryCodable: Codable, Equatable {
    let finishedAtStep: Int
    let candidateWins: Int
    let championWins: Int
    let draws: Int
    let score: Double
    let promoted: Bool
    let promotedID: String?
    let durationSec: Double
    var gamesPlayed: Int?
    var promotionKind: String?
    // Per-side candidate W/L/D — optional for back-compat with
    // session files written before side tracking existed. Missing
    // values decode as `nil` and load-path code substitutes 0 so
    // the display shows "—" for the side breakdown on legacy data.
    var candidateWinsAsWhite: Int?
    var candidateWinsAsBlack: Int?
    var candidateLossesAsWhite: Int?
    var candidateLossesAsBlack: Int?
    var candidateDrawsAsWhite: Int?
    var candidateDrawsAsBlack: Int?
    /// Wall-clock time (seconds since 1970) the tournament
    /// finished. Optional for back-compat with session files
    /// written before the field existed; the Arena History UI
    /// renders "—" when nil.
    var finishedAtUnix: Int64?
    /// Candidate `ModelID` description (e.g. `20260505-3-A1B2`).
    /// Optional for back-compat. Surfaces alongside the verdict
    /// in the Arena History UI.
    var candidateID: String?
    /// Champion `ModelID` description as of arena start, before
    /// any promotion copy. Optional for back-compat.
    var championID: String?
}

// MARK: - Session State

/// Serialized form of a paused training session. Stored at
/// `session.json` inside a `.dcmsession` directory next to the
/// champion and trainer `.dcmmodel` files. Loaded at resume time
/// to re-seed counters, hyperparameter display, and network
/// identity.
struct SessionCheckpointState: Codable, Equatable {
    static let currentFormatVersion: Int = 1

    let formatVersion: Int
    let sessionID: String
    let savedAtUnix: Int64
    let sessionStartUnix: Int64
    /// Accumulated elapsed seconds at save time, measured against
    /// the session's `sessionStart` anchor. On resume, the new
    /// session's anchor is placed at `Date() - elapsedTrainingSec`
    /// so "Total session time" picks up where it left off.
    let elapsedTrainingSec: Double

    // Counters
    let trainingSteps: Int
    let selfPlayGames: Int
    let selfPlayMoves: Int
    let trainingPositionsSeen: Int

    // Hyperparameters (as they were in effect at save time)
    let batchSize: Int
    let learningRate: Float
    var entropyRegularizationCoeff: Float?
    /// Bootstrap-phase draw penalty (0 = disabled). See
    /// `ChessTrainer.drawPenalty`. Optional for back-compat with
    /// session files written before the field was added.
    var drawPenalty: Float?
    let promoteThreshold: Double
    let arenaGames: Int
    /// Number of arena games run concurrently per tournament.
    /// Optional for back-compat with session.json files written
    /// before parallel arena existed; absent → load-side hydrates
    /// to the user's current `effectiveArenaConcurrency`.
    var arenaConcurrency: Int?
    let selfPlayTau: TauConfigCodable
    let arenaTau: TauConfigCodable
    let selfPlayWorkerCount: Int
    var gradClipMaxNorm: Float?
    var weightDecayCoeff: Float?
    /// Policy-loss coefficient applied to the policy term in
    /// `total_loss = valueLossWeight·valueLoss +
    /// policyLossWeight·policyLoss − …`. Optional for back-compat
    /// with session files written before the field became editable.
    /// Renamed from `policyScaleK` (the old K knob); old session
    /// files won't carry this key and load with the user's current
    /// `TrainingParameters.shared.policyLossWeight` instead.
    var policyLossWeight: Float?
    /// Value-loss coefficient applied to the value term in
    /// `total_loss`. Mirrors `policyLossWeight`; optional for
    /// back-compat with session files written before the value
    /// weight existed.
    var valueLossWeight: Float?
    /// Polyak momentum coefficient μ in effect at save time. Optional
    /// for back-compat with session files written before momentum
    /// landed in the schema; absent → loader falls through to the
    /// user's current `TrainingParameters.shared.momentumCoeff`.
    /// The optimizer's velocity buffers themselves are persisted
    /// separately in `trainer.dcmmodel` (v2 layout); this scalar
    /// controls how aggressively the saved velocity is mixed in
    /// going forward.
    var momentumCoeff: Float?
    /// Illegal-mass penalty weight in effect at save time. Multiplied
    /// into the unmasked-softmax illegal-mass term in `total_loss`,
    /// where positive values pull probability mass off illegal cells.
    /// Optional for back-compat with session files written before
    /// the term existed; absent → loader falls through to the user's
    /// current `TrainingParameters.shared.illegalMassWeight`.
    var illegalMassPenaltyWeight: Float?
    /// Policy-CE label-smoothing coefficient ε in effect at save time.
    /// ε=0 → one-hot played-move target; ε>0 → `(1−ε)·oneHot + ε·uniform(legal)`.
    /// Optional for back-compat with session files written before this
    /// term existed; absent → loader falls through to the user's
    /// current `TrainingParameters.shared.policyLabelSmoothingEpsilon`.
    var policyLabelSmoothingEpsilon: Float?
    /// Value-head W/D/L cross-entropy label-smoothing coefficient ε in
    /// effect at save time. ε=0 → hard one-hot on the game result;
    /// ε>0 → `(1−ε)·oneHot(slot) + ε·(⅓,⅓,⅓)`. Optional for back-compat
    /// with session files written before the WDL value head landed;
    /// absent → loader falls through to the user's current
    /// `TrainingParameters.shared.valueLabelSmoothingEpsilon`.
    var valueLabelSmoothingEpsilon: Float?

    // Replay-ratio controller settings. All Optional so older
    // session.json files that lack these keys still decode.
    var replayRatioTarget: Double?
    var replayRatioAutoAdjust: Bool?
    var stepDelayMs: Int?
    /// Self-play-side per-game-per-worker delay (ms) in effect at save
    /// time. Distinct from `stepDelayMs`, which is the training-side
    /// inter-batch delay. Optional for back-compat; absent → loader
    /// falls through to `TrainingParameters.shared.selfPlayDelayMs`.
    var selfPlayDelayMs: Int?
    var lastAutoComputedDelayMs: Int?

    // Training-loop parameters that previously lived only in
    // @AppStorage and so silently drifted between session-save
    // time and session-resume time. All Optional for back-compat
    // with older session.json files that pre-date the schema
    // expansion; on resume, an absent field falls through to the
    // user's current @AppStorage value, while a present field
    // overrides @AppStorage so reload is fully reproducible.
    var lrWarmupSteps: Int?
    var sqrtBatchScalingForLR: Bool?
    var replayBufferMinPositionsBeforeTraining: Int?
    var arenaAutoIntervalSec: Double?
    var candidateProbeIntervalSec: Double?
    var legalMassCollapseThreshold: Double?
    var legalMassCollapseGraceSeconds: Double?
    var legalMassCollapseNoImprovementProbes: Int?
    /// Interval (in training steps) between `[BATCH-STATS]` emissions.
    /// Optional for back-compat; absent → loader falls through to
    /// `TrainingParameters.shared.batchStatsInterval`.
    var batchStatsInterval: Int?
    /// Composition-aware replay-buffer sampler constraints in effect at
    /// save time. All Optional for back-compat with session files written
    /// before these knobs existed; absent → loader falls through to the
    /// user's current `TrainingParameters.shared` value. The sampler reads
    /// these directly off `TrainingParameters.shared` per `sample(count:)`
    /// call (see `ReplayBuffer.swift`), so resume just needs to write the
    /// saved value back onto the singleton.
    var maxPliesFromAnyOneGame: Int?
    var targetSampledGameLengthPlies: Int?
    var maxDrawPercentPerBatch: Int?
    /// Fraction of drawn self-play games kept in the replay buffer
    /// at game end (the rest are dropped). 1.0 = legacy behaviour
    /// (keep everything). Optional for back-compat with sessions
    /// saved before the draw-keep filter existed; absent → loader
    /// falls through to `TrainingParameters.shared.selfPlayDrawKeepFraction`
    /// (which is 1.0 by default).
    var selfPlayDrawKeepFraction: Double?
    /// Lifetime self-play games that were emitted into the replay
    /// buffer (i.e. survived the draw-keep filter). `<= selfPlayGames`;
    /// equal at default keep-fraction. Optional for back-compat.
    var emittedGames: Int?
    /// Lifetime plies emitted into the replay buffer across the
    /// session. `<= selfPlayMoves`; equal at default keep-fraction.
    /// Optional for back-compat.
    var emittedPositions: Int?

    // Game-result breakdown (added v1.1 — Optional for compat)
    var whiteCheckmates: Int?
    var blackCheckmates: Int?
    var stalemates: Int?
    var fiftyMoveDraws: Int?
    var threefoldRepetitionDraws: Int?
    var insufficientMaterialDraws: Int?
    var totalGameWallMs: Double?

    // Per-outcome emitted-game breakdown (added when the Results card
    // gained an Overall vs Kept layout). All Optional for back-compat
    // with sessions saved before these counters existed; loader falls
    // back to the played-side counterparts (the draw-keep filter was
    // either disabled or absent in those sessions, so emitted == played
    // at every outcome category).
    var emittedWhiteCheckmates: Int?
    var emittedBlackCheckmates: Int?
    var emittedStalemates: Int?
    var emittedFiftyMoveDraws: Int?
    var emittedThreefoldRepetitionDraws: Int?
    var emittedInsufficientMaterialDraws: Int?

    // Build metadata captured at save time. Optional for back-compat
    // with older session.json files that lack these fields.
    var buildNumber: Int?
    var buildGitHash: String?
    var buildGitBranch: String?
    var buildDate: String?
    var buildTimestamp: String?
    var buildGitDirty: Bool?

    // Replay-buffer presence (added alongside `replay_buffer.bin`).
    // `true` if the session directory contains a matching
    // replay-buffer file; nil/false for older sessions without it.
    var hasReplayBuffer: Bool?
    var replayBufferStoredCount: Int?
    var replayBufferCapacity: Int?
    var replayBufferTotalPositionsAdded: Int?

    // Chart-data presence (added alongside the optional
    // `training_chart.json` and `progress_rate_chart.json`
    // companion files). `true` iff both files exist in the session
    // directory and the per-ring sample counts agree with the
    // values in those files. Older sessions and sessions saved
    // with chart collection disabled are nil/false here. Loaded
    // by `seedFromRestoredSession` to populate the chart rings on
    // resume so the chart trajectory survives save/resume cycles.
    var hasChartData: Bool?
    var trainingChartSampleCount: Int?
    var progressRateSampleCount: Int?

    // Inline auxiliary chart state, small enough to live next to
    // the rest of session.json instead of in a side file. Both
    // Optional for back-compat; missing/nil decodes the same way
    // older sessions did (no arena bands restored,
    // `legalMassMaxAllTime` resets to 0 on session start).
    var arenaChartEvents: [ArenaChartEvent]?
    var legalMassMaxAllTime: Double?

    /// Per-run training history. Each `TrainingSegment` represents one
    /// continuous Play-and-Train period (start → stop, save, or
    /// session-quit). Cumulative status-bar metrics sum across this
    /// array plus the in-memory current run. Optional for back-compat
    /// with older session files written before segments existed; the
    /// loader treats nil/missing as "no historical segments." Each
    /// save closes the current segment with `endUnix = saveTime` and
    /// appends it; on resume, a new segment begins on the next
    /// Play-and-Train start.
    var trainingSegments: [TrainingSegment]?

    // Network identity — duplicated from the `.dcmmodel` headers so
    // a future "browse saved sessions" UI can read just
    // `session.json` and still show model IDs.
    let championID: String
    let trainerID: String

    // Arena history (audit log — displayed in the UI on resume)
    let arenaHistory: [ArenaHistoryEntryCodable]

    // MARK: - Training Segments

    /// One Play-and-Train run, bounded by start and end wall-clock
    /// times. Status-bar wall-time totals sum `durationSec` across the
    /// session's full segment array — that's "active training time"
    /// and excludes idle gaps when training was stopped. The segment
    /// also captures starting/ending counter snapshots so per-run
    /// progress can be reconstructed (e.g., "this run added 12K
    /// training steps and 3.5M positions to the buffer").
    ///
    /// The build/git fields are captured on segment-start so each
    /// segment is attributable to a specific code version — invaluable
    /// for "which build produced this entropy curve?" forensics across
    /// architecture changes.
    struct TrainingSegment: Codable, Equatable {
        let startUnix: Int64
        let endUnix: Int64
        let durationSec: Double

        let startingTrainingStep: Int
        let endingTrainingStep: Int

        let startingTotalPositions: Int
        let endingTotalPositions: Int

        let startingSelfPlayGames: Int
        let endingSelfPlayGames: Int

        let buildNumber: Int?
        let buildGitHash: String?
        let buildGitDirty: Bool?

        // Optional summary captured at segment-end (last [STATS] tick
        // values). Used by detail views; not required for cumulative
        // status-bar metrics.
        var endPolicyEntropy: Double?
        var endLossTotal: Double?
        var endGradNorm: Double?
    }

    // MARK: JSON serialization

    static func decode(_ data: Data) throws -> SessionCheckpointState {
        do {
            let state = try JSONDecoder().decode(SessionCheckpointState.self, from: data)
            guard state.formatVersion == Self.currentFormatVersion else {
                throw SessionCheckpointError.unsupportedVersion(state.formatVersion)
            }
            return state
        } catch let err as SessionCheckpointError {
            throw err
        } catch {
            throw SessionCheckpointError.invalidJSON(
                error,
                detail: String(data: data.prefix(2000), encoding: .utf8) ?? "(non-utf8)"
            )
        }
    }

    /// Return a copy with `trainingSegments` replaced. Builder helper
    /// that lets `buildCurrentSessionState` construct the bulk of the
    /// state via the synthesized memberwise init (which is already at
    /// the SwiftUI/Swift type-checker complexity threshold) and then
    /// layer the segments in afterward, without forcing the init call
    /// site to grow another argument.
    func withTrainingSegments(_ segments: [TrainingSegment]?) -> SessionCheckpointState {
        var copy = self
        copy.trainingSegments = segments
        return copy
    }

    /// Return a copy with the chart-data fields filled in. Same
    /// reason as `withTrainingSegments`: keeps the memberwise init
    /// call site lean. Pass `nil` for `hasChartData` when no chart
    /// snapshot is being saved (no companion files written).
    func withChartData(
        hasChartData: Bool?,
        trainingChartSampleCount: Int?,
        progressRateSampleCount: Int?,
        arenaChartEvents: [ArenaChartEvent]?,
        legalMassMaxAllTime: Double?
    ) -> SessionCheckpointState {
        var copy = self
        copy.hasChartData = hasChartData
        copy.trainingChartSampleCount = trainingChartSampleCount
        copy.progressRateSampleCount = progressRateSampleCount
        copy.arenaChartEvents = arenaChartEvents
        copy.legalMassMaxAllTime = legalMassMaxAllTime
        return copy
    }

    func encode() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        do {
            return try encoder.encode(self)
        } catch {
            throw SessionCheckpointError.invalidJSON(error)
        }
    }
}

// MARK: - Directory layout

/// Filenames for the three items inside a `.dcmsession` directory.
/// A session is a plain directory (not a bundle) so the
/// constituent `.dcmmodel` files are immediately usable when
/// copied out in Finder.
enum SessionCheckpointLayout {
    static let championFilename = "champion.dcmmodel"
    static let trainerFilename = "trainer.dcmmodel"
    static let stateFilename = "session.json"
    static let replayBufferFilename = "replay_buffer.bin"
    static let trainingChartFilename = "training_chart.json"
    static let progressRateChartFilename = "progress_rate_chart.json"

    static func championURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(championFilename)
    }

    static func trainerURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(trainerFilename)
    }

    static func stateURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(stateFilename)
    }

    static func replayBufferURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(replayBufferFilename)
    }

    static func trainingChartURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(trainingChartFilename)
    }

    static func progressRateChartURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(progressRateChartFilename)
    }

    /// Read the three raw payloads out of a session directory.
    /// Parsing is deferred to the caller so errors from the two
    /// `.dcmmodel` files surface with their original
    /// `ModelCheckpointError` types.
    static func readAll(
        from directoryURL: URL
    ) throws -> (stateData: Data, championData: Data, trainerData: Data) {
        // Normalize the incoming URL to a plain file-path URL.
        // The file importer on macOS can return file-reference URLs
        // or bookmark URLs whose `appendingPathComponent` doesn't
        // resolve to the expected child path. Reconstructing via
        // `URL(fileURLWithPath:isDirectory:)` strips any of that
        // metadata and gives a clean POSIX-path-based URL whose
        // children resolve correctly.
        let normalizedDir = URL(fileURLWithPath: directoryURL.path, isDirectory: true)
        let fm = FileManager.default
        let championURL = championURL(in: normalizedDir)
        let trainerURL = trainerURL(in: normalizedDir)
        let stateURL = stateURL(in: normalizedDir)

        guard fm.fileExists(atPath: championURL.path) else {
            throw SessionCheckpointError.missingChampionFile
        }
        guard fm.fileExists(atPath: trainerURL.path) else {
            throw SessionCheckpointError.missingTrainerFile
        }
        guard fm.fileExists(atPath: stateURL.path) else {
            throw SessionCheckpointError.missingSessionJSON
        }

        let stateData = try Data(contentsOf: stateURL)
        let championData = try Data(contentsOf: championURL)
        let trainerData = try Data(contentsOf: trainerURL)
        return (stateData, championData, trainerData)
    }
}
