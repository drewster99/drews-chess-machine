import Foundation

/// Thread-safe accumulator for the per-run events the `--output`
/// JSON snapshot needs: one entry per `[STATS]` line, one entry per
/// completed arena, one entry per 15-second candidate-test probe.
/// All append paths fire from the training TaskGroup's various
/// background tasks (stats logger, arena coordinator, training
/// worker), so mutation is serialized through a private `NSLock` —
/// same lock-protected-class pattern the rest of this codebase
/// uses (ReplayBuffer, ParallelWorkerStatsBox, etc.).
///
/// The recorder is allocated once per Play-and-Train session when
/// `--output` is active; the app holds it through `ContentView`
/// state and hands it to the training task at session start. On
/// time-limit expiry (or any other flush point) `finalize(...)`
/// assembles the Codable root struct and writes the JSON to disk.
final class CliTrainingRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var arenas: [Arena] = []
    private var stats: [StatsLine] = []
    private var probes: [CandidateTest] = []

    /// Session ID captured at first append for inclusion in the
    /// top-level JSON. Written through `setSessionID(_:)` so the
    /// recorder doesn't have to read the main-actor-isolated
    /// `currentSessionID` directly.
    private var sessionID: String?

    /// Termination reason captured by the writing path (timer task
    /// or collapse detector). Readers should call
    /// `setTerminationReason(_:)` before `writeJSON(...)`.
    private var terminationReason: TerminationReason?

    init() {}

    func setSessionID(_ id: String?) {
        lock.lock(); defer { lock.unlock() }
        sessionID = id
    }

    /// Record how the run ended. Safe to call from any thread — the
    /// value is included in the next snapshot write.
    func setTerminationReason(_ reason: TerminationReason) {
        lock.lock(); defer { lock.unlock() }
        terminationReason = reason
    }

    func appendArena(_ a: Arena) {
        lock.lock(); defer { lock.unlock() }
        arenas.append(a)
    }

    func appendStats(_ s: StatsLine) {
        lock.lock(); defer { lock.unlock() }
        stats.append(s)
    }

    func appendCandidateTest(_ p: CandidateTest) {
        lock.lock(); defer { lock.unlock() }
        probes.append(p)
    }

    /// Cheap lock-protected counts used by the post-write log line
    /// so the caller doesn't have to inspect the JSON file after
    /// writing it to confirm how many events were captured.
    func countsSnapshot() -> (arenas: Int, stats: Int, probes: Int) {
        lock.lock(); defer { lock.unlock() }
        return (arenas.count, stats.count, probes.count)
    }

    /// Encode the Codable snapshot to `Data`. Shared back-end of
    /// `writeJSON(to:)` and `writeJSONToStdout(...)` so both paths
    /// emit byte-identical output. Holds the lock only for the
    /// array copies and releases it before the JSON encode step,
    /// which doesn't need the recorder's state.
    func encodedJSONData(totalTrainingSeconds: Double) throws -> Data {
        lock.lock()
        let snapshot = Snapshot(
            totalTrainingSeconds: totalTrainingSeconds,
            trainingElapsedSeconds: totalTrainingSeconds,
            terminationReason: terminationReason,
            sessionID: sessionID,
            trainingSteps: stats.last?.steps,
            positionsTrained: stats.last?.positionsTrained,
            arenaResults: arenas,
            stats: stats,
            candidateTests: probes
        )
        lock.unlock()

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(snapshot)
    }

    /// Build the Codable root struct and write it to `url`,
    /// overwriting any existing file at the same path. `url` must
    /// name a writable directory; callers resolve tildes and
    /// relative paths before handing the URL here so error
    /// messages point at the fully-resolved path.
    func writeJSON(to url: URL, totalTrainingSeconds: Double) throws {
        let data = try encodedJSONData(totalTrainingSeconds: totalTrainingSeconds)
        // Atomic write so a crash mid-write doesn't leave the
        // previous version half-overwritten. `.atomic` also
        // creates the file if it doesn't exist and replaces it
        // if it does — matching the spec's "filename will be
        // overwritten if it exists" requirement.
        try data.write(to: url, options: [.atomic])
    }

    /// Write the JSON snapshot to stdout, followed by a newline
    /// so the caller's shell prompt doesn't sit at the end of
    /// the closing `}`. Used when `--train` hits its
    /// `training_time_limit` and no `--output <file>` was
    /// provided — the user wants the JSON on stdout for shell
    /// redirection, piping to `jq`, etc. Writes are sent through
    /// `FileHandle.standardOutput` rather than `print()` so the
    /// output is a single binary-safe blob without Swift's
    /// per-line flushing.
    func writeJSONToStdout(totalTrainingSeconds: Double) throws {
        let data = try encodedJSONData(totalTrainingSeconds: totalTrainingSeconds)
        let stdout = FileHandle.standardOutput
        try stdout.write(contentsOf: data)
        try stdout.write(contentsOf: Data([0x0A]))  // trailing newline
    }

    // MARK: - Root snapshot

    /// Reason the --train session ended. Written as a top-level
    /// `termination_reason` string in the output JSON so autotrain
    /// and offline analysis can distinguish a clean deadline expiry
    /// from an aborted collapse without parsing the stats stream.
    /// Snake-case raw values are what lands in the JSON.
    enum TerminationReason: String, Encodable, Sendable {
        /// The `training_time_limit` deadline fired and the snapshot
        /// was written cleanly at its scheduled moment.
        case timerExpired = "timer_expired"
        /// The legal-mass collapse detector found `illegalMass`
        /// above threshold for enough consecutive probes that the
        /// run was aborted early; the snapshot is still written
        /// with whatever telemetry had been captured up to that
        /// point.
        case legalMassCollapse = "legal_mass_collapse"
        /// User-initiated stop (UI Stop button or equivalent)
        /// reached the CLI snapshot writer. Placeholder for future
        /// wiring — the current CLI path doesn't expose a manual-
        /// stop hook, but the enum value is defined so downstream
        /// readers can match on it once it does.
        case manualStop = "manual_stop"
        /// An unrecoverable error during training tripped the
        /// snapshot-then-exit path. Placeholder for future wiring.
        case error = "error"
        /// SIGUSR1 received — autotrain's mid-run hard-reject early-stop
        /// signal. Snapshot is written at the moment the signal lands;
        /// the run did NOT complete its requested window. Treat as a
        /// truncated-window run for analysis (full H1–H7 / S1–S5 /
        /// positive-bands evaluation runs as normal — the data is real).
        case sigusr1Requested = "SIGUSR1-user-requested"
        /// SIGHUP received — typically the controlling-tty disconnect
        /// path on macOS, or `pkill -HUP`. Same truncated-window
        /// semantics as `sigusr1Requested`.
        case sighupReceived = "SIGHUP-received"
        /// AppKit-driven termination (Quit menu, `NSApp.terminate(_:)`,
        /// AppleScript `quit`, logout/shutdown). Routes through the
        /// AppDelegate's applicationShouldTerminate/applicationWillTerminate
        /// flush hooks. Same truncated-window semantics.
        case appWillTerminate = "app-will-terminate"
    }

    struct Snapshot: Encodable, Sendable {
        let totalTrainingSeconds: Double
        /// Duplicate of `totalTrainingSeconds` under a more
        /// self-explanatory key. `total_training_seconds` has
        /// historically been emitted and is kept for backward
        /// compatibility with existing analysis tools, but going
        /// forward the dashboard / autotrain skill reads this
        /// field, which is named to make its meaning obvious at a
        /// glance ("how long did training actually run for," not
        /// "what was the training time budget").
        let trainingElapsedSeconds: Double
        /// How the run ended. See `TerminationReason`. Nil only in
        /// the (historical) case where the snapshot is written by
        /// a path that didn't set a reason; callers are expected to
        /// set it before `writeJSON`.
        let terminationReason: TerminationReason?
        let sessionID: String?
        /// Trainer steps at the moment of the last [STATS] line.
        /// Nil when the run ended before any stats line fired
        /// (e.g. time limit below bootstrap cadence).
        let trainingSteps: Int?
        /// Self-play positions produced at the moment of the last
        /// [STATS] line. Nil for the same reason as above.
        let positionsTrained: Int?
        let arenaResults: [Arena]
        let stats: [StatsLine]
        let candidateTests: [CandidateTest]

        enum CodingKeys: String, CodingKey {
            case totalTrainingSeconds = "total_training_seconds"
            case trainingElapsedSeconds = "training_elapsed_seconds"
            case terminationReason = "termination_reason"
            case sessionID = "session_id"
            case trainingSteps = "training_steps"
            case positionsTrained = "positions_trained"
            case arenaResults = "arena_results"
            case stats
            case candidateTests = "candidate_tests"
        }
    }

    // MARK: - Arena

    struct Arena: Encodable, Sendable {
        let index: Int
        let finishedAtStep: Int
        let gamesPlayed: Int
        let tournamentGames: Int
        let candidateWins: Int
        let championWins: Int
        let draws: Int
        let score: Double
        let drawRate: Double
        let elo: Double?
        let eloLo: Double?
        let eloHi: Double?
        let scoreLo: Double
        let scoreHi: Double
        let candidateWinsAsWhite: Int
        let candidateDrawsAsWhite: Int
        let candidateLossesAsWhite: Int
        let candidateWinsAsBlack: Int
        let candidateDrawsAsBlack: Int
        let candidateLossesAsBlack: Int
        let candidateScoreAsWhite: Double
        let candidateScoreAsBlack: Double
        let promoted: Bool
        let promotionKind: String?
        let promotedID: String?
        let durationSec: Double
        let candidateID: String
        let championID: String
        let trainerID: String
        let learningRate: Double
        let promoteThreshold: Double
        let batchSize: Int
        let workerCount: Int
        let spStartTau: Double
        let spFloorTau: Double
        let spDecayPerPly: Double
        let arStartTau: Double
        let arFloorTau: Double
        let arDecayPerPly: Double
        let diversityUniqueGames: Int
        let diversityGamesInWindow: Int
        let diversityUniquePercent: Double
        let diversityAvgDivergencePly: Double
        let buildNumber: Int

        enum CodingKeys: String, CodingKey {
            case index
            case finishedAtStep = "finished_at_step"
            case gamesPlayed = "games_played"
            case tournamentGames = "arena_games_per_tournament"
            case candidateWins = "candidate_wins"
            case championWins = "champion_wins"
            case draws
            case score
            case drawRate = "draw_rate"
            case elo
            case eloLo = "elo_lo"
            case eloHi = "elo_hi"
            case scoreLo = "score_lo"
            case scoreHi = "score_hi"
            case candidateWinsAsWhite = "candidate_wins_as_white"
            case candidateDrawsAsWhite = "candidate_draws_as_white"
            case candidateLossesAsWhite = "candidate_losses_as_white"
            case candidateWinsAsBlack = "candidate_wins_as_black"
            case candidateDrawsAsBlack = "candidate_draws_as_black"
            case candidateLossesAsBlack = "candidate_losses_as_black"
            case candidateScoreAsWhite = "candidate_score_as_white"
            case candidateScoreAsBlack = "candidate_score_as_black"
            case promoted
            case promotionKind = "promotion_kind"
            case promotedID = "promoted_id"
            case durationSec = "duration_sec"
            case candidateID = "candidate_id"
            case championID = "champion_id"
            case trainerID = "trainer_id"
            case learningRate = "learning_rate"
            case promoteThreshold = "arena_promote_threshold"
            case batchSize = "batch_size"
            case workerCount = "worker_count"
            case spStartTau = "self_play_start_tau"
            case spFloorTau = "self_play_floor_tau"
            case spDecayPerPly = "self_play_decay_per_ply"
            case arStartTau = "arena_start_tau"
            case arFloorTau = "arena_floor_tau"
            case arDecayPerPly = "arena_decay_per_ply"
            case diversityUniqueGames = "diversity_unique_games"
            case diversityGamesInWindow = "diversity_games_in_window"
            case diversityUniquePercent = "diversity_unique_percent"
            case diversityAvgDivergencePly = "diversity_avg_divergence_ply"
            case buildNumber = "build_number"
        }
    }

    // MARK: - [STATS] snapshot

    struct StatsLine: Encodable, Sendable {
        let elapsedSec: Double
        let steps: Int
        let selfPlayGames: Int
        /// Total self-play positions added to the replay buffer
        /// since the session began — this is the "positions
        /// trained" counter in the top-level JSON when it's the
        /// last stats line at exit time.
        let positionsTrained: Int
        let avgLen: Double
        let rollingAvgLen: Double
        let gameLenP50: Int?
        let gameLenP95: Int?
        let bufferCount: Int
        let bufferCapacity: Int
        let policyLoss: Double?
        let valueLoss: Double?
        let policyEntropy: Double?
        let gradGlobalNorm: Double?
        let policyHeadWeightNorm: Double?
        let policyLogitAbsMax: Double?
        let playedMoveProb: Double?
        let playedMoveProbPosAdv: Double?
        let playedMoveProbPosAdvSkipped: Int
        let playedMoveProbNegAdv: Double?
        let playedMoveProbNegAdvSkipped: Int
        let playedMoveCondWindowSize: Int
        let legalMass: Double?
        let top1LegalFraction: Double?
        let valueMean: Double?
        let valueAbsMean: Double?
        let vBaselineDelta: Double?
        let advMean: Double?
        let advStd: Double?
        let advMin: Double?
        let advMax: Double?
        let advFracPositive: Double?
        let advFracSmall: Double?
        let advP05: Double?
        let advP50: Double?
        let advP95: Double?
        let spStartTau: Double
        let spFloorTau: Double
        let spDecayPerPly: Double
        let arStartTau: Double
        let arFloorTau: Double
        let arDecayPerPly: Double
        let diversityUniqueGames: Int
        let diversityGamesInWindow: Int
        let diversityUniquePercent: Double
        let diversityAvgDivergencePly: Double
        let ratioTarget: Double
        let ratioCurrent: Double
        let ratioProductionRate: Double
        let ratioConsumptionRate: Double
        let ratioAutoAdjust: Bool
        let ratioComputedDelayMs: Int
        let whiteCheckmates: Int
        let blackCheckmates: Int
        let stalemates: Int
        let fiftyMoveDraws: Int
        let threefoldRepetitionDraws: Int
        let insufficientMaterialDraws: Int
        let batchSize: Int
        let learningRate: Double
        let promoteThreshold: Double
        let arenaGames: Int
        let workerCount: Int
        let gradClipMaxNorm: Double
        let weightDecayC: Double
        let entropyRegularizationCoeff: Double
        let drawPenalty: Double
        let policyScaleK: Double
        let buildNumber: Int
        let trainerID: String
        let championID: String

        enum CodingKeys: String, CodingKey {
            case elapsedSec = "elapsed_sec"
            case steps
            case selfPlayGames = "self_play_games"
            case positionsTrained = "positions_trained"
            case avgLen = "avg_len"
            case rollingAvgLen = "rolling_avg_len"
            case gameLenP50 = "game_len_p50"
            case gameLenP95 = "game_len_p95"
            case bufferCount = "buffer_count"
            case bufferCapacity = "buffer_capacity"
            case policyLoss = "policy_loss"
            case valueLoss = "value_loss"
            case policyEntropy = "policy_entropy"
            case gradGlobalNorm = "grad_global_norm"
            case policyHeadWeightNorm = "policy_head_weight_norm"
            case policyLogitAbsMax = "policy_logit_abs_max"
            case playedMoveProb = "played_move_prob"
            case playedMoveProbPosAdv = "played_move_prob_pos_adv"
            case playedMoveProbPosAdvSkipped = "played_move_prob_pos_adv_skipped"
            case playedMoveProbNegAdv = "played_move_prob_neg_adv"
            case playedMoveProbNegAdvSkipped = "played_move_prob_neg_adv_skipped"
            case playedMoveCondWindowSize = "played_move_cond_window_size"
            case legalMass = "legal_mass"
            case top1LegalFraction = "top1_legal_fraction"
            case valueMean = "value_mean"
            case valueAbsMean = "value_abs_mean"
            case vBaselineDelta = "v_baseline_delta"
            case advMean = "adv_mean"
            case advStd = "adv_std"
            case advMin = "adv_min"
            case advMax = "adv_max"
            case advFracPositive = "adv_frac_positive"
            case advFracSmall = "adv_frac_small"
            case advP05 = "adv_p05"
            case advP50 = "adv_p50"
            case advP95 = "adv_p95"
            case spStartTau = "self_play_start_tau"
            case spFloorTau = "self_play_floor_tau"
            case spDecayPerPly = "self_play_decay_per_ply"
            case arStartTau = "arena_start_tau"
            case arFloorTau = "arena_floor_tau"
            case arDecayPerPly = "arena_decay_per_ply"
            case diversityUniqueGames = "diversity_unique_games"
            case diversityGamesInWindow = "diversity_games_in_window"
            case diversityUniquePercent = "diversity_unique_percent"
            case diversityAvgDivergencePly = "diversity_avg_divergence_ply"
            case ratioTarget = "ratio_target"
            case ratioCurrent = "ratio_current"
            case ratioProductionRate = "ratio_production_rate"
            case ratioConsumptionRate = "ratio_consumption_rate"
            case ratioAutoAdjust = "ratio_auto_adjust"
            case ratioComputedDelayMs = "ratio_computed_delay_ms"
            case whiteCheckmates = "white_checkmates"
            case blackCheckmates = "black_checkmates"
            case stalemates
            case fiftyMoveDraws = "fifty_move_draws"
            case threefoldRepetitionDraws = "threefold_repetition_draws"
            case insufficientMaterialDraws = "insufficient_material_draws"
            case batchSize = "batch_size"
            case learningRate = "learning_rate"
            case promoteThreshold = "arena_promote_threshold"
            case arenaGames = "arena_games_per_tournament"
            case workerCount = "worker_count"
            case gradClipMaxNorm = "grad_clip_max_norm"
            case weightDecayC = "weight_decay"
            case entropyRegularizationCoeff = "entropy_regularization_coeff"
            case drawPenalty = "draw_penalty"
            case policyScaleK = "policy_scale_k"
            case buildNumber = "build_number"
            case trainerID = "trainer_id"
            case championID = "champion_id"
        }
    }

    // MARK: - Candidate test probe

    struct CandidateTest: Encodable, Sendable {
        let elapsedSec: Double
        /// Monotonic count of probes that have fired this session —
        /// the same counter the UI shows next to the probe results.
        let probeIndex: Int
        let probeNetworkTarget: String
        let inferenceTimeMs: Double
        let valueHead: ValueHead
        let policyHead: PolicyHead

        enum CodingKeys: String, CodingKey {
            case elapsedSec = "elapsed_sec"
            case probeIndex = "probe_index"
            case probeNetworkTarget = "probe_network_target"
            case inferenceTimeMs = "inference_time_ms"
            case valueHead = "value_head"
            case policyHead = "policy_head"
        }

        struct ValueHead: Encodable, Sendable {
            let output: Double
        }

        struct PolicyHead: Encodable, Sendable {
            let policyStats: PolicyStats
            /// Top-10 raw policy cells (by probability), including
            /// illegal candidates — matches the on-screen diagnostic
            /// display which shows illegal moves too so the user can
            /// tell when the policy hasn't yet learned move validity.
            let topRaw: [TopMove]

            enum CodingKeys: String, CodingKey {
                case policyStats = "policy_stats"
                case topRaw = "top_raw"
            }
        }

        struct PolicyStats: Encodable, Sendable {
            let sum: Double
            let top100Sum: Double
            let aboveUniformCount: Int
            let legalMoveCount: Int
            let legalUniformThreshold: Double
            let legalMassSum: Double
            let illegalMassSum: Double
            let min: Double
            let max: Double

            enum CodingKeys: String, CodingKey {
                case sum
                case top100Sum = "top100_sum"
                case aboveUniformCount = "above_uniform_count"
                case legalMoveCount = "legal_move_count"
                case legalUniformThreshold = "legal_uniform_threshold"
                case legalMassSum = "legal_mass_sum"
                case illegalMassSum = "illegal_mass_sum"
                case min
                case max
            }
        }

        struct TopMove: Encodable, Sendable {
            let rank: Int
            let from: String
            let to: String
            let fromRow: Int
            let fromCol: Int
            let toRow: Int
            let toCol: Int
            let probability: Double
            let isLegal: Bool

            enum CodingKeys: String, CodingKey {
                case rank
                case from
                case to
                case fromRow = "from_row"
                case fromCol = "from_col"
                case toRow = "to_row"
                case toCol = "to_col"
                case probability
                case isLegal = "is_legal"
            }
        }
    }
}
