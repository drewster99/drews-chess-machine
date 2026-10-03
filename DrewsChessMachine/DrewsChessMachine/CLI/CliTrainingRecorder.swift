import Foundation
import os

/// Thread-safe accumulator for the per-run events the `--output`
/// JSON snapshot needs: one entry per `[STATS]` line, one entry per
/// completed arena, one entry per 15-second candidate-test probe.
/// All append paths fire from the training TaskGroup's various
/// background tasks (stats logger, arena coordinator, training
/// worker), so mutation is serialized through a private
/// `OSAllocatedUnfairLock<State>` — same lock-protected-class
/// pattern the rest of this codebase uses (ReplayBuffer,
/// ParallelWorkerStatsBox, etc.).
///
/// The recorder is allocated once per Play-and-Train session when
/// `--output` is active; the app holds it through `ContentView`
/// state and hands it to the training task at session start. On
/// time-limit expiry (or any other flush point) `finalize(...)`
/// assembles the Codable root struct and writes the JSON to disk.
final class CliTrainingRecorder: @unchecked Sendable {
    private struct State {
        var arenas: [Arena] = []
        var stats: [StatsLine] = []
        var probes: [CandidateTest] = []
        var layerHealth: [LayerHealthRecord] = []
        /// Session ID captured at first append for inclusion in the
        /// top-level JSON. Written through `setSessionID(_:)` so the
        /// recorder doesn't have to read the main-actor-isolated
        /// `currentSessionID` directly.
        var sessionID: String?
        /// Termination reason captured by the writing path (timer task
        /// or collapse detector). Readers should call
        /// `setTerminationReason(_:)` before `writeJSON(...)`.
        var terminationReason: TerminationReason?
        /// Id of the corpus self-play games are recorded into, surfaced in
        /// results.json provenance. Set via setRecordingCorpusID(_:).
        var recordingCorpusID: String?
        /// Which driver is producing this run. See `RunKind`.
        var runKind: RunKind?
        /// The run's master seed, set once at run start.
        var runRandomSeed: RunRandomSeed?
        /// The lineage of the run's latest save; the last one set is the
        /// run's final lineage.
        var finalLineage: ResultsLineage?
    }
    private let lock = OSAllocatedUnfairLock<State>(initialState: State())

    init() {}

    func setSessionID(_ id: String?) {
        lock.withLock { $0.sessionID = id }
    }

    func setRecordingCorpusID(_ id: String?) {
        lock.withLock { $0.recordingCorpusID = id }
    }

    /// Declare which driver is producing this run. Call once at setup, next to
    /// `setSessionID(_:)`.
    func setRunKind(_ kind: RunKind) {
        lock.withLock { $0.runKind = kind }
    }

    /// Record the run's master seed (resolved once at run start). Call next
    /// to `setRunKind(_:)`.
    func setRunRandomSeed(_ seed: RunRandomSeed) {
        lock.withLock { $0.runRandomSeed = seed }
    }

    /// Record how the run ended. Safe to call from any thread — the
    /// value is included in the next snapshot write.
    /// Record the lineage of a save the run just wrote, with the saved
    /// file's `content_sha256` (nil when the record was not written to a
    /// file). Each save replaces the previous one, so the record in
    /// `results.json` is the run's last.
    func setFinalLineage(_ record: LineageRecord, checkpointSHA256: String?) {
        lock.withLock { $0.finalLineage = ResultsLineage(record: record, checkpointSHA256: checkpointSHA256) }
    }

    /// Record the lineage of a save just written to `url`, with the file's
    /// `content_sha256` read back from its header. A failed read is logged
    /// through `log` and the save's lineage is recorded without the hash; it
    /// never fails the save, which already succeeded.
    func recordSave(of record: LineageRecord, savedAt url: URL, log: (String) -> Void) {
        do {
            let header = try ModelFileCatalog.headerMetadata(at: url)
            setFinalLineage(record, checkpointSHA256: header[SafetensorsFile.contentHashKey])
        } catch {
            log("[RESULTS] reading \(url.lastPathComponent)'s content hash for results.json failed: "
                + "\(error.localizedDescription); its lineage is recorded without checkpoint_sha256")
            setFinalLineage(record, checkpointSHA256: nil)
        }
    }

    func setTerminationReason(_ reason: TerminationReason) {
        lock.withLock { $0.terminationReason = reason }
    }

    func appendArena(_ a: Arena) {
        lock.withLock { $0.arenas.append(a) }
    }

    func appendStats(_ s: StatsLine) {
        lock.withLock { $0.stats.append(s) }
    }

    func appendCandidateTest(_ p: CandidateTest) {
        lock.withLock { $0.probes.append(p) }
    }

    func appendLayerHealth(_ record: LayerHealthRecord) {
        lock.withLock { $0.layerHealth.append(record) }
    }

    /// Cheap lock-protected counts used by the post-write log line
    /// so the caller doesn't have to inspect the JSON file after
    /// writing it to confirm how many events were captured.
    func countsSnapshot() -> (arenas: Int, stats: Int, probes: Int) {
        lock.withLock { ($0.arenas.count, $0.stats.count, $0.probes.count) }
    }

    /// Encode the Codable snapshot to `Data`. Shared back-end of
    /// `writeJSON(to:)` and `writeJSONToStdout(...)` so both paths
    /// emit a byte-identical JSON payload. Holds the lock only for the
    /// array copies (assembling the Snapshot value) and releases it
    /// before the JSON encode step, which doesn't need the
    /// recorder's state.
    func encodedJSONData(totalTrainingSeconds: Double) throws -> Data {
        let snapshot = lock.withLock { state in
            Snapshot(
                runKind: state.runKind,
                totalTrainingSeconds: totalTrainingSeconds,
                trainingElapsedSeconds: totalTrainingSeconds,
                terminationReason: state.terminationReason,
                sessionID: state.sessionID,
                trainingSteps: state.stats.last?.steps,
                positionsTrained: state.stats.last?.positionsTrained,
                arenaResults: state.arenas,
                stats: state.stats,
                candidateTests: state.probes,
                layerHealth: state.layerHealth,
                recordingCorpusID: state.recordingCorpusID,
                randomSeed: state.runRandomSeed.map { String($0.masterSeed) },
                randomSeedMode: state.runRandomSeed.map { $0.effectiveMode.logToken },
                rngStreamDerivation: state.runRandomSeed.map { _ in DCMRandomStreams.derivationVersion },
                lineage: state.finalLineage
            )
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(snapshot)
    }

    /// Build the Codable root struct and write it to `url`, replacing a
    /// regular file already there (never a folder or link). An explicit
    /// replace with no ownership check: runs write their results through
    /// `write(to:totalTrainingSeconds:)` with a `CliResultsOutput` checked
    /// before training, which never replaces a file the run was not told
    /// it may replace.
    func writeJSON(to url: URL, totalTrainingSeconds: Double) throws {
        let data = try encodedJSONData(totalTrainingSeconds: totalTrainingSeconds)
        try FileSafety.replaceRegularFile(data, at: url, expectedIdentity: nil)
    }

    /// Write the results to the destination checked before the run
    /// (`CliResultsOutput.preflight`) and return where they landed.
    ///
    /// A file the pre-flight found is replaced only when `--overwrite-output`
    /// authorized it, and only if it is still that file. If something has
    /// appeared at — or replaced the file at — the destination since the
    /// check, it is left untouched and the results go to a new numbered
    /// sibling (`<name>-2.<ext>`, …) instead, with an alarm naming both: a
    /// long run's results are never lost, and nothing the run does not own is
    /// overwritten.
    @discardableResult
    func write(to output: CliResultsOutput, totalTrainingSeconds: Double) throws -> URL {
        let data = try encodedJSONData(totalTrainingSeconds: totalTrainingSeconds)
        do {
            if let replacing = output.replacing {
                try FileSafety.replaceRegularFile(data, at: output.url, expectedIdentity: replacing)
            } else {
                try FileSafety.publishNewFile(data, to: output.url)
            }
            return output.url
        } catch let refusal as FileSafetyError where refusal.isOwnershipRefusal {
            let sibling = try Self.writeBeside(output.url, data: data)
            let message = "[ALARM] results: \(output.url.path) changed during the run (\(refusal.localizedDescription)); "
                + "left it untouched and wrote this run's results to \(sibling.path)"
            SessionLogger.shared.log(message)
            FileHandle.standardError.write(Data((message + "\n").utf8))
            return sibling
        }
    }

    /// Write `data` to a new numbered sibling of `url`. A name with no
    /// extension gets `.json`, since that is what the file holds.
    private static func writeBeside(_ url: URL, data: Data) throws -> URL {
        let pathExtension = url.pathExtension.isEmpty ? "json" : url.pathExtension
        let stem = url.pathExtension.isEmpty ? url.lastPathComponent : url.deletingPathExtension().lastPathComponent
        let created = try FileSafety.createNewFileWithNumericSuffix(
            in: url.deletingLastPathComponent(), stem: stem, pathExtension: pathExtension,
            maxAttempts: resultsSiblingMaxAttempts)
        do {
            try created.handle.write(contentsOf: data)
            try FileSafety.fullSync(fileDescriptor: created.handle.fileDescriptor, path: created.url.path)
            try created.handle.close()
        } catch {
            do {
                try created.handle.close()
            } catch let closeError {
                SessionLogger.shared.log("[APP] results: closing \(created.url.path) after a failed write: \(closeError.localizedDescription)")
            }
            throw error
        }
        return created.url
    }

    /// How many numbered sibling names `write(to:)` tries.
    private static let resultsSiblingMaxAttempts = 100

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
        /// The `training_step_limit` step budget was reached (trainer
        /// completed-step counter crossed the configured value) and the
        /// snapshot was written cleanly.
        case stepLimitReached = "step_limit_reached"
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

    /// Which driver produced this run. Emitted once at top level so a consumer
    /// can tell what the per-line fields mean before reading them: `StatsLine`
    /// was shaped for self-play, so on the CLI paths some of its keys carry the
    /// analogous quantity under a self-play name (`self_play_games` holds
    /// completed train-vs-UCI games) and others are omitted entirely because the
    /// run never measured them.
    enum RunKind: String, Encodable, Sendable {
        case selfPlay = "self_play"
        case corpusReplay = "corpus_replay"
        case trainVsUci = "train_vs_uci"
    }

    /// A run's lineage totals at one stats tick.
    typealias LineageTotals = LineageTracker.Totals

    /// The run's lineage as `results.json` records it (determinism plan
    /// D5): the record of the run's last save minus its segment history and
    /// derivation history (both stay in the model file), plus the SHA-256
    /// of that saved file, the file the record describes.
    struct ResultsLineage: Encodable, Sendable {
        let record: LineageRecord
        /// `content_sha256` of the file the record was saved in; nil when the
        /// record was not written to a file.
        let checkpointSHA256: String?

        enum CodingKeys: String, CodingKey {
            case schema, run, parent, steps, fed, time, parameters, build, invocation, device, rng
            case checkpointSHA256 = "checkpoint_sha256"
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(record.schema, forKey: .schema)
            try c.encode(record.run, forKey: .run)
            try c.encode(record.parent, forKey: .parent)
            try c.encode(record.steps, forKey: .steps)
            try c.encode(record.fed, forKey: .fed)
            try c.encode(record.time, forKey: .time)
            try c.encode(record.parameters, forKey: .parameters)
            try c.encode(record.build, forKey: .build)
            try c.encode(record.invocation, forKey: .invocation)
            try c.encode(record.device, forKey: .device)
            try c.encode(record.rng, forKey: .rng)
            try c.encode(checkpointSHA256, forKey: .checkpointSHA256)
        }
    }

    struct Snapshot: Encodable, Sendable {
        /// See `RunKind`. Nil only for a snapshot written by a path that never
        /// declared itself.
        let runKind: RunKind?
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
        /// One full layer-health summary per trainer checkpoint save, in
        /// save order. The live `[LAYER-HEALTH]` lines are log-only.
        let layerHealth: [LayerHealthRecord]
        let recordingCorpusID: String?
        /// The run's master seed as a decimal string (a JSON number would lose
        /// the low bits of a seed above 2^53 in most readers), the mode it ran
        /// in (`RandomSeedMode.logToken`; a `--seed` run is seeded) and the
        /// stream-derivation identifier. Passing the seed back with `--seed`
        /// replays the run's random streams. Nil only when a path did not set
        /// the seed; every training path does.
        let randomSeed: String?
        let randomSeedMode: String?
        let rngStreamDerivation: String?
        /// The lineage of the run's last save (`setFinalLineage`); nil, and
        /// left out, for a run that recorded none.
        let lineage: ResultsLineage?

        enum CodingKeys: String, CodingKey {
            case runKind = "run_kind"
            case totalTrainingSeconds = "total_training_seconds"
            case trainingElapsedSeconds = "training_elapsed_seconds"
            case terminationReason = "termination_reason"
            case sessionID = "session_id"
            case trainingSteps = "training_steps"
            case positionsTrained = "positions_trained"
            case arenaResults = "arena_results"
            case stats
            case candidateTests = "candidate_tests"
            case layerHealth = "layer_health"
            case recordingCorpusID = "recording_corpus_id"
            case randomSeed = "random_seed"
            case randomSeedMode = "random_seed_mode"
            case rngStreamDerivation = "rng_stream_derivation"
            case lineage
        }
    }

    // MARK: - Layer health

    /// The full `LayerHealthSummary` of one trainer checkpoint save
    /// (`LayerHealthLog.checkpoint`), tagged with when it was taken.
    struct LayerHealthRecord: Encodable, Sendable {
        /// The run's own step count at the save (segment-local on the CLI
        /// paths, matching the `stats` lines' `steps`).
        let step: Int
        /// The trainer's completed-step clock at the save, when known.
        let trainerStep: Int?
        /// The save's context, e.g. `replay-autosave` (same text as the
        /// `[LAYER-HEALTH] checkpoint` log line).
        let context: String
        let summary: LayerHealthSummary

        enum CodingKeys: String, CodingKey {
            case step
            case trainerStep = "trainer_step"
            case context
            case summary
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

    /// Replay-buffer per-batch observability snapshot. Captured at
    /// each `[STATS]` tick from the trainer's most-recent stats
    /// batch. Field semantics match `ReplayBuffer.BatchStatsSummary`.
    struct BatchStatsSnapshot: Encodable, Sendable {
        /// Composition constraints in effect when the batch summarised
        /// here was sampled. `applied == false` means the batch took
        /// the legacy uniform fast path; the histograms below are then
        /// the pre-constraint distribution. `applied == true` means
        /// the histograms are post-sampling-constraints.
        struct SamplingConstraintsSnapshot: Encodable, Sendable {
            let applied: Bool
            let maxPerGame: Int
            let maxDrawPct: Int
            let targetLength: Int
            /// `true` when material-bucket stratification was active for
            /// this batch (the trainer drew with a per-bucket target
            /// distribution rather than uniformly). When this is true
            /// the W/D/L cap and per-game K cap above were bypassed —
            /// the `bucket_mix` field on the parent record reflects
            /// the post-stratification distribution.
            let stratifyByMaterial: Bool
            enum CodingKeys: String, CodingKey {
                case applied
                case maxPerGame = "max_per_game"
                case maxDrawPct = "max_draw_pct"
                case targetLength = "target_length"
                case stratifyByMaterial = "stratify_by_material"
            }
        }
        let step: Int
        let batchSize: Int
        let samplingConstraints: SamplingConstraintsSnapshot
        let uniqueCount: Int
        let uniquePct: Double
        let dupMax: Int
        /// Counts. `dup_distribution[k]` is the number of distinct
        /// hashes in the batch with multiplicity k; partition-style
        /// histograms (`phase_by_ply` etc.) sum to `batch_size`.
        let dupDistribution: [String: Int]
        let phaseByPlyHistogram: [String: Int]
        let phaseByMaterialHistogram: [String: Int]
        let gameLengthHistogram: [String: Int]
        let samplingTauHistogram: [String: Int]
        let workerIdHistogram: [String: Int]
        let outcomeHistogram: [String: Int]
        let phaseByPlyXOutcomeHistogram: [String: Int]
        /// Same histograms expressed as fractions. Partition
        /// histograms divide by `batchSize`; `dup_distribution_pct`
        /// divides by `uniqueCount` (so its values express "fraction
        /// of distinct hashes with multiplicity k").
        let dupDistributionPct: [String: Double]
        let phaseByPlyHistogramPct: [String: Double]
        let phaseByMaterialHistogramPct: [String: Double]
        let gameLengthHistogramPct: [String: Double]
        let samplingTauHistogramPct: [String: Double]
        let workerIdHistogramPct: [String: Double]
        let outcomeHistogramPct: [String: Double]
        let phaseByPlyXOutcomeHistogramPct: [String: Double]
        let bufferUniquePositions: Int
        let bufferStoredCount: Int
        /// Per-bucket counts in this batch, keyed by the analyzer's
        /// material-bucket labels (`"0-4"`, `"5-8"`, …). Sums to
        /// `batchSize`. Always populated regardless of stratification
        /// state — when stratification is off, this reflects the
        /// natural per-phase mix the buffer was holding.
        let bucketMix: [String: Int]
        /// Same per-bucket counts as fractions of `batchSize`.
        let bucketMixPct: [String: Double]
        /// `bufferUniquePositions / bufferStoredCount`. Range [0, 1].
        /// Distinguishes "buffer full of duplicates" (low) from
        /// "sampler happened to draw duplicates from a diverse
        /// buffer" (high here, low `unique_pct`).
        let bufferUniquePct: Double

        enum CodingKeys: String, CodingKey {
            case step
            case batchSize = "batch_size"
            case samplingConstraints = "sampling_constraints"
            case uniqueCount = "unique_count"
            case uniquePct = "unique_pct"
            case dupMax = "dup_max"
            case dupDistribution = "dup_distribution"
            case phaseByPlyHistogram = "phase_by_ply"
            case phaseByMaterialHistogram = "phase_by_material"
            case gameLengthHistogram = "game_length"
            case samplingTauHistogram = "sampling_tau"
            case workerIdHistogram = "worker_id"
            case outcomeHistogram = "outcome"
            case phaseByPlyXOutcomeHistogram = "phase_by_ply_x_outcome"
            case dupDistributionPct = "dup_distribution_pct"
            case phaseByPlyHistogramPct = "phase_by_ply_pct"
            case phaseByMaterialHistogramPct = "phase_by_material_pct"
            case gameLengthHistogramPct = "game_length_pct"
            case samplingTauHistogramPct = "sampling_tau_pct"
            case workerIdHistogramPct = "worker_id_pct"
            case outcomeHistogramPct = "outcome_pct"
            case phaseByPlyXOutcomeHistogramPct = "phase_by_ply_x_outcome_pct"
            case bufferUniquePositions = "buffer_unique_positions"
            case bufferStoredCount = "buffer_stored_count"
            case bufferUniquePct = "buffer_unique_pct"
            case bucketMix = "bucket_mix"
            case bucketMixPct = "bucket_mix_pct"
        }

        /// Auto-derives all `*_pct` fields from the counts.
        init(
            step: Int,
            batchSize: Int,
            samplingConstraints: SamplingConstraintsSnapshot,
            uniqueCount: Int,
            uniquePct: Double,
            dupMax: Int,
            dupDistribution: [String: Int],
            phaseByPlyHistogram: [String: Int],
            phaseByMaterialHistogram: [String: Int],
            gameLengthHistogram: [String: Int],
            samplingTauHistogram: [String: Int],
            workerIdHistogram: [String: Int],
            outcomeHistogram: [String: Int],
            phaseByPlyXOutcomeHistogram: [String: Int],
            bufferUniquePositions: Int,
            bufferStoredCount: Int,
            bucketMix: [String: Int]
        ) {
            self.step = step
            self.batchSize = batchSize
            self.samplingConstraints = samplingConstraints
            self.uniqueCount = uniqueCount
            self.uniquePct = uniquePct
            self.dupMax = dupMax
            self.dupDistribution = dupDistribution
            self.phaseByPlyHistogram = phaseByPlyHistogram
            self.phaseByMaterialHistogram = phaseByMaterialHistogram
            self.gameLengthHistogram = gameLengthHistogram
            self.samplingTauHistogram = samplingTauHistogram
            self.workerIdHistogram = workerIdHistogram
            self.outcomeHistogram = outcomeHistogram
            self.phaseByPlyXOutcomeHistogram = phaseByPlyXOutcomeHistogram
            self.bufferUniquePositions = bufferUniquePositions
            self.bufferStoredCount = bufferStoredCount
            self.bucketMix = bucketMix
            let bs = batchSize > 0 ? Double(batchSize) : 1
            let uc = uniqueCount > 0 ? Double(uniqueCount) : 1
            func pct(_ d: [String: Int], denom: Double) -> [String: Double] {
                var out: [String: Double] = [:]
                out.reserveCapacity(d.count)
                for (k, v) in d { out[k] = Double(v) / denom }
                return out
            }
            // dup_distribution counts distinct-hashes-by-multiplicity,
            // so the natural denominator is uniqueCount (Σ values =
            // uniqueCount, not batchSize).
            self.dupDistributionPct = pct(dupDistribution, denom: uc)
            self.phaseByPlyHistogramPct = pct(phaseByPlyHistogram, denom: bs)
            self.phaseByMaterialHistogramPct = pct(phaseByMaterialHistogram, denom: bs)
            self.gameLengthHistogramPct = pct(gameLengthHistogram, denom: bs)
            self.samplingTauHistogramPct = pct(samplingTauHistogram, denom: bs)
            self.workerIdHistogramPct = pct(workerIdHistogram, denom: bs)
            self.outcomeHistogramPct = pct(outcomeHistogram, denom: bs)
            self.phaseByPlyXOutcomeHistogramPct = pct(phaseByPlyXOutcomeHistogram, denom: bs)
            self.bucketMixPct = pct(bucketMix, denom: bs)
            self.bufferUniquePct = bufferStoredCount > 0
                ? Double(bufferUniquePositions) / Double(bufferStoredCount)
                : 0
        }
    }

    struct StatsLine: Encodable, Sendable {
        let elapsedSec: Double
        let steps: Int
        let selfPlayGames: Int
        /// Total positions PRODUCED since the session began — every ply played,
        /// including games later dropped by the ply cap or the draw filter. This
        /// is the "positions trained" counter in the top-level JSON when it's
        /// the last stats line at exit time.
        ///
        /// Raw produced, NOT fed-to-buffer: `maxPliesDropped`'s doc below states
        /// the same thing from the other side ("Included in `selfPlayGames` and
        /// `positionsTrained` — the games WERE played"), and self-play's source
        /// `ParallelWorkerStatsBox` bumps this counter in `recordDroppedGame` as
        /// well as `recordCompletedGame`. The fed-to-buffer count is
        /// `emittedPositions`. This doc previously said "added to the replay
        /// buffer", contradicting both — that was wrong, and it is what led the
        /// CLI runners to describe their own value against the wrong field.
        let positionsTrained: Int
        /// Lifetime count of self-play games that survived the
        /// per-game keep/drop filter (`selfPlayDrawKeepFraction`)
        /// and were flushed into the replay buffer. `<= selfPlayGames`;
        /// equal at default keepFraction of 1.0.
        let emittedGames: Int
        /// Lifetime count of plies emitted into the replay buffer
        /// (both colours summed across kept games). `<=
        /// positionsTrained` (raw produced).
        let emittedPositions: Int
        /// `selfPlayDrawKeepFraction` in effect at this stats tick.
        /// 1.0 = keep every drawn game; < 1.0 = filter drawn games
        /// stochastically.
        let selfPlayDrawKeepFraction: Double
        /// `selfPlayMaxPliesPerGame` in effect at this stats tick. Self-play
        /// games hitting this cap are dropped (never emitted) and
        /// counted in `maxPliesDropped` rather than W/D/L.
        let selfPlayMaxPliesPerGame: Int
        /// Lifetime count of self-play games that hit the
        /// `selfPlayMaxPliesPerGame` cap and were dropped. Included in
        /// `selfPlayGames` and `positionsTrained` (the games WERE
        /// played) but never in per-outcome W/D/L counts.
        let maxPliesDropped: Int
        let avgLen: Double
        let rollingAvgLen: Double
        let gameLenP50: Int?
        let gameLenP95: Int?
        let bufferCount: Int
        let bufferCapacity: Int
        let policyLoss: Double?
        let valueLoss: Double?
        let policyEntropy: Double?
        let policyIllegalMassPenalty: Double?
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
        /// Legal-masked Shannon entropy (in nats) over the legal-only
        /// renormalized softmax. Distinguishes "diffuse across legal
        /// moves" from "concentrating onto preferred legal moves" —
        /// the full-policy `policyEntropy` cannot tell those apart.
        let legalEntropy: Double?
        /// Mean policy loss over batch positions where outcome z > 0.5
        /// (the move was played in a winning game). Splitting the
        /// classic `policyLoss` average into win and loss halves
        /// makes the curve unambiguous: pLossWin negative means the
        /// network is concentrating on moves that correlate with
        /// wins (good); pLossLoss negative means it's concentrating
        /// on moves that correlate with losses (bad).
        let policyLossWin: Double?
        /// Mean policy loss over batch positions where z < -0.5.
        let policyLossLoss: Double?
        /// Latest replay-buffer batch-stats summary captured at this
        /// `[STATS]` tick. Mirrors the contents of the `[BATCH-STATS]`
        /// log line — unique-position ratio, ply-phase histogram,
        /// game-length histogram, temperature histogram, worker
        /// histogram, WLD counts, phase×outcome cross product — so
        /// post-run analysis can read them straight from
        /// result.json without parsing the log file. Nil until the
        /// first stats-collection batch lands or when
        /// `batch_stats_interval` is 0.
        let batchStats: BatchStatsSnapshot?
        let valueMean: Double?
        let valueAbsMean: Double?
        let valueProbWin: Double?
        let valueProbDraw: Double?
        let valueProbLoss: Double?
        let advMean: Double?
        let advStd: Double?
        let advMin: Double?
        let advMax: Double?
        let advFracPositive: Double?
        let advFracSmall: Double?
        let advP05: Double?
        let advP50: Double?
        let advP95: Double?
        let spStartTau: Double?
        let spFloorTau: Double?
        let spDecayPerPly: Double?
        let arStartTau: Double?
        let arFloorTau: Double?
        let arDecayPerPly: Double?
        let diversityUniqueGames: Int?
        let diversityGamesInWindow: Int?
        let diversityUniquePercent: Double?
        let diversityAvgDivergencePly: Double?
        let ratioTarget: Double?
        let ratioCurrent: Double
        /// Self-play EMITTED-positions rate (positions/sec) — the
        /// rate the replay-ratio target is computed against. Equals
        /// `ratioProducedRate` when `selfPlayDrawKeepFraction = 1.0`
        /// (default), strictly less when filtering is active.
        let ratioProductionRate: Double
        /// Self-play RAW-produced-positions rate (positions/sec) —
        /// every ply that came off the GPU, kept or dropped. Equal
        /// to `ratioProductionRate` at default keep-fraction;
        /// surfaced separately so the observed keep-fraction can be
        /// read off as `ratioProductionRate / ratioProducedRate`.
        let ratioProducedRate: Double
        let ratioConsumptionRate: Double
        /// Self-play production rate expressed in moves/hour (3600 ×
        /// `ratioProductionRate`). Same rolling 60-s window as the
        /// underlying production rate; provided as a convenience so
        /// post-run analysis doesn't need to re-derive the unit.
        let selfPlayMovesPerHour: Double
        /// Trainer consumption rate expressed in moves/hour (3600 ×
        /// `ratioConsumptionRate`). Same window/source as the
        /// production rate companion above.
        let trainingMovesPerHour: Double
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
        let promoteThreshold: Double?
        let arenaGames: Int?
        let workerCount: Int
        let gradClipMaxNorm: Double
        let weightDecayC: Double
        /// Channel-dropout rate in effect this tick. Surfaced so a dropout
        /// A/B sweep's `results.json` records which rate each arm ran at —
        /// the parameter the harness exists to compare.
        let dropoutRate: Double
        let entropyRegularizationCoeff: Double
        let drawPenalty: Double
        let policyLossWeight: Double
        let valueLossWeight: Double
        /// Effective base learning rate actually applied this tick (the LR
        /// cycle's geometric value when LR cycling is active, else the static
        /// `learningRate`), BEFORE warmup/√batch multipliers. Distinct from
        /// `learningRate` above, which is always the static configured base.
        let lrEffectiveBase: Double
        /// Effective Polyak momentum applied this tick (the momentum cycle's
        /// value when active, else the static coefficient).
        let momentumEffective: Double
        var lrCycleActive: Bool
        var momentumCycleActive: Bool
        let buildNumber: Int
        let trainerID: String
        let championID: String?
        /// Rolling mean of each head's shared logit offset, read before the
        /// loss centers it (`TrainStepTiming.policyLogitMean` /
        /// `valueLogitMean`). Nil when no stats step has landed yet. The
        /// defaults keep memberwise construction that predates these fields
        /// compiling; every production call site passes them explicitly.
        var policyLogitMean: Double? = nil
        var valueLogitMean: Double? = nil
        /// The LR cycle's current (possibly decayed) peak and trough — the
        /// bounds `lrEffectiveBase` is swinging between this tick. Nil when
        /// the LR cycle is not driving the LR, or on a path with no cycle.
        var lrCyclePeak: Double? = nil
        var lrCycleTrough: Double? = nil
        /// The decay horizon and momentum-follow mode in effect, so runs that
        /// differ only in their envelope are distinguishable in `results.json`.
        /// Nil on paths with no LR/momentum cycle.
        var lrCycleDecayHorizonSteps: Int? = nil
        var momentumFollowsLRCycle: Bool? = nil
        /// The policy-target label smoothing in effect this tick: the mode
        /// (`PolicyLabelSmoothingMode.logToken`), the fixed-total ε, and the
        /// per-move δ and cap. All four are recorded whatever the mode, so
        /// arms of a smoothing A/B are distinguishable in `results.json` and
        /// the inactive values are visible. Nil only on a construction site
        /// that predates them; every production call site passes them.
        var policyLabelSmoothingEpsilon: Double? = nil
        var policyLabelSmoothingMode: String? = nil
        var policyLabelSmoothingPerMove: Double? = nil
        var policyLabelSmoothingPerMoveCap: Double? = nil
        /// The process's policy-head tail precision
        /// (`ChessNetwork.PolicyTailPrecision.process`), which every network
        /// and trainer of the run is built with. Nil only on a construction
        /// site that predates it; every production call site sets it.
        var policyTailPrecision: String? = nil
        /// The run's lineage totals at this tick (`LineageTracker.totals`):
        /// the trainer clock, measured trainer-step seconds and games fed,
        /// each continuing across the sessions of the run. Nil where no
        /// predecessor recorded the total (a resume of a file written before
        /// lineage), and then left out of the row.
        let cumTrainerStep: Int?
        let cumTrainStepSec: Double?
        let cumGames: Int?

        enum CodingKeys: String, CodingKey {
            case elapsedSec = "elapsed_sec"
            case steps
            case selfPlayGames = "self_play_games"
            case positionsTrained = "positions_trained"
            case emittedGames = "emitted_games"
            case emittedPositions = "emitted_positions"
            case selfPlayDrawKeepFraction = "self_play_draw_keep_fraction"
            case selfPlayMaxPliesPerGame = "self_play_max_plies_per_game"
            case maxPliesDropped = "max_plies_dropped"
            case avgLen = "avg_len"
            case rollingAvgLen = "rolling_avg_len"
            case gameLenP50 = "game_len_p50"
            case gameLenP95 = "game_len_p95"
            case bufferCount = "buffer_count"
            case bufferCapacity = "buffer_capacity"
            case policyLoss = "policy_loss"
            case valueLoss = "value_loss"
            case policyEntropy = "policy_entropy"
            case policyIllegalMassPenalty = "policy_illegal_mass_penalty"
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
            case legalEntropy = "legal_entropy"
            case policyLossWin = "policy_loss_win"
            case policyLossLoss = "policy_loss_loss"
            case batchStats = "batch_stats"
            case valueMean = "value_mean"
            case valueAbsMean = "value_abs_mean"
            case valueProbWin = "value_prob_win"
            case valueProbDraw = "value_prob_draw"
            case valueProbLoss = "value_prob_loss"
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
            case ratioProducedRate = "ratio_produced_rate"
            case ratioConsumptionRate = "ratio_consumption_rate"
            case selfPlayMovesPerHour = "self_play_moves_per_hour"
            case trainingMovesPerHour = "training_moves_per_hour"
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
            case dropoutRate = "dropout_rate"
            case entropyRegularizationCoeff = "entropy_regularization_coeff"
            case drawPenalty = "draw_penalty"
            case policyLossWeight = "policy_loss_weight"
            case valueLossWeight = "value_loss_weight"
            case lrEffectiveBase = "lr_effective_base"
            case momentumEffective = "momentum_effective"
            case lrCycleActive = "lr_cycle_active"
            case momentumCycleActive = "momentum_cycle_active"
            case buildNumber = "build_number"
            case trainerID = "trainer_id"
            case championID = "champion_id"
            case policyLogitMean = "policy_logit_mean"
            case valueLogitMean = "value_logit_mean"
            case lrCyclePeak = "lr_cycle_peak"
            case lrCycleTrough = "lr_cycle_trough"
            case lrCycleDecayHorizonSteps = "lr_cycle_decay_horizon_steps"
            case momentumFollowsLRCycle = "momentum_follows_lr_cycle"
            case policyLabelSmoothingEpsilon = "policy_label_smoothing_epsilon"
            case policyLabelSmoothingMode = "policy_label_smoothing_mode"
            case policyLabelSmoothingPerMove = "policy_label_smoothing_per_move"
            case policyLabelSmoothingPerMoveCap = "policy_label_smoothing_per_move_cap"
            case policyTailPrecision = "policy_tail_precision"
            case cumTrainerStep = "cum_trainer_step"
            case cumTrainStepSec = "cum_train_step_sec"
            case cumGames = "cum_games"
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

// MARK: - Stats lines for non-self-play CLI runs

extension CliTrainingRecorder.StatsLine {
    /// Build a stats line for a CLI training run that has **no self-play loop
    /// and no arena** — corpus replay (`--replay-corpus`) and train-vs-UCI
    /// (`--train-vs-uci`).
    ///
    /// Those paths previously wrote no `results.json` at all: `--output` was
    /// parsed globally but only ever read by the self-play controller, so
    /// `--replay-corpus … --output x.json` silently produced nothing. That made
    /// the one thing this recorder's `dropoutRate` field exists for — comparing
    /// the arms of a dropout sweep — impossible to do on the CLI paths.
    ///
    /// The zeros and empty strings below are the honest value for "this run type
    /// does not have that", not placeholders standing in for data we failed to
    /// collect: a corpus run plays no self-play games, runs no replay-ratio
    /// controller, tallies no game outcomes, and has no champion lineage. Every
    /// field such a run genuinely does have — step counters, the training step's
    /// losses and diagnostics, and the hyperparameters it trained under — is a
    /// required argument here, so none of them can be forgotten at a call site.
    ///
    /// Declared in an extension deliberately: an initializer in the struct body
    /// would suppress the memberwise init the self-play path relies on.
    init(
        elapsedSec: Double,
        steps: Int,
        positionsFed: Int,
        bufferCount: Int,
        bufferCapacity: Int,
        policyLoss: Double?,
        valueLoss: Double?,
        policyEntropy: Double?,
        policyIllegalMassPenalty: Double?,
        gradGlobalNorm: Double?,
        playedMoveProb: Double?,
        valueMean: Double?,
        valueAbsMean: Double?,
        valueProbWin: Double?,
        valueProbDraw: Double?,
        valueProbLoss: Double?,
        policyLogitMean: Double?,
        valueLogitMean: Double?,
        batchSize: Int,
        learningRate: Double,
        gradClipMaxNorm: Double,
        weightDecayC: Double,
        dropoutRate: Double,
        entropyRegularizationCoeff: Double,
        drawPenalty: Double,
        policyLossWeight: Double,
        valueLossWeight: Double,
        lrEffectiveBase: Double,
        momentumEffective: Double,
        buildNumber: Int,
        trainerID: String,
        /// Positions PRODUCED — every ply played, including games later
        /// dropped. Pass the same value as `positionsFed` on a path with no
        /// drop concept. Pass nil where the count is genuinely UNMEASURED, in
        /// which case `positions_trained` falls back to the fed count and is
        /// therefore a lower bound rather than the self-play convention.
        positionsProduced: Int? = nil,
        /// Games this run actually played, if it plays any. Corpus replay
        /// passes nil (it plays none); train-vs-UCI passes its completed-game
        /// count.
        gamesPlayed: Int? = nil,
        /// Games dropped for hitting the per-game ply cap, and the cap itself.
        /// Both nil on a path with no cap.
        pliesCapDropped: Int? = nil,
        maxPliesPerGame: Int? = nil,
        /// Replay-ratio TARGET, where one governs the run. Corpus replay has a
        /// real one (it sets `perStepFeed = batchSize / target`); train-vs-UCI
        /// never reads it, so passing it there would be a fresh false claim.
        replayRatioTarget: Double? = nil,
        /// The run's lineage totals at this tick (`LineageTracker.totals`).
        lineageTotals: CliTrainingRecorder.LineageTotals
    ) {
        self.init(
            elapsedSec: elapsedSec,
            steps: steps,
            selfPlayGames: gamesPlayed ?? 0,
            positionsTrained: positionsProduced ?? positionsFed,
            emittedGames: 0,
            emittedPositions: positionsFed,
            selfPlayDrawKeepFraction: 0,
            selfPlayMaxPliesPerGame: maxPliesPerGame ?? 0,
            maxPliesDropped: pliesCapDropped ?? 0,
            avgLen: 0,
            rollingAvgLen: 0,
            gameLenP50: nil,
            gameLenP95: nil,
            bufferCount: bufferCount,
            bufferCapacity: bufferCapacity,
            policyLoss: policyLoss,
            valueLoss: valueLoss,
            policyEntropy: policyEntropy,
            policyIllegalMassPenalty: policyIllegalMassPenalty,
            gradGlobalNorm: gradGlobalNorm,
            policyHeadWeightNorm: nil,
            policyLogitAbsMax: nil,
            playedMoveProb: playedMoveProb,
            playedMoveProbPosAdv: nil,
            playedMoveProbPosAdvSkipped: 0,
            playedMoveProbNegAdv: nil,
            playedMoveProbNegAdvSkipped: 0,
            playedMoveCondWindowSize: 0,
            legalMass: nil,
            top1LegalFraction: nil,
            legalEntropy: nil,
            policyLossWin: nil,
            policyLossLoss: nil,
            batchStats: nil,
            valueMean: valueMean,
            valueAbsMean: valueAbsMean,
            valueProbWin: valueProbWin,
            valueProbDraw: valueProbDraw,
            valueProbLoss: valueProbLoss,
            advMean: nil,
            advStd: nil,
            advMin: nil,
            advMax: nil,
            advFracPositive: nil,
            advFracSmall: nil,
            advP05: nil,
            advP50: nil,
            advP95: nil,
            spStartTau: nil,
            spFloorTau: nil,
            spDecayPerPly: nil,
            arStartTau: nil,
            arFloorTau: nil,
            arDecayPerPly: nil,
            diversityUniqueGames: nil,
            diversityGamesInWindow: nil,
            diversityUniquePercent: nil,
            diversityAvgDivergencePly: nil,
            ratioTarget: replayRatioTarget,
            ratioCurrent: 0,
            ratioProductionRate: 0,
            ratioProducedRate: 0,
            ratioConsumptionRate: 0,
            selfPlayMovesPerHour: 0,
            trainingMovesPerHour: 0,
            ratioAutoAdjust: false,
            ratioComputedDelayMs: 0,
            whiteCheckmates: 0,
            blackCheckmates: 0,
            stalemates: 0,
            fiftyMoveDraws: 0,
            threefoldRepetitionDraws: 0,
            insufficientMaterialDraws: 0,
            batchSize: batchSize,
            learningRate: learningRate,
            promoteThreshold: nil,
            arenaGames: nil,
            workerCount: 0,
            gradClipMaxNorm: gradClipMaxNorm,
            weightDecayC: weightDecayC,
            dropoutRate: dropoutRate,
            entropyRegularizationCoeff: entropyRegularizationCoeff,
            drawPenalty: drawPenalty,
            policyLossWeight: policyLossWeight,
            valueLossWeight: valueLossWeight,
            lrEffectiveBase: lrEffectiveBase,
            momentumEffective: momentumEffective,
            lrCycleActive: false,
            momentumCycleActive: false,
            buildNumber: buildNumber,
            trainerID: trainerID,
            championID: nil,
            policyLogitMean: policyLogitMean,
            valueLogitMean: valueLogitMean,
            lrCyclePeak: nil,
            lrCycleTrough: nil,
            lrCycleDecayHorizonSteps: nil,
            momentumFollowsLRCycle: nil,
            cumTrainerStep: lineageTotals.cumTrainerStep,
            cumTrainStepSec: lineageTotals.cumTrainStepSec,
            cumGames: lineageTotals.cumGames
        )
    }

    /// The offline-runner stats line, with every trainer-level hyperparameter
    /// and the LR/momentum cycle telemetry taken from the same
    /// `TrainerHyperparameters` the runner's trainer was configured from —
    /// the fields the self-play path fills from its own trainer. `cycleValues`
    /// is the cycle evaluated at this line's completed-step count (as the
    /// trainer's SGD feed evaluates it), so `lr_effective_base`,
    /// `momentum_effective`, the `*_cycle_active` flags and the envelope
    /// bounds describe what the step actually used.
    init(
        elapsedSec: Double,
        steps: Int,
        positionsFed: Int,
        bufferCount: Int,
        bufferCapacity: Int,
        policyLoss: Double?,
        valueLoss: Double?,
        policyEntropy: Double?,
        policyIllegalMassPenalty: Double?,
        gradGlobalNorm: Double?,
        playedMoveProb: Double?,
        valueMean: Double?,
        valueAbsMean: Double?,
        valueProbWin: Double?,
        valueProbDraw: Double?,
        valueProbLoss: Double?,
        policyLogitMean: Double?,
        valueLogitMean: Double?,
        batchSize: Int,
        trainerHyperparameters hyperparameters: TrainerHyperparameters,
        cycleValues: LRMomentumCycle.Values,
        buildNumber: Int,
        trainerID: String,
        positionsProduced: Int?,
        gamesPlayed: Int?,
        pliesCapDropped: Int?,
        maxPliesPerGame: Int?,
        replayRatioTarget: Double?,
        lineageTotals: CliTrainingRecorder.LineageTotals
    ) {
        self.init(
            elapsedSec: elapsedSec,
            steps: steps,
            positionsFed: positionsFed,
            bufferCount: bufferCount,
            bufferCapacity: bufferCapacity,
            policyLoss: policyLoss,
            valueLoss: valueLoss,
            policyEntropy: policyEntropy,
            policyIllegalMassPenalty: policyIllegalMassPenalty,
            gradGlobalNorm: gradGlobalNorm,
            playedMoveProb: playedMoveProb,
            valueMean: valueMean,
            valueAbsMean: valueAbsMean,
            valueProbWin: valueProbWin,
            valueProbDraw: valueProbDraw,
            valueProbLoss: valueProbLoss,
            policyLogitMean: policyLogitMean,
            valueLogitMean: valueLogitMean,
            batchSize: batchSize,
            learningRate: Double(hyperparameters.learningRate),
            gradClipMaxNorm: Double(hyperparameters.gradClipMaxNorm),
            weightDecayC: Double(hyperparameters.weightDecayC),
            dropoutRate: Double(hyperparameters.dropoutRate),
            entropyRegularizationCoeff: Double(hyperparameters.entropyRegularizationCoeff),
            drawPenalty: Double(hyperparameters.drawPenalty),
            policyLossWeight: Double(hyperparameters.policyLossWeight),
            valueLossWeight: Double(hyperparameters.valueLossWeight),
            lrEffectiveBase: cycleValues.learningRate ?? Double(hyperparameters.learningRate),
            momentumEffective: cycleValues.momentum ?? Double(hyperparameters.momentumCoeff),
            buildNumber: buildNumber,
            trainerID: trainerID,
            positionsProduced: positionsProduced,
            gamesPlayed: gamesPlayed,
            pliesCapDropped: pliesCapDropped,
            maxPliesPerGame: maxPliesPerGame,
            replayRatioTarget: replayRatioTarget,
            lineageTotals: lineageTotals
        )
        lrCycleActive = cycleValues.learningRate != nil
        momentumCycleActive = cycleValues.momentum != nil
        lrCyclePeak = cycleValues.lrPeak
        lrCycleTrough = cycleValues.lrTrough
        lrCycleDecayHorizonSteps = hyperparameters.lrMomentumCycle.envelope.decayHorizonSteps
        momentumFollowsLRCycle = hyperparameters.lrMomentumCycle.envelope.momentumFollowsLRCycle
        policyLabelSmoothingEpsilon = Double(hyperparameters.policyLabelSmoothingEpsilon)
        policyLabelSmoothingMode = hyperparameters.policyLabelSmoothingMode.logToken
        policyLabelSmoothingPerMove = Double(hyperparameters.policyLabelSmoothingPerMove)
        policyLabelSmoothingPerMoveCap = Double(hyperparameters.policyLabelSmoothingPerMoveCap)
        policyTailPrecision = ChessNetwork.PolicyTailPrecision.process.rawValue
    }
}

/// Where a CLI run writes its `results.json` (`--output`), checked before the
/// run starts so a bad destination fails in seconds instead of at the end of a
/// long run, and an earlier run's results are never silently replaced.
///
/// The rules: the folder exists, is a folder and is writable; the name fits a
/// staged write; a folder, link or special file at the path is always refused;
/// an existing regular file is refused unless `--overwrite-output` was passed,
/// and then only that very file may be replaced (`replacing`). Shared by every
/// `--output` the app accepts (GUI `--train`, `--replay-corpus`,
/// `--train-vs-uci`).
struct CliResultsOutput: Sendable, Equatable {
    let url: URL
    /// The existing regular file `--overwrite-output` authorized replacing,
    /// as found by the pre-flight; nil when nothing was at the path.
    let replacing: FileSafety.FileIdentity?

    static func preflight(url: URL, overwriteAuthorized: Bool) throws -> CliResultsOutput {
        try FileSafety.requireStageableDestination(url)
        let folder = url.deletingLastPathComponent()
        var isFolder: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isFolder) else {
            throw CliResultsOutputError.folderUnusable(path: folder.path, reason: "does not exist")
        }
        guard isFolder.boolValue else {
            throw CliResultsOutputError.folderUnusable(path: folder.path, reason: "is not a folder")
        }
        guard FileManager.default.isWritableFile(atPath: folder.path) else {
            throw CliResultsOutputError.folderUnusable(path: folder.path, reason: "is not writable")
        }
        guard let existing = try FileSafety.existingItem(at: url) else {
            return CliResultsOutput(url: url, replacing: nil)
        }
        guard existing.kind == .regularFile else {
            throw FileSafetyError.notARegularFile(path: url.path, kind: existing.kind)
        }
        guard overwriteAuthorized else {
            throw CliResultsOutputError.alreadyExists(path: url.path)
        }
        return CliResultsOutput(url: url, replacing: existing.identity)
    }
}

/// Refusals from `CliResultsOutput.preflight`.
enum CliResultsOutputError: LocalizedError, Equatable {
    case folderUnusable(path: String, reason: String)
    case alreadyExists(path: String)

    var errorDescription: String? {
        switch self {
        case let .folderUnusable(path, reason):
            return "--output folder \(path) \(reason)"
        case let .alreadyExists(path):
            return "--output \(path) already exists; refusing to replace it (pass --overwrite-output to replace it, or choose a new name)"
        }
    }
}
