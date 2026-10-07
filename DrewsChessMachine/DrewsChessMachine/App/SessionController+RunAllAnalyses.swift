import AppKit
import Foundation

/// `SessionController`'s combined-analyzer hook — wired to the
/// `Run All Analyses…` Debug menu item. Runs every analyzer the
/// session has prerequisites for, in sequence, writes each to disk
/// under `CheckpointPaths.analysesDir`, logs a per-analyzer block
/// to the session log, and surfaces a single final NSAlert
/// summarizing the results.
///
/// Analyses run, in order:
///   1. Replay buffer  (skipped if no `replayBuffer` is loaded)
///   2. Value head     (against champion; skipped if no network)
///   3. Network weights (against champion; skipped if no network)
///   4. Network weights (against trainer; skipped if no trainer)
///   5. Numerics audit (champion, then trainer)
///
/// The champion and the trainer are each exported once
/// (`AnalyzedNetworkSnapshot`); every analysis of a network reads that one
/// snapshot, so the files describe the same weights at the same step.
///
/// Each sub-analysis is independent: a failure in one doesn't stop
/// the others. Successes log their JSON path; failures log the
/// error. The final alert summarizes pass/fail per analysis.
extension SessionController {

    /// Result tracking for one sub-analysis inside `Run All`. The
    /// summary line is what appears in the final NSAlert (one per
    /// analysis); the optional URL is the JSON file the analyzer
    /// wrote, used to offer Reveal-in-Finder on the first success.
    private struct AnalysisStepResult: Sendable {
        let summaryLine: String
        let firstSuccessURL: URL?
    }

    /// Entry point invoked by the Debug menu item.
    func runAllAnalysesToFile() {
        SessionLogger.shared.log("[BUTTON] Run All Analyses")

        let bufferRef = replayBuffer
        let championRef = network
        let trainerRef = trainer
        let championTarget = championAnalysisTarget()
        let trainerTarget = trainerAnalysisTarget()
        // Masters are read for the numerics audit only while training is
        // stopped; snapshot that on the main actor.
        let trainingIsRunning = realTraining
        // Snapshot training-progress context once, on the main actor,
        // before the detached work begins. Every file written in this
        // pass shares this snapshot, so step/elapsed values line up
        // across the replay / value-head / weight JSONs produced
        // together.
        let exportMetadata = currentAnalysisExportMetadata()
        let entropySample = ReplayBufferAnalyzer.entropyProbeRandom(
            runSeed: runRandomSeed,
            trainerStep: trainingBox?.snapshot().stats.steps
        )

        if bufferRef == nil && championRef == nil && trainerRef == nil {
            Self.presentRunAllAlert(
                title: "Run All Analyses",
                message: "Nothing to analyze. No replay buffer, no network, and no trainer is loaded.",
                revealURL: nil
            )
            return
        }
        guard beginAnalysis("All Analyses") else { return }

        Task.detached(priority: .utility) {
            defer { Task { @MainActor in self.endAnalysis() } }
            var summaryLines: [String] = []
            var firstSuccessURL: URL?

            // 1. Replay buffer (champion is what generates self-play,
            //    so the buffer is "champion's" data and its label
            //    reflects that). The per-bucket policy-entropy probe,
            //    however, is most useful against the *trainer* — the
            //    champion's policy is frozen between promotions, so
            //    probing it produces bit-identical entropy stats
            //    across snapshots and obscures the "is illegal mass
            //    falling?" signal that's the whole point of the
            //    probe. When a trainer is available, snapshot its
            //    current weights into a fresh inference-mode network
            //    and probe that.
            if let buf = bufferRef {
                let modelLabel = "champion:\(championRef?.identifier?.description ?? "<no-id>")"
                let entropyProbe = await Self.buildTrainerEntropyProbeNetwork(trainer: trainerRef)
                SessionLogger.shared.log("[ANALYSIS] replay-buffer entropy probe positions: \(entropySample.description)")
                let step = await Self.runReplayBufferStep(
                    buffer: buf,
                    network: championRef,
                    modelLabel: modelLabel,
                    entropyProbe: entropyProbe,
                    sampleRandom: entropySample.random,
                    metadata: exportMetadata
                )
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            } else {
                summaryLines.append("• Replay buffer:           SKIPPED — no buffer loaded")
            }

            // One snapshot of each network, shared by its analyses below.
            var championSnapshot: Swift.Result<AnalyzedNetworkSnapshot, Error>?
            if let championTarget {
                do {
                    championSnapshot = .success(try await self.analysisSnapshot(of: championTarget))
                } catch {
                    SessionLogger.shared.log("[ANALYSES] champion snapshot failed: \(error)")
                    championSnapshot = .failure(error)
                }
            }
            var trainerSnapshot: Swift.Result<AnalyzedNetworkSnapshot, Error>?
            if let trainerTarget {
                do {
                    trainerSnapshot = .success(try await self.analysisSnapshot(of: trainerTarget))
                } catch {
                    SessionLogger.shared.log("[ANALYSES] trainer snapshot failed: \(error)")
                    trainerSnapshot = .failure(error)
                }
            }

            // 2. Value head (champion).
            switch championSnapshot {
            case .success(let snapshot)?:
                let step = await Self.runValueHeadStep(snapshot: snapshot, modelLabel: snapshot.modelLabel, metadata: exportMetadata)
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            case .failure(let error)?:
                summaryLines.append("• Value head (champion):    FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Value head (champion):    SKIPPED — no champion loaded")
            }

            // 3. Network weights (champion).
            switch championSnapshot {
            case .success(let snapshot)?:
                let step = await Self.runNetworkWeightsStep(snapshot: snapshot, modelLabel: snapshot.modelLabel, tag: "Champion", metadata: exportMetadata)
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            case .failure(let error)?:
                summaryLines.append("• Network weights (champion): FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Network weights (champion): SKIPPED — no champion loaded")
            }

            // 4. Network weights (trainer).
            switch trainerSnapshot {
            case .success(let snapshot)?:
                let step = await Self.runNetworkWeightsStep(snapshot: snapshot, modelLabel: snapshot.modelLabel, tag: "Trainer", metadata: exportMetadata)
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            case .failure(let error)?:
                summaryLines.append("• Network weights (trainer):  FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Network weights (trainer):  SKIPPED — no trainer initialized")
            }

            // 5. Numerics audit (champion, then trainer).
            switch championSnapshot {
            case .success(let snapshot)?:
                let outcome = await Self.runNumericsAuditStep(
                    snapshot: snapshot,
                    modelLabel: snapshot.modelLabel,
                    mastersSource: .unavailable(NumericsAudit.championMastersNote),
                    metadata: exportMetadata,
                    tag: "RunAll Champion"
                )
                summaryLines.append(Self.numericsSummaryLine(outcome, tag: "champion"))
                if firstSuccessURL == nil, case .saved(let url) = outcome { firstSuccessURL = url }
            case .failure(let error)?:
                summaryLines.append("• Numerics audit (champion): FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Numerics audit (champion): SKIPPED — no champion loaded")
            }
            switch trainerSnapshot {
            case .success(let snapshot)?:
                guard case .trainer(let trainer)? = trainerTarget?.source else {
                    preconditionFailure("a trainer snapshot was taken without a trainer target")
                }
                let outcome = await Self.runNumericsAuditStep(
                    snapshot: snapshot,
                    modelLabel: snapshot.modelLabel,
                    mastersSource: .trainer(trainer, trainingIsRunning: trainingIsRunning),
                    metadata: exportMetadata,
                    tag: "RunAll Trainer"
                )
                summaryLines.append(Self.numericsSummaryLine(outcome, tag: "trainer"))
                if firstSuccessURL == nil, case .saved(let url) = outcome { firstSuccessURL = url }
            case .failure(let error)?:
                summaryLines.append("• Numerics audit (trainer):  FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Numerics audit (trainer):  SKIPPED — no trainer initialized")
            }

            await MainActor.run {
                SessionLogger.shared.log("[ANALYSES] === Run All Analyses summary ===")
                for line in summaryLines {
                    SessionLogger.shared.log("[ANALYSES] \(line)")
                }
                Self.presentRunAllAlert(
                    title: "Run All Analyses Complete",
                    message: "Results:\n\n" + summaryLines.joined(separator: "\n")
                        + (firstSuccessURL == nil ? "" : "\n\nClick Reveal in Finder to open the Analyses folder."),
                    revealURL: firstSuccessURL
                )
            }
        }
    }

    // MARK: - Export metadata

    /// Snapshot the session's context into an `AnalysisExportMetadata` for
    /// stamping onto analysis exports: build, the session's champion and
    /// trainer IDs, self-play volume and training progress. Reads the stats
    /// boxes, replay buffer and model identifiers, so it runs on the main
    /// actor. The `selfPlay` and `training` sub-blocks are present only when
    /// their backing context exists. It describes no network's weights: an
    /// export of a network is stamped with its snapshot through
    /// `describing(_:)`.
    ///
    /// Used by both `Run All Analyses` (one snapshot shared across the
    /// pass) and the single-analysis Debug hooks.
    func currentAnalysisExportMetadata() -> AnalysisExportMetadata {
        let selfPlay: AnalysisExportMetadata.SelfPlay?
        if let snap = parallelWorkerStatsBox?.snapshot() {
            selfPlay = AnalysisExportMetadata.SelfPlay(
                totalGames: snap.selfPlayGames,
                totalMoves: snap.selfPlayPositions,
                emittedGames: snap.emittedGames,
                emittedMoves: snap.emittedPositions
            )
        } else {
            selfPlay = nil
        }

        let training: AnalysisExportMetadata.Training?
        if let snap = trainingBox?.snapshot() {
            let batchSize: Int?
            let batchSizeSetting: Int?
            switch trainingBatchSizeDisplay() {
            case .activeRun(let runBatchSize):
                batchSize = runBatchSize
                batchSizeSetting = nil
            case .setting(let setting):
                batchSize = nil
                batchSizeSetting = setting
            }
            training = AnalysisExportMetadata.Training(
                trainingSteps: snap.stats.steps,
                cumulativeTrainingSeconds: checkpoint?.cumulativeActiveTrainingSec,
                batchSize: batchSize,
                batchSizeSetting: batchSizeSetting,
                promoteThreshold: TrainingParameters.shared.arenaPromoteThreshold,
                replayBufferPlies: replayBuffer?.count
            )
        } else {
            training = nil
        }

        return AnalysisExportMetadata(
            schemaVersion: AnalysisExportMetadata.currentSchemaVersion,
            build: AnalysisExportMetadata.Build(
                buildNumber: BuildInfo.buildNumber,
                buildTimestamp: BuildInfo.buildTimestamp,
                gitHash: BuildInfo.gitHash,
                gitBranch: BuildInfo.gitBranch,
                gitIsDirty: BuildInfo.gitDirty
            ),
            model: AnalysisExportMetadata.Model(
                championModelID: network?.identifier?.description,
                trainerModelID: trainer?.identifier?.description
            ),
            analyzedWeights: nil,
            selfPlay: selfPlay,
            training: training
        )
    }

    // MARK: - Per-analysis runners

    /// Build a fresh inference-mode `ChessMPSNetwork`, load the
    /// trainer's current weights into it, and return it paired with a
    /// "trainer:<id>" label, for use as the replay analyzer's entropy
    /// probe. Returns `nil` if no trainer is available, or if the
    /// snapshot path fails (logged; falls through to "no entropy
    /// probe" in the caller — the analyzer's own fallback then picks
    /// the champion).
    ///
    /// The network is short-lived — allocated for one Run All
    /// Analyses pass and dropped when the closure exits. The build
    /// includes the `.randomWeights` BN warmup whose stats are then
    /// overwritten by `loadWeights`; that's wasted work but ~10s of
    /// ms, negligible for a manual menu action.
    nonisolated static func buildTrainerEntropyProbeNetwork(
        trainer: ChessTrainer?
    ) async -> (network: ChessMPSNetwork, label: String)? {
        guard let trainer else { return nil }
        do {
            let weights = try await trainer.network.exportWeights()
            let probe = try ChessMPSNetwork(.overwrittenByLoad, arch: trainer.arch)
            try await probe.loadWeights(weights)
            // Inherit the trainer's id so logs / JSON record exactly
            // whose weights are being probed — same pattern as
            // `fireCandidateProbeIfNeeded` uses for the candidate-test
            // probe inference network.
            probe.identifier = trainer.identifier
            let label = "trainer:\(trainer.identifier?.description ?? "<no-id>")"
            return (probe, label)
        } catch {
            SessionLogger.shared.log(
                "[ANALYSIS] Trainer entropy-probe snapshot failed: \(error.localizedDescription)"
                + " — replay analyzer will fall back to champion for the entropy probe."
            )
            return nil
        }
    }

    /// Runs the replay-buffer analyzer (with the optional live-network
    /// entropy probe when a network is available), writes the JSON,
    /// logs an `[ANALYSIS]` block. Returns the summary line + first
    /// success URL.
    ///
    /// `entropyProbe`, when non-nil, is the network the analyzer should
    /// probe for per-bucket policy entropy / illegal mass. **It is
    /// deliberately distinct from the champion** — the champion's
    /// policy is frozen between promotions, so probing it produces
    /// bit-identical entropy stats across snapshots and the "is
    /// illegal mass falling?" training-progress signal is invisible.
    /// Run All Analyses passes a fresh inference network freshly
    /// loaded with the trainer's current weights so the entropy
    /// section reflects the trainee's actual learning trajectory.
    nonisolated private static func runReplayBufferStep(
        buffer: ReplayBuffer,
        network: ChessMPSNetwork?,
        modelLabel: String,
        entropyProbe: (network: ChessMPSNetwork, label: String)? = nil,
        sampleRandom: DCMRandom,
        metadata: AnalysisExportMetadata
    ) async -> AnalysisStepResult {
        var result: ReplayBufferAnalyzer.Result
        do {
            if let probe = entropyProbe {
                result = try await ReplayBufferAnalyzer.runWithPolicyEntropy(
                    buffer: buffer,
                    network: probe.network,
                    modelLabel: modelLabel,
                    entropyModelLabel: probe.label,
                    sampleRandom: sampleRandom
                )
            } else if let net = network {
                result = try await ReplayBufferAnalyzer.runWithPolicyEntropy(
                    buffer: buffer,
                    network: net,
                    modelLabel: modelLabel,
                    sampleRandom: sampleRandom
                )
            } else {
                result = ReplayBufferAnalyzer.run(buffer: buffer, modelLabel: modelLabel)
            }
        } catch {
            SessionLogger.shared.log("[ANALYSIS] (RunAll) failed: \(error)")
            return AnalysisStepResult(
                summaryLine: "• Replay buffer:           FAILED — \(error.localizedDescription)",
                firstSuccessURL: nil
            )
        }

        result.exportMetadata = metadata
        let outcome = writeJSON(
            encodable: result,
            filenameStem: "replay_analysis",
            modelLabel: modelLabel
        )
        let summary = result.textSummary()
        await MainActor.run {
            SessionLogger.shared.log("[ANALYSIS] === Replay buffer analysis begin (RunAll) ===")
            for line in summary.split(separator: "\n", omittingEmptySubsequences: false) {
                SessionLogger.shared.log("[ANALYSIS] \(line)")
            }
            SessionLogger.shared.log("[ANALYSIS] === Replay buffer analysis end (RunAll) ===")
        }
        switch outcome {
        case .success(let url):
            return AnalysisStepResult(
                summaryLine: "• Replay buffer:           OK — \(url.lastPathComponent)",
                firstSuccessURL: url
            )
        case .failure(let err):
            return AnalysisStepResult(
                summaryLine: "• Replay buffer:           OK (text) / WRITE FAILED — \(err.localizedDescription)",
                firstSuccessURL: nil
            )
        }
    }

    /// Runs the value-head analyzer, writes the JSON, logs a
    /// `[VALHEAD]` block.
    nonisolated private static func runValueHeadStep(
        snapshot: AnalyzedNetworkSnapshot,
        modelLabel: String,
        metadata: AnalysisExportMetadata
    ) async -> AnalysisStepResult {
        var result: ValueHeadAnalyzer.Result
        do {
            result = try ValueHeadAnalyzer.run(snapshot: snapshot, modelLabel: modelLabel)
        } catch {
            SessionLogger.shared.log("[VALHEAD] (RunAll) failed: \(error)")
            return AnalysisStepResult(
                summaryLine: "• Value head (champion):    FAILED — \(error.localizedDescription)",
                firstSuccessURL: nil
            )
        }

        result.exportMetadata = metadata.describing(snapshot)
        let outcome = writeJSON(
            encodable: result,
            filenameStem: "valuehead_analysis",
            modelLabel: modelLabel
        )
        let summary = result.textSummary()
        await MainActor.run {
            SessionLogger.shared.log("[VALHEAD] === Value head analysis begin (RunAll) ===")
            for line in summary.split(separator: "\n", omittingEmptySubsequences: false) {
                SessionLogger.shared.log("[VALHEAD] \(line)")
            }
            SessionLogger.shared.log("[VALHEAD] === Value head analysis end (RunAll) ===")
        }
        switch outcome {
        case .success(let url):
            return AnalysisStepResult(
                summaryLine: "• Value head (champion):    OK — \(url.lastPathComponent)",
                firstSuccessURL: url
            )
        case .failure(let err):
            return AnalysisStepResult(
                summaryLine: "• Value head (champion):    OK (text) / WRITE FAILED — \(err.localizedDescription)",
                firstSuccessURL: nil
            )
        }
    }

    nonisolated private static func numericsSummaryLine(_ outcome: NumericsAuditStepOutcome, tag: String) -> String {
        switch outcome {
        case .saved(let url): return "• Numerics audit (\(tag)): OK — \(url.lastPathComponent)"
        case .writeFailed(let error): return "• Numerics audit (\(tag)): OK (text) / WRITE FAILED — \(error)"
        case .failed(let error): return "• Numerics audit (\(tag)): FAILED — \(error)"
        }
    }

    /// Runs the network weight analyzer on `snapshot`. `tag` is
    /// "Champion" / "Trainer" — used in the summary line and the log block
    /// header so the two paths are distinguishable.
    nonisolated private static func runNetworkWeightsStep(
        snapshot: AnalyzedNetworkSnapshot,
        modelLabel: String,
        tag: String,
        metadata: AnalysisExportMetadata
    ) async -> AnalysisStepResult {
        var result: NetworkWeightAnalyzer.Result
        do {
            result = try await NetworkWeightAnalyzer.runOffPool(snapshot: snapshot, modelLabel: modelLabel)
        } catch {
            SessionLogger.shared.log("[NETW] (RunAll \(tag)) failed: \(error)")
            return AnalysisStepResult(
                summaryLine: "• Network weights (\(tag.lowercased())):  FAILED — \(error.localizedDescription)",
                firstSuccessURL: nil
            )
        }

        result.exportMetadata = metadata.describing(snapshot)
        let outcome = writeJSON(
            encodable: result,
            filenameStem: "network_weights",
            modelLabel: modelLabel
        )
        let summary = result.textSummary()
        await MainActor.run {
            SessionLogger.shared.log("[NETW] === Network weight analysis begin (RunAll \(tag)) ===")
            for line in summary.split(separator: "\n", omittingEmptySubsequences: false) {
                SessionLogger.shared.log("[NETW] \(line)")
            }
            SessionLogger.shared.log("[NETW] === Network weight analysis end (RunAll \(tag)) ===")
        }
        switch outcome {
        case .success(let url):
            return AnalysisStepResult(
                summaryLine: "• Network weights (\(tag.lowercased())):  OK — \(url.lastPathComponent)",
                firstSuccessURL: url
            )
        case .failure(let err):
            return AnalysisStepResult(
                summaryLine: "• Network weights (\(tag.lowercased())):  OK (text) / WRITE FAILED — \(err.localizedDescription)",
                firstSuccessURL: nil
            )
        }
    }

    // MARK: - Shared JSON write

    /// Generic JSON writer used by every sub-step. Same encoder
    /// options the individual analyzer extensions use; filename stem
    /// distinguishes the artifact families inside the same
    /// `Analyses/` folder.
    nonisolated private static func writeJSON<T: Encodable>(
        encodable: T,
        filenameStem: String,
        modelLabel: String
    ) -> Result<URL, Error> {
        let fm = FileManager.default
        let dir = CheckpointPaths.analysesDir
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            return .failure(error)
        }

        let stamp = filenameTimestamp()
        let safeModel = modelLabel
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: ":", with: "_")
        let url = dir.appendingPathComponent("\(filenameStem)_\(stamp)_\(safeModel).json")

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        do {
            let data = try encoder.encode(encodable)
            try data.write(to: url, options: [.atomic])
            return .success(url)
        } catch {
            return .failure(error)
        }
    }

    nonisolated private static func filenameTimestamp() -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        df.locale = Locale(identifier: "en_US_POSIX")
        return df.string(from: Date())
    }

    // MARK: - Alert

    @MainActor
    private static func presentRunAllAlert(
        title: String,
        message: String,
        revealURL: URL? = nil
    ) {
        NonBlockingAlert.presentInformational(
            title: title,
            message: message,
            revealURL: revealURL
        )
    }
}
