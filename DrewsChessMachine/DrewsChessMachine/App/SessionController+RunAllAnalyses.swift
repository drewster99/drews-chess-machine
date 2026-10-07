import AppKit
import Foundation

/// `SessionController`'s combined-analyzer hook — wired to the
/// `Run All Analyses…` Debug menu item. Runs every analyzer the
/// session has prerequisites for, in sequence, writes each to disk
/// under `CheckpointPaths.analysesDir` (`AnalysisJSONExport`), logs a
/// per-analyzer block to the session log, and surfaces a single final
/// NSAlert summarizing the results.
///
/// The champion and the trainer are each exported once, before any
/// analysis runs (`analysisSnapshot(of:initReferences:)`, identity checked
/// across the export); every analysis of a network — the numerics audit's
/// masters and velocity, and the replay analyzer's entropy probe, which runs
/// on an inference network carrying the trainer's snapshot — reads that one
/// cut, so the files describe the same weights at the same step. Both
/// snapshots share one init-reference cache, so a champion and trainer of
/// one architecture and seed cost one reference build.
///
/// Analyses run, in order:
///   1. Replay buffer  (skipped if no `replayBuffer` is loaded)
///   2. Value head     (against champion; skipped if no network)
///   3. Network weights (against champion; skipped if no network)
///   4. Network weights (against trainer; skipped if no trainer)
///   5. Numerics audit (champion, then trainer)
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
        // Which networks are analyzed is decided at the press; each is
        // exported, identity checked across the export, before any analysis
        // runs (`analysisSnapshot(of:initReferences:)`), so every file — and
        // the replay step's trainer entropy probe — describes those captured
        // weights.
        let analyzesChampion = championRef != nil
        let analyzesTrainer = trainer != nil
        // One init-reference cache for the request: the champion and trainer
        // of one architecture and seed share one build.
        let initReferences = AnalysisInitReferenceCache()
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

        if bufferRef == nil && !analyzesChampion && !analyzesTrainer {
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

            // One capture of each network, taken first and shared by every
            // analysis below.
            var championCapture: Swift.Result<AnalysisCapture, Error>?
            if analyzesChampion {
                do {
                    championCapture = .success(try await self.analysisSnapshot(
                        of: .champion, initReferences: initReferences))
                } catch {
                    SessionLogger.shared.log("[ANALYSES] champion snapshot failed: \(error)")
                    championCapture = .failure(error)
                }
            }
            var trainerCapture: Swift.Result<AnalysisCapture, Error>?
            if analyzesTrainer {
                do {
                    trainerCapture = .success(try await self.analysisSnapshot(
                        of: .trainer, initReferences: initReferences))
                } catch {
                    SessionLogger.shared.log("[ANALYSES] trainer snapshot failed: \(error)")
                    trainerCapture = .failure(error)
                }
            }

            // 1. Replay buffer, after both captures: its entropy probe is the
            //    trainer capture's weights, so it describes the same step as
            //    the trainer's other files. The champion is what generates
            //    self-play, so the buffer is "champion's" data and its label
            //    reflects that. The probe runs on the trainer because the
            //    champion's policy is frozen between promotions, so probing it
            //    gives bit-identical entropy stats across snapshots and hides
            //    the "is illegal mass falling?" signal that is the whole point
            //    of the probe.
            if let buf = bufferRef {
                let modelLabel = "champion:\(championRef?.identifier?.description ?? "<no-id>")"
                SessionLogger.shared.log("[ANALYSIS] replay-buffer entropy probe positions: \(entropySample.description)")
                let entropyProbe: (network: ChessMPSNetwork, label: String)?
                switch trainerCapture {
                case .success(let capture)?:
                    entropyProbe = await Self.buildEntropyProbe(from: capture.snapshot)
                case .failure?:
                    SessionLogger.shared.log("[ANALYSIS] Entropy probe: the trainer snapshot failed — replay analyzer will fall back to champion for the entropy probe.")
                    entropyProbe = nil
                case nil:
                    entropyProbe = nil
                }
                let step = await Self.runReplayBufferStep(
                    buffer: buf, champion: championRef, modelLabel: modelLabel,
                    entropyProbe: entropyProbe, sampleRandom: entropySample.random, metadata: exportMetadata)
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            } else {
                summaryLines.append("• Replay buffer:           SKIPPED — no buffer loaded")
            }

            // 2. Value head (champion).
            switch championCapture {
            case .success(let capture)?:
                let step = await Self.runValueHeadStep(
                    snapshot: capture.snapshot, modelLabel: capture.snapshot.modelLabel, metadata: exportMetadata)
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            case .failure(let error)?:
                summaryLines.append("• Value head (champion):    FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Value head (champion):    SKIPPED — no champion loaded")
            }

            // 3. Network weights (champion).
            switch championCapture {
            case .success(let capture)?:
                let step = await Self.runNetworkWeightsStep(
                    snapshot: capture.snapshot, modelLabel: capture.snapshot.modelLabel, tag: "Champion", metadata: exportMetadata)
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            case .failure(let error)?:
                summaryLines.append("• Network weights (champion): FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Network weights (champion): SKIPPED — no champion loaded")
            }

            // 4. Network weights (trainer).
            switch trainerCapture {
            case .success(let capture)?:
                let step = await Self.runNetworkWeightsStep(
                    snapshot: capture.snapshot, modelLabel: capture.snapshot.modelLabel, tag: "Trainer", metadata: exportMetadata)
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            case .failure(let error)?:
                summaryLines.append("• Network weights (trainer):  FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Network weights (trainer):  SKIPPED — no trainer initialized")
            }

            // 5. Numerics audit (champion, then trainer).
            switch championCapture {
            case .success(let capture)?:
                let outcome = await Self.runNumericsAuditStep(capture: capture, metadata: exportMetadata, tag: "RunAll Champion")
                summaryLines.append(Self.numericsSummaryLine(outcome, tag: "champion"))
                if firstSuccessURL == nil, case .saved(let url) = outcome { firstSuccessURL = url }
            case .failure(let error)?:
                summaryLines.append("• Numerics audit (champion): FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Numerics audit (champion): SKIPPED — no champion loaded")
            }
            switch trainerCapture {
            case .success(let capture)?:
                let outcome = await Self.runNumericsAuditStep(capture: capture, metadata: exportMetadata, tag: "RunAll Trainer")
                summaryLines.append(Self.numericsSummaryLine(outcome, tag: "trainer"))
                if firstSuccessURL == nil, case .saved(let url) = outcome { firstSuccessURL = url }
            case .failure(let error)?:
                summaryLines.append("• Numerics audit (trainer):  FAILED — \(error.localizedDescription)")
            case nil:
                summaryLines.append("• Numerics audit (trainer):  SKIPPED — no trainer initialized")
            }

            SessionLogger.shared.log("[ANALYSES] === Run All Analyses summary ===")
            for line in summaryLines {
                SessionLogger.shared.log("[ANALYSES] \(line)")
            }
            let revealURL = firstSuccessURL
            let message = "Results:\n\n" + summaryLines.joined(separator: "\n")
                + (revealURL == nil ? "" : "\n\nClick Reveal in Finder to open the Analyses folder.")
            await Self.presentRunAllAlert(
                title: "Run All Analyses Complete",
                message: message,
                revealURL: revealURL
            )
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

    /// Runs the replay-buffer analyzer (`analyzeReplayBuffer`; the entropy
    /// probe on `entropyProbe`, else the champion), writes the JSON and logs
    /// an `[ANALYSIS]` block. A failed probe fails this step.
    nonisolated private static func runReplayBufferStep(
        buffer: ReplayBuffer,
        champion: ChessMPSNetwork?,
        modelLabel: String,
        entropyProbe: (network: ChessMPSNetwork, label: String)?,
        sampleRandom: DCMRandom,
        metadata: AnalysisExportMetadata
    ) async -> AnalysisStepResult {
        var result: ReplayBufferAnalyzer.Result
        do {
            result = try await analyzeReplayBuffer(
                buffer: buffer, champion: champion, modelLabel: modelLabel,
                entropyProbe: entropyProbe, sampleRandom: sampleRandom)
        } catch {
            SessionLogger.shared.log("[ANALYSIS] (RunAll) failed: \(error)")
            return AnalysisStepResult(
                summaryLine: "• Replay buffer:           FAILED — \(error.localizedDescription)",
                firstSuccessURL: nil
            )
        }

        result.exportMetadata = metadata
        let outcome = await AnalysisJSONExport.summarizeAndPublishOffPool(
            result, modelLabel: modelLabel, directory: CheckpointPaths.analysesDir)
        AnalysisJSONExport.logSummary(outcome.summary, family: .replayBuffer, context: "RunAll")
        switch outcome.written {
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
            result = try await ValueHeadAnalyzer.runOffPool(snapshot: snapshot, modelLabel: modelLabel)
        } catch {
            SessionLogger.shared.log("[VALHEAD] (RunAll) failed: \(error)")
            return AnalysisStepResult(
                summaryLine: "• Value head (champion):    FAILED — \(error.localizedDescription)",
                firstSuccessURL: nil
            )
        }

        result.exportMetadata = metadata.describing(snapshot)
        let outcome = await AnalysisJSONExport.summarizeAndPublishOffPool(
            result, modelLabel: modelLabel, directory: CheckpointPaths.analysesDir)
        AnalysisJSONExport.logSummary(outcome.summary, family: .valueHead, context: "RunAll")
        switch outcome.written {
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
        let outcome = await AnalysisJSONExport.summarizeAndPublishOffPool(
            result, modelLabel: modelLabel, directory: CheckpointPaths.analysesDir)
        AnalysisJSONExport.logSummary(outcome.summary, family: .networkWeights, context: "RunAll \(tag)")
        switch outcome.written {
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
