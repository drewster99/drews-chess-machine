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
        // Snapshot training-progress context once, on the main actor,
        // before the detached work begins. Every file written in this
        // pass shares this snapshot, so step/elapsed values line up
        // across the replay / value-head / weight JSONs produced
        // together.
        let exportMetadata = currentAnalysisExportMetadata()

        if bufferRef == nil && championRef == nil && trainerRef == nil {
            Self.presentRunAllAlert(
                title: "Run All Analyses",
                message: "Nothing to analyze. No replay buffer, no network, and no trainer is loaded.",
                revealURL: nil
            )
            return
        }

        Task.detached(priority: .utility) {
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
                let step = await Self.runReplayBufferStep(
                    buffer: buf,
                    network: championRef,
                    modelLabel: modelLabel,
                    entropyProbe: entropyProbe,
                    metadata: exportMetadata
                )
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            } else {
                summaryLines.append("• Replay buffer:           SKIPPED — no buffer loaded")
            }

            // 2. Value head (champion).
            if let net = championRef {
                let modelLabel = "champion:\(net.identifier?.description ?? "<no-id>")"
                let step = await Self.runValueHeadStep(
                    network: net,
                    modelLabel: modelLabel,
                    metadata: exportMetadata
                )
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            } else {
                summaryLines.append("• Value head (champion):    SKIPPED — no champion loaded")
            }

            // 3. Network weights (champion).
            if let net = championRef {
                let modelLabel = "champion:\(net.identifier?.description ?? "<no-id>")"
                let step = await Self.runNetworkWeightsStep(
                    networkInner: net.network,
                    modelLabel: modelLabel,
                    tag: "Champion",
                    metadata: exportMetadata
                )
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            } else {
                summaryLines.append("• Network weights (champion): SKIPPED — no champion loaded")
            }

            // 4. Network weights (trainer).
            if let trainer = trainerRef {
                let modelLabel = "trainer:\(trainer.identifier?.description ?? "<no-id>")"
                let step = await Self.runNetworkWeightsStep(
                    networkInner: trainer.network,
                    modelLabel: modelLabel,
                    tag: "Trainer",
                    metadata: exportMetadata
                )
                summaryLines.append(step.summaryLine)
                if firstSuccessURL == nil { firstSuccessURL = step.firstSuccessURL }
            } else {
                summaryLines.append("• Network weights (trainer):  SKIPPED — no trainer initialized")
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

    /// Snapshot the live training-progress context into an
    /// `AnalysisExportMetadata` for stamping onto analysis exports.
    /// Reads the trainer / self-play stats boxes, replay buffer, model
    /// identifiers, and the static architecture constants, so it must run
    /// on the main actor (the class's isolation) where that state is
    /// reachable. The `selfPlay` and `training` sub-blocks are present
    /// only when their backing context exists, so an export taken before
    /// any run simply omits them rather than reporting misleading zeros.
    ///
    /// Used by both `Run All Analyses` (one snapshot shared across the
    /// pass) and the three single-analysis Debug hooks.
    func currentAnalysisExportMetadata() -> AnalysisExportMetadata {
        let archHashHex = String(format: "0x%08x", ModelCheckpointFile.currentArchHash)
        let notes = ChessNetwork.architectureNotes
        let architecture = AnalysisExportMetadata.Architecture(
            archHash: archHashHex,
            architectureVersion: ChessNetwork.architectureVersion,
            parameterCount: ChessNetwork.parameterCount,
            numBlocks: ChessNetwork.numBlocks,
            channels: ChessNetwork.channels,
            convKernelSize: ChessNetwork.towerConvKernelSize,
            inputPlanes: ChessNetwork.inputPlanes,
            boardSize: ChessNetwork.boardSize,
            policyChannels: ChessNetwork.policyChannels,
            policySize: ChessNetwork.policySize,
            seReductionRatio: ChessNetwork.seReductionRatio,
            valueHead: AnalysisExportMetadata.Architecture.ValueHead(
                classes: ChessNetwork.valueHeadClasses,
                convChannels: ChessNetwork.valueHeadConvChannels,
                hiddenUnits: ChessNetwork.valueHeadHiddenUnits
            ),
            summary: ChessNetwork.architectureSummary,
            notes: notes.isEmpty ? nil : notes
        )

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
            training = AnalysisExportMetadata.Training(
                trainingSteps: snap.stats.steps,
                cumulativeTrainingSeconds: checkpoint?.cumulativeActiveTrainingSec,
                batchSize: TrainingParameters.shared.trainingBatchSize,
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
            architecture: architecture,
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
            let probe = try ChessMPSNetwork(.randomWeights)
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
        metadata: AnalysisExportMetadata
    ) async -> AnalysisStepResult {
        var result: ReplayBufferAnalyzer.Result
        do {
            if let probe = entropyProbe {
                result = try await ReplayBufferAnalyzer.runWithPolicyEntropy(
                    buffer: buffer,
                    network: probe.network,
                    modelLabel: modelLabel,
                    entropyModelLabel: probe.label
                )
            } else if let net = network {
                result = try await ReplayBufferAnalyzer.runWithPolicyEntropy(
                    buffer: buffer,
                    network: net,
                    modelLabel: modelLabel
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
        network: ChessMPSNetwork,
        modelLabel: String,
        metadata: AnalysisExportMetadata
    ) async -> AnalysisStepResult {
        var result: ValueHeadAnalyzer.Result
        do {
            result = try await ValueHeadAnalyzer.run(
                network: network,
                modelLabel: modelLabel
            )
        } catch {
            SessionLogger.shared.log("[VALHEAD] (RunAll) failed: \(error)")
            return AnalysisStepResult(
                summaryLine: "• Value head (champion):    FAILED — \(error.localizedDescription)",
                firstSuccessURL: nil
            )
        }

        result.exportMetadata = metadata
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

    /// Runs the network weight analyzer against `networkInner`. `tag`
    /// is "Champion" / "Trainer" — used in the summary line and the
    /// log block header so the two paths are distinguishable.
    nonisolated private static func runNetworkWeightsStep(
        networkInner: ChessNetwork,
        modelLabel: String,
        tag: String,
        metadata: AnalysisExportMetadata
    ) async -> AnalysisStepResult {
        var result: NetworkWeightAnalyzer.Result
        do {
            result = try await NetworkWeightAnalyzer.run(
                network: networkInner,
                modelLabel: modelLabel
            )
        } catch {
            SessionLogger.shared.log("[NETW] (RunAll \(tag)) failed: \(error)")
            return AnalysisStepResult(
                summaryLine: "• Network weights (\(tag.lowercased())):  FAILED — \(error.localizedDescription)",
                firstSuccessURL: nil
            )
        }

        result.exportMetadata = metadata
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
