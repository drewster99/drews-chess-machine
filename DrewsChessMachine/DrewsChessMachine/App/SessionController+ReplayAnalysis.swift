import AppKit
import Foundation

/// `SessionController`'s offline replay-buffer analyzer hook — wired to
/// the `Analyze Replay Buffer…` Debug menu item. Runs
/// `ReplayBufferAnalyzer` on the currently-loaded buffer, writes a
/// timestamped JSON file under `~/Library/Application Support/
/// DrewsChessMachine/Analyses/` (`AnalysisJSONExport`), logs an
/// `[ANALYSIS]` text-summary block to the session log, and surfaces an
/// NSAlert with a Reveal-in-Finder action so the JSON file is one click
/// away.
///
/// The buffer walks hold the buffer's lock for their duration; they, the
/// entropy probe's network build, the summary and the JSON write all run on
/// GCD (`ReplayBufferAnalyzer.runOffPool` / `entropyProbeSamplesOffPool`,
/// `InferenceNetworkFactory`, `AnalysisJSONExport`), so neither the main
/// actor nor a cooperative thread is held through them. Only the alert runs
/// on the main actor.
///
/// The per-bucket policy-entropy probe (analysis #7) runs on the trainer
/// when there is one: the champion's policy is frozen between promotions,
/// so probing it gives bit-identical entropy stats across snapshots and
/// hides the "is illegal mass falling?" signal. The probe network is a
/// fresh inference network carrying the trainer's analysis snapshot
/// (`analysisSnapshot(of: .trainer, initReferences:)`, identity checked).
extension SessionController {

    /// Entry point invoked by the Debug menu item. Runs the analyzer
    /// against `replayBuffer`; if no buffer is loaded, surfaces an
    /// explanatory alert instead of silently doing nothing.
    func analyzeReplayBufferToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Replay Buffer")
        guard let buf = replayBuffer else {
            Self.presentAnalyzeAlert(
                title: "Analyze Replay Buffer",
                message: "No replay buffer is loaded. Start Play-and-Train or load a saved session first.",
                revealURL: nil
            )
            return
        }
        // The champion is the entropy probe's fallback when there is no
        // trainer (or its snapshot fails); the ref captured here outlives the
        // task regardless of any concurrent session change.
        let champion = network
        let modelLabel = champion?.identifier?.description ?? "<no-id>"
        let hasTrainer = trainer != nil
        guard beginAnalysis("Replay Buffer") else { return }
        // Snapshot training-progress context on the main actor before
        // the detached work; stamped onto the result below.
        let exportMetadata = currentAnalysisExportMetadata()
        let entropySample = ReplayBufferAnalyzer.entropyProbeRandom(
            runSeed: runRandomSeed,
            trainerStep: trainingBox?.snapshot().stats.steps
        )
        SessionLogger.shared.log("[ANALYSIS] replay-buffer entropy probe positions: \(entropySample.description)")

        Task.detached(priority: .utility) {
            defer { Task { @MainActor in self.endAnalysis() } }
            var entropyProbe: (network: ChessMPSNetwork, label: String)?
            if hasTrainer {
                do {
                    let capture = try await self.analysisSnapshot(
                        of: .trainer, initReferences: AnalysisInitReferenceCache())
                    entropyProbe = await Self.buildEntropyProbe(from: capture.snapshot)
                } catch {
                    SessionLogger.shared.log(
                        "[ANALYSIS] Trainer entropy-probe snapshot failed: \(error.localizedDescription)"
                        + " — replay analyzer will fall back to champion for the entropy probe.")
                }
            }
            var result: ReplayBufferAnalyzer.Result
            do {
                result = try await Self.analyzeReplayBuffer(
                    buffer: buf, champion: champion, modelLabel: modelLabel,
                    entropyProbe: entropyProbe, sampleRandom: entropySample.random)
            } catch {
                // Only the entropy probe's forward passes throw (e.g. an
                // MPSGraph transient error). Still produce a file without
                // section (7); this line says why it is missing.
                SessionLogger.shared.log("[ANALYSIS] Policy-entropy probe failed: \(error). Falling back to pure-buffer analysis.")
                result = await ReplayBufferAnalyzer.runOffPool(buffer: buf, modelLabel: modelLabel)
            }
            result.exportMetadata = exportMetadata
            let outcome = await AnalysisJSONExport.summarizeAndPublishOffPool(
                result, modelLabel: modelLabel, directory: CheckpointPaths.analysesDir)
            AnalysisJSONExport.logSummary(outcome.summary, family: .replayBuffer, context: nil)

            switch outcome.written {
            case .success(let url):
                SessionLogger.shared.log("[ANALYSIS] Saved JSON: \(url.path)")
                await Self.presentAnalyzeAlert(
                    title: "Replay Buffer Analysis Complete",
                    message: """
                        Saved JSON to:
                        \(url.path)

                        A text summary was written to the session log under [ANALYSIS]; \
                        click Reveal in Finder to open the JSON in the output folder.
                        """,
                    revealURL: url
                )
            case .failure(let err):
                SessionLogger.shared.log("[ANALYSIS] JSON write failed: \(err)")
                await Self.presentAnalyzeAlert(
                    title: "Replay Buffer Analysis — JSON Write Failed",
                    message: """
                        The analyzer ran and a text summary was written to the session \
                        log, but writing the JSON file failed:

                        \(err.localizedDescription)
                        """,
                    revealURL: nil
                )
            }
        }
    }

    // MARK: - Shared with Run All Analyses

    /// The replay analyzer over `buffer`, with the per-bucket policy-entropy
    /// probe on `entropyProbe` when given (the trainer: the champion's policy
    /// is frozen between promotions, so probing it shows nothing between
    /// them), else on the live `champion`, else without the probe. The one
    /// branch both the single analysis and Run All take; each decides what a
    /// thrown probe failure costs.
    nonisolated static func analyzeReplayBuffer(
        buffer: ReplayBuffer,
        champion: ChessMPSNetwork?,
        modelLabel: String,
        entropyProbe: (network: ChessMPSNetwork, label: String)?,
        sampleRandom: DCMRandom
    ) async throws -> ReplayBufferAnalyzer.Result {
        if let entropyProbe {
            return try await ReplayBufferAnalyzer.runWithPolicyEntropy(
                buffer: buffer, network: entropyProbe.network, modelLabel: modelLabel,
                entropyModelLabel: entropyProbe.label, sampleRandom: sampleRandom)
        }
        if let champion {
            return try await ReplayBufferAnalyzer.runWithPolicyEntropy(
                buffer: buffer, network: champion, modelLabel: modelLabel, sampleRandom: sampleRandom)
        }
        return await ReplayBufferAnalyzer.runOffPool(buffer: buffer, modelLabel: modelLabel)
    }

    /// The entropy probe from a trainer snapshot: an inference network
    /// carrying `snapshot.weights` (the trainer's weights at
    /// `snapshot.trainingStep`, the same weights the trainer's other analyses
    /// read), labelled "trainer:<id> step <N>" — an unknown step is written
    /// as "unknown", never dropped. Built through `InferenceNetworkFactory`,
    /// so the MPSGraph construction runs on GCD, never on the cooperative
    /// pool. Nil — logged — when the build fails; the analyzer then probes the
    /// champion. Short-lived: dropped with the analysis.
    nonisolated static func buildEntropyProbe(
        from snapshot: AnalyzedNetworkSnapshot
    ) async -> (network: ChessMPSNetwork, label: String)? {
        let label = "\(snapshot.modelLabel) step \(snapshot.trainingStep.map { String($0) } ?? "unknown")"
        do {
            let network = try await InferenceNetworkFactory.build(loading: snapshot.weights, arch: snapshot.architecture)
            return (network, label)
        } catch {
            SessionLogger.shared.log(
                "[ANALYSIS] Entropy-probe network build for \(label) failed: \(error.localizedDescription)"
                + " — replay analyzer will fall back to champion for the entropy probe.")
            return nil
        }
    }

    // MARK: - Alert + Reveal in Finder

    /// Present a non-blocking result sheet. If `revealURL` is non-nil,
    /// adds a "Reveal in Finder" button that selects the file. Routed
    /// through `NonBlockingAlert` so a left-open dialog can never stall
    /// the training pipeline — see that type's doc comment.
    @MainActor
    private static func presentAnalyzeAlert(
        title: String,
        message: String,
        revealURL: URL?
    ) {
        NonBlockingAlert.presentInformational(
            title: title,
            message: message,
            revealURL: revealURL
        )
    }
}
