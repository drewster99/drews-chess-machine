import AppKit
import Foundation

/// `SessionController`'s whole-network weight analyzer hook — wired to
/// the `Analyze Network Weights…` Debug menu items (champion and trainer).
/// Takes one weights snapshot of the chosen network when the task runs
/// (`analysisSnapshot(of:initReferences:)`, identity checked across its
/// export), runs `NetworkWeightAnalyzer` and then `NumericsAudit` on that
/// same snapshot — the audit's masters and velocity from the same cut —
/// writes a timestamped JSON file for each under `CheckpointPaths.analysesDir`
/// (`AnalysisJSONExport`), logs `[NETW]` / `[NUMERICS]` text summaries to the
/// session log, and surfaces an NSAlert with a Reveal-in-Finder action. The
/// analyses, summaries and writes run on GCD; only the logging of the blocks
/// and the alert are on the main actor.
extension SessionController {

    /// Entry point invoked by the "Analyze Network Weights (Champion)…"
    /// Debug menu item; surfaces an explanatory alert if no champion is
    /// loaded.
    func analyzeNetworkWeightsToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Network Weights (Champion)")
        guard network != nil else {
            Self.presentNetworkWeightsAlert(
                title: "Analyze Network Weights",
                message: "No champion network is loaded. Build a network or load a saved session first.",
                revealURL: nil
            )
            return
        }
        runNetworkWeightsAnalysis(of: .champion, buttonContext: "Champion")
    }

    /// Entry point invoked by the "Analyze Network Weights (Trainer)…"
    /// Debug menu item; surfaces an explanatory alert if no trainer is
    /// initialized.
    func analyzeNetworkWeightsTrainerToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Network Weights (Trainer)")
        guard trainer != nil else {
            Self.presentNetworkWeightsAlert(
                title: "Analyze Network Weights — Trainer",
                message: "No trainer is initialized. Start Play-and-Train first so the trainer network exists.",
                revealURL: nil
            )
            return
        }
        runNetworkWeightsAnalysis(of: .trainer, buttonContext: "Trainer")
    }

    /// Shared runner for the champion + trainer paths: one snapshot (taken
    /// when the task runs, identity checked across its export), then the
    /// weight analysis and the numerics audit of it, JSON writes, log blocks
    /// and the alert. `buttonContext` is a short tag (e.g. "Champion" /
    /// "Trainer") that appears in the alert titles.
    private func runNetworkWeightsAnalysis(of role: AnalyzedNetworkSnapshot.Role, buttonContext: String) {
        guard beginAnalysis("Network Weights (\(buttonContext))") else { return }
        // The session's context, on the main actor; each file is stamped
        // with the analyzed weights below.
        let exportMetadata = currentAnalysisExportMetadata()
        Task {
            defer { self.endAnalysis() }
            let capture: AnalysisCapture
            var result: NetworkWeightAnalyzer.Result
            do {
                capture = try await self.analysisSnapshot(of: role, initReferences: AnalysisInitReferenceCache())
                result = try await NetworkWeightAnalyzer.runOffPool(
                    snapshot: capture.snapshot, modelLabel: capture.snapshot.modelLabel)
            } catch {
                SessionLogger.shared.log("[NETW] analyzer failed (\(buttonContext)): \(error)")
                Self.presentNetworkWeightsAlert(
                    title: "Network Weight Analyzer — \(buttonContext) Failed",
                    message: "The analyzer threw an error:\n\n\(error.localizedDescription)",
                    revealURL: nil
                )
                return
            }

            let snapshot = capture.snapshot
            let modelLabel = snapshot.modelLabel
            result.exportMetadata = exportMetadata.describing(snapshot)
            let outcome = await AnalysisJSONExport.summarizeAndPublishOffPool(
                result, modelLabel: modelLabel, directory: CheckpointPaths.analysesDir)
            AnalysisJSONExport.logSummary(outcome.summary, family: .networkWeights, context: nil)
            switch outcome.written {
            case .success(let url):
                SessionLogger.shared.log("[NETW] Saved JSON (\(buttonContext)): \(url.path)")
            case .failure(let err):
                SessionLogger.shared.log("[NETW] JSON write failed (\(buttonContext)): \(err)")
            }
            let numerics = await Self.runNumericsAuditStep(capture: capture, metadata: exportMetadata, tag: buttonContext)
            let numericsLine = numerics.description

            switch outcome.written {
            case .success(let url):
                Self.presentNetworkWeightsAlert(
                    title: "Network Weight Analysis Complete — \(buttonContext)",
                    message: """
                        Saved JSON to:
                        \(url.path)

                        \(numericsLine)

                        Text summaries were written to the session log under [NETW] and \
                        [NUMERICS]; click Reveal in Finder to open the JSON in the output folder.
                        """,
                    revealURL: url
                )
            case .failure(let err):
                Self.presentNetworkWeightsAlert(
                    title: "Network Weight Analysis — \(buttonContext) JSON Write Failed",
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

    // MARK: - Alert

    @MainActor
    private static func presentNetworkWeightsAlert(
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
