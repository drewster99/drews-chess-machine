import AppKit
import Foundation

/// `SessionController`'s value-head analyzer hook — wired to the
/// `Analyze Value Head Weights…` Debug menu item. Takes a weights snapshot
/// of the champion when the task runs (`analysisSnapshot(of: .champion,
/// initReferences:)`, identity checked across its export), runs
/// `ValueHeadAnalyzer` on it, writes a timestamped JSON file under
/// `~/Library/Application Support/DrewsChessMachine/Analyses/`
/// (`AnalysisJSONExport`), logs a `[VALHEAD]` text-summary block to the
/// session log, and surfaces an NSAlert with a Reveal-in-Finder action. The
/// analyzer, the summary and the write run on GCD; only the logging of the
/// block and the alert are on the main actor.
///
/// Independent of the replay-buffer analyzer — the value-head pass reads
/// only the network's weights, never touches the replay buffer.
extension SessionController {

    /// Entry point invoked by the Debug menu item. Runs the analyzer on the
    /// champion; if no network is loaded, surfaces an explanatory alert
    /// instead of silently doing nothing.
    func analyzeValueHeadToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Value Head Weights")
        guard network != nil else {
            Self.presentValueHeadAlert(
                title: "Analyze Value Head Weights",
                message: "No network is loaded. Build a network or load a saved session first.",
                revealURL: nil
            )
            return
        }
        guard beginAnalysis("Value Head") else { return }
        // The session's context, on the main actor; the file is stamped
        // with the analyzed weights below.
        let exportMetadata = currentAnalysisExportMetadata()

        Task {
            defer { self.endAnalysis() }
            let snapshot: AnalyzedNetworkSnapshot
            var result: ValueHeadAnalyzer.Result
            do {
                snapshot = try await self.analysisSnapshot(
                    of: .champion, initReferences: AnalysisInitReferenceCache()).snapshot
                result = try await ValueHeadAnalyzer.runOffPool(snapshot: snapshot, modelLabel: snapshot.modelLabel)
            } catch {
                SessionLogger.shared.log("[VALHEAD] analyzer failed: \(error)")
                Self.presentValueHeadAlert(
                    title: "Value Head Analyzer — Failed",
                    message: "The value-head analyzer threw an error:\n\n\(error.localizedDescription)",
                    revealURL: nil
                )
                return
            }

            result.exportMetadata = exportMetadata.describing(snapshot)
            let outcome = await AnalysisJSONExport.summarizeAndPublishOffPool(
                result, modelLabel: snapshot.modelLabel, directory: CheckpointPaths.analysesDir)
            AnalysisJSONExport.logSummary(outcome.summary, family: .valueHead, context: nil)

            switch outcome.written {
            case .success(let url):
                SessionLogger.shared.log("[VALHEAD] Saved JSON: \(url.path)")
                Self.presentValueHeadAlert(
                    title: "Value Head Analysis Complete",
                    message: """
                        Saved JSON to:
                        \(url.path)

                        A text summary was written to the session log under [VALHEAD]; \
                        click Reveal in Finder to open the JSON in the output folder.
                        """,
                    revealURL: url
                )
            case .failure(let err):
                SessionLogger.shared.log("[VALHEAD] JSON write failed: \(err)")
                Self.presentValueHeadAlert(
                    title: "Value Head Analysis — JSON Write Failed",
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

    // MARK: - Alert + Reveal in Finder

    @MainActor
    private static func presentValueHeadAlert(
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
