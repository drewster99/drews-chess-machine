import AppKit
import Foundation

/// `SessionController`'s whole-network weight analyzer hook — wired to
/// the `Analyze Network Weights…` Debug menu item. Runs
/// `NetworkWeightAnalyzer.run(...)` against the currently-loaded
/// champion network, writes a timestamped JSON file under
/// `CheckpointPaths.analysesDir`, logs a `[NETW]` text summary
/// block to the session log, and surfaces an NSAlert with a
/// Reveal-in-Finder action.
///
/// Independent of the buffer / value-head analyzers — reads only
/// the network's weights and runs in well under a second on the
/// project's ~2.4M-parameter network.
extension SessionController {

    /// Entry point invoked by the Debug menu item. Runs the analyzer
    /// against `network`; surfaces an explanatory alert if none is
    /// loaded.
    func analyzeNetworkWeightsToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Network Weights")
        guard let net = network else {
            Self.presentNetworkWeightsAlert(
                title: "Analyze Network Weights",
                message: "No network is loaded. Build a network or load a saved session first.",
                revealURL: nil
            )
            return
        }
        let modelLabel = net.identifier?.description ?? "<no-id>"

        Task.detached(priority: .utility) {
            // Same threading shape as the value-head analyzer: the
            // exportWeights await yields, so off-main; alert + log
            // surfacing hops back to MainActor at the end.
            let result: NetworkWeightAnalyzer.Result
            do {
                result = try await NetworkWeightAnalyzer.run(
                    network: net,
                    modelLabel: modelLabel
                )
            } catch {
                SessionLogger.shared.log("[NETW] analyzer failed: \(error)")
                await MainActor.run {
                    Self.presentNetworkWeightsAlert(
                        title: "Network Weight Analyzer — Failed",
                        message: "The analyzer threw an error:\n\n\(error.localizedDescription)",
                        revealURL: nil
                    )
                }
                return
            }

            let summary = result.textSummary()
            let writeOutcome = Self.writeNetworkWeightsJSON(
                result: result,
                modelLabel: modelLabel
            )

            await MainActor.run {
                SessionLogger.shared.log("[NETW] === Network weight analysis begin ===")
                for line in summary.split(separator: "\n", omittingEmptySubsequences: false) {
                    SessionLogger.shared.log("[NETW] \(line)")
                }
                SessionLogger.shared.log("[NETW] === Network weight analysis end ===")

                switch writeOutcome {
                case .success(let url):
                    SessionLogger.shared.log("[NETW] Saved JSON: \(url.path)")
                    Self.presentNetworkWeightsAlert(
                        title: "Network Weight Analysis Complete",
                        message: """
                            Saved JSON to:
                            \(url.path)

                            A text summary was written to the session log under [NETW]; \
                            click Reveal in Finder to open the JSON in the output folder.
                            """,
                        revealURL: url
                    )
                case .failure(let err):
                    SessionLogger.shared.log("[NETW] JSON write failed: \(err)")
                    Self.presentNetworkWeightsAlert(
                        title: "Network Weight Analysis — JSON Write Failed",
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
    }

    // MARK: - JSON write

    nonisolated private static func writeNetworkWeightsJSON(
        result: NetworkWeightAnalyzer.Result,
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
        let filename = "network_weights_\(stamp)_\(safeModel).json"
        let url = dir.appendingPathComponent(filename)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        do {
            let data = try encoder.encode(result)
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
    private static func presentNetworkWeightsAlert(
        title: String,
        message: String,
        revealURL: URL?
    ) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        if revealURL != nil {
            alert.addButton(withTitle: "Reveal in Finder")
        }
        let response = alert.runModal()
        if let url = revealURL,
           response == .alertSecondButtonReturn {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
    }
}
