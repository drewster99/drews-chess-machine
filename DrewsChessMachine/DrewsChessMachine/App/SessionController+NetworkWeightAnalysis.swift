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

    /// Entry point invoked by the "Analyze Network Weights (Champion)…"
    /// Debug menu item. Runs the analyzer against the champion's
    /// network (`self.network`); surfaces an explanatory alert if no
    /// champion is loaded.
    func analyzeNetworkWeightsToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Network Weights (Champion)")
        guard let net = network else {
            Self.presentNetworkWeightsAlert(
                title: "Analyze Network Weights",
                message: "No champion network is loaded. Build a network or load a saved session first.",
                revealURL: nil
            )
            return
        }
        let modelLabel = "champion:\(net.identifier?.description ?? "<no-id>")"
        runNetworkWeightsAnalysis(
            networkInner: net.network,
            modelLabel: modelLabel,
            buttonContext: "Champion"
        )
    }

    /// Entry point invoked by the "Analyze Network Weights (Trainer)…"
    /// Debug menu item. Runs against the trainer's inner network
    /// (`self.trainer?.network`); surfaces an explanatory alert if
    /// no trainer is initialized.
    func analyzeNetworkWeightsTrainerToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Network Weights (Trainer)")
        guard let trainer = trainer else {
            Self.presentNetworkWeightsAlert(
                title: "Analyze Network Weights — Trainer",
                message: "No trainer is initialized. Start Play-and-Train first so the trainer network exists.",
                revealURL: nil
            )
            return
        }
        let modelLabel = "trainer:\(trainer.identifier?.description ?? "<no-id>")"
        runNetworkWeightsAnalysis(
            networkInner: trainer.network,
            modelLabel: modelLabel,
            buttonContext: "Trainer"
        )
    }

    /// Shared runner for the champion + trainer paths. Each path picks
    /// the right `ChessNetwork` (the type with `trainableVariables`)
    /// and a descriptive `modelLabel`; from there the analyzer pipeline,
    /// JSON write, log block, and alert presentation are identical.
    /// `buttonContext` is a short tag (e.g. "Champion" / "Trainer")
    /// that appears in the alert titles so the user knows which path
    /// produced the result.
    private func runNetworkWeightsAnalysis(
        networkInner: ChessNetwork,
        modelLabel: String,
        buttonContext: String
    ) {
        // Snapshot training-progress context on the main actor before
        // the detached work; stamped onto the result below.
        let exportMetadata = currentAnalysisExportMetadata()
        Task.detached(priority: .utility) {
            var result: NetworkWeightAnalyzer.Result
            do {
                result = try await NetworkWeightAnalyzer.run(
                    network: networkInner,
                    modelLabel: modelLabel
                )
            } catch {
                SessionLogger.shared.log("[NETW] analyzer failed (\(buttonContext)): \(error)")
                await MainActor.run {
                    Self.presentNetworkWeightsAlert(
                        title: "Network Weight Analyzer — \(buttonContext) Failed",
                        message: "The analyzer threw an error:\n\n\(error.localizedDescription)",
                        revealURL: nil
                    )
                }
                return
            }

            result.exportMetadata = exportMetadata
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
                    SessionLogger.shared.log("[NETW] Saved JSON (\(buttonContext)): \(url.path)")
                    Self.presentNetworkWeightsAlert(
                        title: "Network Weight Analysis Complete — \(buttonContext)",
                        message: """
                            Saved JSON to:
                            \(url.path)

                            A text summary was written to the session log under [NETW]; \
                            click Reveal in Finder to open the JSON in the output folder.
                            """,
                        revealURL: url
                    )
                case .failure(let err):
                    SessionLogger.shared.log("[NETW] JSON write failed (\(buttonContext)): \(err)")
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
        NonBlockingAlert.presentInformational(
            title: title,
            message: message,
            revealURL: revealURL
        )
    }
}
