import AppKit
import Foundation

/// `SessionController`'s whole-network weight analyzer hook — wired to
/// the `Analyze Network Weights…` Debug menu items (champion and trainer).
/// Takes one weights snapshot of the chosen network
/// (`AnalyzedNetworkSnapshot`), runs `NetworkWeightAnalyzer` and then
/// `NumericsAudit` on that same snapshot, writes a timestamped JSON file for
/// each under `CheckpointPaths.analysesDir`, logs `[NETW]` / `[NUMERICS]`
/// text summaries to the session log, and surfaces an NSAlert with a
/// Reveal-in-Finder action.
extension SessionController {

    /// Entry point invoked by the "Analyze Network Weights (Champion)…"
    /// Debug menu item; surfaces an explanatory alert if no champion is
    /// loaded.
    func analyzeNetworkWeightsToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Network Weights (Champion)")
        guard let target = championAnalysisTarget() else {
            Self.presentNetworkWeightsAlert(
                title: "Analyze Network Weights",
                message: "No champion network is loaded. Build a network or load a saved session first.",
                revealURL: nil
            )
            return
        }
        runNetworkWeightsAnalysis(
            target: target,
            mastersSource: .unavailable(NumericsAudit.championMastersNote),
            buttonContext: "Champion"
        )
    }

    /// Entry point invoked by the "Analyze Network Weights (Trainer)…"
    /// Debug menu item; surfaces an explanatory alert if no trainer is
    /// initialized.
    func analyzeNetworkWeightsTrainerToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Network Weights (Trainer)")
        guard let trainer, let target = trainerAnalysisTarget() else {
            Self.presentNetworkWeightsAlert(
                title: "Analyze Network Weights — Trainer",
                message: "No trainer is initialized. Start Play-and-Train first so the trainer network exists.",
                revealURL: nil
            )
            return
        }
        runNetworkWeightsAnalysis(
            target: target,
            mastersSource: .trainer(trainer, trainingIsRunning: realTraining),
            buttonContext: "Trainer"
        )
    }

    /// Shared runner for the champion + trainer paths: one snapshot, then
    /// the weight analysis and the numerics audit of that snapshot, JSON
    /// writes, log blocks and the alert. `buttonContext` is a short tag
    /// (e.g. "Champion" / "Trainer") that appears in the alert titles.
    private func runNetworkWeightsAnalysis(
        target: AnalysisTarget,
        mastersSource: NumericsMastersSource,
        buttonContext: String
    ) {
        guard beginAnalysis("Network Weights (\(buttonContext))") else { return }
        // The session's context, on the main actor; each file is stamped
        // with the analyzed weights below.
        let exportMetadata = currentAnalysisExportMetadata()
        let modelLabel = target.modelLabel
        Task {
            defer { self.endAnalysis() }
            let snapshot: AnalyzedNetworkSnapshot
            var result: NetworkWeightAnalyzer.Result
            do {
                snapshot = try await self.analysisSnapshot(of: target)
                result = try await NetworkWeightAnalyzer.runOffPool(snapshot: snapshot, modelLabel: modelLabel)
            } catch {
                SessionLogger.shared.log("[NETW] analyzer failed (\(buttonContext)): \(error)")
                Self.presentNetworkWeightsAlert(
                    title: "Network Weight Analyzer — \(buttonContext) Failed",
                    message: "The analyzer threw an error:\n\n\(error.localizedDescription)",
                    revealURL: nil
                )
                return
            }

            result.exportMetadata = exportMetadata.describing(snapshot)
            let summary = result.textSummary()
            let writeOutcome = await Self.writeNetworkWeightsJSONOffMain(result: result, modelLabel: modelLabel)
            let numerics = await Self.runNumericsAuditStep(
                snapshot: snapshot,
                modelLabel: modelLabel,
                mastersSource: mastersSource,
                metadata: exportMetadata,
                tag: buttonContext
            )
            let numericsLine = numerics.description

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

                        \(numericsLine)

                        Text summaries were written to the session log under [NETW] and \
                        [NUMERICS]; click Reveal in Finder to open the JSON in the output folder.
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

    /// `writeNetworkWeightsJSON` on a GCD queue: encoding a multi-megabyte
    /// result and writing it is synchronous work.
    nonisolated private static func writeNetworkWeightsJSONOffMain(
        result: NetworkWeightAnalyzer.Result,
        modelLabel: String
    ) async -> Result<URL, Error> {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: writeNetworkWeightsJSON(result: result, modelLabel: modelLabel))
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
