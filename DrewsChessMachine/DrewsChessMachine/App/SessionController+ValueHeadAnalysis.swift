import AppKit
import Foundation

/// `SessionController`'s value-head analyzer hook — wired to the
/// `Analyze Value Head Weights…` Debug menu item. Takes a weights snapshot
/// of the champion (`AnalyzedNetworkSnapshot`), runs `ValueHeadAnalyzer` on
/// it, writes a timestamped JSON file under
/// `~/Library/Application Support/DrewsChessMachine/Analyses/`, logs a
/// `[VALHEAD]` text-summary block to the session log, and surfaces an
/// NSAlert with a Reveal-in-Finder action.
///
/// Independent of the replay-buffer analyzer — the value-head pass reads
/// only the network's weights, never touches the replay buffer.
extension SessionController {

    /// Entry point invoked by the Debug menu item. Runs the analyzer on the
    /// champion; if no network is loaded, surfaces an explanatory alert
    /// instead of silently doing nothing.
    func analyzeValueHeadToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Value Head Weights")
        guard let target = championAnalysisTarget() else {
            Self.presentValueHeadAlert(
                title: "Analyze Value Head Weights",
                message: "No network is loaded. Build a network or load a saved session first.",
                revealURL: nil
            )
            return
        }
        let modelLabel = target.modelLabel
        guard beginAnalysis("Value Head") else { return }
        // The session's context, on the main actor; the file is stamped
        // with the analyzed weights below.
        let exportMetadata = currentAnalysisExportMetadata()

        Task {
            defer { self.endAnalysis() }
            let snapshot: AnalyzedNetworkSnapshot
            var result: ValueHeadAnalyzer.Result
            do {
                snapshot = try await self.analysisSnapshot(of: target)
                result = try ValueHeadAnalyzer.run(snapshot: snapshot, modelLabel: modelLabel)
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
            let summary = result.textSummary()
            let writeOutcome = Self.writeValueHeadJSON(
                result: result,
                modelLabel: modelLabel
            )

            SessionLogger.shared.log("[VALHEAD] === Value head analysis begin ===")
            for line in summary.split(separator: "\n", omittingEmptySubsequences: false) {
                SessionLogger.shared.log("[VALHEAD] \(line)")
            }
            SessionLogger.shared.log("[VALHEAD] === Value head analysis end ===")

            switch writeOutcome {
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

    // MARK: - JSON write

    /// Encode `result` and write it to a timestamped JSON file under
    /// `CheckpointPaths.analysesDir`. Creates the directory on demand.
    /// Filename embeds both a timestamp and a sanitized model label so
    /// consecutive analyses don't collide and remain attributable to
    /// a specific model snapshot.
    nonisolated private static func writeValueHeadJSON(
        result: ValueHeadAnalyzer.Result,
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
        let filename = "valuehead_analysis_\(stamp)_\(safeModel).json"
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

    /// Filesystem-safe timestamp matching the replay-buffer analyzer's
    /// filenames so the two artifact families sort interleaved by
    /// time in Finder.
    nonisolated private static func filenameTimestamp() -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        df.locale = Locale(identifier: "en_US_POSIX")
        return df.string(from: Date())
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
