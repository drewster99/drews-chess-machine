import Foundation

/// An analysis result exported as one JSON file and summarized as text for
/// the session log (or a headless CLI's stderr).
protocol AnalysisReport: Encodable, Sendable {
    /// The file family: the file-name prefix and the log block's tag.
    static var exportFamily: AnalysisJSONExport.Family { get }
    /// The human-readable digest of the result.
    func textSummary() -> String
}

extension ReplayBufferAnalyzer.Result: AnalysisReport {
    static var exportFamily: AnalysisJSONExport.Family { .replayBuffer }
}
extension ValueHeadAnalyzer.Result: AnalysisReport {
    static var exportFamily: AnalysisJSONExport.Family { .valueHead }
}
extension NetworkWeightAnalyzer.Result: AnalysisReport {
    static var exportFamily: AnalysisJSONExport.Family { .networkWeights }
}
extension NumericsAudit.Result: AnalysisReport {
    static var exportFamily: AnalysisJSONExport.Family { .numericsAudit }
}

/// The one writer of analysis JSON files and their `[TAG]` log blocks.
///
/// Five copies of this writer once lived beside the analyses, each running
/// `textSummary()` and the encode + write wherever its caller happened to be —
/// the main actor for the single value-head path, the cooperative pool for
/// Run All — and each writing with `.atomic` onto a name stamped to the
/// second, which silently replaced an earlier file of the same label in the
/// same second (`--analyze-numerics` over a run's checkpoints, which share a
/// model ID). Now the summary and the write run on a GCD queue, and the file
/// is published through `FileSafety` without ever replacing anything.
enum AnalysisJSONExport {

    /// The analysis file families. The raw value is the file-name prefix, so
    /// families sort apart in the folder and each sorts by time within.
    enum Family: String, Sendable {
        case replayBuffer = "replay_analysis"
        case valueHead = "valuehead_analysis"
        case networkWeights = "network_weights"
        case numericsAudit = "numerics_audit"

        /// The session-log tag of the family's summary block.
        var logTag: String {
            switch self {
            case .replayBuffer: return "[ANALYSIS]"
            case .valueHead: return "[VALHEAD]"
            case .networkWeights: return "[NETW]"
            case .numericsAudit: return "[NUMERICS]"
            }
        }

        /// The summary block's title on its begin / end lines.
        var logTitle: String {
            switch self {
            case .replayBuffer: return "Replay buffer analysis"
            case .valueHead: return "Value head analysis"
            case .networkWeights: return "Network weight analysis"
            case .numericsAudit: return "Numerics audit"
            }
        }
    }

    /// What `summarizeAndPublishOffPool` produced: the text summary always,
    /// and the published file or why it could not be written.
    struct Outcome: Sendable {
        let summary: String
        let written: Result<URL, any Error>
    }

    /// Analyses of one label in the same second (a CLI folder audit of one
    /// run's checkpoints) get `-2` … up to this; more than this per second is
    /// not a real workload, and the error names the folder.
    static let maxNumericSuffixAttempts = 100

    /// `report.textSummary()` and `publish` on a GCD queue: both walk the
    /// whole result, and encoding and durably writing a multi-megabyte file
    /// takes real time, so neither may hold the main actor or a cooperative
    /// thread. A failed write is returned rather than thrown: the summary is
    /// still worth logging.
    static func summarizeAndPublishOffPool<Report: AnalysisReport>(
        _ report: Report,
        modelLabel: String,
        directory: URL
    ) async -> Outcome {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                let summary = report.textSummary()
                let written = Result(catching: { try publish(report, modelLabel: modelLabel, directory: directory) })
                continuation.resume(returning: Outcome(summary: summary, written: written))
            }
        }
    }

    /// Encode `report` and publish it as a new file in `directory` (created
    /// when missing) named `<family>_<yyyyMMdd-HHmmss>_<label>.json`, or with
    /// `-2`, `-3`, … before `.json` when that name is taken. Nothing in the
    /// folder is ever replaced. Synchronous: call it through
    /// `summarizeAndPublishOffPool`, or from a headless CLI's own thread.
    static func publish<Report: AnalysisReport>(
        _ report: Report,
        modelLabel: String,
        directory: URL
    ) throws -> URL {
        try CheckpointPaths.ensureDirectory(directory)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(report)
        return try FileSafety.publishNewFileWithNumericSuffix(
            data,
            in: directory,
            stem: fileStem(family: Report.exportFamily, modelLabel: modelLabel, at: Date()),
            pathExtension: "json",
            maxAttempts: maxNumericSuffixAttempts
        )
    }

    /// `<family>_<yyyyMMdd-HHmmss>_<label>` in local time, the label with
    /// `/`, space and `:` replaced by `_` so a "champion:<id>" or
    /// "file:<name>" label is one plain path component.
    static func fileStem(family: Family, modelLabel: String, at date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let safeLabel = modelLabel
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: ":", with: "_")
        return "\(family.rawValue)_\(formatter.string(from: date))_\(safeLabel)"
    }

    /// Write `summary` to the session log as one block under the family's
    /// tag. `context` (e.g. "RunAll Champion") goes in parentheses on the
    /// begin / end lines so the blocks of one Run All pass are told apart.
    static func logSummary(_ summary: String, family: Family, context: String?) {
        let contextSuffix = context.map { " (\($0))" } ?? ""
        SessionLogger.shared.log("\(family.logTag) === \(family.logTitle) begin\(contextSuffix) ===")
        for line in summary.split(separator: "\n", omittingEmptySubsequences: false) {
            SessionLogger.shared.log("\(family.logTag) \(line)")
        }
        SessionLogger.shared.log("\(family.logTag) === \(family.logTitle) end\(contextSuffix) ===")
    }
}
