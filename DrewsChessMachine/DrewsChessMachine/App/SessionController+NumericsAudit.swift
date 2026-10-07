import Foundation

/// What one numerics-audit run produced, for the alert and Run All's summary
/// (the `[NUMERICS]` block is already in the log).
enum NumericsAuditStepOutcome: Sendable {
    case saved(URL)
    case writeFailed(String)
    case failed(String)

    /// One line for an alert or a summary.
    var description: String {
        switch self {
        case .saved(let url): return "Numerics audit JSON:\n\(url.path)"
        case .writeFailed(let error): return "Numerics audit ran (see [NUMERICS] in the log) but its JSON write failed: \(error)"
        case .failed(let error): return "Numerics audit failed: \(error)"
        }
    }
}

/// `SessionController`'s numerics-audit hook (head numerics plan Phase 0):
/// run alongside the network weight analyzer from the Debug menu and from
/// Run All Analyses, against the champion or the trainer.
extension SessionController {

    /// Run the audit on `capture` — its snapshot's weights and the optimizer
    /// state read in the same cut — log the `[NUMERICS]` block and write its
    /// JSON under `CheckpointPaths.analysesDir` (`AnalysisJSONExport`, on
    /// GCD). `metadata` is the session-level snapshot; the file is stamped
    /// with the audited weights through `describing(_:)`.
    nonisolated static func runNumericsAuditStep(
        capture: AnalysisCapture,
        metadata: AnalysisExportMetadata,
        tag: String
    ) async -> NumericsAuditStepOutcome {
        let snapshot = capture.snapshot
        let modelLabel = snapshot.modelLabel
        var result: NumericsAudit.Result
        do {
            result = try await NumericsAudit.run(
                snapshot: snapshot,
                optimizerState: capture.optimizerState,
                modelLabel: modelLabel,
                corpusShardURL: nil,
                lichessDirectory: .standard
            )
        } catch {
            SessionLogger.shared.log("[NUMERICS] audit failed (\(tag)): \(error)")
            return .failed(error.localizedDescription)
        }
        result.exportMetadata = metadata.describing(snapshot)
        let outcome = await AnalysisJSONExport.summarizeAndPublishOffPool(
            result, modelLabel: modelLabel, directory: CheckpointPaths.analysesDir)
        AnalysisJSONExport.logSummary(outcome.summary, family: .numericsAudit, context: tag)
        switch outcome.written {
        case .success(let url):
            SessionLogger.shared.log("[NUMERICS] Saved JSON (\(tag)): \(url.path)")
            return .saved(url)
        case .failure(let error):
            SessionLogger.shared.log("[NUMERICS] JSON write failed (\(tag)): \(error)")
            return .writeFailed(error.localizedDescription)
        }
    }
}
