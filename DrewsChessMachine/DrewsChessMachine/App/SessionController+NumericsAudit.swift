import Foundation

/// Where a numerics audit gets fp32 masters to compare against the working
/// weights: a trainer (read only while training is stopped), or nowhere,
/// with the reason.
enum NumericsMastersSource: Sendable {
    case unavailable(String)
    case trainer(ChessTrainer, trainingIsRunning: Bool)
}

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

    /// Run the audit on `snapshot`, write its JSON under
    /// `CheckpointPaths.analysesDir`, and log the `[NUMERICS]` block.
    /// `metadata` is the session-level snapshot; the file is stamped with
    /// the audited weights through `describing(_:)`.
    nonisolated static func runNumericsAuditStep(
        snapshot: AnalyzedNetworkSnapshot,
        modelLabel: String,
        mastersSource: NumericsMastersSource,
        metadata: AnalysisExportMetadata,
        tag: String
    ) async -> NumericsAuditStepOutcome {
        var result: NumericsAudit.Result
        do {
            let masters: [[Float]]?
            let mastersNote: String?
            let velocity: LayerHealth.VelocitySource
            switch mastersSource {
            case .unavailable(let note):
                masters = nil
                mastersNote = note
                velocity = .unavailable(reason: NumericsAudit.liveNetworkVelocityNote)
            case .trainer(let trainer, let running):
                (masters, mastersNote) = try await NumericsAudit.trainerMasters(trainer: trainer, trainingIsRunning: running)
                velocity = try await NumericsAudit.trainerVelocity(
                    trainer: trainer, networkTensorCount: snapshot.names.count,
                    trainableCount: snapshot.trainableCount, trainingIsRunning: running)
            }
            result = try await NumericsAudit.run(
                snapshot: snapshot,
                modelLabel: modelLabel,
                masters: masters,
                mastersNote: mastersNote,
                velocity: velocity,
                corpusShardURL: nil,
                lichessDirectory: .standard
            )
        } catch {
            SessionLogger.shared.log("[NUMERICS] audit failed (\(tag)): \(error)")
            return .failed(error.localizedDescription)
        }
        result.exportMetadata = metadata.describing(snapshot)
        let summary = result.textSummary()
        SessionLogger.shared.log("[NUMERICS] === Numerics audit begin (\(tag)) ===")
        for line in summary.split(separator: "\n", omittingEmptySubsequences: false) {
            SessionLogger.shared.log("[NUMERICS] \(line)")
        }
        SessionLogger.shared.log("[NUMERICS] === Numerics audit end (\(tag)) ===")
        switch writeNumericsAuditJSON(result: result, modelLabel: modelLabel) {
        case .success(let url):
            SessionLogger.shared.log("[NUMERICS] Saved JSON (\(tag)): \(url.path)")
            return .saved(url)
        case .failure(let error):
            SessionLogger.shared.log("[NUMERICS] JSON write failed (\(tag)): \(error)")
            return .writeFailed(error.localizedDescription)
        }
    }

    nonisolated static func writeNumericsAuditJSON(
        result: NumericsAudit.Result,
        modelLabel: String,
        directory: URL = CheckpointPaths.analysesDir
    ) -> Result<URL, Error> {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            return .failure(error)
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let safeModel = modelLabel
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: ":", with: "_")
        let url = directory.appendingPathComponent("numerics_audit_\(formatter.string(from: Date()))_\(safeModel).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        do {
            try encoder.encode(result).write(to: url, options: [.atomic])
            return .success(url)
        } catch {
            return .failure(error)
        }
    }
}
