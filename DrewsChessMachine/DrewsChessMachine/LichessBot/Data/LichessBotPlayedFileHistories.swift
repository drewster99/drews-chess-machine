import Foundation

/// The training history of the weights a game generation played, for the
/// index (`MODEL_TRAINING_METHOD_PLAN.md`): the one the generation recorded,
/// or — for a generation recorded before generations kept it — the history
/// in the header of the file it played (owner decision 2026-10-08: past
/// games read the played file's header; records are never rewritten).
///
/// A played file counts only while it still holds the generation's
/// `model_id`. Not its exact bytes: a rolling `-latest` file is rewritten in
/// place by its own run, which keeps one `model_id`, so the file's history
/// is still the history of the weights played; a whole-file hash would call
/// every rolling file unknown. A file that is gone, unreadable or holds
/// another model leaves the history unknown, logged once per path per use.
///
/// One instance per index rebuild or filing, on the index's file queue (it
/// reads model headers and is not thread-safe); each path is read once.
final class LichessBotPlayedFileHistories {
    /// Reads a model file's catalog entry: its header (a whole legacy
    /// `.dcmmodel`, which has none) in the app; scripted in a test.
    private let readEntry: (URL) throws -> ModelFileEntry
    private var entries: [String: Result<ModelFileEntry, Error>] = [:]
    private var loggedMismatches: Set<String> = []

    /// Reads the played files' headers through the model catalog.
    convenience init() {
        self.init(readEntry: Self.catalogEntry(at:))
    }

    init(readEntry: @escaping (URL) throws -> ModelFileEntry) {
        self.readEntry = readEntry
    }

    /// The catalog's entry for a model file, by its extension.
    static func catalogEntry(at url: URL) throws -> ModelFileEntry {
        url.pathExtension == "dcmmodel" ? try ModelFileCatalog.legacyEntry(for: url) : try ModelFileCatalog.entry(for: url)
    }

    func history(of generation: LichessBotGenerationInfo) -> ModelTrainingHistory? {
        if let recorded = generation.trainingHistory {
            return recorded
        }
        guard let path = generation.filePath else { return nil }
        let entry: ModelFileEntry
        switch cachedEntry(atPath: path) {
        case .success(let read):
            entry = read
        case .failure:
            return nil
        }
        guard entry.modelID == generation.modelID else {
            if loggedMismatches.insert(path).inserted {
                SessionLogger.shared.log("[LICHESS-BOT] training method of played file \(URL(fileURLWithPath: path).lastPathComponent) not read: it now holds \(entry.modelID), not \(generation.modelID)")
            }
            return nil
        }
        return entry.trainingHistory
    }

    private func cachedEntry(atPath path: String) -> Result<ModelFileEntry, Error> {
        if let cached = entries[path] {
            return cached
        }
        let url = URL(fileURLWithPath: path)
        let result = Result { try readEntry(url) }
        if case .failure(let error) = result {
            SessionLogger.shared.log("[LICHESS-BOT] training method of played file \(url.lastPathComponent) not read: \(error.localizedDescription)")
        }
        entries[path] = result
        return result
    }
}
