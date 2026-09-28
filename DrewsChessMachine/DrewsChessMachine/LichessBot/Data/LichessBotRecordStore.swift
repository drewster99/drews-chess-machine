import Foundation

/// Where a finalized game's files went.
struct LichessBotFinalizedGame: Sendable, Equatable {
    let record: LichessBotGameRecord
    let recordURL: URL
    let pgnURL: URL
    let journalURL: URL
}

/// Finalizes games and answers questions about the records on disk (plan
/// §10.2). Stateless apart from its configuration: the files are the state,
/// and every operation runs on the file queue.
///
/// **Finalize** reads a game's `InProgress/` journal, builds its record
/// (reconciled against the export when there is one), writes the `.json`
/// and `.pgn` atomically into `Games/YYYY/MM/`, moves the journal beside
/// them, and updates `index.json`. The order makes a crash at any point
/// recoverable: until the journal leaves `InProgress/`, the game is simply
/// finalized again on the next launch, overwriting any record already
/// written.
final class LichessBotRecordStore: Sendable {
    let directory: LichessBotDataDirectory
    private let fileQueue: LichessBotFileQueue
    private let ourAccountID: String

    init(directory: LichessBotDataDirectory, fileQueue: LichessBotFileQueue, ourAccountID: String) {
        self.directory = directory
        self.fileQueue = fileQueue
        self.ourAccountID = ourAccountID
    }

    /// Game ids with a journal still in `InProgress/`: live games, or games
    /// that finished while the app was down or before their export was
    /// reconciled (plan §10.2 launch recovery).
    func inProgressGameIDs() async throws -> [String] {
        let directory = self.directory
        return try await fileQueue.run {
            let fm = FileManager.default
            guard fm.fileExists(atPath: directory.inProgressDirectory.path) else { return [] }
            let suffix = "." + LichessBotDataDirectory.journalExtension
            return try fm.contentsOfDirectory(atPath: directory.inProgressDirectory.path)
                .filter { $0.hasSuffix(suffix) }
                .map { String($0.dropLast(suffix.count)) }
                .sorted()
        }
    }

    func readJournal(gameID: String) async throws -> LichessBotJSONLines.Decoded<LichessBotJournalEntry> {
        let url = directory.inProgressJournalURL(gameID: gameID)
        return try await fileQueue.run {
            try LichessBotJournal.read(url)
        }
    }

    /// The journal's own finished status, if it recorded one.
    func journalFinishedStatus(gameID: String) async throws -> String? {
        let journal = try await readJournal(gameID: gameID)
        for entry in journal.elements.reversed() {
            if case .finished(let status, _, _) = entry.event {
                return status
            }
        }
        return nil
    }

    /// Build, write and file the game's record. `export` must be terminal
    /// when given; pass nil with a reason only when Lichess has no export
    /// for the game.
    func finalize(gameID: String, export: LichessBotGameExport?, exportUnavailableReason: String?) async throws -> LichessBotFinalizedGame {
        let directory = self.directory
        let ourAccountID = self.ourAccountID
        return try await fileQueue.run {
            let journalURL = directory.inProgressJournalURL(gameID: gameID)
            let journal = try LichessBotJournal.read(journalURL)
            let built = try LichessBotRecordBuilder.build(
                gameID: gameID,
                journal: journal,
                export: export,
                exportUnavailableReason: exportUnavailableReason,
                ourAccountID: ourAccountID,
                checkedAt: Date()
            )

            // The record handed back is decoded from the bytes written, so
            // it is exactly what is on disk (timestamps are stored to the
            // millisecond).
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            let recordData = try encoder.encode(built)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
            let record = try decoder.decode(LichessBotGameRecord.self, from: recordData)

            let folder = directory.gamesMonthDirectory(createdAt: record.createdAt)
            let stem = LichessBotDataDirectory.fileStem(gameID: gameID, createdAt: record.createdAt)
            let recordURL = folder.appendingPathComponent("\(stem).json", isDirectory: false)
            let pgnURL = folder.appendingPathComponent("\(stem).pgn", isDirectory: false)
            let keptJournalURL = folder.appendingPathComponent("\(stem).\(LichessBotDataDirectory.journalExtension)", isDirectory: false)

            let fm = FileManager.default
            let journalHasGame = journal.elements.contains { entry in
                guard case .streamLine(let raw) = entry.event else { return false }
                do {
                    if case .gameFull = try LichessBotGameStreamLine.decode(Data(raw.utf8)) {
                        return true
                    }
                } catch {
                    // An undecodable line is not a gameFull; the record
                    // builder reports it as an anomaly.
                }
                return false
            }
            if !journalHasGame && fm.fileExists(atPath: recordURL.path) {
                // A fragment written after the game was filed: keep it beside
                // the record, but never let it replace the complete record.
                let fragmentURL = folder.appendingPathComponent("\(stem)-fragment-\(Int(Date().timeIntervalSince1970)).\(LichessBotDataDirectory.journalExtension)", isDirectory: false)
                try fm.moveItem(at: journalURL, to: fragmentURL)
                let existing = try LichessBotIndex.readRecord(at: recordURL)
                return LichessBotFinalizedGame(record: existing, recordURL: recordURL, pgnURL: pgnURL, journalURL: fragmentURL)
            }

            try LichessBotAtomicWrite.write(recordData, to: recordURL)
            try LichessBotAtomicWrite.write(Data(LichessBotPGNWriter.pgn(for: record).utf8), to: pgnURL)

            let filedJournalURL: URL
            if fm.fileExists(atPath: keptJournalURL.path) {
                // A journal for this game was already filed (the game was
                // finalized once, then more was written): keep both.
                filedJournalURL = folder.appendingPathComponent("\(stem)-\(Int(Date().timeIntervalSince1970)).\(LichessBotDataDirectory.journalExtension)", isDirectory: false)
            } else {
                filedJournalURL = keptJournalURL
            }
            try fm.moveItem(at: journalURL, to: filedJournalURL)

            _ = try LichessBotIndex.upsert(LichessBotGameSummary(record: record), recordURL: recordURL, in: directory)
            return LichessBotFinalizedGame(record: record, recordURL: recordURL, pgnURL: pgnURL, journalURL: filedJournalURL)
        }
    }

    /// The index, rebuilt first if stale.
    func loadIndex() async throws -> LichessBotIndex.File {
        let directory = self.directory
        return try await fileQueue.run {
            try LichessBotIndex.load(directory)
        }
    }

    /// Discard and rebuild the index ("Rebuild index").
    func rebuildIndex() async throws -> LichessBotIndex.File {
        let directory = self.directory
        return try await fileQueue.run {
            let rebuilt = try LichessBotIndex.rebuild(directory)
            try LichessBotIndex.write(rebuilt, to: directory)
            return rebuilt
        }
    }

    /// One-line summary for the session log (plan §10.4).
    static func summaryLine(_ record: LichessBotGameRecord) -> String {
        let opponent = record.opponent.name ?? record.opponent.aiLevel.map { "lichess AI level \($0)" } ?? record.opponent.id ?? "?"
        let clock: String
        if let initial = record.setup.clockInitialMilliseconds, let increment = record.setup.clockIncrementMilliseconds {
            clock = "\(initial / 1000)+\(increment / 1000)"
        } else {
            clock = "-"
        }
        let models = record.generations.map(\.modelID)
        var seen: Set<String> = []
        let uniqueModels = models.filter { seen.insert($0).inserted }.joined(separator: ",")
        return "[LICHESS-BOT] game \(record.gameID) \(record.outcome.pgnResult) (\(record.outcome.status)) as \(record.ourColor.rawValue) vs \(opponent) \(record.setup.rated ? "rated" : "casual") \(clock) model \(uniqueModels.isEmpty ? "-" : uniqueModels) plies \(record.outcome.plies) reconciliation \(record.reconciliation.outcome.rawValue)"
    }
}
