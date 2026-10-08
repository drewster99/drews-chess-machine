import Foundation

/// Where a finalized game's files went.
struct LichessBotFinalizedGame: Sendable, Equatable {
    let record: LichessBotGameRecord
    let recordURL: URL
    let pgnURL: URL
    let journalURL: URL
    /// Why `index.json` wasn't updated, when it wasn't. The game is filed
    /// either way (its journal has left InProgress/); the next index load sees
    /// the signature mismatch and rebuilds the cache.
    let indexUpdateFailure: String?
}

/// Where a game stands with respect to filing, from the files alone.
enum LichessBotFilingState: Sendable, Equatable {
    /// Its journal is still in `InProgress/`: filing has work to do.
    case journalInProgress
    /// No journal in `InProgress/`, and a readable record is in the index:
    /// the game is filed.
    case filed
    /// No journal in `InProgress/`, and a record file for the game exists
    /// but doesn't decode.
    case filedButUnreadable(path: String, error: String)
    /// No journal in `InProgress/` and no record: there is nothing to file.
    case missing
}

/// Finalizes games and answers questions about the records on disk (plan
/// §10.2). Stateless apart from its configuration: the files are the state.
/// Journal work runs on the journal queue — the same queue the journal
/// writer appends on, so an append and the move that files its journal never
/// interleave — and index work on the index queue, so a rebuild (which
/// decodes every record) never sits in front of a live game's append.
///
/// **Finalize** reads a game's `InProgress/` journal, builds its record
/// (reconciled against the export when there is one), writes the `.json`
/// and `.pgn` atomically into `Games/YYYY/MM/`, and moves the journal beside
/// them, all on the journal queue; then it updates `index.json` on the index
/// queue. The order makes a crash at any point recoverable: until the journal
/// leaves `InProgress/`, the game is simply finalized again on the next
/// launch, overwriting any record already written. Once it has left, the game
/// is filed, and an index failure is reported rather than thrown: the index
/// is a cache its next load rebuilds.
final class LichessBotRecordStore: Sendable {
    let directory: LichessBotDataDirectory
    /// Journals: listing, reading, and finalize's read-build-write-move.
    private let journalQueue: LichessBotFileQueue
    /// `index.json`: loads, rebuilds and upserts.
    private let indexQueue: LichessBotFileQueue
    private let ourAccountID: String

    init(directory: LichessBotDataDirectory, journalQueue: LichessBotFileQueue, indexQueue: LichessBotFileQueue, ourAccountID: String) {
        self.directory = directory
        self.journalQueue = journalQueue
        self.indexQueue = indexQueue
        self.ourAccountID = ourAccountID
    }

    /// One queue for journals and index alike: every operation in one order.
    /// Correct, without the isolation the two-queue form gives live play.
    convenience init(directory: LichessBotDataDirectory, fileQueue: LichessBotFileQueue, ourAccountID: String) {
        self.init(directory: directory, journalQueue: fileQueue, indexQueue: fileQueue, ourAccountID: ourAccountID)
    }

    /// Game ids with a journal still in `InProgress/`: live games, or games
    /// that finished while the app was down or before their export was
    /// reconciled (plan §10.2 launch recovery).
    func inProgressGameIDs() async throws -> [String] {
        let directory = self.directory
        return try await journalQueue.run {
            let fm = FileManager.default
            guard fm.fileExists(atPath: directory.inProgressDirectory.path) else { return [] }
            let suffix = "." + LichessBotDataDirectory.journalExtension
            return try fm.contentsOfDirectory(atPath: directory.inProgressDirectory.path)
                .filter { $0.hasSuffix(suffix) }
                .map { String($0.dropLast(suffix.count)) }
                .sorted()
        }
    }

    /// When each `InProgress/` journal was created — roughly when its game
    /// started, for seeding today's game counts.
    func inProgressJournalCreationDates() async throws -> [String: Date] {
        let directory = self.directory
        return try await journalQueue.run {
            let fm = FileManager.default
            guard fm.fileExists(atPath: directory.inProgressDirectory.path) else { return [:] }
            let suffix = "." + LichessBotDataDirectory.journalExtension
            var dates: [String: Date] = [:]
            for name in try fm.contentsOfDirectory(atPath: directory.inProgressDirectory.path) where name.hasSuffix(suffix) {
                let url = directory.inProgressDirectory.appendingPathComponent(name, isDirectory: false)
                guard let created = try url.resourceValues(forKeys: [.creationDateKey]).creationDate else {
                    throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: url.path, NSLocalizedDescriptionKey: "no creation date for \(name)"])
                }
                dates[String(name.dropLast(suffix.count))] = created
            }
            return dates
        }
    }

    func readJournal(gameID: String) async throws -> LichessBotJSONLines.Decoded<LichessBotJournalEntry> {
        let url = try directory.validatedInProgressJournalURL(gameID: gameID)
        return try await journalQueue.run {
            try LichessBotJournal.read(url)
        }
    }

    /// Whether the game still has a journal to file, is filed, or neither.
    /// The journal check runs on the journal queue behind any append already
    /// queued, so a journal being created is never missed; the index is
    /// loaded (rebuilt first when stale), so a record whose index update
    /// failed still counts as filed.
    func filingState(gameID: String) async throws -> LichessBotFilingState {
        let journalURL = try directory.validatedInProgressJournalURL(gameID: gameID)
        let journalExists = try await journalQueue.run {
            FileManager.default.fileExists(atPath: journalURL.path)
        }
        if journalExists {
            return .journalInProgress
        }
        let index = try await loadIndex()
        if index.rows.contains(where: { $0.gameID == gameID }) {
            return .filed
        }
        // Record stems end in "-<gameID>", and a game id is letters and
        // digits only (`LichessBotGameIDPathSafety`), so this names only
        // this game.
        let recordSuffix = "-\(gameID).json"
        if let unreadable = index.unreadableRecords.first(where: { $0.path.hasSuffix(recordSuffix) }) {
            return .filedButUnreadable(path: unreadable.path, error: unreadable.error)
        }
        return .missing
    }

    /// A leftover journal, checked and prepared for resuming its game
    /// (`LichessBotResumedJournal.make`), on the journal queue: reading and
    /// decoding a long game's journal is real work, kept off the cooperative
    /// pool and the main actor.
    func resumedJournal(gameID: String) async throws -> LichessBotResumedJournal {
        let url = try directory.validatedInProgressJournalURL(gameID: gameID)
        let ourAccountID = self.ourAccountID
        return try await journalQueue.run {
            try LichessBotResumedJournal.make(gameID: gameID, journal: try LichessBotJournal.read(url), ourAccountID: ourAccountID)
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
        let filing = try await journalQueue.run {
            try Self.fileGame(gameID: gameID, export: export, exportUnavailableReason: exportUnavailableReason, ourAccountID: ourAccountID, directory: directory)
        }
        guard filing.recordWritten else {
            return filing.game
        }
        // The journal has left InProgress/: the game is filed whatever happens
        // next. A throw here would send the game back to the reconciler to be
        // filed again from a journal that is gone.
        let record = filing.game.record
        let recordURL = filing.game.recordURL
        do {
            try await indexQueue.run {
                // On the index queue: a generation recorded before generations
                // kept their training history reads the played file's header.
                let summary = LichessBotGameSummary(record: record, playedFiles: LichessBotPlayedFileHistories())
                _ = try LichessBotIndex.upsert(summary, recordURL: recordURL, in: directory)
            }
            return filing.game
        } catch {
            let game = filing.game
            return LichessBotFinalizedGame(record: game.record, recordURL: game.recordURL, pgnURL: game.pgnURL, journalURL: game.journalURL, indexUpdateFailure: error.localizedDescription)
        }
    }

    private struct Filing: Sendable {
        let game: LichessBotFinalizedGame
        /// False for a fragment filed beside an untouched record.
        let recordWritten: Bool
    }

    /// The journal-queue part of `finalize`: read, build, write and move.
    private static func fileGame(gameID: String, export: LichessBotGameExport?, exportUnavailableReason: String?, ourAccountID: String, directory: LichessBotDataDirectory) throws -> Filing {
        let fm = FileManager.default
        let journalURL = try directory.validatedInProgressJournalURL(gameID: gameID)
        let journal = try LichessBotJournal.read(journalURL)
        let journalFull = LichessBotRecordBuilder.firstGameFull(in: journal.elements)
        let createdAt: Date
        if let journalFull {
            createdAt = LichessBotRecordBuilder.creationDate(of: journalFull)
        } else if let export, let exportCreatedAt = LichessBotRecordBuilder.creationDate(of: export) {
            createdAt = exportCreatedAt
        } else {
            throw LichessBotRecordError.noGameInformation(gameID: gameID)
        }
        let folder = directory.gamesMonthDirectory(createdAt: createdAt)
        let stem = try LichessBotDataDirectory.validatedFileStem(gameID: gameID, createdAt: createdAt)
        let journalSuffix = "." + LichessBotDataDirectory.journalExtension
        let recordURL = folder.appendingPathComponent("\(stem).json", isDirectory: false)
        let pgnURL = folder.appendingPathComponent("\(stem).pgn", isDirectory: false)
        let keptJournalURL = folder.appendingPathComponent(stem + journalSuffix, isDirectory: false)
        let recordExists = fm.fileExists(atPath: recordURL.path)

        if journalFull == nil && recordExists {
            // A fragment written after the game was filed. Checked before any
            // build: a fragment alone may not describe the game. It is kept
            // beside the record and never replaces it. The existing record is
            // read before the move, so an unreadable one leaves the fragment
            // where it is.
            let existing = try LichessBotIndex.readRecord(at: recordURL)
            let fragmentURL = folder.appendingPathComponent("\(stem)-fragment-\(uniqueJournalSuffix())\(journalSuffix)", isDirectory: false)
            try fm.moveItem(at: journalURL, to: fragmentURL)
            return Filing(game: LichessBotFinalizedGame(record: existing, recordURL: recordURL, pgnURL: pgnURL, journalURL: fragmentURL, indexUpdateFailure: nil), recordWritten: false)
        }

        // A record already filed means the game's journal was filed too, and
        // this is a later one (the game was seen again after filing). The
        // record is rebuilt from every journal kept for the game, oldest
        // first, then this one: concatenated journals are exactly what a
        // resumed journal looks like to the builder, so nothing an earlier
        // journal held (DCM's decisions, chat) is lost. After a crash between
        // writing the record and moving the journal, no journal is kept yet,
        // so this is the current journal alone, as before.
        let earlier = recordExists ? try filedJournals(in: folder, stem: stem) : []
        let combined = LichessBotJSONLines.Decoded(
            elements: earlier.flatMap(\.elements) + journal.elements,
            droppedTrailingByteCount: earlier.map(\.droppedTrailingByteCount).reduce(0, +) + journal.droppedTrailingByteCount
        )
        let hasFinish = combined.elements.contains { entry in
            if case .finished = entry.event { return true }
            return false
        }
        if export == nil && !hasFinish {
            throw LichessBotRecordError.noOutcome(gameID: gameID)
        }
        let built = try LichessBotRecordBuilder.build(
            gameID: gameID,
            journal: combined,
            export: export,
            exportUnavailableReason: exportUnavailableReason,
            ourAccountID: ourAccountID,
            checkedAt: Date()
        )
        if combined.droppedTrailingByteCount > 0 {
            SessionLogger.shared.log("[LICHESS-BOT] game \(gameID): dropped \(combined.droppedTrailingByteCount) bytes of unterminated journal line(s) left by an interrupted write")
        }

        // The record handed back is decoded from the bytes written, so it is
        // exactly what is on disk (timestamps are stored to the millisecond).
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let recordData = try encoder.encode(built)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        let record = try decoder.decode(LichessBotGameRecord.self, from: recordData)

        try LichessBotAtomicWrite.write(recordData, to: recordURL)
        try LichessBotAtomicWrite.write(Data(LichessBotPGNWriter.pgn(for: record).utf8), to: pgnURL)
        let filedJournalURL: URL
        if fm.fileExists(atPath: keptJournalURL.path) {
            // A journal for this game was already filed: keep both.
            filedJournalURL = folder.appendingPathComponent("\(stem)-\(uniqueJournalSuffix())\(journalSuffix)", isDirectory: false)
        } else {
            filedJournalURL = keptJournalURL
        }
        try fm.moveItem(at: journalURL, to: filedJournalURL)
        return Filing(game: LichessBotFinalizedGame(record: record, recordURL: recordURL, pgnURL: pgnURL, journalURL: filedJournalURL, indexUpdateFailure: nil), recordWritten: true)
    }

    /// Every journal already filed for the game with this stem (the kept one,
    /// later ones, fragments), oldest first by first entry. The stem ends at
    /// the game id, so the stem plus a separator names only this game.
    private static func filedJournals(in folder: URL, stem: String) throws -> [LichessBotJSONLines.Decoded<LichessBotJournalEntry>] {
        let suffix = "." + LichessBotDataDirectory.journalExtension
        return try FileManager.default.contentsOfDirectory(atPath: folder.path)
            .filter { $0.hasSuffix(suffix) && ($0 == stem + suffix || $0.hasPrefix(stem + "-")) }
            .map { try LichessBotJournal.read(folder.appendingPathComponent($0, isDirectory: false)) }
            .filter { !$0.elements.isEmpty }
            .sorted { $0.elements[0].at < $1.elements[0].at }
    }

    /// A name suffix no other filed journal can have. Whole-second timestamps
    /// collided when two were filed within one second, and a collision makes
    /// the move throw.
    private static func uniqueJournalSuffix() -> String {
        UUID().uuidString
    }

    /// The index, rebuilt first if stale.
    func loadIndex() async throws -> LichessBotIndex.File {
        let directory = self.directory
        return try await indexQueue.run {
            try LichessBotIndex.load(directory)
        }
    }

    /// Discard and rebuild the index ("Rebuild index").
    func rebuildIndex() async throws -> LichessBotIndex.File {
        let directory = self.directory
        return try await indexQueue.run {
            let rebuilt = try LichessBotIndex.rebuild(directory, reason: "rebuild requested")
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
