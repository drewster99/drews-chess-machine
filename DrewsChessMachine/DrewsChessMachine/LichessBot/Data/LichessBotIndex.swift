import Foundation

/// The per-game fields Stats needs (plan §10.5, §11), derived from a
/// `LichessBotGameRecord`.
struct LichessBotGameSummary: Sendable, Codable, Equatable {
    let gameID: String
    let createdAt: Date
    let speed: String
    let perf: String?
    let rated: Bool
    let ourColor: LichessBotColorName
    let opponentID: String?
    let opponentName: String?
    let opponentKind: LichessBotOpponentKind
    let opponentTitle: String?
    let opponentRating: Int?
    let ourRatingBefore: Int?
    let ourRatingDiff: Int?
    let status: String
    let winner: String?
    let ourScore: Double?
    let plies: Int
    let modelIDs: [String]
    let sourceKinds: [String]
    let builds: [Int]
    let reconciliation: LichessBotGameRecord.Reconciliation.Outcome
    let anomalyCount: Int

    init(record: LichessBotGameRecord) {
        gameID = record.gameID
        createdAt = record.createdAt
        speed = record.setup.speed
        perf = record.setup.perf
        rated = record.setup.rated
        ourColor = record.ourColor
        opponentID = record.opponent.id
        opponentName = record.opponent.name
        opponentKind = record.opponent.kind
        opponentTitle = record.opponent.title
        opponentRating = record.opponent.ratingBefore
        ourRatingBefore = record.us.ratingBefore
        ourRatingDiff = record.us.ratingDiff
        status = record.outcome.status
        winner = record.outcome.winner
        ourScore = record.outcome.ourScore
        plies = record.outcome.plies
        var seenModels: Set<String> = []
        modelIDs = record.generations.map(\.modelID).filter { seenModels.insert($0).inserted }
        var seenSources: Set<String> = []
        sourceKinds = record.generations.map(\.sourceKind.rawValue).filter { seenSources.insert($0).inserted }
        builds = record.builds
        reconciliation = record.reconciliation.outcome
        anomalyCount = record.anomalies.count
    }
}

/// `index.json`: a derived cache of `LichessBotGameSummary` rows, never
/// authoritative (plan §10.5). It records how many record files it was
/// built from and the newest one's modification time; when either no
/// longer matches the `Games/` folder, it is rebuilt from the records.
/// Every function here runs on `LichessBotFileQueue`.
enum LichessBotIndex {
    static let schemaVersion = 1

    struct File: Sendable, Codable, Equatable {
        let schemaVersion: Int
        let recordCount: Int
        /// Seconds since the epoch, stored as a Double so it survives the
        /// JSON round trip exactly and the staleness check can compare it.
        let newestRecordModified: Double?
        /// Newest game first.
        let rows: [LichessBotGameSummary]
    }

    /// Every `*.json` record file under `Games/` (journals and PGNs
    /// excluded), with its modification time.
    static func recordFiles(in directory: LichessBotDataDirectory) throws -> [(url: URL, modified: Double)] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.gamesDirectory.path) else {
            return []
        }
        guard let enumerator = fm.enumerator(
            at: directory.gamesDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: directory.gamesDirectory.path])
        }
        var files: [(url: URL, modified: Double)] = []
        for case let url as URL in enumerator where url.pathExtension == "json" {
            let values = try url.resourceValues(forKeys: [.contentModificationDateKey, .isRegularFileKey])
            guard values.isRegularFile == true else { continue }
            guard let modified = values.contentModificationDate else {
                throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: url.path, NSLocalizedDescriptionKey: "no modification date for \(url.lastPathComponent)"])
            }
            files.append((url, modified.timeIntervalSince1970))
        }
        return files
    }

    static func readRecord(at url: URL) throws -> LichessBotGameRecord {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        return try decoder.decode(LichessBotGameRecord.self, from: Data(contentsOf: url))
    }

    /// Build the index from the record files.
    static func rebuild(_ directory: LichessBotDataDirectory) throws -> File {
        let files = try recordFiles(in: directory)
        let rows = try files.map { try LichessBotGameSummary(record: readRecord(at: $0.url)) }
        return File(
            schemaVersion: schemaVersion,
            recordCount: files.count,
            newestRecordModified: files.map(\.modified).max(),
            rows: sorted(rows)
        )
    }

    /// The index, rebuilt and rewritten first if it is missing, stale or
    /// from another schema.
    static func load(_ directory: LichessBotDataDirectory) throws -> File {
        let files = try recordFiles(in: directory)
        if let stored = readStored(directory),
           stored.recordCount == files.count,
           stored.newestRecordModified == files.map(\.modified).max() {
            return stored
        }
        let rebuilt = try rebuild(directory)
        try write(rebuilt, to: directory)
        return rebuilt
    }

    /// Add one game's row after its record file (`recordURL`) was written.
    /// Incremental when the stored index exactly covered every *other*
    /// record; otherwise (a stale index, or a record written again) a full
    /// rebuild — either way the result equals `rebuild`.
    static func upsert(_ summary: LichessBotGameSummary, recordURL: URL, in directory: LichessBotDataDirectory) throws -> File {
        let files = try recordFiles(in: directory)
        let target = recordURL.standardizedFileURL.path
        let others = files.filter { $0.url.standardizedFileURL.path != target }
        guard others.count == files.count - 1,
              let stored = readStored(directory),
              !stored.rows.contains(where: { $0.gameID == summary.gameID }),
              stored.recordCount == others.count,
              stored.newestRecordModified == others.map(\.modified).max() else {
            let rebuilt = try rebuild(directory)
            try write(rebuilt, to: directory)
            return rebuilt
        }
        let updated = File(
            schemaVersion: schemaVersion,
            recordCount: files.count,
            newestRecordModified: files.map(\.modified).max(),
            rows: sorted(stored.rows + [summary])
        )
        try write(updated, to: directory)
        return updated
    }

    /// The stored index if it exists, decodes and has the current schema;
    /// otherwise nil, and the caller rebuilds (the records are the source
    /// of truth). An unreadable cache is logged so a recurring cause shows.
    private static func readStored(_ directory: LichessBotDataDirectory) -> File? {
        guard FileManager.default.fileExists(atPath: directory.indexURL.path) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        let stored: File
        do {
            stored = try decoder.decode(File.self, from: Data(contentsOf: directory.indexURL))
        } catch {
            SessionLogger.shared.log("[LICHESS-BOT] index.json unreadable (\(error.localizedDescription)); rebuilding")
            return nil
        }
        return stored.schemaVersion == schemaVersion ? stored : nil
    }

    static func write(_ file: File, to directory: LichessBotDataDirectory) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        encoder.outputFormatting = [.sortedKeys]
        try LichessBotAtomicWrite.write(encoder.encode(file), to: directory.indexURL)
    }

    private static func sorted(_ rows: [LichessBotGameSummary]) -> [LichessBotGameSummary] {
        rows.sorted { lhs, rhs in
            lhs.createdAt != rhs.createdAt ? lhs.createdAt > rhs.createdAt : lhs.gameID < rhs.gameID
        }
    }
}
