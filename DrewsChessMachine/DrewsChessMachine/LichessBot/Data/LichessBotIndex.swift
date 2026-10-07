import CryptoKit
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
    /// The per-game facts the Record card's statistics read
    /// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §4.2), reduced from the record
    /// once, here, so the statistics never open a record file. Optional so
    /// a row without it decodes; an index of an older schema is rejected and
    /// rebuilt anyway, so nil occurs only in a hand-written test fixture,
    /// and the statistics count such a row as "no move data".
    let facts: LichessBotGameFacts?

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
        facts = LichessBotGameFacts(record: record)
    }
}

/// `index.json`: a derived cache of `LichessBotGameSummary` rows, never
/// authoritative (plan §10.5). It records a signature of the record files it
/// was built from (every path, size and modification time); when that no
/// longer matches the `Games/` folder, it is rebuilt from the records. Every
/// function here runs on a `LichessBotFileQueue` — the controller's
/// general-purpose one, never the journal queue, since a rebuild decodes
/// every record.
enum LichessBotIndex {
    /// 3: rows carry `facts` (`LICHESS_BOT_RECORD_STATS_PLAN.md` §4.2).
    /// Bump it again whenever `LichessBotSelfAssessmentDefinition`'s
    /// thresholds change: the facts are reduced with them, so rows of the
    /// old thresholds must not be kept. A stored index of another version is
    /// rejected and rebuilt from the records.
    static let schemaVersion = 3

    /// A record file the index left out because it doesn't decode.
    struct UnreadableRecord: Sendable, Codable, Equatable {
        let path: String
        let error: String
    }

    struct File: Sendable, Codable, Equatable {
        let schemaVersion: Int
        let recordCount: Int
        /// Digest of every record file's path, size and modification time:
        /// any record added, removed, replaced or touched changes it, so a
        /// stale index is always noticed.
        let recordSignature: String
        /// Newest game first.
        let rows: [LichessBotGameSummary]
        /// Record files that don't decode, left out of `rows` (see `rebuild`).
        let unreadableRecords: [UnreadableRecord]
    }

    struct RecordFile {
        let url: URL
        let size: Int
        let modified: Date
    }

    /// Every `*.json` record file under `Games/` (journals and PGNs
    /// excluded), with its size and modification time.
    static func recordFiles(in directory: LichessBotDataDirectory) throws -> [RecordFile] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.gamesDirectory.path) else {
            return []
        }
        let keys: [URLResourceKey] = [.contentModificationDateKey, .isRegularFileKey, .fileSizeKey]
        guard let enumerator = fm.enumerator(
            at: directory.gamesDirectory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else {
            throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: directory.gamesDirectory.path])
        }
        var files: [RecordFile] = []
        for case let url as URL in enumerator where url.pathExtension == "json" {
            let values = try url.resourceValues(forKeys: Set(keys))
            guard values.isRegularFile == true else { continue }
            guard let modified = values.contentModificationDate else {
                throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: url.path, NSLocalizedDescriptionKey: "no modification date for \(url.lastPathComponent)"])
            }
            guard let size = values.fileSize else {
                throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: url.path, NSLocalizedDescriptionKey: "no size for \(url.lastPathComponent)"])
            }
            files.append(RecordFile(url: url, size: size, modified: modified))
        }
        return files
    }

    /// A cheap signature of the record files: one stat each (already taken by
    /// `recordFiles`), no reads. The modification time goes in by bit
    /// pattern, so it compares exactly.
    static func signature(of files: [RecordFile]) -> String {
        var hasher = SHA256()
        for file in files.sorted(by: { $0.url.path < $1.url.path }) {
            hasher.update(data: Data("\(file.url.path)\t\(file.size)\t\(file.modified.timeIntervalSince1970.bitPattern)\n".utf8))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func readRecord(at url: URL) throws -> LichessBotGameRecord {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        return try decoder.decode(LichessBotGameRecord.self, from: Data(contentsOf: url))
    }

    /// Build the index from the record files. A record that doesn't decode
    /// is left out and reported, never thrown: one bad file must not stop the
    /// index — and with it the filing of every later game — from working.
    ///
    /// Every rebuild logs how long it took and why it ran
    /// (`[LICHESS-BOT] index rebuilt: <n> records in <s> s (<reason>)`):
    /// it decodes every record on the controller's general file queue,
    /// which also carries the Keychain, the instance lock and the protocol
    /// log, so its real cost at scale must be visible
    /// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §4.2).
    static func rebuild(_ directory: LichessBotDataDirectory, reason: String) throws -> File {
        let started = DispatchTime.now().uptimeNanoseconds
        let files = try recordFiles(in: directory)
        var rows: [LichessBotGameSummary] = []
        var unreadable: [UnreadableRecord] = []
        for file in files {
            do {
                rows.append(LichessBotGameSummary(record: try readRecord(at: file.url)))
            } catch {
                unreadable.append(UnreadableRecord(path: file.url.path, error: String(describing: error)))
            }
        }
        if !unreadable.isEmpty {
            SessionLogger.shared.log("[ALARM] LICHESS-BOT index: \(unreadable.count) record file(s) don't decode and are left out: \(unreadable.map(\.path).joined(separator: ", "))")
        }
        SessionLogger.shared.log("[LICHESS-BOT] index rebuilt: \(files.count) records in \(seconds(since: started)) s (\(reason))")
        return File(
            schemaVersion: schemaVersion,
            recordCount: files.count,
            recordSignature: signature(of: files),
            rows: sorted(rows),
            unreadableRecords: unreadable.sorted { $0.path < $1.path }
        )
    }

    /// The index, rebuilt and rewritten first if it is missing, stale or
    /// from another schema.
    static func load(_ directory: LichessBotDataDirectory) throws -> File {
        let started = DispatchTime.now().uptimeNanoseconds
        let files = try recordFiles(in: directory)
        let reason: String
        switch readStored(directory) {
        case .usable(let stored):
            if stored.recordCount == files.count, stored.recordSignature == signature(of: files) {
                SessionLogger.shared.log("[LICHESS-BOT] index loaded: \(stored.rows.count) rows in \(seconds(since: started)) s")
                return stored
            }
            reason = "stale: the record files changed"
        case .missing:
            reason = "no index"
        case .unreadable(let error):
            reason = "unreadable: \(error)"
        case .otherSchema(let version):
            reason = "schema \(version), now \(schemaVersion)"
        }
        let rebuilt = try rebuild(directory, reason: reason)
        try write(rebuilt, to: directory)
        return rebuilt
    }

    /// Add one game's row after its record file (`recordURL`) was written.
    /// Incremental when the stored index exactly covered every *other*
    /// record; otherwise (a stale index, or a record written again) a full
    /// rebuild — either way the result equals `rebuild`.
    static func upsert(_ summary: LichessBotGameSummary, recordURL: URL, in directory: LichessBotDataDirectory) throws -> File {
        let started = DispatchTime.now().uptimeNanoseconds
        let files = try recordFiles(in: directory)
        let target = recordURL.standardizedFileURL.path
        let others = files.filter { $0.url.standardizedFileURL.path != target }
        guard others.count == files.count - 1,
              case .usable(let stored) = readStored(directory),
              !stored.rows.contains(where: { $0.gameID == summary.gameID }),
              stored.recordCount == others.count,
              stored.recordSignature == signature(of: others) else {
            let rebuilt = try rebuild(directory, reason: "filing \(summary.gameID): the stored index does not cover the other records exactly")
            try write(rebuilt, to: directory)
            return rebuilt
        }
        let updated = File(
            schemaVersion: schemaVersion,
            recordCount: files.count,
            recordSignature: signature(of: files),
            rows: sorted(stored.rows + [summary]),
            unreadableRecords: stored.unreadableRecords
        )
        try write(updated, to: directory)
        SessionLogger.shared.log("[LICHESS-BOT] index updated: \(summary.gameID) added, \(updated.rows.count) rows in \(seconds(since: started)) s")
        return updated
    }

    /// What `index.json` holds, as `load` and `upsert` judge it.
    private enum StoredIndex {
        case usable(File)
        case missing
        case unreadable(String)
        case otherSchema(Int)
    }

    /// The stored index if it exists, decodes and has the current schema;
    /// otherwise why not, and the caller rebuilds (the records are the
    /// source of truth). An unreadable cache is logged so a recurring cause
    /// shows. An index of an older schema still decodes (every field added
    /// since is optional) and is then rejected by its version.
    private static func readStored(_ directory: LichessBotDataDirectory) -> StoredIndex {
        guard FileManager.default.fileExists(atPath: directory.indexURL.path) else {
            return .missing
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        let stored: File
        do {
            stored = try decoder.decode(File.self, from: Data(contentsOf: directory.indexURL))
        } catch {
            SessionLogger.shared.log("[LICHESS-BOT] index.json unreadable (\(error.localizedDescription)); rebuilding")
            return .unreadable(error.localizedDescription)
        }
        return stored.schemaVersion == schemaVersion ? .usable(stored) : .otherSchema(stored.schemaVersion)
    }

    /// Seconds since `started` (a `DispatchTime` uptime), three decimals.
    private static func seconds(since started: UInt64) -> String {
        String(format: "%.3f", Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000)
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
