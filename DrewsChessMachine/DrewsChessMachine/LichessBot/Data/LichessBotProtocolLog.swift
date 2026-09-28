import Foundation

/// What a protocol-log entry is about, for filtering in the Events view.
enum LichessBotProtocolEventKind: String, Sendable, Codable, CaseIterable {
    /// Stream open, close, stall, reconnect.
    case stream
    /// One Lichess request: category, status, latency.
    case request
    /// A 429 and the cooldown it started.
    case rateLimit
    /// A circuit breaker tripped.
    case breaker
    /// A challenge decision, with its rule.
    case challenge
    /// Token and account checks.
    case account
    /// Model generation snapshots.
    case model
    /// Game lifecycle at account level: started, finished, reconciled.
    case game
    /// The bot going online, draining, offline.
    case lifecycle
    /// Anything unexpected: unknown values, disagreements, failures.
    case anomaly
}

/// One protocol-log line (plan §10.3).
struct LichessBotProtocolEntry: Sendable, Codable, Equatable {
    let at: Date
    let kind: LichessBotProtocolEventKind
    let gameID: String?
    let message: String
    let fields: [String: String]
}

/// The account-level protocol event log: `Protocol/events-YYYYMMDD.jsonl`,
/// one file per local day, kept indefinitely (plan §10.3).
///
/// `record` is synchronous so it can be called from any callback (the
/// request gate's event hook, stream loops); the write happens on the file
/// queue. Every message is passed through `LichessBotRedaction`. A failed
/// write is reported through `onWriteFailure` — the controller raises it as
/// an alarm — and never silently dropped.
final class LichessBotProtocolLog: Sendable {
    private let directory: LichessBotDataDirectory
    private let fileQueue: LichessBotFileQueue
    private let onWriteFailure: @Sendable (Error) -> Void

    init(directory: LichessBotDataDirectory, fileQueue: LichessBotFileQueue, onWriteFailure: @escaping @Sendable (Error) -> Void) {
        self.directory = directory
        self.fileQueue = fileQueue
        self.onWriteFailure = onWriteFailure
    }

    func record(_ kind: LichessBotProtocolEventKind, _ message: String, gameID: String? = nil, fields: [String: String] = [:], at: Date = Date()) {
        let entry = LichessBotProtocolEntry(
            at: at,
            kind: kind,
            gameID: gameID,
            message: LichessBotRedaction.redact(message),
            fields: fields.mapValues(LichessBotRedaction.redact)
        )
        let url = fileURL(for: at)
        let onWriteFailure = self.onWriteFailure
        fileQueue.enqueue {
            do {
                try LichessBotJSONLines.append(try LichessBotJSONLines.encodeLine(entry), to: url, synchronize: false)
            } catch {
                onWriteFailure(error)
            }
        }
    }

    /// Wait until every entry recorded so far is on disk.
    func flush() async throws {
        try await fileQueue.run {}
    }

    func fileURL(for date: Date) -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyyMMdd"
        return directory.protocolDirectory.appendingPathComponent("events-\(formatter.string(from: date)).jsonl", isDirectory: false)
    }

    /// Entries for the local day containing `date`, oldest first.
    func entries(on date: Date) async throws -> LichessBotJSONLines.Decoded<LichessBotProtocolEntry> {
        let url = fileURL(for: date)
        return try await fileQueue.run {
            guard FileManager.default.fileExists(atPath: url.path) else {
                return LichessBotJSONLines.Decoded(elements: [], droppedTrailingByteCount: 0)
            }
            return try LichessBotJSONLines.decode(LichessBotProtocolEntry.self, from: Data(contentsOf: url), fileName: url.lastPathComponent)
        }
    }
}

