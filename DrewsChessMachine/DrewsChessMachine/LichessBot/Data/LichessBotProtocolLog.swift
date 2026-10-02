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
/// one file per UTC day (plan E50: file names use UTC, so a day's file never
/// changes with the Mac's time zone), kept indefinitely (plan §10.3).
///
/// `record` is synchronous so it can be called from any callback (the
/// request gate's event hook, stream loops); the write happens on the file
/// queue. Every message is passed through `LichessBotRedaction`. A failed
/// write is reported through `onWriteFailure` — the controller raises it as
/// an alarm — and never silently dropped. An entry recorded after the file
/// queue was closed (the controller shut down) is not written; the queue
/// writes the refusal, with the entry's kind and message, to the session log.
///
/// Never synchronized to disk (plan E38 says otherwise; this is the actual
/// behavior): entries still queued when the app crashes are lost, and written
/// ones survive an app crash but not a power loss.
final class LichessBotProtocolLog: Sendable {
    private let directory: LichessBotDataDirectory
    private let fileQueue: LichessBotFileQueue
    private let onWriteFailure: @Sendable (Error) -> Void
    /// Day files whose end this launch has vouched for (its own last append to
    /// them succeeded). Read and changed only inside file-queue closures.
    private let tailVerifiedPaths = SyncBox<Set<String>>([])

    /// The day-file name part, in UTC: a value-type style, so nothing is
    /// allocated per entry.
    private static let dayFileNameStyle = Date.VerbatimFormatStyle(
        format: "\(year: .padded(4))\(month: .twoDigits)\(day: .twoDigits)",
        timeZone: .gmt,
        calendar: Calendar(identifier: .gregorian)
    )

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
        let tailVerifiedPaths = self.tailVerifiedPaths
        fileQueue.enqueue("protocol event (\(entry.kind.rawValue)) \(entry.message)") {
            do {
                var data = Data()
                if !tailVerifiedPaths.value.contains(url.path), FileManager.default.fileExists(atPath: url.path) {
                    let cut = try LichessBotJSONLines.cutUnterminatedFinalLine(of: url)
                    if !cut.isEmpty {
                        SessionLogger.shared.log("[ALARM] LICHESS-BOT \(url.lastPathComponent): cut an unterminated final line of \(cut.count) bytes left by an interrupted write; base64 \(cut.base64EncodedString())")
                        let note = LichessBotProtocolEntry(
                            at: Date(),
                            kind: .anomaly,
                            gameID: nil,
                            message: "cut an unterminated final line left by an interrupted write",
                            fields: ["bytes": "\(cut.count)", "base64": cut.base64EncodedString()]
                        )
                        data.append(try LichessBotJSONLines.encodeLine(note))
                    }
                }
                data.append(try LichessBotJSONLines.encodeLine(entry))
                do {
                    try LichessBotJSONLines.append(data, to: url, synchronize: false)
                } catch {
                    // A failed append may have left part of a line: check the
                    // end again before the next one.
                    tailVerifiedPaths.modify { $0.remove(url.path) }
                    throw error
                }
                tailVerifiedPaths.modify { $0.insert(url.path) }
            } catch {
                onWriteFailure(error)
            }
        }
    }

    /// Wait until every entry recorded so far is on disk. Throws
    /// `LichessBotFileQueueError.closed` once the file queue is closed.
    func flush() async throws {
        try await fileQueue.run {}
    }

    /// The file for the UTC day containing `date`. A local day spans two
    /// files wherever the local zone isn't UTC; a reader that wants a local
    /// day reads both.
    func fileURL(for date: Date) -> URL {
        directory.protocolDirectory.appendingPathComponent("events-\(date.formatted(Self.dayFileNameStyle)).jsonl", isDirectory: false)
    }

    /// Entries for the UTC day containing `date`, oldest first.
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

