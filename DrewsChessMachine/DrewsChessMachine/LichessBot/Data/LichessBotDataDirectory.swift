import Foundation

/// The on-disk layout of the Lichess bot's data (plan §10.1):
///
/// ```
/// LichessBot/
///   Games/YYYY/MM/<YYYYMMDD-HHMMSS>-<gameId>.json           final record
///   Games/YYYY/MM/<YYYYMMDD-HHMMSS>-<gameId>.pgn            PGN
///   Games/YYYY/MM/<YYYYMMDD-HHMMSS>-<gameId>.journal.jsonl  raw journal, kept
///   InProgress/<gameId>.journal.jsonl                       journal while live
///   Protocol/events-YYYYMMDD.jsonl                          protocol event log, one file per UTC day
///   index.json                                              derived stats cache
///   player-notes.json                                       favorites, bot limit times
///   bot.lock                                                instance lock
/// ```
///
/// A value, not a singleton: production uses `standard`, under the app's one
/// root (`CheckpointPaths.rootURL`); tests point it at a temporary folder.
struct LichessBotDataDirectory: Sendable, Equatable {
    let root: URL

    static var standard: LichessBotDataDirectory {
        LichessBotDataDirectory(root: CheckpointPaths.rootURL.appendingPathComponent("LichessBot", isDirectory: true))
    }

    var gamesDirectory: URL { root.appendingPathComponent("Games", isDirectory: true) }
    var inProgressDirectory: URL { root.appendingPathComponent("InProgress", isDirectory: true) }
    var protocolDirectory: URL { root.appendingPathComponent("Protocol", isDirectory: true) }
    var indexURL: URL { root.appendingPathComponent("index.json", isDirectory: false) }
    var lockURL: URL { root.appendingPathComponent("bot.lock", isDirectory: false) }
    var playerNotesURL: URL { root.appendingPathComponent("player-notes.json", isDirectory: false) }

    static let journalExtension = "journal.jsonl"

    func inProgressJournalURL(gameID: String) -> URL {
        inProgressDirectory.appendingPathComponent("\(gameID).\(Self.journalExtension)", isDirectory: false)
    }

    /// `Games/YYYY/MM/` for a game created at `createdAt`, in UTC so a game's
    /// files never move when the Mac's time zone changes (plan E50).
    func gamesMonthDirectory(createdAt: Date) -> URL {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        let year = String(format: "%04d", calendar.component(.year, from: createdAt))
        let month = String(format: "%02d", calendar.component(.month, from: createdAt))
        return gamesDirectory
            .appendingPathComponent(year, isDirectory: true)
            .appendingPathComponent(month, isDirectory: true)
    }

    /// `<YYYYMMDD-HHMMSS>-<gameId>`, from the game's creation time in UTC
    /// (plan E50).
    static func fileStem(gameID: String, createdAt: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = .gmt
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "\(formatter.string(from: createdAt))-\(gameID)"
    }

    func createDirectories() throws {
        let fm = FileManager.default
        for directory in [root, gamesDirectory, inProgressDirectory, protocolDirectory] {
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }
}

/// A serial work executor for Lichess bot file operations (plan §10.2,
/// §15). File I/O never runs on a cooperative-pool thread: callers hop on
/// through a continuation and come back with the result or the error. The
/// controller runs two: one for game journals and filing — the play path,
/// which every game-stream line waits on — and one for everything else
/// (index, protocol log, player notes, Keychain, lock), so nothing slow sits
/// in front of a journal append. Each queue orders its own work; work that
/// must stay ordered shares a queue.
final class LichessBotFileQueue: Sendable {
    private let queue: DispatchQueue

    init(label: String, qos: DispatchQoS) {
        queue = DispatchQueue(label: label, qos: qos)
    }

    /// The general-purpose queue: everything except journals and filing.
    convenience init() {
        self.init(label: "drewschess.lichessbot.files", qos: .utility)
    }

    func run<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result { try body() })
            }
        }
    }

    /// Enqueue without waiting, for callers that can't await (the protocol
    /// log is written from synchronous callbacks). `body` must handle its
    /// own errors.
    func enqueue(_ body: @escaping @Sendable () -> Void) {
        queue.async(execute: body)
    }
}

/// Atomic file replacement: write a sibling temporary file, then rename it
/// over the destination, so a crash never leaves a partial file under the
/// final name (plan §10.2).
enum LichessBotAtomicWrite {
    static func write(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
}

/// Removes secrets from anything about to be logged (plan §10.3). The token
/// is never passed to a logger on purpose; this is the backstop for text
/// that might embed one (an error description echoing a header, a URL).
enum LichessBotRedaction {
    private static let patterns: [NSRegularExpression] = {
        sources.map(\.pattern).map { source in
            do {
                return try NSRegularExpression(pattern: source)
            } catch {
                preconditionFailure("invalid redaction pattern \(source): \(error)")
            }
        }
    }()

    /// Each secret's pattern and what replaces it.
    private static let sources: [(pattern: String, template: String)] = [
        (#"lip_[A-Za-z0-9_-]+"#, "[REDACTED]"),
        (#"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"#, "[REDACTED]"),
        // `gameStart.game.fullId` embeds the player's secret suffix (E26).
        (#""fullId"\s*:\s*"[^"]*""#, "\"fullId\":\"[REDACTED]\""),
    ]

    static func redact(_ text: String) -> String {
        var result = text
        for (pattern, source) in zip(patterns, sources) {
            let range = NSRange(result.startIndex..<result.endIndex, in: result)
            result = pattern.stringByReplacingMatches(in: result, range: range, withTemplate: source.template)
        }
        return result
    }
}
