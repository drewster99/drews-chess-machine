import Foundation

/// One journal line's payload (plan §10.2). This is a persistence schema,
/// kept separate from the in-memory `LichessBotGameEvent` so the file format
/// changes only deliberately (with `LichessBotJournal.schemaVersion`).
///
/// Raw stream lines are stored as received, as text: they are the game's
/// primary record, and everything else in a finished record is rebuilt from
/// them plus DCM's own decisions.
enum LichessBotJournalEvent: Sendable, Codable, Equatable {
    /// First line of every journal file, and again whenever an app launch
    /// resumes writing to an existing one.
    case header(schemaVersion: Int, gameID: String, build: Int, gitHash: String, resumed: Bool)
    case streamOpened(attempt: Int)
    /// A game-stream line exactly as received (UTF-8 JSON text).
    case streamLine(raw: String)
    /// A stream line that was not valid UTF-8, base64-encoded.
    case streamLineBytes(base64: String)
    case streamEnded(reason: String)
    case positionSynced(kind: LichessBotJournalSyncKind, fromPly: Int, toPly: Int)
    case moveDecided(ply: Int, decision: LichessBotMoveDecision, generation: LichessBotGenerationInfo)
    case movePosted(ply: Int, uci: String, offeringDraw: Bool, milliseconds: Double)
    case moveRejected(ply: Int, uci: String, error: String)
    case action(String)
    case anomaly(String)
    case finished(status: String, winner: String?, localDrawCondition: ChessDrawCondition?)
}

enum LichessBotJournalSyncKind: String, Sendable, Codable, Equatable {
    case extended
    case rebuilt
}

struct LichessBotJournalEntry: Sendable, Codable, Equatable {
    let at: Date
    let event: LichessBotJournalEvent
}

enum LichessBotJournal {
    static let schemaVersion = 1

    /// The journal form of a game-session event, or nil for events the
    /// journal doesn't store because the raw stream already holds them
    /// (`gameInfo` and `chat` are decoded from stream lines; `unchanged`
    /// syncs carry nothing).
    static func event(for gameEvent: LichessBotGameEvent) -> LichessBotJournalEvent? {
        switch gameEvent {
        case .streamOpened(let attempt):
            return .streamOpened(attempt: attempt)
        case .streamLine(let data, _):
            if let text = String(data: data, encoding: .utf8) {
                return .streamLine(raw: text)
            }
            return .streamLineBytes(base64: data.base64EncodedString())
        case .streamEnded(let reason):
            return .streamEnded(reason: reason)
        case .gameInfo, .chat:
            return nil
        case .positionSynced(let sync, _):
            switch sync {
            case .unchanged:
                return nil
            case .extended(let fromPly, let toPly):
                return .positionSynced(kind: .extended, fromPly: fromPly, toPly: toPly)
            case .rebuilt(let fromPly, let toPly):
                return .positionSynced(kind: .rebuilt, fromPly: fromPly, toPly: toPly)
            }
        case .moveDecided(let ply, let decision, let generation):
            return .moveDecided(ply: ply, decision: decision, generation: generation)
        case .movePosted(let ply, let uci, let offeringDraw, let milliseconds):
            return .movePosted(ply: ply, uci: uci, offeringDraw: offeringDraw, milliseconds: milliseconds)
        case .moveRejected(let ply, let uci, let error):
            return .moveRejected(ply: ply, uci: uci, error: error)
        case .action(let text):
            return .action(text)
        case .anomaly(let text):
            return .anomaly(text)
        case .finished(let status, let winner, let localDrawCondition):
            return .finished(status: status.raw, winner: winner?.raw, localDrawCondition: localDrawCondition)
        }
    }

    /// The receive time for stream lines (captured when the bytes arrived),
    /// otherwise now.
    static func timestamp(for gameEvent: LichessBotGameEvent) -> Date {
        if case .streamLine(_, let receivedAt) = gameEvent {
            return receivedAt
        }
        return Date()
    }

    /// Read a journal file. An unterminated final line — a crash mid-append
    /// — is dropped and its size reported.
    static func read(_ url: URL) throws -> LichessBotJSONLines.Decoded<LichessBotJournalEntry> {
        try LichessBotJSONLines.decode(LichessBotJournalEntry.self, from: Data(contentsOf: url), fileName: url.lastPathComponent)
    }
}

/// Writes game journals into `InProgress/` (plan §10.2): it is the game
/// observer that turns session events into journal lines.
///
/// Each append hops onto the file queue and back, so a game session never
/// does file I/O on a cooperative-pool thread, and events land in the order
/// the session reported them. The file is synchronized to disk at the end
/// of each move cycle (after a move is posted) and when the game finishes.
/// The first time a game is seen by this app launch, a header is written —
/// marked `resumed` when the journal already existed (plan §10.2 launch
/// recovery).
///
/// Write failures are reported through `onWriteFailure` (an alarm), never
/// dropped silently; the game keeps playing either way.
final class LichessBotJournalWriter: LichessBotGameObserver {
    private let directory: LichessBotDataDirectory
    private let fileQueue: LichessBotFileQueue
    private let onWriteFailure: @Sendable (String, Error) -> Void
    private let onGameFinished: @Sendable (String) async -> Void
    private let headerWritten = SyncBox<Set<String>>([])

    init(
        directory: LichessBotDataDirectory,
        fileQueue: LichessBotFileQueue,
        onWriteFailure: @escaping @Sendable (String, Error) -> Void,
        onGameFinished: @escaping @Sendable (String) async -> Void
    ) {
        self.directory = directory
        self.fileQueue = fileQueue
        self.onWriteFailure = onWriteFailure
        self.onGameFinished = onGameFinished
    }

    func gameEvent(gameID: String, _ event: LichessBotGameEvent) async {
        guard let journalEvent = LichessBotJournal.event(for: event) else { return }
        let entry = LichessBotJournalEntry(at: LichessBotJournal.timestamp(for: event), event: journalEvent)
        let isFinish: Bool
        let synchronize: Bool
        switch journalEvent {
        case .finished:
            isFinish = true
            synchronize = true
        case .movePosted:
            isFinish = false
            synchronize = true
        default:
            isFinish = false
            synchronize = false
        }
        do {
            try await append([entry], gameID: gameID, synchronize: synchronize)
        } catch {
            onWriteFailure(gameID, error)
        }
        if isFinish {
            await onGameFinished(gameID)
        }
    }

    /// Append entries (with a header first if this launch hasn't written
    /// one for the game yet).
    func append(_ entries: [LichessBotJournalEntry], gameID: String, synchronize: Bool) async throws {
        let url = directory.inProgressJournalURL(gameID: gameID)
        let needsHeader = headerWritten.mutate { written -> Bool in
            written.insert(gameID).inserted
        }
        let headerBox = headerWritten
        do {
            try await fileQueue.run {
                var data = Data()
                if needsHeader {
                    let resumed = FileManager.default.fileExists(atPath: url.path)
                    let header = LichessBotJournalEntry(
                        at: Date(),
                        event: .header(
                            schemaVersion: LichessBotJournal.schemaVersion,
                            gameID: gameID,
                            build: BuildInfo.buildNumber,
                            gitHash: BuildInfo.gitHash,
                            resumed: resumed
                        )
                    )
                    data.append(try LichessBotJSONLines.encodeLine(header))
                }
                for entry in entries {
                    data.append(try LichessBotJSONLines.encodeLine(entry))
                }
                try LichessBotJSONLines.append(data, to: url, synchronize: synchronize)
            }
        } catch {
            if needsHeader {
                // Nothing reached the file; write the header next time.
                headerBox.modify { $0.remove(gameID) }
            }
            throw error
        }
    }

    /// Forget which games have headers, so the next event for `gameID`
    /// writes one. Called when a journal is moved out of `InProgress/`.
    func forget(gameID: String) {
        headerWritten.modify { $0.remove(gameID) }
    }
}

/// Delivers each game event to several observers, in order.
struct LichessBotGameObserverFanOut: LichessBotGameObserver {
    let observers: [any LichessBotGameObserver]

    func gameEvent(gameID: String, _ event: LichessBotGameEvent) async {
        for observer in observers {
            await observer.gameEvent(gameID: gameID, event)
        }
    }
}
