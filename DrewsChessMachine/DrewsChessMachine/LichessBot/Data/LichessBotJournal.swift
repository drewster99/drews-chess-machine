import Foundation

/// One journal line's payload (plan §10.2). This is a persistence schema,
/// kept separate from the in-memory `LichessBotGameEvent` so the file format
/// changes only deliberately.
///
/// Cases are added without changing `LichessBotJournal.schemaVersion`, which
/// is written into every header but checked only when a journal is resumed
/// (a newer version is refused there). What actually decides compatibility
/// is the decoder: a build that doesn't know a case can't decode a journal
/// holding it, so a journal written by a newer build can't be filed by an
/// older one.
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
    /// A keep-alive on the game stream (the entry's time is its receive time).
    case keepAlive
    /// A request DCM made for this game (plan §14.3a transcript).
    case request(LichessBotRequestRecord)
    case streamEnded(reason: String)
    case positionSynced(kind: LichessBotJournalSyncKind, fromPly: Int, toPly: Int)
    case moveDecided(ply: Int, decision: LichessBotMoveDecision, generation: LichessBotGenerationInfo)
    case movePosted(ply: Int, uci: String, offeringDraw: Bool, milliseconds: Double)
    case moveRejected(ply: Int, uci: String, error: String)
    case action(String)
    /// A chat message DCM posted successfully (see `LichessBotGameEvent.chatSent`).
    case chatSent(room: String, text: String, origin: String)
    /// A player-room message fetched after the game (see
    /// `LichessBotGameEvent.chatFetched`); the entry's time is the fetch.
    case chatFetched(username: String, text: String)
    case anomaly(String)
    case finished(status: String, winner: String?, localDrawCondition: ChessDrawCondition?)
    /// DCM accepted the opponent's takeback (see
    /// `LichessBotGameEvent.takebackAccepted`).
    case takebackAccepted
    /// DCM decided to answer a chat command (see
    /// `LichessBotGameEvent.commandReplyQueued`).
    case commandReplyQueued(command: String, username: String, room: String)
    /// How the game began (challenge-log plan §3.5), written by the
    /// controller once it is known, or as undetermined at session end. A
    /// build from before this case can't decode a journal holding it (the
    /// rule above), the documented cost of downgrading mid-game.
    case gameOrigin(LichessBotGameOrigin)
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
        case .keepAlive:
            return .keepAlive
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
        case .moveHeld(let ply, let uci, _):
            return .action("holding move \(uci) at ply \(ply) (operator delay or hold)")
        case .moveReleased(let ply, let reason):
            return .action("released the held move at ply \(ply): \(reason)")
        case .chatSent(let room, let text, let origin):
            return .chatSent(room: room.rawValue, text: text, origin: origin.rawValue)
        case .chatFetched(let username, let text):
            return .chatFetched(username: username, text: text)
        case .anomaly(let text):
            return .anomaly(text)
        case .stoppedMoving(let reason):
            return .anomaly("stopped moving: \(reason)")
        case .tokenRejected(let detail):
            return .anomaly("token rejected: \(detail)")
        case .finished(let status, let winner, let localDrawCondition):
            return .finished(status: status.raw, winner: winner?.raw, localDrawCondition: localDrawCondition)
        case .takebackAccepted:
            return .takebackAccepted
        case .commandReplyQueued(let command, let username, let room):
            return .commandReplyQueued(command: command.rawValue, username: username, room: room.rawValue)
        }
    }

    /// The receive time for stream lines (captured when the bytes arrived),
    /// otherwise now.
    static func timestamp(for gameEvent: LichessBotGameEvent) -> Date {
        switch gameEvent {
        case .streamLine(_, let receivedAt), .keepAlive(let receivedAt):
            return receivedAt
        default:
            return Date()
        }
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
/// the session reported them. The first time a game is seen by this app
/// launch, a header is written — marked `resumed` when the journal already
/// existed (plan §10.2 launch recovery).
///
/// The file is synchronized to disk after each posted move and at the
/// finish, and at no other time — not per anomaly, although plan E38 says
/// so. Every other line is written to the file before the session
/// continues, so it survives an app crash (it is in the kernel's cache);
/// only a kernel panic or power loss can lose lines written since the last
/// synchronize (`fsync`, not `F_FULLFSYNC`).
///
/// Every append goes through `LichessBotJSONLines.append`, so a symbolic
/// link, folder or FIFO at a journal's path is refused and left untouched
/// (an `onWriteFailure`), never written through.
///
/// Write failures are reported through `onWriteFailure` (an alarm), never
/// dropped silently; the game keeps playing either way.
final class LichessBotJournalWriter: LichessBotGameObserver {
    private let directory: LichessBotDataDirectory
    private let fileQueue: LichessBotFileQueue
    private let onWriteFailure: @Sendable (String, Error) -> Void
    private let onGameFinished: @Sendable (String) async -> Void
    /// Games this launch has written a header for. Read and changed only
    /// inside file-queue closures, so each append's decision and write are
    /// one serial step.
    private let headerWritten = SyncBox<Set<String>>([])
    /// Games whose journal has been filed; nothing more is written for them.
    private let finalized = SyncBox<Set<String>>([])

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
        guard !finalized.value.contains(gameID),
              let journalEvent = LichessBotJournal.event(for: event) else { return }
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
    /// one for the game yet). The header decision and the append (which
    /// checks the file's end, and cuts and records a fragment an interrupted
    /// write left) happen inside one file-queue closure, so concurrent
    /// appends (a session's events and the controller's request records)
    /// can't reorder a headerless write ahead of the one carrying the header.
    ///
    /// `synchronize` maps to `LichessBotJSONLines.Synchronization`: `true` is
    /// `.fsync` (posted moves and the finish), `false` is `.none`.
    func append(_ entries: [LichessBotJournalEntry], gameID: String, synchronize: Bool) async throws {
        let url = try directory.validatedInProgressJournalURL(gameID: gameID)
        let headerWritten = self.headerWritten
        let finalized = self.finalized
        let synchronization: LichessBotJSONLines.Synchronization = synchronize ? .fsync : .none
        try await fileQueue.run {
            let fileExists = try FileSafety.existingItem(at: url) != nil
            let needsHeader = !headerWritten.value.contains(gameID)
            if !fileExists && (!needsHeader || finalized.value.contains(gameID)) {
                // This launch wrote the journal, or has filed the game, and
                // the file is gone: the game was filed. A late write must not
                // start a fragment in InProgress/.
                SessionLogger.shared.log("[LICHESS-BOT] game \(gameID): \(entries.count) late journal entr(ies) not written: the game is filed")
                return
            }
            let header = needsHeader
                ? try LichessBotJSONLines.encodeLine(LichessBotJournalEntry(
                    at: Date(),
                    event: .header(
                        schemaVersion: LichessBotJournal.schemaVersion,
                        gameID: gameID,
                        build: BuildInfo.buildNumber,
                        gitHash: BuildInfo.gitHash,
                        resumed: fileExists
                    )
                ))
                : Data()
            try LichessBotJSONLines.append(to: url, synchronization: synchronization, systemCalls: .system) { cut in
                var data = header
                if !cut.isEmpty {
                    let note = "cut an unterminated final line of \(cut.count) bytes left by an interrupted write; base64 \(cut.base64EncodedString())"
                    SessionLogger.shared.log("[ALARM] LICHESS-BOT game \(gameID) journal: \(note)")
                    data.append(try LichessBotJSONLines.encodeLine(LichessBotJournalEntry(at: Date(), event: .anomaly(note))))
                }
                for entry in entries {
                    data.append(try LichessBotJSONLines.encodeLine(entry))
                }
                return data
            }
            // A failed append leaves `headerWritten` alone, so the next one
            // writes a header again (the failed one may have carried it).
            if needsHeader {
                headerWritten.modify { $0.insert(gameID) }
            }
        }
    }

    /// Record a request DCM made for a game, into that game's journal. A
    /// request for a game already filed is not written (the protocol log
    /// still has it).
    func recordRequest(_ record: LichessBotRequestRecord) async {
        guard let gameID = record.gameID, !finalized.value.contains(gameID) else { return }
        do {
            try await append([LichessBotJournalEntry(at: record.startedAt, event: .request(record))], gameID: gameID, synchronize: false)
        } catch {
            onWriteFailure(gameID, error)
        }
    }

    /// Record how the game began, into its journal, synchronized like a
    /// posted move. A game already filed gets nothing (logged by `append`):
    /// its origin then shows through the challenge log instead.
    func recordOrigin(_ origin: LichessBotGameOrigin, gameID: String) async {
        guard !finalized.value.contains(gameID) else {
            SessionLogger.shared.log("[LICHESS-BOT] game \(gameID): origin \(origin.token) not written to the journal: the game is filed")
            return
        }
        do {
            try await append([LichessBotJournalEntry(at: Date(), event: .gameOrigin(origin))], gameID: gameID, synchronize: true)
        } catch {
            onWriteFailure(gameID, error)
        }
    }

    /// The game's journal has been moved out of `InProgress/`: write nothing
    /// more for it. The game stays in `headerWritten` on purpose: a write
    /// already past the `finalized` check then finds "headered, file gone" on
    /// the queue and skips, instead of starting a new fragment.
    func markFinalized(gameID: String) {
        finalized.modify { $0.insert(gameID) }
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
