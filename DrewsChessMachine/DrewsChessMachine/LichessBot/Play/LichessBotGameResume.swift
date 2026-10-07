import Foundation

/// What a game session carries over from earlier sessions of the same game:
/// the per-game allowances and streaks that must not start over when the
/// app is relaunched (or goes offline and back) while the game is live.
/// Folded from the game's journal, the single record of what DCM did.
struct LichessBotGameSessionCarryover: Sendable, Equatable {
    /// The greeting was sent: a resumed session must not greet again.
    var greeted: Bool
    /// The goodbye was sent: a resumed session must not say it again.
    var farewellSent: Bool
    /// Takebacks DCM accepted in this game (the per-game allowance).
    var takebacksAccepted: Int
    /// Chat-command replies DCM decided to send in this game (the per-game
    /// reply budget, counted when decided, as the session counts them).
    var commandRepliesQueued: Int
    /// DCM's value readings, one per ply it decided at, oldest first: what
    /// the resign and draw-offer streaks count.
    var readings: [LichessBotValueReading]

    /// A game this runtime is the first to see.
    static let newGame = LichessBotGameSessionCarryover(greeted: false, farewellSent: false, takebacksAccepted: 0, commandRepliesQueued: 0, readings: [])

    /// A resumed game whose journal can't be read: nothing optional is done
    /// again, since there is no telling whether it already was. No greeting
    /// or goodbye, no takeback accepted, no command answered; the streaks
    /// start empty (they only ever delay a resignation or a draw offer).
    static let unknownHistory = LichessBotGameSessionCarryover(
        greeted: true,
        farewellSent: true,
        takebacksAccepted: Int.max,
        commandRepliesQueued: LichessBotChatCommandBudget.maximumRepliesPerGame,
        readings: []
    )

    /// The carryover the journal's entries add up to.
    ///
    /// Readings follow the session's own rule: a decision at a ply replaces
    /// any reading at that ply or later, and a rebuilt position (a takeback)
    /// drops every reading at or after the ply it went back to.
    static func fold(_ entries: [LichessBotJournalEntry]) -> LichessBotGameSessionCarryover {
        var carryover = newGame
        for entry in entries {
            switch entry.event {
            case .moveDecided(let ply, let decision, _):
                carryover.readings.removeAll { $0.ply >= ply }
                carryover.readings.append(LichessBotValueReading(ply: ply, decision: decision))
            case .positionSynced(let kind, _, let toPly):
                if kind == .rebuilt {
                    carryover.readings.removeAll { $0.ply >= toPly }
                }
            case .takebackAccepted:
                carryover.takebacksAccepted += 1
            case .commandReplyQueued:
                carryover.commandRepliesQueued += 1
            case .chatSent(_, _, let origin):
                if origin == LichessBotChatOrigin.greeting.rawValue {
                    carryover.greeted = true
                } else if origin == LichessBotChatOrigin.goodbye.rawValue {
                    carryover.farewellSent = true
                }
            case .header, .streamOpened, .streamLine, .streamLineBytes, .keepAlive, .request, .streamEnded,
                 .movePosted, .moveRejected, .action, .chatFetched, .anomaly, .finished, .gameOrigin:
                break
            }
        }
        return carryover
    }
}

extension LichessBotValueReading {
    /// The reading a move decision gives for its ply. The session and the
    /// journal fold both build readings here.
    init(ply: Int, decision: LichessBotMoveDecision) {
        self.init(ply: ply, win: decision.win, draw: decision.draw, loss: decision.loss)
    }
}

extension LichessBotGameFull {
    /// The color `accountID` plays, or nil if neither player is that account.
    func color(of accountID: String) -> LichessBotColorName? {
        if white.id == accountID {
            return .white
        }
        if black.id == accountID {
            return .black
        }
        return nil
    }
}

/// Why a game's leftover journal can't be used to resume it.
enum LichessBotResumeError: Error, LocalizedError, Equatable {
    case noHeader(gameID: String)
    case headerForAnotherGame(gameID: String, headerGameID: String)
    case newerSchema(gameID: String, schemaVersion: Int)
    case notOurGame(gameID: String)

    var errorDescription: String? {
        switch self {
        case .noHeader(let gameID):
            return "game \(gameID)'s journal doesn't start with a header"
        case .headerForAnotherGame(let gameID, let headerGameID):
            return "game \(gameID)'s journal belongs to game \(headerGameID)"
        case .newerSchema(let gameID, let schemaVersion):
            return "game \(gameID)'s journal was written by a newer build (schema \(schemaVersion), this build reads up to \(LichessBotJournal.schemaVersion))"
        case .notOurGame(let gameID):
            return "game \(gameID)'s journal shows a game DCM's account doesn't play"
        }
    }
}

/// A game's leftover journal, checked and prepared for resuming: the
/// session's carryover and the live view's history. Built off the main
/// actor (the stream lines are decoded here), so replaying it into the live
/// view is cheap.
struct LichessBotResumedJournal: Sendable {
    /// One journal entry, its stream line decoded when it carries one.
    struct Item: Sendable {
        let entry: LichessBotJournalEntry
        /// For a `streamLine` entry: the decoded line, or why it doesn't
        /// decode. Nil for every other entry.
        let decodedLine: Result<LichessBotGameStreamLine, LichessBotStreamLineDecodeFailure>?
    }

    let gameID: String
    let items: [Item]
    /// Bytes of an unterminated final line dropped when the journal was read
    /// (an interrupted write).
    let droppedTrailingByteCount: Int
    /// When the journal was started: when this runtime's predecessor first
    /// saw the game.
    let firstJournaledAt: Date
    /// When the journal's last entry was written: roughly when DCM stopped
    /// following the game.
    let lastJournaledAt: Date
    let carryover: LichessBotGameSessionCarryover
    /// How the game began, as the journal records it
    /// (`LichessBotGameOrigin.recorded(from:)`): nil when it holds none — a
    /// journal from before origins were recorded, or a game whose origin
    /// was still unknown.
    let recordedOrigin: LichessBotGameOrigin?

    /// Check a leftover journal and prepare it, or refuse it with a reason.
    static func make(gameID: String, journal: LichessBotJSONLines.Decoded<LichessBotJournalEntry>, ourAccountID: String) throws -> LichessBotResumedJournal {
        guard let first = journal.elements.first, case .header(let schemaVersion, let headerGameID, _, _, _) = first.event else {
            throw LichessBotResumeError.noHeader(gameID: gameID)
        }
        guard headerGameID == gameID else {
            throw LichessBotResumeError.headerForAnotherGame(gameID: gameID, headerGameID: headerGameID)
        }
        guard schemaVersion <= LichessBotJournal.schemaVersion else {
            throw LichessBotResumeError.newerSchema(gameID: gameID, schemaVersion: schemaVersion)
        }
        var items: [Item] = []
        items.reserveCapacity(journal.elements.count)
        var checkedPlayers = false
        for entry in journal.elements {
            guard case .streamLine(let raw) = entry.event else {
                items.append(Item(entry: entry, decodedLine: nil))
                continue
            }
            do {
                let line = try LichessBotGameStreamLine.decode(Data(raw.utf8))
                if !checkedPlayers, case .gameFull(let full) = line {
                    checkedPlayers = true
                    guard full.color(of: ourAccountID) != nil else {
                        throw LichessBotResumeError.notOurGame(gameID: gameID)
                    }
                }
                items.append(Item(entry: entry, decodedLine: .success(line)))
            } catch let error as LichessBotResumeError {
                throw error
            } catch {
                items.append(Item(entry: entry, decodedLine: .failure(LichessBotStreamLineDecodeFailure(description: String(describing: error)))))
            }
        }
        return LichessBotResumedJournal(
            gameID: gameID,
            items: items,
            droppedTrailingByteCount: journal.droppedTrailingByteCount,
            firstJournaledAt: first.at,
            // Not empty: it starts with the header checked above.
            lastJournaledAt: journal.elements[journal.elements.count - 1].at,
            carryover: .fold(journal.elements),
            recordedOrigin: LichessBotGameOrigin.recorded(from: journal.elements.compactMap { entry in
                if case .gameOrigin(let origin) = entry.event { return origin }
                return nil
            }).origin
        )
    }
}

/// Why a journaled stream line didn't decode.
struct LichessBotStreamLineDecodeFailure: Error, Sendable, Equatable {
    let description: String
}

/// How a game session came to start.
enum LichessBotSessionOrigin: Sendable {
    /// A game this runtime is the first to see.
    case new
    /// A game with a usable journal from before this runtime.
    case resumed(LichessBotResumedJournal)
    /// A game with a journal from before this runtime that can't be used;
    /// `journalStartedAt` is when that journal file was created.
    case resumedWithUnreadableJournal(journalStartedAt: Date, reason: String)
}
