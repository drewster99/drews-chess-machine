import Foundation

/// Records store colors directly (not through `LichessBotOpenValue`): a
/// record is written by DCM, so its colors are always known values.
extension LichessBotColorName: Codable {}

/// Who an opponent is, for the Stats slices (plan §11, E24).
enum LichessBotOpponentKind: String, Sendable, Codable, Equatable, CaseIterable {
    case human
    case bot
    /// Lichess's own engine (`aiLevel` set, no account, no rating).
    case lichessAI
}

/// A finished game: the single source of truth for everything the bot
/// shows about it (plan §10.6). Written when the game is filed, after
/// reconciliation with Lichess's export, and rewritten only if the game is
/// filed again: after a crash before its journal left InProgress/, or when a
/// later journal for the game arrives (then rebuilt from every journal kept
/// for it). Its journals are kept beside it.
struct LichessBotGameRecord: Sendable, Codable, Equatable {
    static let currentSchemaVersion = 1

    struct Setup: Sendable, Codable, Equatable {
        let speed: String
        let perf: String?
        let rated: Bool
        let variant: String
        let initialFen: String
        let clockInitialMilliseconds: Int?
        let clockIncrementMilliseconds: Int?
    }

    struct Player: Sendable, Codable, Equatable {
        /// Lowercase Lichess id; nil for Lichess's AI.
        let id: String?
        let name: String?
        let title: String?
        let kind: LichessBotOpponentKind
        let aiLevel: Int?
        let ratingBefore: Int?
        /// From the export; nil for casual games and the AI.
        let ratingDiff: Int?
        let provisional: Bool?
    }

    struct Outcome: Sendable, Codable, Equatable {
        /// Lichess's status name, spelled as Lichess spells it (see
        /// `LichessBotGameStatusName`).
        let status: String
        let winner: String?
        /// The PGN result token (`LichessBotRecordBuilder.pgnResult`): a win
        /// for either side, a draw, or the unfinished-game token for a game
        /// that never counted.
        let pgnResult: String
        /// DCM's score: a win, half a point for a draw, or a loss; nil for a
        /// game that never counted.
        let ourScore: Double?
        let plies: Int
        /// The draw rule DCM's own engine saw in the final position (E10).
        let localDrawCondition: ChessDrawCondition?
    }

    struct Move: Sendable, Codable, Equatable {
        let ply: Int
        let color: LichessBotColorName
        /// Nil only if the move list could not be replayed from this ply.
        let san: String?
        /// The token exactly as Lichess sent it (opponent) or as DCM sent it
        /// (ours) — never normalized (E1). For a move known only from the
        /// export (`receivedAt` nil), the UCI DCM derived from the export's
        /// SAN.
        let uciAsGiven: String
        /// Server clocks after this move, when the state that first carried
        /// it carried only it (a reconnect's `gameFull` brings several moves
        /// with one pair of clocks, which belongs to the last).
        let whiteClockMilliseconds: Int?
        let blackClockMilliseconds: Int?
        /// When the stream line that first carried the move arrived; nil for
        /// a move known only from Lichess's export (the journal ended before
        /// it).
        let receivedAt: Date?
        let ours: Bool
        let decision: LichessBotMoveDecision?
        /// The decision's generation number (`LichessBotGenerationInfo.generationID`).
        /// Numbering restarts at 1 every time the bot goes online, so in a
        /// game resumed after a relaunch one number can belong to two
        /// generations; `generationIndex` names the one that decided.
        let generationID: Int?
        /// Index into the record's `generations` of the generation that made
        /// `decision`; nil when there is no decision. Also nil in records
        /// written before this field existed: those list at most one
        /// generation per `generationID`, so the ID finds it (for a game
        /// resumed across a relaunch it is the first session's, whichever
        /// model decided the move; only rebuilding the record from its kept
        /// journal corrects that).
        let generationIndex: Int?
        let postMilliseconds: Double?
        let offeredDraw: Bool?
    }

    /// A move removed by a takeback (E30).
    struct Retraction: Sendable, Codable, Equatable {
        let ply: Int
        let uciAsGiven: String
        let retractedAt: Date
    }

    struct TimedNote: Sendable, Codable, Equatable {
        let at: Date
        let text: String
    }

    struct ChatMessage: Sendable, Codable, Equatable {
        let at: Date
        let room: String
        let username: String
        let text: String
    }

    struct RejectedMove: Sendable, Codable, Equatable {
        let at: Date
        let ply: Int
        let uci: String
        let error: String
    }

    struct Reconciliation: Sendable, Codable, Equatable {
        enum Outcome: String, Sendable, Codable, Equatable {
            /// The export agrees with the journal.
            case matched
            /// The export disagreed; the export's values were used.
            case corrected
            /// No export exists (Lichess keeps no export for some aborted
            /// games); the journal is the only source.
            case exportUnavailable
        }
        let outcome: Outcome
        let checkedAt: Date
        let mismatches: [String]
        let note: String?
    }

    let schemaVersion: Int
    let gameID: String
    let url: String
    let createdAt: Date
    let firstEventAt: Date?
    let lastEventAt: Date?
    /// App builds that wrote this game's journal (more than one if the game
    /// spanned a relaunch).
    let builds: [Int]
    let setup: Setup
    let ourColor: LichessBotColorName
    let us: Player
    let opponent: Player
    let outcome: Outcome
    let openingECO: String?
    let openingName: String?
    /// Every model generation that chose a move, in order of first use, each
    /// once. Told apart by their whole info, not their `generationID`: a game
    /// resumed after a relaunch can hold two generations numbered alike.
    let generations: [LichessBotGenerationInfo]
    let moves: [Move]
    let retractions: [Retraction]
    let events: [TimedNote]
    let chat: [ChatMessage]
    let anomalies: [TimedNote]
    let rejectedMoves: [RejectedMove]
    /// Game-stream connections after the first.
    let streamReconnects: Int
    /// Bytes of unterminated final journal lines dropped when the record was
    /// built. A line cut before an append is recorded as an anomaly instead
    /// (see `LichessBotJSONLines.cutUnterminatedFinalLine`).
    let droppedTrailingJournalBytes: Int
    let reconciliation: Reconciliation
}

enum LichessBotRecordError: LocalizedError, Equatable {
    /// Neither the journal nor the export describes the game.
    case noGameInformation(gameID: String)
    /// Neither player in the game is our account.
    case notOurGame(gameID: String, ourAccountID: String)
    /// There is no export, and the journal never recorded how the game ended.
    case noOutcome(gameID: String)

    var errorDescription: String? {
        switch self {
        case .noGameInformation(let gameID):
            return "Game \(gameID): the journal has no gameFull and there is no export to build a record from"
        case .notOurGame(let gameID, let ourAccountID):
            return "Game \(gameID): neither player is \(ourAccountID)"
        case .noOutcome(let gameID):
            return "Game \(gameID): there is no export and the journal never recorded how the game ended"
        }
    }
}

/// Builds a `LichessBotGameRecord` from a journal and, when available,
/// Lichess's export (plan §10.2). Pure: the same inputs always give the
/// same record, so a record can be rebuilt from its kept journal at any
/// time.
///
/// The move list is replayed from the journal's raw stream lines — the
/// same lines the game session played from — and DCM's own decisions and
/// POSTs are attached to our moves by ply once the whole journal is
/// replayed. The export wins for status,
/// winner, rating changes and opening; every disagreement is listed in the
/// record's reconciliation block.
enum LichessBotRecordBuilder {

    /// The status of a record built with neither an export nor a finish in
    /// the journal. Only a direct `build` can produce one:
    /// `LichessBotRecordStore.finalize` refuses to file such a game
    /// (`noOutcome`), so it never reaches disk. It maps to the unfinished
    /// result and no score.
    static let statusNotRecorded = "unknown"

    /// The journal's first `gameFull`, if any.
    static func firstGameFull(in entries: [LichessBotJournalEntry]) -> LichessBotGameFull? {
        for entry in entries {
            guard case .streamLine(let raw) = entry.event else { continue }
            do {
                if case .gameFull(let full) = try LichessBotGameStreamLine.decode(Data(raw.utf8)) {
                    return full
                }
            } catch {
                // An undecodable line is not a gameFull; `build` reports it
                // as an anomaly.
            }
        }
        return nil
    }

    /// When the game was created, from its `gameFull`.
    static func creationDate(of full: LichessBotGameFull) -> Date {
        Date(timeIntervalSince1970: Double(full.createdAt) / 1000)
    }

    /// When the game was created, from Lichess's export; nil when the export
    /// doesn't say.
    static func creationDate(of export: LichessBotGameExport) -> Date? {
        export.createdAt.map { Date(timeIntervalSince1970: Double($0) / 1000) }
    }

    static func build(
        gameID: String,
        journal: LichessBotJSONLines.Decoded<LichessBotJournalEntry>,
        export: LichessBotGameExport?,
        exportUnavailableReason: String?,
        ourAccountID: String,
        checkedAt: Date
    ) throws -> LichessBotGameRecord {
        var replay = Replay(ourAccountID: ourAccountID)
        for entry in journal.elements {
            replay.apply(entry)
        }

        guard let facts = GameFacts(full: replay.firstFull, export: export) else {
            throw LichessBotRecordError.noGameInformation(gameID: gameID)
        }
        guard let ourColor = facts.color(of: ourAccountID) else {
            throw LichessBotRecordError.notOurGame(gameID: gameID, ourAccountID: ourAccountID)
        }

        var mismatches: [String] = []
        let journalStatus = replay.finish?.status
        let journalWinner = replay.finish?.winner
        let status: String
        let winner: String?
        if let export {
            status = export.status.raw
            winner = export.winner?.raw
            if let journalStatus {
                if journalStatus != status {
                    mismatches.append("status: journal \(journalStatus), export \(status)")
                }
                if journalWinner != winner {
                    mismatches.append("winner: journal \(journalWinner ?? "none"), export \(winner ?? "none")")
                }
            } else {
                // The resignation-stream gap (E22), or a game that ended
                // while the app was down.
                mismatches.append("journal has no finished status; export says \(status)")
            }
            let exportSANs = export.sanMoves
            let journalSANs = replay.pending.compactMap(\.san)
            if journalSANs.count == replay.pending.count {
                // The export is authoritative. A journal that agrees with it
                // up to its own end, and ends early (the app was down, or
                // crashed before journaling the last lines), is extended
                // from the export so the record holds the whole game.
                if exportSANs.count > journalSANs.count && Array(exportSANs.prefix(journalSANs.count)) == journalSANs {
                    let tail = Array(exportSANs[journalSANs.count...])
                    if let failure = replay.extendFromExport(tail, clocksCentiseconds: export.clocks) {
                        mismatches.append("the journal ends at ply \(journalSANs.count); the export's later moves could not all be added: \(failure)")
                    } else {
                        mismatches.append("the journal ends at ply \(journalSANs.count); the export's \(tail.count) later move(s) were added")
                    }
                }
                let recordSANs = replay.pending.compactMap(\.san)
                if recordSANs != exportSANs {
                    let firstDifference = zip(recordSANs, exportSANs).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                        ?? min(recordSANs.count, exportSANs.count)
                    mismatches.append("moves differ from ply \(firstDifference): record has \(recordSANs.count), export has \(exportSANs.count)")
                }
            } else {
                mismatches.append("journal moves could not be fully replayed; export has \(exportSANs.count) moves")
            }
        } else if let journalFinish = replay.finish {
            status = journalFinish.status
            winner = journalFinish.winner
        } else {
            status = Self.statusNotRecorded
            winner = nil
            mismatches.append("no export and no finished status in the journal")
        }
        replay.finishMoves(ourColor: ourColor, exportNoteTime: checkedAt)
        if journal.droppedTrailingByteCount > 0 {
            replay.anomalies.append(.init(at: checkedAt, text: "the journal ended in an unterminated line of \(journal.droppedTrailingByteCount) bytes (an interrupted write), which was dropped"))
        }

        let reconciliation: LichessBotGameRecord.Reconciliation
        if export == nil {
            reconciliation = .init(outcome: .exportUnavailable, checkedAt: checkedAt, mismatches: mismatches, note: exportUnavailableReason)
        } else {
            reconciliation = .init(outcome: mismatches.isEmpty ? .matched : .corrected, checkedAt: checkedAt, mismatches: mismatches, note: nil)
        }

        let opponentColor: LichessBotColorName = ourColor == .white ? .black : .white
        let outcome = LichessBotGameRecord.Outcome(
            status: status,
            winner: winner,
            pgnResult: pgnResult(status: status, winner: winner),
            ourScore: ourScore(status: status, winner: winner, ourColor: ourColor),
            plies: replay.moves.count,
            localDrawCondition: replay.finish?.localDrawCondition
        )

        return LichessBotGameRecord(
            schemaVersion: LichessBotGameRecord.currentSchemaVersion,
            gameID: gameID,
            url: "https://lichess.org/\(gameID)",
            createdAt: facts.createdAt,
            firstEventAt: journal.elements.first?.at,
            lastEventAt: journal.elements.last?.at,
            builds: replay.builds,
            setup: facts.setup,
            ourColor: ourColor,
            us: facts.player(ourColor, export: export),
            opponent: facts.player(opponentColor, export: export),
            outcome: outcome,
            openingECO: export?.opening?.eco,
            openingName: export?.opening?.name,
            generations: replay.generations,
            moves: replay.moves,
            retractions: replay.retractions,
            events: replay.events,
            chat: replay.mergedChat(),
            anomalies: replay.anomalies,
            rejectedMoves: replay.rejectedMoves,
            streamReconnects: max(0, replay.streamOpens - 1),
            droppedTrailingJournalBytes: journal.droppedTrailingByteCount,
            reconciliation: reconciliation
        )
    }

    static func pgnResult(status: String, winner: String?) -> String {
        switch winner {
        case "white": return "1-0"
        case "black": return "0-1"
        default:
            let status = LichessBotGameStatusName(rawValue: status)
            switch status {
            case .aborted, .noStart, .created, .started, .unknownFinish, .none:
                return "*"
            default:
                return "1/2-1/2"
            }
        }
    }

    static func ourScore(status: String, winner: String?, ourColor: LichessBotColorName) -> Double? {
        if let winner {
            return winner == ourColor.rawValue ? 1 : 0
        }
        return pgnResult(status: status, winner: nil) == "*" ? nil : 0.5
    }

    // MARK: - Game facts (setup and players)

    /// Setup and players, from the journal's first `gameFull` when there is
    /// one, otherwise from the export.
    private struct GameFacts {
        let createdAt: Date
        let setup: LichessBotGameRecord.Setup
        let white: (id: String?, name: String?, title: String?, rating: Int?, provisional: Bool?, aiLevel: Int?)
        let black: (id: String?, name: String?, title: String?, rating: Int?, provisional: Bool?, aiLevel: Int?)

        init?(full: LichessBotGameFull?, export: LichessBotGameExport?) {
            if let full {
                createdAt = LichessBotRecordBuilder.creationDate(of: full)
                setup = LichessBotGameRecord.Setup(
                    speed: full.speed.raw,
                    perf: full.perf?.name,
                    rated: full.rated,
                    variant: full.variant.key.raw,
                    initialFen: full.initialFen,
                    clockInitialMilliseconds: full.clock?.initial.value,
                    clockIncrementMilliseconds: full.clock?.increment.value
                )
                white = (full.white.id, full.white.name, full.white.title, full.white.rating, full.white.provisional, full.white.aiLevel)
                black = (full.black.id, full.black.name, full.black.title, full.black.rating, full.black.provisional, full.black.aiLevel)
            } else if let export, let exportCreatedAt = LichessBotRecordBuilder.creationDate(of: export), let speed = export.speed, let variant = export.variant, let rated = export.rated {
                createdAt = exportCreatedAt
                setup = LichessBotGameRecord.Setup(
                    speed: speed,
                    perf: export.perf,
                    rated: rated,
                    variant: variant,
                    initialFen: export.initialPosition,
                    clockInitialMilliseconds: export.clock?.initial.map { $0 * 1000 },
                    clockIncrementMilliseconds: export.clock?.increment.map { $0 * 1000 }
                )
                let w = export.players.white
                let b = export.players.black
                white = (w.user?.id, w.user?.name, w.user?.title, w.rating, w.provisional, w.aiLevel)
                black = (b.user?.id, b.user?.name, b.user?.title, b.rating, b.provisional, b.aiLevel)
            } else {
                return nil
            }
        }

        func color(of accountID: String) -> LichessBotColorName? {
            if white.id == accountID { return .white }
            if black.id == accountID { return .black }
            return nil
        }

        func player(_ color: LichessBotColorName, export: LichessBotGameExport?) -> LichessBotGameRecord.Player {
            let p = color == .white ? white : black
            let kind: LichessBotOpponentKind
            if p.aiLevel != nil {
                kind = .lichessAI
            } else if p.title == "BOT" {
                kind = .bot
            } else {
                kind = .human
            }
            return LichessBotGameRecord.Player(
                id: p.id,
                name: p.name,
                title: p.title,
                kind: kind,
                aiLevel: p.aiLevel,
                ratingBefore: p.rating,
                ratingDiff: export?.player(color).ratingDiff,
                provisional: p.provisional
            )
        }
    }

    // MARK: - Journal replay

    private struct Finish {
        let status: String
        let winner: String?
        let localDrawCondition: ChessDrawCondition?
    }

    private struct PendingMove {
        let ply: Int
        let uciAsGiven: String
        let san: String?
        let whiteClock: Int?
        let blackClock: Int?
        /// Nil for a move known only from Lichess's export.
        let receivedAt: Date?
    }

    private struct Replay {
        let ourAccountID: String
        var firstFull: LichessBotGameFull?
        var tracker: LichessBotPositionTracker?
        var replayable = true
        var pending: [PendingMove] = []
        var moves: [LichessBotGameRecord.Move] = []
        var retractions: [LichessBotGameRecord.Retraction] = []
        var events: [LichessBotGameRecord.TimedNote] = []
        var chat: [LichessBotGameRecord.ChatMessage] = []
        /// Every message DCM sent, from its successful POST. Merged into
        /// `chat` by `mergedChat()`: the stream echoes a message only while
        /// it is open, so the goodbye (sent after it closes) is known only
        /// from here.
        var sentChat: [LichessBotGameRecord.ChatMessage] = []
        /// Player-room lines found by fetching the chat after the game. A
        /// session resumed after a relaunch fetches lines the journal
        /// already holds, so these are matched against it when merged.
        var fetchedChat: [LichessBotGameRecord.ChatMessage] = []
        var anomalies: [LichessBotGameRecord.TimedNote] = []
        var rejectedMoves: [LichessBotGameRecord.RejectedMove] = []
        var generations: [LichessBotGenerationInfo] = []
        var builds: [Int] = []
        var streamOpens = 0
        var finish: Finish?
        /// Our decisions and POSTs by ply, each in journal order. A ply can
        /// have several (a takeback and replay). They are matched to the move
        /// finally at each ply only in `finishMoves`, once the whole journal
        /// is replayed: a held move's echo is often journaled before its
        /// POST, so matching at the stream line would miss it. Each carries
        /// the index into `generations` of the generation that made it, the
        /// only exact reference once a game spans two going-online sessions.
        var decisions: [Int: [(decision: LichessBotMoveDecision, generationID: Int, generationIndex: Int)]] = [:]
        var posts: [Int: [(uci: String, offeringDraw: Bool, milliseconds: Double)]] = [:]
        var offerFlags: [String: Bool] = [:]

        init(ourAccountID: String) {
            self.ourAccountID = ourAccountID
        }

        mutating func apply(_ entry: LichessBotJournalEntry) {
            switch entry.event {
            case .header(_, _, let build, _, _):
                if !builds.contains(build) {
                    builds.append(build)
                }
            case .streamOpened:
                streamOpens += 1
            case .streamLine(let raw):
                applyLine(Data(raw.utf8), at: entry.at)
            case .streamLineBytes:
                anomalies.append(.init(at: entry.at, text: "stream line was not UTF-8"))
            case .streamEnded, .positionSynced, .keepAlive, .request:
                break
            case .moveDecided(let ply, let decision, let generation):
                // Matched on the whole info, never the ID alone: IDs restart
                // at 1 every time the bot goes online, so a game resumed
                // after a relaunch can journal a different model under an ID
                // the first session already used. Two sessions' generations
                // always differ (`snapshotAt` is when each was built), while
                // one generation, journaled with every move it decides, is
                // listed once.
                let generationIndex: Int
                if let listedIndex = generations.firstIndex(of: generation) {
                    generationIndex = listedIndex
                } else {
                    generationIndex = generations.count
                    generations.append(generation)
                }
                decisions[ply, default: []].append((decision, generation.generationID, generationIndex))
            case .movePosted(let ply, let uci, let offeringDraw, let milliseconds):
                posts[ply, default: []].append((uci, offeringDraw, milliseconds))
            case .moveRejected(let ply, let uci, let error):
                // A refused attempt is journaled instead of `movePosted`, never
                // after it, so there is no POST of this attempt to take back.
                // An earlier accepted POST of the same move stays: it happened.
                rejectedMoves.append(.init(at: entry.at, ply: ply, uci: uci, error: error))
            case .action(let text):
                events.append(.init(at: entry.at, text: text))
            case .chatSent(let room, let text, _):
                sentChat.append(.init(at: entry.at, room: room, username: ourAccountID, text: text))
            case .chatFetched(let username, let text):
                fetchedChat.append(.init(at: entry.at, room: LichessBotChatRoom.player.rawValue, username: username, text: text))
            case .anomaly(let text):
                anomalies.append(.init(at: entry.at, text: text))
            case .finished(let status, let winner, let localDrawCondition):
                finish = Finish(status: status, winner: winner, localDrawCondition: localDrawCondition)
            case .takebackAccepted:
                // The same timeline note the journal's earlier free-text
                // action produced.
                events.append(.init(at: entry.at, text: "accepted takeback"))
            case .commandReplyQueued:
                // The reply itself is a request and a sent chat message,
                // both journaled on their own.
                break
            }
        }

        /// The stream's chat plus each sent message the stream never echoed,
        /// in time order. Each echo (our account, same room and text)
        /// accounts for one sent message.
        func mergedChat() -> [LichessBotGameRecord.ChatMessage] {
            var echoes = chat.filter { $0.username.lowercased() == ourAccountID.lowercased() }
            var merged = chat
            // Our display name as the game showed it, when known.
            let ourName: String
            if let full = firstFull, full.white.id == ourAccountID, let name = full.white.name {
                ourName = name
            } else if let full = firstFull, full.black.id == ourAccountID, let name = full.black.name {
                ourName = name
            } else {
                ourName = ourAccountID
            }
            for sent in sentChat {
                if let index = echoes.firstIndex(where: { $0.room == sent.room && $0.text == sent.text }) {
                    echoes.remove(at: index)
                } else {
                    merged.append(.init(at: sent.at, room: sent.room, username: ourName, text: sent.text))
                }
            }
            // Each fetched line already known (same author and text, player
            // room) accounts for one known line; the rest are new.
            let player = LichessBotChatRoom.player.rawValue
            var known = merged.filter { $0.room == player }.map { (user: $0.username.lowercased(), text: $0.text) }
            let ourNameLowercased = ourName.lowercased()
            for fetched in fetchedChat {
                let user = fetched.username.lowercased()
                if let index = known.firstIndex(where: { ($0.user == user || ($0.user == ourNameLowercased && user == ourAccountID.lowercased())) && $0.text == fetched.text }) {
                    known.remove(at: index)
                } else {
                    merged.append(fetched)
                }
            }
            return merged.sorted { $0.at < $1.at }
        }

        private mutating func applyLine(_ data: Data, at: Date) {
            let line: LichessBotGameStreamLine
            do {
                line = try LichessBotGameStreamLine.decode(data)
            } catch {
                anomalies.append(.init(at: at, text: "undecodable stream line in journal: \(error)"))
                return
            }
            switch line {
            case .gameFull(let full):
                if firstFull == nil {
                    firstFull = full
                    if full.variant.key.known == .standard && LichessBotPositionTracker.isStandardStart(full.initialFen) {
                        do {
                            tracker = try LichessBotPositionTracker(initialFen: full.initialFen)
                        } catch {
                            replayable = false
                        }
                    } else {
                        replayable = false
                    }
                }
                applyState(full.state, at: at)
            case .gameState(let state):
                applyState(state, at: at)
            case .chatLine(let line):
                chat.append(.init(at: at, room: line.room.raw, username: line.username, text: line.text))
            case .opponentGone(let gone):
                if gone.gone {
                    events.append(.init(at: at, text: "opponent gone" + (gone.claimWinInSeconds.map { "; claimable in \($0)s" } ?? "")))
                } else {
                    events.append(.init(at: at, text: "opponent back"))
                }
            case .unknown(let type):
                anomalies.append(.init(at: at, text: "unknown stream line type \(type)"))
            }
        }

        private mutating func applyState(_ state: LichessBotGameState, at: Date) {
            let tokens = state.moveTokens
            let common = zip(pending, tokens).prefix { $0.0.uciAsGiven == $0.1 }.count
            if common < pending.count {
                for removed in pending[common...].reversed() {
                    retractions.append(.init(ply: removed.ply, uciAsGiven: removed.uciAsGiven, retractedAt: at))
                }
                pending.removeLast(pending.count - common)
                if let tracker, replayable {
                    do {
                        try tracker.sync(to: pending.map(\.uciAsGiven))
                    } catch {
                        replayable = false
                    }
                }
            }
            if tokens.count > common {
                for index in common..<tokens.count {
                    let token = tokens[index]
                    let isLast = index == tokens.count - 1
                    pending.append(PendingMove(
                        ply: index,
                        uciAsGiven: token,
                        san: san(for: token, ply: index, at: at),
                        whiteClock: isLast ? state.wtime.value : nil,
                        blackClock: isLast ? state.btime.value : nil,
                        receivedAt: at
                    ))
                }
            }
            noteOffers(state, at: at)
        }

        private mutating func san(for token: String, ply: Int, at: Date) -> String? {
            guard replayable, let tracker else { return nil }
            let engine = tracker.engine
            guard let move = ChessMove.parseUCI(token, legal: engine.currentLegalMoves, state: engine.state) else {
                anomalies.append(.init(at: at, text: "move \(token) at ply \(ply) does not replay"))
                replayable = false
                return nil
            }
            do {
                let san = try SANFormatter.san(for: move, in: engine.state, legalMoves: engine.currentLegalMoves)
                try tracker.sync(to: pending.map(\.uciAsGiven) + [token])
                return san
            } catch {
                anomalies.append(.init(at: at, text: "move \(token) at ply \(ply) does not replay: \(error.localizedDescription)"))
                replayable = false
                return nil
            }
        }

        private mutating func noteOffers(_ state: LichessBotGameState, at: Date) {
            let flags: [(key: String, value: Bool?, text: String)] = [
                ("wdraw", state.wdraw, "white offers a draw"),
                ("bdraw", state.bdraw, "black offers a draw"),
                ("wtakeback", state.wtakeback, "white proposes a takeback"),
                ("btakeback", state.btakeback, "black proposes a takeback"),
            ]
            for flag in flags {
                let now = flag.value ?? false
                if now && !(offerFlags[flag.key] ?? false) {
                    events.append(.init(at: at, text: flag.text))
                }
                offerFlags[flag.key] = now
            }
        }

        private static let millisecondsPerCentisecond = 10

        /// Append the export's moves past the end of the journal's list — a
        /// game that went on after the journal stopped (the app was down, or
        /// crashed before journaling the last lines). Called only when the
        /// journal's SAN list is a strict prefix of the export's. Returns why
        /// it stopped early, if it did. The export's `clocks` give each ply
        /// the mover's clock after the move, so only the mover's clock is set.
        mutating func extendFromExport(_ sans: [String], clocksCentiseconds: [Int]?) -> String? {
            guard replayable, let tracker else {
                return "the journal's position can't be extended"
            }
            for exportSAN in sans {
                let ply = pending.count
                let engine = tracker.engine
                guard let move = PGNImporter.resolveLegalSANMove(exportSAN, state: engine.state) else {
                    replayable = false
                    return "export move \(exportSAN) at ply \(ply) does not replay"
                }
                let san: String
                do {
                    san = try SANFormatter.san(for: move, in: engine.state, legalMoves: engine.currentLegalMoves)
                    try tracker.sync(to: pending.map(\.uciAsGiven) + [move.uci])
                } catch {
                    replayable = false
                    return "export move \(exportSAN) at ply \(ply) does not replay: \(error.localizedDescription)"
                }
                let moverClock: Int?
                if let clocksCentiseconds, ply < clocksCentiseconds.count {
                    moverClock = clocksCentiseconds[ply] * Self.millisecondsPerCentisecond
                } else {
                    moverClock = nil
                }
                let whiteMoved = ply % 2 == 0
                pending.append(PendingMove(
                    ply: ply,
                    uciAsGiven: move.uci,
                    san: san,
                    whiteClock: whiteMoved ? moverClock : nil,
                    blackClock: whiteMoved ? nil : moverClock,
                    receivedAt: nil
                ))
            }
            return nil
        }

        /// Turn the move list into record moves, now that our color is known
        /// and the whole journal has been replayed. Our move at a ply takes
        /// the last POST of its exact token at that ply, and — only then —
        /// the last decision for that token. A move on our side that this
        /// client did not post is the other-client signal (plan §6.1 B) and
        /// is recorded as an anomaly; one known only from the export is noted
        /// at `exportNoteTime`, the reconciliation time.
        mutating func finishMoves(ourColor: LichessBotColorName, exportNoteTime: Date) {
            moves = pending.map { move in
                let color: LichessBotColorName = move.ply % 2 == 0 ? .white : .black
                let ours = color == ourColor
                let post = ours ? posts[move.ply]?.last(where: { $0.uci == move.uciAsGiven }) : nil
                let decision = post == nil ? nil : decisions[move.ply]?.last(where: { $0.decision.uci == move.uciAsGiven })
                if ours && post == nil {
                    if let receivedAt = move.receivedAt {
                        anomalies.append(.init(at: receivedAt, text: "move \(move.uciAsGiven) at ply \(move.ply) is on our side but was not posted by this client"))
                    } else {
                        anomalies.append(.init(at: exportNoteTime, text: "move \(move.uciAsGiven) at ply \(move.ply), known only from the export, is on our side but was not posted by this client"))
                    }
                }
                return LichessBotGameRecord.Move(
                    ply: move.ply,
                    color: color,
                    san: move.san,
                    uciAsGiven: move.uciAsGiven,
                    whiteClockMilliseconds: move.whiteClock,
                    blackClockMilliseconds: move.blackClock,
                    receivedAt: move.receivedAt,
                    ours: ours,
                    decision: decision?.decision,
                    generationID: decision?.generationID,
                    generationIndex: decision?.generationIndex,
                    postMilliseconds: post?.milliseconds,
                    offeredDraw: post?.offeringDraw
                )
            }
        }
    }
}
