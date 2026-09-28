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
/// shows about it (plan §10.6). Written once, after reconciliation with
/// Lichess's export; the journal it was built from is kept beside it.
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
        /// Lichess's status name (`mate`, `resign`, `outoftime`, …).
        let status: String
        let winner: String?
        /// `1-0`, `0-1`, `1/2-1/2`, or `*` for a game that never counted.
        let pgnResult: String
        /// 1, ½ or 0 for DCM; nil for aborted games.
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
        /// (ours) — never normalized (E1).
        let uciAsGiven: String
        /// Server clocks after this move, when the state that first carried
        /// it carried only it (a reconnect's `gameFull` brings several moves
        /// with one pair of clocks, which belongs to the last).
        let whiteClockMilliseconds: Int?
        let blackClockMilliseconds: Int?
        let receivedAt: Date
        let ours: Bool
        let decision: LichessBotMoveDecision?
        let generationID: Int?
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
    /// Every model generation that chose a move, in order of first use.
    let generations: [LichessBotGenerationInfo]
    let moves: [Move]
    let retractions: [Retraction]
    let events: [TimedNote]
    let chat: [ChatMessage]
    let anomalies: [TimedNote]
    let rejectedMoves: [RejectedMove]
    /// Game-stream connections after the first.
    let streamReconnects: Int
    let droppedTrailingJournalBytes: Int
    let reconciliation: Reconciliation
}

enum LichessBotRecordError: LocalizedError, Equatable {
    /// Neither the journal nor the export describes the game.
    case noGameInformation(gameID: String)
    /// Neither player in the game is our account.
    case notOurGame(gameID: String, ourAccountID: String)

    var errorDescription: String? {
        switch self {
        case .noGameInformation(let gameID):
            return "Game \(gameID): the journal has no gameFull and there is no export to build a record from"
        case .notOurGame(let gameID, let ourAccountID):
            return "Game \(gameID): neither player is \(ourAccountID)"
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
/// POSTs are attached to our moves by ply. The export wins for status,
/// winner, rating changes and opening; every disagreement is listed in the
/// record's reconciliation block.
enum LichessBotRecordBuilder {

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
        replay.finishMoves(ourColor: ourColor)

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
            let ourSANs = replay.moves.map(\.san)
            let exportSANs = export.sanMoves
            if ourSANs.allSatisfy({ $0 != nil }) {
                let journalSANs = ourSANs.compactMap { $0 }
                if journalSANs != exportSANs {
                    let firstDifference = zip(journalSANs, exportSANs).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                        ?? min(journalSANs.count, exportSANs.count)
                    mismatches.append("moves differ from ply \(firstDifference): journal has \(journalSANs.count), export has \(exportSANs.count)")
                }
            } else {
                mismatches.append("journal moves could not be fully replayed; export has \(exportSANs.count) moves")
            }
        } else if let journalFinish = replay.finish {
            status = journalFinish.status
            winner = journalFinish.winner
        } else {
            status = "unknown"
            winner = nil
            mismatches.append("no export and no finished status in the journal")
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
                createdAt = Date(timeIntervalSince1970: Double(full.createdAt) / 1000)
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
            } else if let export, let exportCreatedAt = export.createdAt, let speed = export.speed, let variant = export.variant, let rated = export.rated {
                createdAt = Date(timeIntervalSince1970: Double(exportCreatedAt) / 1000)
                setup = LichessBotGameRecord.Setup(
                    speed: speed,
                    perf: export.perf,
                    rated: rated,
                    variant: variant,
                    initialFen: "startpos",
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
        let receivedAt: Date
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
        /// Our decisions and POSTs, by ply. A decision is consumed by the
        /// move it produced; a takeback leaves room for a new one.
        var decisions: [Int: (decision: LichessBotMoveDecision, generationID: Int)] = [:]
        var posts: [Int: (uci: String, offeringDraw: Bool, milliseconds: Double)] = [:]
        /// Our moves, resolved when the move list is final.
        var ourMoveAttachments: [Int: (decision: LichessBotMoveDecision?, generationID: Int?, postMilliseconds: Double?, offeredDraw: Bool?)] = [:]
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
                decisions[ply] = (decision, generation.generationID)
                if !generations.contains(where: { $0.generationID == generation.generationID }) {
                    generations.append(generation)
                }
            case .movePosted(let ply, let uci, let offeringDraw, let milliseconds):
                posts[ply] = (uci, offeringDraw, milliseconds)
            case .moveRejected(let ply, let uci, let error):
                rejectedMoves.append(.init(at: entry.at, ply: ply, uci: uci, error: error))
                if posts[ply]?.uci == uci {
                    posts[ply] = nil
                }
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
                    ourMoveAttachments[removed.ply] = nil
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
                    attachOurMove(ply: index, token: token)
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

        /// Record what DCM decided and posted for this ply, if this client
        /// posted it. Which side the move belongs to is settled later, once
        /// our color is known; a post whose token differs is left for the
        /// our-side check to flag.
        private mutating func attachOurMove(ply: Int, token: String) {
            guard let post = posts[ply], post.uci == token else {
                ourMoveAttachments[ply] = (nil, nil, nil, nil)
                return
            }
            let decision = decisions[ply].flatMap { $0.decision.uci == token ? $0 : nil }
            ourMoveAttachments[ply] = (decision?.decision, decision?.generationID, post.milliseconds, post.offeringDraw)
            posts[ply] = nil
            decisions[ply] = nil
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

        /// Turn the replayed move list into record moves, now that our color
        /// is known. A move on our side that this client did not post is
        /// the other-client signal (plan §6.1 B) and is recorded as an
        /// anomaly.
        mutating func finishMoves(ourColor: LichessBotColorName) {
            moves = pending.map { move in
                let color: LichessBotColorName = move.ply % 2 == 0 ? .white : .black
                let ours = color == ourColor
                let attachment = ourMoveAttachments[move.ply]
                if ours && attachment?.postMilliseconds == nil {
                    anomalies.append(.init(at: move.receivedAt, text: "move \(move.uciAsGiven) at ply \(move.ply) is on our side but was not posted by this client"))
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
                    decision: ours ? attachment?.decision : nil,
                    generationID: ours ? attachment?.generationID : nil,
                    postMilliseconds: ours ? attachment?.postMilliseconds : nil,
                    offeredDraw: ours ? attachment?.offeredDraw : nil
                )
            }
        }
    }
}
