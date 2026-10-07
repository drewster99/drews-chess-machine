import Foundation

/// One line of a game's protocol transcript (plan §14.3a).
struct LichessBotTranscriptEntry: Identifiable, Sendable, Equatable {
    enum Direction: Sendable, Equatable {
        /// From Lichess (a stream line).
        case incoming
        /// From DCM (a request).
        case outgoing
        /// A local note: a stream opening or ending, an anomaly.
        case note
    }

    let id: Int
    let at: Date
    let direction: Direction
    /// A one-line summary: the stream line's type, or the request's method,
    /// label, status and time.
    let title: String
    /// The full raw text: the JSON line, or the request's method, path and
    /// fields.
    let detail: String
    let isProblem: Bool
    /// For collapsed keep-alives: how many arrived in a row.
    var repeatCount: Int
}

/// The live view of one Lichess game, fed by its session's events on the
/// main actor (plan §14.3a). Keeps the position after every ply so views can
/// browse earlier positions without touching the game.
@MainActor
@Observable
final class LichessBotLiveGame: Identifiable {
    struct Ply: Identifiable {
        /// 0-based ply index.
        let id: Int
        let uciAsGiven: String
        let move: ChessMove
        let san: String
        let color: PieceColor
        /// The position after this ply.
        let stateAfter: GameState
    }

    struct Player: Equatable {
        let id: String?
        let name: String
        let rating: Int?
        let title: String?
    }

    struct ChatMessage: Identifiable, Equatable {
        let id: Int
        let at: Date
        let room: String
        let username: String
        let text: String
        /// Set for a message DCM sent, from its successful POST; nil for a
        /// message known only from the stream.
        let origin: LichessBotChatOrigin?
        /// A sent message whose echo Lichess streamed back.
        var echoed: Bool
    }

    let id: String
    let startedAt: Date
    private let ourAccountID: String

    /// How the game began (challenge-log plan §3.5); nil while not yet
    /// known — shown as "Not yet known", never blank. Set by the controller
    /// when it decides, and by `replay` from a resumed journal.
    private(set) var origin: LichessBotGameOrigin?

    private(set) var ourColor: PieceColor?
    private(set) var white: Player?
    private(set) var black: Player?
    private(set) var speed: String?
    private(set) var rated: Bool?
    private(set) var clockInitialMilliseconds: Int?
    private(set) var clockIncrementMilliseconds: Int?

    private(set) var plies: [Ply] = []
    private(set) var retractedPlyCount = 0
    /// Our decision for each ply we moved at.
    private(set) var decisions: [Int: LichessBotMoveDecision] = [:]
    private(set) var generations: [Int: LichessBotGenerationInfo] = [:]
    private(set) var latestDecision: LichessBotMoveDecision?

    private(set) var status: String = "started"
    private(set) var winner: String?
    /// The draw rule DCM's own engine sees in the final position, if any.
    /// Lichess's status says only "draw"; this says which rule.
    private(set) var localDrawCondition: ChessDrawCondition?
    private(set) var finishedAt: Date?
    private(set) var whiteClockMilliseconds: Int?
    private(set) var blackClockMilliseconds: Int?
    /// When the clocks above were received; views tick the side to move
    /// locally from here.
    private(set) var clocksReceivedAt: Date?
    private(set) var opponentOffersDraw = false
    private(set) var opponentProposesTakeback = false
    private(set) var opponentGoneClaimableInSeconds: Int?

    struct HeldMove: Equatable {
        let ply: Int
        let uci: String
        let san: String
    }

    /// The operator's move pacing for this game (plan §14.3c): a delay
    /// before each of DCM's moves after its first, or holding each move
    /// until Play move. Belongs to this game only and is never persisted.
    var moveDelaySeconds = 0
    var holdsMoves = false
    /// The move DCM has decided and is holding or delaying, if any.
    private(set) var heldMove: HeldMove?
    private(set) var releaseRequested = false

    /// A message DCM sent: shown at once, and matched to Lichess's echo if
    /// that already arrived (the echo and the POST's reply can come in
    /// either order), so it is listed once.
    private func recordSentChat(room: String, text: String, origin: LichessBotChatOrigin, at: Date) {
        if let index = chat.firstIndex(where: { $0.origin == nil && !$0.echoed && $0.room == room && $0.text == text && isFromUs($0) }) {
            let echo = chat[index]
            chat[index] = ChatMessage(id: echo.id, at: echo.at, room: echo.room, username: echo.username, text: echo.text, origin: origin, echoed: true)
            return
        }
        let ourName = (ourColor == .white ? white?.name : black?.name) ?? ourAccountID
        chat.append(ChatMessage(id: nextChatID, at: at, room: room, username: ourName, text: text, origin: origin, echoed: false))
        nextChatID += 1
    }

    /// A message from the stream. Our own echo confirms the matching sent
    /// message instead of listing it twice.
    private func recordStreamedChat(_ line: LichessBotChatLine, at: Date) {
        if line.username.lowercased() == ourAccountID.lowercased(),
           let index = chat.firstIndex(where: { $0.origin != nil && !$0.echoed && $0.room == line.room.raw && $0.text == line.text }) {
            chat[index].echoed = true
            return
        }
        chat.append(ChatMessage(id: nextChatID, at: at, room: line.room.raw, username: line.username, text: line.text, origin: nil, echoed: false))
        nextChatID += 1
    }

    /// The lines of a fetched player-room chat not already shown, in order.
    /// Each known player-room message (streamed or sent) accounts for one
    /// fetched line with the same author and text.
    func unseenChatLines(in fetched: [LichessBotFetchedChatLine]) -> [LichessBotFetchedChatLine] {
        var known = chat
            .filter { $0.room == LichessBotChatRoom.player.rawValue }
            .map { (user: isFromUs($0) ? ourAccountID.lowercased() : $0.username.lowercased(), text: $0.text) }
        var unseen: [LichessBotFetchedChatLine] = []
        for line in fetched {
            let user = line.user.lowercased()
            if let index = known.firstIndex(where: { $0.user == user && $0.text == line.text }) {
                known.remove(at: index)
            } else {
                unseen.append(line)
            }
        }
        return unseen
    }

    /// Whether a chat line was written by our own account (DCM's automatic
    /// messages and the operator's).
    func isFromUs(_ message: ChatMessage) -> Bool {
        message.username.lowercased() == ourAccountID.lowercased()
    }

    /// The operator's Play move for the held move.
    func requestRelease() {
        guard heldMove != nil else { return }
        releaseRequested = true
    }

    var pacingSnapshot: LichessBotMovePacingSnapshot {
        LichessBotMovePacingSnapshot(delaySeconds: moveDelaySeconds, holds: holdsMoves, releaseRequested: releaseRequested)
    }

    private(set) var transcript: [LichessBotTranscriptEntry] = []
    private(set) var chat: [ChatMessage] = []
    private(set) var anomalies: [String] = []
    private(set) var streamConnections = 0

    /// A move token that failed to replay, reported once per ply: every later
    /// state repeats the whole move list.
    private struct UnreplayableMove: Hashable {
        let ply: Int
        let token: String
    }
    @ObservationIgnored private var reportedUnreplayableMoves: Set<UnreplayableMove> = []

    private var nextTranscriptID = 0
    private var nextChatID = 0

    init(id: String, startedAt: Date, ourAccountID: String) {
        self.id = id
        self.startedAt = startedAt
        self.ourAccountID = ourAccountID
    }

    var isFinished: Bool {
        finishedAt != nil
    }

    /// DCM's score (1, ½ or 0) by the same rule as the game record's; nil
    /// while playing, for a game without a result (aborted, never started,
    /// left unfinished), or before our color is known.
    var ourScore: Double? {
        guard isFinished, let ourColor else { return nil }
        return LichessBotRecordBuilder.ourScore(status: status, winner: winner, ourColor: ourColor == .white ? .white : .black)
    }

    var opponent: Player? {
        switch ourColor {
        case .white: return black
        case .black: return white
        case .none: return nil
        }
    }

    /// The side to move in the current position.
    var sideToMove: PieceColor {
        plies.last?.stateAfter.currentPlayer ?? GameState.starting.currentPlayer
    }

    /// The position after `ply` plies (0 = the start).
    func state(afterPlies count: Int) -> GameState {
        count == 0 ? GameState.starting : plies[count - 1].stateAfter
    }

    // MARK: - Applying events

    /// A live event, as it happens.
    func apply(_ event: LichessBotGameEvent) {
        apply(event, at: Date())
    }

    /// An event that happened at `at`: now for a live event, the journal
    /// entry's time for one replayed from a resumed game's journal.
    private func apply(_ event: LichessBotGameEvent, at: Date) {
        switch event {
        case .streamOpened(let attempt):
            streamConnections += 1
            note("stream opened" + (attempt > 0 ? " (reconnect, attempt \(attempt))" : ""), isProblem: attempt > 0, at: at)
        case .streamLine(let data, let receivedAt):
            applyLine(data, at: receivedAt)
        case .keepAlive(let receivedAt):
            if var last = transcript.last, last.title == "keep-alive" {
                last.repeatCount += 1
                transcript[transcript.count - 1] = last
            } else {
                appendTranscript(at: receivedAt, direction: .incoming, title: "keep-alive", detail: "", isProblem: false)
            }
        case .streamEnded(let reason):
            note("stream ended: \(reason)", isProblem: !isFinished, at: at)
        case .gameInfo(let full, let ourColor):
            self.ourColor = ourColor == .white ? .white : .black
            white = player(full.white)
            black = player(full.black)
            speed = full.speed.raw
            rated = full.rated
            clockInitialMilliseconds = full.clock?.initial.value
            clockIncrementMilliseconds = full.clock?.increment.value
        case .positionSynced:
            break
        case .moveDecided(let ply, let decision, let generation):
            decisions[ply] = decision
            generations[ply] = generation
            latestDecision = decision
        case .movePosted:
            break
        case .moveRejected(let ply, let uci, let error):
            anomalies.append("move \(uci) at ply \(ply) rejected: \(error)")
        case .action(let text):
            note(text, isProblem: false, at: at)
        case .moveHeld(let ply, let uci, let san):
            heldMove = HeldMove(ply: ply, uci: uci, san: san)
            releaseRequested = false
            note("holding \(san) at ply \(ply)", isProblem: false, at: at)
        case .moveReleased(let ply, let reason):
            heldMove = nil
            releaseRequested = false
            note("released the held move at ply \(ply): \(reason)", isProblem: false, at: at)
        case .chat:
            break
        case .chatSent(let room, let text, let origin):
            recordSentChat(room: room.rawValue, text: text, origin: origin, at: at)
        case .chatFetched(let username, let text):
            chat.append(ChatMessage(id: nextChatID, at: at, room: LichessBotChatRoom.player.rawValue, username: username, text: text, origin: nil, echoed: false))
            nextChatID += 1
            note("post-game chat from \(username)", isProblem: false, at: at)
        case .anomaly(let text):
            anomalies.append(text)
            note(text, isProblem: true, at: at)
        case .stoppedMoving(let reason):
            anomalies.append("stopped moving: \(reason)")
            note("stopped moving: \(reason)", isProblem: true, at: at)
        case .tokenRejected(let detail):
            anomalies.append("token rejected: \(detail)")
            note("token rejected: \(detail)", isProblem: true, at: at)
        case .finished(let status, let winner, let localDrawCondition):
            self.status = status.raw
            self.winner = winner?.raw
            self.localDrawCondition = localDrawCondition
            if finishedAt == nil {
                finishedAt = at
            }
        case .takebackAccepted:
            note("accepted takeback", isProblem: false, at: at)
        case .commandReplyQueued:
            // The reply shows as its requests and its sent chat.
            break
        }
    }

    // MARK: - Replaying a resumed game's journal

    /// Rebuild this game's history from the journal an earlier run left, so
    /// a resumed game shows its earlier moves, DCM's decisions, the chat and
    /// the transcript instead of starting blank. Called once, before the
    /// game is listed, so no view watches it happen. Each entry is applied
    /// the way its live event was, at the entry's own time.
    func replay(_ journal: LichessBotResumedJournal) {
        for item in journal.items {
            let entry = item.entry
            switch entry.event {
            case .header(_, _, let build, _, let resumed):
                if resumed {
                    note("DCM resumed following this game (build \(build))", isProblem: false, at: entry.at)
                }
            case .streamOpened(let attempt):
                apply(.streamOpened(attempt: attempt), at: entry.at)
            case .streamLine(let raw):
                switch item.decodedLine {
                case .success(let line):
                    applyDecodedLine(line, raw: raw, at: entry.at)
                    if case .gameFull(let full) = line, let color = full.color(of: ourAccountID) {
                        // Live, the session reports the game's facts right
                        // after this line.
                        apply(.gameInfo(full, ourColor: color), at: entry.at)
                    }
                case .failure(let failure):
                    appendTranscript(at: entry.at, direction: .incoming, title: "undecodable line", detail: raw + "\n" + failure.description, isProblem: true)
                case .none:
                    appendTranscript(at: entry.at, direction: .incoming, title: "line not decoded", detail: raw, isProblem: true)
                }
            case .streamLineBytes:
                appendTranscript(at: entry.at, direction: .incoming, title: "stream line was not UTF-8", detail: "", isProblem: true)
            case .keepAlive:
                apply(.keepAlive(receivedAt: entry.at), at: entry.at)
            case .request(let record):
                applyRequest(record)
            case .streamEnded(let reason):
                apply(.streamEnded(reason: reason), at: entry.at)
            case .positionSynced, .movePosted:
                break
            case .moveDecided(let ply, let decision, let generation):
                apply(.moveDecided(ply: ply, decision: decision, generation: generation), at: entry.at)
            case .moveRejected(let ply, let uci, let error):
                apply(.moveRejected(ply: ply, uci: uci, error: error), at: entry.at)
            case .action(let text):
                apply(.action(text), at: entry.at)
            case .chatSent(let room, let text, let origin):
                guard let knownOrigin = LichessBotChatOrigin(rawValue: origin) else {
                    apply(.anomaly("journal: sent chat with unknown origin \"\(origin)\": \(text)"), at: entry.at)
                    continue
                }
                recordSentChat(room: room, text: text, origin: knownOrigin, at: entry.at)
            case .chatFetched(let username, let text):
                apply(.chatFetched(username: username, text: text), at: entry.at)
            case .anomaly(let text):
                apply(.anomaly(text), at: entry.at)
            case .finished(let status, let winner, let localDrawCondition):
                self.status = status
                self.winner = winner
                self.localDrawCondition = localDrawCondition
                if finishedAt == nil {
                    finishedAt = entry.at
                }
            case .takebackAccepted:
                apply(.takebackAccepted, at: entry.at)
            case .commandReplyQueued:
                break
            case .gameOrigin(let recorded):
                // The journal's rule: the first determined origin, else the
                // latest undetermined one.
                if origin?.isDetermined != true {
                    origin = recorded
                }
            }
        }
        if journal.droppedTrailingByteCount > 0 {
            note("the journal ended in \(journal.droppedTrailingByteCount) bytes of an unfinished line (an interrupted write)", isProblem: true, at: journal.lastJournaledAt)
        }
        note("DCM was not following this game from \(journal.lastJournaledAt.formatted(date: .omitted, time: .standard)) until now", isProblem: true, at: Date())
    }

    /// The controller decided how the game began.
    func setOrigin(_ decided: LichessBotGameOrigin) {
        origin = decided
    }

    /// The status of a game DCM stopped following before it ended: the only
    /// ending a later session for the same game undoes.
    static let leftUnfinishedStatus = "left unfinished"

    /// The session for this game ended without the game finishing (the bot
    /// went offline, the gate closed, the token was rejected). The view
    /// stops its clocks; the record is settled later by reconciliation.
    func markSessionEnded(_ reason: String) {
        guard finishedAt == nil else { return }
        finishedAt = Date()
        status = Self.leftUnfinishedStatus
        // A held move ends with its session.
        heldMove = nil
        releaseRequested = false
        note("DCM stopped following this game: \(reason)", isProblem: true, at: Date())
    }

    /// A new session for this game started: it was left unfinished (DCM went
    /// offline, or its session ended) while the game was still live on
    /// Lichess. Undo that ending so the clocks, pacing controls and chat input
    /// come back; the session's `gameFull` then restores status, clocks and
    /// offers. A game that really ended is left alone. The operator's pacing
    /// stays: it is their choice for this game, and a held or delayed move is
    /// still played automatically when DCM's clock gets low.
    func resumeFollowing() {
        guard finishedAt != nil, status == Self.leftUnfinishedStatus else { return }
        finishedAt = nil
        status = "started"
        note("DCM is following this game again", isProblem: false, at: Date())
    }

    /// A request DCM made for this game.
    func applyRequest(_ record: LichessBotRequestRecord) {
        var title = "\(record.method) \(record.label)"
        if let status = record.status {
            title += " · \(status)"
        } else {
            title += " · failed"
        }
        if let roundTrip = record.roundTripMilliseconds {
            title += String(format: " · %.0f ms", roundTrip)
        }
        var detail = "\(record.method) \(record.path)"
        for (name, value) in record.formFields.sorted(by: { $0.key < $1.key }) {
            detail += "\n  \(name)=\(value)"
        }
        detail += String(format: "\nqueued %.1f ms", record.queuedMilliseconds)
        if let networkProtocol = record.networkProtocol {
            detail += " · \(networkProtocol)"
        }
        if let message = record.errorMessage {
            detail += "\nerror: \(message)"
        }
        if let failure = record.failure {
            detail += "\nfailed: \(failure)"
        }
        let isProblem = record.status.map { !(200..<300).contains($0) } ?? true
        appendTranscript(at: record.startedAt, direction: .outgoing, title: title, detail: detail, isProblem: isProblem)
    }

    private func applyLine(_ data: Data, at: Date) {
        let raw = String(decoding: data, as: UTF8.self)
        let line: LichessBotGameStreamLine
        do {
            line = try LichessBotGameStreamLine.decode(data)
        } catch {
            appendTranscript(at: at, direction: .incoming, title: "undecodable line", detail: raw, isProblem: true)
            return
        }
        applyDecodedLine(line, raw: raw, at: at)
    }

    private func applyDecodedLine(_ line: LichessBotGameStreamLine, raw: String, at: Date) {
        switch line {
        case .gameFull(let full):
            appendTranscript(at: at, direction: .incoming, title: "gameFull · \(full.state.moveTokens.count) plies · \(full.state.status.raw)", detail: raw, isProblem: false)
            applyState(full.state, at: at)
        case .gameState(let state):
            appendTranscript(at: at, direction: .incoming, title: "gameState · \(state.moveTokens.count) plies · \(state.status.raw)", detail: raw, isProblem: false)
            applyState(state, at: at)
        case .chatLine(let message):
            appendTranscript(at: at, direction: .incoming, title: "chat · \(message.username)", detail: raw, isProblem: false)
            recordStreamedChat(message, at: at)
        case .opponentGone(let gone):
            appendTranscript(at: at, direction: .incoming, title: gone.gone ? "opponentGone" : "opponent back", detail: raw, isProblem: false)
            opponentGoneClaimableInSeconds = gone.gone ? gone.claimWinInSeconds : nil
        case .unknown(let type):
            appendTranscript(at: at, direction: .incoming, title: "unknown line type \(type)", detail: raw, isProblem: true)
        }
    }

    private func applyState(_ state: LichessBotGameState, at: Date) {
        let tokens = state.moveTokens
        let common = zip(plies, tokens).prefix { $0.0.uciAsGiven == $0.1 }.count
        if common < plies.count {
            retractedPlyCount += plies.count - common
            plies.removeLast(plies.count - common)
        }
        if tokens.count > common {
            var current = self.state(afterPlies: common)
            for index in common..<tokens.count {
                let token = tokens[index]
                let legal = MoveGenerator.legalMoves(for: current)
                guard let move = ChessMove.parseUCI(token, legal: legal, state: current) else {
                    if reportedUnreplayableMoves.insert(UnreplayableMove(ply: index, token: token)).inserted {
                        anomalies.append("move \(token) at ply \(index) does not replay; the board stops here")
                    }
                    break
                }
                let san: String
                do {
                    san = try SANFormatter.san(for: move, in: current, legalMoves: legal)
                } catch {
                    if reportedUnreplayableMoves.insert(UnreplayableMove(ply: index, token: token)).inserted {
                        anomalies.append("SAN for \(token) at ply \(index): \(error.localizedDescription)")
                    }
                    break
                }
                let color = current.currentPlayer
                current = MoveGenerator.applyMove(move, to: current)
                plies.append(Ply(id: index, uciAsGiven: token, move: move, san: san, color: color, stateAfter: current))
            }
        }
        whiteClockMilliseconds = state.wtime.value
        blackClockMilliseconds = state.btime.value
        clocksReceivedAt = at
        if let ourColor {
            let opponentColor: LichessBotColorName = ourColor == .white ? .black : .white
            opponentOffersDraw = state.isOfferingDraw(opponentColor)
            opponentProposesTakeback = state.isProposingTakeback(opponentColor)
        }
        status = state.status.raw
        if state.status.isLive == false {
            winner = state.winner?.raw
            if finishedAt == nil {
                finishedAt = at
            }
        }
    }

    private func note(_ text: String, isProblem: Bool, at: Date) {
        appendTranscript(at: at, direction: .note, title: text, detail: "", isProblem: isProblem)
    }

    private func appendTranscript(at: Date, direction: LichessBotTranscriptEntry.Direction, title: String, detail: String, isProblem: Bool) {
        transcript.append(LichessBotTranscriptEntry(id: nextTranscriptID, at: at, direction: direction, title: title, detail: detail, isProblem: isProblem, repeatCount: 1))
        nextTranscriptID += 1
    }

    private func player(_ player: LichessBotGamePlayer) -> Player {
        let name: String
        if let given = player.name {
            name = given
        } else if let level = player.aiLevel {
            name = "lichess AI level \(level)"
        } else if let id = player.id {
            name = id
        } else {
            anomalies.append("a player in gameFull has no name, id or AI level")
            name = "unidentified player"
        }
        return Player(id: player.id, name: name, rating: player.rating, title: player.title)
    }
}
