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
    /// A one-line summary ("gameState ply 12", "POST move e2e4 · 200 · 41 ms").
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
    }

    let id: String
    let startedAt: Date
    private let ourAccountID: String

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
    private(set) var finishedAt: Date?
    private(set) var whiteClockMilliseconds: Int?
    private(set) var blackClockMilliseconds: Int?
    /// When the clocks above were received; views tick the side to move
    /// locally from here.
    private(set) var clocksReceivedAt: Date?
    private(set) var opponentOffersDraw = false
    private(set) var opponentProposesTakeback = false
    private(set) var opponentGoneClaimableInSeconds: Int?

    private(set) var transcript: [LichessBotTranscriptEntry] = []
    private(set) var chat: [ChatMessage] = []
    private(set) var anomalies: [String] = []
    private(set) var streamConnections = 0

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

    func apply(_ event: LichessBotGameEvent) {
        switch event {
        case .streamOpened(let attempt):
            streamConnections += 1
            note("stream opened" + (attempt > 0 ? " (reconnect, attempt \(attempt))" : ""), isProblem: attempt > 0)
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
            note("stream ended: \(reason)", isProblem: !isFinished)
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
            note(text, isProblem: false)
        case .chat:
            break
        case .anomaly(let text):
            anomalies.append(text)
            note(text, isProblem: true)
        case .stoppedMoving(let reason):
            anomalies.append("stopped moving: \(reason)")
            note("stopped moving: \(reason)", isProblem: true)
        case .tokenRejected(let detail):
            anomalies.append("token rejected: \(detail)")
            note("token rejected: \(detail)", isProblem: true)
        case .finished(let status, let winner, _):
            self.status = status.raw
            self.winner = winner?.raw
            if finishedAt == nil {
                finishedAt = Date()
            }
        }
    }

    /// The session for this game ended without the game finishing (the bot
    /// went offline, the gate closed, the token was rejected). The view
    /// stops its clocks; the record is settled later by reconciliation.
    func markSessionEnded(_ reason: String) {
        guard finishedAt == nil else { return }
        finishedAt = Date()
        status = "left unfinished"
        note("DCM stopped following this game: \(reason)", isProblem: true)
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
        switch line {
        case .gameFull(let full):
            appendTranscript(at: at, direction: .incoming, title: "gameFull · \(full.state.moveTokens.count) plies · \(full.state.status.raw)", detail: raw, isProblem: false)
            applyState(full.state, at: at)
        case .gameState(let state):
            appendTranscript(at: at, direction: .incoming, title: "gameState · \(state.moveTokens.count) plies · \(state.status.raw)", detail: raw, isProblem: false)
            applyState(state, at: at)
        case .chatLine(let message):
            appendTranscript(at: at, direction: .incoming, title: "chat · \(message.username)", detail: raw, isProblem: false)
            chat.append(ChatMessage(id: nextChatID, at: at, room: message.room.raw, username: message.username, text: message.text))
            nextChatID += 1
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
                    anomalies.append("move \(token) at ply \(index) does not replay; the board stops here")
                    break
                }
                let san: String
                do {
                    san = try SANFormatter.san(for: move, in: current, legalMoves: legal)
                } catch {
                    anomalies.append("SAN for \(token) at ply \(index): \(error.localizedDescription)")
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

    private func note(_ text: String, isProblem: Bool) {
        appendTranscript(at: Date(), direction: .note, title: text, detail: "", isProblem: isProblem)
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
