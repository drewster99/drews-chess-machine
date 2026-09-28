import Foundation

/// One candidate move and the probability the network's own policy (legal
/// moves only, temperature 1) gives it.
struct LichessBotMoveCandidate: Sendable, Equatable, Codable {
    let uci: String
    let probability: Float
}

/// Everything DCM decided for one move, and what it cost. Recorded per move
/// in the game journal (plan §10.6).
struct LichessBotMoveDecision: Sendable, Equatable, Codable {
    /// The move in DCM's standard UCI form — what is sent to Lichess.
    let uci: String
    let san: String
    /// Probability the *sampling* distribution (after temperature) put on
    /// the chosen move.
    let chosenProbability: Float
    /// The network's own policy over legal moves (temperature 1), best
    /// first, truncated.
    let topMoves: [LichessBotMoveCandidate]
    /// Value head, from the side to move — our side.
    let win: Float
    let draw: Float
    let loss: Float
    let temperature: Float
    let legalMoveCount: Int
    /// The post-temperature distribution was essentially uniform: the move
    /// was close to a random pick rather than a network opinion.
    let randomish: Bool
    let encodeMilliseconds: Double
    let inferenceMilliseconds: Double
    let sampleMilliseconds: Double

    /// Expected score for our side: `p_win + ½·p_draw`.
    var expectedScore: Float {
        win + 0.5 * draw
    }
}

enum LichessBotMoveChooserError: LocalizedError, Equatable {
    case noLegalMoves
    case evaluationProducedNoOutput

    var errorDescription: String? {
        switch self {
        case .noLegalMoves:
            return "No legal moves: the game is over, so there is no move to choose"
        case .evaluationProducedNoOutput:
            return "The network returned no policy for the position"
        }
    }
}

/// The inputs a move decision needs, captured on the game session's actor so
/// only `Sendable` values cross into the network call.
struct LichessBotMoveRequest: Sendable {
    let state: GameState
    /// `ChessGameEngine.recentStates` — the history the encoder threads, the
    /// same as self-play and `--uci`.
    let history: [GameState]
    let legalMoves: [ChessMove]
    /// Game-total ply, which drives the temperature schedule.
    let ply: Int
}

/// Chooses a move with one forward pass: encode, evaluate (policy and W/D/L
/// together), then sample through `MoveSampler` — the same sampler every
/// other DCM player uses. No search.
enum LichessBotMoveChooser {

    static let defaultTopMoveCount = 5

    static func choose(
        _ request: LichessBotMoveRequest,
        network: ChessMPSNetwork,
        schedule: SamplingSchedule,
        topMoveCount: Int = defaultTopMoveCount
    ) async throws -> LichessBotMoveDecision {
        guard !request.legalMoves.isEmpty else {
            throw LichessBotMoveChooserError.noLegalMoves
        }
        let clock = ContinuousClock()

        let encodeStart = clock.now
        let board = BoardEncoder.encode(request.state, history: request.history, encoding: network.inputEncoding)
        let encodeElapsed = clock.now - encodeStart

        let inferenceStart = clock.now
        let output = SyncBox<(logits: [Float], win: Float, draw: Float, loss: Float)?>(nil)
        try await network.evaluateWithValueDistribution(board: board) { policy, wdl in
            output.value = (logits: Array(policy), win: wdl.win, draw: wdl.draw, loss: wdl.loss)
        }
        let inferenceElapsed = clock.now - inferenceStart
        guard let evaluated = output.value else {
            throw LichessBotMoveChooserError.evaluationProducedNoOutput
        }

        let sampleStart = clock.now
        let player = request.state.currentPlayer
        let sample = sampleMove(logits: evaluated.logits, request: request, schedule: schedule)
        let topMoves = policyTopMoves(logits: evaluated.logits, legalMoves: request.legalMoves, currentPlayer: player, count: topMoveCount)
        let san = try SANFormatter.san(for: sample.move, in: request.state, legalMoves: request.legalMoves)
        let sampleElapsed = clock.now - sampleStart

        return LichessBotMoveDecision(
            uci: sample.move.uci,
            san: san,
            chosenProbability: sample.chosenProbability,
            topMoves: topMoves,
            win: evaluated.win,
            draw: evaluated.draw,
            loss: evaluated.loss,
            temperature: schedule.tau(forPly: request.ply),
            legalMoveCount: request.legalMoves.count,
            randomish: sample.randomish,
            encodeMilliseconds: milliseconds(encodeElapsed),
            inferenceMilliseconds: milliseconds(inferenceElapsed),
            sampleMilliseconds: milliseconds(sampleElapsed)
        )
    }

    private static func sampleMove(logits: [Float], request: LichessBotMoveRequest, schedule: SamplingSchedule) -> MoveSampler.Result {
        let probs = UnsafeMutableBufferPointer<Float>.allocate(capacity: MoveSampler.scratchCapacity)
        let eta = UnsafeMutableBufferPointer<Float>.allocate(capacity: MoveSampler.scratchCapacity)
        probs.initialize(repeating: 0)
        eta.initialize(repeating: 0)
        defer {
            probs.deallocate()
            eta.deallocate()
        }
        return logits.withUnsafeBufferPointer { logitsBuffer in
            MoveSampler.sampleMove(
                logits: logitsBuffer,
                legalMoves: request.legalMoves,
                currentPlayer: request.state.currentPlayer,
                ply: request.ply,
                schedule: schedule,
                probsScratch: probs,
                etaScratch: eta
            )
        }
    }

    /// The network's own opinion over legal moves: a softmax over the legal
    /// moves' logits at temperature 1, best first. Independent of the
    /// temperature used to sample, so records show what the network thought
    /// rather than what the schedule did to it.
    static func policyTopMoves(logits: [Float], legalMoves: [ChessMove], currentPlayer: PieceColor, count: Int) -> [LichessBotMoveCandidate] {
        guard !legalMoves.isEmpty, count > 0 else { return [] }
        let legalLogits = legalMoves.map { logits[PolicyEncoding.policyIndex($0, currentPlayer: currentPlayer)] }
        guard let maximum = legalLogits.max() else { return [] }
        let exponentials = legalLogits.map { expf($0 - maximum) }
        let total = exponentials.reduce(0, +)
        let ranked = zip(legalMoves, exponentials)
            .map { LichessBotMoveCandidate(uci: $0.0.uci, probability: $0.1 / total) }
            .sorted { $0.probability > $1.probability }
        return Array(ranked.prefix(count))
    }

    private static func milliseconds(_ duration: Duration) -> Double {
        LichessBotBackoff.seconds(duration) * 1000
    }
}
