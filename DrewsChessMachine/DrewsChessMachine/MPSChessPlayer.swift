import Foundation

// MARK: - Training Data

/// One recorded position from a game, for training the network.
/// The outcome is filled in after the game ends.
struct TrainingPosition: Sendable {
    /// Board tensor from the current player's perspective (18×8×8 = 1,152 floats).
    let inputTensor: [Float]
    /// Policy index (0-4095) of the move that was played, in the network's coordinate
    /// system (flipped for black). Matches the network's policy output encoding.
    let policyIndex: Int
    /// Game outcome from this position's player perspective: +1 win, 0 draw, -1 loss.
    /// Set to 0 initially, updated when the game ends.
    var outcome: Float
}

// MARK: - MPS Chess Player

/// A chess player that uses a neural network (ChessMPSNetwork) to choose moves.
///
/// Encodes the board from the current player's perspective, runs inference,
/// masks illegal moves, renormalizes, and samples from the policy distribution.
/// Records all positions played for later use as training data.
final class MPSChessPlayer: ChessPlayer {
    let identifier: String
    let name: String
    private let network: ChessMPSNetwork
    /// Optional replay buffer. When non-nil, this player pushes each finished
    /// game's labeled positions into the buffer from `onGameEnded`. The
    /// default-nil path is what Play Game and Play Continuous use — they
    /// have no use for training data, and passing nil keeps their behavior
    /// identical to before the buffer existed.
    private let replayBuffer: ReplayBuffer?
    private var isWhite = true

    /// Positions recorded during the current game. Cleared on each new game.
    /// Outcomes are filled in when onGameEnded is called.
    private(set) var gamePositions: [TrainingPosition] = []

    /// Create a player backed by a neural network.
    /// For self-play, two players can share the same network — they take turns,
    /// and BoardEncoder.encode handles the perspective flip automatically.
    /// Pass a `replayBuffer` to have this player contribute its labeled
    /// positions to a shared training pool at game end; leave it nil for
    /// normal (non-training) play.
    init(name: String, network: ChessMPSNetwork, replayBuffer: ReplayBuffer? = nil) {
        self.identifier = UUID().uuidString
        self.name = name
        self.network = network
        self.replayBuffer = replayBuffer
    }

    func onNewGame(_ isWhite: Bool) {
        self.isWhite = isWhite
        gamePositions = []
    }

    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove {
        guard !legalMoves.isEmpty else {
            throw ChessPlayerError.noLegalMoves
        }

        let flip = gameState.currentPlayer == .black
        let tensor = BoardEncoder.encode(gameState)
        let (policy, _) = try network.evaluate(board: tensor)
        let move = sampleMove(from: policy, legalMoves: legalMoves, flip: flip)

        // Record this position for training — policyIndex in the network's coordinate system
        gamePositions.append(TrainingPosition(
            inputTensor: tensor,
            policyIndex: Self.networkPolicyIndex(for: move, flip: flip),
            outcome: 0
        ))

        return move
    }

    func onGameEnded(_ result: GameResult, finalState: GameState) {
        let myOutcome: Float
        switch result {
        case .checkmate(let winner):
            myOutcome = (winner == .white) == isWhite ? 1.0 : -1.0
        case .stalemate,
             .drawByFiftyMoveRule,
             .drawByInsufficientMaterial,
             .drawByThreefoldRepetition:
            myOutcome = 0.0
        }

        for i in gamePositions.indices {
            gamePositions[i].outcome = myOutcome
        }

        // Push this half-game's now-labeled positions into the shared
        // replay buffer if one was provided. Both players in a self-play
        // session push independently so the buffer sees positions from
        // both perspectives.
        if let replayBuffer {
            replayBuffer.append(contentsOf: gamePositions)
        }
    }

    // MARK: - Coordinate Mapping

    /// Convert a move's absolute coordinates to the network's policy index.
    ///
    /// The network always sees the board from the current player's perspective.
    /// When black plays, BoardEncoder flips rows (row → 7-row). The policy output
    /// uses this same flipped coordinate system, so move indices must be flipped
    /// to match. Files (columns) stay unchanged — only ranks flip.
    private static func networkPolicyIndex(for move: ChessMove, flip: Bool) -> Int {
        let fromRow = flip ? (7 - move.fromRow) : move.fromRow
        let toRow = flip ? (7 - move.toRow) : move.toRow
        let fromSquare = fromRow * 8 + move.fromCol
        let toSquare = toRow * 8 + move.toCol
        return fromSquare * 64 + toSquare
    }

    // MARK: - Move Sampling

    /// Sample a move from the policy distribution over legal moves.
    ///
    /// The network emits raw logits, not a softmax distribution — softmax is
    /// fused with the legal-move mask here on the CPU. We exponentiate only
    /// the ~30 legal-move logits (with max-subtract for numerical stability)
    /// rather than running softmax over all 4096 slots and then masking.
    ///
    /// Performance-critical: runs once per ply. legalMoves is guaranteed
    /// non-empty by the caller (game-end is detected before this call).
    private func sampleMove(from logits: [Float], legalMoves: [ChessMove], flip: Bool) -> ChessMove {
        // Gather logits for legal moves only.
        var values = legalMoves.map { logits[Self.networkPolicyIndex(for: $0, flip: flip)] }

        // Numerically stable softmax: subtract max, exp, then normalize.
        let maxLogit = values.max() ?? 0
        var sum: Float = 0
        for i in values.indices {
            let e = expf(values[i] - maxLogit)
            values[i] = e
            sum += e
        }
        // exp() is strictly positive, so sum is strictly positive whenever
        // legalMoves is non-empty.
        for i in values.indices { values[i] /= sum }

        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for i in 0..<values.count {
            cumulative += values[i]
            if r < cumulative {
                return legalMoves[i]
            }
        }

        // Floating-point rounding can leave the cumulative just shy of 1.0;
        // the last legal move catches that.
        return legalMoves[legalMoves.count - 1]
    }
}
