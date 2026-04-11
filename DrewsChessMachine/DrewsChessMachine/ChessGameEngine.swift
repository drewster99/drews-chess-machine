import Foundation

// MARK: - Game Result

enum GameResult: Sendable {
    case checkmate(winner: PieceColor)
    case stalemate
    case drawByFiftyMoveRule
    case drawByInsufficientMaterial
}

// MARK: - Errors

enum ChessGameError: LocalizedError {
    case illegalMove(ChessMove)
    case gameAlreadyOver

    var errorDescription: String? {
        switch self {
        case .illegalMove(let move):
            return "Illegal move: \(move.notation)"
        case .gameAlreadyOver:
            return "The game has already ended"
        }
    }
}

// MARK: - Chess Game Engine

/// Manages a chess game between two players: validates moves, updates state,
/// detects game-ending conditions (checkmate, stalemate, draws).
final class ChessGameEngine {
    private(set) var state: GameState
    private(set) var result: GameResult?
    private(set) var moveHistory: [ChessMove] = []

    init(state: GameState = .starting) {
        self.state = state
    }

    /// All legal moves in the current position.
    func legalMoves() -> [ChessMove] {
        MoveGenerator.legalMoves(for: state)
    }

    /// Whether the current player is in check.
    func isInCheck() -> Bool {
        MoveGenerator.isInCheck(state, color: state.currentPlayer)
    }

    /// Apply a legal move, updating the game state. Throws if the move is illegal or the game is over.
    func applyMove(_ move: ChessMove) throws {
        guard result == nil else {
            throw ChessGameError.gameAlreadyOver
        }

        let legal = legalMoves()
        guard legal.contains(move) else {
            throw ChessGameError.illegalMove(move)
        }

        state = MoveGenerator.applyMove(move, to: state)
        moveHistory.append(move)
        updateResult()
    }

    /// Play a complete game between two players. Returns the result.
    func playGame(white: ChessPlayer, black: ChessPlayer) async -> GameResult {
        white.onNewGame(true)
        black.onNewGame(false)

        var lastMove: ChessMove?

        while result == nil {
            let currentPlayer = state.currentPlayer == .white ? white : black

            do {
                let move = try await currentPlayer.onChooseNextMove(
                    opponentMove: lastMove,
                    newGameState: state
                )
                try applyMove(move)
                lastMove = move
            } catch {
                // Player threw or returned illegal move — game over
                break
            }
        }

        guard let finalResult = result else {
            let finalResult = GameResult.stalemate
            white.onGameEnded(finalResult, finalState: state)
            black.onGameEnded(finalResult, finalState: state)
            return finalResult
        }

        white.onGameEnded(finalResult, finalState: state)
        black.onGameEnded(finalResult, finalState: state)
        return finalResult
    }

    // MARK: - Game End Detection

    private func updateResult() {
        let moves = MoveGenerator.legalMoves(for: state)

        if moves.isEmpty {
            if MoveGenerator.isInCheck(state, color: state.currentPlayer) {
                result = .checkmate(winner: state.currentPlayer.opposite)
            } else {
                result = .stalemate
            }
            return
        }

        if state.halfmoveClock >= 100 {
            result = .drawByFiftyMoveRule
            return
        }

        if isInsufficientMaterial() {
            result = .drawByInsufficientMaterial
        }
    }

    /// Detects positions where neither side can checkmate:
    /// K vs K, K+B vs K, K+N vs K.
    private func isInsufficientMaterial() -> Bool {
        var whitePieces: [PieceType] = []
        var blackPieces: [PieceType] = []

        for row in state.board {
            for square in row {
                guard let piece = square else { continue }
                if piece.color == .white {
                    whitePieces.append(piece.type)
                } else {
                    blackPieces.append(piece.type)
                }
            }
        }

        // Any pawns, rooks, or queens → sufficient material
        let hasMatingPiece: (PieceType) -> Bool = { type in
            type == .pawn || type == .rook || type == .queen
        }

        if whitePieces.contains(where: hasMatingPiece) || blackPieces.contains(where: hasMatingPiece) {
            return false
        }

        // Only kings, bishops, and knights remain
        let whiteMinors = whitePieces.filter { $0 != .king }.count
        let blackMinors = blackPieces.filter { $0 != .king }.count

        // K vs K
        if whiteMinors == 0 && blackMinors == 0 { return true }
        // K+minor vs K
        if whiteMinors <= 1 && blackMinors == 0 { return true }
        if whiteMinors == 0 && blackMinors <= 1 { return true }

        return false
    }
}
