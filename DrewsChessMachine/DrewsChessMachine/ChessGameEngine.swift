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
    case gameAlreadyOver

    var errorDescription: String? {
        switch self {
        case .gameAlreadyOver:
            return "The game has already ended"
        }
    }
}

// MARK: - Chess Game Engine

/// Manages a chess game between two players: applies moves, updates state,
/// detects game-ending conditions (checkmate, stalemate, draws).
///
/// Move legality is **not** validated here — the caller (`ChessMachine`) is
/// required to pass moves drawn from `MoveGenerator.legalMoves(for:)`. The
/// game loop already needs that list for player choice and end-detection, so
/// re-deriving it inside `applyMove` would be wasted work.
final class ChessGameEngine {
    private(set) var state: GameState
    private(set) var result: GameResult?
    private(set) var moveHistory: [ChessMove] = []

    init(state: GameState = .starting) {
        self.state = state
    }

    /// Apply a move and recompute the result given the legal moves available
    /// in the *new* position. The caller is responsible for generating
    /// `nextLegalMoves` after the move is applied — the engine reuses that
    /// list for end-of-game detection (no second `legalMoves` call).
    ///
    /// Returns the legal moves for the next position so the caller can hand
    /// them straight back to the next player.
    @discardableResult
    func applyMoveAndAdvance(_ move: ChessMove) throws -> [ChessMove] {
        guard result == nil else {
            throw ChessGameError.gameAlreadyOver
        }

        state = MoveGenerator.applyMove(move, to: state)
        moveHistory.append(move)

        let nextMoves = MoveGenerator.legalMoves(for: state)
        updateResult(nextLegalMoves: nextMoves)
        return nextMoves
    }

    // MARK: - Game End Detection

    private func updateResult(nextLegalMoves: [ChessMove]) {
        if nextLegalMoves.isEmpty {
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

        for square in state.board {
            guard let piece = square else { continue }
            if piece.color == .white {
                whitePieces.append(piece.type)
            } else {
                blackPieces.append(piece.type)
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
