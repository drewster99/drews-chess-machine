import Foundation

// MARK: - Errors

enum ChessPlayerError: LocalizedError {
    case noPlayerAvailable
    case noLegalMoves

    var errorDescription: String? {
        switch self {
        case .noPlayerAvailable:
            return "No player is available to make a move"
        case .noLegalMoves:
            return "No legal moves available"
        }
    }
}

// MARK: - Protocol

/// A chess player that can participate in games managed by ChessMachine.
///
/// Implementations: MPSChessPlayer (neural network), RandomPlayer, NullPlayer.
protocol ChessPlayer: AnyObject {
    var identifier: String { get }
    var name: String { get }

    /// Called at the start of a new game.
    func onNewGame(_ isWhite: Bool)

    /// Called when it's this player's turn. Returns a legal move or throws.
    /// Thrown errors abort the game — the engine treats it as a forfeit.
    ///
    /// `legalMoves` is the precomputed legal-move list for `gameState`. The
    /// engine generates this once per ply and shares it with the player to
    /// avoid duplicate generation. The player must return a move from this
    /// list (or one that is otherwise legal).
    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove

    /// Called when the game ends, regardless of outcome.
    func onGameEnded(_ result: GameResult, finalState: GameState)
}

// MARK: - Random Player

/// Picks a random legal move each turn. Useful as a baseline opponent or for testing.
final class RandomPlayer: ChessPlayer {
    let identifier = UUID().uuidString
    let name: String

    init(name: String = "Random") {
        self.name = name
    }

    func onNewGame(_ isWhite: Bool) {}

    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove {
        guard !legalMoves.isEmpty else {
            throw ChessPlayerError.noLegalMoves
        }
        return legalMoves[Int.random(in: 0..<legalMoves.count)]
    }

    func onGameEnded(_ result: GameResult, finalState: GameState) {}
}

// MARK: - Null Player

/// Always throws — a placeholder when no real player is available.
/// Useful for testing error handling in the game loop.
final class NullPlayer: ChessPlayer {
    let identifier = UUID().uuidString
    let name = "Null"

    func onNewGame(_ isWhite: Bool) {}

    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove {
        throw ChessPlayerError.noPlayerAvailable
    }

    func onGameEnded(_ result: GameResult, finalState: GameState) {}
}
