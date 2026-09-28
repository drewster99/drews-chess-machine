import Foundation

enum LichessBotPositionError: LocalizedError, Equatable {
    /// The game starts from a position other than the standard start.
    /// DCM's history and repetition planes would start empty there, out of
    /// its training distribution, so such games are not played (plan E3).
    case unsupportedInitialPosition(fen: String)
    /// A move in the server's list is not legal in the position DCM has
    /// reconstructed — local state and the server disagree (plan E5).
    case illegalMove(token: String, ply: Int, fen: String)

    var errorDescription: String? {
        switch self {
        case .unsupportedInitialPosition(let fen):
            return "Game starts from a non-standard position (\(fen)); DCM only plays from the standard start"
        case .illegalMove(let token, let ply, let fen):
            return "Server move \(token) at ply \(ply) is illegal in the reconstructed position \(fen)"
        }
    }
}

/// How a `sync` changed the tracked position.
enum LichessBotPositionSync: Equatable, Sendable {
    case unchanged
    /// New moves were applied on top of the existing ones.
    case extended(fromPly: Int, toPly: Int)
    /// The server's list was not an extension of the local one — a takeback
    /// or a divergence — so the position was rebuilt from the start
    /// (plan E4, E5).
    case rebuilt(fromPly: Int, toPly: Int)
}

/// The position of one Lichess game, as DCM reconstructs it from the
/// server's move list.
///
/// The position is a pure function of `(initialFen, moves)`: the server's
/// list is the truth, and DCM replays it through `ChessGameEngine` from the
/// start. Replaying produces exactly the board, repetition counts and
/// `recentStates` history self-play produces, so DCM's encoder sees the same
/// planes it was trained on (plan §4). When the new list extends the old
/// one, only the new moves are applied; otherwise — a takeback shortened
/// it, or a reconnect's `gameFull` disagrees — the position is rebuilt from
/// scratch rather than patched.
///
/// The engine runs with `.serverAuthoritative` adjudication: only checkmate
/// and stalemate end the game locally, so a difference between DCM's and
/// Lichess's draw-rule definitions can never make DCM stop moving while the
/// server still considers the game live (plan §8.2, E8).
///
/// Not `Sendable`: each game session actor owns its tracker.
final class LichessBotPositionTracker {
    private(set) var engine: ChessGameEngine
    /// The server's move tokens exactly as given (castling may be in the
    /// king-to-rook form; plan E1).
    private(set) var tokens: [String] = []
    /// The moves those tokens resolved to, in DCM's internal form.
    private(set) var moves: [ChessMove] = []

    init(initialFen: String) throws {
        guard Self.isStandardStart(initialFen) else {
            throw LichessBotPositionError.unsupportedInitialPosition(fen: initialFen)
        }
        engine = ChessGameEngine(adjudication: .serverAuthoritative)
    }

    /// Game-total ply: the number of moves played.
    var ply: Int {
        tokens.count
    }

    var sideToMove: PieceColor {
        engine.state.currentPlayer
    }

    /// Bring the position to the server's move list.
    @discardableResult
    func sync(to newTokens: [String]) throws -> LichessBotPositionSync {
        let oldPly = tokens.count
        if newTokens == tokens {
            return .unchanged
        }
        if newTokens.count > tokens.count && Array(newTokens.prefix(tokens.count)) == tokens {
            for token in newTokens[tokens.count...] {
                try apply(token)
            }
            return .extended(fromPly: oldPly, toPly: tokens.count)
        }
        let rebuiltEngine = ChessGameEngine(adjudication: .serverAuthoritative)
        engine = rebuiltEngine
        tokens = []
        moves = []
        for token in newTokens {
            try apply(token)
        }
        return .rebuilt(fromPly: oldPly, toPly: tokens.count)
    }

    private func apply(_ token: String) throws {
        let state = engine.state
        guard let move = ChessMove.parseUCI(token, legal: engine.currentLegalMoves, state: state) else {
            throw LichessBotPositionError.illegalMove(token: token, ply: tokens.count, fen: FENParser.fen(from: state))
        }
        do {
            try engine.applyMoveAndAdvance(move)
        } catch {
            // `parseUCI` only returns moves from `currentLegalMoves`, so the
            // only way to get here is a game the engine has already ended by
            // mate or stalemate receiving a further move — a server/local
            // disagreement, reported the same way.
            throw LichessBotPositionError.illegalMove(token: token, ply: tokens.count, fen: FENParser.fen(from: state))
        }
        tokens.append(token)
        moves.append(move)
    }

    /// Whether `fen` is `"startpos"` or a FEN of the standard starting
    /// position (move counters ignored). Lichess reports a game from the
    /// start either way (plan E3).
    static func isStandardStart(_ fen: String) -> Bool {
        if fen == "startpos" {
            return true
        }
        let fields = fen.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count >= 4 else {
            return false
        }
        return fields[0] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
            && fields[1] == "w"
            && fields[2] == "KQkq"
            && fields[3] == "-"
    }
}
