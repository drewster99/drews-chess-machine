import Foundation

/// Generates legal chess moves, applies moves to produce new game states,
/// and detects check/checkmate/stalemate.
enum MoveGenerator {

    // MARK: - Public API

    /// All legal moves for the current player in the given state.
    static func legalMoves(for state: GameState) -> [ChessMove] {
        var moves: [ChessMove] = []
        let color = state.currentPlayer

        for row in 0..<8 {
            for col in 0..<8 {
                guard let piece = state.board[row][col], piece.color == color else { continue }

                switch piece.type {
                case .pawn:   moves.append(contentsOf: pawnMoves(row: row, col: col, color: color, state: state))
                case .knight: moves.append(contentsOf: jumpMoves(row: row, col: col, color: color, offsets: knightOffsets, state: state))
                case .bishop: moves.append(contentsOf: slidingMoves(row: row, col: col, color: color, directions: diagonals, state: state))
                case .rook:   moves.append(contentsOf: slidingMoves(row: row, col: col, color: color, directions: straights, state: state))
                case .queen:  moves.append(contentsOf: slidingMoves(row: row, col: col, color: color, directions: allDirections, state: state))
                case .king:   moves.append(contentsOf: kingMoves(row: row, col: col, color: color, state: state))
                }
            }
        }

        // Filter: only keep moves that don't leave our own king in check
        return moves.filter { move in
            let newState = applyMove(move, to: state)
            return !isInCheck(newState, color: color)
        }
    }

    /// Whether the given color's king is in check.
    static func isInCheck(_ state: GameState, color: PieceColor) -> Bool {
        for row in 0..<8 {
            for col in 0..<8 {
                if let piece = state.board[row][col], piece.type == .king, piece.color == color {
                    return isSquareAttacked(state, row: row, col: col, by: color.opposite)
                }
            }
        }
        return false
    }

    /// Apply a move to produce a new game state. Does not validate legality.
    static func applyMove(_ move: ChessMove, to state: GameState) -> GameState {
        var board = state.board
        guard let piece = board[move.fromRow][move.fromCol] else {
            // No piece at source — caller error. Return state unchanged.
            return state
        }

        // Detect en passant capture before modifying the board
        let isEnPassant = piece.type == .pawn
            && move.toCol != move.fromCol
            && state.board[move.toRow][move.toCol] == nil
        let isCapture = state.board[move.toRow][move.toCol] != nil || isEnPassant

        // Move the piece
        board[move.fromRow][move.fromCol] = nil
        if let promo = move.promotion {
            board[move.toRow][move.toCol] = Piece(type: promo, color: piece.color)
        } else {
            board[move.toRow][move.toCol] = piece
        }

        // En passant: remove the captured pawn
        if isEnPassant {
            board[move.fromRow][move.toCol] = nil
        }

        // Castling: move the rook
        if piece.type == .king && abs(move.toCol - move.fromCol) == 2 {
            if move.toCol > move.fromCol {
                // Kingside: rook h-file → f-file
                board[move.fromRow][5] = board[move.fromRow][7]
                board[move.fromRow][7] = nil
            } else {
                // Queenside: rook a-file → d-file
                board[move.fromRow][3] = board[move.fromRow][0]
                board[move.fromRow][0] = nil
            }
        }

        // Update castling rights
        var wk = state.whiteKingsideCastle
        var wq = state.whiteQueensideCastle
        var bk = state.blackKingsideCastle
        var bq = state.blackQueensideCastle

        // King moves → lose both sides
        if piece.type == .king {
            if piece.color == .white { wk = false; wq = false }
            else                     { bk = false; bq = false }
        }

        // Rook leaves or is captured on its home square
        if move.fromRow == 7 && move.fromCol == 7 { wk = false }
        if move.fromRow == 7 && move.fromCol == 0 { wq = false }
        if move.fromRow == 0 && move.fromCol == 7 { bk = false }
        if move.fromRow == 0 && move.fromCol == 0 { bq = false }
        if move.toRow == 7 && move.toCol == 7 { wk = false }
        if move.toRow == 7 && move.toCol == 0 { wq = false }
        if move.toRow == 0 && move.toCol == 7 { bk = false }
        if move.toRow == 0 && move.toCol == 0 { bq = false }

        // En passant target: set if pawn double-pushed, clear otherwise
        var ep: (row: Int, col: Int)? = nil
        if piece.type == .pawn && abs(move.toRow - move.fromRow) == 2 {
            ep = (row: (move.fromRow + move.toRow) / 2, col: move.fromCol)
        }

        // Halfmove clock: reset on pawn move or capture, otherwise increment
        let halfmove = (piece.type == .pawn || isCapture) ? 0 : state.halfmoveClock + 1

        return GameState(
            board: board,
            currentPlayer: state.currentPlayer.opposite,
            whiteKingsideCastle: wk,
            whiteQueensideCastle: wq,
            blackKingsideCastle: bk,
            blackQueensideCastle: bq,
            enPassantSquare: ep,
            halfmoveClock: halfmove
        )
    }

    // MARK: - Attack Detection

    /// Whether a square is attacked by any piece of the given color.
    static func isSquareAttacked(_ state: GameState, row: Int, col: Int, by attackerColor: PieceColor) -> Bool {
        // Pawn attacks — an attacking pawn sits one row "behind" the target from its perspective
        let pawnSourceRow = row + (attackerColor == .white ? 1 : -1)
        if pawnSourceRow >= 0, pawnSourceRow < 8 {
            for dc in [-1, 1] {
                let pc = col + dc
                if pc >= 0, pc < 8,
                   let p = state.board[pawnSourceRow][pc],
                   p.color == attackerColor, p.type == .pawn {
                    return true
                }
            }
        }

        // Knight attacks
        for (dr, dc) in knightOffsets {
            let r = row + dr, c = col + dc
            if r >= 0, r < 8, c >= 0, c < 8,
               let p = state.board[r][c],
               p.color == attackerColor, p.type == .knight {
                return true
            }
        }

        // King attacks
        for (dr, dc) in allDirections {
            let r = row + dr, c = col + dc
            if r >= 0, r < 8, c >= 0, c < 8,
               let p = state.board[r][c],
               p.color == attackerColor, p.type == .king {
                return true
            }
        }

        // Sliding: bishop or queen on diagonals
        for (dr, dc) in diagonals {
            if let p = firstPieceAlong(state: state, row: row, col: col, dr: dr, dc: dc) {
                if p.color == attackerColor, p.type == .bishop || p.type == .queen {
                    return true
                }
            }
        }

        // Sliding: rook or queen on straights
        for (dr, dc) in straights {
            if let p = firstPieceAlong(state: state, row: row, col: col, dr: dr, dc: dc) {
                if p.color == attackerColor, p.type == .rook || p.type == .queen {
                    return true
                }
            }
        }

        return false
    }

    // MARK: - Piece Move Generators (Pseudo-Legal)

    private static func pawnMoves(row: Int, col: Int, color: PieceColor, state: GameState) -> [ChessMove] {
        var moves: [ChessMove] = []
        let dir = color == .white ? -1 : 1
        let startRank = color == .white ? 6 : 1
        let promoRank = color == .white ? 0 : 7
        let oneForward = row + dir

        guard oneForward >= 0, oneForward < 8 else { return moves }

        // Forward one
        if state.board[oneForward][col] == nil {
            appendPawnMove(&moves, from: (row, col), to: (oneForward, col), promoRank: promoRank)

            // Forward two from starting rank
            let twoForward = row + 2 * dir
            if row == startRank, twoForward >= 0, twoForward < 8, state.board[twoForward][col] == nil {
                moves.append(ChessMove(fromRow: row, fromCol: col, toRow: twoForward, toCol: col, promotion: nil))
            }
        }

        // Diagonal captures + en passant
        for dc in [-1, 1] {
            let cc = col + dc
            guard cc >= 0, cc < 8 else { continue }

            if let target = state.board[oneForward][cc], target.color != color {
                appendPawnMove(&moves, from: (row, col), to: (oneForward, cc), promoRank: promoRank)
            } else if let ep = state.enPassantSquare, ep.row == oneForward, ep.col == cc {
                moves.append(ChessMove(fromRow: row, fromCol: col, toRow: oneForward, toCol: cc, promotion: nil))
            }
        }

        return moves
    }

    private static func appendPawnMove(
        _ moves: inout [ChessMove],
        from: (Int, Int),
        to: (Int, Int),
        promoRank: Int
    ) {
        if to.0 == promoRank {
            for promo in [PieceType.queen, .rook, .bishop, .knight] {
                moves.append(ChessMove(fromRow: from.0, fromCol: from.1, toRow: to.0, toCol: to.1, promotion: promo))
            }
        } else {
            moves.append(ChessMove(fromRow: from.0, fromCol: from.1, toRow: to.0, toCol: to.1, promotion: nil))
        }
    }

    private static func slidingMoves(
        row: Int, col: Int, color: PieceColor,
        directions: [(Int, Int)], state: GameState
    ) -> [ChessMove] {
        var moves: [ChessMove] = []
        for (dr, dc) in directions {
            var r = row + dr, c = col + dc
            while r >= 0, r < 8, c >= 0, c < 8 {
                if let p = state.board[r][c] {
                    if p.color != color {
                        moves.append(ChessMove(fromRow: row, fromCol: col, toRow: r, toCol: c, promotion: nil))
                    }
                    break
                }
                moves.append(ChessMove(fromRow: row, fromCol: col, toRow: r, toCol: c, promotion: nil))
                r += dr; c += dc
            }
        }
        return moves
    }

    private static func jumpMoves(
        row: Int, col: Int, color: PieceColor,
        offsets: [(Int, Int)], state: GameState
    ) -> [ChessMove] {
        var moves: [ChessMove] = []
        for (dr, dc) in offsets {
            let r = row + dr, c = col + dc
            guard r >= 0, r < 8, c >= 0, c < 8 else { continue }
            if let p = state.board[r][c], p.color == color { continue }
            moves.append(ChessMove(fromRow: row, fromCol: col, toRow: r, toCol: c, promotion: nil))
        }
        return moves
    }

    private static func kingMoves(row: Int, col: Int, color: PieceColor, state: GameState) -> [ChessMove] {
        // Normal king moves (one square in any direction)
        var moves = jumpMoves(row: row, col: col, color: color, offsets: allDirections, state: state)

        // Castling — king must be on its home square
        let homeRow = color == .white ? 7 : 0
        guard row == homeRow, col == 4 else { return moves }
        guard !isSquareAttacked(state, row: homeRow, col: 4, by: color.opposite) else { return moves }

        // Kingside
        let hasKingside = color == .white ? state.whiteKingsideCastle : state.blackKingsideCastle
        if hasKingside,
           state.board[homeRow][5] == nil,
           state.board[homeRow][6] == nil,
           !isSquareAttacked(state, row: homeRow, col: 5, by: color.opposite),
           !isSquareAttacked(state, row: homeRow, col: 6, by: color.opposite) {
            moves.append(ChessMove(fromRow: homeRow, fromCol: 4, toRow: homeRow, toCol: 6, promotion: nil))
        }

        // Queenside
        let hasQueenside = color == .white ? state.whiteQueensideCastle : state.blackQueensideCastle
        if hasQueenside,
           state.board[homeRow][3] == nil,
           state.board[homeRow][2] == nil,
           state.board[homeRow][1] == nil,
           !isSquareAttacked(state, row: homeRow, col: 3, by: color.opposite),
           !isSquareAttacked(state, row: homeRow, col: 2, by: color.opposite) {
            moves.append(ChessMove(fromRow: homeRow, fromCol: 4, toRow: homeRow, toCol: 2, promotion: nil))
        }

        return moves
    }

    // MARK: - Helpers

    /// First piece encountered along a ray from (row, col) in direction (dr, dc), exclusive of start.
    private static func firstPieceAlong(state: GameState, row: Int, col: Int, dr: Int, dc: Int) -> Piece? {
        var r = row + dr, c = col + dc
        while r >= 0, r < 8, c >= 0, c < 8 {
            if let p = state.board[r][c] { return p }
            r += dr; c += dc
        }
        return nil
    }

    // MARK: - Direction Tables

    private static let knightOffsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    private static let diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    private static let straights = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    private static let allDirections = diagonals + straights
}
