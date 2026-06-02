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
            let rowBase = row * 8
            for col in 0..<8 {
                guard let piece = state.board[rowBase + col], piece.color == color else { continue }

                switch piece.type {
                case .pawn:
                    moves.append(contentsOf: pawnMoves(row: row, col: col, color: color, state: state))
                case .knight:
                    moves.append(contentsOf: jumpMoves(
                        row: row, col: col, color: color, offsets: knightOffsets, state: state
                    ))
                case .bishop:
                    moves.append(contentsOf: slidingMoves(
                        row: row, col: col, color: color, directions: diagonals, state: state
                    ))
                case .rook:
                    moves.append(contentsOf: slidingMoves(
                        row: row, col: col, color: color, directions: straights, state: state
                    ))
                case .queen:
                    moves.append(contentsOf: slidingMoves(
                        row: row, col: col, color: color, directions: allDirections, state: state
                    ))
                case .king:
                    moves.append(contentsOf: kingMoves(row: row, col: col, color: color, state: state))
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
        for i in 0..<64 {
            if let piece = state.board[i], piece.type == .king, piece.color == color {
                return isSquareAttacked(state, row: i / 8, col: i % 8, by: color.opposite)
            }
        }
        return false
    }

    /// Apply a move to produce a new game state.
    ///
    /// **The move must be legal.** This is performance-critical code in the
    /// inner game loop, so legality is not checked here — callers (e.g.
    /// `ChessMachine`) are required to pass only moves drawn from
    /// `legalMoves(for:)`. If the source square is empty this traps via the
    /// `preconditionFailure` below; that's intentional — it surfaces caller
    /// bugs immediately rather than silently corrupting state.
    static func applyMove(_ move: ChessMove, to state: GameState) -> GameState {
        var board = state.board
        let fromIndex = move.fromRow * 8 + move.fromCol
        let toIndex = move.toRow * 8 + move.toCol

        // All board mutations run inside a single unsafe-buffer borrow. This
        // forces the copy-on-write check exactly once (the buffer is made
        // unique on entry), after which every read/write is a direct pointer
        // access — collapsing the half-dozen `subscript.modify` coroutine
        // invocations (each with its own uniqueness + bounds check) that this
        // function showed as a hot spot. `piece` and `isCapture` are carried
        // back out for the castling-rights and clock updates below; `isEnPassant`
        // is only needed inside, to remove the captured pawn and to derive
        // `isCapture`.
        let (piece, isCapture) = board.withUnsafeMutableBufferPointer {
            buf -> (Piece, Bool) in
            guard let piece = buf[fromIndex] else {
                preconditionFailure(
                    "applyMove: source square at row=\(move.fromRow) col=\(move.fromCol) is empty — caller passed an illegal/corrupt move"
                )
            }

            // Detect en passant capture before modifying the board
            let target = buf[toIndex]
            let isEnPassant = piece.type == .pawn
                && move.toCol != move.fromCol
                && target == nil
            let isCapture = target != nil || isEnPassant

            // Move the piece
            buf[fromIndex] = nil
            if let promo = move.promotion {
                buf[toIndex] = Piece(type: promo, color: piece.color)
            } else {
                buf[toIndex] = piece
            }

            // En passant: remove the captured pawn
            if isEnPassant {
                buf[move.fromRow * 8 + move.toCol] = nil
            }

            // Castling: move the rook
            if piece.type == .king && abs(move.toCol - move.fromCol) == 2 {
                let rowBase = move.fromRow * 8
                if move.toCol > move.fromCol {
                    // Kingside: rook h-file → f-file
                    buf[rowBase + 5] = buf[rowBase + 7]
                    buf[rowBase + 7] = nil
                } else {
                    // Queenside: rook a-file → d-file
                    buf[rowBase + 3] = buf[rowBase + 0]
                    buf[rowBase + 0] = nil
                }
            }

            return (piece, isCapture)
        }

        // Update castling rights
        var wk = state.whiteKingsideCastle
        var wq = state.whiteQueensideCastle
        var bk = state.blackKingsideCastle
        var bq = state.blackQueensideCastle

        // King moves → lose both sides
        if piece.type == .king {
            if piece.color == .white {
                wk = false
                wq = false
            } else {
                bk = false
                bq = false
            }
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
        var ep: (row: Int, col: Int)?
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
        // Bind the board buffer once: every lookup below reads `board` instead
        // of re-borrowing `state.board`, which keeps the array buffer's
        // refcount traffic out of this hot path.
        let board = state.board

        // Pawn attacks — an attacking pawn sits one row "behind" the target
        // from its perspective. The two diagonal source squares are checked
        // explicitly rather than via a `[-1, 1]` literal, which would heap-
        // allocate an array on every call.
        let pawnSourceRow = row + (attackerColor == .white ? 1 : -1)
        if pawnSourceRow >= 0, pawnSourceRow < 8 {
            let pawnRowBase = pawnSourceRow * 8
            let leftCol = col - 1
            if leftCol >= 0,
               let p = board[pawnRowBase + leftCol],
               p.color == attackerColor, p.type == .pawn {
                return true
            }
            let rightCol = col + 1
            if rightCol < 8,
               let p = board[pawnRowBase + rightCol],
               p.color == attackerColor, p.type == .pawn {
                return true
            }
        }

        // Knight attacks
        for o in knightOffsets {
            let r = row + o.dr, c = col + o.dc
            if r >= 0, r < 8, c >= 0, c < 8,
               let p = board[r * 8 + c],
               p.color == attackerColor, p.type == .knight {
                return true
            }
        }

        // Sliding pieces and the adjacent king, in a single pass over the
        // eight directions. Each ray is walked once: a piece found at distance
        // 1 that is the attacker's king is itself an attacker, and the first
        // piece encountered along the ray is a slider attacker when it matches
        // the ray's orientation — bishop/queen on a diagonal, rook/queen on a
        // straight. Folding the king test into each ray's first step replaces
        // the separate king loop, which previously re-read the eight squares
        // adjacent to the target that these rays already touch.
        for o in diagonals {
            let dr = o.dr, dc = o.dc
            var r = row + dr, c = col + dc
            var distance = 1
            while r >= 0, r < 8, c >= 0, c < 8 {
                if let p = board[r * 8 + c] {
                    if p.color == attackerColor,
                       p.type == .bishop || p.type == .queen || (distance == 1 && p.type == .king) {
                        return true
                    }
                    break
                }
                r += dr; c += dc
                distance += 1
            }
        }
        for o in straights {
            let dr = o.dr, dc = o.dc
            var r = row + dr, c = col + dc
            var distance = 1
            while r >= 0, r < 8, c >= 0, c < 8 {
                if let p = board[r * 8 + c] {
                    if p.color == attackerColor,
                       p.type == .rook || p.type == .queen || (distance == 1 && p.type == .king) {
                        return true
                    }
                    break
                }
                r += dr; c += dc
                distance += 1
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

        let oneForwardBase = oneForward * 8

        // Forward one
        if state.board[oneForwardBase + col] == nil {
            appendPawnMove(&moves, fromRow: row, fromCol: col, toRow: oneForward, toCol: col, promoRank: promoRank)

            // Forward two from starting rank
            let twoForward = row + 2 * dir
            if row == startRank, twoForward >= 0, twoForward < 8, state.board[twoForward * 8 + col] == nil {
                moves.append(ChessMove(fromRow: row, fromCol: col, toRow: twoForward, toCol: col, promotion: nil))
            }
        }

        // Diagonal captures + en passant
        for dc in [-1, 1] {
            let cc = col + dc
            guard cc >= 0, cc < 8 else { continue }

            if let target = state.board[oneForwardBase + cc], target.color != color {
                appendPawnMove(&moves, fromRow: row, fromCol: col, toRow: oneForward, toCol: cc, promoRank: promoRank)
            } else if let ep = state.enPassantSquare, ep.row == oneForward, ep.col == cc {
                moves.append(ChessMove(fromRow: row, fromCol: col, toRow: oneForward, toCol: cc, promotion: nil))
            }
        }

        return moves
    }

    private static func appendPawnMove(
        _ moves: inout [ChessMove],
        fromRow: Int, fromCol: Int,
        toRow: Int, toCol: Int,
        promoRank: Int
    ) {
        if toRow == promoRank {
            for promo in [PieceType.queen, .rook, .bishop, .knight] {
                moves.append(ChessMove(fromRow: fromRow, fromCol: fromCol, toRow: toRow, toCol: toCol, promotion: promo))
            }
        } else {
            moves.append(ChessMove(fromRow: fromRow, fromCol: fromCol, toRow: toRow, toCol: toCol, promotion: nil))
        }
    }

    private static func slidingMoves(
        row: Int, col: Int, color: PieceColor,
        directions: [Offset], state: GameState
    ) -> [ChessMove] {
        var moves: [ChessMove] = []
        for o in directions {
            let dr = o.dr, dc = o.dc
            var r = row + dr, c = col + dc
            while r >= 0, r < 8, c >= 0, c < 8 {
                if let p = state.board[r * 8 + c] {
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
        offsets: [Offset], state: GameState
    ) -> [ChessMove] {
        var moves: [ChessMove] = []
        for o in offsets {
            let dr = o.dr, dc = o.dc
            let r = row + dr, c = col + dc
            guard r >= 0, r < 8, c >= 0, c < 8 else { continue }
            if let p = state.board[r * 8 + c], p.color == color { continue }
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

        let homeBase = homeRow * 8
        let kingsideRook = Piece(type: .rook, color: color)
        let queensideRook = Piece(type: .rook, color: color)

        // Kingside
        let hasKingside = color == .white ? state.whiteKingsideCastle : state.blackKingsideCastle
        if hasKingside,
           state.board[homeBase + 7] == kingsideRook,
           state.board[homeBase + 5] == nil,
           state.board[homeBase + 6] == nil,
           !isSquareAttacked(state, row: homeRow, col: 5, by: color.opposite),
           !isSquareAttacked(state, row: homeRow, col: 6, by: color.opposite) {
            moves.append(ChessMove(fromRow: homeRow, fromCol: 4, toRow: homeRow, toCol: 6, promotion: nil))
        }

        // Queenside
        let hasQueenside = color == .white ? state.whiteQueensideCastle : state.blackQueensideCastle
        if hasQueenside,
           state.board[homeBase + 0] == queensideRook,
           state.board[homeBase + 3] == nil,
           state.board[homeBase + 2] == nil,
           state.board[homeBase + 1] == nil,
           !isSquareAttacked(state, row: homeRow, col: 3, by: color.opposite),
           !isSquareAttacked(state, row: homeRow, col: 2, by: color.opposite) {
            moves.append(ChessMove(fromRow: homeRow, fromCol: 4, toRow: homeRow, toCol: 2, promotion: nil))
        }

        return moves
    }

    // MARK: - Direction Tables

    /// A single (row, col) step. A concrete struct, deliberately **not** an
    /// `(Int, Int)` tuple: these tables are iterated on the per-ply move-
    /// generation hot path across many concurrent self-play workers, and tuple
    /// element types have no static type metadata — every access (iterating
    /// `[(Int,Int)]`, passing it to a function) calls `swift_getTupleTypeMetadata`,
    /// which serializes on a global locking metadata cache. Under high worker
    /// counts that cache contention dominated CPU (~66% in an Instruments
    /// trace). A struct has static metadata, so the runtime cache is never
    /// consulted. Order within each table is preserved from the original tuple
    /// form so generated move lists are byte-identical.
    struct Offset { let dr: Int; let dc: Int }

    private static let knightOffsets = [
        Offset(dr: -2, dc: -1), Offset(dr: -2, dc: 1), Offset(dr: -1, dc: -2), Offset(dr: -1, dc: 2),
        Offset(dr: 1, dc: -2), Offset(dr: 1, dc: 2), Offset(dr: 2, dc: -1), Offset(dr: 2, dc: 1)
    ]
    private static let diagonals = [
        Offset(dr: -1, dc: -1), Offset(dr: -1, dc: 1), Offset(dr: 1, dc: -1), Offset(dr: 1, dc: 1)
    ]
    private static let straights = [
        Offset(dr: -1, dc: 0), Offset(dr: 1, dc: 0), Offset(dr: 0, dc: -1), Offset(dr: 0, dc: 1)
    ]
    private static let allDirections = diagonals + straights
}
