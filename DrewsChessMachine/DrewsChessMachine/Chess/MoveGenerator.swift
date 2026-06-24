import Foundation

/// Generates legal chess moves, applies moves to produce new game states,
/// and detects check/checkmate/stalemate.
enum MoveGenerator {

    // MARK: - Public API

    /// All legal moves for the current player. Uses the pin-based generator
    /// (`legalMovesPinBased`); `legalMovesCopyMake` is the equivalent reference
    /// implementation, validated against it by perft and the live cross-check.
    static func legalMoves(for state: GameState) -> [ChessMove] {
        let legal = legalMovesPinBased(for: state)
        if crosscheckMovegen { crosscheckGenerators(pinBased: legal, state: state) }
        return legal
    }

    /// Reference legality filter: copy-make — allocate a fresh state per
    /// candidate and keep those that don't leave the king in check. Retained as
    /// the gold reference for perft and the cross-check oracle; the production
    /// `legalMoves` uses the faster `legalMovesPinBased`.
    static func legalMovesCopyMake(for state: GameState) -> [ChessMove] {
        let color = state.currentPlayer
        return pseudoLegalMoves(for: state).filter { move in
            !isInCheck(applyMove(move, to: state), color: color)
        }
    }

    /// When `--crosscheck-movegen` is passed, every `legalMoves` call also runs
    /// the make/unmake (B) and pin-based (C) generators and logs any position
    /// whose move set disagrees with this reference. A self-play soak guard,
    /// off (a single bool check) otherwise. Evaluated once, thread-safe.
    static let crosscheckMovegen = CommandLine.arguments.contains("--crosscheck-movegen")

    /// One-time stderr notice on the first cross-check, so a soak can positively
    /// confirm the cross-check is active rather than silently disabled.
    private static let announceCrosscheck: Void = {
        FileHandle.standardError.write(Data("[MOVEGEN-CROSSCHECK] active\n".utf8))
    }()

    private static func crosscheckGenerators(pinBased: [ChessMove], state: GameState) {
        _ = announceCrosscheck
        let ref = Set(legalMovesCopyMake(for: state))   // copy-make oracle
        let makeUnmake = Set(legalMovesMakeUnmake(for: state))
        let pin = Set(pinBased)                          // the production result
        guard makeUnmake != ref || pin != ref else { return }
        func diff(_ other: Set<ChessMove>) -> String {
            "missing=\(ref.subtracting(other).map(uciString)) extra=\(other.subtracting(ref).map(uciString))"
        }
        var parts: [String] = []
        if makeUnmake != ref { parts.append("B[\(diff(makeUnmake))]") }
        if pin != ref { parts.append("C[\(diff(pin))]") }
        // Log AND write to stderr: a divergence is a correctness failure that
        // must surface regardless of whether SessionLogger is started (it isn't
        // in UCI mode).
        let msg = "[MOVEGEN-CROSSCHECK] divergence FEN='\(FENParser.fen(from: state))' \(parts.joined(separator: " "))"
        SessionLogger.shared.log(msg)
        FileHandle.standardError.write(Data((msg + "\n").utf8))
    }

    private static func uciString(_ m: ChessMove) -> String {
        let files = Array("abcdefgh")
        let promo = m.promotion.map { "=\($0)" } ?? ""
        return "\(files[m.fromCol])\(8 - m.fromRow)\(files[m.toCol])\(8 - m.toRow)\(promo)"
    }

    /// Every move the current player's pieces can make by the movement rules,
    /// WITHOUT removing those that leave the mover's own king in check. This is
    /// the unfiltered basis `legalMoves` filters. It's much cheaper than
    /// `legalMoves` (no per-move apply + check scan), so a caller that only
    /// needs to resolve one specific move — e.g. matching a single SAN token —
    /// can generate these and legality-check just the matched candidate instead
    /// of legality-filtering the whole list.
    static func pseudoLegalMoves(for state: GameState) -> [ChessMove] {
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
        return moves
    }

    /// (B) Make/unmake variant of `legalMoves`: the same result set, but it
    /// tests each pseudo-legal move's legality by applying it **in place** on a
    /// single board buffer and undoing it — no fresh `GameState` (and no board
    /// array) allocated per candidate, the dominant cost of the copy-make
    /// filter. Validated move-for-move against `legalMoves` by perft.
    static func legalMovesMakeUnmake(for state: GameState) -> [ChessMove] {
        let pseudo = pseudoLegalMoves(for: state)
        let color = state.currentPlayer
        let kingHome = kingSquare(of: color, board: state.board)

        var legal: [ChessMove] = []
        legal.reserveCapacity(pseudo.count)
        var board = state.board
        board.withUnsafeMutableBufferPointer { buf in
            for move in pseudo where moveLeavesKingSafe(move, on: buf, kingHome: kingHome, color: color) {
                legal.append(move)
            }
        }
        return legal
    }

    /// (C) Pin/check-aware legality. A pin set and the in-check flag are
    /// computed **once** per position; then any move that is not in check, not a
    /// king move, not en passant, and not from a pinned square is legal with NO
    /// per-move apply — such a move provably cannot expose the king. The
    /// residual (king moves, pinned pieces, in-check evasions, en passant) is
    /// verified with the same make/unmake test. Validated move-for-move against
    /// `legalMoves` by perft.
    static func legalMovesPinBased(for state: GameState) -> [ChessMove] {
        let pseudo = pseudoLegalMoves(for: state)
        let color = state.currentPlayer
        let kingHome = kingSquare(of: color, board: state.board)

        var legal: [ChessMove] = []
        legal.reserveCapacity(pseudo.count)
        var board = state.board
        board.withUnsafeMutableBufferPointer { buf in
            let inCheck = isSquareAttacked(board: UnsafeBufferPointer(buf),
                                           row: kingHome / 8, col: kingHome % 8, by: color.opposite)
            let pinned = pinnedSquares(board: UnsafeBufferPointer(buf), kingIndex: kingHome, color: color)
            for move in pseudo {
                let fromIndex = move.fromRow * 8 + move.fromCol
                let isKingMove = fromIndex == kingHome
                let isEnPassant = !isKingMove
                    && move.toCol != move.fromCol
                    && buf[fromIndex]?.type == .pawn
                    && buf[move.toRow * 8 + move.toCol] == nil
                if !inCheck && !isKingMove && !isEnPassant && (pinned & (UInt64(1) << fromIndex)) == 0 {
                    legal.append(move)            // provably cannot expose the king
                } else if moveLeavesKingSafe(move, on: buf, kingHome: kingHome, color: color) {
                    legal.append(move)
                }
            }
        }
        return legal
    }

    /// Index of `color`'s king on `board` (0 if absent — only in malformed
    /// positions, which the generators are never asked about).
    private static func kingSquare(of color: PieceColor, board: [Piece?]) -> Int {
        let king = Piece(type: .king, color: color)
        for i in 0..<64 where board[i] == king { return i }
        return 0
    }

    /// Apply `move` to `buf`, test whether `color`'s king (relocated to the move
    /// destination if the king itself moved) is attacked, then restore `buf`
    /// exactly. Replicates `applyMove`'s board mutations (capture, en passant,
    /// castling rook move, promotion). The single make/unmake legality
    /// primitive shared by both alternative generators.
    private static func moveLeavesKingSafe(_ move: ChessMove,
                                           on buf: UnsafeMutableBufferPointer<Piece?>,
                                           kingHome: Int,
                                           color: PieceColor) -> Bool {
        let fromIndex = move.fromRow * 8 + move.fromCol
        let toIndex = move.toRow * 8 + move.toCol
        guard let moving = buf[fromIndex] else {
            preconditionFailure("moveLeavesKingSafe: move from an empty square")
        }
        let captured = buf[toIndex]
        let isEnPassant = moving.type == .pawn && move.toCol != move.fromCol && captured == nil
        let epIndex = move.fromRow * 8 + move.toCol
        let epCaptured = isEnPassant ? buf[epIndex] : nil
        let isCastle = moving.type == .king && abs(move.toCol - move.fromCol) == 2
        let rowBase = move.fromRow * 8
        let kingside = move.toCol > move.fromCol
        let rookFrom = isCastle ? (kingside ? rowBase + 7 : rowBase + 0) : 0
        let rookTo = isCastle ? (kingside ? rowBase + 5 : rowBase + 3) : 0
        let rookPiece = isCastle ? buf[rookFrom] : nil

        buf[fromIndex] = nil
        buf[toIndex] = move.promotion.map { Piece(type: $0, color: moving.color) } ?? moving
        if isEnPassant { buf[epIndex] = nil }
        if isCastle { buf[rookTo] = rookPiece; buf[rookFrom] = nil }

        let kingIndex = moving.type == .king ? toIndex : kingHome
        let safe = !isSquareAttacked(board: UnsafeBufferPointer(buf),
                                     row: kingIndex / 8, col: kingIndex % 8, by: color.opposite)

        buf[fromIndex] = moving
        buf[toIndex] = captured
        if isEnPassant { buf[epIndex] = epCaptured }
        if isCastle { buf[rookFrom] = rookPiece; buf[rookTo] = nil }

        return safe
    }

    /// Squares of `color`'s pieces absolutely pinned to its king — a friendly
    /// piece alone on a king ray with an enemy slider of the matching ray type
    /// (or a queen) behind it. Computed once from the king outward along the
    /// eight rays.
    /// Bitboard of `color`'s pinned-piece squares (bit `i` = square `i`), so the
    /// hot path tests membership with a bit-and instead of a `Set` lookup and
    /// the per-call set allocation.
    private static func pinnedSquares(board: UnsafeBufferPointer<Piece?>, kingIndex: Int, color: PieceColor) -> UInt64 {
        var pinned: UInt64 = 0
        let kr = kingIndex / 8, kc = kingIndex % 8
        for o in diagonals {
            if let s = pinAlongRay(board: board, kr: kr, kc: kc, dr: o.dr, dc: o.dc, color: color, slider: .bishop) {
                pinned |= UInt64(1) << s
            }
        }
        for o in straights {
            if let s = pinAlongRay(board: board, kr: kr, kc: kc, dr: o.dr, dc: o.dc, color: color, slider: .rook) {
                pinned |= UInt64(1) << s
            }
        }
        return pinned
    }

    /// Walk one ray from the king: if the first piece is ours and the next piece
    /// along the ray is an enemy `slider` (or queen), our piece is pinned —
    /// return its square. Otherwise nil.
    private static func pinAlongRay(board: UnsafeBufferPointer<Piece?>,
                                    kr: Int, kc: Int, dr: Int, dc: Int,
                                    color: PieceColor, slider: PieceType) -> Int? {
        var r = kr + dr, c = kc + dc
        var ownSquare: Int? = nil
        while r >= 0, r < 8, c >= 0, c < 8 {
            if let p = board[r * 8 + c] {
                if p.color == color {
                    if ownSquare != nil { return nil }   // second friendly blocker → no pin
                    ownSquare = r * 8 + c
                } else {
                    guard let own = ownSquare else { return nil }   // enemy adjacent → check, not pin
                    return (p.type == slider || p.type == .queen) ? own : nil
                }
            }
            r += dr; c += dc
        }
        return nil
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
        state.board.withUnsafeBufferPointer { board in
            isSquareAttacked(board: board, row: row, col: col, by: attackerColor)
        }
    }

    /// Board-buffer overload of `isSquareAttacked`, so a legality test needs no
    /// `GameState` wrapper. Used by the make/unmake generator, which mutates a
    /// single board buffer in place rather than allocating a state per move.
    static func isSquareAttacked(board: UnsafeBufferPointer<Piece?>, row: Int, col: Int, by attackerColor: PieceColor) -> Bool {
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
