import Foundation

/// Errors from `SANFormatter`.
enum SANFormatterError: LocalizedError {
    /// The move is not in the legal-move list for the position, so it has no
    /// SAN — formatting it would describe a move that cannot be played.
    case illegalMove(ChessMove)

    var errorDescription: String? {
        switch self {
        case .illegalMove(let move):
            return "Cannot format an illegal move as SAN: \(move.uci)"
        }
    }
}

/// Formats moves in Standard Algebraic Notation (SAN), as used in PGN:
/// `e4`, `Nf3`, `exd6`, `O-O`, `e8=Q+`, `Raxd1#`.
///
/// The codebase already *parses* SAN (`PGNImporter.resolveLegalSANMove`);
/// this is the writing side. Output follows the PGN standard's export
/// format:
/// - piece letters `K Q R B N`; pawns carry no letter
/// - disambiguation only when another piece of the same type can reach the
///   same square: the origin file if that suffices, otherwise the origin
///   rank, otherwise both
/// - `x` for captures; a pawn capture (including en passant) is prefixed
///   with its origin file
/// - `=Q` style promotion
/// - `O-O` / `O-O-O` for castling
/// - `+` for check and `#` for checkmate
///
/// Pure and stateless. Legality comes from `MoveGenerator`, the single
/// source of truth for which moves exist.
enum SANFormatter {

    /// SAN for `move` in `state`, generating the legal-move list itself.
    static func san(for move: ChessMove, in state: GameState) throws -> String {
        try san(for: move, in: state, legalMoves: MoveGenerator.legalMoves(for: state))
    }

    /// SAN for `move` in `state` given that position's legal moves — for
    /// callers that already hold them (a game engine's `currentLegalMoves`).
    ///
    /// Throws `SANFormatterError.illegalMove` if `move` is not in
    /// `legalMoves`.
    static func san(
        for move: ChessMove,
        in state: GameState,
        legalMoves: [ChessMove]
    ) throws -> String {
        guard legalMoves.contains(move),
              let mover = state.board[move.fromRow * 8 + move.fromCol] else {
            throw SANFormatterError.illegalMove(move)
        }

        let body: String
        if mover.type == .king && abs(move.toCol - move.fromCol) == 2 {
            body = move.toCol > move.fromCol ? "O-O" : "O-O-O"
        } else {
            let destination = BoardEncoder.squareName(move.toRow * 8 + move.toCol)
            let targetOccupied = state.board[move.toRow * 8 + move.toCol] != nil
            if mover.type == .pawn {
                // A pawn that changes file always captures — onto an occupied
                // square, or onto the empty en-passant target.
                let isCapture = move.fromCol != move.toCol
                var text = isCapture ? "\(fileLetter(move.fromCol))x\(destination)" : destination
                if let promotion = move.promotion {
                    text += "=\(pieceLetter(promotion))"
                }
                body = text
            } else {
                body = pieceLetter(mover.type)
                    + disambiguation(for: move, pieceType: mover.type, in: state, legalMoves: legalMoves)
                    + (targetOccupied ? "x" : "")
                    + destination
            }
        }

        return body + checkSuffix(after: move, in: state)
    }

    // MARK: - Private helpers

    /// The minimal origin qualifier that distinguishes `move` from every
    /// other legal move by a piece of the same type to the same square.
    private static func disambiguation(
        for move: ChessMove,
        pieceType: PieceType,
        in state: GameState,
        legalMoves: [ChessMove]
    ) -> String {
        let rivals = legalMoves.filter { other in
            other != move
                && other.toRow == move.toRow
                && other.toCol == move.toCol
                && state.board[other.fromRow * 8 + other.fromCol]?.type == pieceType
        }
        guard !rivals.isEmpty else { return "" }
        if !rivals.contains(where: { $0.fromCol == move.fromCol }) {
            return fileLetter(move.fromCol)
        }
        if !rivals.contains(where: { $0.fromRow == move.fromRow }) {
            return rankDigit(move.fromRow)
        }
        return fileLetter(move.fromCol) + rankDigit(move.fromRow)
    }

    /// `#` if `move` checkmates, `+` if it gives check, otherwise empty.
    private static func checkSuffix(after move: ChessMove, in state: GameState) -> String {
        let next = MoveGenerator.applyMove(move, to: state)
        guard MoveGenerator.isInCheck(next, color: next.currentPlayer) else {
            return ""
        }
        return MoveGenerator.legalMoves(for: next).isEmpty ? "#" : "+"
    }

    private static func pieceLetter(_ type: PieceType) -> String {
        switch type {
        case .pawn: return ""
        case .knight: return "N"
        case .bishop: return "B"
        case .rook: return "R"
        case .queen: return "Q"
        case .king: return "K"
        }
    }

    /// File letter for a board column (0 = a).
    private static func fileLetter(_ col: Int) -> String {
        String(UnicodeScalar(UInt8(ascii: "a") + UInt8(col)))
    }

    /// Rank digit for a board row (row 0 = rank 8).
    private static func rankDigit(_ row: Int) -> String {
        String(8 - row)
    }
}
