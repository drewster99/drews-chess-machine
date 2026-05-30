import Foundation

/// UCI long algebraic move notation helpers used by the UCI bridge
/// (cutechess, c-chess-cli, etc.). The UCI protocol speaks moves as
/// `<from><to>[promo]` where `<from>` and `<to>` are two-character
/// algebraic square names (a1-h8) and the optional promotion suffix
/// is a single lowercase piece letter from `q/r/b/n`.
///
/// Castling in UCI is encoded as the king's source-and-destination
/// pair (`e1g1`, `e1c1`, `e8g8`, `e8c8`). That happens to match the
/// engine's internal representation of castling (king moves two
/// squares), so no special-casing is needed.
///
/// En passant in UCI is just the diagonal pawn move (e.g. `e5d6`);
/// again, no special notation. The engine's legal-move list already
/// carries the en-passant move with the same from/to, so a match
/// against the legal list resolves it.
extension ChessMove {

    /// Render this move in UCI long algebraic notation.
    /// Examples: `"e2e4"`, `"e7e8q"`, `"e1g1"`.
    var uci: String {
        let from = BoardEncoder.squareName(fromRow * 8 + fromCol)
        let to = BoardEncoder.squareName(toRow * 8 + toCol)
        guard let promo = promotion else {
            return "\(from)\(to)"
        }
        let suffix: String
        switch promo {
        case .queen:  suffix = "q"
        case .rook:   suffix = "r"
        case .bishop: suffix = "b"
        case .knight: suffix = "n"
        default:      suffix = ""
        }
        return "\(from)\(to)\(suffix)"
    }

    /// Parse a UCI LAN move string against the supplied list of legal
    /// moves and return the matching `ChessMove`, or nil if the string
    /// is malformed or names a move that is not legal in the current
    /// position.
    ///
    /// Matches by `(fromRow, fromCol, toRow, toCol, promotion)` against
    /// the legal-move list, which is what lets castling / en passant /
    /// promotion all resolve without re-implementing chess rules on
    /// the UCI side — `MoveGenerator` is the single source of truth
    /// for what moves are legal.
    static func parseUCI(
        _ token: String,
        legal: [ChessMove]
    ) -> ChessMove? {
        let chars = Array(token)
        guard chars.count == 4 || chars.count == 5 else { return nil }
        guard let fromSq = parseSquare(file: chars[0], rank: chars[1]),
              let toSq = parseSquare(file: chars[2], rank: chars[3]) else {
            return nil
        }
        let promo: PieceType?
        if chars.count == 5 {
            switch chars[4] {
            case "q", "Q": promo = .queen
            case "r", "R": promo = .rook
            case "b", "B": promo = .bishop
            case "n", "N": promo = .knight
            default: return nil
            }
        } else {
            promo = nil
        }
        return legal.first { move in
            move.fromRow == fromSq.row
                && move.fromCol == fromSq.col
                && move.toRow == toSq.row
                && move.toCol == toSq.col
                && move.promotion == promo
        }
    }

    /// Parse one algebraic square (file char `a`-`h` + rank char
    /// `1`-`8`) into the engine's (row, col) coordinates where row
    /// 0 = rank 8 and row 7 = rank 1.
    private static func parseSquare(
        file: Character,
        rank: Character
    ) -> (row: Int, col: Int)? {
        guard let fileScalar = file.lowercased().unicodeScalars.first,
              let rankDigit = rank.wholeNumberValue else {
            return nil
        }
        // 97 = ASCII 'a' — file letters in UCI LAN are always lowercase
        // ASCII a-h after the `.lowercased()` normalization above.
        let fileValue = Int(fileScalar.value) - 97
        guard fileValue >= 0, fileValue < 8,
              rankDigit >= 1, rankDigit <= 8 else {
            return nil
        }
        return (row: 8 - rankDigit, col: fileValue)
    }
}
