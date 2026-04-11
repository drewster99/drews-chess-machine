import Foundation

/// A chess move defined by source and destination squares, with optional promotion.
struct ChessMove: Sendable, Equatable {
    let fromRow: Int
    let fromCol: Int
    let toRow: Int
    let toCol: Int
    /// Promotion piece type when a pawn reaches the last rank. Nil for non-promotion moves.
    let promotion: PieceType?

    /// Policy tensor index for this move: from_square * 64 + to_square.
    /// Squares numbered row-by-row from rank 8: 0=a8, 7=h8, ..., 56=a1, 63=h1.
    var policyIndex: Int {
        let fromSquare = fromRow * 8 + fromCol
        let toSquare = toRow * 8 + toCol
        return fromSquare * 64 + toSquare
    }

    /// Algebraic notation for display (e.g., "e2-e4", "a7-a8=Q").
    var notation: String {
        let from = BoardEncoder.squareName(fromRow * 8 + fromCol)
        let to = BoardEncoder.squareName(toRow * 8 + toCol)
        if let promo = promotion {
            let suffix: String
            switch promo {
            case .queen:  suffix = "=Q"
            case .rook:   suffix = "=R"
            case .bishop: suffix = "=B"
            case .knight: suffix = "=N"
            default:      suffix = ""
            }
            return "\(from)-\(to)\(suffix)"
        }
        return "\(from)-\(to)"
    }
}
