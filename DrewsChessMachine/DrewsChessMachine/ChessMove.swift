import Foundation

/// A chess move defined by source and destination squares, with optional promotion.
struct ChessMove: Sendable, Equatable, Hashable {
    let fromRow: Int
    let fromCol: Int
    let toRow: Int
    let toCol: Int
    /// Promotion piece type when a pawn reaches the last rank. Nil for non-promotion moves.
    let promotion: PieceType?

    // Note: there is no `policyIndex` property on `ChessMove`. The policy
    // index depends on the current player (encoder-frame perspective flip
    // for black) and on the AlphaZero-style 76-channel encoding, neither
    // of which a `ChessMove` knows about on its own. Use
    // `PolicyEncoding.policyIndex(_:currentPlayer:)` instead — the caller
    // must supply the side to move so the encoding picks the right
    // perspective. The compile-time absence of this convenience property
    // is intentional: it forces every callsite to think about which
    // encoding (and which side) it's working with.

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
