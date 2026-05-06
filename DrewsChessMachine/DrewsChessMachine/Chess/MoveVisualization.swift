import Foundation

/// A move to visualize on the board with an arrow and ghost piece.
struct MoveVisualization: Sendable {
    let fromRow: Int
    let fromCol: Int
    let toRow: Int
    let toCol: Int
    let probability: Float
    let piece: String?
    /// True if this move is legal in the source state. The Forward
    /// Pass demo (and Candidate Test) shows the top-K policy cells
    /// regardless of legality so the user can see whether the network
    /// has learned move-validity — illegal candidates surfacing in the
    /// top-K is a diagnostic signal that the policy hasn't yet learned
    /// to suppress them. Defaults to `true` for callers that don't
    /// know or care about legality (legacy code paths).
    let isLegal: Bool
    /// Promotion piece, when this move came from one of the 12
    /// promotion channels (queen-promo or underpromo). nil for
    /// non-promotion channels. Carried through so the displayed
    /// top-K text can render the promotion suffix (`=Q`, `=R`,
    /// `=B`, `=N`) — without it, two distinct policy cells like
    /// "NE1 from g6" (chan 7, no promotion) and "queen-promo
    /// cap-right from g6" (chan 75, =Q) both render as "g6-h7"
    /// in the move text, which makes it impossible to tell which
    /// channel a top-K entry actually came from.
    let promotion: PieceType?

    init(
        fromRow: Int,
        fromCol: Int,
        toRow: Int,
        toCol: Int,
        probability: Float,
        piece: String?,
        isLegal: Bool = true,
        promotion: PieceType? = nil
    ) {
        self.fromRow = fromRow
        self.fromCol = fromCol
        self.toRow = toRow
        self.toCol = toCol
        self.probability = probability
        self.piece = piece
        self.isLegal = isLegal
        self.promotion = promotion
    }
}
