import Foundation
import SwiftUI

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
    /// Optional explicit arrow color. When nil, `ChessBoardView`
    /// falls back to its rank-based hue gradient (the default for
    /// the forward-pass demo and candidate-test panel). The Tactical
    /// Probe popover passes per-arrow colors here to encode
    /// expected-vs-actual outcome at a glance (green = expected at
    /// rank 1, yellow-green = expected in top-5, red = expected
    /// missed top-5, blue = actual top-1 that wasn't expected,
    /// tan = other top-5 moves).
    let color: Color?

    init(
        fromRow: Int,
        fromCol: Int,
        toRow: Int,
        toCol: Int,
        probability: Float,
        piece: String?,
        isLegal: Bool = true,
        promotion: PieceType? = nil,
        color: Color? = nil
    ) {
        self.fromRow = fromRow
        self.fromCol = fromCol
        self.toRow = toRow
        self.toCol = toCol
        self.probability = probability
        self.piece = piece
        self.isLegal = isLegal
        self.promotion = promotion
        self.color = color
    }
}
