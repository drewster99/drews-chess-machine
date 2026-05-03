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

    init(
        fromRow: Int,
        fromCol: Int,
        toRow: Int,
        toCol: Int,
        probability: Float,
        piece: String?,
        isLegal: Bool = true
    ) {
        self.fromRow = fromRow
        self.fromCol = fromCol
        self.toRow = toRow
        self.toCol = toCol
        self.probability = probability
        self.piece = piece
        self.isLegal = isLegal
    }
}
