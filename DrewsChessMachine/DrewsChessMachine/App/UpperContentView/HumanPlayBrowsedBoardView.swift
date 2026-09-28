import SwiftUI

/// The human game's board at an earlier position the user is browsing
/// (plan §14.3a). Read-only: it replays the game's moves from the start to
/// the chosen ply and never touches the game itself.
struct HumanPlayBrowsedBoardView: View {
    let history: [HumanPlayPacer.HistoryEntry]
    /// How many plies to apply (0 = the start).
    let plyCount: Int
    let flipped: Bool

    var body: some View {
        let count = min(plyCount, history.count)
        let state = history.prefix(count).reduce(GameState.starting) { position, entry in
            MoveGenerator.applyMove(entry.move, to: position)
        }
        let lastMove = count > 0 ? history[count - 1].move : nil
        HumanPlayBoardView(
            pieces: flipped ? Array(state.board.reversed()) : state.board,
            selectedFromSquare: nil,
            legalMoveTargets: [],
            lastMoveDestinationSquare: lastMove.map { visual($0.toRow * 8 + $0.toCol) },
            lastMoveSourceSquare: lastMove.map { visual($0.fromRow * 8 + $0.fromCol) },
            checkSquare: checkSquare(in: state),
            humanMoveActive: false,
            humanColor: nil,
            pendingPromotion: nil,
            promotionVisualSquare: nil,
            onTapSquare: { _ in },
            onSelectPromotion: { _ in },
            onCancelPromotion: {},
            onAnimationCompleted: {}
        )
    }

    private func visual(_ square: Int) -> Int {
        flipped ? 63 - square : square
    }

    private func checkSquare(in state: GameState) -> Int? {
        let color = state.currentPlayer
        guard MoveGenerator.isInCheck(state, color: color) else { return nil }
        for square in 0..<64 {
            if let piece = state.board[square], piece.type == .king, piece.color == color {
                return visual(square)
            }
        }
        return nil
    }
}
