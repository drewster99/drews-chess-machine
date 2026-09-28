import SwiftUI

/// A read-only board for a Lichess game at a chosen position (plan §14.3a),
/// oriented to DCM's color, with last-move and check highlights. It takes no
/// input: browsing is done by the surrounding view's controls.
struct LichessBotBoardView: View {
    let game: LichessBotLiveGame
    /// The ply count of the position to show.
    let plyCount: Int

    var body: some View {
        let flipped = game.ourColor == .black
        let state = game.state(afterPlies: plyCount)
        let lastMove = plyCount > 0 ? game.plies[plyCount - 1].move : nil
        HumanPlayBoardView(
            pieces: flipped ? Array(state.board.reversed()) : state.board,
            selectedFromSquare: nil,
            legalMoveTargets: [],
            lastMoveDestinationSquare: lastMove.map { Self.visual($0.toRow * 8 + $0.toCol, flipped: flipped) },
            lastMoveSourceSquare: lastMove.map { Self.visual($0.fromRow * 8 + $0.fromCol, flipped: flipped) },
            checkSquare: Self.checkSquare(in: state, flipped: flipped),
            humanMoveActive: false,
            humanColor: nil,
            pendingPromotion: nil,
            promotionVisualSquare: nil,
            onTapSquare: { _ in },
            onSelectPromotion: { _ in },
            onCancelPromotion: {},
            onAnimationCompleted: {}
        )
        .accessibilityLabel("Board after ply \(plyCount)")
    }

    private static func visual(_ square: Int, flipped: Bool) -> Int {
        flipped ? 63 - square : square
    }

    /// The side to move's king square if it is in check.
    private static func checkSquare(in state: GameState, flipped: Bool) -> Int? {
        let color = state.currentPlayer
        guard MoveGenerator.isInCheck(state, color: color) else { return nil }
        for square in 0..<64 {
            if let piece = state.board[square], piece.type == .king, piece.color == color {
                return visual(square, flipped: flipped)
            }
        }
        return nil
    }
}
