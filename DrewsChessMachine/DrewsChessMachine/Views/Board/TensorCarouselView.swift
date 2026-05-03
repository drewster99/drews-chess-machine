import SwiftUI

/// Renders one 8x8 tensor channel as a mini chessboard with active squares highlighted.
struct ChannelBoardView: View {
    let values: [Float]

    private static let lightSquare = Color(red: 0.94, green: 0.85, blue: 0.71)
    private static let darkSquare = Color(red: 0.71, green: 0.53, blue: 0.39)
    private static let activeColor = Color.blue

    var body: some View {
        Canvas { context, size in
            let sq = min(size.width, size.height) / 8

            for row in 0..<8 {
                for col in 0..<8 {
                    let rect = CGRect(
                        x: CGFloat(col) * sq,
                        y: CGFloat(row) * sq,
                        width: sq, height: sq
                    )
                    let isLight = (row + col) % 2 == 0
                    context.fill(
                        Path(rect),
                        with: .color(isLight ? Self.lightSquare : Self.darkSquare)
                    )

                    let value = values[row * 8 + col]
                    if value > 0.001 {
                        context.fill(
                            Path(rect),
                            with: .color(Self.activeColor.opacity(Double(value) * 0.65))
                        )
                    }
                }
            }
        }
        .aspectRatio(1, contentMode: .fit)
    }
}

/// Channel name lookup for the 20 input planes (post-arch-refresh —
/// added planes 18 and 19 for threefold-repetition signals).
enum TensorChannelNames {
    static let names = [
        "My Pawns", "My Knights", "My Bishops",
        "My Rooks", "My Queens", "My King",
        "Opp Pawns", "Opp Knights", "Opp Bishops",
        "Opp Rooks", "Opp Queens", "Opp King",
        "My Kingside Castle", "My Queenside Castle",
        "Opp Kingside Castle", "Opp Queenside Castle",
        "En Passant", "Halfmove Clock",
        "Repetition ≥1×", "Repetition ≥2×"
    ]

    /// Short labels for the strip thumbnails.
    static let shortNames = [
        "♙", "♘", "♗", "♖", "♕", "♔",
        "♟", "♞", "♝", "♜", "♛", "♚",
        "K-side", "Q-side", "K-side", "Q-side",
        "e.p.", "50-mv",
        "rep≥1", "rep≥2"
    ]
}
