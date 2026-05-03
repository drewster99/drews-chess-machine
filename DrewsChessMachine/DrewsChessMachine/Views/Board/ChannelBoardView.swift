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
