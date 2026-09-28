import SwiftUI

/// A small disc in a side's piece color — white with a dark rim, or black
/// with a light rim — so it reads in both light and dark mode.
struct PieceColorDisc: View {
    let color: PieceColor
    var diameter: CGFloat = 12

    var body: some View {
        Circle()
            .fill(color == .white ? Color.white : Color.black)
            .overlay(Circle().strokeBorder(Color.gray.opacity(0.7), lineWidth: 1))
            .frame(width: diameter, height: diameter)
            .accessibilityLabel(color == .white ? "White" : "Black")
    }
}
