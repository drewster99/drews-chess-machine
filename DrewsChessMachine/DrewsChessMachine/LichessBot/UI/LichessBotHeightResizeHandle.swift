import SwiftUI

/// A grip along a card's bottom edge that the operator drags to make the
/// card taller or shorter. The height belongs to the caller (usually
/// `@AppStorage`, so it is remembered across launches); the handle only
/// changes it, clamped to `range`.
struct LichessBotHeightResizeHandle: View {
    @Binding var height: Double
    let range: ClosedRange<Double>
    /// The height when the current drag began. Each drag update sets the
    /// height from this plus the total translation, not by accumulating
    /// deltas, so a clamped update doesn't make the grip drift from the
    /// pointer.
    @State private var heightAtDragStart: Double?

    /// How far one accessibility increment or decrement moves the edge.
    private static let accessibilityStep: Double = 40

    var body: some View {
        Capsule()
            .fill(Color.secondary.opacity(0.5))
            .frame(width: 36, height: 4)
            .frame(maxWidth: .infinity)
            .frame(height: 12)
            .contentShape(Rectangle())
            .pointerStyle(.frameResize(position: .bottom))
            .gesture(
                // Global coordinates: the handle moves as the card resizes, so a
                // translation measured in its own space would lag the pointer.
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { value in
                        let start: Double
                        if let heightAtDragStart {
                            start = heightAtDragStart
                        } else {
                            start = height
                            heightAtDragStart = height
                        }
                        height = clamped(start + value.translation.height)
                    }
                    .onEnded { _ in
                        heightAtDragStart = nil
                    }
            )
            .help("Drag to resize")
            .accessibilityElement()
            .accessibilityLabel("Resize")
            .accessibilityValue("\(Int(height)) points tall")
            .accessibilityAdjustableAction { direction in
                switch direction {
                case .increment:
                    height = clamped(height + Self.accessibilityStep)
                case .decrement:
                    height = clamped(height - Self.accessibilityStep)
                @unknown default:
                    break
                }
            }
    }

    private func clamped(_ proposed: Double) -> Double {
        min(max(proposed, range.lowerBound), range.upperBound)
    }
}
