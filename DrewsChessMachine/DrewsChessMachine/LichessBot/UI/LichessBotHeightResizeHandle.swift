import SwiftUI

/// A grip along a card's bottom edge that the operator drags to make the
/// card taller or shorter. The height belongs to the caller (usually
/// `@AppStorage`, so it is remembered across launches); the handle only
/// changes it, clamped to `range`.
///
/// The content can be laid out taller than the stored height (the stored
/// height is its minimum, not a fixed height). A drag therefore starts from
/// the displayed height, the edge the operator sees: starting from the
/// stored height would make the first part of a downward drag move nothing.
struct LichessBotHeightResizeHandle: View {
    @Binding var height: Double
    /// The content's laid-out height: `height` or more. Nil until the
    /// content's first layout, which comes before any pointer reaches the
    /// handle; until then the content is shown at `height`.
    let displayedHeight: Double?
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
                            start = Self.shownHeight(stored: height, displayed: displayedHeight, range: range)
                            heightAtDragStart = start
                        }
                        height = Self.clamped(start + value.translation.height, to: range)
                    }
                    .onEnded { _ in
                        heightAtDragStart = nil
                        height = Self.heightAfterDrag(stored: height, displayed: displayedHeight, range: range)
                    }
            )
            .help("Drag to resize")
            .accessibilityElement()
            .accessibilityLabel("Resize")
            .accessibilityValue("\(Int(Self.shownHeight(stored: height, displayed: displayedHeight, range: range))) points tall")
            .accessibilityAdjustableAction { direction in
                let shown = Self.shownHeight(stored: height, displayed: displayedHeight, range: range)
                switch direction {
                case .increment:
                    height = Self.clamped(shown + Self.accessibilityStep, to: range)
                case .decrement:
                    height = Self.clamped(shown - Self.accessibilityStep, to: range)
                @unknown default:
                    break
                }
            }
    }

    /// The height the edge is at, where a drag or an accessibility step
    /// starts: the displayed height once laid out, the stored height before
    /// (which is then what the content is shown at), clamped to `range`.
    nonisolated static func shownHeight(stored: Double, displayed: Double?, range: ClosedRange<Double>) -> Double {
        clamped(displayed ?? stored, to: range)
    }

    /// The stored height once a drag ends. A drag that went above the
    /// content's own minimum leaves the content at that minimum, so the
    /// stored height is raised to what is shown and never sits where moving
    /// it moves nothing.
    nonisolated static func heightAfterDrag(stored: Double, displayed: Double?, range: ClosedRange<Double>) -> Double {
        guard let displayed else { return stored }
        return clamped(max(stored, displayed), to: range)
    }

    nonisolated static func clamped(_ proposed: Double, to range: ClosedRange<Double>) -> Double {
        min(max(proposed, range.lowerBound), range.upperBound)
    }
}
