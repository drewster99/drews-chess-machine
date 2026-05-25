import SwiftUI

/// Tiny line chart drawn straight to a `Path` — no axes, no labels,
/// no SwiftUI Charts dependency. Sized by the caller via `.frame`.
///
/// Empty / single-value series renders as a centered horizontal stub
/// (so the row's spark column doesn't collapse to zero width before
/// the first two ticks land). Otherwise: min/max-normalized to the
/// view's height, swept across the width at uniform x spacing.
///
/// The line color is wired to the row's value-delta color (green for
/// `current > previous`, red for `current < previous`, otherwise the
/// neutral fill) so the spark visually agrees with the value cell
/// next to it.
struct TacticalProbeSparkView: View {
    let values: [Float]
    let stroke: Color

    var body: some View {
        Canvas { context, size in
            guard values.count >= 2 else {
                drawStub(context: context, size: size)
                return
            }

            let lo = values.min() ?? 0
            let hi = values.max() ?? 1
            let range = max(hi - lo, 1e-6)
            let n = values.count
            let stepX = n > 1 ? size.width / CGFloat(n - 1) : 0

            var path = Path()
            for (i, v) in values.enumerated() {
                let normalized = (v - lo) / range
                let x = CGFloat(i) * stepX
                let y = size.height * (1.0 - CGFloat(normalized))
                if i == 0 {
                    path.move(to: CGPoint(x: x, y: y))
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
            context.stroke(path, with: .color(stroke), lineWidth: 1.5)
        }
        .accessibilityHidden(true)
    }

    private func drawStub(context: GraphicsContext, size: CGSize) {
        var path = Path()
        let y = size.height / 2
        path.move(to: CGPoint(x: 0, y: y))
        path.addLine(to: CGPoint(x: size.width, y: y))
        context.stroke(path, with: .color(stroke.opacity(0.35)), lineWidth: 1)
    }
}
