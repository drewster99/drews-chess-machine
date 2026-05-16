import CoreGraphics

/// Affine mapping from data-space `(x, y)` to view-space points
/// inside a plot rect. Stored as concrete scale + origin pairs so
/// the per-point transform is two FMA operations — the path
/// builder calls this for every visible sample on every render.
///
/// Y axis is flipped: data Y increases upward, view Y increases
/// downward. The conversion `view.y = rect.maxY - yNorm * height`
/// is folded into `yScale` (negative) + `yOffset` so the per-point
/// math stays branchless.
struct ChartCoordTransform {
    let xScale: Double
    let xOffset: Double
    let yScale: Double
    let yOffset: Double
    let rect: CGRect

    init(
        xDomain: ClosedRange<Double>,
        yDomain: ClosedRange<Double>,
        rect: CGRect
    ) {
        self.rect = rect
        let xSpan = xDomain.upperBound - xDomain.lowerBound
        let ySpan = yDomain.upperBound - yDomain.lowerBound
        // Degenerate domain → collapse to the rect's center to avoid
        // dividing by zero. The caller is responsible for not asking
        // for a degenerate domain in normal operation; this is the
        // safety floor.
        let xSafe = xSpan == 0 ? 1 : xSpan
        let ySafe = ySpan == 0 ? 1 : ySpan
        self.xScale = Double(rect.width) / xSafe
        self.xOffset = Double(rect.origin.x) - xDomain.lowerBound * xScale
        self.yScale = -Double(rect.height) / ySafe
        self.yOffset = Double(rect.origin.y) + Double(rect.height) - yDomain.lowerBound * yScale
    }

    @inline(__always)
    func point(_ x: Double, _ y: Double) -> CGPoint {
        CGPoint(x: x * xScale + xOffset, y: y * yScale + yOffset)
    }

    @inline(__always)
    func viewX(_ x: Double) -> CGFloat {
        CGFloat(x * xScale + xOffset)
    }

    @inline(__always)
    func viewY(_ y: Double) -> CGFloat {
        CGFloat(y * yScale + yOffset)
    }
}
