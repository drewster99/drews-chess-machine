import CoreGraphics

/// Pixel-space description of the chart's geometry inside a parent
/// rect. Computes once per render based on requested margins and is
/// shared by the Canvas content (knows where the plot area is) and
/// the axis-label overlays (know where to place each `Text`).
struct ChartAxisLayout {
    let totalSize: CGSize
    /// Width reserved on the left for Y axis labels.
    let yLabelWidth: CGFloat
    /// Height reserved at the bottom for X axis labels.
    let xLabelHeight: CGFloat
    let plotRect: CGRect

    init(
        totalSize: CGSize,
        yLabelWidth: CGFloat,
        xLabelHeight: CGFloat
    ) {
        self.totalSize = totalSize
        self.yLabelWidth = yLabelWidth
        self.xLabelHeight = xLabelHeight
        self.plotRect = CGRect(
            x: yLabelWidth,
            y: 0,
            width: max(0, totalSize.width - yLabelWidth),
            height: max(0, totalSize.height - xLabelHeight)
        )
    }

    /// Evenly-spaced tick values across a domain, including both
    /// endpoints. `count >= 2`.
    static func evenlySpacedTicks(
        domain: ClosedRange<Double>,
        count: Int
    ) -> [Double] {
        guard count >= 2 else { return [domain.lowerBound] }
        let span = domain.upperBound - domain.lowerBound
        guard span > 0 else { return [domain.lowerBound] }
        var ticks: [Double] = []
        ticks.reserveCapacity(count)
        let step = span / Double(count - 1)
        for i in 0..<count {
            ticks.append(domain.lowerBound + Double(i) * step)
        }
        return ticks
    }

    /// Resolved tick values for one axis. If `explicit` is non-nil,
    /// use it after clipping to `domain` (so a caller can specify
    /// `[0,1,2,3,4,5]` against a `0...5.5` axis and never see ticks
    /// outside the plot). Otherwise fall back to `fallback`.
    static func resolvedTicks(
        explicit: [Double]?,
        domain: ClosedRange<Double>,
        fallback: (ClosedRange<Double>) -> [Double]
    ) -> [Double] {
        if let explicit {
            return explicit.filter { $0 >= domain.lowerBound && $0 <= domain.upperBound }
        }
        return fallback(domain)
    }

    /// "Nice number" ticks — picks a tick step from
    /// `{1, 2, 2.5, 5} × 10ⁿ` so labels land on round values like
    /// `0, 20, 40` rather than `0, 18, 36, 54`. Approximately
    /// `approxCount` ticks; actual count may be slightly less or
    /// more depending on how the domain aligns to nice multiples.
    /// Used as the Y-axis default.
    static func niceTicks(
        domain: ClosedRange<Double>,
        approxCount: Int
    ) -> [Double] {
        let span = domain.upperBound - domain.lowerBound
        guard span > 0, approxCount >= 2 else { return [domain.lowerBound] }
        let roughStep = span / Double(approxCount - 1)
        guard roughStep > 0, roughStep.isFinite else {
            return evenlySpacedTicks(domain: domain, count: approxCount)
        }
        let magnitude = pow(10.0, floor(log10(roughStep)))
        let normalized = roughStep / magnitude
        let niceNormalized: Double
        if normalized < 1.5 { niceNormalized = 1 }
        else if normalized < 3 { niceNormalized = 2 }
        else if normalized < 4 { niceNormalized = 2.5 }
        else if normalized < 7 { niceNormalized = 5 }
        else { niceNormalized = 10 }
        let niceStep = niceNormalized * magnitude
        // First tick at or inside the domain's lower bound, snapped
        // to a multiple of `niceStep`.
        let firstTick = ceil(domain.lowerBound / niceStep) * niceStep
        var ticks: [Double] = []
        var t = firstTick
        // Epsilon margin so a tick that mathematically lands on the
        // upper bound isn't dropped by floating-point round-off.
        let epsilon = niceStep * 1e-9
        while t <= domain.upperBound + epsilon {
            ticks.append(t)
            t += niceStep
        }
        return ticks
    }
}
