import SwiftUI

/// Input to `FastChartDecimator`. Carries everything the decimator
/// needs to turn an arbitrarily long raw-points stream into a
/// chart-ready bucket series — the styling fields are passed through
/// so the caller doesn't have to re-stitch them onto the decimator
/// output before handing it to the chart.
public struct FastChartRawSeries: Sendable, Equatable, Identifiable {
    public let id: String
    public let color: Color
    public let lineWidth: CGFloat
    public let interpolation: FastChartInterpolation
    /// Raw points. Must be non-decreasing in `x`. NaN-y values are
    /// preserved and become NaN buckets (line breaks across the gap).
    public let points: [CGPoint]
    /// Forwarded to the decimator's output series. Honored by the
    /// chart only when `data` is `.buckets` — which it always is on
    /// decimator output.
    public let showMinMaxBand: Bool

    public init(
        id: String,
        color: Color,
        lineWidth: CGFloat = 1.5,
        interpolation: FastChartInterpolation = .linear,
        points: [CGPoint],
        showMinMaxBand: Bool = false
    ) {
        self.id = id
        self.color = color
        self.lineWidth = lineWidth
        self.interpolation = interpolation
        self.points = points
        self.showMinMaxBand = showMinMaxBand
    }
}
