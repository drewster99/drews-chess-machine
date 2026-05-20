import SwiftUI

/// How successive series points are connected by the path builder.
public enum FastChartInterpolation: Sendable, Equatable {
    /// Straight line between (x_i, y_i) and (x_{i+1}, y_{i+1}).
    case linear
    /// `(x_i, y_i)` → `(x_{i+1}, y_i)` → `(x_{i+1}, y_{i+1})` — the
    /// horizontal hold lands AFTER the sample. Matches Swift Charts'
    /// `.interpolationMethod(.stepEnd)` semantics, used by the
    /// PowerThermal step trace.
    case stepEnd
    /// `(x_i, y_i)` → `(x_i, y_{i+1})` → `(x_{i+1}, y_{i+1})` — the
    /// horizontal hold lands BEFORE the next sample.
    case stepStart
}

/// Data payload for a single series.
public enum FastChartSeriesData: Sendable, Equatable {
    /// Raw points. Must be non-decreasing in `x`. The chart draws a
    /// stroke through every point in the visible X range.
    case points([CGPoint])
    /// Pre-decimated buckets. The chart strokes through `(x, yMax)`
    /// for each bucket; when `showMinMaxBand` is set on the series,
    /// the chart additionally fills a faint vertical band from
    /// `yMin` to `yMax`.
    case buckets([FastChartBucket])
}

/// One series on a `FastLineChart`.
public struct FastChartSeries: Sendable, Equatable, Identifiable {
    public let id: String
    public let color: Color
    public let lineWidth: CGFloat
    public let interpolation: FastChartInterpolation
    public let data: FastChartSeriesData
    /// Honored only when `data` is `.buckets`. Renders a faint
    /// vertical band between each bucket's `yMin` and `yMax` to make
    /// per-bucket dispersion visible — useful at zoomed-out levels
    /// where one bucket aggregates many raw samples.
    public let showMinMaxBand: Bool

    public init(
        id: String,
        color: Color,
        lineWidth: CGFloat = 1.5,
        interpolation: FastChartInterpolation = .linear,
        data: FastChartSeriesData,
        showMinMaxBand: Bool = false
    ) {
        self.id = id
        self.color = color
        self.lineWidth = lineWidth
        self.interpolation = interpolation
        self.data = data
        self.showMinMaxBand = showMinMaxBand
    }
}
