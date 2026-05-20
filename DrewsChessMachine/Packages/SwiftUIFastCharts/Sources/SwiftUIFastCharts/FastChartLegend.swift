import SwiftUI

/// One legend entry — a colored swatch followed by a label.
public struct FastChartLegendItem: Sendable, Identifiable, Equatable {
    public var id: String { label }
    public let label: String
    public let color: Color

    public init(label: String, color: Color) {
        self.label = label
        self.color = color
    }
}

/// Legend rendering policy.
public enum FastChartLegend: Sendable, Equatable {
    /// Show automatically iff `series.count > 1`, deriving items
    /// from the series' `id` and `color`. Rendered lower-left in
    /// a single horizontal row beneath the plot.
    case auto
    /// Never render a legend, even with multiple series.
    case off
    /// Render exactly these items. Caller controls labels and order.
    case custom([FastChartLegendItem])
}
