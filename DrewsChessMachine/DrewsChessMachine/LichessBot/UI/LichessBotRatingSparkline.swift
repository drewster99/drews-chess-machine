import Charts
import SwiftUI

/// A speed's rating over its recent rated games (§3.5, OD-9): a line with
/// no axes, 120 × 22 points, the y range fitted to the data. Its tooltip
/// gives the range and dates; the accessibility label says the same.
struct LichessBotRatingSparkline: View {
    let series: LichessBotRatingSparklineSeries

    static let size = CGSize(width: 120, height: 22)

    var body: some View {
        let low = series.ratings.min() ?? 0
        let high = series.ratings.max() ?? 0
        Chart(Array(series.ratings.enumerated()), id: \.offset) { point in
            LineMark(x: .value("Game", point.offset), y: .value("Rating", point.element))
                .foregroundStyle(LichessBotStatsStyle.actual)
                .interpolationMethod(.linear)
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(domain: low...max(high, low + 1))
        .chartLegend(.hidden)
        .frame(width: Self.size.width, height: Self.size.height)
        .help(series.help)
        .accessibilityElement()
        .accessibilityLabel("Rating trend: \(series.help)")
        .opacity(series.ratings.count >= 2 ? 1 : 0)
    }
}
