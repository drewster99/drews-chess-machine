import Charts
import SwiftUI

/// A speed's rating over its recent rated games (§3.5, OD-9): a line with
/// no axes, 120 × 22 points, the y range fitted to the data. Its tooltip
/// gives the range and dates; the accessibility label says the same. Below
/// two points there is no line, and the frame stays empty.
struct LichessBotRatingSparkline: View {
    let series: LichessBotRatingSparklineSeries

    static let size = CGSize(width: 120, height: 22)

    var body: some View {
        ZStack {
            ForEach(series.drawableRanges) { drawable in
                LichessBotRatingSparklineChart(ratings: series.ratings, range: drawable.range)
            }
        }
        .frame(width: Self.size.width, height: Self.size.height)
        .help(series.help)
        .accessibilityElement()
        .accessibilityLabel("Rating trend: \(series.help)")
    }
}

/// The sparkline's line over `range`. A flat series still gets a range one
/// point tall, so its line is drawn mid-frame rather than not at all.
struct LichessBotRatingSparklineChart: View {
    let ratings: [Int]
    let range: ClosedRange<Int>

    var body: some View {
        Chart(Array(ratings.enumerated()), id: \.offset) { point in
            LineMark(x: .value("Game", point.offset), y: .value("Rating", point.element))
                .foregroundStyle(LichessBotStatsStyle.actual)
                .interpolationMethod(.linear)
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(domain: range.lowerBound...max(range.upperBound, range.lowerBound + 1))
        .chartLegend(.hidden)
    }
}
