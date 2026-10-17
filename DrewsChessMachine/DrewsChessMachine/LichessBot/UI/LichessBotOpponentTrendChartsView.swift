import Charts
import SwiftUI

/// The detail window's trend charts over the games against one opponent,
/// oldest first: DCM's starting rating, the opponent's starting rating, and
/// DCM's running score fraction. Every point is a value Lichess reported (or
/// a score DCM scored); nothing is interpolated. A series with fewer than two
/// points draws nothing.
struct LichessBotOpponentTrendChartsView: View {
    let stats: LichessBotOpponentProfileStats?

    var body: some View {
        GroupBox("Trends over our games vs them") {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 16) {
                    LichessBotOpponentTrendChart(title: "DCM rating", points: ourRating, style: .rating)
                    LichessBotOpponentTrendChart(title: "Their rating", points: theirRating, style: .rating)
                    LichessBotOpponentTrendChart(title: "DCM score", points: score, style: .score)
                }
                VStack(alignment: .leading, spacing: 12) {
                    LichessBotOpponentTrendChart(title: "DCM rating", points: ourRating, style: .rating)
                    LichessBotOpponentTrendChart(title: "Their rating", points: theirRating, style: .rating)
                    LichessBotOpponentTrendChart(title: "DCM score", points: score, style: .score)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
        .shown(hasAnySeries)
    }

    private var ourRating: [LichessBotOpponentTrendPoint] { stats?.ourRatingTrend ?? [] }
    private var theirRating: [LichessBotOpponentTrendPoint] { stats?.theirRatingTrend ?? [] }
    private var score: [LichessBotOpponentTrendPoint] { stats?.scoreTrend ?? [] }

    private var hasAnySeries: Bool {
        ourRating.count >= 2 || theirRating.count >= 2 || score.count >= 2
    }
}

/// One trend chart with a caption, or a placeholder when there is no line to
/// draw (fewer than two points).
struct LichessBotOpponentTrendChart: View {
    enum Style { case rating, score }

    let title: String
    let points: [LichessBotOpponentTrendPoint]
    let style: Style

    static let size = CGSize(width: 220, height: 130)

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Chart(points) { point in
                LineMark(x: .value("Game", point.index), y: .value(title, point.value))
                    .foregroundStyle(LichessBotStatsStyle.actual)
                    .interpolationMethod(.linear)
            }
            .chartYScale(domain: domain)
            .chartYAxis {
                if style == .score {
                    AxisMarks(format: FloatingPointFormatStyle<Double>.Percent().precision(.fractionLength(0)))
                } else {
                    AxisMarks()
                }
            }
            .chartXAxis(.hidden)
            .frame(width: Self.size.width, height: Self.size.height)
            .opacity(points.count >= 2 ? 1 : 0)
            .overlay {
                Text("Not enough games")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .shown(points.count < 2)
            }
        }
    }

    private var domain: ClosedRange<Double> {
        switch style {
        case .score:
            return 0...1
        case .rating:
            let values = points.map(\.value)
            guard let low = values.min(), let high = values.max() else { return 0...1 }
            return low == high ? (low - 1)...(high + 1) : low...high
        }
    }
}
