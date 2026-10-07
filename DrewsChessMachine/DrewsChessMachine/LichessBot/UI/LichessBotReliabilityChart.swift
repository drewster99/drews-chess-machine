import Charts
import SwiftUI

/// Reliability over every DCM decision (§3.6): for each tenth of the value
/// head's expected score, the mean prediction against the mean actual score
/// of those positions' games, point size by the number of positions. A
/// calibrated head sits on the diagonal. Positions of one game share its
/// result, so long games weigh more.
struct LichessBotReliabilityChart: View {
    let buckets: [LichessBotReliabilityBucket]

    var body: some View {
        Chart {
            LineMark(x: .value("Predicted", 0.0), y: .value("Actual", 0.0), series: .value("Line", "Calibrated"))
                .foregroundStyle(LichessBotStatsStyle.neutral.opacity(0.5))
            LineMark(x: .value("Predicted", 1.0), y: .value("Actual", 1.0), series: .value("Line", "Calibrated"))
                .foregroundStyle(LichessBotStatsStyle.neutral.opacity(0.5))
            ForEach(buckets) { bucket in
                PointMark(x: .value("Predicted", bucket.meanPredicted), y: .value("Actual", bucket.meanActual))
                    .symbolSize(by: .value("Positions", bucket.positions))
                    .foregroundStyle(LichessBotStatsStyle.actual)
                    .accessibilityLabel("Predicted \(LichessBotStatsFormat.percent(bucket.meanPredicted))")
                    .accessibilityValue("Actual \(LichessBotStatsFormat.percent(bucket.meanActual)), \(bucket.positions) positions")
            }
        }
        .chartXScale(domain: 0...1)
        .chartYScale(domain: 0...1)
        .chartXAxisLabel("Predicted expected score")
        .chartYAxisLabel("Actual score")
        .chartLegend(.hidden)
        .chartSymbolSizeScale(range: 16...160)
        .frame(width: 240, height: 200)
        .help("Each point is a tenth of the value head's expected score over every DCM decision; points on the diagonal are calibrated. Positions of one game share its result, so long games weigh more.")
    }
}
