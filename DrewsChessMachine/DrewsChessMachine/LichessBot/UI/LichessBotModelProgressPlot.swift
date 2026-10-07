import Charts
import SwiftUI

/// The progression chart's plot: a point per bin, with a vertical rule for
/// its score interval, colored by run.
struct LichessBotModelProgressPlot: View {
    let marks: [LichessBotProgressChartMark]
    let metric: LichessBotProgressMetric
    let stepAxisLabel: String

    var body: some View {
        Chart(marks) { mark in
            RuleMark(x: .value("Step", mark.step), yStart: .value("Low", mark.lower ?? mark.value), yEnd: .value("High", mark.upper ?? mark.value))
                .foregroundStyle(by: .value("Run", mark.series))
                .opacity(0.5)
            PointMark(x: .value("Step", mark.step), y: .value(metric.label, mark.value))
                .foregroundStyle(by: .value("Run", mark.series))
                .accessibilityLabel("\(mark.series) at step \(mark.step)")
                .accessibilityValue(metric == .score ? String(format: "%.1f%%", mark.value) : "\(Int(mark.value.rounded()))")
        }
        .chartXAxisLabel(stepAxisLabel)
        .chartYAxisLabel(metric == .score ? "Score %" : "Performance rating")
        .frame(height: 180)
    }
}
