import Charts
import SwiftUI

/// How DCM did as its model trained (§3.8, OD-12): one point per group of
/// consecutive checkpoints of a run holding at least 30 scored games,
/// placed at the run's cumulative trainer step when recorded, else its
/// training step; the score with its 95% interval, or the performance
/// rating. One color per run.
struct LichessBotModelProgressChart: View {
    let points: [LichessBotProgressionPoint]
    @State private var metric: LichessBotProgressMetric = .score

    var body: some View {
        let marks = LichessBotProgressChartMark.marks(points, metric: metric)
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Progression")
                    .font(LichessBotStatsStyle.sectionFont)
                Picker("Plot", selection: $metric) {
                    ForEach(LichessBotProgressMetric.allCases) { metric in
                        Text(metric.label).tag(metric)
                    }
                }
                .pickerStyle(.menu)
                .fixedSize()
            }
            Chart(marks) { mark in
                RuleMark(x: .value("Step", mark.step), yStart: .value("Low", mark.lower ?? mark.value), yEnd: .value("High", mark.upper ?? mark.value))
                    .foregroundStyle(by: .value("Run", mark.series))
                    .opacity(0.5)
                PointMark(x: .value("Step", mark.step), y: .value(metric.label, mark.value))
                    .foregroundStyle(by: .value("Run", mark.series))
                    .accessibilityLabel("\(mark.series) at step \(mark.step)")
                    .accessibilityValue(metric == .score ? String(format: "%.1f%%", mark.value) : "\(Int(mark.value.rounded()))")
            }
            .chartXAxisLabel(points.contains { $0.stepIsCumulative } ? "Cumulative trainer step" : "Training step")
            .chartYAxisLabel(metric == .score ? "Score %" : "Performance rating")
            .frame(height: 180)
        }
    }
}
