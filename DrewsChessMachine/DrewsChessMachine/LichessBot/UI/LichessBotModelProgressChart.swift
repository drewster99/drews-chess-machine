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
        VStack(alignment: .leading, spacing: 6) {
            LichessBotProgressMetricHeader(metric: $metric)
            LichessBotModelProgressPlot(
                marks: LichessBotProgressChartMark.marks(points, metric: metric),
                metric: metric,
                stepIsCumulative: points.contains { $0.stepIsCumulative }
            )
        }
    }
}
