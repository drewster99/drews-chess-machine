import Charts
import SwiftUI

/// Small progress-rate sparkline tile inside the grid (the big
/// version lives in the upper section of the app).
struct SmallProgressRateChart: View {
    let buckets: [ProgressRateBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(
                title: "Progress rate (self play + train)",
                value: headerText
            )
            ProgressRateChartBody(
                buckets: buckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            .frame(height: 60)
        }
        .frame(height: 75)
        .chartCard()
    }

    private var headerText: String {
        let tolerance = Swift.max(
            TrainingChartGridView.hoverMatchToleranceSec,
            context.bucketWidthSec * 1.5
        )
        if let t = hoveredSec {
            if let nearest = TrainingChartGridView.nearestProgressBucket(
                at: t, in: buckets, tolerance: tolerance
            ) {
                let combined = nearest.combinedMovesPerHour?.max ?? 0
                let selfPlay = nearest.selfPlayMovesPerHour?.max ?? 0
                let training = nearest.trainingMovesPerHour?.max ?? 0
                return "comb=\(TrainingChartGridView.compactLabel(combined)) "
                    + "sp=\(TrainingChartGridView.compactLabel(selfPlay)) "
                    + "tr=\(TrainingChartGridView.compactLabel(training))"
            } else {
                return "— no data"
            }
        } else if let last = buckets.last,
                  let combined = last.combinedMovesPerHour?.max {
            return "\(TrainingChartGridView.compactLabel(combined)) moves/hour"
        } else {
            return "-- moves/hour"
        }
    }
}
