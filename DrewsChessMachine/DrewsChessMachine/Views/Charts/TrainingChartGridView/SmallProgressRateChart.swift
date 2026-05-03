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
                return "t=\(TrainingChartGridView.formatElapsedAxis(nearest.elapsedSec)) "
                    + "comb=\(TrainingChartGridView.compactLabel(combined)) "
                    + "sp=\(TrainingChartGridView.compactLabel(selfPlay)) "
                    + "tr=\(TrainingChartGridView.compactLabel(training))"
            } else {
                return "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
            }
        } else if let last = buckets.last,
                  let combined = last.combinedMovesPerHour?.max {
            return "\(TrainingChartGridView.compactLabel(combined)) moves/hour"
        } else {
            return "-- moves/hour"
        }
    }
}

/// Inner Chart for `SmallProgressRateChart` — split out so the
/// parent's body stays focused on layout while the marks + axes
/// chain lives on its own type.
struct ProgressRateChartBody: View {
    let buckets: [ProgressRateBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    var body: some View {
        Chart {
            chartMarks
        }
        .chartForegroundStyleScale([
            "Self-play": Color.blue,
            "Training": Color.orange,
            "Combined": Color.green
        ])
        .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(TrainingChartGridView.compactLabel(v))
                            .font(.system(size: 7))
                            .monospacedDigit()
                    }
                }
            }
        }
        .chartLegend(.hidden)
        .chartXScale(domain: context.timeSeriesXDomain)
        .chartScrollableAxes(.horizontal)
        .chartXVisibleDomain(length: context.visibleDomainSec)
        .chartScrollPosition(x: $scrollX)
        .chartOverlay { proxy in
            ChartHoverOverlay(proxy: proxy, hoveredSec: $hoveredSec)
        }
    }

    @ChartContentBuilder
    private var chartMarks: some ChartContent {
        ForEach(buckets) { b in
            LineMark(
                x: .value("Time", b.elapsedSec),
                y: .value("Moves/hr", b.combinedMovesPerHour?.max ?? .nan)
            )
            .foregroundStyle(by: .value("Series", "Combined"))
        }
        ForEach(buckets) { b in
            LineMark(
                x: .value("Time", b.elapsedSec),
                y: .value("Moves/hr", b.selfPlayMovesPerHour?.max ?? .nan)
            )
            .foregroundStyle(by: .value("Series", "Self-play"))
        }
        ForEach(buckets) { b in
            LineMark(
                x: .value("Time", b.elapsedSec),
                y: .value("Moves/hr", b.trainingMovesPerHour?.max ?? .nan)
            )
            .foregroundStyle(by: .value("Series", "Training"))
        }
        if let t = hoveredSec {
            RuleMark(x: .value("Time", t))
                .foregroundStyle(Color.gray.opacity(0.5))
                .lineStyle(StrokeStyle(lineWidth: 1))
        }
    }
}
