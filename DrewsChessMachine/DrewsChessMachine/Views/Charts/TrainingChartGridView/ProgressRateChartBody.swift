import Charts
import SwiftUI

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
