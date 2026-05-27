import SwiftUI
import SwiftUIFastCharts

/// Small progress-rate sparkline tile inside the grid. One series —
/// training-rate (moves consumed by the trainer per hour) — on its
/// own Y axis, with the moves/hour readout in the chart's own header.
/// The self-play rate series was previously charted alongside but
/// removed: it tracks training rate × the configured replay ratio,
/// which is already visualized in the Replay-ratio tile, so showing
/// it here was redundant.
struct SmallProgressRateChart: View {
    let buckets: [ProgressRateBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    private static let trainingColor: Color = .orange

    var body: some View {
        FastLineChart(
            title: "Training rate",
            titleHelp: AttributedString("""
                Training throughput in moves per hour — positions consumed by the trainer. \
                Self-play (producer) rate has its own implicit visualization via the \
                Replay-ratio tile: producer ≈ trainer × replay-ratio-target.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...yMaxObserved(),
            series: [
                FastChartSeries(
                    id: "Training",
                    color: Self.trainingColor,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.trainingMovesPerHour?.min ?? .nan,
                            yMax: b.trainingMovesPerHour?.max ?? .nan
                        )
                    })
                )
            ],
            legend: .off,
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func yMaxObserved() -> Double {
        let c = buckets.compactMap { $0.trainingMovesPerHour?.max }.max() ?? 0
        return Swift.max(c * 1.1, 10)
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let tolerance = Swift.max(
            TrainingChartGridView.hoverMatchToleranceSec,
            bucketWidthSec * 1.5
        )
        if let t = hoveredX {
            if let nearest = TrainingChartGridView.nearestProgressBucket(
                at: t, in: buckets, tolerance: tolerance
            ) {
                let training = nearest.trainingMovesPerHour?.max ?? 0
                var trPart = AttributedString(
                    "tr=\(FastChartFormatters.compact(training))"
                )
                trPart.foregroundColor = Self.trainingColor
                return trPart
            }
            return AttributedString("— no data")
        }
        if let last = buckets.last,
           let training = last.trainingMovesPerHour?.max {
            return AttributedString("\(FastChartFormatters.compact(training)) moves/hour (tr)")
        }
        return AttributedString("-- moves/hour")
    }
}

extension SmallProgressRateChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
