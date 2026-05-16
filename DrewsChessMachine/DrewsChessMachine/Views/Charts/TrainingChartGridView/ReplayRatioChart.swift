import SwiftUI
import SwiftUIFastCharts

/// Replay ratio tile — line + dashed reference at the user target.
struct ReplayRatioChart: View {
    let buckets: [TrainingBucket]
    let target: Double
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    var body: some View {
        let yMax = yMaxObserved()
        return FastLineChart(
            title: "Replay ratio",
            titleHelp: AttributedString("""
                Ratio of training positions consumed to self-play positions produced, computed \
                continuously by the ReplayRatioController. Dashed red line is the configured target; \
                when auto-adjust is on, the controller nudges the training step delay to keep the \
                trace near that target.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...yMax,
            series: [
                FastChartSeries(
                    id: "Replay ratio",
                    color: .green,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.replayRatio?.min ?? .nan,
                            yMax: b.replayRatio?.max ?? .nan
                        )
                    })
                )
            ],
            referenceLines: [
                FastChartReferenceLine(
                    id: "target",
                    y: target,
                    label: String(format: "target %.2f", target),
                    color: Color.red.opacity(0.6),
                    lineWidth: 1,
                    dashed: true
                )
            ],
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func yMaxObserved() -> Double {
        let observed = buckets.compactMap { $0.replayRatio?.max }.max() ?? 0
        return Swift.max(observed * 1.1, target * 1.1, 0.1)
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        if let t = hoveredX {
            if let v = nearest(at: t)?.replayRatio?.max {
                return AttributedString(String(format: "%.2f (target %.2f)", v, target))
            }
            return AttributedString("— no data")
        }
        if let v = buckets.last?.replayRatio?.max {
            return AttributedString(String(format: "%.2f (target %.2f)", v, target))
        }
        return AttributedString(String(format: "-- (target %.2f)", target))
    }

    private func nearest(at t: Double) -> TrainingBucket? {
        TrainingChartGridView.nearestTrainingBucket(
            at: t,
            in: buckets,
            tolerance: Swift.max(
                TrainingChartGridView.hoverMatchToleranceSec,
                bucketWidthSec * 1.5
            )
        )
    }
}

extension ReplayRatioChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.target == rhs.target
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
