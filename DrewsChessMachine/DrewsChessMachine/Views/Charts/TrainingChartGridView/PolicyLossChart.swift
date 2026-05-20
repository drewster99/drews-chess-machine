import SwiftUI
import SwiftUIFastCharts

/// Single-series tile for the outcome-weighted policy cross-entropy
/// (`pLoss`). `pLoss` is unbounded on both sides and routinely
/// negative — chart includes a faint dashed 0-line so the sign is
/// readable.
struct PolicyLossChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    var body: some View {
        let (yMin, yMax) = observedYRange()
        return FastLineChart(
            title: "pLoss (outcome-weighted CE)",
            titleHelp: AttributedString("""
                Outcome-weighted policy cross-entropy. Each batch position's CE is signed by the \
                game's outcome from the current player's perspective (+1 win, -1 loss, 0 draw), so \
                the loss is unbounded on both sides and routinely negative when winning predictions \
                dominate the batch. Always read alongside policy entropy — pLoss alone doesn't say \
                whether the head is concentrating or diffusing.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: yMin...yMax,
            series: [
                FastChartSeries(
                    id: "pLoss",
                    color: .orange,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.policyLoss?.min ?? .nan,
                            yMax: b.policyLoss?.max ?? .nan
                        )
                    })
                )
            ],
            referenceLines: [
                FastChartReferenceLine(
                    id: "zero",
                    y: 0,
                    label: nil,
                    color: Color.gray.opacity(0.4),
                    lineWidth: 0.5,
                    dashed: true
                )
            ],
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func observedYRange() -> (Double, Double) {
        let values = buckets.compactMap { $0.policyLoss?.max }
        let mins = buckets.compactMap { $0.policyLoss?.min }
        let lo = (mins.min() ?? -0.5)
        let hi = (values.max() ?? 0.5)
        // Always include 0 in the visible range so the dashed
        // reference line stays on-chart.
        let yMin = Swift.min(lo, 0) - 0.05
        let yMax = Swift.max(hi, 0) + 0.05
        if yMin == yMax { return (yMin - 0.5, yMax + 0.5) }
        return (yMin, yMax)
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        if let t = hoveredX {
            if let v = nearest(at: t)?.policyLoss?.max {
                return AttributedString(String(format: "%+.4f", v))
            }
            return AttributedString("— no data")
        }
        if let v = buckets.last?.policyLoss?.max {
            return AttributedString(String(format: "%+.4f", v))
        }
        return AttributedString("--")
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

extension PolicyLossChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
