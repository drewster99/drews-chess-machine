import SwiftUI
import SwiftUIFastCharts

/// Two outcome-partitioned policy-loss series — `pLossWin` (green)
/// and `pLossLoss` (red). Includes a faint dashed 0-line so the
/// sign of each curve is readable at a glance.
struct PolicyLossSplitChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    var body: some View {
        let yRange = observedYRange()
        return FastLineChart(
            title: "pLoss split (W vs L)",
            titleHelp: AttributedString("""
                Policy loss restricted to batch positions whose game ended in a win (pLossWin, green) \
                versus a loss (pLossLoss, red), from the current player's perspective. Outcome-weighted \
                CE flips sign with the game's z, so the total-loss curve is hard to read; splitting by \
                outcome makes each side individually interpretable.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: yRange,
            series: [
                FastChartSeries(
                    id: "pLossWin",
                    color: .green,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.policyLossWin?.min ?? .nan,
                            yMax: b.policyLossWin?.max ?? .nan
                        )
                    })
                ),
                FastChartSeries(
                    id: "pLossLoss",
                    color: .red,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.policyLossLoss?.min ?? .nan,
                            yMax: b.policyLossLoss?.max ?? .nan
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
            legend: .off,
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func observedYRange() -> ClosedRange<Double> {
        let allMax = buckets.compactMap { $0.policyLossWin?.max }
            + buckets.compactMap { $0.policyLossLoss?.max }
        let allMin = buckets.compactMap { $0.policyLossWin?.min }
            + buckets.compactMap { $0.policyLossLoss?.min }
        let lo = Swift.min(allMin.min() ?? -0.5, 0) - 0.05
        let hi = Swift.max(allMax.max() ?? 0.5, 0) + 0.05
        if lo == hi { return (lo - 0.5)...(hi + 0.5) }
        return lo...hi
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let winV: Double?
        let lossV: Double?
        let isHovering = hoveredX != nil
        if let t = hoveredX {
            if let b = nearest(at: t) {
                winV = b.policyLossWin?.max
                lossV = b.policyLossLoss?.max
            } else {
                winV = nil
                lossV = nil
            }
        } else {
            winV = buckets.last?.policyLossWin?.max
            lossV = buckets.last?.policyLossLoss?.max
        }
        if isHovering && winV == nil && lossV == nil {
            return AttributedString("— no data")
        }
        if winV == nil && lossV == nil {
            return AttributedString("--")
        }
        let winStr = winV.map { String(format: "%+.4f", $0) } ?? "--"
        let lossStr = lossV.map { String(format: "%+.4f", $0) } ?? "--"
        var out = AttributedString("win ")
        var winPart = AttributedString(winStr)
        winPart.foregroundColor = .green
        out.append(winPart)
        out.append(AttributedString(" / loss "))
        var lossPart = AttributedString(lossStr)
        lossPart.foregroundColor = .red
        out.append(lossPart)
        return out
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
