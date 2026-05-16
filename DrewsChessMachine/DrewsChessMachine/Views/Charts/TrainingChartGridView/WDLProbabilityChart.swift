import SwiftUI
import SwiftUIFastCharts

/// Value-head row tile: the W/D/L softmax batch means (`pW` green,
/// `pD` gray, `pL` red), summing to ≈ 1 and clamped to `[0, 1]`.
/// 0.75 dashed reference marks the draw-bias init — `pD` trending
/// up to and staying there is the regression-toward-collapse signal.
struct WDLProbabilityChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    var body: some View {
        FastLineChart(
            title: "value W/D/L probabilities",
            titleHelp: AttributedString("""
                Batch-mean of the value head's three-way softmax probabilities — pW (win, green), \
                pD (draw, gray), pL (loss, red). At init, the head's [0, ln 6, 0] bias produces \
                pD ≈ 0.75 with pW = pL ≈ 0.125 (the dashed reference line). Healthy training pulls \
                pD down toward the buffer's true draw rate while pW and pL rise and stay roughly \
                symmetric. pD → 1 is the regression-toward-collapse signal the WDL head was adopted \
                to avoid.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...1,
            series: [
                FastChartSeries(
                    id: "pW",
                    color: .green,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.valueProbWin?.min ?? .nan,
                            yMax: b.valueProbWin?.max ?? .nan
                        )
                    })
                ),
                FastChartSeries(
                    id: "pD",
                    color: .gray,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.valueProbDraw?.min ?? .nan,
                            yMax: b.valueProbDraw?.max ?? .nan
                        )
                    })
                ),
                FastChartSeries(
                    id: "pL",
                    color: .red,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.valueProbLoss?.min ?? .nan,
                            yMax: b.valueProbLoss?.max ?? .nan
                        )
                    })
                )
            ],
            referenceLines: [
                FastChartReferenceLine(
                    id: "init",
                    y: 0.75,
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

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let w: Double?, d: Double?, l: Double?
        let isHovering = hoveredX != nil
        if let t = hoveredX {
            if let b = nearest(at: t) {
                w = b.valueProbWin?.max
                d = b.valueProbDraw?.max
                l = b.valueProbLoss?.max
            } else {
                w = nil; d = nil; l = nil
            }
        } else {
            w = buckets.last?.valueProbWin?.max
            d = buckets.last?.valueProbDraw?.max
            l = buckets.last?.valueProbLoss?.max
        }
        if isHovering && w == nil && d == nil && l == nil {
            return AttributedString("— no data")
        }
        if w == nil && d == nil && l == nil {
            return AttributedString("--")
        }
        let wStr = w.map { String(format: "%.3f", $0) } ?? "--"
        let dStr = d.map { String(format: "%.3f", $0) } ?? "--"
        let lStr = l.map { String(format: "%.3f", $0) } ?? "--"
        var out = AttributedString("W ")
        var wPart = AttributedString(wStr); wPart.foregroundColor = .green; out.append(wPart)
        out.append(AttributedString(" / D "))
        var dPart = AttributedString(dStr); dPart.foregroundColor = .gray; out.append(dPart)
        out.append(AttributedString(" / L "))
        var lPart = AttributedString(lStr); lPart.foregroundColor = .red; out.append(lPart)
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
