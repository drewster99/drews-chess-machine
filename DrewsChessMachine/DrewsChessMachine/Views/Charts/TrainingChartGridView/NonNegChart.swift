import SwiftUI
import SwiftUIFastCharts

/// Above-uniform policy count chart — legal vs illegal counts on a
/// fixed `0...policySize` Y axis.
struct NonNegChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    var body: some View {
        let policyMax = ChessNetwork.policySize
        return FastLineChart(
            title: "Above-uniform policy count",
            titleHelp: AttributedString("""
                How many of the 4864 policy cells have a softmax probability above uniform \
                (1 / 4864). Legal (mint) is the count over legal moves; Illegal (red) is the count \
                over illegal moves. A rising illegal count means the policy is putting non-trivial \
                mass on illegal moves — wasted probability that legal-renorm has to clean up.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...Double(policyMax),
            series: [
                FastChartSeries(
                    id: "Legal",
                    color: .mint,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.policyNonNegCount?.min ?? .nan,
                            yMax: b.policyNonNegCount?.max ?? .nan
                        )
                    })
                ),
                FastChartSeries(
                    id: "Illegal",
                    color: .red,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.policyNonNegIllegalCount?.min ?? .nan,
                            yMax: b.policyNonNegIllegalCount?.max ?? .nan
                        )
                    })
                )
            ],
            legend: .auto,
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let legal: Double?
        let illegal: Double?
        let isHovering = hoveredX != nil
        if let t = hoveredX {
            if let b = nearest(at: t) {
                legal = b.policyNonNegCount?.max
                illegal = b.policyNonNegIllegalCount?.max
            } else {
                legal = nil
                illegal = nil
            }
        } else {
            legal = buckets.last?.policyNonNegCount?.max
            illegal = buckets.last?.policyNonNegIllegalCount?.max
        }
        if isHovering && legal == nil && illegal == nil {
            return AttributedString("— no data")
        }
        let legalStr = legal.map { String(Int($0)) } ?? "--"
        let illegalStr = illegal.map { String(Int($0)) } ?? "--"
        return AttributedString("legal \(legalStr) • illegal \(illegalStr)")
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

extension NonNegChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
