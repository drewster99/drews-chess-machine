import SwiftUI
import SwiftUIFastCharts

/// Legal mass sum tile — fraction of softmax mass on legal moves at
/// the probed position. Y axis is tiered to give early-training
/// signal more vertical room: top is 0.50 while the session's
/// running maximum stays below 0.25, expands to 0.75 when the max
/// crosses 0.25, and to 1.00 once the max crosses 0.50.
struct LegalMassChart: View {
    let buckets: [TrainingBucket]
    /// Session-wide running max of `rollingLegalMass`. Drives the tier.
    let allTimeMax: Double
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    private var yAxisTop: Double {
        if allTimeMax >= 0.5 { return 1.0 }
        if allTimeMax >= 0.25 { return 0.75 }
        return 0.5
    }

    /// Quartile ticks 0, 0.25, … up through the current top.
    private var yLabelCount: Int {
        switch yAxisTop {
        case 1.0: return 5
        case 0.75: return 4
        default: return 3
        }
    }

    var body: some View {
        FastLineChart(
            title: "Legal mass sum",
            titleHelp: AttributedString("""
                Fraction of the softmax policy mass placed on legal moves at the probed position. \
                Climbs from near 0 at init toward 1.0 once the network has learned to put most \
                probability on legal moves. The Y axis is tiered to give early-training signal more \
                vertical room: top expands from 0.5 to 0.75 to 1.0 as the session's all-time max grows.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...yAxisTop,
            series: [
                FastChartSeries(
                    id: "Legal mass",
                    color: .cyan,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.legalMass?.min ?? .nan,
                            yMax: b.legalMass?.max ?? .nan
                        )
                    })
                )
            ],
            yLabelCount: yLabelCount,
            yLabelFormatter: { String(format: "%.2f", $0) },
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        if let t = hoveredX {
            if let v = nearest(at: t)?.legalMass?.max {
                return AttributedString(String(format: "%.4f%%", v * 100))
            }
            return AttributedString("— no data")
        }
        if let v = buckets.last?.legalMass?.max {
            return AttributedString(String(format: "%.4f%%", v * 100))
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

extension LegalMassChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.allTimeMax == rhs.allTimeMax
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
