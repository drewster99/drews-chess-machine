import SwiftUI
import SwiftUIFastCharts

/// Legal-only policy-entropy chart — Shannon entropy in nats over the
/// legal-renormalized softmax at the probed position.
///
/// The full-head `pEnt` (entropy over the entire 4864-cell policy
/// head) was removed because it conflates "is the policy concentrating
/// on a single legal move" with "is policy mass leaking onto illegal
/// moves." `pEntLegal` answers the first question on its own.
struct EntropyChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    /// Reference uniform-distribution entropy for the legal-only
    /// renormalized softmax. A typical chess position has ~30 legal
    /// moves, so log(30) ≈ 3.40 nats is a reasonable "fully diffuse
    /// over legal" baseline; the header reports the current value as
    /// a percentage of this reference so 100 % means uniform-over-legal.
    nonisolated static let maxLegalEntropy = log(30.0)

    var body: some View {
        FastLineChart(
            title: "Policy entropy",
            titleHelp: AttributedString("""
                Shannon entropy (in nats) of the legal-renormalized policy softmax — i.e. how diffuse \
                the model's move preferences are at the probed position, after illegal moves are \
                masked out. The header shows the current value as a percentage of log(30) ≈ 3.40 nats, \
                the diffuse-over-30-legal-moves baseline. Lower entropy means the model is committing \
                to a small number of moves; very low can signal collapse.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...yMaxObserved(),
            series: [
                FastChartSeries(
                    id: "pEntLegal",
                    color: .green,
                    lineWidth: 1.5,
                    data: .buckets(fastBuckets())
                )
            ],
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func yMaxObserved() -> Double {
        let observed = buckets.compactMap { $0.legalEntropy?.max }.max() ?? 0
        return Swift.max(observed * 1.05, Self.maxLegalEntropy * 1.05)
    }

    private func fastBuckets() -> [FastChartBucket] {
        var out: [FastChartBucket] = []
        out.reserveCapacity(buckets.count)
        for (i, b) in buckets.enumerated() {
            let r = b.legalEntropy
            out.append(FastChartBucket(
                id: i,
                x: b.elapsedSec,
                yMin: r?.min ?? .nan,
                yMax: r?.max ?? .nan
            ))
        }
        return out
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        func format(_ v: Double) -> String {
            String(format: "%.3f (%.1f%%)", v, v / Self.maxLegalEntropy * 100)
        }
        if let t = hoveredX {
            if let bucket = nearestBucket(at: t),
               let v = bucket.legalEntropy?.max {
                return AttributedString("pEntLegal \(format(v))")
            }
            return AttributedString("— no data")
        }
        if let v = buckets.last?.legalEntropy?.max {
            return AttributedString("pEntLegal \(format(v))")
        }
        return AttributedString("--")
    }

    private func nearestBucket(at t: Double) -> TrainingBucket? {
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

extension EntropyChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
