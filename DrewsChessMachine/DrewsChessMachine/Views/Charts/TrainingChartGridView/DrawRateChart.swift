import SwiftUI
import SwiftUIFastCharts

/// Two draw-rate series on one tile: the self-play rolling draw rate
/// (orange — `drawFraction`, the per-game fraction of completed self-play
/// games that ended in a natural draw) and the sampled-minibatch rolling
/// draw rate (blue — `sampledBatchDrawFraction`, the per-position fraction
/// of the positions the trainer draws whose game was a draw).
///
/// Overlaying the two shows the effect of the draw-cap sampling
/// constraint: with the cap active the batch line is clamped *below* the
/// self-play line. Even with no constraint the two differ — the batch
/// rate is position-weighted, and draws tend to be longer games, so draws
/// are over-represented per-position relative to the per-game self-play
/// rate.
struct DrawRateChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    private static let selfPlayColor: Color = .orange
    private static let batchColor: Color = .blue

    var body: some View {
        return FastLineChart(
            title: "draw rate (self-play vs batch)",
            titleHelp: AttributedString("""
                Two rolling draw-rate series, in [0, 1]. Orange = self-play (per-game fraction of \
                completed self-play games that ended in a natural draw — stalemate / 50-move / 3-fold \
                / insufficient material; 60-second window). Blue = sampled batch (per-position fraction \
                of the minibatch positions the trainer drew whose game was a draw; 512-step window). \
                With the draw-cap sampling constraint active the batch line is clamped below self-play; \
                even without it the position-weighted batch rate runs higher because draws are longer \
                games and contribute more positions.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: observedYRange(),
            series: [
                FastChartSeries(
                    id: "self-play",
                    color: Self.selfPlayColor,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.drawFraction?.min ?? .nan,
                            yMax: b.drawFraction?.max ?? .nan
                        )
                    })
                ),
                FastChartSeries(
                    id: "batch",
                    color: Self.batchColor,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.sampledBatchDrawFraction?.min ?? .nan,
                            yMax: b.sampledBatchDrawFraction?.max ?? .nan
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

    private func observedYRange() -> ClosedRange<Double> {
        let allMax = buckets.compactMap { $0.drawFraction?.max }
            + buckets.compactMap { $0.sampledBatchDrawFraction?.max }
        let hi = allMax.max() ?? 0
        // Draw rate is a fraction in [0, 1]; pin the floor at 0 and give the
        // ceiling a little headroom, with a small minimum so a near-zero rate
        // still renders a usable axis rather than collapsing to a flat line.
        return 0...Swift.max(hi * 1.1, 0.05)
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let spV: Double?
        let batchV: Double?
        let isHovering = hoveredX != nil
        if let t = hoveredX {
            if let b = nearest(at: t) {
                spV = b.drawFraction?.max
                batchV = b.sampledBatchDrawFraction?.max
            } else {
                spV = nil
                batchV = nil
            }
        } else {
            spV = buckets.last?.drawFraction?.max
            batchV = buckets.last?.sampledBatchDrawFraction?.max
        }
        if spV == nil && batchV == nil {
            return AttributedString(isHovering ? "— no data" : "--")
        }
        func fmt(_ v: Double?) -> String {
            guard let v, v.isFinite else { return "--" }
            return String(format: "%.3f", v)
        }
        var out = AttributedString("sp ")
        var spPart = AttributedString(fmt(spV))
        spPart.foregroundColor = Self.selfPlayColor
        out.append(spPart)
        out.append(AttributedString(" / batch "))
        var batchPart = AttributedString(fmt(batchV))
        batchPart.foregroundColor = Self.batchColor
        out.append(batchPart)
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

extension DrawRateChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
