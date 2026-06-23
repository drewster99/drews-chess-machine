import SwiftUI
import SwiftUIFastCharts

/// Two game-length series on one tile: the self-play rolling mean game
/// length (orange — `gameLength`, mean plies of completed self-play
/// games) and the sampled-minibatch rolling mean game length (blue —
/// `sampledBatchGameLength`, the position-weighted mean game-length-of-
/// origin over what the trainer actually pulls from the replay buffer).
///
/// The two answer different questions — "how long are the games we're
/// producing?" vs. "how long are the games we're training on?" — and are
/// not expected to coincide even with uniform sampling: the batch number
/// is position-weighted (a 300-ply game contributes 300 rows, a 40-ply
/// game 40), so it sits structurally above the per-game self-play mean.
/// The gap *widens* when the length-tilt sampling constraint is active;
/// that divergence is the signal the overlay is here to surface.
struct GameLengthChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    private static let selfPlayColor: Color = .orange
    private static let batchColor: Color = .blue

    var body: some View {
        return FastLineChart(
            title: "game length (self-play vs batch)",
            titleHelp: AttributedString("""
                Two rolling-mean game-length series, in plies. Orange = self-play (mean length of \
                completed self-play games, the rollingAvgLen on the [STATS] line — game-weighted, \
                60-second window). Blue = sampled batch (mean game-length-of-origin over the positions \
                the trainer draws from the replay buffer — position-weighted, 512-step window). The \
                batch line sits above self-play because position-weighting over-represents long games, \
                and diverges further when the length-tilt sampling constraint is active.
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
                            yMin: b.gameLength?.min ?? .nan,
                            yMax: b.gameLength?.max ?? .nan
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
                            yMin: b.sampledBatchGameLength?.min ?? .nan,
                            yMax: b.sampledBatchGameLength?.max ?? .nan
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
        let allMax = buckets.compactMap { $0.gameLength?.max }
            + buckets.compactMap { $0.sampledBatchGameLength?.max }
        let hi = allMax.max() ?? 1
        // Both series are non-negative ply counts — pin the floor at 0 so the
        // axis doesn't render a meaningless negative band.
        if hi <= 0 { return 0...1 }
        return 0...(hi * 1.1)
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let spV: Double?
        let batchV: Double?
        let isHovering = hoveredX != nil
        if let t = hoveredX {
            if let b = nearest(at: t) {
                spV = b.gameLength?.max
                batchV = b.sampledBatchGameLength?.max
            } else {
                spV = nil
                batchV = nil
            }
        } else {
            spV = buckets.last?.gameLength?.max
            batchV = buckets.last?.sampledBatchGameLength?.max
        }
        if spV == nil && batchV == nil {
            return AttributedString(isHovering ? "— no data" : "--")
        }
        func fmt(_ v: Double?) -> String {
            guard let v, v.isFinite else { return "--" }
            return String(Int(v.rounded()))
        }
        var out = AttributedString("sp ")
        var spPart = AttributedString("\(fmt(spV)) plies")
        spPart.foregroundColor = Self.selfPlayColor
        out.append(spPart)
        out.append(AttributedString(" / batch "))
        var batchPart = AttributedString("\(fmt(batchV)) plies")
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

extension GameLengthChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
