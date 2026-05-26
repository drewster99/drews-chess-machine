import Charts
import SwiftUI

/// Stealth-mode `pDraw` watch histogram tile. Per-game bars: bucket
/// `i` shows the count of GAMES (in the rolling window) whose 8-ply
/// streak first completed at a ply in `[i*width, (i+1)*width)`. Hover
/// a bar to see that bucket's flag→draw precision — of the games in
/// the bucket, the fraction that actually finished as draws
/// (excluding ply-cap-terminated games per the locked plan decision).
///
/// Header summary surfaces the same two metrics aggregated across all
/// buckets: `games=A.A%` (of completed games, how many flagged at
/// least once) and `→draw=B.B%` (of flagged-non-cap-terminated games,
/// the fraction that drew).
///
/// `snapshot` is nil before the first `DrawWatchSnapshot` has been
/// mirrored off the tracker by the heartbeat; the tile renders an
/// all-zero placeholder in that case so column alignment in the chart
/// grid stays stable.
struct DrawWatchHistogramChart: View {
    let snapshot: DrawWatchSnapshot?

    @State private var hoveredBucketIndex: Int?

    private struct Bar: Identifiable, Equatable {
        let id: Int          // bucket index 0..<histogramBucketCount
        let label: String    // e.g. "0-40"
        let count: Int       // gamesFlaggedByBucket[id]
    }

    private static let barColor: Color = Color(hue: 0.58, saturation: 0.65, brightness: 0.85)

    private static func makeBars(from snap: DrawWatchSnapshot?) -> [Bar] {
        let n = DrawWatchTracker.histogramBucketCount
        let w = DrawWatchTracker.histogramBucketWidthPlies
        var out: [Bar] = []
        out.reserveCapacity(n)
        for i in 0..<n {
            let low = i * w
            let high = low + w
            // Last bucket is the "and up" overflow bin — labeled so a
            // future raise of `selfPlayMaxPliesPerGame` past `n*w`
            // makes the accumulation point obvious.
            let label = (i == n - 1) ? "\(low)+" : "\(low)-\(high)"
            let count = snap?.gamesFlaggedByBucket[safe: i] ?? 0
            out.append(Bar(id: i, label: label, count: count))
        }
        return out
    }

    var body: some View {
        let bars = Self.makeBars(from: snapshot)
        let maxCount = bars.map(\.count).max() ?? 0
        let header = headerText(snapshot: snapshot, bars: bars)
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(
                title: "Draw-watch (8-ply pDraw streak)",
                value: header,
                titleHelp: AttributedString("""
                    Stealth-mode monitor of the W/D/L value head during self-play. Each bar = number \
                    of distinct GAMES in the last 30 min whose 8-ply pDraw streak first completed at \
                    a ply inside that 40-ply bucket. Hover a bar to see that bucket's draw-precision \
                    (of those games, excluding ply-cap-terminated, the fraction that ended in a draw). \
                    Header: games=% of all completed games that flagged at least once · →draw=% of \
                    those (excluding ply-cap-terminated) that finished as draws. Threshold is the \
                    "Draw-Watch pDraw Threshold" param in Self-Play Sampling (default 0.95). \
                    Flagging does NOT terminate the game — purely observational.
                    """)
            )
            Chart(bars) { bar in
                BarMark(
                    x: .value("Bucket", bar.label),
                    y: .value("Count", bar.count)
                )
                .foregroundStyle(Self.barColor)
                .opacity(hoveredBucketIndex == nil || hoveredBucketIndex == bar.id ? 1.0 : 0.4)
            }
            .chartYScale(domain: 0...(maxCount > 0 ? Int(Double(maxCount) * 1.1) + 1 : 1))
            .chartXAxis {
                AxisMarks(preset: .aligned, values: .automatic) { value in
                    AxisValueLabel {
                        if let label = value.as(String.self) {
                            Text(label)
                                .font(.system(size: 6))
                                .monospacedDigit()
                        }
                    }
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(TrainingChartGridView.compactLabel(v))
                                .font(.system(size: 7))
                                .monospacedDigit()
                        }
                    }
                }
            }
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let point):
                                let origin = (proxy.plotFrame.map { geo[$0].origin } ?? .zero)
                                let xInPlot = point.x - origin.x
                                if let label: String = proxy.value(atX: xInPlot),
                                   let match = bars.first(where: { $0.label == label }) {
                                    if hoveredBucketIndex != match.id {
                                        hoveredBucketIndex = match.id
                                    }
                                } else if hoveredBucketIndex != nil {
                                    hoveredBucketIndex = nil
                                }
                            case .ended:
                                if hoveredBucketIndex != nil {
                                    hoveredBucketIndex = nil
                                }
                            }
                        }
                }
            }
        }
        .frame(height: 75)
        .chartCard()
    }

    private func headerText(snapshot: DrawWatchSnapshot?, bars: [Bar]) -> String {
        if let hoveredID = hoveredBucketIndex,
           let bar = bars.first(where: { $0.id == hoveredID }),
           let snap = snapshot {
            let acc = snap.drawAccuracyForBucket(hoveredID)
                .map { String(format: "%.1f%%", $0 * 100) } ?? "--"
            return "\(bar.label): games=\(bar.count) →draw=\(acc)"
        }
        guard let s = snapshot, s.totalGames > 0 else { return "-- (last 30 min)" }
        let pctG = s.fractionOfGamesFlagged.map { String(format: "%.1f%%", $0 * 100) } ?? "--"
        let acc = s.flagDrawAccuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "--"
        return "games=\(pctG) · →draw=\(acc) (last 30 min, N=\(s.totalGames))"
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
