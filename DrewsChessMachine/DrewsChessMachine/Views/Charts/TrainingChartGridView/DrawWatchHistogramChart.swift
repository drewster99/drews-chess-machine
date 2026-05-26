import Charts
import SwiftUI

/// Stealth-mode `pDraw` watch histogram tile. X axis is fixed-width
/// 20-ply buckets (`[0,20), [20,40), ..., [(B-1)*20, B*20)`); bars
/// show count of flag-fires whose `plyIndex` falls in the bucket.
/// Header summary surfaces session-wide `% games flagged`, `% plies
/// in flagged streaks`, and the v1 calibration metric `flag→draw
/// precision` (excluding cap-terminated games from the precision
/// denominator — see `DRAW_WATCH_PLAN.md` locked decision #5).
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
        let label: String    // e.g. "0-20"
        let count: Int
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
            // Last bucket is the "and up" overflow bin — label it so
            // a future raise of `selfPlayMaxPliesPerGame` past
            // `n * w` makes the histogram's accumulation point obvious.
            let label = (i == n - 1) ? "\(low)+" : "\(low)-\(high)"
            let count = snap?.plyBucketHistogram[safe: i] ?? 0
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
                title: "Draw-watch (pDraw ≥ 0.95 × 8 plies)",
                value: header,
                titleHelp: AttributedString("""
                    Stealth-mode monitor of the W/D/L value head during self-play. Each bar is the count \
                    of "draw-watch flag" events whose firing ply fell in that 20-ply bucket. A flag \
                    fires when pDraw ≥ 0.95 for 8 consecutive plies on the same game; the game continues \
                    playing — flagging does NOT terminate it. Header reads: flags total · % of games \
                    that flagged at least once · % of plies inside flagged streaks · "→draw" = of \
                    flagged games (excluding ply-cap-terminated), the fraction that actually finished \
                    as draws (the flag's draw-precision calibration).
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
           let bar = bars.first(where: { $0.id == hoveredID }) {
            return "\(bar.label) plies: \(bar.count)"
        }
        guard let s = snapshot else { return "--" }
        let pctG = s.fractionOfGamesFlagged.map { String(format: "%.1f%%", $0 * 100) } ?? "--"
        let pctP = s.fractionOfPliesInFlaggedStreaks.map { String(format: "%.1f%%", $0 * 100) } ?? "--"
        let acc = s.flagDrawAccuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "--"
        return "flags=\(s.flags.count) · games=\(pctG) · plies=\(pctP) · →draw=\(acc)"
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
