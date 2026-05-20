import Charts
import SwiftUI

/// Three-up histogram block embedded in `ArenaDetailPopover` when the
/// row's `TournamentRecord` carries an `extendedSummary`. The user
/// opens it by clicking an arena-history row.
///
/// Charts (top to bottom):
///   1. **Outcome by game length.** Stacked bar — wins (green), draws
///      (gray), losses (red) per 20-ply bucket. Tells you whether
///      losses are concentrated in short blunders or long endgame
///      grinds.
///   2. **Score by ply.** Line chart over 5-ply candidate-to-move
///      buckets, where the per-bucket score is computed the same way
///      a tournament's overall score is — `(W + 0.5·D) / N`, but
///      attributed per ply: every candidate-to-move ply gets a credit
///      of 1 / 0.5 / 0 for the eventual outcome of its game. So the
///      first dot tells you "of all candidate moves at plies 0-4,
///      what fraction belonged to games the candidate eventually
///      scored on?". Marker size encodes sample count so the eye can
///      down-weight sparse late-game buckets.
///   3. **Score by progress.** Same metric, bucketed by `ply /
///      gameLength` in 5% increments — orthogonal view that
///      normalizes out short-vs-long games.
///
/// The 0.5 dashed reference line on the score charts is the "even"
/// mark — bullets above mean the candidate is on track to score in
/// games at this stage; below means it's losing.
struct ArenaBreakdownsView: View {
    let summary: ArenaExtendedSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Breakdowns")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            wdlByLengthSection
            scoreByPlySection
            scoreByProgressSection
        }
    }

    // MARK: - WDL by length

    private var wdlByLengthSection: some View {
        // Filter to populated buckets; an arena with no games above
        // 100 plies shouldn't dedicate half the chart to empty rows.
        let buckets = summary.wdlByLength.filter { $0.count > 0 }
        return VStack(alignment: .leading, spacing: 4) {
            Text("Outcome by game length")
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)

            if buckets.isEmpty {
                emptyChartPlaceholder
            } else {
                WDLByLengthChart(buckets: buckets)
                    .frame(height: 110)
            }
        }
    }

    // MARK: - Score by ply

    private var scoreByPlySection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Win rate by ply (5-ply buckets, (W+0.5·D)/N)")
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)

            if summary.valueByPly.isEmpty {
                emptyChartPlaceholder
            } else {
                ScoreByPlyChart(buckets: summary.valueByPly)
                    .frame(height: 110)
            }
        }
    }

    // MARK: - Score by progress

    private var scoreByProgressSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Win rate by game progress (5% buckets)")
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)

            if summary.valueByProgress.isEmpty {
                emptyChartPlaceholder
            } else {
                ScoreByProgressChart(buckets: summary.valueByProgress)
                    .frame(height: 110)
            }
        }
    }

    private var emptyChartPlaceholder: some View {
        Text("no data")
            .font(.caption)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, minHeight: 60, alignment: .center)
    }
}

// MARK: - WDL by length chart

private struct WDLByLengthChart: View {
    let buckets: [ArenaWDLByLengthBucket]

    /// Three plot rows per bucket so Swift Charts can stack them
    /// inside a single `BarMark` series. Naming the outcome as a
    /// String keeps `.foregroundStyle(by:)` happy and gives us a
    /// readable legend.
    private struct Row: Identifiable {
        let id = UUID()
        let bucketLabel: String      // X-axis category, e.g. "0-19"
        let outcome: String          // "W" / "D" / "L"
        let count: Int
    }

    private var rows: [Row] {
        buckets.flatMap { bucket -> [Row] in
            let lab = label(for: bucket)
            return [
                Row(bucketLabel: lab, outcome: "W", count: bucket.wins),
                Row(bucketLabel: lab, outcome: "D", count: bucket.draws),
                Row(bucketLabel: lab, outcome: "L", count: bucket.losses)
            ]
        }
    }

    private var xOrder: [String] {
        buckets.map { label(for: $0) }
    }

    var body: some View {
        Chart(rows) { row in
            BarMark(
                x: .value("Length", row.bucketLabel),
                y: .value("Games", row.count)
            )
            .foregroundStyle(by: .value("Outcome", row.outcome))
        }
        .chartXScale(domain: xOrder)
        .chartForegroundStyleScale([
            "W": Color.green.opacity(0.85),
            "D": Color.gray.opacity(0.6),
            "L": Color.red.opacity(0.85)
        ])
        .chartXAxis {
            AxisMarks(values: .automatic) {
                AxisGridLine()
                AxisValueLabel()
                    .font(.system(size: 8))
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading) {
                AxisGridLine()
                AxisValueLabel()
                    .font(.system(size: 8))
            }
        }
        .chartLegend(position: .top, alignment: .trailing, spacing: 4)
    }

    private func label(for bucket: ArenaWDLByLengthBucket) -> String {
        if let hi = bucket.upperInclusive {
            return "\(bucket.lowerInclusive)-\(hi)"
        }
        return "\(bucket.lowerInclusive)+"
    }
}

// MARK: - Score by ply chart

private struct ScoreByPlyChart: View {
    let buckets: [ArenaValueByPlyBucket]

    private struct Point: Identifiable {
        let id = UUID()
        let midPly: Int
        let score: Double
        let count: Int
    }

    private var points: [Point] {
        buckets.map { b in
            // Use the bucket's midpoint as the x-coordinate so the
            // line traces through 2.5, 7.5, 12.5, ... and looks
            // visually centered in each 5-ply span.
            let mid = Double(b.lowerInclusive + b.upperInclusive) / 2.0
            return Point(midPly: Int(mid.rounded()), score: b.candidateScore, count: b.count)
        }
    }

    var body: some View {
        Chart(points) { p in
            LineMark(
                x: .value("Ply", p.midPly),
                y: .value("Score", p.score)
            )
            .foregroundStyle(Color.accentColor)
            PointMark(
                x: .value("Ply", p.midPly),
                y: .value("Score", p.score)
            )
            // Larger dots for buckets with more samples → small-n
            // bumps stop visually dominating the line.
            .symbolSize(Double(8 + min(p.count, 32)))
            .foregroundStyle(Color.accentColor)
        }
        .chartYScale(domain: 0...1)
        .chartYAxis {
            AxisMarks(position: .leading, values: [0, 0.5, 1]) { val in
                AxisGridLine()
                AxisValueLabel {
                    if let v = val.as(Double.self) {
                        Text(String(format: "%.1f", v))
                            .font(.system(size: 8))
                    }
                }
            }
        }
        .chartXAxis {
            AxisMarks {
                AxisGridLine()
                AxisValueLabel()
                    .font(.system(size: 8))
            }
        }
        .chartOverlay { proxy in
            referenceLine(at: 0.5, proxy: proxy)
        }
    }

    @ViewBuilder
    private func referenceLine(at y: Double, proxy: ChartProxy) -> some View {
        GeometryReader { geo in
            if let plotFrame = proxy.plotFrame,
               let yPos = proxy.position(forY: y) {
                let rect = geo[plotFrame]
                Path { path in
                    path.move(to: CGPoint(x: rect.minX, y: rect.origin.y + yPos))
                    path.addLine(to: CGPoint(x: rect.maxX, y: rect.origin.y + yPos))
                }
                .stroke(
                    Color.secondary.opacity(0.45),
                    style: StrokeStyle(lineWidth: 1, dash: [3, 3])
                )
            }
        }
    }
}

// MARK: - Score by progress chart

private struct ScoreByProgressChart: View {
    let buckets: [ArenaValueByProgressBucket]

    private struct Point: Identifiable {
        let id = UUID()
        let midPct: Double
        let score: Double
        let count: Int
    }

    private var points: [Point] {
        buckets.map { b in
            let mid = Double(b.lowerPercent + b.upperPercent) / 2.0
            return Point(midPct: mid, score: b.candidateScore, count: b.count)
        }
    }

    var body: some View {
        Chart(points) { p in
            LineMark(
                x: .value("Game progress %", p.midPct),
                y: .value("Score", p.score)
            )
            .foregroundStyle(Color.accentColor)
            PointMark(
                x: .value("Game progress %", p.midPct),
                y: .value("Score", p.score)
            )
            .symbolSize(Double(8 + min(p.count, 32)))
            .foregroundStyle(Color.accentColor)
        }
        .chartYScale(domain: 0...1)
        .chartXScale(domain: 0...100)
        .chartYAxis {
            AxisMarks(position: .leading, values: [0, 0.5, 1]) { val in
                AxisGridLine()
                AxisValueLabel {
                    if let v = val.as(Double.self) {
                        Text(String(format: "%.1f", v))
                            .font(.system(size: 8))
                    }
                }
            }
        }
        .chartXAxis {
            AxisMarks(values: [0, 25, 50, 75, 100]) { val in
                AxisGridLine()
                AxisValueLabel {
                    if let v = val.as(Int.self) {
                        Text("\(v)%")
                            .font(.system(size: 8))
                    }
                }
            }
        }
        .chartOverlay { proxy in
            referenceLine(at: 0.5, proxy: proxy)
        }
    }

    @ViewBuilder
    private func referenceLine(at y: Double, proxy: ChartProxy) -> some View {
        GeometryReader { geo in
            if let plotFrame = proxy.plotFrame,
               let yPos = proxy.position(forY: y) {
                let rect = geo[plotFrame]
                Path { path in
                    path.move(to: CGPoint(x: rect.minX, y: rect.origin.y + yPos))
                    path.addLine(to: CGPoint(x: rect.maxX, y: rect.origin.y + yPos))
                }
                .stroke(
                    Color.secondary.opacity(0.45),
                    style: StrokeStyle(lineWidth: 1, dash: [3, 3])
                )
            }
        }
    }
}
