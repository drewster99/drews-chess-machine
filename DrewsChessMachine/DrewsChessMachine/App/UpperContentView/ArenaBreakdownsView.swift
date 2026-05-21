import Charts
import SwiftUI

/// Three-up histogram block embedded in `ArenaDetailView` when the
/// row's `TournamentRecord` carries an `extendedSummary`. The user
/// opens it by clicking an arena-history row.
///
/// Charts (top to bottom):
///   1. **Outcome by game length.** Stacked bar — wins (green), draws
///      (gray), losses (red) per 20-ply bucket. Tells you whether
///      losses are concentrated in short blunders or long endgame
///      grinds.
///   2. **Samples per ply bucket.** Bar chart of the per-ply-bucket
///      sample count, X-aligned with the win-rate-by-ply chart below
///      so each bar sits under the win-rate dot whose `N` it is.
///   3. **Score by ply.** Line chart over 20-ply candidate-to-move
///      buckets, where the per-bucket score is computed the same way
///      a tournament's overall score is — `(W + 0.5·D) / N`, but
///      attributed per ply: every candidate-to-move ply gets a credit
///      of 1 / 0.5 / 0 for the eventual outcome of its game. So the
///      first dot tells you "of all candidate moves at plies 0-4,
///      what fraction belonged to games the candidate eventually
///      scored on?". Marker size encodes sample count so the eye can
///      down-weight sparse late-game buckets.
///   4. **Value-head by ply.** Line chart of the mean value-head
///      scalar `p_win − p_loss` over the same 20-ply buckets — what
///      the network *thought* of its position, vs. how the game
///      actually scored.
///   5. **Score by progress.** Same `(W + 0.5·D)/N` metric, bucketed
///      by `ply / gameLength` in 5% increments — orthogonal view that
///      normalizes out short-vs-long games.
///   6. **Value-head by progress.** Mean value-head scalar bucketed
///      the same way, in 5% game-progress increments.
///
/// The 0.5 dashed reference line on the score charts is the "even"
/// mark — bullets above mean the candidate is on track to score in
/// games at this stage; below means it's losing.
///
/// Hovering any of these charts surfaces a readout card with the
/// hovered bucket's underlying counts.
struct ArenaBreakdownsView: View {
    let summary: ArenaExtendedSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Breakdowns")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            wdlByLengthSection
            plySampleCountSection
            scoreByPlySection
            valueByPlySection
            scoreByProgressSection
            valueByProgressSection
        }
    }

    // MARK: - WDL by length

    private var wdlByLengthSection: some View {
        // Filter to populated buckets; an arena with no games above
        // 100 plies shouldn't dedicate half the chart to empty rows.
        let buckets = summary.wdlByLength.filter { $0.count > 0 }
        return VStack(alignment: .leading, spacing: 4) {
            sectionHeader(
                "Outcome by game length",
                info: "Every completed arena game, bucketed by length in 20-ply bands and stacked by the candidate's outcome — win (green), draw (gray), loss (red). Bar height is the game count. Shows whether losses cluster in short games or long ones."
            )

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
            sectionHeader(
                "Win rate by ply (20-ply buckets, (W+0.5·D)/N)",
                info: "Each candidate-to-move ply is credited with its game's final result: 1 for a win, ½ for a draw, 0 for a loss. Per 20-ply bucket the chart plots (W + 0.5·D) / N over those plies — the score the arena assigns overall, resolved by how deep into the game the ply occurred. Dot size grows with the bucket's sample count."
            )

            if summary.valueByPly.isEmpty {
                emptyChartPlaceholder
            } else {
                ScoreByPlyChart(buckets: summary.valueByPly)
                    .frame(height: 110)
            }
        }
    }

    // MARK: - Samples per ply bucket

    private var plySampleCountSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader(
                "Candidate moves per 20-ply bucket",
                info: "How many candidate-to-move plies fell in each 20-ply bucket — the exact denominator N behind the matching point on the win-rate-by-ply chart below. Each game contributes about one sample per candidate move (~10 per bucket), so tall bars mark a well-supported curve and a short tail means only a few long games reached that depth."
            )

            if summary.valueByPly.isEmpty {
                emptyChartPlaceholder
            } else {
                PlySampleCountChart(buckets: summary.valueByPly)
                    .frame(height: 110)
            }
        }
    }

    // MARK: - Value-head by ply

    private var valueByPlySection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader(
                "Value-head by ply (mean p_win − p_loss)",
                info: "The mean of the candidate network's value-head output — p_win − p_loss, in [−1, +1] — over every candidate-to-move ply in each 20-ply bucket. This is what the network believed about its position; win-rate-by-ply is what actually happened. Comparing the two is a calibration check."
            )

            if summary.valueByPly.isEmpty {
                emptyChartPlaceholder
            } else {
                ValueByPlyChart(buckets: summary.valueByPly)
                    .frame(height: 110)
            }
        }
    }

    // MARK: - Score by progress

    private var scoreByProgressSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader(
                "Win rate by game progress (ply ÷ length)",
                info: "The same (W + 0.5·D) / N scoring as win-rate-by-ply, but bucketed by game progress — ply ÷ total game length — in 5% bands. Dividing by game length makes the curve comparable across short and long games."
            )

            if summary.valueByProgress.isEmpty {
                emptyChartPlaceholder
            } else {
                ScoreByProgressChart(buckets: summary.valueByProgress)
                    .frame(height: 110)
            }
        }
    }

    // MARK: - Value-head by progress

    private var valueByProgressSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader(
                "Value-head by game progress (ply ÷ length)",
                info: "The mean value-head scalar (p_win − p_loss) over candidate-to-move plies, bucketed by game progress (ply ÷ length) in 5% bands — the value-head companion to win-rate-by-game-progress."
            )

            if summary.valueByProgress.isEmpty {
                emptyChartPlaceholder
            } else {
                ValueByProgressChart(buckets: summary.valueByProgress)
                    .frame(height: 110)
            }
        }
    }

    /// A chart section's title row: the caption plus an info button
    /// that explains what the chart plots and how it is computed.
    @ViewBuilder
    private func sectionHeader(_ title: String, info: String) -> some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
            ChartInfoButton(explanation: info)
        }
    }

    private var emptyChartPlaceholder: some View {
        Text("no data")
            .font(.caption)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, minHeight: 60, alignment: .center)
    }
}

// MARK: - Chart info button

/// A small `ⓘ` button shown next to a chart's title. Tapping it opens
/// a popover that explains, in plain language, what the chart plots
/// and how its value is computed.
private struct ChartInfoButton: View {
    let explanation: String
    @State private var showing = false

    var body: some View {
        Button(action: { showing = true }, label: {
            Image(systemName: "info.circle")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        })
        .buttonStyle(.plain)
        .popover(isPresented: $showing, arrowEdge: .top) {
            Text(explanation)
                .font(.callout)
                .multilineTextAlignment(.leading)
                .padding(14)
                .frame(width: 320)
        }
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
                // Emitted loss-first so the stack reads L (bottom),
                // D, W (top) — wins crown the bar.
                Row(bucketLabel: lab, outcome: "L", count: bucket.losses),
                Row(bucketLabel: lab, outcome: "D", count: bucket.draws),
                Row(bucketLabel: lab, outcome: "W", count: bucket.wins)
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
        .chartOverlay { proxy in
            ArenaChartHoverOverlay(
                proxy: proxy,
                columnXs: xOrder,
                markerY: { _ in nil },
                readout: { readout(for: buckets[$0]) }
            )
        }
    }

    private func readout(for bucket: ArenaWDLByLengthBucket) -> ChartHoverReadout {
        ChartHoverReadout(
            title: pliesTitle(for: bucket),
            rows: [
                ChartHoverReadout.Row(label: "Games", value: "\(bucket.count)"),
                ChartHoverReadout.Row(
                    label: "W·D·L",
                    value: "\(bucket.wins) · \(bucket.draws) · \(bucket.losses)"
                ),
                ChartHoverReadout.Row(
                    label: "Win rate",
                    value: arenaBreakdownPercent(bucket.candidateScore)
                )
            ]
        )
    }

    private func pliesTitle(for bucket: ArenaWDLByLengthBucket) -> String {
        if let hi = bucket.upperInclusive {
            return "\(bucket.lowerInclusive)–\(hi) plies"
        }
        return "\(bucket.lowerInclusive)+ plies"
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
            // Use the bucket's midpoint as the x-coordinate so each
            // dot sits visually centered in its ply span.
            let mid = Double(b.lowerInclusive + b.upperInclusive) / 2.0
            return Point(midPly: Int(mid.rounded()), score: b.candidateScore, count: b.count)
        }
    }

    var body: some View {
        let pts = points
        return Chart(pts) { p in
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
        .winRateYAxis(scores: pts.map(\.score))
        .chartXAxis {
            AxisMarks {
                AxisGridLine()
                AxisValueLabel()
                    .font(.system(size: 8))
            }
        }
        .chartOverlay { proxy in
            ArenaChartHoverOverlay(
                proxy: proxy,
                columnXs: pts.map(\.midPly),
                referenceLineY: 0.5,
                markerY: { pts[$0].score },
                readout: { arenaPlyReadout(for: buckets[$0]) }
            )
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
        let pts = points
        return Chart(pts) { p in
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
        .winRateYAxis(scores: pts.map(\.score))
        .chartXScale(domain: 0...100)
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
            ArenaChartHoverOverlay(
                proxy: proxy,
                columnXs: pts.map(\.midPct),
                referenceLineY: 0.5,
                markerY: { pts[$0].score },
                readout: { arenaProgressReadout(for: buckets[$0]) }
            )
        }
    }
}

// MARK: - Value by ply chart

private struct ValueByPlyChart: View {
    let buckets: [ArenaValueByPlyBucket]

    private struct Point: Identifiable {
        let id = UUID()
        let midPly: Int
        let value: Double
        let count: Int
    }

    private var points: [Point] {
        buckets.map { b in
            // Use the bucket's midpoint as the x-coordinate so each
            // dot sits visually centered in its ply span.
            let mid = Double(b.lowerInclusive + b.upperInclusive) / 2.0
            return Point(midPly: Int(mid.rounded()), value: Double(b.mean), count: b.count)
        }
    }

    var body: some View {
        let pts = points
        return Chart(pts) { p in
            LineMark(
                x: .value("Ply", p.midPly),
                y: .value("Value", p.value)
            )
            .foregroundStyle(Color.accentColor)
            PointMark(
                x: .value("Ply", p.midPly),
                y: .value("Value", p.value)
            )
            // Larger dots for buckets with more samples → small-n
            // bumps stop visually dominating the line.
            .symbolSize(Double(8 + min(p.count, 32)))
            .foregroundStyle(Color.accentColor)
        }
        .valueYAxis(values: pts.map(\.value))
        .chartXAxis {
            AxisMarks {
                AxisGridLine()
                AxisValueLabel()
                    .font(.system(size: 8))
            }
        }
        .chartOverlay { proxy in
            ArenaChartHoverOverlay(
                proxy: proxy,
                columnXs: pts.map(\.midPly),
                referenceLineY: 0.0,
                markerY: { pts[$0].value },
                readout: { arenaPlyReadout(for: buckets[$0]) }
            )
        }
    }
}

// MARK: - Value by progress chart

private struct ValueByProgressChart: View {
    let buckets: [ArenaValueByProgressBucket]

    private struct Point: Identifiable {
        let id = UUID()
        let midPct: Double
        let value: Double
        let count: Int
    }

    private var points: [Point] {
        buckets.map { b in
            let mid = Double(b.lowerPercent + b.upperPercent) / 2.0
            return Point(midPct: mid, value: Double(b.mean), count: b.count)
        }
    }

    var body: some View {
        let pts = points
        return Chart(pts) { p in
            LineMark(
                x: .value("Game progress %", p.midPct),
                y: .value("Value", p.value)
            )
            .foregroundStyle(Color.accentColor)
            PointMark(
                x: .value("Game progress %", p.midPct),
                y: .value("Value", p.value)
            )
            .symbolSize(Double(8 + min(p.count, 32)))
            .foregroundStyle(Color.accentColor)
        }
        .valueYAxis(values: pts.map(\.value))
        .chartXScale(domain: 0...100)
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
            ArenaChartHoverOverlay(
                proxy: proxy,
                columnXs: pts.map(\.midPct),
                referenceLineY: 0.0,
                markerY: { pts[$0].value },
                readout: { arenaProgressReadout(for: buckets[$0]) }
            )
        }
    }
}

// MARK: - Samples per ply bucket chart

private struct PlySampleCountChart: View {
    let buckets: [ArenaValueByPlyBucket]

    private struct Point: Identifiable {
        let id = UUID()
        let midPly: Int
        let count: Int
    }

    private var points: [Point] {
        buckets.map { b in
            // Use the bucket's midpoint as the x-coordinate so each
            // bar sits under the matching win-rate-by-ply dot.
            let mid = Double(b.lowerInclusive + b.upperInclusive) / 2.0
            return Point(midPly: Int(mid.rounded()), count: b.count)
        }
    }

    var body: some View {
        let pts = points
        return Chart(pts) { p in
            BarMark(
                x: .value("Ply", p.midPly),
                y: .value("Samples", p.count)
            )
            .foregroundStyle(Color.accentColor.opacity(0.55))
        }
        .chartXAxis {
            AxisMarks {
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
        .chartOverlay { proxy in
            ArenaChartHoverOverlay(
                proxy: proxy,
                columnXs: pts.map(\.midPly),
                markerY: { _ in nil },
                readout: { arenaPlyReadout(for: buckets[$0]) }
            )
        }
    }
}

// MARK: - Shared hover overlay

/// Hover + reference-line overlay shared by every arena-breakdown
/// chart. Placed inside `.chartOverlay`, it draws — in back-to-front
/// order — an optional dashed horizontal reference line, a transparent
/// hover-capture layer over the plot, and the `ChartHoverDecorations`
/// card for whatever column the cursor lands on.
///
/// Generic over the chart's X value: `Int` ply, `Double` progress %,
/// or `String` length-bucket label. `columnXs` are the plottable X
/// coordinates of the chart's columns in index order — the hit test
/// snaps the cursor to the nearest one and reports that index.
/// `markerY` returns a column's data-space Y for the hover ring, or
/// `nil` for bar charts that have no single point to ring; `readout`
/// builds the card for the resolved index.
///
/// The hovered-column state lives on this overlay rather than on each
/// chart, so moving the cursor invalidates only the overlay, not the
/// whole `Chart`.
private struct ArenaChartHoverOverlay<X: Plottable>: View {
    let proxy: ChartProxy
    let columnXs: [X]
    /// Data-space Y for the dashed "even" reference line, or `nil` for
    /// charts that draw none.
    var referenceLineY: Double?
    /// Data-space Y of column `i`'s data point, for the hover ring.
    /// `nil` suppresses the ring (bar charts).
    let markerY: (Int) -> Double?
    let readout: (Int) -> ChartHoverReadout

    @State private var hoveredIndex: Int?

    var body: some View {
        GeometryReader { geo in
            if let plotFrame = proxy.plotFrame {
                let plotRect = geo[plotFrame]
                ZStack {
                    if let y = referenceLineY {
                        referenceLine(at: y, plotRect: plotRect)
                    }
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onContinuousHover { phase in
                            handleHover(phase, plotRect: plotRect)
                        }
                    if let idx = hoveredIndex, columnXs.indices.contains(idx) {
                        decorations(for: idx, plotRect: plotRect)
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
    }

    private func decorations(for idx: Int, plotRect: CGRect) -> ChartHoverDecorations {
        let highlightX = (proxy.position(forX: columnXs[idx]) ?? 0) + plotRect.minX
        let marker = markerY(idx)
            .flatMap { proxy.position(forY: $0) }
            .map { $0 + plotRect.minY }
        return ChartHoverDecorations(
            highlightX: highlightX,
            markerY: marker,
            plotRect: plotRect,
            readout: readout(idx)
        )
    }

    /// Snap the cursor to the nearest column and publish its index.
    private func handleHover(_ phase: HoverPhase, plotRect: CGRect) {
        switch phase {
        case .active(let location):
            guard location.x >= plotRect.minX, location.x <= plotRect.maxX else {
                hoveredIndex = nil
                return
            }
            var bestIndex: Int?
            var bestDistance = CGFloat.greatestFiniteMagnitude
            for (i, x) in columnXs.enumerated() {
                guard let px = proxy.position(forX: x) else { continue }
                let distance = abs(px + plotRect.minX - location.x)
                if distance < bestDistance {
                    bestDistance = distance
                    bestIndex = i
                }
            }
            hoveredIndex = bestIndex
        case .ended:
            hoveredIndex = nil
        }
    }

    @ViewBuilder
    private func referenceLine(at y: Double, plotRect: CGRect) -> some View {
        if let yPos = proxy.position(forY: y) {
            Path { path in
                path.move(to: CGPoint(x: plotRect.minX, y: plotRect.minY + yPos))
                path.addLine(to: CGPoint(x: plotRect.maxX, y: plotRect.minY + yPos))
            }
            .stroke(
                Color.secondary.opacity(0.45),
                style: StrokeStyle(lineWidth: 1, dash: [3, 3])
            )
        }
    }
}

// MARK: - Shared win-rate / value Y-axis

private extension View {
    /// Y-axis treatment shared by the score-by-ply and
    /// score-by-progress charts. Arena win rate hugs the 0.5 "even"
    /// mark tightly — a full-scale axis flattens it into a
    /// featureless band — so the axis instead pins to a narrow fixed
    /// window straddling 0.5 symmetrically, with one gridline + label
    /// per stride step. The window widens only far enough to contain
    /// any score that genuinely falls outside it, so a lopsided arena
    /// still fits while the common, near-even case stays
    /// high-resolution.
    func winRateYAxis(scores: [Double]) -> some View {
        let lower = min(0.46, scores.min() ?? 0.46)
        let upper = max(0.54, scores.max() ?? 0.54)
        return self
            .chartYScale(domain: lower...upper)
            .chartYAxis {
                AxisMarks(position: .leading, values: .stride(by: 0.01)) { val in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = val.as(Double.self) {
                            Text(String(format: "%.2f", v))
                                .font(.system(size: 8))
                        }
                    }
                }
            }
    }

    /// Y-axis treatment shared by the value-head-by-ply and
    /// value-head-by-progress charts. The value scalar `p_win − p_loss`
    /// hugs 0 for a near-even engine — a full `[-1, +1]` axis flattens
    /// it into a featureless band — so the axis instead pins to a
    /// narrow window straddling 0, with one gridline + label per
    /// stride step. The window widens only far enough to contain any
    /// value that genuinely falls outside it, so a lopsided arena
    /// still fits while the common, near-even case stays
    /// high-resolution.
    func valueYAxis(values: [Double]) -> some View {
        let lower = min(-0.05, values.min() ?? -0.05)
        let upper = max(0.05, values.max() ?? 0.05)
        return self
            .chartYScale(domain: lower...upper)
            .chartYAxis {
                AxisMarks(position: .leading, values: .stride(by: 0.01)) { val in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = val.as(Double.self) {
                            Text(String(format: "%+.2f", v))
                                .font(.system(size: 8))
                        }
                    }
                }
            }
    }
}

// MARK: - Hover readout

/// Compact label/value card shown while hovering an arena breakdown
/// chart. Sized to its own content; the caller pins it into a chart
/// corner via `ChartHoverDecorations`.
private struct ChartHoverReadout: View {
    struct Row: Identifiable {
        var id: String { label }
        let label: String
        let value: String
    }

    let title: String
    let rows: [Row]

    /// Fixed width so `ChartHoverDecorations` can place the card
    /// adjacent to the cursor with known geometry.
    static let cardWidth: CGFloat = 168

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 9, weight: .semibold))
            ForEach(rows) { row in
                HStack(spacing: 12) {
                    Text(row.label)
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                    Text(row.value)
                        .monospacedDigit()
                }
                .font(.system(size: 9))
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .frame(width: Self.cardWidth, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 5))
        .overlay {
            RoundedRectangle(cornerRadius: 5)
                .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 0.5)
        }
    }
}

/// Hover decorations drawn over an arena breakdown chart: a thin
/// vertical rule at the hovered column, an optional ring on the
/// hovered data point, and the readout card placed just beside the
/// cursor — to its right when the cursor is in the plot's left half,
/// to its left otherwise — clamped to stay within the plot so it
/// reads close to whatever is being pointed at. Hit-testing is
/// disabled so the chart's transparent hover-capture layer keeps
/// receiving events through it.
private struct ChartHoverDecorations: View {
    /// Screen-space X of the hovered column (GeometryReader space).
    let highlightX: CGFloat
    /// Screen-space Y of the hovered data point; `nil` for the bar
    /// chart, where a stacked column has no single point to ring.
    let markerY: CGFloat?
    let plotRect: CGRect
    let readout: ChartHoverReadout

    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color.secondary.opacity(0.35))
                .frame(width: 1, height: plotRect.height)
                .position(x: highlightX, y: plotRect.midY)

            if let markerY {
                Circle()
                    .stroke(Color.accentColor, lineWidth: 1.5)
                    .frame(width: 9, height: 9)
                    .position(x: highlightX, y: markerY)
            }

            readout
                .position(x: cardCenterX, y: cardCenterY)
        }
        .allowsHitTesting(false)
    }

    /// Card center X: one gap beside the hovered column — to its right
    /// when the cursor is in the plot's left half, to its left
    /// otherwise — then clamped so the card stays inside the plot.
    private var cardCenterX: CGFloat {
        let halfWidth = ChartHoverReadout.cardWidth / 2
        let gap: CGFloat = 12
        let onLeftHalf = highlightX < plotRect.midX
        let raw = onLeftHalf
            ? highlightX + gap + halfWidth
            : highlightX - gap - halfWidth
        return min(max(raw, plotRect.minX + halfWidth), plotRect.maxX - halfWidth)
    }

    /// Card center Y: level with the hovered point (or the plot
    /// middle for the bar chart), clamped to keep the card in-plot.
    /// The card's height isn't known here, so a generous half-height
    /// estimate is used for the clamp.
    private var cardCenterY: CGFloat {
        let anchor = markerY ?? plotRect.midY
        let halfHeight: CGFloat = 46
        return min(max(anchor, plotRect.minY + halfHeight), plotRect.maxY - halfHeight)
    }
}

/// A `(W + 0.5·D)/N`-style fraction rendered as a one-decimal percent.
private func arenaBreakdownPercent(_ fraction: Double) -> String {
    String(format: "%.1f%%", fraction * 100)
}

/// Hover-readout card for an `ArenaValueByPlyBucket`, shared by the
/// win-rate-by-ply, value-head-by-ply, and samples-per-ply charts so
/// all three surface the same per-bucket counts.
private func arenaPlyReadout(for bucket: ArenaValueByPlyBucket) -> ChartHoverReadout {
    var rows: [ChartHoverReadout.Row] = [
        ChartHoverReadout.Row(
            label: "Win rate",
            value: arenaBreakdownPercent(bucket.candidateScore)
        ),
        ChartHoverReadout.Row(
            label: "Value",
            value: arenaBreakdownSignedValue(bucket.mean)
        )
    ]
    // `meanPolicyProbability` is nil only for summaries persisted
    // before the field existed — show the row when present.
    if let policy = bucket.meanPolicyProbability {
        rows.append(ChartHoverReadout.Row(
            label: "Policy",
            value: arenaBreakdownPercent(Double(policy))
        ))
    }
    rows.append(ChartHoverReadout.Row(
        label: "W·D·L",
        value: "\(bucket.wins) · \(bucket.draws) · \(bucket.losses)"
    ))
    rows.append(ChartHoverReadout.Row(label: "Samples", value: "\(bucket.count)"))
    return ChartHoverReadout(
        title: "Ply \(bucket.lowerInclusive)–\(bucket.upperInclusive)",
        rows: rows
    )
}

/// Hover-readout card for an `ArenaValueByProgressBucket`, shared by
/// the win-rate-by-progress and value-head-by-progress charts.
private func arenaProgressReadout(for bucket: ArenaValueByProgressBucket) -> ChartHoverReadout {
    var rows: [ChartHoverReadout.Row] = [
        ChartHoverReadout.Row(
            label: "Win rate",
            value: arenaBreakdownPercent(bucket.candidateScore)
        ),
        ChartHoverReadout.Row(
            label: "Value",
            value: arenaBreakdownSignedValue(bucket.mean)
        )
    ]
    // `meanPolicyProbability` is nil only for summaries persisted
    // before the field existed — show the row when present.
    if let policy = bucket.meanPolicyProbability {
        rows.append(ChartHoverReadout.Row(
            label: "Policy",
            value: arenaBreakdownPercent(Double(policy))
        ))
    }
    rows.append(ChartHoverReadout.Row(
        label: "W·D·L",
        value: "\(bucket.wins) · \(bucket.draws) · \(bucket.losses)"
    ))
    rows.append(ChartHoverReadout.Row(label: "Samples", value: "\(bucket.count)"))
    return ChartHoverReadout(
        title: "\(bucket.lowerPercent)–\(bucket.upperPercent)% of game",
        rows: rows
    )
}

/// Signed three-decimal rendering of a value-head scalar in `[-1, +1]`.
private func arenaBreakdownSignedValue(_ value: Float) -> String {
    String(format: "%+.3f", Double(value))
}
