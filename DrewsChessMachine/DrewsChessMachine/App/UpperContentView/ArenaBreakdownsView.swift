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
///   2. **Score by ply.** Line chart over 20-ply candidate-to-move
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
///
/// Hovering any of the three charts surfaces a readout card with the
/// hovered bucket's underlying counts.
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
            Text("Win rate by ply (20-ply buckets, (W+0.5·D)/N)")
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

    /// Bucket label of the column currently under the cursor, or
    /// `nil` when not hovering. Drives the readout card.
    @State private var hoveredLabel: String?

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
        .chartOverlay { proxy in
            hoverOverlay(proxy: proxy)
        }
    }

    @ViewBuilder
    private func hoverOverlay(proxy: ChartProxy) -> some View {
        GeometryReader { geo in
            if let plotFrame = proxy.plotFrame {
                let plotRect = geo[plotFrame]
                ZStack {
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onContinuousHover { phase in
                            handleHover(phase, plotRect: plotRect, proxy: proxy)
                        }
                    if let lbl = hoveredLabel,
                       let bucket = buckets.first(where: { label(for: $0) == lbl }),
                       let xPos = proxy.position(forX: lbl) {
                        ChartHoverDecorations(
                            highlightX: xPos + plotRect.minX,
                            markerY: nil,
                            plotRect: plotRect,
                            readout: readout(for: bucket)
                        )
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
    }

    private func handleHover(_ phase: HoverPhase, plotRect: CGRect, proxy: ChartProxy) {
        switch phase {
        case .active(let location):
            guard location.x >= plotRect.minX, location.x <= plotRect.maxX else {
                hoveredLabel = nil
                return
            }
            var bestLabel: String?
            var bestDistance = CGFloat.greatestFiniteMagnitude
            for lbl in xOrder {
                guard let px = proxy.position(forX: lbl) else { continue }
                let distance = abs(px + plotRect.minX - location.x)
                if distance < bestDistance {
                    bestDistance = distance
                    bestLabel = lbl
                }
            }
            hoveredLabel = bestLabel
        case .ended:
            hoveredLabel = nil
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

    /// Index (into `buckets` / `points`) of the dot currently under
    /// the cursor, or `nil` when not hovering.
    @State private var hoveredIndex: Int?

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
        .winRateYAxis(scores: points.map(\.score))
        .chartXAxis {
            AxisMarks {
                AxisGridLine()
                AxisValueLabel()
                    .font(.system(size: 8))
            }
        }
        .chartOverlay { proxy in
            ZStack {
                referenceLine(at: 0.5, proxy: proxy)
                hoverOverlay(proxy: proxy)
            }
        }
    }

    @ViewBuilder
    private func hoverOverlay(proxy: ChartProxy) -> some View {
        GeometryReader { geo in
            if let plotFrame = proxy.plotFrame {
                let plotRect = geo[plotFrame]
                ZStack {
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onContinuousHover { phase in
                            handleHover(phase, plotRect: plotRect, proxy: proxy)
                        }
                    if let idx = hoveredIndex, idx >= 0, idx < points.count {
                        let p = points[idx]
                        let highlightX = (proxy.position(forX: p.midPly) ?? 0) + plotRect.minX
                        let markerY = proxy.position(forY: p.score).map { $0 + plotRect.minY }
                        ChartHoverDecorations(
                            highlightX: highlightX,
                            markerY: markerY,
                            plotRect: plotRect,
                            readout: readout(for: buckets[idx])
                        )
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
    }

    private func handleHover(_ phase: HoverPhase, plotRect: CGRect, proxy: ChartProxy) {
        switch phase {
        case .active(let location):
            guard location.x >= plotRect.minX, location.x <= plotRect.maxX else {
                hoveredIndex = nil
                return
            }
            var bestIndex: Int?
            var bestDistance = CGFloat.greatestFiniteMagnitude
            for (i, p) in points.enumerated() {
                guard let px = proxy.position(forX: p.midPly) else { continue }
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

    private func readout(for bucket: ArenaValueByPlyBucket) -> ChartHoverReadout {
        ChartHoverReadout(
            title: "Ply \(bucket.lowerInclusive)–\(bucket.upperInclusive)",
            rows: [
                ChartHoverReadout.Row(
                    label: "Win rate",
                    value: arenaBreakdownPercent(bucket.candidateScore)
                ),
                ChartHoverReadout.Row(
                    label: "Value",
                    value: arenaBreakdownSignedValue(bucket.mean)
                ),
                ChartHoverReadout.Row(
                    label: "W·D·L",
                    value: "\(bucket.wins) · \(bucket.draws) · \(bucket.losses)"
                ),
                ChartHoverReadout.Row(label: "Samples", value: "\(bucket.count)")
            ]
        )
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

    /// Index (into `buckets` / `points`) of the dot currently under
    /// the cursor, or `nil` when not hovering.
    @State private var hoveredIndex: Int?

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
        .winRateYAxis(scores: points.map(\.score))
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
            ZStack {
                referenceLine(at: 0.5, proxy: proxy)
                hoverOverlay(proxy: proxy)
            }
        }
    }

    @ViewBuilder
    private func hoverOverlay(proxy: ChartProxy) -> some View {
        GeometryReader { geo in
            if let plotFrame = proxy.plotFrame {
                let plotRect = geo[plotFrame]
                ZStack {
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onContinuousHover { phase in
                            handleHover(phase, plotRect: plotRect, proxy: proxy)
                        }
                    if let idx = hoveredIndex, idx >= 0, idx < points.count {
                        let p = points[idx]
                        let highlightX = (proxy.position(forX: p.midPct) ?? 0) + plotRect.minX
                        let markerY = proxy.position(forY: p.score).map { $0 + plotRect.minY }
                        ChartHoverDecorations(
                            highlightX: highlightX,
                            markerY: markerY,
                            plotRect: plotRect,
                            readout: readout(for: buckets[idx])
                        )
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
    }

    private func handleHover(_ phase: HoverPhase, plotRect: CGRect, proxy: ChartProxy) {
        switch phase {
        case .active(let location):
            guard location.x >= plotRect.minX, location.x <= plotRect.maxX else {
                hoveredIndex = nil
                return
            }
            var bestIndex: Int?
            var bestDistance = CGFloat.greatestFiniteMagnitude
            for (i, p) in points.enumerated() {
                guard let px = proxy.position(forX: p.midPct) else { continue }
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

    private func readout(for bucket: ArenaValueByProgressBucket) -> ChartHoverReadout {
        ChartHoverReadout(
            title: "\(bucket.lowerPercent)–\(bucket.upperPercent)% of game",
            rows: [
                ChartHoverReadout.Row(
                    label: "Win rate",
                    value: arenaBreakdownPercent(bucket.candidateScore)
                ),
                ChartHoverReadout.Row(
                    label: "Value",
                    value: arenaBreakdownSignedValue(bucket.mean)
                ),
                ChartHoverReadout.Row(
                    label: "W·D·L",
                    value: "\(bucket.wins) · \(bucket.draws) · \(bucket.losses)"
                ),
                ChartHoverReadout.Row(label: "Samples", value: "\(bucket.count)")
            ]
        )
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

// MARK: - Shared win-rate Y-axis

private extension View {
    /// Y-axis treatment shared by the score-by-ply and
    /// score-by-progress charts. Arena win rate hugs the 0.5 "even"
    /// mark tightly — a full-scale axis flattens it into a
    /// featureless band — so the axis instead pins to a narrow fixed
    /// window straddling 0.5, with one gridline + label per stride
    /// step. The window widens only far enough to contain any score
    /// that genuinely falls outside it, so a lopsided arena still
    /// fits while the common, near-even case stays high-resolution.
    func winRateYAxis(scores: [Double]) -> some View {
        let lower = min(0.47, scores.min() ?? 0.47)
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
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 5))
        .overlay {
            RoundedRectangle(cornerRadius: 5)
                .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 0.5)
        }
        .fixedSize()
    }
}

/// Hover decorations drawn over an arena breakdown chart: a thin
/// vertical rule at the hovered column, an optional ring on the
/// hovered data point, and the readout card pinned to whichever top
/// corner of the plot is opposite the cursor — so the card never
/// hides the thing being pointed at. Hit-testing is disabled so the
/// chart's transparent hover-capture layer keeps receiving events
/// through it.
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
                .padding(4)
                .frame(
                    width: plotRect.width,
                    height: plotRect.height,
                    alignment: highlightX < plotRect.midX ? .topTrailing : .topLeading
                )
                .offset(x: plotRect.minX, y: plotRect.minY)
        }
        .allowsHitTesting(false)
    }
}

/// A `(W + 0.5·D)/N`-style fraction rendered as a one-decimal percent.
private func arenaBreakdownPercent(_ fraction: Double) -> String {
    String(format: "%.1f%%", fraction * 100)
}

/// Signed three-decimal rendering of a value-head scalar in `[-1, +1]`.
private func arenaBreakdownSignedValue(_ value: Float) -> String {
    String(format: "%+.3f", Double(value))
}
