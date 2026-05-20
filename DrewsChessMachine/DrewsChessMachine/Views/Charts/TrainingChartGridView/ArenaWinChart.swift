import Charts
import SwiftUI

/// Arena win-percentage trend over time. One point per completed
/// arena (X = arena end time in session-elapsed seconds, Y = the
/// candidate's score in `[0, 1]` — fraction of games won with
/// draws counting 0.5). Points are connected by a line so the
/// trend reads at a glance; promoted arenas are highlighted in
/// green vs gray for kept-champion. A dashed orange rule shows the
/// configured promotion threshold so the eye can see how often the
/// candidate clears it.
///
/// Companion to `ArenaActivityChart` (same data source, same X
/// axis): the activity chart shows arenas as duration bands so the
/// reader can see WHEN training paused for arena play; this chart
/// shows the same scores as a connected trend so the reader can
/// see the strength trajectory at a glance.
struct ArenaWinChart: View {
    let events: [ArenaChartEvent]
    let promoteThreshold: Double
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    /// Hovered arena. Snaps to the arena whose `endElapsedSec` (where
    /// each point/line node is plotted) is closest to the cursor's
    /// data X. The earlier interval-containment rule
    /// (`t ∈ [start, end]`) only matched a few percent of cursor
    /// positions, because arena durations (~20s) are tiny relative to
    /// the visible window (~1h), so the header collapsed to "show the
    /// last arena" as a fallback almost everywhere.
    private var hoverArena: ArenaChartEvent? {
        guard let t = hoveredSec, !events.isEmpty else { return nil }
        if let inInterval = events.first(where: { t >= $0.startElapsedSec && t <= $0.endElapsedSec }) {
            return inInterval
        }
        return events.min(by: { abs($0.endElapsedSec - t) < abs($1.endElapsedSec - t) })
    }

    private var headerText: String {
        if let e = hoverArena {
            let verdict = e.promoted ? "PROMOTED" : "kept"
            return String(format: "#%d  %.2f  %@", e.id + 1, e.score, verdict)
        }
        if let last = events.last {
            return String(format: "#%d  %.2f", last.id + 1, last.score)
        }
        return "no arenas yet"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(
                title: "Arena win %",
                value: headerText,
                titleHelp: AttributedString("""
                    Candidate score from each arena tournament — fraction of games won, with draws \
                    counting 0.5 — plotted as a connected trend. Dashed orange line is the \
                    promotion threshold; dots are green when the arena promoted, gray when it \
                    was kept. Companion to the Arena activity tile.
                    """)
            )
            Chart {
                // Connecting line — shows the score trajectory across
                // arenas. Drawn first so the per-point colored markers
                // sit on top.
                ForEach(events) { e in
                    LineMark(
                        x: .value("Time", e.endElapsedSec),
                        y: .value("Score", e.score)
                    )
                    .foregroundStyle(Color.blue.opacity(0.7))
                    .interpolationMethod(.linear)
                }
                // Per-arena point markers — green for promoted, gray
                // for kept. Drawn in two passes so promoted dots paint
                // AFTER kept dots: at wide zoom-out levels where 100+
                // arenas cluster around the threshold line, gray dots
                // would otherwise occlude rare green ones and the
                // promotion events would visually disappear. With this
                // split the green ones always sit on top in any
                // overlapping pixel.
                //
                // Larger when this arena is the hovered one so the
                // visual lookup matches the chart-grid's shared
                // crosshair selection.
                ForEach(events.filter { !$0.promoted }) { e in
                    PointMark(
                        x: .value("Time", e.endElapsedSec),
                        y: .value("Score", e.score)
                    )
                    .foregroundStyle(Color.gray)
                    // Small dots so 100+ arenas don't visually
                    // merge into a continuous band at wide zoom;
                    // hover state still bumps the dot up to a
                    // readable size.
                    .symbolSize(hoverArena?.id == e.id ? 60 : 10)
                }
                ForEach(events.filter { $0.promoted }) { e in
                    PointMark(
                        x: .value("Time", e.endElapsedSec),
                        y: .value("Score", e.score)
                    )
                    .foregroundStyle(Color.green)
                    .symbolSize(hoverArena?.id == e.id ? 60 : 10)
                }
                // Promotion threshold reference line — dashed orange
                // so the reader can see at a glance how often the
                // candidate's score clears it.
                RuleMark(y: .value("Threshold", promoteThreshold))
                    .foregroundStyle(Color.orange.opacity(0.6))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                // Shared hover crosshair — same time-axis as the rest
                // of the chart grid so hovering this chart highlights
                // the matching second on every other chart.
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
            }
            .chartYScale(domain: 0.40...0.60)
            .chartYAxis {
                AxisMarks(position: .leading, values: [0.40, 0.45, 0.50, 0.55, 0.60]) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.2f", v))
                                .font(.system(size: 7))
                                .monospacedDigit()
                        }
                    }
                }
            }
            .modifier(StandardTimeSeriesChartModifiers(
                context: context,
                scrollX: $scrollX,
                hoveredSec: $hoveredSec
            ))
            .overlay {
                if events.isEmpty {
                    Text("No data to display")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        // Disable hit-testing so the chart's hover
                        // overlay still receives cursor events even
                        // while the empty-state label is showing.
                        .allowsHitTesting(false)
                }
            }
        }
        .frame(height: 75)
        .chartCard()
    }
}
