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

    /// Hovered arena — the one whose duration band contains the
    /// crosshair time (or `nil` if the hover is between arenas).
    private var hoverArena: ArenaChartEvent? {
        guard let t = hoveredSec else { return nil }
        return events.first { t >= $0.startElapsedSec && t <= $0.endElapsedSec }
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
            ChartTileHeader(title: "Arena win %", value: headerText)
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
                // for kept. Larger when this arena is the hovered one
                // so the visual lookup matches the chart-grid's shared
                // crosshair selection.
                ForEach(events) { e in
                    PointMark(
                        x: .value("Time", e.endElapsedSec),
                        y: .value("Score", e.score)
                    )
                    .foregroundStyle(e.promoted ? Color.green : Color.gray)
                    .symbolSize(hoverArena?.id == e.id ? 80 : 30)
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
            .chartYScale(domain: 0...1.05)
            .chartYAxis {
                AxisMarks(position: .leading, values: [0, 0.25, 0.5, 0.75, 1.0]) { value in
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
