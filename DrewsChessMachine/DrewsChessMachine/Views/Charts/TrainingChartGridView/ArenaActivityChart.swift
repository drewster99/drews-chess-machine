import Charts
import SwiftUI

/// Arena activity chart — one band per completed arena, plus a
/// live band for the in-progress arena (if any).
struct ArenaActivityChart: View {
    let events: [ArenaChartEvent]
    let activeArenaStartElapsed: Double?
    /// Latest training-sample elapsed time. Used to draw the live
    /// arena band's "now" edge.
    let lastTrainingElapsedSec: Double?
    let promoteThreshold: Double
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    private var liveNow: Double? {
        guard let start = activeArenaStartElapsed else { return nil }
        return max(start, lastTrainingElapsedSec ?? start)
    }

    private var hoverArenaID: Int? {
        guard let t = hoveredSec else { return nil }
        for e in events where t >= e.startElapsedSec && t <= e.endElapsedSec {
            return e.id
        }
        return nil
    }

    /// "M:SS" from a duration in seconds.
    private static func mmss(_ seconds: Double) -> String {
        let total = Int(max(0, seconds))
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    private func hoveredArenaHeader(_ e: ArenaChartEvent) -> String {
        let verdict = e.promoted ? "PROMOTED" : "kept"
        let dur = Self.mmss(e.endElapsedSec - e.startElapsedSec)
        let scoreStr = String(format: "%.2f", e.score)
        return "#\(e.id + 1)  \(verdict)  \(scoreStr)  \(dur)"
    }

    private var headerText: String {
        if let start = activeArenaStartElapsed, let now = liveNow {
            return "ARENA RUNNING  \(Self.mmss(now - start))"
        }
        if let id = hoverArenaID, let e = events.first(where: { $0.id == id }) {
            return hoveredArenaHeader(e)
        }
        if let last = events.last {
            let verdict = last.promoted ? "PROMOTED" : "kept"
            let scoreStr = String(format: "%.2f", last.score)
            return "\(events.count) ran · last \(verdict) \(scoreStr)"
        }
        return "no arenas yet"
    }

    /// True when the chart has nothing meaningful to render — no
    /// completed arenas and no live arena in progress. Surfaces a
    /// "No data to display" overlay so the empty state reads as
    /// "nothing yet" rather than "the threshold line is the data".
    private var isEmpty: Bool {
        events.isEmpty && activeArenaStartElapsed == nil
    }

    /// Score-bar fill color for a completed arena (brighter when hovered).
    private func barColor(for e: ArenaChartEvent, hoveredID: Int?) -> Color {
        let hovered = hoveredID == e.id
        if e.promoted {
            return Color.green.opacity(hovered ? 1.0 : 0.7)
        }
        return Color.gray.opacity(hovered ? 1.0 : 0.5)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(
                title: "Arena activity",
                value: headerText,
                titleHelp: AttributedString("""
                    Duration bands — one per arena tournament — across the same X axis as the rest \
                    of the chart grid, so you can see exactly when training paused for arena play. \
                    Band color is green when the candidate promoted, gray when the champion was \
                    kept; band height is the candidate's score. Dashed orange line is the \
                    promotion threshold; a blue tint marks the in-progress arena.
                    """)
            )
            Chart {
                chartContent
            }
            .chartYScale(domain: 0...1.05)
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
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
            // Visible-window-only X domain — matches the migrated
            // tiles, so the layout doesn't jump every time a new
            // training sample bumps `lastElapsed` upward.
            .chartXScale(domain:
                max(0, scrollX)...(max(0, scrollX) + max(0.001, context.visibleDomainSec))
            )
            .chartOverlay { proxy in
                ChartHoverOverlay(proxy: proxy, hoveredSec: $hoveredSec)
            }
            .overlay {
                if isEmpty {
                    Text("No data to display")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        // The chart's own `.chartOverlay` provides
                        // hover-detection via `.onContinuousHover`;
                        // a hit-testing-enabled Text on top would
                        // shadow it. Disabling hit-testing keeps the
                        // hover gestures reaching the chart even
                        // though there's nothing meaningful to hover
                        // over while empty.
                        .allowsHitTesting(false)
                }
            }
            .frame(height: 60)
        }
        .frame(height: 75)
        .chartCard()
    }

    @ChartContentBuilder
    private var chartContent: some ChartContent {
        let hoveredID = hoverArenaID
        ForEach(events) { e in
            RectangleMark(
                xStart: .value("Start", e.startElapsedSec),
                xEnd: .value("End", e.endElapsedSec),
                yStart: .value("Floor", 0.0),
                yEnd: .value("Top", 1.0)
            )
            .foregroundStyle(Color.secondary.opacity(hoveredID == e.id ? 0.25 : 0.12))
        }
        ForEach(events) { e in
            RectangleMark(
                xStart: .value("Start", e.startElapsedSec),
                xEnd: .value("End", e.endElapsedSec),
                yStart: .value("Floor", 0.0),
                yEnd: .value("Score", e.score)
            )
            .foregroundStyle(barColor(for: e, hoveredID: hoveredID))
        }
        if let start = activeArenaStartElapsed, let now = liveNow {
            RectangleMark(
                xStart: .value("Start", start),
                xEnd: .value("Now", now),
                yStart: .value("Floor", 0.0),
                yEnd: .value("Top", 1.0)
            )
            .foregroundStyle(Color.blue.opacity(0.35))
        }
        RuleMark(y: .value("Threshold", promoteThreshold))
            .foregroundStyle(Color.orange.opacity(0.6))
            .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
        if let t = hoveredSec {
            RuleMark(x: .value("Time", t))
                .foregroundStyle(Color.gray.opacity(0.5))
                .lineStyle(StrokeStyle(lineWidth: 1))
        }
    }
}
