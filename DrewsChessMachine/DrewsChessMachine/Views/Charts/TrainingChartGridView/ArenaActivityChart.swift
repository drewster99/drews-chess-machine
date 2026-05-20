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

    /// Single source of truth for the visible window — used both for
    /// `chartXScale` and for the hover overlay's manual screen→data
    /// mapping. If these two drift, the hover crosshair lands on a
    /// different time than what the chart is showing.
    private var visibleXDomain: ClosedRange<Double> {
        let lo = max(0, scrollX)
        let hi = lo + max(0.001, context.visibleDomainSec)
        return lo...hi
    }

    /// Hovered arena ID. Snaps to the arena whose band midpoint is
    /// closest to the cursor when no band contains the cursor — arena
    /// durations (~20s) are tiny relative to the visible window
    /// (~1h), so strict interval-containment left the hover dead in
    /// ~95% of cursor positions.
    private var hoverArenaID: Int? {
        guard let t = hoveredSec, !events.isEmpty else { return nil }
        for e in events where t >= e.startElapsedSec && t <= e.endElapsedSec {
            return e.id
        }
        let nearest = events.min(by: {
            let m0 = ($0.startElapsedSec + $0.endElapsedSec) * 0.5
            let m1 = ($1.startElapsedSec + $1.endElapsedSec) * 0.5
            return abs(m0 - t) < abs(m1 - t)
        })
        return nearest?.id
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

    /// Full-height fill color for a completed arena, encoding the
    /// verdict categorically:
    ///   - promoted:        bright green
    ///   - kept, score≥0.5: medium gray (candidate competitive but
    ///                      didn't clear the promotion threshold)
    ///   - kept, score<0.5: muted red (candidate actively lost ground
    ///                      vs the champion — surfaces regressions
    ///                      that would otherwise blend into "kept")
    /// Brighter when hovered. Bars fill the full Y range (0..1) so
    /// every arena is at least one pixel tall even at full-history
    /// zoom-out, and color does the work of conveying the verdict
    /// (score-vs-time lives on the companion `ArenaWinChart`).
    private func barColor(for e: ArenaChartEvent, hoveredID: Int?) -> Color {
        let hovered = hoveredID == e.id
        if e.promoted {
            return Color.green.opacity(hovered ? 1.0 : 0.7)
        }
        if e.score < 0.5 {
            return Color.red.opacity(hovered ? 0.8 : 0.5)
        }
        return Color.gray.opacity(hovered ? 0.9 : 0.45)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(
                title: "Arena activity",
                value: headerText,
                titleHelp: AttributedString("""
                    Full-height duration bands — one per arena tournament — across the same X axis \
                    as the rest of the chart grid, so you can see exactly when training paused for \
                    arena play. Color encodes the verdict: green = candidate promoted; gray = kept \
                    (score ≥ 0.5, candidate competitive); red = kept but candidate lost ground \
                    (score < 0.5). A blue tint marks the in-progress arena. Score-vs-time trend \
                    is on the companion 'Arena win %' tile.
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
            .chartXScale(domain: visibleXDomain)
            .chartOverlay { proxy in
                ChartHoverOverlay(proxy: proxy, xDomain: visibleXDomain, hoveredSec: $hoveredSec)
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
        // Full-height verdict bars. Drawn in two passes so promotions
        // (green) paint AFTER every other arena and always sit on top
        // if any visual overlap occurs (rare — arenas are disjoint in
        // time — but cheap insurance against future stacking changes).
        ForEach(events.filter { !$0.promoted }) { e in
            RectangleMark(
                xStart: .value("Start", e.startElapsedSec),
                xEnd: .value("End", e.endElapsedSec),
                yStart: .value("Floor", 0.0),
                yEnd: .value("Top", 1.0)
            )
            .foregroundStyle(barColor(for: e, hoveredID: hoveredID))
        }
        ForEach(events.filter { $0.promoted }) { e in
            RectangleMark(
                xStart: .value("Start", e.startElapsedSec),
                xEnd: .value("End", e.endElapsedSec),
                yStart: .value("Floor", 0.0),
                yEnd: .value("Top", 1.0)
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
        if let t = hoveredSec {
            RuleMark(x: .value("Time", t))
                .foregroundStyle(Color.gray.opacity(0.5))
                .lineStyle(StrokeStyle(lineWidth: 1))
        }
    }
}
