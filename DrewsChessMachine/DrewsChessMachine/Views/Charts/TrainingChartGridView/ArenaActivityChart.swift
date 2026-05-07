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

    private var headerText: String {
        if let start = activeArenaStartElapsed, let now = liveNow {
            let elapsed = max(0, now - start)
            let durMin = Int(elapsed) / 60
            let durSec = Int(elapsed) % 60
            return String(format: "ARENA RUNNING  %d:%02d", durMin, durSec)
        } else if let id = hoverArenaID,
                  let e = events.first(where: { $0.id == id }) {
            let verdict = e.promoted ? "PROMOTED" : "kept"
            let durMin = Int(e.endElapsedSec - e.startElapsedSec) / 60
            let durSec = Int(e.endElapsedSec - e.startElapsedSec) % 60
            return String(
                format: "#%d  %@  %.2f  %d:%02d",
                e.id + 1, verdict, e.score, durMin, durSec
            )
        } else if let last = events.last {
            let verdict = last.promoted ? "PROMOTED" : "kept"
            return String(format: "%d ran · last %@ %.2f", events.count, verdict, last.score)
        } else {
            return "no arenas yet"
        }
    }

    /// True when the chart has nothing meaningful to render — no
    /// completed arenas and no live arena in progress. Surfaces a
    /// "No data to display" overlay so the empty state reads as
    /// "nothing yet" rather than "the threshold line is the data".
    private var isEmpty: Bool {
        events.isEmpty && activeArenaStartElapsed == nil
    }

    var body: some View {
        let hoveredID = hoverArenaID
        let nowMark = liveNow
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "Arena activity", value: headerText)
            Chart {
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
                    .foregroundStyle(
                        e.promoted
                            ? Color.green.opacity(hoveredID == e.id ? 1.0 : 0.7)
                            : Color.gray.opacity(hoveredID == e.id ? 1.0 : 0.5)
                    )
                }
                if let start = activeArenaStartElapsed, let now = nowMark {
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
            .chartXScale(domain: context.timeSeriesXDomain)
            .chartScrollableAxes(.horizontal)
            .chartXVisibleDomain(length: context.visibleDomainSec)
            .chartScrollPosition(x: $scrollX)
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
}
