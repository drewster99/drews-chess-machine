import SwiftUI

/// Sibling of `UpperContentView` under `ContentView`. Renders the
/// chart layer: the zoom-control row plus the
/// `TrainingChartGridView`. Owns no `@State`; reads chart-grid
/// inputs (decimated frame, scroll position, hover, zoom index, etc.)
/// from the shared `ChartCoordinator` that `ContentView` constructs
/// and passes to both child views.
///
/// Carving the chart layer into its own struct view scopes its
/// SwiftUI body invalidation: `UpperContentView`'s frequent
/// re-evaluations (busy chip, alarm banner, training stats) no
/// longer touch the chart grid because they don't mutate any of
/// the `@Observable` properties this view reads.
struct LowerContentView: View {
    /// Promotion threshold drawn as a horizontal reference line on
    /// the arena-activity chart. Lives on `UpperContentView` (it
    /// is a tunable training parameter), forwarded here as a
    /// `let` so the chart grid stays decoupled from
    /// `TrainingParameters`.
    let promoteThreshold: Double
    /// Target replay ratio rendered as a dashed horizontal
    /// reference line on the replay-ratio tile. Same forwarding
    /// pattern as `promoteThreshold`.
    let replayRatioTarget: Double
    /// Unified-memory total used by the App-memory and GPU-memory
    /// tiles to render `used / total (pct%)` headers. Derived
    /// from `UpperContentView`'s `memoryStatsSnap`; nil before the
    /// first sample lands.
    let appMemoryTotalGB: Double?
    let gpuMemoryTotalGB: Double?

    @Bindable var chartCoordinator: ChartCoordinator

    var body: some View {
        VStack(spacing: 0) {
            chartZoomControlRow
            TrainingChartGridView(
                frame: chartCoordinator.decimatedFrame,
                diversityHistogram: chartCoordinator.currentDiversityHistogramBars,
                arenaEvents: chartCoordinator.arenaChartEvents,
                activeArenaStartElapsed: chartCoordinator.activeArenaStartElapsed,
                promoteThreshold: promoteThreshold,
                replayRatioTarget: replayRatioTarget,
                appMemoryTotalGB: appMemoryTotalGB,
                gpuMemoryTotalGB: gpuMemoryTotalGB,
                visibleDomainSec: ChartZoom.stops[chartCoordinator.chartZoomIdx],
                scrollX: $chartCoordinator.scrollX,
                hoveredSec: $chartCoordinator.hoveredSec
            )
        }
        .onChange(of: chartCoordinator.scrollX) { _, newValue in
            chartCoordinator.handleScrollChange(newValue)
        }
        .onChange(of: chartCoordinator.chartZoomIdx) { _, _ in
            // Visible window length changed — re-decimate so the
            // chart marks land on the new bucket grid. Without
            // this the chart would be stuck on the prior zoom's
            // bucket positions until the next 1 Hz sample append.
            chartCoordinator.recomputeDecimatedFrame()
        }
    }

    /// Compact row — same font size and weight for every element so
    /// it lays out as one tight cluster on the left rather than the
    /// bold-zoom + tiny-hint + far-right-Auto layout it used to be.
    @ViewBuilder
    private var chartZoomControlRow: some View {
        HStack(spacing: 8) {
            Text(ChartZoom.labels[chartCoordinator.chartZoomIdx])
                .font(.caption.bold())
                .monospacedDigit()
                .foregroundStyle(.secondary)
            Text("⌘=")
                .font(.caption)
                .foregroundStyle(chartCoordinator.canZoomIn ? Color.secondary : Color.secondary.opacity(0.4))
            Text("⌘-")
                .font(.caption)
                .foregroundStyle(chartCoordinator.canZoomOut ? Color.secondary : Color.secondary.opacity(0.4))
            Toggle(isOn: Binding(
                get: { chartCoordinator.chartZoomAuto },
                set: { chartCoordinator.setAutoZoom($0) }
            )) {
                Text("Auto")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .toggleStyle(.checkbox)
            Spacer()
        }
        .padding(.horizontal, 4)
    }
}
