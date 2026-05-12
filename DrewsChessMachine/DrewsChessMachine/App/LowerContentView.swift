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
    /// Current gradient-clip global L2 cap, drawn as a dashed
    /// horizontal reference line on the gNorm tile. Same
    /// forwarding pattern as `replayRatioTarget`.
    let gradClipMaxNorm: Double
    /// Unified-memory total used by the App-memory and GPU-memory
    /// tiles to render `used / total (pct%)` headers. Derived
    /// from `UpperContentView`'s `memoryStatsSnap`; nil before the
    /// first sample lands.
    let appMemoryTotalGB: Double?
    let gpuMemoryTotalGB: Double?

    @Bindable var chartCoordinator: ChartCoordinator

    var body: some View {
        VStack(spacing: 0) {
            ChartZoomControlRow(coordinator: chartCoordinator)
            TrainingChartGridView(
                frame: chartCoordinator.decimatedFrame,
                diversityHistogram: chartCoordinator.currentDiversityHistogramBars,
                arenaEvents: chartCoordinator.arenaChartEvents,
                activeArenaStartElapsed: chartCoordinator.activeArenaStartElapsed,
                promoteThreshold: promoteThreshold,
                replayRatioTarget: replayRatioTarget,
                gradClipMaxNorm: gradClipMaxNorm,
                appMemoryTotalGB: appMemoryTotalGB,
                gpuMemoryTotalGB: gpuMemoryTotalGB,
                legalMassMaxAllTime: chartCoordinator.legalMassMaxAllTime,
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
            // bucket positions until the next heartbeat sample append.
            chartCoordinator.recomputeDecimatedFrame()
        }
    }

}
