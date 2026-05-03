import Charts
import SwiftUI

/// Grid of compact training-metric charts. All charts share a
/// synchronized horizontal scroll position and a single hover
/// crosshair — mousing over any time-series chart highlights the
/// same elapsed second across all of them so you can read every
/// metric at that moment.
///
/// The grid itself owns no data: every per-chart subview is
/// constructed from `frame.trainingBuckets` (or
/// `frame.progressRateBuckets`) plus the static configuration the
/// chart needs. The hover position is owned by the parent
/// (`LowerContentView`) so it can drive both the grid and the
/// large progress-rate chart in one shared selection state.
struct TrainingChartGridView: View {
    let frame: DecimatedChartFrame
    /// Current divergence-ply histogram bars (one per bucket) from
    /// the self-play diversity tracker. `nil` or empty while Play-
    /// and-Train isn't running or before the first game finishes.
    let diversityHistogram: [DiversityHistogramBar]
    /// Completed arena tournaments for this session, in order. Each
    /// carries its elapsed start/end time plus score + promotion
    /// flag so the arena activity chart can show duration bands
    /// colored by outcome.
    let arenaEvents: [ArenaChartEvent]
    /// Elapsed-second mark when the currently-in-progress arena
    /// began, or `nil` if no arena is running. When non-nil, the
    /// arena activity chart renders a live band in a distinctive
    /// blue tint from this start up to the latest chart sample so
    /// an arena is visible ON the chart the whole time it runs,
    /// not just after it ends.
    let activeArenaStartElapsed: Double?
    /// Promotion threshold used by the arena chart's horizontal
    /// reference line. Matches `ContentView.tournamentPromoteThreshold`;
    /// pulled through as a parameter so the grid stays decoupled
    /// from ContentView's compile unit.
    let promoteThreshold: Double
    /// Target replay ratio (consumption / production). Drawn as a
    /// dashed horizontal reference line on the Replay ratio tile so
    /// the reader can see how far the auto-adjust controller is
    /// missing at a glance.
    let replayRatioTarget: Double
    /// Unified-memory total in GB, used by the App memory and GPU
    /// memory tiles to render "used / total (pct%)" headers.
    let appMemoryTotalGB: Double?
    let gpuMemoryTotalGB: Double?
    let visibleDomainSec: Double
    @Binding var scrollX: Double
    /// Shared hover selection across every time-series chart. Set
    /// by each chart's `chartOverlay` and read back by the others
    /// to draw a synchronized crosshair.
    @Binding var hoveredSec: Double?

    /// Derived shared context handed to every chart subview.
    private var context: Context {
        let last = frame.lastTrainingElapsedSec
            ?? frame.lastProgressRateElapsedSec
            ?? 0
        let end = max(last, visibleDomainSec)
        let bucketCount = max(frame.trainingBuckets.count, 1)
        let bucketWidth = visibleDomainSec / Double(bucketCount)
        return Context(
            timeSeriesXDomain: 0...end,
            visibleDomainSec: visibleDomainSec,
            bucketWidthSec: bucketWidth
        )
    }

    private static let columns = Array(
        repeating: GridItem(.flexible(), spacing: 1),
        count: 5
    )

    var body: some View {
        // 5 columns × 2 rows. Layout by column (left to right):
        //   Col 1: Loss Total, Replay Ratio
        //   Col 2: Policy Entropy, Non-Negligible Policy Count
        //   Col 3: Progress Rate, (empty — progress rate is tall)
        //   Col 4: CPU %, GPU %
        //   Col 5: App Memory, GPU RAM
        // But LazyVGrid fills row-major, so we order accordingly.
        LazyVGrid(columns: Self.columns, spacing: 1) {
            // Row 1
            PolicyLossSplitChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            EntropyChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            SmallProgressRateChart(
                buckets: frame.progressRateBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            MiniLineChart(
                title: "CPU",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.cpuPercent },
                unit: "%",
                color: .blue,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            MemoryChart(
                title: "App memory (RAM)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.appMemoryGB },
                totalGB: appMemoryTotalGB,
                color: .brown,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            // Row 2
            PolicyValueLossChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            NonNegChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            ReplayRatioChart(
                buckets: frame.trainingBuckets,
                target: replayRatioTarget,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            MiniLineChart(
                title: "GPU",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.gpuBusyPercent },
                unit: "%",
                color: .indigo,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            MemoryChart(
                title: "GPU memory (RAM)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.gpuMemoryGB },
                totalGB: gpuMemoryTotalGB,
                color: .teal,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            // Row 3
            LegalMassChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            MiniLineChart(
                title: "gNorm (gradient L2 norm)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.gradNorm },
                unit: "",
                color: .pink,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            DiversityHistogramChart(bars: diversityHistogram)
            PowerThermalChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            ArenaActivityChart(
                events: arenaEvents,
                activeArenaStartElapsed: activeArenaStartElapsed,
                lastTrainingElapsedSec: frame.lastTrainingElapsedSec,
                promoteThreshold: promoteThreshold,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
        }
        .background(Color(nsColor: .separatorColor))
    }

    // MARK: - Public statics (consumed by ContentView's big chart)

    /// Maximum distance in seconds between the hovered cursor time
    /// and the nearest bucket / sample for the data point to count
    /// as "data at this hover time". Original samples were emitted
    /// at 1 Hz; 1.5 s accommodates jitter while rejecting hovers
    /// past the data's last sample.
    nonisolated static let hoverMatchToleranceSec: Double = 1.5

    /// Convert a Double to a compact short label suitable for chart
    /// axis labels and inline headers.
    nonisolated static func compactLabel(_ value: Double) -> String {
        let abs = Swift.abs(value)
        if abs >= 1_000_000 {
            return String(format: "%.1fM", value / 1_000_000)
        } else if abs >= 1_000 {
            return String(format: "%.1fK", value / 1_000)
        } else if abs >= 100 {
            return String(format: "%.0f", value)
        } else if abs >= 10 {
            return String(format: "%.1f", value)
        } else if abs >= 1 {
            return String(format: "%.2f", value)
        } else if abs >= 0.01 {
            return String(format: "%.3f", value)
        } else if abs == 0 {
            return "0"
        } else {
            return String(format: "%.1e", value)
        }
    }

    /// Compact mm:ss / h:mm:ss formatter for chart hover headers
    /// and X-axis labels.
    nonisolated static func formatElapsedAxis(_ seconds: Double) -> String {
        let secs = max(0, Int(seconds.rounded()))
        let h = secs / 3600
        let m = (secs % 3600) / 60
        let s = secs % 60
        if secs < 60 {
            return String(format: "0:%02d", s)
        } else if secs < 3600 {
            return String(format: "%d:%02d", m, s)
        } else {
            return String(format: "%d:%02d:%02d", h, m, s)
        }
    }
}
