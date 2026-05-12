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
    /// Current gradient-clip global L2 cap. Drawn as a dashed
    /// horizontal reference line on the gNorm tile so the reader can
    /// see at a glance whether the optimizer is being clipped — when
    /// the gNorm trace sits above this line, every weight's gradient
    /// is being scaled by `clipMax/gNorm`. Sourced from
    /// `TrainingParameters.shared.gradClipMaxNorm`; updates next
    /// re-render after the user changes it via the popover.
    let gradClipMaxNorm: Double
    /// Unified-memory total in GB, used by the combined memory tile
    /// to render the `App X · GPU Y / Total GB (pct%)` header. The
    /// `app` and `gpu` totals are both derived from
    /// `ProcessInfo.physicalMemory` (unified memory), so either one
    /// is sufficient — both are accepted only because the upstream
    /// plumbing predates the chart consolidation.
    let appMemoryTotalGB: Double?
    let gpuMemoryTotalGB: Double?
    /// Session-wide running max of rolling legal-mass. Drives the
    /// tiered Y-axis on the legal-mass tile (top = 0.5 / 0.75 / 1.0
    /// based on the highest value seen all session, not just in the
    /// visible window). Sourced from `ChartCoordinator`.
    let legalMassMaxAllTime: Double
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
        // 5 columns × 4 rows, filled row-major:
        //   Row 1: legal-mass, policy entropy, progress rate (small),
        //          CPU %, RAM (App + GPU on shared axes)
        //   Row 2: pLoss, non-negligible policy count, replay
        //          ratio, pwNorm, power / thermal
        //   Row 3: pLoss split, gNorm, ||v|| (velocity L2 norm),
        //          longest move prefix histogram, arena activity
        //   Row 4 (value head): W/D/L probabilities, vLoss, vMean,
        //          vAbs, (empty)
        LazyVGrid(columns: Self.columns, spacing: 1) {
            // Row 1
            LegalMassChart(
                buckets: frame.trainingBuckets,
                allTimeMax: legalMassMaxAllTime,
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
            CpuGpuChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            MemoryChart(
                buckets: frame.trainingBuckets,
                totalGB: appMemoryTotalGB ?? gpuMemoryTotalGB,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            // Row 2
            PolicyLossChart(
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
                title: "pwNorm (policy head weight L2 norm)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.policyHeadWeightNorm },
                unit: "",
                color: .indigo,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            PowerThermalChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            // Row 3
            PolicyLossSplitChart(
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
                context: context,
                referenceLine: gradClipMaxNorm,
                referenceLineLabel: String(format: "clip %.0f", gradClipMaxNorm)
            )
            MiniLineChart(
                title: "||v|| (velocity L2 norm)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.velocityNorm },
                unit: "",
                color: .purple,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            DiversityHistogramChart(bars: diversityHistogram)
            ArenaActivityChart(
                events: arenaEvents,
                activeArenaStartElapsed: activeArenaStartElapsed,
                lastTrainingElapsedSec: frame.lastTrainingElapsedSec,
                promoteThreshold: promoteThreshold,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            // Row 4 — value head (post-WDL switch)
            WDLProbabilityChart(
                buckets: frame.trainingBuckets,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            MiniLineChart(
                title: "vLoss (W/D/L categorical CE)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.valueLoss },
                unit: "",
                color: .cyan,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            MiniLineChart(
                title: "vMean (p_win − p_loss)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.valueMean },
                unit: "",
                color: .teal,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context,
                referenceLine: 0,
                referenceLineColor: Color.gray.opacity(0.4)
            )
            MiniLineChart(
                title: "vAbs |p_win − p_loss|",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.valueAbsMean },
                unit: "",
                color: .mint,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
            // 5th cell of the value-head row intentionally left empty.
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
