import Charts
import SwiftUI
import SwiftUIFastCharts

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
/// (`LowerContentView`) — the migrated tiles drive it through
/// `fastChartGroup.hoveredX`; the unmigrated Arena and Diversity
/// tiles still write to `hoveredSec` via SwiftUI Charts'
/// `chartOverlay`. The grid bridges the two so the crosshair
/// stays in sync.
struct TrainingChartGridView: View {
    let frame: DecimatedChartFrame
    let diversityHistogram: [DiversityHistogramBar]
    let arenaEvents: [ArenaChartEvent]
    let activeArenaStartElapsed: Double?
    let promoteThreshold: Double
    let replayRatioTarget: Double
    let gradClipMaxNorm: Double
    let appMemoryTotalGB: Double?
    let gpuMemoryTotalGB: Double?
    let legalMassMaxAllTime: Double
    let visibleDomainSec: Double
    @Binding var scrollX: Double
    @Binding var hoveredSec: Double?
    /// Shared hover state for the new path-based chart tiles. The
    /// grid's `.onChange` blocks below mirror this to/from
    /// `hoveredSec` so the still-on-Swift-Charts Arena and Diversity
    /// tiles share the crosshair sync.
    let fastChartGroup: FastChartGroup

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

    /// Visible-X domain in elapsed seconds, computed once per
    /// render — `[scrollX, scrollX + visibleDomainSec]`. Handed to
    /// every migrated `FastLineChart` tile. The path builder uses
    /// this as the rendered slice; widening the visible window or
    /// auto-following the latest sample re-renders every tile by
    /// changing this value.
    private var migratedXDomain: ClosedRange<Double> {
        let lo = max(0, scrollX)
        let hi = lo + max(0.001, visibleDomainSec)
        return lo...hi
    }

    private static let columns = Array(
        repeating: GridItem(.flexible(), spacing: 1),
        count: 5
    )

    var body: some View {
        let xDomain = migratedXDomain
        let bucketWidthSec = context.bucketWidthSec
        return LazyVGrid(columns: Self.columns, spacing: 1) {
            // Row 1
            LegalMassChart(
                buckets: frame.trainingBuckets,
                allTimeMax: legalMassMaxAllTime,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            EntropyChart(
                buckets: frame.trainingBuckets,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            SmallProgressRateChart(
                buckets: frame.progressRateBuckets,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            CpuGpuChart(
                buckets: frame.trainingBuckets,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            MemoryChart(
                buckets: frame.trainingBuckets,
                totalGB: appMemoryTotalGB ?? gpuMemoryTotalGB,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            // Row 2
            PolicyLossChart(
                buckets: frame.trainingBuckets,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            NonNegChart(
                buckets: frame.trainingBuckets,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            ReplayRatioChart(
                buckets: frame.trainingBuckets,
                target: replayRatioTarget,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            MiniLineChart(
                title: "pwNorm (policy head weight L2 norm)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.policyHeadWeightNorm },
                unit: "",
                color: .indigo,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec,
                titleHelp: AttributedString("""
                    L2 norm of the policy head's weights. Steady growth tracks learning; a sudden \
                    collapse can indicate the head is being driven toward a degenerate solution.
                    """)
            )
            PowerThermalChart(
                buckets: frame.trainingBuckets,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            // Row 3
            PolicyLossSplitChart(
                buckets: frame.trainingBuckets,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            MiniLineChart(
                title: "gNorm (gradient L2 norm)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.gradNorm },
                unit: "",
                color: .pink,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec,
                referenceLine: gradClipMaxNorm,
                referenceLineLabel: String(format: "clip %.0f", gradClipMaxNorm),
                titleHelp: AttributedString("""
                    Pre-clip global L2 norm of the gradient across all trainable parameters. Reported \
                    every step. Dashed red line is the gradient-clip ceiling — values above mean the \
                    optimizer is rescaling the step by clip / gNorm. Frequent clipping right after a \
                    promotion is common while the trainer absorbs the weight swap.
                    """)
            )
            MiniLineChart(
                title: "||v|| (velocity L2 norm)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.velocityNorm },
                unit: "",
                color: .purple,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec,
                titleHelp: AttributedString("""
                    L2 norm of the optimizer's velocity (momentum buffer) across all parameters. \
                    Slow drift or steady growth is normal; sudden drops or unbounded growth can \
                    indicate trainer instability.
                    """)
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
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec
            )
            .equatable()
            MiniLineChart(
                title: "vLoss (W/D/L categorical CE)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.valueLoss },
                unit: "",
                color: .cyan,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec,
                titleHelp: AttributedString("""
                    Categorical cross-entropy of the value head's W/D/L softmax against the game's \
                    one-hot result. Range is roughly [0, ln 3 ≈ 1.10]; values below ln 3 mean the \
                    head is doing better than uniform.
                    """)
            )
            MiniLineChart(
                title: "vMean (p_win − p_loss)",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.valueMean },
                unit: "",
                color: .teal,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec,
                referenceLine: 0,
                referenceLineColor: Color.gray.opacity(0.4),
                titleHelp: AttributedString("""
                    Batch mean of the derived value scalar v = p_win − p_loss, in [-1, +1]. \
                    Negative means the head leans "losing" on average across the batch; zero is \
                    the neutral expectation.
                    """)
            )
            MiniLineChart(
                title: "vAbs |p_win − p_loss|",
                buckets: frame.trainingBuckets,
                rangeAccessor: { $0.valueAbsMean },
                unit: "",
                color: .mint,
                group: fastChartGroup,
                xDomain: xDomain,
                bucketWidthSec: bucketWidthSec,
                titleHelp: AttributedString("""
                    Batch mean of |p_win − p_loss|. Higher = more confident value-head predictions \
                    on average; very low means "everything looks like a draw" — a classic \
                    value-head collapse symptom.
                    """)
            )
            ArenaWinChart(
                events: arenaEvents,
                promoteThreshold: promoteThreshold,
                hoveredSec: $hoveredSec,
                scrollX: $scrollX,
                context: context
            )
        }
        .background(Color(nsColor: .separatorColor))
        // No hover bridge: `chartCoordinator.hoveredSec` is now a
        // computed pass-through onto `fastChartGroup.hoveredX`, so
        // unmigrated Arena/Diversity tiles writing through the
        // `$hoveredSec` binding and migrated FastLineChart tiles
        // writing to `fastChartGroup.hoveredX` land on the same
        // Observable storage. Every hover invalidation now fans
        // out from a single source.
    }

    // MARK: - Public statics (consumed by ContentView's big chart)

    /// Maximum distance in seconds between the hovered cursor time
    /// and the nearest bucket / sample for the data point to count
    /// as "data at this hover time". Wants to be a bit more than the
    /// heartbeat interval — large enough that a hover anywhere inside
    /// the data range still matches the nearest sample even after
    /// decimation leaves gaps between sparsely-populated buckets,
    /// small enough to still reject hovers that land out past the
    /// data's last sample.
    nonisolated static let hoverMatchToleranceSec: Double = 7.5

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
