import Charts
import SwiftUI

/// Per-second sample of training metrics for the chart grid.
/// Populated by the heartbeat alongside `ProgressRateSample`.
struct TrainingChartSample: Identifiable, Sendable {
    let id: Int
    let elapsedSec: Double

    let rollingPolicyLoss: Double?
    let rollingValueLoss: Double?
    let rollingPolicyEntropy: Double?
    let rollingPolicyNonNegCount: Double?
    let rollingPolicyNonNegIllegalCount: Double?
    let rollingGradNorm: Double?
    let replayRatio: Double?
    /// Outcome-partitioned policy loss — mean over batch positions
    /// where outcome z > +0.5 (winning game) / z < -0.5 (losing game).
    /// Splitting the conventional `rollingPolicyLoss` by outcome makes
    /// the curve unambiguous; rendered together on the upper-left
    /// chart instead of total loss.
    let rollingPolicyLossWin: Double?
    let rollingPolicyLossLoss: Double?
    /// Legal-masked Shannon entropy (in nats) over the legal-only
    /// renormalized policy softmax. Distinct from `rollingPolicyEntropy`
    /// (which is over the full 4864-dim head): a high value means
    /// "diffuse across legal moves" while the full-head pEnt can be
    /// high while concentrating on illegals. Charted on the same
    /// tile as `rollingPolicyEntropy` so the two trajectories can
    /// be compared directly.
    let rollingLegalEntropy: Double?
    /// Sum of softmax probability mass that lands on legal moves at
    /// the probed position. In `[0, 1]` — the complement is mass on
    /// illegal moves. Pulled from the periodic `LegalMassSnapshot`
    /// probe (same source as `rollingLegalEntropy`).
    let rollingLegalMass: Double?

    // System metrics
    let cpuPercent: Double?
    let gpuBusyPercent: Double?
    let gpuMemoryMB: Double?
    let appMemoryMB: Double?

    /// Whether macOS Low Power Mode was on at sample time. Charted
    /// as 0 (off) / 1 (on) — a step trace that sits along the
    /// bottom and pops up to 1 only while the user (or the system
    /// automatically on battery) has enabled the mode.
    let lowPowerMode: Bool?
    /// `ProcessInfo.ThermalState` at sample time. Charted as the
    /// raw-value offset by +2 (so nominal=2, fair=3, serious=4,
    /// critical=5), keeping the line strictly above the low-power
    /// step trace at 0/1 so they never overlap. Hover resolves the
    /// offset back to the named thermal state.
    let thermalState: ProcessInfo.ThermalState?

    var rollingTotalLoss: Double? {
        guard let p = rollingPolicyLoss, let v = rollingValueLoss else { return nil }
        return p + v
    }

    var appMemoryGB: Double? {
        appMemoryMB.map { $0 / 1024.0 }
    }

    var gpuMemoryGB: Double? {
        gpuMemoryMB.map { $0 / 1024.0 }
    }
}

/// Bar slice in the diversity histogram. Identifiable so SwiftUI's
/// `Chart` can key BarMarks correctly as counts update; Equatable so
/// `[DiversityHistogramBar]` parameters on chart views compare cheaply
/// during SwiftUI's view-diff (lets `LowerContentView` skip body
/// re-eval when the histogram hasn't changed).
struct DiversityHistogramBar: Identifiable, Sendable, Equatable {
    let id: Int          // bucket index (stable across updates)
    let label: String    // bucket label like "0-2", "41+"
    let count: Int
}

/// One completed arena tournament, positioned on the chart grid's
/// shared elapsed-time X axis. The `startElapsedSec` / `endElapsedSec`
/// pair lets us render arenas as duration bands rather than point
/// events, so the reader can see WHEN training was paused for
/// arena play and for HOW LONG.
struct ArenaChartEvent: Identifiable, Sendable, Equatable {
    let id: Int
    /// Session-elapsed seconds when the arena began.
    let startElapsedSec: Double
    /// Session-elapsed seconds when the arena ended (may extend
    /// past the visible chart window for very recent arenas).
    let endElapsedSec: Double
    /// Candidate score in `[0, 1]` — fraction of games the
    /// candidate won (draws count 0.5).
    let score: Double
    /// Whether the candidate was promoted to champion. Drives the
    /// bar color (green) vs kept-champion (gray) and the promotion
    /// marker.
    let promoted: Bool
}

/// Shared chart-rendering inputs forwarded from `LowerContentView`
/// to every per-chart subview. Bundles the X-axis configuration so
/// each chart subview's parameter list stays compact, and so a
/// future change to (for example) the visible-domain semantics
/// only has to touch one type definition.
struct ChartGridContext: Equatable {
    /// Full-data X domain used by `chartXScale(domain:)`. Computed
    /// in `LowerContentView` as `0...max(lastElapsed, visibleDomainSec)`.
    let timeSeriesXDomain: ClosedRange<Double>
    /// Length in seconds of the visible scroll window. Passed
    /// directly to `chartXVisibleDomain(length:)`.
    let visibleDomainSec: Double
    /// Width of one decimation bucket in seconds. Used by hover-
    /// readout helpers to size the "is the cursor near a bucket"
    /// tolerance — at very wide zoom levels, the per-sample 1.5 s
    /// tolerance would cause every hover to read as no-data.
    let bucketWidthSec: Double
}

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
    private var context: ChartGridContext {
        let last = frame.lastTrainingElapsedSec
            ?? frame.lastProgressRateElapsedSec
            ?? 0
        let end = max(last, visibleDomainSec)
        let bucketCount = max(frame.trainingBuckets.count, 1)
        let bucketWidth = visibleDomainSec / Double(bucketCount)
        return ChartGridContext(
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

// MARK: - Card wrapper

/// Background card chrome shared by every grid tile. Pulled out as a
/// free function so chart subviews can apply it without depending on
/// the parent `TrainingChartGridView`.
@ViewBuilder
fileprivate func chartCard<Content: View>(
    @ViewBuilder content: () -> Content
) -> some View {
    content()
        .padding(6)
        .background(Color(nsColor: .controlBackgroundColor))
}

// MARK: - Hover overlay helper

/// Three-way result of a per-chart hover query.
fileprivate enum HoverReadout {
    /// The cursor isn't over any time-series chart right now.
    case notHovering
    /// The cursor IS over a chart, but no bucket is within tolerance
    /// (or the bucket has no value for this series). Carries the
    /// raw hovered time so the header can still display
    /// "t=M:SS — no data".
    case hoveringNoData(hoveredTime: Double)
    /// The cursor is over a chart and the nearest bucket has a
    /// value for this series. Carries the bucket's anchor time and
    /// the bucket's representative value (range max).
    case hoveringWithData(sampleTime: Double, value: Double)
}

/// Transparent overlay that captures mouse position on a time-series
/// chart and pipes the converted elapsed-second into the shared
/// `hoveredSec` binding. Pulled out of every chart's body so each
/// chart's hover wiring is one line.
fileprivate struct ChartHoverOverlay: View {
    let proxy: ChartProxy
    @Binding var hoveredSec: Double?

    var body: some View {
        GeometryReader { geo in
            Rectangle()
                .fill(Color.clear)
                .contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let point):
                        let origin = (proxy.plotFrame.map { geo[$0].origin } ?? .zero)
                        let xInPlot = point.x - origin.x
                        if let sec: Double = proxy.value(atX: xInPlot) {
                            if sec < 0 {
                                if hoveredSec != nil { hoveredSec = nil }
                                return
                            }
                            if hoveredSec != sec {
                                hoveredSec = sec
                            }
                        }
                    case .ended:
                        if hoveredSec != nil {
                            hoveredSec = nil
                        }
                    }
                }
        }
    }
}

// MARK: - Per-chart hover lookup helpers

/// Linear-scan nearest-bucket lookup across the visible-window
/// bucket array. Returns `nil` if the array is empty OR if the
/// nearest bucket's anchor is farther than `tolerance` seconds
/// from `t`. Tolerance is sized by the caller — typically
/// `max(hoverMatchToleranceSec, 1.5 * bucketWidthSec)` so wide
/// zoom levels still register hover correctly.
fileprivate func nearestTrainingBucket(
    at t: Double,
    in buckets: [TrainingBucket],
    tolerance: Double
) -> TrainingBucket? {
    guard !buckets.isEmpty else { return nil }
    var best = buckets[0]
    var bestDist = Swift.abs(best.elapsedSec - t)
    for b in buckets.dropFirst() {
        let d = Swift.abs(b.elapsedSec - t)
        if d < bestDist { best = b; bestDist = d }
    }
    return bestDist <= tolerance ? best : nil
}

fileprivate func nearestProgressBucket(
    at t: Double,
    in buckets: [ProgressRateBucket],
    tolerance: Double
) -> ProgressRateBucket? {
    guard !buckets.isEmpty else { return nil }
    var best = buckets[0]
    var bestDist = Swift.abs(best.elapsedSec - t)
    for b in buckets.dropFirst() {
        let d = Swift.abs(b.elapsedSec - t)
        if d < bestDist { best = b; bestDist = d }
    }
    return bestDist <= tolerance ? best : nil
}

/// Resolve the hover state for one numeric series on a chart. The
/// per-chart subview reads the relevant `ChartBucketRange?` field
/// off the nearest bucket and converts it into the three-way
/// `HoverReadout`. We use the bucket range's `max` as the
/// representative value (preserves spike visibility — same logic
/// the chart's line marks use).
fileprivate func hoverReadoutTraining(
    hoveredSec: Double?,
    buckets: [TrainingBucket],
    accessor: (TrainingBucket) -> ChartBucketRange?,
    bucketWidthSec: Double
) -> HoverReadout {
    guard let t = hoveredSec else { return .notHovering }
    let tolerance = Swift.max(
        TrainingChartGridView.hoverMatchToleranceSec,
        bucketWidthSec * 1.5
    )
    guard let bucket = nearestTrainingBucket(
        at: t, in: buckets, tolerance: tolerance
    ) else {
        return .hoveringNoData(hoveredTime: t)
    }
    guard let range = accessor(bucket) else {
        return .hoveringNoData(hoveredTime: t)
    }
    return .hoveringWithData(sampleTime: bucket.elapsedSec, value: range.max)
}

// MARK: - Per-chart subviews

/// Upper-left tile (replaces the legacy "Loss (pLoss + vLoss)"
/// total-loss sparkgraph). Plots two outcome-partitioned series:
/// the policy loss restricted to win-outcome batch positions
/// (`pLossWin`) and to loss-outcome positions (`pLossLoss`).
/// The total-loss curve is ambiguous because outcome-weighted CE
/// flips sign with z; splitting by outcome makes both lines
/// individually interpretable.
fileprivate struct PolicyLossSplitChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    var body: some View {
        let lastWin = buckets.last?.policyLossWin?.max
        let lastLoss = buckets.last?.policyLossLoss?.max
        let headerText: String
        switch (lastWin, lastLoss) {
        case (let w?, let l?):
            headerText = String(format: "win %+.4f / loss %+.4f", w, l)
        case (let w?, nil):
            headerText = String(format: "win %+.4f / loss --", w)
        case (nil, let l?):
            headerText = String(format: "win -- / loss %+.4f", l)
        default:
            headerText = "--"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: "pLoss split (W vs L)", value: headerText)
                Chart {
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("pLoss", b.policyLossWin?.max ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pLossWin"))
                    }
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("pLoss", b.policyLossLoss?.max ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pLossLoss"))
                    }
                    RuleMark(y: .value("Zero", 0.0))
                        .foregroundStyle(Color.gray.opacity(0.4))
                        .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                }
                .chartForegroundStyleScale([
                    "pLossWin": Color.green,
                    "pLossLoss": Color.red
                ])
                .modifier(StandardTimeSeriesChartModifiers(
                    context: context,
                    scrollX: $scrollX,
                    hoveredSec: $hoveredSec
                ))
            }
            .frame(height: 75)
        }
    }
}

/// Combo chart showing pLoss (policy loss, orange) and vLoss
/// (value loss, cyan) on a shared Y axis.
fileprivate struct PolicyValueLossChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    var body: some View {
        let lastP = buckets.last?.policyLoss?.max
        let lastV = buckets.last?.valueLoss?.max
        let headerText: String
        switch (lastP, lastV) {
        case (let p?, let v?):
            headerText = String(format: "pLoss %.3f / vLoss %.3f", p, v)
        case (let p?, nil):
            headerText = String(format: "pLoss %.3f / vLoss --", p)
        case (nil, let v?):
            headerText = String(format: "pLoss -- / vLoss %.3f", v)
        default:
            headerText = "--"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: "pLoss + vLoss", value: headerText)
                Chart {
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("Loss", b.policyLoss?.max ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pLoss"))
                    }
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("Loss", b.valueLoss?.max ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "vLoss"))
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                }
                .chartForegroundStyleScale([
                    "pLoss": Color.orange,
                    "vLoss": Color.cyan
                ])
                .modifier(StandardTimeSeriesChartModifiers(
                    context: context,
                    scrollX: $scrollX,
                    hoveredSec: $hoveredSec
                ))
            }
            .frame(height: 75)
        }
    }
}

/// Policy entropy chart with an extra `pEntLegal` series.
fileprivate struct EntropyChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    nonisolated static let maxEntropy = log(Double(ChessNetwork.policySize))
    /// Reference uniform-distribution entropy for the legal-only
    /// renormalized softmax. A typical chess position has ~30 legal
    /// moves, so log(30) ≈ 3.40 nats is a reasonable "fully diffuse
    /// over legal" baseline.
    nonisolated static let maxLegalEntropy = log(30.0)

    var body: some View {
        let readout = hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.policyEntropy },
            bucketWidthSec: context.bucketWidthSec
        )
        let latestPEnt = buckets.last?.policyEntropy?.max
        let latestPEntLegal = buckets.last?.legalEntropy?.max
        let headerText = entropyHeader(
            readout: readout,
            latestPEnt: latestPEnt,
            latestPEntLegal: latestPEntLegal
        )

        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: "Policy entropy", value: headerText)
                Chart {
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("Entropy", b.policyEntropy?.max ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pEnt"))
                    }
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("Entropy", b.legalEntropy?.max ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pEntLegal"))
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    if case .hoveringWithData(let t, let v) = readout {
                        PointMark(x: .value("Time", t), y: .value("Entropy", v))
                            .foregroundStyle(.purple)
                            .symbolSize(40)
                    }
                }
                .chartForegroundStyleScale([
                    "pEnt": Color.purple,
                    "pEntLegal": Color.green
                ])
                .modifier(StandardTimeSeriesChartModifiers(
                    context: context,
                    scrollX: $scrollX,
                    hoveredSec: $hoveredSec
                ))
            }
            .frame(height: 75)
        }
    }

    private func entropyHeader(
        readout: HoverReadout,
        latestPEnt: Double?,
        latestPEntLegal: Double?
    ) -> String {
        func formatEntropy(_ v: Double?) -> String {
            guard let v else { return "--" }
            return String(format: "%.3f", v)
        }
        switch readout {
        case .notHovering:
            let pEntStr: String
            if let v = latestPEnt {
                pEntStr = String(format: "%.3f (%.1f%%)", v, v / Self.maxEntropy * 100)
            } else {
                pEntStr = "--"
            }
            let pEntLegalStr: String
            if let v = latestPEntLegal {
                pEntLegalStr = String(format: "%.3f (%.1f%%)", v, v / Self.maxLegalEntropy * 100)
            } else {
                pEntLegalStr = "--"
            }
            return "pEnt \(pEntStr) / pEntLegal \(pEntLegalStr)"
        case .hoveringNoData(let t):
            return "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            let pEntStr = String(format: "%.3f (%.1f%%)", v, v / Self.maxEntropy * 100)
            return "t=\(TrainingChartGridView.formatElapsedAxis(t)) pEnt \(pEntStr) / pEntLegal \(formatEntropy(latestPEntLegal))"
        }
    }
}

/// Legal mass sum tile — fraction of softmax mass on legal moves
/// at the probed position. Y axis is fixed to 0…1.
fileprivate struct LegalMassChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    var body: some View {
        let readout = hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.legalMass },
            bucketWidthSec: context.bucketWidthSec
        )
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = buckets.last?.legalMass?.max {
                headerText = String(format: "%.4f%%", v * 100)
            } else {
                headerText = "--"
            }
        case .hoveringNoData(let t):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            headerText = String(
                format: "t=%@ %.4f%%",
                TrainingChartGridView.formatElapsedAxis(t), v * 100
            )
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: "Legal mass sum", value: headerText)
                Chart {
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("Legal mass", b.legalMass?.max ?? .nan)
                        )
                        .foregroundStyle(.cyan)
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    if case .hoveringWithData(let t, let v) = readout {
                        PointMark(x: .value("Time", t), y: .value("Legal mass", v))
                            .foregroundStyle(.cyan)
                            .symbolSize(40)
                    }
                }
                .chartYScale(domain: 0...1)
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
            }
            .frame(height: 75)
        }
    }
}

/// Above-uniform policy count chart — legal vs illegal counts on
/// a fixed `0...policySize` Y axis.
fileprivate struct NonNegChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    var body: some View {
        let legalReadout = hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.policyNonNegCount },
            bucketWidthSec: context.bucketWidthSec
        )
        let policyMax = ChessNetwork.policySize
        let headerText: String
        switch legalReadout {
        case .notHovering:
            let lastLegal = buckets.last?.policyNonNegCount?.max
            let lastIllegal = buckets.last?.policyNonNegIllegalCount?.max
            let legalStr = lastLegal.map { String(Int($0)) } ?? "--"
            let illegalStr = lastIllegal.map { String(Int($0)) } ?? "--"
            headerText = "legal \(legalStr) • illegal \(illegalStr)"
        case .hoveringNoData(let t):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            // Pull the matching illegal value from the same hovered bucket.
            let tolerance = Swift.max(
                TrainingChartGridView.hoverMatchToleranceSec,
                context.bucketWidthSec * 1.5
            )
            let illegalAtHover = nearestTrainingBucket(
                at: t, in: buckets, tolerance: tolerance
            )?.policyNonNegIllegalCount?.max
            let illegalStr = illegalAtHover.map { String(Int($0)) } ?? "--"
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t))  legal \(Int(v)) • illegal \(illegalStr)"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: "Above-uniform policy count", value: headerText)
                Chart {
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("Count", b.policyNonNegCount?.max ?? .nan),
                            series: .value("Series", "Legal")
                        )
                        .foregroundStyle(by: .value("Series", "Legal"))
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("Count", b.policyNonNegIllegalCount?.max ?? .nan),
                            series: .value("Series", "Illegal")
                        )
                        .foregroundStyle(by: .value("Series", "Illegal"))
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    if case .hoveringWithData(let t, let v) = legalReadout {
                        PointMark(x: .value("Time", t), y: .value("Count", v))
                            .foregroundStyle(.mint)
                            .symbolSize(40)
                    }
                }
                .chartForegroundStyleScale([
                    "Legal": Color.mint,
                    "Illegal": Color.red
                ])
                .chartLegend(position: .bottom, alignment: .leading, spacing: 4)
                .chartYScale(domain: 0...Double(policyMax))
                .modifier(StandardTimeSeriesChartModifiers(
                    context: context,
                    scrollX: $scrollX,
                    hoveredSec: $hoveredSec
                ))
            }
            .frame(height: 75)
        }
    }
}

/// Replay ratio tile — line + dashed reference at the user target.
fileprivate struct ReplayRatioChart: View {
    let buckets: [TrainingBucket]
    let target: Double
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    var body: some View {
        let readout = hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.replayRatio },
            bucketWidthSec: context.bucketWidthSec
        )
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = buckets.last?.replayRatio?.max {
                headerText = String(format: "%.2f (target %.2f)", v, target)
            } else {
                headerText = String(format: "-- (target %.2f)", target)
            }
        case .hoveringNoData(let t):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            headerText = String(
                format: "t=%@ %.2f (target %.2f)",
                TrainingChartGridView.formatElapsedAxis(t), v, target
            )
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: "Replay ratio", value: headerText)
                Chart {
                    RuleMark(y: .value("Target", target))
                        .foregroundStyle(Color.red.opacity(0.6))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value("Replay ratio", b.replayRatio?.max ?? .nan)
                        )
                        .foregroundStyle(.green)
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    if case .hoveringWithData(let t, let v) = readout {
                        PointMark(x: .value("Time", t), y: .value("Replay ratio", v))
                            .foregroundStyle(.green)
                            .symbolSize(40)
                    }
                }
                .modifier(StandardTimeSeriesChartModifiers(
                    context: context,
                    scrollX: $scrollX,
                    hoveredSec: $hoveredSec
                ))
            }
            .frame(height: 75)
        }
    }
}

/// Generic single-series mini chart used by CPU%, GPU%, and gNorm.
fileprivate struct MiniLineChart: View {
    let title: String
    let buckets: [TrainingBucket]
    let rangeAccessor: (TrainingBucket) -> ChartBucketRange?
    let unit: String
    let color: Color
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext
    var wholeNumber: Bool = false

    var body: some View {
        let readout = hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: rangeAccessor,
            bucketWidthSec: context.bucketWidthSec
        )
        let unitSuffix = unit.isEmpty ? "" : " \(unit)"
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = buckets.last.flatMap({ rangeAccessor($0)?.max }) {
                let valueStr = wholeNumber ? String(Int(v)) : TrainingChartGridView.compactLabel(v)
                headerText = "\(valueStr)\(unitSuffix)"
            } else {
                headerText = "--"
            }
        case .hoveringNoData(let t):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            let valueStr = wholeNumber ? String(Int(v)) : TrainingChartGridView.compactLabel(v)
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) \(valueStr)\(unitSuffix)"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: title, value: headerText)
                Chart {
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value(title, rangeAccessor(b)?.max ?? .nan)
                        )
                        .foregroundStyle(color)
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    if case .hoveringWithData(let t, let v) = readout {
                        PointMark(x: .value("Time", t), y: .value(title, v))
                            .foregroundStyle(color)
                            .symbolSize(40)
                    }
                }
                .modifier(StandardTimeSeriesChartModifiers(
                    context: context,
                    scrollX: $scrollX,
                    hoveredSec: $hoveredSec
                ))
            }
            .frame(height: 75)
        }
    }
}

/// Memory tile variant: same chart shape as `MiniLineChart` but the
/// header reads `X.X GB / Y.Y GB (Z.Z%)`.
fileprivate struct MemoryChart: View {
    let title: String
    let buckets: [TrainingBucket]
    let rangeAccessor: (TrainingBucket) -> ChartBucketRange?
    let totalGB: Double?
    let color: Color
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    var body: some View {
        let readout = hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: rangeAccessor,
            bucketWidthSec: context.bucketWidthSec
        )
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = buckets.last.flatMap({ rangeAccessor($0)?.max }) {
                headerText = Self.formatMemoryHeader(usedGB: v, totalGB: totalGB)
            } else {
                headerText = "--"
            }
        case .hoveringNoData(let t):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) \(Self.formatMemoryHeader(usedGB: v, totalGB: totalGB))"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: title, value: headerText)
                Chart {
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value(title, rangeAccessor(b)?.max ?? .nan)
                        )
                        .foregroundStyle(color)
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    if case .hoveringWithData(let t, let v) = readout {
                        PointMark(x: .value("Time", t), y: .value(title, v))
                            .foregroundStyle(color)
                            .symbolSize(40)
                    }
                }
                .modifier(StandardTimeSeriesChartModifiers(
                    context: context,
                    scrollX: $scrollX,
                    hoveredSec: $hoveredSec
                ))
            }
            .frame(height: 75)
        }
    }

    private static func formatMemoryHeader(usedGB: Double, totalGB: Double?) -> String {
        guard let totalGB, totalGB > 0 else {
            return String(format: "%.1f GB", usedGB)
        }
        let pct = usedGB / totalGB * 100
        return String(format: "%.1f GB / %.0f GB (%.0f%%)", usedGB, totalGB, pct)
    }
}

/// Power / thermal step-trace chart (categorical).
fileprivate struct PowerThermalChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    var body: some View {
        let readout = hoverReadout()
        let headerText = headerText(for: readout)
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: "Power / thermal", value: headerText)
                Chart {
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value(
                                "Power",
                                b.lowPowerMode.map { $0 ? 1.0 : 0.0 } ?? .nan
                            )
                        )
                        .foregroundStyle(by: .value("Series", "Low power"))
                        .interpolationMethod(.stepEnd)
                    }
                    ForEach(buckets) { b in
                        LineMark(
                            x: .value("Time", b.elapsedSec),
                            y: .value(
                                "Thermal",
                                b.thermalState.map { Self.thermalY($0) } ?? .nan
                            )
                        )
                        .foregroundStyle(by: .value("Series", "Thermal"))
                        .interpolationMethod(.stepEnd)
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                }
                .chartForegroundStyleScale([
                    "Low power": Color.gray,
                    "Thermal": Color.orange
                ])
                .chartLegend(.hidden)
                .chartYScale(domain: 0...5.5)
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    AxisMarks(position: .leading, values: [0, 1, 2, 3, 4, 5]) { _ in
                        AxisGridLine()
                    }
                }
                .chartXScale(domain: context.timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: context.visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    ChartHoverOverlay(proxy: proxy, hoveredSec: $hoveredSec)
                }
            }
            .frame(height: 75)
        }
    }

    private enum Readout {
        case notHovering
        case hoveringNoData(time: Double)
        case hoveringWithData(time: Double, lowPower: Bool, thermal: ProcessInfo.ThermalState)
    }

    private func hoverReadout() -> Readout {
        guard let t = hoveredSec else { return .notHovering }
        let tolerance = Swift.max(
            TrainingChartGridView.hoverMatchToleranceSec,
            context.bucketWidthSec * 1.5
        )
        guard let bucket = nearestTrainingBucket(
            at: t, in: buckets, tolerance: tolerance
        ) else {
            return .hoveringNoData(time: t)
        }
        guard let lp = bucket.lowPowerMode, let ts = bucket.thermalState else {
            return .hoveringNoData(time: t)
        }
        return .hoveringWithData(time: bucket.elapsedSec, lowPower: lp, thermal: ts)
    }

    private func headerText(for readout: Readout) -> String {
        switch readout {
        case .notHovering:
            if let latest = buckets.last {
                let powerStr = (latest.lowPowerMode ?? false) ? "on" : "off"
                let thermStr: String
                if let ts = latest.thermalState {
                    thermStr = Self.thermalStateName(ts)
                } else {
                    thermStr = "--"
                }
                return "lowpwr=\(powerStr)  thermal=\(thermStr)"
            } else {
                return "--"
            }
        case .hoveringNoData(let t):
            return "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let lp, let ts):
            let powerStr = lp ? "on" : "off"
            let thermStr = Self.thermalStateName(ts)
            return "t=\(TrainingChartGridView.formatElapsedAxis(t))  lowpwr=\(powerStr)  thermal=\(thermStr)"
        }
    }

    private static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func thermalY(_ state: ProcessInfo.ThermalState) -> Double {
        Double(state.rawValue) + 2
    }
}

/// Diversity histogram tile (categorical X axis).
fileprivate struct DiversityHistogramChart: View {
    let bars: [DiversityHistogramBar]

    @State private var hoveredHistogramBarID: Int?

    private static let bucketColors: [Color] = [
        .green, .mint, .yellow, .orange, .red, Color(red: 0.6, green: 0, blue: 0)
    ]

    var body: some View {
        let total = bars.reduce(0) { $0 + $1.count }
        let maxCount = bars.map(\.count).max() ?? 0
        let headerText: String
        if let hoveredID = hoveredHistogramBarID,
           let bar = bars.first(where: { $0.id == hoveredID }) {
            headerText = "\(bar.label) plies: \(bar.count)"
        } else if total > 0 {
            headerText = "\(total) games"
        } else {
            headerText = "--"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(title: "Longest move prefix", value: headerText)
                Chart(bars) { bar in
                    BarMark(
                        x: .value("Bucket", bar.label),
                        y: .value("Count", bar.count)
                    )
                    .foregroundStyle(
                        Self.bucketColors.indices.contains(bar.id)
                            ? Self.bucketColors[bar.id]
                            : Color.gray
                    )
                    .opacity(hoveredHistogramBarID == nil || hoveredHistogramBarID == bar.id ? 1.0 : 0.4)
                }
                .chartYScale(domain: 0...(maxCount > 0 ? Int(Double(maxCount) * 1.1) + 1 : 1))
                .chartXAxis {
                    AxisMarks(preset: .aligned, values: .automatic) { value in
                        AxisValueLabel {
                            if let label = value.as(String.self) {
                                Text(label)
                                    .font(.system(size: 6))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(TrainingChartGridView.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartOverlay { proxy in
                    GeometryReader { geo in
                        Rectangle()
                            .fill(Color.clear)
                            .contentShape(Rectangle())
                            .onContinuousHover { phase in
                                switch phase {
                                case .active(let point):
                                    let origin = (proxy.plotFrame.map { geo[$0].origin } ?? .zero)
                                    let xInPlot = point.x - origin.x
                                    if let label: String = proxy.value(atX: xInPlot),
                                       let match = bars.first(where: { $0.label == label }) {
                                        if hoveredHistogramBarID != match.id {
                                            hoveredHistogramBarID = match.id
                                        }
                                    } else if hoveredHistogramBarID != nil {
                                        hoveredHistogramBarID = nil
                                    }
                                case .ended:
                                    if hoveredHistogramBarID != nil {
                                        hoveredHistogramBarID = nil
                                    }
                                }
                            }
                    }
                }
            }
            .frame(height: 75)
        }
    }
}

/// Arena activity chart — one band per completed arena, plus a
/// live band for the in-progress arena (if any).
fileprivate struct ArenaActivityChart: View {
    let events: [ArenaChartEvent]
    let activeArenaStartElapsed: Double?
    /// Latest training-sample elapsed time. Used to draw the live
    /// arena band's "now" edge.
    let lastTrainingElapsedSec: Double?
    let promoteThreshold: Double
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

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

    var body: some View {
        let hoveredID = hoverArenaID
        let nowMark = liveNow
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
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
                .frame(height: 60)
            }
            .frame(height: 75)
        }
    }
}

/// Small progress-rate sparkline tile inside the grid (the big
/// version lives in the upper section of the app).
fileprivate struct SmallProgressRateChart: View {
    let buckets: [ProgressRateBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: ChartGridContext

    var body: some View {
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                ChartTileHeader(
                    title: "Progress rate (self play + train)",
                    value: headerText
                )
                progressChartBody.frame(height: 60)
            }
            .frame(height: 75)
        }
    }

    private var headerText: String {
        let tolerance = Swift.max(
            TrainingChartGridView.hoverMatchToleranceSec,
            context.bucketWidthSec * 1.5
        )
        if let t = hoveredSec {
            if let nearest = nearestProgressBucket(
                at: t, in: buckets, tolerance: tolerance
            ) {
                let combined = nearest.combinedMovesPerHour?.max ?? 0
                let selfPlay = nearest.selfPlayMovesPerHour?.max ?? 0
                let training = nearest.trainingMovesPerHour?.max ?? 0
                return "t=\(TrainingChartGridView.formatElapsedAxis(nearest.elapsedSec)) "
                    + "comb=\(TrainingChartGridView.compactLabel(combined)) "
                    + "sp=\(TrainingChartGridView.compactLabel(selfPlay)) "
                    + "tr=\(TrainingChartGridView.compactLabel(training))"
            } else {
                return "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
            }
        } else if let last = buckets.last,
                  let combined = last.combinedMovesPerHour?.max {
            return "\(TrainingChartGridView.compactLabel(combined)) moves/hour"
        } else {
            return "-- moves/hour"
        }
    }

    @ChartContentBuilder
    private var progressChartMarks: some ChartContent {
        ForEach(buckets) { b in
            LineMark(
                x: .value("Time", b.elapsedSec),
                y: .value("Moves/hr", b.combinedMovesPerHour?.max ?? .nan)
            )
            .foregroundStyle(by: .value("Series", "Combined"))
        }
        ForEach(buckets) { b in
            LineMark(
                x: .value("Time", b.elapsedSec),
                y: .value("Moves/hr", b.selfPlayMovesPerHour?.max ?? .nan)
            )
            .foregroundStyle(by: .value("Series", "Self-play"))
        }
        ForEach(buckets) { b in
            LineMark(
                x: .value("Time", b.elapsedSec),
                y: .value("Moves/hr", b.trainingMovesPerHour?.max ?? .nan)
            )
            .foregroundStyle(by: .value("Series", "Training"))
        }
        if let t = hoveredSec {
            RuleMark(x: .value("Time", t))
                .foregroundStyle(Color.gray.opacity(0.5))
                .lineStyle(StrokeStyle(lineWidth: 1))
        }
    }

    private var progressChartBody: some View {
        Chart { progressChartMarks }
            .chartForegroundStyleScale([
                "Self-play": Color.blue,
                "Training": Color.orange,
                "Combined": Color.green
            ])
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(TrainingChartGridView.compactLabel(v))
                                .font(.system(size: 7))
                                .monospacedDigit()
                        }
                    }
                }
            }
            .chartLegend(.hidden)
            .chartXScale(domain: context.timeSeriesXDomain)
            .chartScrollableAxes(.horizontal)
            .chartXVisibleDomain(length: context.visibleDomainSec)
            .chartScrollPosition(x: $scrollX)
            .chartOverlay { proxy in
                ChartHoverOverlay(proxy: proxy, hoveredSec: $hoveredSec)
            }
    }
}

// MARK: - Shared modifiers / helpers

/// Bundles the standard X-axis + Y-axis + scroll modifier chain
/// shared by every line-series tile. Pulled into a `ViewModifier`
/// so each chart subview's body stays focused on its marks rather
/// than a 7-line modifier chain.
fileprivate struct StandardTimeSeriesChartModifiers: ViewModifier {
    let context: ChartGridContext
    @Binding var scrollX: Double
    @Binding var hoveredSec: Double?

    func body(content: Content) -> some View {
        content
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(TrainingChartGridView.compactLabel(v))
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
    }
}

/// Compact tile header — title on the left, value on the right. The
/// header layout is identical across every tile, so this one view
/// replaces a repetitive `HStack { Text(...).font(.caption2)... }`
/// pattern in each chart subview.
fileprivate struct ChartTileHeader: View {
    let title: String
    let value: String

    var body: some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.caption2)
                .monospacedDigit()
                .foregroundStyle(.primary)
        }
    }
}
