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
/// `Chart` can key BarMarks correctly as counts update.
struct DiversityHistogramBar: Identifiable, Sendable {
    let id: Int          // bucket index (stable across updates)
    let label: String    // bucket label like "0-2", "41+"
    let count: Int
}

/// One completed arena tournament, positioned on the chart grid's
/// shared elapsed-time X axis. The `startElapsedSec` / `endElapsedSec`
/// pair lets us render arenas as duration bands rather than point
/// events, so the reader can see WHEN training was paused for
/// arena play and for HOW LONG.
struct ArenaChartEvent: Identifiable, Sendable {
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

/// Grid of compact training-metric charts. All charts share a
/// synchronized horizontal scroll position and a single hover
/// crosshair — mousing over any time-series chart highlights the
/// same elapsed second across all of them so you can read every
/// metric at that moment.
struct TrainingChartGridView: View {
    let progressRateSamples: [ProgressRateSample]
    let trainingChartSamples: [TrainingChartSample]
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
    /// Unified-memory total in GB (`ProcessInfo.physicalMemory`),
    /// plumbed through so the App memory and GPU memory tiles can
    /// render "used / total (pct%)" headers.
    let appMemoryTotalGB: Double?
    let gpuMemoryTotalGB: Double?
    let visibleDomainSec: Double
    @Binding var scrollX: Double

    /// Full-data X domain used by every time-series chart. Computed
    /// lazily from the current sample buffers. When the data span
    /// `[0, maxElapsed]` is shorter than `visibleDomainSec`, the
    /// domain is widened to the visible-window length so the chart
    /// still renders the full visible area (and scrolling is a no-op
    /// — `chartScrollableAxes` won't allow scrolling past the data,
    /// which is the correct behavior while the session is young).
    /// When data is longer than the visible window, the full domain
    /// is what lets `chartScrollableAxes(.horizontal)` actually
    /// expose a scroll region — without an explicit `chartXScale`
    /// the Charts framework auto-fits to the plot frame and scroll
    /// becomes a no-op even with plenty of data.
    private var timeSeriesXDomain: ClosedRange<Double> {
        let lastElapsed = trainingChartSamples.last?.elapsedSec
            ?? progressRateSamples.last?.elapsedSec
            ?? 0
        let end = max(lastElapsed, visibleDomainSec)
        return 0...end
    }

    /// Shared hover selection across every time-series chart. Set
    /// by `.onContinuousHover` on each chart's overlay; `nil` means
    /// the mouse isn't over any time-series chart right now. When
    /// non-nil, every time-series chart draws a crosshair at this
    /// elapsed-second and swaps its header value to the sample
    /// nearest this time instead of the latest.
    @State private var hoveredSec: Double?

    /// Local hover state for the diversity histogram (categorical
    /// X-axis, so separate from the time-series `hoveredSec`).
    @State private var hoveredHistogramBarID: Int?

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
            // Row 1: pLossWin/pLossLoss | Entropy + LegalEntropy | Progress Rate | CPU % | App Memory
            policyLossSplitChart
            entropyChart
            progressRateChart
            miniChart(
                title: "CPU",
                yPath: \.cpuPercent,
                unit: "%",
                color: .blue
            )
            memoryChart(
                title: "App memory (RAM)",
                yPath: \.appMemoryGB,
                totalGB: appMemoryTotalGB,
                color: .brown
            )
            // Row 2: pLoss + vLoss combo | Non-Neg Count | Replay Ratio | GPU % | GPU RAM
            policyValueLossChart
            nonNegChart
            replayRatioChart
            miniChart(
                title: "GPU",
                yPath: \.gpuBusyPercent,
                unit: "%",
                color: .indigo
            )
            memoryChart(
                title: "GPU memory (RAM)",
                yPath: \.gpuMemoryGB,
                totalGB: gpuMemoryTotalGB,
                color: .teal
            )
            // Row 3: Legal mass sum | gNorm | Diversity histogram | Power / Thermal | Arena activity
            legalMassChart
            miniChart(
                title: "gNorm (gradient L2 norm)",
                yPath: \.rollingGradNorm,
                unit: "",
                color: .pink
            )
            diversityHistogramChart
            powerThermalChart
            arenaActivityChart
        }
        .background(Color(nsColor: .separatorColor))
    }

    // MARK: - Arena activity chart

    /// Local hover state for arena duration bands. Populated by a
    /// `chartOverlay` that converts the cursor X to an elapsed
    /// time, then finds the arena whose interval contains that
    /// time (rather than nearest-sample logic, which would snap to
    /// the closest boundary instead of the containing band).
    private var arenaActivityChart: some View {
        chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Arena activity")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(arenaActivityHeaderText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                arenaActivityChartBody
                    .frame(height: 60)
            }
            .frame(height: 75)
        }
    }

    /// Latest sample's elapsed time gives the "now" X coordinate to
    /// draw the live band out to. Falls back to the active start itself
    /// if no samples have landed yet (so the band still appears as a
    /// thin slice rather than being omitted). Pulled out of
    /// `arenaActivityChart` so the chart's getter type-checks faster.
    private var arenaActivityLiveNow: Double? {
        guard let start = activeArenaStartElapsed else { return nil }
        return max(start, trainingChartSamples.last?.elapsedSec ?? start)
    }

    /// `id` of the arena event currently under the cursor (if any).
    /// Used both for color-emphasis on the chart and as the hovered-
    /// arena header text branch.
    private var arenaActivityHoverArenaID: Int? {
        guard let t = hoveredSec else { return nil }
        for e in arenaEvents where t >= e.startElapsedSec && t <= e.endElapsedSec {
            return e.id
        }
        return nil
    }

    /// Header logic: live arena wins if one is running (so the reader
    /// knows arena play is active). Otherwise the hovered arena's
    /// stats if the cursor is over one; then the latest completed
    /// arena's summary; finally a running count.
    private var arenaActivityHeaderText: String {
        let events = arenaEvents
        if let start = activeArenaStartElapsed, let now = arenaActivityLiveNow {
            let elapsed = max(0, now - start)
            let durMin = Int(elapsed) / 60
            let durSec = Int(elapsed) % 60
            return String(format: "ARENA RUNNING  %d:%02d", durMin, durSec)
        } else if let hoverArenaID = arenaActivityHoverArenaID,
                  let e = events.first(where: { $0.id == hoverArenaID }) {
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

    /// The Chart and its modifier chain. Pulled out of
    /// `arenaActivityChart` so the if/else header logic and the chart
    /// generic stack type-check independently — before this split the
    /// combined getter was 117 ms.
    private var arenaActivityChartBody: some View {
        let events = arenaEvents
        let hoverArenaID = arenaActivityHoverArenaID
        let liveNow = arenaActivityLiveNow
        return Chart {
            // Two marks per arena:
            // 1. Faint full-height band showing the DURATION (always
            //    visible, even if the candidate scored 0 — a
            //    RectangleMark with yEnd=0 would otherwise collapse to
            //    invisible height and hide the arena entirely from
            //    the chart).
            // 2. Score bar from y=0 to y=score colored by promotion
            //    outcome (green promoted / gray kept). Score bar on
            //    top of the band so the "outcome" mark is the loud
            //    part, and the band just marks "arena ran here".
            ForEach(events) { e in
                RectangleMark(
                    xStart: .value("Start", e.startElapsedSec),
                    xEnd: .value("End", e.endElapsedSec),
                    yStart: .value("Floor", 0.0),
                    yEnd: .value("Top", 1.0)
                )
                .foregroundStyle(Color.secondary.opacity(hoverArenaID == e.id ? 0.25 : 0.12))
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
                        ? Color.green.opacity(hoverArenaID == e.id ? 1.0 : 0.7)
                        : Color.gray.opacity(hoverArenaID == e.id ? 1.0 : 0.5)
                )
            }
            // Live band for the in-progress arena: a full-height blue
            // rectangle from the arena's start up to "now" (the latest
            // chart sample). Drawn BEFORE the threshold / crosshair
            // so those remain visible on top. Distinct color from the
            // gray/green completed-arena bars so the "arena is
            // actively running" state is unambiguous.
            if let start = activeArenaStartElapsed, let now = liveNow {
                RectangleMark(
                    xStart: .value("Start", start),
                    xEnd: .value("Now", now),
                    yStart: .value("Floor", 0.0),
                    yEnd: .value("Top", 1.0)
                )
                .foregroundStyle(Color.blue.opacity(0.35))
            }
            // Promotion threshold line.
            RuleMark(y: .value("Threshold", promoteThreshold))
                .foregroundStyle(Color.orange.opacity(0.6))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
            // Shared hover crosshair.
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
        .chartXScale(domain: timeSeriesXDomain)
        .chartScrollableAxes(.horizontal)
        .chartXVisibleDomain(length: visibleDomainSec)
        .chartScrollPosition(x: $scrollX)
        .chartOverlay { proxy in
            hoverOverlay(proxy: proxy)
        }
    }

    // MARK: - Power / thermal chart

    /// Human-readable name for a `ProcessInfo.ThermalState`. Used
    /// by both the chart's header text and the hover readout so
    /// the reader sees "serious" rather than `.serious` or `2`.
    private static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// Convert a thermal state to its chart Y position:
    /// `rawValue + 2` → 2…5. Keeps the trace strictly above the
    /// 0/1 low-power step so the two series never overlap.
    private static func thermalY(_ state: ProcessInfo.ThermalState) -> Double {
        Double(state.rawValue) + 2
    }

    private var powerThermalChart: some View {
        let readout = hoverReadoutPowerThermal()
        let headerText: String
        // Three-way header logic matching the other hover-aware
        // tiles. Shows latest state when not hovering, the hovered
        // sample's power + thermal when hovering with data, and an
        // explicit "no data" when the hover is outside the sampled
        // range.
        switch readout {
        case .notHovering:
            if let latest = trainingChartSamples.last {
                let powerStr = (latest.lowPowerMode ?? false) ? "on" : "off"
                let thermStr: String
                if let ts = latest.thermalState {
                    thermStr = Self.thermalStateName(ts)
                } else {
                    thermStr = "--"
                }
                headerText = "lowpwr=\(powerStr)  thermal=\(thermStr)"
            } else {
                headerText = "--"
            }
        case .hoveringNoData(let t):
            headerText = "t=\(Self.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let lowPower, let thermal):
            let powerStr = lowPower ? "on" : "off"
            let thermStr = Self.thermalStateName(thermal)
            headerText = "t=\(Self.formatElapsedAxis(t))  lowpwr=\(powerStr)  thermal=\(thermStr)"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Power / thermal")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    // Low power as a 0/1 step trace. Always emit a
                    // LineMark so the view shape is stable across
                    // samples — nil `lowPowerMode` maps to `.nan`,
                    // which Swift Charts renders as a gap.
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value(
                                "Power",
                                sample.lowPowerMode.map { $0 ? 1.0 : 0.0 } ?? .nan
                            )
                        )
                        .foregroundStyle(by: .value("Series", "Low power"))
                        .interpolationMethod(.stepEnd)
                    }
                    // Thermal state trace, offset by +2 so it sits
                    // in the 2-5 range strictly above the 0/1
                    // low-power band. Step interpolation because
                    // thermal state is a discrete level, not a
                    // continuous measurement. Same NaN-for-gap
                    // pattern for structural stability.
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value(
                                "Thermal",
                                sample.thermalState.map { Self.thermalY($0) } ?? .nan
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
                // Force the Y domain to 0-5.5 so the two traces
                // sit in their intended bands regardless of what
                // values have been observed. Without this the
                // axis would auto-scale to the current max and
                // the trace band positions would shift.
                .chartYScale(domain: 0...5.5)
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    // Hide Y axis labels — the numeric scale is
                    // meaningless on its own (raw thermal values +
                    // 2 mixed with 0/1 low-power flags). Hover and
                    // header text carry the actual semantics.
                    AxisMarks(position: .leading, values: [0, 1, 2, 3, 4, 5]) { value in
                        AxisGridLine()
                    }
                }
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
            }
            .frame(height: 75)
        }
    }

    /// Three-way hover result for the power/thermal chart. Parallels
    /// the generic `HoverReadout` but carries the two fields we
    /// need for this chart's header simultaneously.
    private enum PowerThermalReadout {
        case notHovering
        case hoveringNoData(hoveredTime: Double)
        case hoveringWithData(time: Double, lowPower: Bool, thermal: ProcessInfo.ThermalState)
    }

    private func hoverReadoutPowerThermal() -> PowerThermalReadout {
        guard let hoverT = hoveredSec else { return .notHovering }
        guard let sample = Self.nearestTrainingSample(
            at: hoverT,
            samples: trainingChartSamples
        ) else {
            return .hoveringNoData(hoveredTime: hoverT)
        }
        guard let lp = sample.lowPowerMode,
              let ts = sample.thermalState else {
            return .hoveringNoData(hoveredTime: hoverT)
        }
        return .hoveringWithData(time: sample.elapsedSec, lowPower: lp, thermal: ts)
    }

    // MARK: - Diversity histogram chart

    /// Color ramp for the histogram buckets, matched to severity:
    /// green bands are the healthy-diversity region, yellow/orange
    /// is "watch", red is "games sharing deep-middlegame play —
    /// policy is collapsing". Aligned index-wise with
    /// `GameDiversityTracker.histogramLabels`.
    private static let diversityBucketColors: [Color] = [
        .green, .mint, .yellow, .orange, .red, Color(red: 0.6, green: 0, blue: 0)
    ]

    private var diversityHistogramChart: some View {
        let bars = diversityHistogram
        let total = bars.reduce(0) { $0 + $1.count }
        let maxCount = bars.map(\.count).max() ?? 0
        let headerText: String
        // Header shows the hovered bucket's label and count when
        // hovering, falls back to "N games" total otherwise.
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
                HStack(spacing: 4) {
                    Text("Longest move prefix")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart(bars) { bar in
                    BarMark(
                        x: .value("Bucket", bar.label),
                        y: .value("Count", bar.count)
                    )
                    .foregroundStyle(
                        Self.diversityBucketColors.indices.contains(bar.id)
                            ? Self.diversityBucketColors[bar.id]
                            : Color.gray
                    )
                    .opacity(hoveredHistogramBarID == nil || hoveredHistogramBarID == bar.id ? 1.0 : 0.4)
                }
                // `maxCount * 1.1` so the tallest bar doesn't touch
                // the ceiling. Fall back to 1...1 domain on empty so
                // the axis renders an empty frame instead of NaN'ing.
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
                                Text(Self.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartOverlay { proxy in
                    // Categorical hover: convert the mouse X into a
                    // bucket label via `proxy.value(atX:)`, then find
                    // the bar with that label and cache its ID so the
                    // header swaps to its count. Using the ID rather
                    // than the label index lets the bar ordering
                    // shift without stale-pointer hazards.
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

    // MARK: - Progress rate chart (3 series)

    private var progressRateChart: some View {
        chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Progress rate (self play + train)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(progressRateHeaderText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                progressRateChartBody
                    .frame(height: 60)
            }
            .frame(height: 75)
        }
    }

    // Three-way header logic mirroring the single-series charts.
    // "Hovering-no-data" applies when the cursor is over a chart but
    // the nearest progress sample is outside tolerance — e.g. the
    // user scrubbed to a time before the session actually started
    // sampling. Extracted to its own computed property so
    // `progressRateChart` doesn't have to type-check the if/else
    // alongside the Chart's deep generic stack.
    private var progressRateHeaderText: String {
        if let t = hoveredSec {
            if let nearest = Self.nearestProgressSample(at: t, samples: progressRateSamples) {
                let combined = Self.compactLabel(nearest.combinedMovesPerHour)
                let selfPlay = Self.compactLabel(nearest.selfPlayMovesPerHour)
                let training = Self.compactLabel(nearest.trainingMovesPerHour)
                return "t=\(Self.formatElapsedAxis(nearest.elapsedSec)) comb=\(combined) sp=\(selfPlay) tr=\(training)"
            } else {
                return "t=\(Self.formatElapsedAxis(t)) — no data"
            }
        } else if let last = progressRateSamples.last {
            return "\(Self.compactLabel(last.combinedMovesPerHour)) moves/hour"
        } else {
            return "-- moves/hour"
        }
    }

    /// Chart marks for the small progress-rate sparkline. Pulled out
    /// as a `@ChartContentBuilder` helper so it type-checks
    /// separately from the modifier chain on `progressRateChartBody`.
    ///
    /// One ForEach per series — SwiftUI Charts only connects
    /// LineMarks that share a single enclosing ForEach. Packing all
    /// three series into one ForEach produced spurious flat lines at
    /// y=0 because Charts couldn't disambiguate series within the
    /// shared iteration.
    @ChartContentBuilder
    private var progressRateChartMarks: some ChartContent {
        ForEach(progressRateSamples) { sample in
            LineMark(
                x: .value("Time", sample.elapsedSec),
                y: .value("Moves/hr", sample.combinedMovesPerHour)
            )
            .foregroundStyle(by: .value("Series", "Combined"))
        }
        ForEach(progressRateSamples) { sample in
            LineMark(
                x: .value("Time", sample.elapsedSec),
                y: .value("Moves/hr", sample.selfPlayMovesPerHour)
            )
            .foregroundStyle(by: .value("Series", "Self-play"))
        }
        ForEach(progressRateSamples) { sample in
            LineMark(
                x: .value("Time", sample.elapsedSec),
                y: .value("Moves/hr", sample.trainingMovesPerHour)
            )
            .foregroundStyle(by: .value("Series", "Training"))
        }
        // Crosshair: vertical line at the hovered time, only rendered
        // when a hover is active.
        if let t = hoveredSec {
            RuleMark(x: .value("Time", t))
                .foregroundStyle(Color.gray.opacity(0.5))
                .lineStyle(StrokeStyle(lineWidth: 1))
        }
    }

    // The Chart and its modifier chain. Pulled out so it type-checks
    // independently of `progressRateChart`'s outer scaffolding —
    // before this split, the combined getter was 223 ms; with marks
    // inlined, body alone was 147 ms.
    private var progressRateChartBody: some View {
        Chart { progressRateChartMarks }
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
                        Text(Self.compactLabel(v))
                            .font(.system(size: 7))
                            .monospacedDigit()
                    }
                }
            }
        }
        .chartLegend(.hidden)
        .chartXScale(domain: timeSeriesXDomain)
        .chartScrollableAxes(.horizontal)
        .chartXVisibleDomain(length: visibleDomainSec)
        .chartScrollPosition(x: $scrollX)
        .chartOverlay { proxy in
            hoverOverlay(proxy: proxy)
        }
    }

    // MARK: - Entropy chart (custom header with percentage)

    private static let maxEntropy = log(Double(ChessNetwork.policySize))

    /// Reference uniform-distribution entropy for the legal-only
    /// renormalized softmax. A typical chess position has ~30 legal
    /// moves, so log(30) ≈ 3.40 nats is a reasonable "fully diffuse
    /// over legal" baseline. Used to render pEntLegal as a percentage
    /// of its own uniform reference, parallel to how pEnt is rendered
    /// against `maxEntropy`.
    private static let maxLegalEntropy = log(30.0)

    /// Upper-left tile (replaces the legacy "Loss (pLoss + vLoss)"
    /// total-loss sparkgraph). Plots two outcome-partitioned series:
    /// the policy loss restricted to win-outcome batch positions
    /// (`pLossWin`) and to loss-outcome positions (`pLossLoss`).
    /// The total-loss curve is ambiguous because outcome-weighted CE
    /// flips sign with z; splitting by outcome makes both lines
    /// individually interpretable — both should drift below zero
    /// as the network learns to favor moves correlating with their
    /// own game outcomes.
    private var policyLossSplitChart: some View {
        let lastWin = trainingChartSamples.last?.rollingPolicyLossWin
        let lastLoss = trainingChartSamples.last?.rollingPolicyLossLoss
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
                HStack(spacing: 4) {
                    Text("pLoss split (W vs L)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("pLoss", sample.rollingPolicyLossWin ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pLossWin"))
                    }
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("pLoss", sample.rollingPolicyLossLoss ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pLossLoss"))
                    }
                    // Zero baseline — pLoss under the advantage-
                    // weighted CE can be negative; the y=0 reference
                    // line makes the sign visually obvious.
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
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Self.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
            }
            .frame(height: 75)
        }
    }

    /// Combo chart showing pLoss (policy loss, orange) and vLoss
    /// (value loss, cyan) on a shared Y axis. Both curves carry
    /// useful training-health signal but live on similar magnitudes,
    /// so plotting them together keeps the row-2 column-1 tile dense
    /// and frees row-3 column-1 for the legal-mass tile below.
    private var policyValueLossChart: some View {
        let lastP = trainingChartSamples.last?.rollingPolicyLoss
        let lastV = trainingChartSamples.last?.rollingValueLoss
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
                HStack(spacing: 4) {
                    Text("pLoss + vLoss")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Loss", sample.rollingPolicyLoss ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pLoss"))
                    }
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Loss", sample.rollingValueLoss ?? .nan)
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
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Self.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
            }
            .frame(height: 75)
        }
    }

    /// Legal mass sum tile — fraction of the full softmax that lands
    /// on legal moves at the probed position. Y axis is fixed to
    /// `0...1` so the reader sees absolute health (a healthy network
    /// concentrates ≈1.0 on legal moves) rather than auto-scaled
    /// noise. Header reads as a percentage to match the format used
    /// in the Policy Stats panel.
    private var legalMassChart: some View {
        let readout = hoverReadout(path: \.rollingLegalMass)
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = trainingChartSamples.last?.rollingLegalMass {
                headerText = String(format: "%.4f%%", v * 100)
            } else {
                headerText = "--"
            }
        case .hoveringNoData(let t):
            headerText = "t=\(Self.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            headerText = String(
                format: "t=%@ %.4f%%",
                Self.formatElapsedAxis(t), v * 100
            )
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Legal mass sum")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Legal mass", sample.rollingLegalMass ?? .nan)
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
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
            }
            .frame(height: 75)
        }
    }

    private var entropyChart: some View {
        let readout = hoverReadout(path: \.rollingPolicyEntropy)
        let headerText: String
        // Pull the two latest values once so the header logic below
        // can format them side-by-side ("pEnt X.XXX / pEntLegal Y.YYY")
        // analogous to the pLoss-split chart's "win / loss" header.
        let latestPEnt = trainingChartSamples.last?.rollingPolicyEntropy
        let latestPEntLegal = trainingChartSamples.last?.rollingLegalEntropy

        func formatEntropy(_ v: Double?) -> String {
            guard let v else { return "--" }
            return String(format: "%.3f", v)
        }

        switch readout {
        case .notHovering:
            // Two-value form. pEnt is normalized against the
            // full-policy max (log(4864) ≈ 8.49); pEntLegal is
            // normalized against log(30) ≈ 3.4 — a typical legal-move
            // count. Both percentages are diagnostic-only so the
            // reader can scan how concentrated each distribution is
            // relative to its own uniform reference.
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
            headerText = "pEnt \(pEntStr) / pEntLegal \(pEntLegalStr)"
        case .hoveringNoData(let t):
            headerText = "t=\(Self.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            // Hover targets the pEnt series; show pEnt at the hover
            // time and the most-recent pEntLegal alongside so both
            // numbers stay visible while the user inspects history.
            let pEntStr = String(format: "%.3f (%.1f%%)", v, v / Self.maxEntropy * 100)
            headerText = "t=\(Self.formatElapsedAxis(t)) pEnt \(pEntStr) / pEntLegal \(formatEntropy(latestPEntLegal))"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Policy entropy")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    // Stable view shape: `nil` becomes `.nan`, which
                    // Swift Charts renders as a gap — see the
                    // matching comment in `miniChart`.
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Entropy", sample.rollingPolicyEntropy ?? .nan)
                        )
                        .foregroundStyle(by: .value("Series", "pEnt"))
                    }
                    // pEntLegal — entropy over the legal-only
                    // renormalized softmax. log(N_legal) ≈ 3.4 nats at
                    // 30 legal moves; falls toward 0 as the policy
                    // concentrates on preferred legal moves.
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Entropy", sample.rollingLegalEntropy ?? .nan)
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
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Self.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
            }
            .frame(height: 75)
        }
    }

    // MARK: - Non-negligible count chart (fixed Y-axis 0..policySize)

    private var nonNegChart: some View {
        // Header tracks the LEGAL series under the hover cursor — same
        // primary signal as before — but the body of the chart renders
        // both legal (mint) and illegal (red) so the user can see the
        // mask leakage as a second curve climbing off zero.
        let legalReadout = hoverReadout(path: \.rollingPolicyNonNegCount)
        let policyMax = ChessNetwork.policySize
        let headerText: String
        switch legalReadout {
        case .notHovering:
            let lastLegal = trainingChartSamples.last?.rollingPolicyNonNegCount
            let lastIllegal = trainingChartSamples.last?.rollingPolicyNonNegIllegalCount
            let legalStr = lastLegal.map { String(Int($0)) } ?? "--"
            let illegalStr = lastIllegal.map { String(Int($0)) } ?? "--"
            headerText = "legal \(legalStr) • illegal \(illegalStr)"
        case .hoveringNoData(let t):
            headerText = "t=\(Self.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            // Look up the matching illegal value at the same hover
            // index so both numbers in the readout came from the same
            // sample.
            let illegalAtHover = trainingChartSamples
                .min(by: { abs($0.elapsedSec - t) < abs($1.elapsedSec - t) })?
                .rollingPolicyNonNegIllegalCount
            let illegalStr = illegalAtHover.map { String(Int($0)) } ?? "--"
            headerText = "t=\(Self.formatElapsedAxis(t))  legal \(Int(v)) • illegal \(illegalStr)"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Above-uniform policy count")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    // Stable view shape: `nil` becomes `.nan`, which
                    // Swift Charts renders as a gap — see the
                    // matching comment in `miniChart`.
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Count", sample.rollingPolicyNonNegCount ?? .nan),
                            series: .value("Series", "Legal")
                        )
                        .foregroundStyle(by: .value("Series", "Legal"))
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Count", sample.rollingPolicyNonNegIllegalCount ?? .nan),
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
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Self.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
            }
            .frame(height: 75)
        }
    }

    // MARK: - Memory chart (used + total + percent)

    /// Specialized mini chart for memory tiles — same plot shape as
    /// `miniChart` but the header reads `X.X GB / Y.Y GB (Z.Z%)` so
    /// the reader sees both the current footprint and the fraction of
    /// the unified-memory pool it represents. Falls back to the bare
    /// value string when the total is unavailable.
    private func memoryChart(
        title: String,
        yPath: KeyPath<TrainingChartSample, Double?>,
        totalGB: Double?,
        color: Color
    ) -> some View {
        let readout = hoverReadout(path: yPath)
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = trainingChartSamples.last?[keyPath: yPath] {
                headerText = Self.formatMemoryHeader(usedGB: v, totalGB: totalGB)
            } else {
                headerText = "--"
            }
        case .hoveringNoData(let t):
            headerText = "t=\(Self.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            headerText = "t=\(Self.formatElapsedAxis(t)) \(Self.formatMemoryHeader(usedGB: v, totalGB: totalGB))"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(title)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value(title, sample[keyPath: yPath] ?? .nan)
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
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Self.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
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

    // MARK: - Replay ratio chart (line + target reference)

    /// Dedicated chart for the live replay ratio. Same shape as the
    /// generic `miniChart` but draws a red dashed `RuleMark` at
    /// `replayRatioTarget` so the reader can see target vs. actual
    /// on the same tile — the auto-adjust controller's error signal
    /// is the gap between the green line and the red dashes.
    private var replayRatioChart: some View {
        let readout = hoverReadout(path: \.replayRatio)
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = trainingChartSamples.last?.replayRatio {
                headerText = String(format: "%.2f (target %.2f)", v, replayRatioTarget)
            } else {
                headerText = String(format: "-- (target %.2f)", replayRatioTarget)
            }
        case .hoveringNoData(let t):
            headerText = "t=\(Self.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            headerText = String(
                format: "t=%@ %.2f (target %.2f)",
                Self.formatElapsedAxis(t), v, replayRatioTarget
            )
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Replay ratio")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    RuleMark(y: .value("Target", replayRatioTarget))
                        .foregroundStyle(Color.red.opacity(0.6))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Replay ratio", sample.replayRatio ?? .nan)
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
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Self.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
            }
            .frame(height: 75)
        }
    }

    // MARK: - Generic single-series mini chart

    private func miniChart(
        title: String,
        yPath: KeyPath<TrainingChartSample, Double?>,
        unit: String,
        color: Color,
        wholeNumber: Bool = false
    ) -> some View {
        let readout = hoverReadout(path: yPath)
        let unitSuffix = unit.isEmpty ? "" : " \(unit)"
        let headerText: String
        // Three-way header logic: not hovering (show latest if any),
        // hovering-with-data (show nearest sample's time + value),
        // hovering-but-no-data (show the raw hover time + "no data"
        // instead of silently falling back to the last sample,
        // which misled the reader into thinking the last-known
        // value extended past its actual range).
        switch readout {
        case .notHovering:
            if let v = trainingChartSamples.last?[keyPath: yPath] {
                let valueStr = wholeNumber ? String(Int(v)) : Self.compactLabel(v)
                headerText = "\(valueStr)\(unitSuffix)"
            } else {
                headerText = "--"
            }
        case .hoveringNoData(let t):
            headerText = "t=\(Self.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            let valueStr = wholeNumber ? String(Int(v)) : Self.compactLabel(v)
            headerText = "t=\(Self.formatElapsedAxis(t)) \(valueStr)\(unitSuffix)"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(title)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    // Always emit one LineMark per sample so the
                    // view-tree shape is identical on every update —
                    // missing samples become `NaN` y-values, which
                    // Swift Charts renders as a gap in the line
                    // without disturbing the axis or the attribute
                    // graph. Putting the `if let` inside the ForEach
                    // body made every sample's view identity flip
                    // between two different shapes, which is what
                    // the profiler flagged as the dominant hang cost
                    // in `GeometryReader.Child.updateValue`.
                    ForEach(trainingChartSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value(title, sample[keyPath: yPath] ?? .nan)
                        )
                        .foregroundStyle(color)
                    }
                    // Crosshair at hovered time across every mini
                    // chart — shared hoveredSec means all charts
                    // show the rule in lockstep. Always drawn when
                    // hovering, even if this chart has no data at
                    // that time (so the reader sees WHERE they are
                    // even on silent series).
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    // Point marker only when we have a real sample
                    // value — explicitly NOT drawn in the
                    // hoveringNoData case to avoid misleading dots.
                    if case .hoveringWithData(let t, let v) = readout {
                        PointMark(x: .value("Time", t), y: .value(title, v))
                            .foregroundStyle(color)
                            .symbolSize(40)
                    }
                }
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Self.compactLabel(v))
                                    .font(.system(size: 7))
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                .chartXScale(domain: timeSeriesXDomain)
                .chartScrollableAxes(.horizontal)
                .chartXVisibleDomain(length: visibleDomainSec)
                .chartScrollPosition(x: $scrollX)
                .chartOverlay { proxy in
                    hoverOverlay(proxy: proxy)
                }
            }
            .frame(height: 75)
        }
    }

    // MARK: - Hover overlay helper

    /// Transparent overlay that captures mouse position on a
    /// time-series chart and pipes the converted elapsed-second
    /// into the shared `hoveredSec` @State. `.contentShape` on the
    /// rectangle is what makes the whole plot area hit-test even
    /// though the fill is clear. The `ChartProxy` maps the pixel
    /// X position (relative to the plot area, not the card) back
    /// into the chart's X-data space.
    private func hoverOverlay(proxy: ChartProxy) -> some View {
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
                            // Clamp to the samples' time range to
                            // avoid edge-case "hovering beyond the
                            // last sample shows the last sample's
                            // values" surprises — we specifically
                            // want no point marker past the end.
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

    // MARK: - Hover lookup helpers

    /// Maximum distance in seconds between the hovered cursor time
    /// and the nearest sample's time for the sample to count as
    /// "data at this hover time". Training samples land at 1 Hz, so
    /// 1.5 s accommodates normal sampling jitter while rejecting
    /// hovers that fall well outside any sample's reach (e.g. past
    /// the last sample or before the first one). Without this gate,
    /// a pure nearest-sample lookup would silently show stale
    /// first/last-sample values for arbitrary off-range hover
    /// positions and mislead the reader.
    static let hoverMatchToleranceSec: Double = 1.5

    /// Three-way result of a per-chart hover query.
    private enum HoverReadout {
        /// The cursor isn't over any time-series chart right now.
        case notHovering
        /// The cursor IS over a chart, but either (a) the nearest
        /// sample is outside `hoverMatchToleranceSec` of the
        /// hovered time, or (b) this specific series has no value
        /// at the nearest sample. Carries the raw hovered time so
        /// the header can still display "t=M:SS — no data".
        case hoveringNoData(hoveredTime: Double)
        /// The cursor is over a chart and this series has a value
        /// at a sample within the match tolerance. Carries the
        /// actual sample's time (not the raw hover time, so point
        /// markers land exactly on the line) plus the value.
        case hoveringWithData(sampleTime: Double, value: Double)
    }

    /// Per-series hover lookup. Returns a `HoverReadout` covering
    /// the three possible UI states so the chart header can render
    /// unambiguous text and the PointMark can be drawn only when a
    /// real value exists.
    private func hoverReadout(
        path: KeyPath<TrainingChartSample, Double?>
    ) -> HoverReadout {
        guard let hoverT = hoveredSec else { return .notHovering }
        guard let sample = Self.nearestTrainingSample(
            at: hoverT,
            samples: trainingChartSamples
        ) else {
            return .hoveringNoData(hoveredTime: hoverT)
        }
        guard let v = sample[keyPath: path] else {
            return .hoveringNoData(hoveredTime: hoverT)
        }
        return .hoveringWithData(sampleTime: sample.elapsedSec, value: v)
    }

    /// Linear-scan nearest-sample lookup. Returns `nil` when the
    /// array is empty OR when the nearest sample is farther than
    /// `hoverMatchToleranceSec` from `t` — the caller interprets
    /// `nil` as "no data at this hover time".
    private static func nearestTrainingSample(
        at t: Double,
        samples: [TrainingChartSample]
    ) -> TrainingChartSample? {
        guard !samples.isEmpty else { return nil }
        var best: TrainingChartSample = samples[0]
        var bestDist = Swift.abs(best.elapsedSec - t)
        for s in samples.dropFirst() {
            let d = Swift.abs(s.elapsedSec - t)
            if d < bestDist {
                best = s
                bestDist = d
            }
        }
        guard bestDist <= hoverMatchToleranceSec else { return nil }
        return best
    }

    /// Same as `nearestTrainingSample(...)` but for the three-series
    /// progress-rate chart. Same tolerance gate applies so hovers
    /// past the last sample don't masquerade as stale last-sample
    /// values.
    private static func nearestProgressSample(
        at t: Double,
        samples: [ProgressRateSample]
    ) -> ProgressRateSample? {
        guard !samples.isEmpty else { return nil }
        var best: ProgressRateSample = samples[0]
        var bestDist = Swift.abs(best.elapsedSec - t)
        for s in samples.dropFirst() {
            let d = Swift.abs(s.elapsedSec - t)
            if d < bestDist {
                best = s
                bestDist = d
            }
        }
        guard bestDist <= hoverMatchToleranceSec else { return nil }
        return best
    }

    // MARK: - Card wrapper

    private func chartCard<Content: View>(
        @ViewBuilder content: () -> Content
    ) -> some View {
        content()
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Compact label formatter

    static func compactLabel(_ value: Double) -> String {
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

    /// Compact mm:ss / h:mm:ss formatter for the hover header's
    /// time stamp. Mirrors `ContentView.formatElapsedAxis` semantics
    /// but lives here so the grid doesn't reach back into
    /// `ContentView` (the compilation unit's public API surface).
    static func formatElapsedAxis(_ seconds: Double) -> String {
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
