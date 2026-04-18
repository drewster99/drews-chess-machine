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
    let replayRatio: Double?

    // System metrics
    let cpuPercent: Double?
    let gpuBusyPercent: Double?
    let gpuMemoryMB: Double?
    let appMemoryMB: Double?

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
    let visibleDomainSec: Double
    @Binding var scrollX: Double

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
            // Row 1: Loss Total | Entropy | Progress Rate | CPU % | App Memory
            miniChart(
                title: "Loss total",
                yPath: \.rollingTotalLoss,
                unit: "",
                color: .red
            )
            entropyChart
            progressRateChart
            miniChart(
                title: "CPU",
                yPath: \.cpuPercent,
                unit: "%",
                color: .blue
            )
            miniChart(
                title: "App memory",
                yPath: \.appMemoryGB,
                unit: "GB",
                color: .brown
            )
            // Row 2: Loss Policy | Non-Neg Count | Replay Ratio | GPU % | GPU RAM
            miniChart(
                title: "Loss policy",
                yPath: \.rollingPolicyLoss,
                unit: "",
                color: .orange
            )
            nonNegChart
            miniChart(
                title: "Replay ratio",
                yPath: \.replayRatio,
                unit: "train/move",
                color: .green
            )
            miniChart(
                title: "GPU",
                yPath: \.gpuBusyPercent,
                unit: "%",
                color: .indigo
            )
            miniChart(
                title: "GPU RAM",
                yPath: \.gpuMemoryGB,
                unit: "GB",
                color: .teal
            )
            // Row 3: Loss Value | Diversity histogram
            miniChart(
                title: "Loss value",
                yPath: \.rollingValueLoss,
                unit: "",
                color: .cyan
            )
            diversityHistogramChart
        }
        .background(Color(nsColor: .separatorColor))
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
                    Text("Diversity histogram")
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
        // When hovering, swap the header to show all three series
        // values at the hovered time; otherwise show the latest
        // combined rate.
        let headerText: String
        if let t = hoveredSec,
           let nearest = Self.nearestProgressSample(at: t, samples: progressRateSamples) {
            let combined = Self.compactLabel(nearest.combinedMovesPerHour)
            let selfPlay = Self.compactLabel(nearest.selfPlayMovesPerHour)
            let training = Self.compactLabel(nearest.trainingMovesPerHour)
            headerText = "t=\(Self.formatElapsedAxis(nearest.elapsedSec)) comb=\(combined) sp=\(selfPlay) tr=\(training)"
        } else if let last = progressRateSamples.last {
            headerText = "\(Self.compactLabel(last.combinedMovesPerHour)) moves/hour"
        } else {
            headerText = "-- moves/hour"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Progress rate")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(headerText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart {
                    ForEach(progressRateSamples) { sample in
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Moves/hr", sample.combinedMovesPerHour)
                        )
                        .foregroundStyle(by: .value("Series", "Combined"))
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Moves/hr", sample.selfPlayMovesPerHour)
                        )
                        .foregroundStyle(by: .value("Series", "Self-play"))
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Moves/hr", sample.trainingMovesPerHour)
                        )
                        .foregroundStyle(by: .value("Series", "Training"))
                    }
                    // Crosshair: vertical line at the hovered time,
                    // only rendered when a hover is active.
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                }
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

    // MARK: - Entropy chart (custom header with percentage)

    private static let maxEntropy = log(Double(ChessNetwork.policySize))

    private var entropyChart: some View {
        let hoveredValue = hoveredSampleValue(path: \.rollingPolicyEntropy)
        let hoveredTime = hoveredSampleTime()
        let headerText: String
        if let t = hoveredTime, let v = hoveredValue {
            let pct = v / Self.maxEntropy * 100
            headerText = "t=\(Self.formatElapsedAxis(t)) \(String(format: "%.3f (%.1f%%)", v, pct))"
        } else if let lastValue = trainingChartSamples.last?.rollingPolicyEntropy {
            headerText = String(format: "%.3f (%.1f%%)", lastValue, lastValue / Self.maxEntropy * 100)
        } else {
            headerText = "--"
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
                    ForEach(trainingChartSamples) { sample in
                        if let y = sample.rollingPolicyEntropy {
                            LineMark(
                                x: .value("Time", sample.elapsedSec),
                                y: .value("Entropy", y)
                            )
                            .foregroundStyle(.purple)
                        }
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    if let t = hoveredTime, let v = hoveredValue {
                        PointMark(x: .value("Time", t), y: .value("Entropy", v))
                            .foregroundStyle(.purple)
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

    // MARK: - Non-negligible count chart (fixed Y-axis 0-4096)

    private var nonNegChart: some View {
        let hoveredValue = hoveredSampleValue(path: \.rollingPolicyNonNegCount)
        let hoveredTime = hoveredSampleTime()
        let headerText: String
        if let t = hoveredTime, let v = hoveredValue {
            headerText = "t=\(Self.formatElapsedAxis(t)) \(Int(v)) / 4096"
        } else if let lastValue = trainingChartSamples.last?.rollingPolicyNonNegCount {
            headerText = "\(Int(lastValue)) / 4096"
        } else {
            headerText = "-- / 4096"
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Non-negligible policy count")
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
                        if let y = sample.rollingPolicyNonNegCount {
                            LineMark(
                                x: .value("Time", sample.elapsedSec),
                                y: .value("Count", y)
                            )
                            .foregroundStyle(.mint)
                        }
                    }
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    if let t = hoveredTime, let v = hoveredValue {
                        PointMark(x: .value("Time", t), y: .value("Count", v))
                            .foregroundStyle(.mint)
                            .symbolSize(40)
                    }
                }
                .chartYScale(domain: 0...4096)
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
        let hoveredValue = hoveredSampleValue(path: yPath)
        let hoveredTime = hoveredSampleTime()
        let unitSuffix = unit.isEmpty ? "" : " \(unit)"
        let headerText: String
        // When hovering, swap the header to show the hovered time +
        // the sample's value at that time. Otherwise show the
        // latest value (pre-hover behavior).
        if let t = hoveredTime, let v = hoveredValue {
            let valueStr = wholeNumber ? String(Int(v)) : Self.compactLabel(v)
            headerText = "t=\(Self.formatElapsedAxis(t)) \(valueStr)\(unitSuffix)"
        } else if let v = trainingChartSamples.last?[keyPath: yPath] {
            let valueStr = wholeNumber ? String(Int(v)) : Self.compactLabel(v)
            headerText = "\(valueStr)\(unitSuffix)"
        } else {
            headerText = "--"
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
                        if let y = sample[keyPath: yPath] {
                            LineMark(
                                x: .value("Time", sample.elapsedSec),
                                y: .value(title, y)
                            )
                            .foregroundStyle(color)
                        }
                    }
                    // Crosshair at hovered time across every mini
                    // chart — shared hoveredSec means all charts
                    // show the rule in lockstep.
                    if let t = hoveredSec {
                        RuleMark(x: .value("Time", t))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                    // Point marker at the nearest sample for this
                    // chart's series.
                    if let t = hoveredTime, let v = hoveredValue {
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

    /// Return the training-sample-nearest-to-hovered-time's value
    /// at `path`. Linear scan — `trainingChartSamples` is small (a
    /// few thousand samples at most during a long session) and the
    /// call rate is bounded by the mouse's physical move cadence,
    /// so a binary search isn't worth the complexity yet.
    private func hoveredSampleValue(
        path: KeyPath<TrainingChartSample, Double?>
    ) -> Double? {
        guard let t = hoveredSec,
              let sample = Self.nearestTrainingSample(at: t, samples: trainingChartSamples) else {
            return nil
        }
        return sample[keyPath: path]
    }

    /// Return the elapsed-second of the training-sample-nearest-to-
    /// hovered-time. Separated from `hoveredSampleValue` so per-
    /// chart headers can show "t=..." even when a specific series
    /// has no value at that sample.
    private func hoveredSampleTime() -> Double? {
        guard let t = hoveredSec,
              let sample = Self.nearestTrainingSample(at: t, samples: trainingChartSamples) else {
            return nil
        }
        return sample.elapsedSec
    }

    /// Linear-scan nearest-sample lookup. Returns `nil` on empty
    /// input (so headers gracefully degrade to "latest" mode).
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
        return best
    }

    /// Same as `nearestTrainingSample(...)` but for the three-series
    /// progress-rate chart.
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
