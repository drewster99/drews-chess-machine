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

/// Grid of compact training-metric charts. All charts share a
/// synchronized horizontal scroll position.
struct TrainingChartGridView: View {
    let progressRateSamples: [ProgressRateSample]
    let trainingChartSamples: [TrainingChartSample]
    let visibleDomainSec: Double
    @Binding var scrollX: Double

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
                title: "Loss Total",
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
                title: "App Memory",
                yPath: \.appMemoryGB,
                unit: "GB",
                color: .brown
            )
            // Row 2: Loss Policy | Non-Neg Count | Replay Ratio | GPU % | GPU RAM
            miniChart(
                title: "Loss Policy",
                yPath: \.rollingPolicyLoss,
                unit: "",
                color: .orange
            )
            nonNegChart
            miniChart(
                title: "Replay Ratio",
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
            // Row 3: Loss Value
            miniChart(
                title: "Loss Value",
                yPath: \.rollingValueLoss,
                unit: "",
                color: .cyan
            )
        }
        .background(Color(nsColor: .separatorColor))
    }

    // MARK: - Progress rate chart (3 series)

    private var progressRateChart: some View {
        let lastCombined = progressRateSamples.last?.combinedMovesPerHour
        let headerValue = lastCombined.map { Self.compactLabel($0) } ?? "--"
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Progress Rate")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(headerValue) moves/hour")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart(progressRateSamples) { sample in
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
            }
            .frame(height: 75)
        }
    }

    // MARK: - Entropy chart (custom header with percentage)

    private static let maxEntropy = log(Double(ChessNetwork.policySize))

    private var entropyChart: some View {
        let lastValue = trainingChartSamples.last?.rollingPolicyEntropy
        let valueStr: String
        let pctStr: String
        if let v = lastValue {
            valueStr = String(format: "%.3f", v)
            pctStr = String(format: "(%.1f%%)", v / Self.maxEntropy * 100)
        } else {
            valueStr = "--"
            pctStr = ""
        }
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Policy Entropy")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(valueStr) \(pctStr)")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart(trainingChartSamples) { sample in
                    if let y = sample.rollingPolicyEntropy {
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Entropy", y)
                        )
                        .foregroundStyle(.purple)
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
            }
            .frame(height: 75)
        }
    }

    // MARK: - Non-negligible count chart (fixed Y-axis 0-4096)

    private var nonNegChart: some View {
        let lastValue = trainingChartSamples.last?.rollingPolicyNonNegCount
        let headerValue = lastValue.map { String(Int($0)) } ?? "--"
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("Non-Negligible Policy Count")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(headerValue) / 4096")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart(trainingChartSamples) { sample in
                    if let y = sample.rollingPolicyNonNegCount {
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value("Count", y)
                        )
                        .foregroundStyle(.mint)
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
        let lastValue = trainingChartSamples.last?[keyPath: yPath]
        let headerValue: String
        if let v = lastValue {
            headerValue = wholeNumber ? String(Int(v)) : Self.compactLabel(v)
        } else {
            headerValue = "--"
        }
        let unitSuffix = unit.isEmpty ? "" : " \(unit)"
        return chartCard {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(title)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(headerValue)\(unitSuffix)")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
                Chart(trainingChartSamples) { sample in
                    if let y = sample[keyPath: yPath] {
                        LineMark(
                            x: .value("Time", sample.elapsedSec),
                            y: .value(title, y)
                        )
                        .foregroundStyle(color)
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
            }
            .frame(height: 75)
        }
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
}
