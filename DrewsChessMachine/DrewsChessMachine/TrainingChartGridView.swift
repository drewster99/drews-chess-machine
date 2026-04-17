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
    let gpuMemoryMB: Double?
    let appMemoryMB: Double?

    var rollingTotalLoss: Double? {
        guard let p = rollingPolicyLoss, let v = rollingValueLoss else { return nil }
        return p + v
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
        LazyVGrid(columns: Self.columns, spacing: 1) {
            // Row 1
            progressRateChart
            miniChart(
                title: "Policy Entropy",
                yPath: \.rollingPolicyEntropy,
                unit: "0=focused 8.3=uniform",
                color: .purple
            )
            miniChart(
                title: "Loss Total",
                yPath: \.rollingTotalLoss,
                unit: "policy+value",
                color: .red
            )
            miniChart(
                title: "Loss Policy",
                yPath: \.rollingPolicyLoss,
                unit: "CE weighted",
                color: .orange
            )
            miniChart(
                title: "Loss Value",
                yPath: \.rollingValueLoss,
                unit: "MSE",
                color: .cyan
            )
            // Row 2
            miniChart(
                title: "Replay Ratio",
                yPath: \.replayRatio,
                unit: "train/move",
                color: .green
            )
            miniChart(
                title: "Non-Neg Count",
                yPath: \.rollingPolicyNonNegCount,
                unit: "/ 4096",
                color: .mint,
                wholeNumber: true
            )
            miniChart(
                title: "CPU",
                yPath: \.cpuPercent,
                unit: "%",
                color: .blue
            )
            miniChart(
                title: "GPU RAM",
                yPath: \.gpuMemoryMB,
                unit: "MB",
                color: .indigo
            )
            miniChart(
                title: "App Memory",
                yPath: \.appMemoryMB,
                unit: "MB",
                color: .brown
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
