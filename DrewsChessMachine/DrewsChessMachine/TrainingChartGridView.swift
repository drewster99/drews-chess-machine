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

    var rollingTotalLoss: Double? {
        guard let p = rollingPolicyLoss, let v = rollingValueLoss else { return nil }
        return p + v
    }
}

/// Grid of compact training-metric charts. 4 columns × 2 rows,
/// all sharing a synchronized horizontal scroll position so
/// panning any chart pans all of them.
struct TrainingChartGridView: View {
    let progressRateSamples: [ProgressRateSample]
    let trainingChartSamples: [TrainingChartSample]
    let visibleDomainSec: Double
    @Binding var scrollX: Double

    private static let columns = Array(
        repeating: GridItem(.flexible(), spacing: 6),
        count: 4
    )

    var body: some View {
        LazyVGrid(columns: Self.columns, spacing: 6) {
            // Row 1
            progressRateChart
            miniChart(
                title: "Policy Entropy",
                samples: trainingChartSamples,
                yPath: \.rollingPolicyEntropy,
                yLabel: "nats",
                color: .purple
            )
            miniChart(
                title: "Loss Total",
                samples: trainingChartSamples,
                yPath: \.rollingTotalLoss,
                yLabel: "loss",
                color: .red
            )
            miniChart(
                title: "Loss Policy",
                samples: trainingChartSamples,
                yPath: \.rollingPolicyLoss,
                yLabel: "loss",
                color: .orange
            )
            // Row 2
            miniChart(
                title: "Loss Value",
                samples: trainingChartSamples,
                yPath: \.rollingValueLoss,
                yLabel: "loss",
                color: .cyan
            )
            miniChart(
                title: "Replay Ratio",
                samples: trainingChartSamples,
                yPath: \.replayRatio,
                yLabel: "ratio",
                color: .green
            )
            miniChart(
                title: "Non-Negligible Count",
                samples: trainingChartSamples,
                yPath: \.rollingPolicyNonNegCount,
                yLabel: "count",
                color: .mint
            )
        }
    }

    // MARK: - Progress rate chart (3 series)

    private var progressRateChart: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Progress Rate")
                .font(.caption)
                .foregroundStyle(.secondary)
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
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 3)) { _ in
                    AxisGridLine()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(v.formatted(.number.notation(.compactName)))
                                .font(.system(size: 8))
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
        .frame(height: 85)
    }

    // MARK: - Generic single-series mini chart

    private func miniChart(
        title: String,
        samples: [TrainingChartSample],
        yPath: KeyPath<TrainingChartSample, Double?>,
        yLabel: String,
        color: Color
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Chart(samples) { sample in
                if let y = sample[keyPath: yPath] {
                    LineMark(
                        x: .value("Time", sample.elapsedSec),
                        y: .value(yLabel, y)
                    )
                    .foregroundStyle(color)
                }
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 3)) { _ in
                    AxisGridLine()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(Self.compactLabel(v))
                                .font(.system(size: 8))
                                .monospacedDigit()
                        }
                    }
                }
            }
            .chartScrollableAxes(.horizontal)
            .chartXVisibleDomain(length: visibleDomainSec)
            .chartScrollPosition(x: $scrollX)
        }
        .frame(height: 85)
    }

    private static func compactLabel(_ value: Double) -> String {
        let abs = Swift.abs(value)
        if abs >= 1_000_000 {
            return String(format: "%.1fM", value / 1_000_000)
        } else if abs >= 1_000 {
            return String(format: "%.1fK", value / 1_000)
        } else if abs >= 10 {
            return String(format: "%.0f", value)
        } else if abs >= 1 {
            return String(format: "%.1f", value)
        } else if abs >= 0.01 {
            return String(format: "%.2f", value)
        } else {
            return String(format: "%.1e", value)
        }
    }
}
