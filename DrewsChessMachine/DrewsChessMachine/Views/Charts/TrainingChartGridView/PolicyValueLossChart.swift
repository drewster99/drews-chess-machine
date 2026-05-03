import Charts
import SwiftUI

/// Combo chart showing pLoss (policy loss, orange) and vLoss
/// (value loss, cyan) on a shared Y axis.
struct PolicyValueLossChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

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
        return VStack(alignment: .leading, spacing: 1) {
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
        .chartCard()
    }
}
