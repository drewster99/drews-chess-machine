import Charts
import SwiftUI

/// Upper-left tile (replaces the legacy "Loss (pLoss + vLoss)"
/// total-loss sparkgraph). Plots two outcome-partitioned series:
/// the policy loss restricted to win-outcome batch positions
/// (`pLossWin`) and to loss-outcome positions (`pLossLoss`).
/// The total-loss curve is ambiguous because outcome-weighted CE
/// flips sign with z; splitting by outcome makes both lines
/// individually interpretable.
struct PolicyLossSplitChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

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
        return VStack(alignment: .leading, spacing: 1) {
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
        .chartCard()
    }
}
