import Charts
import SwiftUI

/// Replay ratio tile — line + dashed reference at the user target.
struct ReplayRatioChart: View {
    let buckets: [TrainingBucket]
    let target: Double
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    var body: some View {
        let readout = TrainingChartGridView.hoverReadoutTraining(
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
        return VStack(alignment: .leading, spacing: 1) {
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
        .chartCard()
    }
}
