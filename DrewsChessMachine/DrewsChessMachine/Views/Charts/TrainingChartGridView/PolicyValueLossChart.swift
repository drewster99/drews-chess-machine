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
        // Per-series hover readouts — when the cursor moves, read the
        // bucket that corresponds to the hovered time rather than the
        // most-recent bucket. Without this the crosshair RuleMark
        // moves but the header value sits stuck at the last sample.
        let pReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.policyLoss },
            bucketWidthSec: context.bucketWidthSec
        )
        let vReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.valueLoss },
            bucketWidthSec: context.bucketWidthSec
        )
        func value(
            for readout: TrainingChartGridView.HoverReadout,
            lastBucketValue: Double?
        ) -> String {
            switch readout {
            case .notHovering:
                return lastBucketValue.map { String(format: "%.3f", $0) } ?? "--"
            case .hoveringNoData:
                return "--"
            case .hoveringWithData(_, let v):
                return String(format: "%.3f", v)
            }
        }
        let pStr = value(
            for: pReadout,
            lastBucketValue: buckets.last?.policyLoss?.max
        )
        let vStr = value(
            for: vReadout,
            lastBucketValue: buckets.last?.valueLoss?.max
        )
        let headerText: String
        if pStr == "--" && vStr == "--" {
            switch pReadout {
            case .hoveringNoData, .hoveringWithData:
                headerText = "— no data"
            case .notHovering:
                headerText = "--"
            }
        } else {
            headerText = "pLoss \(pStr) / vLoss \(vStr)"
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
                if case .hoveringWithData(let t, let v) = pReadout {
                    PointMark(x: .value("Time", t), y: .value("Loss", v))
                        .foregroundStyle(Color.orange)
                        .symbolSize(40)
                }
                if case .hoveringWithData(let t, let v) = vReadout {
                    PointMark(x: .value("Time", t), y: .value("Loss", v))
                        .foregroundStyle(Color.cyan)
                        .symbolSize(40)
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
