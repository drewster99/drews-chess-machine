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
        // Per-series hover readouts — when the cursor moves, read the
        // bucket that corresponds to the hovered time rather than the
        // most-recent bucket. Without this the crosshair RuleMark
        // moves but the header value sits stuck at the last sample.
        let winReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.policyLossWin },
            bucketWidthSec: context.bucketWidthSec
        )
        let lossReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.policyLossLoss },
            bucketWidthSec: context.bucketWidthSec
        )
        func value(
            for readout: TrainingChartGridView.HoverReadout,
            lastBucketValue: Double?
        ) -> String {
            switch readout {
            case .notHovering:
                return lastBucketValue.map { String(format: "%+.4f", $0) } ?? "--"
            case .hoveringNoData:
                return "--"
            case .hoveringWithData(_, let v):
                return String(format: "%+.4f", v)
            }
        }
        let winStr = value(
            for: winReadout,
            lastBucketValue: buckets.last?.policyLossWin?.max
        )
        let lossStr = value(
            for: lossReadout,
            lastBucketValue: buckets.last?.policyLossLoss?.max
        )
        let headerText: String
        if winStr == "--" && lossStr == "--" {
            switch winReadout {
            case .hoveringNoData, .hoveringWithData:
                headerText = "— no data"
            case .notHovering:
                headerText = "--"
            }
        } else {
            headerText = "win \(winStr) / loss \(lossStr)"
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
                if case .hoveringWithData(let t, let v) = winReadout {
                    PointMark(x: .value("Time", t), y: .value("pLoss", v))
                        .foregroundStyle(Color.green)
                        .symbolSize(40)
                }
                if case .hoveringWithData(let t, let v) = lossReadout {
                    PointMark(x: .value("Time", t), y: .value("pLoss", v))
                        .foregroundStyle(Color.red)
                        .symbolSize(40)
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
