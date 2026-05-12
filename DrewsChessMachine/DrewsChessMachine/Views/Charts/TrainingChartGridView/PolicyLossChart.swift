import Charts
import SwiftUI

/// Single-series tile for the outcome-weighted policy cross-entropy
/// (`pLoss`). Was "pLoss + vLoss" on a shared Y axis until the WDL
/// switch made the two unsharable: post-2026-05-12 `vLoss` is
/// categorical-CE-scale (~[0, ln 3 ≈ 1.10], shrinking toward ~0.5),
/// while `pLoss` is outcome-weighted CE that is unbounded on both
/// sides and routinely negative — crushing one against the other on
/// one axis. `vLoss` now lives on its own tile in the value-head
/// row; the win/loss-partitioned policy loss is on `PolicyLossSplitChart`.
struct PolicyLossChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    var body: some View {
        // Per-series hover readout — when the cursor moves, read the
        // bucket that corresponds to the hovered time rather than the
        // most-recent bucket. Without this the crosshair RuleMark
        // moves but the header value sits stuck at the last sample.
        let readout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.policyLoss },
            bucketWidthSec: context.bucketWidthSec
        )
        let headerText: String
        switch readout {
        case .notHovering:
            headerText = (buckets.last?.policyLoss?.max)
                .map { String(format: "%+.4f", $0) } ?? "--"
        case .hoveringNoData:
            headerText = "— no data"
        case .hoveringWithData(_, let v):
            headerText = String(format: "%+.4f", v)
        }
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "pLoss (outcome-weighted CE)", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("pLoss", b.policyLoss?.max ?? .nan)
                    )
                    .foregroundStyle(Color.orange)
                }
                RuleMark(y: .value("Zero", 0.0))
                    .foregroundStyle(Color.gray.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let v) = readout {
                    PointMark(x: .value("Time", t), y: .value("pLoss", v))
                        .foregroundStyle(Color.orange)
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
