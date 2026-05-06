import Charts
import SwiftUI

/// Generic single-series mini chart used by CPU%, GPU%, and gNorm.
struct MiniLineChart: View {
    let title: String
    let buckets: [TrainingBucket]
    let rangeAccessor: (TrainingBucket) -> ChartBucketRange?
    let unit: String
    let color: Color
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context
    var wholeNumber: Bool = false

    var body: some View {
        let readout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: rangeAccessor,
            bucketWidthSec: context.bucketWidthSec
        )
        let unitSuffix = unit.isEmpty ? "" : " \(unit)"
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = buckets.last.flatMap({ rangeAccessor($0)?.max }) {
                let valueStr = wholeNumber ? String(Int(v)) : TrainingChartGridView.compactLabel(v)
                headerText = "\(valueStr)\(unitSuffix)"
            } else {
                headerText = "--"
            }
        case .hoveringNoData:
            headerText = "— no data"
        case .hoveringWithData(_, let v):
            let valueStr = wholeNumber ? String(Int(v)) : TrainingChartGridView.compactLabel(v)
            headerText = "\(valueStr)\(unitSuffix)"
        }
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: title, value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value(title, rangeAccessor(b)?.max ?? .nan)
                    )
                    .foregroundStyle(color)
                }
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let v) = readout {
                    PointMark(x: .value("Time", t), y: .value(title, v))
                        .foregroundStyle(color)
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
