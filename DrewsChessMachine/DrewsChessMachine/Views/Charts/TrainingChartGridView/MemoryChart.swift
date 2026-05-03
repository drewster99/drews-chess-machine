import Charts
import SwiftUI

/// Memory tile variant: same chart shape as `MiniLineChart` but the
/// header reads `X.X GB / Y.Y GB (Z.Z%)`.
struct MemoryChart: View {
    let title: String
    let buckets: [TrainingBucket]
    let rangeAccessor: (TrainingBucket) -> ChartBucketRange?
    let totalGB: Double?
    let color: Color
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    var body: some View {
        let readout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: rangeAccessor,
            bucketWidthSec: context.bucketWidthSec
        )
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = buckets.last.flatMap({ rangeAccessor($0)?.max }) {
                headerText = Self.formatMemoryHeader(usedGB: v, totalGB: totalGB)
            } else {
                headerText = "--"
            }
        case .hoveringNoData(let t):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) \(Self.formatMemoryHeader(usedGB: v, totalGB: totalGB))"
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

    private static func formatMemoryHeader(usedGB: Double, totalGB: Double?) -> String {
        guard let totalGB, totalGB > 0 else {
            return String(format: "%.1f GB", usedGB)
        }
        let pct = usedGB / totalGB * 100
        return String(format: "%.1f GB / %.0f GB (%.0f%%)", usedGB, totalGB, pct)
    }
}
