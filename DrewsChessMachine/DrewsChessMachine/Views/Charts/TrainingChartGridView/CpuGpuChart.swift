import Charts
import SwiftUI

/// Two-series mini chart that renders CPU% and GPU% on the same tile.
/// Combining the two reclaims a chart slot that previously held one
/// series alone, and the visual relationship between CPU and GPU
/// utilization is more useful than either curve in isolation —
/// e.g., GPU utilization that drops while CPU stays pinned points to
/// a CPU-side bottleneck like data prep.
///
/// The header readout shows whichever series the cursor is closest
/// to, with both current values when nothing is hovered. The
/// per-series colors match the legend order: CPU blue, GPU indigo
/// (the same colors the standalone tiles previously used).
struct CpuGpuChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    private static let cpuColor: Color = .blue
    private static let gpuColor: Color = .indigo

    var body: some View {
        let cpuReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.cpuPercent },
            bucketWidthSec: context.bucketWidthSec
        )
        let gpuReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.gpuBusyPercent },
            bucketWidthSec: context.bucketWidthSec
        )
        let headerText = Self.headerText(
            cpuReadout: cpuReadout,
            gpuReadout: gpuReadout,
            buckets: buckets
        )
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "CPU / GPU", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("CPU", b.cpuPercent?.max ?? .nan),
                        series: .value("series", "CPU")
                    )
                    .foregroundStyle(Self.cpuColor)
                }
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("GPU", b.gpuBusyPercent?.max ?? .nan),
                        series: .value("series", "GPU")
                    )
                    .foregroundStyle(Self.gpuColor)
                }
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let v) = cpuReadout {
                    PointMark(x: .value("Time", t), y: .value("CPU", v))
                        .foregroundStyle(Self.cpuColor)
                        .symbolSize(40)
                }
                if case .hoveringWithData(let t, let v) = gpuReadout {
                    PointMark(x: .value("Time", t), y: .value("GPU", v))
                        .foregroundStyle(Self.gpuColor)
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

    /// Build the header readout. Three cases:
    ///   - not hovering → "CPU N% / GPU M%" using the latest non-nil
    ///     bucket value for each series.
    ///   - hovering, both have data → "CPU N% / GPU M%" at hover.
    ///   - hovering, neither bucket has any data on this tile →
    ///     "— no data".
    /// Mixed cases (one series has data, the other doesn't) fall
    /// through to "CPU N% / GPU --" or vice versa.
    private static func headerText(
        cpuReadout: TrainingChartGridView.HoverReadout,
        gpuReadout: TrainingChartGridView.HoverReadout,
        buckets: [TrainingBucket]
    ) -> String {
        func value(for readout: TrainingChartGridView.HoverReadout, lastBucketValue: Double?) -> String {
            switch readout {
            case .notHovering:
                if let v = lastBucketValue {
                    return String(Int(v))
                }
                return "--"
            case .hoveringNoData:
                return "--"
            case .hoveringWithData(_, let v):
                return String(Int(v))
            }
        }
        let cpuValue = value(
            for: cpuReadout,
            lastBucketValue: buckets.last.flatMap { $0.cpuPercent?.max }
        )
        let gpuValue = value(
            for: gpuReadout,
            lastBucketValue: buckets.last.flatMap { $0.gpuBusyPercent?.max }
        )
        if cpuValue == "--" && gpuValue == "--" {
            switch cpuReadout {
            case .hoveringNoData, .hoveringWithData:
                return "— no data"
            case .notHovering:
                return "--"
            }
        }
        return "CPU \(cpuValue)% / GPU \(gpuValue)%"
    }
}
