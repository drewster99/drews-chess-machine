import Charts
import SwiftUI

/// Combined memory tile — plots App-process RSS and GPU resident
/// memory on the same axes so the two halves of unified-memory
/// usage can be read at a glance. Header reads
/// `App X.X GB / GPU Y.Y GB / SYS Z GB` when a unified-memory total
/// is known, where `Z` is the host's physical memory in whole GB.
struct MemoryChart: View {
    let buckets: [TrainingBucket]
    let totalGB: Double?
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    private static let appColor: Color = .brown
    private static let gpuColor: Color = .teal

    var body: some View {
        let readout = hoverReadout()
        let headerText = headerText(for: readout)
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "RAM", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("Memory", b.appMemoryGB?.max ?? .nan)
                    )
                    .foregroundStyle(by: .value("Series", "App"))
                }
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("Memory", b.gpuMemoryGB?.max ?? .nan)
                    )
                    .foregroundStyle(by: .value("Series", "GPU"))
                }
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let app, let gpu) = readout {
                    if let app {
                        PointMark(x: .value("Time", t), y: .value("Memory", app))
                            .foregroundStyle(Self.appColor)
                            .symbolSize(40)
                    }
                    if let gpu {
                        PointMark(x: .value("Time", t), y: .value("Memory", gpu))
                            .foregroundStyle(Self.gpuColor)
                            .symbolSize(40)
                    }
                }
            }
            .chartForegroundStyleScale([
                "App": Self.appColor,
                "GPU": Self.gpuColor
            ])
            .chartLegend(.hidden)
            .modifier(StandardTimeSeriesChartModifiers(
                context: context,
                scrollX: $scrollX,
                hoveredSec: $hoveredSec
            ))
        }
        .frame(height: 75)
        .chartCard()
    }

    private enum Readout {
        case notHovering
        case hoveringNoData(time: Double)
        case hoveringWithData(time: Double, app: Double?, gpu: Double?)
    }

    private func hoverReadout() -> Readout {
        guard let t = hoveredSec else { return .notHovering }
        let tolerance = Swift.max(
            TrainingChartGridView.hoverMatchToleranceSec,
            context.bucketWidthSec * 1.5
        )
        guard let bucket = TrainingChartGridView.nearestTrainingBucket(
            at: t, in: buckets, tolerance: tolerance
        ) else {
            return .hoveringNoData(time: t)
        }
        let app = bucket.appMemoryGB?.max
        let gpu = bucket.gpuMemoryGB?.max
        if app == nil && gpu == nil {
            return .hoveringNoData(time: t)
        }
        return .hoveringWithData(time: bucket.elapsedSec, app: app, gpu: gpu)
    }

    private func headerText(for readout: Readout) -> String {
        switch readout {
        case .notHovering:
            let app = buckets.last?.appMemoryGB?.max
            let gpu = buckets.last?.gpuMemoryGB?.max
            return Self.formatHeader(app: app, gpu: gpu, totalGB: totalGB)
        case .hoveringNoData:
            return "— no data"
        case .hoveringWithData(_, let app, let gpu):
            return Self.formatHeader(app: app, gpu: gpu, totalGB: totalGB)
        }
    }

    private static func formatHeader(
        app: Double?,
        gpu: Double?,
        totalGB: Double?
    ) -> String {
        let appStr = app.map { String(format: "%.1f", $0) } ?? "--"
        let gpuStr = gpu.map { String(format: "%.1f", $0) } ?? "--"
        if let totalGB, totalGB > 0 {
            let sysStr = String(format: "%.0f", totalGB)
            return "App \(appStr) GB / GPU \(gpuStr) GB / SYS \(sysStr) GB"
        } else {
            return "App \(appStr) GB / GPU \(gpuStr) GB"
        }
    }
}
