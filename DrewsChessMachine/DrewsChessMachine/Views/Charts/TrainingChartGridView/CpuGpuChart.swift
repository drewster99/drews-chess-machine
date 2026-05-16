import SwiftUI
import SwiftUIFastCharts

/// Two-series mini chart that renders CPU% and GPU% on the same tile.
/// Combining the two reclaims a chart slot that previously held one
/// series alone, and the visual relationship is more useful than
/// either curve in isolation.
struct CpuGpuChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    private static let cpuColor: Color = .blue
    private static let gpuColor: Color = .indigo

    var body: some View {
        FastLineChart(
            title: "CPU / GPU",
            titleHelp: AttributedString("""
                CPU usage % across all cores (blue) and GPU busy % (indigo). GPU dropping while CPU \
                stays pinned points to a CPU-side bottleneck such as data prep or replay sampling; \
                both pinned near 100 % means you're saturating the machine.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...yMaxObserved(),
            series: [
                FastChartSeries(
                    id: "CPU",
                    color: Self.cpuColor,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.cpuPercent?.min ?? .nan,
                            yMax: b.cpuPercent?.max ?? .nan
                        )
                    })
                ),
                FastChartSeries(
                    id: "GPU",
                    color: Self.gpuColor,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.gpuBusyPercent?.min ?? .nan,
                            yMax: b.gpuBusyPercent?.max ?? .nan
                        )
                    })
                )
            ],
            legend: .off,
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func yMaxObserved() -> Double {
        let cpu = buckets.compactMap { $0.cpuPercent?.max }.max() ?? 0
        let gpu = buckets.compactMap { $0.gpuBusyPercent?.max }.max() ?? 0
        return Swift.max(Swift.max(cpu, gpu) * 1.1, 10)
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let cpu: Double?
        let gpu: Double?
        let isHovering = hoveredX != nil
        if let t = hoveredX {
            if let b = nearest(at: t) {
                cpu = b.cpuPercent?.max
                gpu = b.gpuBusyPercent?.max
            } else {
                cpu = nil
                gpu = nil
            }
        } else {
            cpu = buckets.last?.cpuPercent?.max
            gpu = buckets.last?.gpuBusyPercent?.max
        }
        if isHovering && cpu == nil && gpu == nil {
            return AttributedString("— no data")
        }
        if cpu == nil && gpu == nil {
            return AttributedString("--")
        }
        let cpuStr = cpu.map { String(Int($0)) } ?? "--"
        let gpuStr = gpu.map { String(Int($0)) } ?? "--"
        var out = AttributedString("")
        var cpuPart = AttributedString("CPU \(cpuStr)%")
        cpuPart.foregroundColor = Self.cpuColor
        out.append(cpuPart)
        out.append(AttributedString(" / "))
        var gpuPart = AttributedString("GPU \(gpuStr)%")
        gpuPart.foregroundColor = Self.gpuColor
        out.append(gpuPart)
        return out
    }

    private func nearest(at t: Double) -> TrainingBucket? {
        TrainingChartGridView.nearestTrainingBucket(
            at: t,
            in: buckets,
            tolerance: Swift.max(
                TrainingChartGridView.hoverMatchToleranceSec,
                bucketWidthSec * 1.5
            )
        )
    }
}
