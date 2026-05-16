import SwiftUI
import SwiftUIFastCharts

/// Combined memory tile — plots App-process RSS and GPU resident
/// memory on the same axes so the two halves of unified-memory
/// usage can be read at a glance. Header reads
/// `App X.X GB / GPU Y.Y GB / SYS Z GB` when a unified-memory total
/// is known, where `Z` is the host's physical memory in whole GB.
struct MemoryChart: View {
    let buckets: [TrainingBucket]
    let totalGB: Double?
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    private static let appColor: Color = .brown
    private static let gpuColor: Color = .teal
    private static let sysColor: Color = .gray

    var body: some View {
        FastLineChart(
            title: "RAM",
            titleHelp: AttributedString("""
                App-process resident memory (App, brown) and GPU resident memory (GPU, teal), in GB. \
                SYS is the host's physical memory total — both halves of unified memory. App + GPU \
                creeping toward SYS means you're at risk of swapping.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...yMax(),
            series: [
                FastChartSeries(
                    id: "App",
                    color: Self.appColor,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.appMemoryGB?.min ?? .nan,
                            yMax: b.appMemoryGB?.max ?? .nan
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
                            yMin: b.gpuMemoryGB?.min ?? .nan,
                            yMax: b.gpuMemoryGB?.max ?? .nan
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

    private func yMax() -> Double {
        let appMax = buckets.compactMap { $0.appMemoryGB?.max }.max() ?? 0
        let gpuMax = buckets.compactMap { $0.gpuMemoryGB?.max }.max() ?? 0
        let dataMax = Swift.max(appMax, gpuMax)
        return Swift.max(dataMax * 1.1, 1.0)
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let app: Double?
        let gpu: Double?
        let isHovering = hoveredX != nil
        let inDataRange: Bool
        if let t = hoveredX {
            if let bucket = nearest(at: t) {
                app = bucket.appMemoryGB?.max
                gpu = bucket.gpuMemoryGB?.max
                inDataRange = (app != nil || gpu != nil)
            } else {
                app = nil
                gpu = nil
                inDataRange = false
            }
        } else {
            app = buckets.last?.appMemoryGB?.max
            gpu = buckets.last?.gpuMemoryGB?.max
            inDataRange = (app != nil || gpu != nil)
        }
        if isHovering && !inDataRange {
            return AttributedString("— no data")
        }
        return Self.buildAttributed(app: app, gpu: gpu, totalGB: totalGB)
    }

    private static func buildAttributed(
        app: Double?,
        gpu: Double?,
        totalGB: Double?
    ) -> AttributedString {
        let appStr = app.map { String(format: "%.1f", $0) } ?? "--"
        let gpuStr = gpu.map { String(format: "%.1f", $0) } ?? "--"
        var out = AttributedString("")
        var appPart = AttributedString("App \(appStr) GB")
        appPart.foregroundColor = appColor
        out.append(appPart)
        out.append(AttributedString(" / "))
        var gpuPart = AttributedString("GPU \(gpuStr) GB")
        gpuPart.foregroundColor = gpuColor
        out.append(gpuPart)
        if let totalGB, totalGB > 0 {
            let sysStr = String(format: "%.0f", totalGB)
            out.append(AttributedString(" / "))
            var sysPart = AttributedString("SYS \(sysStr) GB")
            sysPart.foregroundColor = sysColor
            out.append(sysPart)
        }
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

extension MemoryChart: Equatable {
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.bucketWidthSec == rhs.bucketWidthSec
            && lhs.totalGB == rhs.totalGB
            && lhs.xDomain == rhs.xDomain
            && lhs.group === rhs.group
            && lhs.buckets == rhs.buckets
    }
}
