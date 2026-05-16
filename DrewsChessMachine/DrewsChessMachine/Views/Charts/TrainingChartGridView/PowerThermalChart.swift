import SwiftUI
import SwiftUIFastCharts

/// Power / thermal step-trace chart (categorical). Low-power is the
/// 0/1 binary at the bottom; thermal-state is the 2-5 ladder
/// (nominal/fair/serious/critical, mapped from `ThermalState.rawValue + 2`)
/// above it.
struct PowerThermalChart: View {
    let buckets: [TrainingBucket]
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double

    var body: some View {
        FastLineChart(
            title: "Power / thermal",
            titleHelp: AttributedString("""
                Two step traces: low-power mode (gray, 0 = off / 1 = on, at the bottom) and macOS \
                thermal state (orange, nominal = 2 / fair = 3 / serious = 4 / critical = 5). The \
                thermal state escalating means the system is starting to throttle — expect a \
                training rate dip on the Progress rate tile.
                """),
            group: group,
            xDomain: xDomain,
            yDomain: 0...5.5,
            series: [
                FastChartSeries(
                    id: "Low power",
                    color: .gray,
                    lineWidth: 1.5,
                    interpolation: .stepEnd,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: b.lowPowerMode.map { $0 ? 1.0 : 0.0 } ?? .nan,
                            yMax: b.lowPowerMode.map { $0 ? 1.0 : 0.0 } ?? .nan
                        )
                    })
                ),
                FastChartSeries(
                    id: "Thermal",
                    color: .orange,
                    lineWidth: 1.5,
                    interpolation: .stepEnd,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        let y = b.thermalState.map { Double($0.rawValue) + 2 } ?? .nan
                        return FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: y,
                            yMax: y
                        )
                    })
                )
            ],
            yTickValues: [0, 1, 2, 3, 4, 5],
            yLabelFormatter: { String(format: "%.0f", $0) },
            legend: .off,
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let lp: Bool?
        let ts: ProcessInfo.ThermalState?
        let isHovering = hoveredX != nil
        if let t = hoveredX {
            if let b = nearest(at: t) {
                lp = b.lowPowerMode
                ts = b.thermalState
            } else {
                lp = nil
                ts = nil
            }
        } else if let b = buckets.last {
            lp = b.lowPowerMode
            ts = b.thermalState
        } else {
            lp = nil
            ts = nil
        }
        if isHovering && lp == nil && ts == nil {
            return AttributedString("— no data")
        }
        if lp == nil && ts == nil {
            return AttributedString("--")
        }
        let powerStr = (lp ?? false) ? "on" : "off"
        let thermStr = ts.map(Self.thermalStateName) ?? "--"
        return AttributedString("lowpwr=\(powerStr)  thermal=\(thermStr)")
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

    private static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
