import Charts
import SwiftUI

/// Power / thermal step-trace chart (categorical).
struct PowerThermalChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    var body: some View {
        let readout = hoverReadout()
        let headerText = headerText(for: readout)
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "Power / thermal", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value(
                            "Power",
                            b.lowPowerMode.map { $0 ? 1.0 : 0.0 } ?? .nan
                        )
                    )
                    .foregroundStyle(by: .value("Series", "Low power"))
                    .interpolationMethod(.stepEnd)
                }
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value(
                            "Thermal",
                            b.thermalState.map { Self.thermalY($0) } ?? .nan
                        )
                    )
                    .foregroundStyle(by: .value("Series", "Thermal"))
                    .interpolationMethod(.stepEnd)
                }
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
            }
            .chartForegroundStyleScale([
                "Low power": Color.gray,
                "Thermal": Color.orange
            ])
            .chartLegend(.hidden)
            .chartYScale(domain: 0...5.5)
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
            .chartYAxis {
                AxisMarks(position: .leading, values: [0, 1, 2, 3, 4, 5]) { _ in
                    AxisGridLine()
                }
            }
            .chartXScale(domain: context.timeSeriesXDomain)
            .chartScrollableAxes(.horizontal)
            .chartXVisibleDomain(length: context.visibleDomainSec)
            .chartScrollPosition(x: $scrollX)
            .chartOverlay { proxy in
                ChartHoverOverlay(proxy: proxy, hoveredSec: $hoveredSec)
            }
        }
        .frame(height: 75)
        .chartCard()
    }

    private enum Readout {
        case notHovering
        case hoveringNoData(time: Double)
        case hoveringWithData(time: Double, lowPower: Bool, thermal: ProcessInfo.ThermalState)
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
        guard let lp = bucket.lowPowerMode, let ts = bucket.thermalState else {
            return .hoveringNoData(time: t)
        }
        return .hoveringWithData(time: bucket.elapsedSec, lowPower: lp, thermal: ts)
    }

    private func headerText(for readout: Readout) -> String {
        switch readout {
        case .notHovering:
            if let latest = buckets.last {
                let powerStr = (latest.lowPowerMode ?? false) ? "on" : "off"
                let thermStr: String
                if let ts = latest.thermalState {
                    thermStr = Self.thermalStateName(ts)
                } else {
                    thermStr = "--"
                }
                return "lowpwr=\(powerStr)  thermal=\(thermStr)"
            } else {
                return "--"
            }
        case .hoveringNoData:
            return "— no data"
        case .hoveringWithData(_, let lp, let ts):
            let powerStr = lp ? "on" : "off"
            let thermStr = Self.thermalStateName(ts)
            return "lowpwr=\(powerStr)  thermal=\(thermStr)"
        }
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

    private static func thermalY(_ state: ProcessInfo.ThermalState) -> Double {
        Double(state.rawValue) + 2
    }
}
