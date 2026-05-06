import Charts
import SwiftUI

/// Legal mass sum tile — fraction of softmax mass on legal moves
/// at the probed position. Y axis is tiered to give early-training
/// signal more vertical room: top is 0.50 while the session's
/// running maximum stays below 0.25, expands to 0.75 when the max
/// crosses 0.25, and to 1.00 once the max crosses 0.50. The
/// bottom always stays at 0 so the absolute level is still
/// readable. Keying off the session-wide max (not the visible
/// window's max) means scrolling or zooming never flickers the
/// scale, and the tier monotonically expands as training improves.
struct LegalMassChart: View {
    let buckets: [TrainingBucket]
    /// Session-wide running max of `rollingLegalMass` (sample
    /// granularity, not bucket envelope). Drives the tier choice.
    let allTimeMax: Double
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    /// Tiered Y-axis top, keyed off the session-wide running max.
    private var yAxisTop: Double {
        if allTimeMax >= 0.5 { return 1.0 }
        if allTimeMax >= 0.25 { return 0.75 }
        return 0.5
    }

    /// Quartile ticks 0, 0.25, … up through the current top.
    private var yAxisTicks: [Double] {
        let top = yAxisTop
        var ticks: [Double] = [0]
        var t = 0.25
        while t <= top + 1e-9 {
            ticks.append(t)
            t += 0.25
        }
        return ticks
    }

    var body: some View {
        let readout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.legalMass },
            bucketWidthSec: context.bucketWidthSec
        )
        let headerText: String
        switch readout {
        case .notHovering:
            if let v = buckets.last?.legalMass?.max {
                headerText = String(format: "%.4f%%", v * 100)
            } else {
                headerText = "--"
            }
        case .hoveringNoData:
            headerText = "— no data"
        case .hoveringWithData(_, let v):
            headerText = String(format: "%.4f%%", v * 100)
        }
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "Legal mass sum", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("Legal mass", b.legalMass?.max ?? .nan)
                    )
                    .foregroundStyle(.cyan)
                }
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let v) = readout {
                    PointMark(x: .value("Time", t), y: .value("Legal mass", v))
                        .foregroundStyle(.cyan)
                        .symbolSize(40)
                }
            }
            .chartYScale(domain: 0...yAxisTop)
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
            .chartYAxis {
                AxisMarks(position: .leading, values: yAxisTicks) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.2f", v))
                                .font(.system(size: 7))
                                .monospacedDigit()
                        }
                    }
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
}
