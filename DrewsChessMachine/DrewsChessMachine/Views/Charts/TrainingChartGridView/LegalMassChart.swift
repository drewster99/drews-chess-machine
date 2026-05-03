import Charts
import SwiftUI

/// Legal mass sum tile — fraction of softmax mass on legal moves
/// at the probed position. Y axis is fixed to 0…1.
struct LegalMassChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

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
        case .hoveringNoData(let t):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            headerText = String(
                format: "t=%@ %.4f%%",
                TrainingChartGridView.formatElapsedAxis(t), v * 100
            )
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
            .chartYScale(domain: 0...1)
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
            .chartYAxis {
                AxisMarks(position: .leading, values: [0, 0.25, 0.5, 0.75, 1.0]) { value in
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
