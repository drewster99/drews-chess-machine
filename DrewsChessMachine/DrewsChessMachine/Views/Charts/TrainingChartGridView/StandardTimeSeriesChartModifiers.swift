import Charts
import SwiftUI

/// Bundles the standard X-axis + Y-axis + scroll modifier chain
/// shared by every line-series tile. Pulled into a `ViewModifier`
/// so each chart subview's body stays focused on its marks rather
/// than a 7-line modifier chain.
struct StandardTimeSeriesChartModifiers: ViewModifier {
    let context: TrainingChartGridView.Context
    @Binding var scrollX: Double
    @Binding var hoveredSec: Double?

    func body(content: Content) -> some View {
        content
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(TrainingChartGridView.compactLabel(v))
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
}
