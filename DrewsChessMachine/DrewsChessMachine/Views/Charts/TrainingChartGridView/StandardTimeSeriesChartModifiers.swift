import Charts
import SwiftUI

/// X-axis + Y-axis + scroll-position modifier chain shared by the
/// still-on-SwiftCharts Arena tiles. Uses a visible-window-only X
/// domain (`scrollX...(scrollX + visibleDomainSec)`) to match the
/// migrated FastLineChart tiles — the prior 0...max(lastElapsed,
/// visibleDomainSec) caused the chart layout to re-distribute on
/// every data tick because the full-data domain kept growing.
struct StandardTimeSeriesChartModifiers: ViewModifier {
    let context: TrainingChartGridView.Context
    @Binding var scrollX: Double
    @Binding var hoveredSec: Double?

    func body(content: Content) -> some View {
        let lo = max(0, scrollX)
        let hi = lo + max(0.001, context.visibleDomainSec)
        return content
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
            .chartXScale(domain: lo...hi)
            .chartOverlay { proxy in
                ChartHoverOverlay(proxy: proxy, hoveredSec: $hoveredSec)
            }
    }
}
