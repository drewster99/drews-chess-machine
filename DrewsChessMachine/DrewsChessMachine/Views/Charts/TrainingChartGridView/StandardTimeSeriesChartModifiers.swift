import Charts
import SwiftUI

/// X-axis + scroll-position modifier chain shared by the still-on-
/// SwiftCharts Arena tiles. Uses a visible-window-only X domain
/// (`scrollX...(scrollX + visibleDomainSec)`) to match the migrated
/// FastLineChart tiles — the prior 0...max(lastElapsed,
/// visibleDomainSec) caused the chart layout to re-distribute on
/// every data tick because the full-data domain kept growing.
///
/// Y-axis is owned by the caller — each arena tile has its own custom
/// Y ticks (0.40–0.60 stepping for win %, 0.00–1.05 for activity bars)
/// that shouldn't be clobbered by a generic .automatic ladder.
struct StandardTimeSeriesChartModifiers: ViewModifier {
    let context: TrainingChartGridView.Context
    @Binding var scrollX: Double
    @Binding var hoveredSec: Double?

    func body(content: Content) -> some View {
        let lo = max(0, scrollX)
        let hi = lo + max(0.001, context.visibleDomainSec)
        return content
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { _ in AxisGridLine() } }
            .chartXScale(domain: lo...hi)
            .chartOverlay { proxy in
                ChartHoverOverlay(proxy: proxy, xDomain: lo...hi, hoveredSec: $hoveredSec)
            }
    }
}
