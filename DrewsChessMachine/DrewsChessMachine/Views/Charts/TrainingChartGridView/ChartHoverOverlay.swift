import Charts
import SwiftUI

/// Transparent overlay that captures mouse position on a time-series
/// chart and pipes the converted elapsed-second into the shared
/// `hoveredSec` binding. Pulled out of every chart's body so each
/// chart's hover wiring is one line.
///
/// Coordinate mapping is done manually from the caller-supplied
/// `xDomain` and the plot rect (via `proxy.plotFrame`), matching the
/// approach used by the migrated `FastLineChart` tiles. The earlier
/// reliance on `proxy.value(atX:)` produced positions wildly off the
/// caller's configured `chartXScale` when the chart's data range was
/// much narrower than the visible window, which made hover on the
/// arena tiles collapse to "the cursor is past the last point" almost
/// everywhere.
struct ChartHoverOverlay: View {
    let proxy: ChartProxy
    let xDomain: ClosedRange<Double>
    @Binding var hoveredSec: Double?

    var body: some View {
        GeometryReader { geo in
            Rectangle()
                .fill(Color.clear)
                .contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let point):
                        guard let frameAnchor = proxy.plotFrame else {
                            if hoveredSec != nil { hoveredSec = nil }
                            return
                        }
                        let plot = geo[frameAnchor]
                        guard plot.width > 0 else { return }
                        let xInPlot = point.x - plot.origin.x
                        if xInPlot < 0 || xInPlot > plot.width {
                            if hoveredSec != nil { hoveredSec = nil }
                            return
                        }
                        let normalized = Double(xInPlot / plot.width)
                        let span = xDomain.upperBound - xDomain.lowerBound
                        let dataX = xDomain.lowerBound + normalized * span
                        if hoveredSec != dataX {
                            hoveredSec = dataX
                        }
                    case .ended:
                        if hoveredSec != nil {
                            hoveredSec = nil
                        }
                    }
                }
        }
    }
}
