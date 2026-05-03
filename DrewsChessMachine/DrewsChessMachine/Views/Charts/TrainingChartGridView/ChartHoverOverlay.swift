import Charts
import SwiftUI

/// Transparent overlay that captures mouse position on a time-series
/// chart and pipes the converted elapsed-second into the shared
/// `hoveredSec` binding. Pulled out of every chart's body so each
/// chart's hover wiring is one line.
struct ChartHoverOverlay: View {
    let proxy: ChartProxy
    @Binding var hoveredSec: Double?

    var body: some View {
        GeometryReader { geo in
            Rectangle()
                .fill(Color.clear)
                .contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let point):
                        let origin = (proxy.plotFrame.map { geo[$0].origin } ?? .zero)
                        let xInPlot = point.x - origin.x
                        if let sec: Double = proxy.value(atX: xInPlot) {
                            if sec < 0 {
                                if hoveredSec != nil { hoveredSec = nil }
                                return
                            }
                            if hoveredSec != sec {
                                hoveredSec = sec
                            }
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
