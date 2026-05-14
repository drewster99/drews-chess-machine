import Charts
import SwiftUI

/// Diversity histogram tile (categorical X axis).
struct DiversityHistogramChart: View {
    let bars: [DiversityHistogramBar]

    @State private var hoveredHistogramBarID: Int?

    // Eleven-step green→red gradient aligned with
    // `GameDiversityTracker.histogramLabels`. Buckets 0–5 (one ply
    // each) cover the healthy steady-state region and step through
    // greens / yellow-greens / yellow. Buckets 6–10 (the widening
    // tail) progress through orange and red into very dark red so
    // a deep policy-share regime visually pops.
    private static let bucketColors: [Color] = [
        Color(hue: 0.33, saturation: 0.85, brightness: 0.80),  // 0
        Color(hue: 0.28, saturation: 0.85, brightness: 0.82),  // 1
        Color(hue: 0.23, saturation: 0.85, brightness: 0.85),  // 2
        Color(hue: 0.18, saturation: 0.85, brightness: 0.88),  // 3
        Color(hue: 0.14, saturation: 0.90, brightness: 0.92),  // 4
        Color(hue: 0.11, saturation: 0.95, brightness: 0.94),  // 5
        Color(hue: 0.08, saturation: 0.95, brightness: 0.90),  // 6-7
        Color(hue: 0.05, saturation: 1.00, brightness: 0.85),  // 8-10
        Color(hue: 0.02, saturation: 1.00, brightness: 0.78),  // 11-20
        Color(hue: 0.00, saturation: 1.00, brightness: 0.65),  // 21-40
        Color(red: 0.40, green: 0, blue: 0)                    // 41+
    ]

    var body: some View {
        let total = bars.reduce(0) { $0 + $1.count }
        let maxCount = bars.map(\.count).max() ?? 0
        let headerText: String
        if let hoveredID = hoveredHistogramBarID,
           let bar = bars.first(where: { $0.id == hoveredID }) {
            headerText = "\(bar.label) plies: \(bar.count)"
        } else if total > 0 {
            headerText = "\(total) games"
        } else {
            headerText = "--"
        }
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "Longest move prefix", value: headerText)
            Chart(bars) { bar in
                BarMark(
                    x: .value("Bucket", bar.label),
                    y: .value("Count", bar.count)
                )
                .foregroundStyle(
                    Self.bucketColors.indices.contains(bar.id)
                        ? Self.bucketColors[bar.id]
                        : Color.gray
                )
                .opacity(hoveredHistogramBarID == nil || hoveredHistogramBarID == bar.id ? 1.0 : 0.4)
            }
            .chartYScale(domain: 0...(maxCount > 0 ? Int(Double(maxCount) * 1.1) + 1 : 1))
            .chartXAxis {
                AxisMarks(preset: .aligned, values: .automatic) { value in
                    AxisValueLabel {
                        if let label = value.as(String.self) {
                            Text(label)
                                .font(.system(size: 6))
                                .monospacedDigit()
                        }
                    }
                }
            }
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
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let point):
                                let origin = (proxy.plotFrame.map { geo[$0].origin } ?? .zero)
                                let xInPlot = point.x - origin.x
                                if let label: String = proxy.value(atX: xInPlot),
                                   let match = bars.first(where: { $0.label == label }) {
                                    if hoveredHistogramBarID != match.id {
                                        hoveredHistogramBarID = match.id
                                    }
                                } else if hoveredHistogramBarID != nil {
                                    hoveredHistogramBarID = nil
                                }
                            case .ended:
                                if hoveredHistogramBarID != nil {
                                    hoveredHistogramBarID = nil
                                }
                            }
                        }
                }
            }
        }
        .frame(height: 75)
        .chartCard()
    }
}
