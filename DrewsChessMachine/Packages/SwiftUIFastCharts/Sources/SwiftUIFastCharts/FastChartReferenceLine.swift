import SwiftUI

/// A constant-Y horizontal reference line drawn on the chart. Used
/// for the gradient-clip ceiling on gNorm, the W/D/L draw-bias init
/// at 0.75, the replay-ratio target, and similar diagnostic lines.
public struct FastChartReferenceLine: Sendable, Equatable, Identifiable {
    public let id: String
    public let y: Double
    /// Short label rendered at the right edge of the line. Caller
    /// keeps it short — long labels are not wrapped.
    public let label: String?
    public let color: Color
    public let lineWidth: CGFloat
    public let dashed: Bool

    public init(
        id: String,
        y: Double,
        label: String? = nil,
        color: Color = Color.red.opacity(0.55),
        lineWidth: CGFloat = 1.0,
        dashed: Bool = true
    ) {
        self.id = id
        self.y = y
        self.label = label
        self.color = color
        self.lineWidth = lineWidth
        self.dashed = dashed
    }
}
