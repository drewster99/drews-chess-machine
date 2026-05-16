import Foundation

/// What a `FastLineChart`'s header closure receives when it is asked
/// to render the upper-right header value. `hoveredX == nil` means
/// no chart in the shared group is under the cursor — caller should
/// typically render the "latest" value or a placeholder.
public struct FastChartHoverContext: Sendable, Equatable {
    public let hoveredX: Double?
    public let xDomain: ClosedRange<Double>
    public let yDomain: ClosedRange<Double>

    public init(
        hoveredX: Double?,
        xDomain: ClosedRange<Double>,
        yDomain: ClosedRange<Double>
    ) {
        self.hoveredX = hoveredX
        self.xDomain = xDomain
        self.yDomain = yDomain
    }
}
