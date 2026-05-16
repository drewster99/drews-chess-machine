import Foundation

/// One pre-decimated chart bucket — carries its X anchor (typically
/// the bucket's representative time) and the min/max envelope of all
/// raw samples that landed in it. The chart strokes a line through
/// (`x`, `yMax`) — preserving spike visibility, mirroring what the
/// DCM source charts do — and optionally fills a faint vertical span
/// from `yMin` to `yMax` behind that line.
///
/// `yMin == yMax` is legal (single-sample bucket). NaN in either
/// field means "this bucket has no value for this series" and the
/// chart will break the line across the gap.
public struct FastChartBucket: Sendable, Equatable, Identifiable {
    public let id: Int
    public let x: Double
    public let yMin: Double
    public let yMax: Double

    public init(id: Int, x: Double, yMin: Double, yMax: Double) {
        self.id = id
        self.x = x
        self.yMin = yMin
        self.yMax = yMax
    }
}
