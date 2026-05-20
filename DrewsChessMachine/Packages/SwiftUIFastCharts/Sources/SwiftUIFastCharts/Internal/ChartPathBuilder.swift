import CoreGraphics
import SwiftUI

/// Builds the `Path` for one series, walking only the visible X
/// range. NaN-y points (and bucket envelopes with NaN max) break the
/// stroke — i.e. the builder emits a fresh `move(to:)` on the next
/// valid point rather than drawing a line through the gap.
enum ChartPathBuilder {

    /// Visible-range slice for `[CGPoint]` series. Returns the index
    /// of the first point with `x >= xDomain.lowerBound - margin`
    /// (one point of margin so the stroke leading into the visible
    /// window is drawn) and the count of points to include past
    /// `xDomain.upperBound + margin`.
    static func visibleRange(
        in points: [CGPoint],
        xDomain: ClosedRange<Double>
    ) -> Range<Int> {
        guard !points.isEmpty else { return 0..<0 }
        let lo = lowerBound(points, key: { Double($0.x) }, target: xDomain.lowerBound)
        let hi = upperBound(points, key: { Double($0.x) }, target: xDomain.upperBound)
        // Pull one point back / forward so the leading and trailing
        // line segments cross the viewport edges instead of starting
        // exactly at the first/last visible sample.
        let lowerIdx = Swift.max(0, lo - 1)
        let upperIdx = Swift.min(points.count, hi + 1)
        return lowerIdx..<upperIdx
    }

    static func visibleRange(
        in buckets: [FastChartBucket],
        xDomain: ClosedRange<Double>
    ) -> Range<Int> {
        guard !buckets.isEmpty else { return 0..<0 }
        let lo = lowerBound(buckets, key: { $0.x }, target: xDomain.lowerBound)
        let hi = upperBound(buckets, key: { $0.x }, target: xDomain.upperBound)
        let lowerIdx = Swift.max(0, lo - 1)
        let upperIdx = Swift.min(buckets.count, hi + 1)
        return lowerIdx..<upperIdx
    }

    /// Stroke path for a `.points` series.
    static func strokePath(
        points: [CGPoint],
        visible: Range<Int>,
        interpolation: FastChartInterpolation,
        transform: ChartCoordTransform
    ) -> Path {
        var path = Path()
        var lastEmitted: CGPoint?
        for i in visible {
            let p = points[i]
            let y = Double(p.y)
            guard y.isFinite else {
                lastEmitted = nil
                continue
            }
            let view = transform.point(Double(p.x), y)
            switch interpolation {
            case .linear:
                if lastEmitted == nil {
                    path.move(to: view)
                } else {
                    path.addLine(to: view)
                }
            case .stepEnd:
                if let last = lastEmitted {
                    path.addLine(to: CGPoint(x: view.x, y: last.y))
                    path.addLine(to: view)
                } else {
                    path.move(to: view)
                }
            case .stepStart:
                if let last = lastEmitted {
                    path.addLine(to: CGPoint(x: last.x, y: view.y))
                    path.addLine(to: view)
                } else {
                    path.move(to: view)
                }
            }
            lastEmitted = view
        }
        return path
    }

    /// Stroke path for a `.buckets` series. Strokes through
    /// `(x, yMax)` per bucket — matches the DCM source charts'
    /// "use the bucket's max as the representative" rule, preserving
    /// spike visibility.
    static func strokePath(
        buckets: [FastChartBucket],
        visible: Range<Int>,
        interpolation: FastChartInterpolation,
        transform: ChartCoordTransform
    ) -> Path {
        var path = Path()
        var lastEmitted: CGPoint?
        for i in visible {
            let b = buckets[i]
            let y = b.yMax
            guard y.isFinite else {
                lastEmitted = nil
                continue
            }
            let view = transform.point(b.x, y)
            switch interpolation {
            case .linear:
                if lastEmitted == nil {
                    path.move(to: view)
                } else {
                    path.addLine(to: view)
                }
            case .stepEnd:
                if let last = lastEmitted {
                    path.addLine(to: CGPoint(x: view.x, y: last.y))
                    path.addLine(to: view)
                } else {
                    path.move(to: view)
                }
            case .stepStart:
                if let last = lastEmitted {
                    path.addLine(to: CGPoint(x: last.x, y: view.y))
                    path.addLine(to: view)
                } else {
                    path.move(to: view)
                }
            }
            lastEmitted = view
        }
        return path
    }

    /// Min/max-envelope band path for a `.buckets` series. Renders
    /// each bucket's `yMin → yMax` extent as a thin vertical line.
    /// Filled later as one stroke — much cheaper than a polygon.
    static func bandPath(
        buckets: [FastChartBucket],
        visible: Range<Int>,
        transform: ChartCoordTransform
    ) -> Path {
        var path = Path()
        for i in visible {
            let b = buckets[i]
            guard b.yMin.isFinite, b.yMax.isFinite, b.yMin < b.yMax else { continue }
            let top = transform.point(b.x, b.yMax)
            let bottom = transform.point(b.x, b.yMin)
            path.move(to: top)
            path.addLine(to: bottom)
        }
        return path
    }

    // MARK: - Binary search

    /// First index `i` with `key(array[i]) >= target`. Returns
    /// `array.count` if no such index exists.
    private static func lowerBound<T>(
        _ array: [T],
        key: (T) -> Double,
        target: Double
    ) -> Int {
        var lo = 0
        var hi = array.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if key(array[mid]) < target { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    /// First index `i` with `key(array[i]) > target`. Returns
    /// `array.count` if no such index exists.
    private static func upperBound<T>(
        _ array: [T],
        key: (T) -> Double,
        target: Double
    ) -> Int {
        var lo = 0
        var hi = array.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if key(array[mid]) <= target { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }
}
