import CoreGraphics
import Foundation

/// Pure-function bucket min/max decimator. Takes raw point arrays
/// and produces chart-ready bucket arrays at roughly the requested
/// resolution. Caller owns the input lifetime; the decimator does
/// not retain anything between calls.
///
/// Decimation bucket width is `(xDomain.upperBound - xDomain.lowerBound) / maxBucketCount`.
/// Each bucket holds the `min` and `max` of every raw point whose
/// `x` lands in `[bucketStart, bucketEnd)`. The bucket's `x` is
/// fixed to the bucket's *midpoint* in X — gives the chart's stroke
/// a stable left-to-right position regardless of how raw points are
/// distributed within the bucket.
///
/// NaN-y points are skipped (they do not contribute to min/max).
/// Empty buckets are skipped (no entry produced) — the chart's path
/// builder then naturally breaks the stroke across the gap.
public struct FastChartDecimator: Sendable {

    public init() {}

    public func decimate(
        rawSeries: [FastChartRawSeries],
        xDomain: ClosedRange<Double>,
        maxBucketCount: Int
    ) -> [FastChartSeries] {
        guard maxBucketCount > 0 else { return [] }
        let span = xDomain.upperBound - xDomain.lowerBound
        guard span > 0 else { return [] }
        let bucketWidth = span / Double(maxBucketCount)
        return rawSeries.map { raw in
            let buckets = bucketize(
                points: raw.points,
                xDomain: xDomain,
                bucketWidth: bucketWidth,
                bucketCount: maxBucketCount
            )
            return FastChartSeries(
                id: raw.id,
                color: raw.color,
                lineWidth: raw.lineWidth,
                interpolation: raw.interpolation,
                data: .buckets(buckets),
                showMinMaxBand: raw.showMinMaxBand
            )
        }
    }

    private func bucketize(
        points: [CGPoint],
        xDomain: ClosedRange<Double>,
        bucketWidth: Double,
        bucketCount: Int
    ) -> [FastChartBucket] {
        guard !points.isEmpty else { return [] }
        // Pre-allocate one mutable accumulator per bucket index; we
        // emit only the populated ones at the end. Inverted floats
        // mark the "still empty" state so the first valid sample
        // always wins both min and max.
        var minY = [Double](repeating: .infinity, count: bucketCount)
        var maxY = [Double](repeating: -.infinity, count: bucketCount)
        var hasAny = [Bool](repeating: false, count: bucketCount)

        let lower = xDomain.lowerBound
        let upper = xDomain.upperBound
        for p in points {
            let x = Double(p.x)
            if x < lower || x >= upper { continue }
            let y = Double(p.y)
            if !y.isFinite { continue }
            let idx = Swift.min(
                bucketCount - 1,
                Swift.max(0, Int((x - lower) / bucketWidth))
            )
            if y < minY[idx] { minY[idx] = y }
            if y > maxY[idx] { maxY[idx] = y }
            hasAny[idx] = true
        }

        var out: [FastChartBucket] = []
        out.reserveCapacity(bucketCount)
        var emittedId = 0
        for i in 0..<bucketCount {
            if !hasAny[i] { continue }
            let bucketX = lower + (Double(i) + 0.5) * bucketWidth
            out.append(FastChartBucket(
                id: emittedId,
                x: bucketX,
                yMin: minY[i],
                yMax: maxY[i]
            ))
            emittedId += 1
        }
        return out
    }
}
