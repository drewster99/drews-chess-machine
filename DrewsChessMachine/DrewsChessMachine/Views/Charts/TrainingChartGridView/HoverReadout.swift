import Foundation

extension TrainingChartGridView {
    /// Three-way result of a per-chart hover query.
    enum HoverReadout {
        /// The cursor isn't over any time-series chart right now.
        case notHovering
        /// The cursor IS over a chart, but no bucket is within tolerance
        /// (or the bucket has no value for this series). Carries the
        /// raw hovered time so the header can still display
        /// "t=M:SS — no data".
        case hoveringNoData(hoveredTime: Double)
        /// The cursor is over a chart and the nearest bucket has a
        /// value for this series. Carries the bucket's anchor time
        /// and the bucket's representative value (range max).
        case hoveringWithData(sampleTime: Double, value: Double)
    }

    /// Linear-scan nearest-bucket lookup across the visible-window
    /// bucket array. Returns `nil` if the array is empty OR if the
    /// nearest bucket's anchor is farther than `tolerance` seconds
    /// from `t`. Tolerance is sized by the caller — typically
    /// `max(hoverMatchToleranceSec, 1.5 * bucketWidthSec)` so wide
    /// zoom levels still register hover correctly.
    static func nearestTrainingBucket(
        at t: Double,
        in buckets: [TrainingBucket],
        tolerance: Double
    ) -> TrainingBucket? {
        guard !buckets.isEmpty else { return nil }
        var best = buckets[0]
        var bestDist = Swift.abs(best.elapsedSec - t)
        for b in buckets.dropFirst() {
            let d = Swift.abs(b.elapsedSec - t)
            if d < bestDist { best = b; bestDist = d }
        }
        return bestDist <= tolerance ? best : nil
    }

    static func nearestProgressBucket(
        at t: Double,
        in buckets: [ProgressRateBucket],
        tolerance: Double
    ) -> ProgressRateBucket? {
        guard !buckets.isEmpty else { return nil }
        var best = buckets[0]
        var bestDist = Swift.abs(best.elapsedSec - t)
        for b in buckets.dropFirst() {
            let d = Swift.abs(b.elapsedSec - t)
            if d < bestDist { best = b; bestDist = d }
        }
        return bestDist <= tolerance ? best : nil
    }

    /// Resolve the hover state for one numeric series on a chart. The
    /// per-chart subview reads the relevant `ChartBucketRange?` field
    /// off the nearest bucket and converts it into the three-way
    /// `HoverReadout`. We use the bucket range's `max` as the
    /// representative value (preserves spike visibility — same logic
    /// the chart's line marks use).
    static func hoverReadoutTraining(
        hoveredSec: Double?,
        buckets: [TrainingBucket],
        accessor: (TrainingBucket) -> ChartBucketRange?,
        bucketWidthSec: Double
    ) -> HoverReadout {
        guard let t = hoveredSec else { return .notHovering }
        let tolerance = Swift.max(
            TrainingChartGridView.hoverMatchToleranceSec,
            bucketWidthSec * 1.5
        )
        guard let bucket = nearestTrainingBucket(
            at: t, in: buckets, tolerance: tolerance
        ) else {
            return .hoveringNoData(hoveredTime: t)
        }
        guard let range = accessor(bucket) else {
            return .hoveringNoData(hoveredTime: t)
        }
        return .hoveringWithData(sampleTime: bucket.elapsedSec, value: range.max)
    }
}
