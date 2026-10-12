import Foundation

// MARK: - Sampling the schedule for display

/// The schedule over a range of trainer steps, bucketed for a chart or a
/// table: per bucket the bounds at its first step and the minimum, maximum
/// and last value of the fed LR and momentum across its steps.
///
/// Min / max per bucket is what keeps the oscillation visible when one pixel
/// spans hundreds of steps: a plain every-Nth sample aliases the cosine and
/// can draw a flat line through a schedule that swings tenfold. The values
/// come from a caller-supplied function of the step — in the app, the fed
/// values (`LRMomentumCycleReadout`), so the picture is what the trainer is
/// fed, warmup and √batch included, with no formula of its own.
struct LRScheduleSample: Sendable, Equatable {
    struct Point: Sendable, Equatable {
        let learningRate: Double
        let momentum: Double
        let lrPeak: Double?
        let lrTrough: Double?
    }

    struct Bucket: Sendable, Equatable {
        let firstStep: Int
        let lastStep: Int
        let lrPeak: Double?
        let lrTrough: Double?
        let lrMin: Double
        let lrMax: Double
        let lrLast: Double
        let momentumMin: Double
        let momentumMax: Double
        let momentumLast: Double
    }

    let buckets: [Bucket]

    /// Samples every step in `range` (no step is skipped, so no extreme is
    /// missed) into at most `bucketCount` buckets of equal width (the last may
    /// be shorter). `isCancelled` is polled once per bucket, so an editor can
    /// abandon a stale sampling when the draft changes.
    static func sample(steps range: ClosedRange<Int>, bucketCount: Int,
                       isCancelled: () -> Bool = { false },
                       valueAt: (Int) -> Point) -> LRScheduleSample? {
        precondition(bucketCount > 0, "LRScheduleSample needs at least one bucket")
        let total = range.upperBound - range.lowerBound + 1
        let width = max(1, (total + bucketCount - 1) / bucketCount)
        var buckets: [Bucket] = []
        buckets.reserveCapacity(min(bucketCount, total))
        var first = range.lowerBound
        while first <= range.upperBound {
            if isCancelled() { return nil }
            let last = min(range.upperBound, first + width - 1)
            let head = valueAt(first)
            var lrMin = head.learningRate, lrMax = head.learningRate, lrLast = head.learningRate
            var momMin = head.momentum, momMax = head.momentum, momLast = head.momentum
            if last > first {
                for step in (first + 1)...last {
                    let p = valueAt(step)
                    lrMin = min(lrMin, p.learningRate)
                    lrMax = max(lrMax, p.learningRate)
                    lrLast = p.learningRate
                    momMin = min(momMin, p.momentum)
                    momMax = max(momMax, p.momentum)
                    momLast = p.momentum
                }
            }
            buckets.append(Bucket(firstStep: first, lastStep: last, lrPeak: head.lrPeak, lrTrough: head.lrTrough,
                                  lrMin: lrMin, lrMax: lrMax, lrLast: lrLast,
                                  momentumMin: momMin, momentumMax: momMax, momentumLast: momLast))
            first = last + 1
        }
        return LRScheduleSample(buckets: buckets)
    }
}
