import Foundation

/// Pure, host-testable interpolation that assigns a trainer-step value
/// to a chart sample that predates the `TrainingChartSample.trainingStep`
/// field, so the combined Training-vs-Eval-Loss window can place those
/// historical samples on its shared step axis.
///
/// The mapping is built from exact `(elapsedSec, step)` anchors — the
/// Lichess probe stamps its overall-series ticks with both an absolute
/// timestamp (convertible to `elapsedSec` via the chart's elapsed
/// anchor) and the authoritative `completedTrainSteps`, so each tick is
/// a ground-truth point on the step-vs-time curve. The caller appends a
/// terminal anchor at `(current elapsedSec, current completedTrainSteps)`
/// so the region after the last probe tick is pinned to the real step
/// total rather than extrapolated blindly.
///
/// Piecewise-linear between anchors. This is deliberately better than a
/// naive "evenly spaced samples across the total step count" map: chart
/// samples are spaced evenly in *wall time*, not in *steps*, so during an
/// arena (training paused, time advancing, step flat) the anchors keep
/// the interpolated step correctly flat instead of marching it forward.
enum TrainingStepBackfill {
    /// One ground-truth point on the step-vs-time curve.
    struct Anchor: Equatable {
        let elapsedSec: Double
        let step: Int
    }

    /// Normalize raw anchor candidates into a clean, sorted, strictly
    /// step-monotonic sequence suitable for `interpolatedStep`:
    /// - drops non-finite / negative elapsed,
    /// - sorts by elapsed ascending,
    /// - enforces non-decreasing step (a later-in-time anchor can never
    ///   report fewer steps; clamp up if timestamps and the global
    ///   counter momentarily disagree),
    /// - collapses duplicate elapsed values to their max step.
    static func normalize(_ anchors: [Anchor]) -> [Anchor] {
        let sorted = anchors
            .filter { $0.elapsedSec.isFinite && $0.elapsedSec >= 0 && $0.step >= 0 }
            .sorted { $0.elapsedSec < $1.elapsedSec }
        var out: [Anchor] = []
        out.reserveCapacity(sorted.count)
        var maxStep = 0
        for a in sorted {
            let step = max(maxStep, a.step)
            maxStep = step
            if let last = out.last, last.elapsedSec == a.elapsedSec {
                // Same instant — keep the larger step.
                out[out.count - 1] = Anchor(elapsedSec: a.elapsedSec, step: step)
            } else {
                out.append(Anchor(elapsedSec: a.elapsedSec, step: step))
            }
        }
        return out
    }

    /// Interpolate the step at `elapsedSec` against `anchors`, which MUST
    /// already be normalized (sorted, step-monotonic). Returns `nil` only
    /// when there are no anchors at all.
    ///
    /// - Before the first anchor: linear from the origin `(0, 0)` up to
    ///   the first anchor (training starts at step 0 at t≈0).
    /// - Between anchors: linear in the bracketing segment.
    /// - After the last anchor: linear extrapolation along the final
    ///   segment's slope (flat if only one anchor), never below the last
    ///   anchor's step.
    static func interpolatedStep(elapsedSec: Double, anchors: [Anchor]) -> Int? {
        guard let first = anchors.first, let last = anchors.last else { return nil }

        if elapsedSec <= first.elapsedSec {
            guard first.elapsedSec > 0 else { return first.step }
            let frac = max(0, elapsedSec) / first.elapsedSec
            return Int((Double(first.step) * frac).rounded())
        }

        if elapsedSec >= last.elapsedSec {
            guard anchors.count >= 2 else { return last.step }
            let prev = anchors[anchors.count - 2]
            let dt = last.elapsedSec - prev.elapsedSec
            guard dt > 0 else { return last.step }
            let slope = Double(last.step - prev.step) / dt
            let projected = Double(last.step) + (elapsedSec - last.elapsedSec) * slope
            return max(last.step, Int(projected.rounded()))
        }

        // Binary-search the bracketing segment [lo, hi].
        var lo = 0
        var hi = anchors.count - 1
        while lo + 1 < hi {
            let mid = (lo + hi) / 2
            if anchors[mid].elapsedSec <= elapsedSec { lo = mid } else { hi = mid }
        }
        let a = anchors[lo]
        let b = anchors[hi]
        let dt = b.elapsedSec - a.elapsedSec
        guard dt > 0 else { return a.step }
        let frac = (elapsedSec - a.elapsedSec) / dt
        let value = Double(a.step) + frac * Double(b.step - a.step)
        return Int(value.rounded())
    }
}
