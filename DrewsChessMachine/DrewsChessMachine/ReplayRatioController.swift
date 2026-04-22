import Foundation

/// Tracks the 1-minute rolling ratio of training consumption to
/// self-play production and optionally auto-adjusts the sleep
/// between training steps (or, when training can't keep up, the
/// sleep between self-play games) so the two sides stay in balance.
///
/// The auto-adjuster works by:
/// 1. Measuring the actual per-step overhead (GPU + buffer locks +
///    scheduling — everything except the sleep delay) from the
///    observed consumption rate minus the known delay.
/// 2. Computing the desired cycle time from the target ratio and
///    the observed production rate.
/// 3. Setting signed delay = desiredCycle − overhead, with damping so
///    large changes converge smoothly over ~15 seconds instead of
///    oscillating against the 60-second measurement window.
///
/// The "signed delay" convention: one signed integer is the single
/// source of truth. A positive value means "training should sleep
/// this many ms between steps" (self-play is already the bottleneck,
/// slow training down to match). A negative value means "self-play
/// should sleep |value| ms between games" (training is the bottleneck
/// — the GPU+overhead per training step is already slower than the
/// target cycle, so the only lever left is to slow game production).
/// Zero means neither side sleeps. By construction, at most one side
/// is non-zero at any moment, so the invariant "one of the two
/// delays is always zero" holds trivially — the two public getters
/// `computedTrainingDelayMs` and `computedSelfPlayDelayMs` just
/// project the signed value onto the positive side each cares about.
///
/// Thread-safe via a private serial `DispatchQueue`. All accessors
/// hop through `queue.sync` so the object presents the same atomic
/// read/write semantics the old `NSLock` version did.
final class ReplayRatioController: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.replayratiocontroller.serial")

    // MARK: - Configuration

    private var _targetRatio: Double
    private var _autoAdjust: Bool
    private var _manualDelayMs: Int
    let batchSize: Int
    /// Hard ceiling on the training-side (positive) delay.
    let maxDelayMs: Int
    /// Hard ceiling on the self-play-side (negative) delay magnitude.
    /// Per-game sleeps larger than this are unlikely to be useful —
    /// the controller reacts on the 60-second window, and a
    /// several-second pause between games per slot is already enough
    /// to cut production throughput in half.
    let maxSelfPlayDelayMs: Int

    // MARK: - Internal state

    private var _totalConsumed: Int = 0

    /// Signed delay. `> 0` → training sleeps; `< 0` → self-play sleeps;
    /// `== 0` → neither. This is the single source of truth; the two
    /// getters exposed to callers are just projections.
    private var _computedSignedDelayMs: Int

    /// EMA of GPU step time (from TrainStepTiming.totalMs). Used
    /// only as a FLOOR on the overhead estimate — prevents the
    /// overhead from collapsing to zero when the delay exceeds
    /// the stale cycle measurement during a transition.
    private var _emaStepTimeMs: Double = 0
    private var _emaInitialized: Bool = false
    private let emaAlpha: Double = 0.05

    /// Per-step damping factor. Each step moves 10% of the way
    /// from current delay to the computed target. At ~3 steps/sec
    /// this converges in ~15 seconds. Low enough to avoid
    /// oscillation from stale window data, high enough to track
    /// production rate changes within a minute.
    private let damping: Double = 0.1

    private struct Sample {
        let time: Date
        let produced: Int
        let consumed: Int
    }
    private var samples: [Sample] = []
    private let windowSeconds: Double = 60.0

    // MARK: - Init

    init(
        batchSize: Int,
        targetRatio: Double = 1.0,
        autoAdjust: Bool = true,
        initialDelayMs: Int = 50,
        maxDelayMs: Int = 500,
        maxSelfPlayDelayMs: Int = 2000
    ) {
        self.batchSize = batchSize
        self._targetRatio = targetRatio
        self._autoAdjust = autoAdjust
        self._manualDelayMs = initialDelayMs
        // `initialDelayMs` is conventionally non-negative (the training
        // delay persisted from the previous session, or a manual UI
        // value). Seed the signed delay from it directly — if the
        // controller needs to swing negative, the damping step will
        // get there on its own.
        self._computedSignedDelayMs = initialDelayMs
        self.maxDelayMs = maxDelayMs
        self.maxSelfPlayDelayMs = maxSelfPlayDelayMs
    }

    // MARK: - Training worker interface

    /// Called by the training worker after each `trainStep`.
    ///
    /// - Parameters:
    ///   - currentBufferTotal: `buffer.totalPositionsAdded`.
    ///   - stepTimeMs: Wall-clock duration of the just-completed
    ///     training step (GPU + data prep + readback, NOT including
    ///     the sleep delay).
    /// - Returns: The delay in ms the training worker should
    ///   `Task.sleep` before the next step.
    func recordStepAndGetDelay(
        currentBufferTotal: Int,
        stepTimeMs: Double
    ) -> Int {
        queue.sync {
            // Update the GPU step-time EMA (used as overhead floor).
            if _emaInitialized {
                _emaStepTimeMs = emaAlpha * stepTimeMs + (1 - emaAlpha) * _emaStepTimeMs
            } else {
                _emaStepTimeMs = stepTimeMs
                _emaInitialized = true
            }

            _totalConsumed += batchSize

            let now = Date()
            samples.append(Sample(
                time: now,
                produced: currentBufferTotal,
                consumed: _totalConsumed
            ))

            let cutoff = now.addingTimeInterval(-windowSeconds)
            while let first = samples.first, first.time < cutoff {
                samples.removeFirst()
            }

            let warmupThreshold = windowSeconds * 0.5
            guard samples.count >= 2,
                  let oldest = samples.first,
                  let newest = samples.last else {
                return _autoAdjust ? max(0, _computedSignedDelayMs) : _manualDelayMs
            }

            let dt = newest.time.timeIntervalSince(oldest.time)
            guard dt >= warmupThreshold else {
                return _autoAdjust ? max(0, _computedSignedDelayMs) : _manualDelayMs
            }

            let productionRate = Double(newest.produced - oldest.produced) / dt
            let consumptionRate = Double(newest.consumed - oldest.consumed) / dt

            if _autoAdjust && productionRate > 0 && consumptionRate > 0 {
                // 1. Estimate per-step overhead from the measured
                //    consumption rate minus the known delay we set.
                //    Floor at the EMA of GPU step time so the estimate
                //    never collapses to zero during a transition where
                //    the delay exceeds the stale cycle measurement.
                //    Only the positive-side (training) delay affects
                //    the consumption cycle — when the controller is in
                //    self-play-slowdown mode the training cycle is
                //    pure overhead, so `currentTrainingDelaySec` is 0.
                let actualCycleSec = Double(batchSize) / consumptionRate
                let currentTrainingDelaySec = Double(max(0, _computedSignedDelayMs)) / 1000.0
                let emaFloorSec = _emaStepTimeMs / 1000.0
                let overheadSec = max(emaFloorSec, actualCycleSec - currentTrainingDelaySec)

                // 2. Desired cycle time from the target ratio.
                let desiredCycleSec = Double(batchSize) / (_targetRatio * productionRate)

                // 3. Signed target delay = desiredCycle − overhead.
                //    Positive: training too fast relative to production,
                //    slow training down. Negative: training can't keep
                //    up even at zero delay — slow self-play instead, by
                //    a magnitude that would push the cycle balance back
                //    toward the target on the next measurement window.
                let targetSignedDelayMs = (desiredCycleSec - overheadSec) * 1000.0

                // 4. Damped move toward the target, symmetric across
                //    the sign boundary — one control law governs both
                //    training-slowdown and self-play-slowdown regimes.
                let newDelayMs = Double(_computedSignedDelayMs)
                    + damping * (targetSignedDelayMs - Double(_computedSignedDelayMs))
                _computedSignedDelayMs = max(-maxSelfPlayDelayMs, min(maxDelayMs, Int(newDelayMs)))
            }

            return _autoAdjust ? max(0, _computedSignedDelayMs) : _manualDelayMs
        }
    }

    // MARK: - UI snapshot

    struct RatioSnapshot: Sendable {
        let productionRate: Double
        let consumptionRate: Double
        let currentRatio: Double
        let targetRatio: Double
        let autoAdjust: Bool
        /// Training-side delay, >= 0. Zero whenever self-play is the
        /// side being slowed.
        let computedDelayMs: Int
        /// Self-play-side delay (per-game sleep per slot), >= 0. Zero
        /// whenever training is the side being slowed. At most one of
        /// `computedDelayMs` and `computedSelfPlayDelayMs` is non-zero
        /// at any moment — the controller enforces this by storing a
        /// single signed source of truth.
        let computedSelfPlayDelayMs: Int
    }

    func snapshot() -> RatioSnapshot {
        queue.sync {
            let now = Date()
            let cutoff = now.addingTimeInterval(-windowSeconds)

            var productionRate: Double = 0
            var consumptionRate: Double = 0

            if samples.count >= 2 {
                var oldestInWindow: Sample?
                for s in samples where s.time >= cutoff {
                    oldestInWindow = s
                    break
                }
                if let oldest = oldestInWindow, let newest = samples.last {
                    let dt = newest.time.timeIntervalSince(oldest.time)
                    if dt > 3.0 {
                        productionRate = Double(newest.produced - oldest.produced) / dt
                        consumptionRate = Double(newest.consumed - oldest.consumed) / dt
                    }
                }
            }

            let ratio = productionRate > 0
                ? consumptionRate / productionRate
                : 0

            return RatioSnapshot(
                productionRate: productionRate,
                consumptionRate: consumptionRate,
                currentRatio: ratio,
                targetRatio: _targetRatio,
                autoAdjust: _autoAdjust,
                computedDelayMs: max(0, _computedSignedDelayMs),
                computedSelfPlayDelayMs: max(0, -_computedSignedDelayMs)
            )
        }
    }

    // MARK: - UI setters

    var targetRatio: Double {
        get { queue.sync { _targetRatio } }
        set { queue.async { self._targetRatio = newValue } }
    }

    var autoAdjust: Bool {
        get { queue.sync { _autoAdjust } }
        set { queue.async { self._autoAdjust = newValue } }
    }

    var manualDelayMs: Int {
        get { queue.sync { _manualDelayMs } }
        set { queue.async { self._manualDelayMs = max(0, min(self.maxDelayMs, newValue)) } }
    }

    /// Training-side delay projection of the signed source of truth.
    /// Getter returns `max(0, signed)`. Setter accepts a non-negative
    /// training delay and writes it verbatim into the signed state,
    /// which implicitly zeroes any self-play-side delay. This is the
    /// path the Auto-Adjust toggle uses to seed the controller from a
    /// persisted manual delay so the displayed value doesn't jump
    /// on mode change.
    var computedDelayMs: Int {
        get { queue.sync { max(0, _computedSignedDelayMs) } }
        set {
            queue.async {
                let clamped = max(0, min(self.maxDelayMs, newValue))
                self._computedSignedDelayMs = clamped
            }
        }
    }

    /// Self-play-side delay projection. Read-only; the controller is
    /// the only writer (via the signed auto-adjust loop).
    var computedSelfPlayDelayMs: Int {
        queue.sync { max(0, -_computedSignedDelayMs) }
    }
}
