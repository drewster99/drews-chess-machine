import Foundation

/// Tracks the 1-minute rolling ratio of training consumption to
/// self-play production and optionally auto-adjusts the training
/// step delay so the two sides stay in balance.
///
/// The auto-adjuster works by:
/// 1. Measuring the actual per-step overhead (GPU + buffer locks +
///    scheduling — everything except the sleep delay) from the
///    observed consumption rate minus the known delay.
/// 2. Computing the desired cycle time from the target ratio and
///    the observed production rate.
/// 3. Setting delay = desiredCycle − overhead, with damping so
///    large changes converge smoothly over ~15 seconds instead of
///    oscillating against the 60-second measurement window.
///
/// Thread-safe via `NSLock`.
final class ReplayRatioController: @unchecked Sendable {
    private let lock = NSLock()

    // MARK: - Configuration

    private var _targetRatio: Double
    private var _autoAdjust: Bool
    private var _manualDelayMs: Int
    let batchSize: Int
    let maxDelayMs: Int

    // MARK: - Internal state

    private var _totalConsumed: Int = 0
    private var _computedDelayMs: Int

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
        maxDelayMs: Int = 500
    ) {
        self.batchSize = batchSize
        self._targetRatio = targetRatio
        self._autoAdjust = autoAdjust
        self._manualDelayMs = initialDelayMs
        self._computedDelayMs = initialDelayMs
        self.maxDelayMs = maxDelayMs
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
        lock.lock()
        defer { lock.unlock() }

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
            return _autoAdjust ? _computedDelayMs : _manualDelayMs
        }

        let dt = newest.time.timeIntervalSince(oldest.time)
        guard dt >= warmupThreshold else {
            return _autoAdjust ? _computedDelayMs : _manualDelayMs
        }

        let productionRate = Double(newest.produced - oldest.produced) / dt
        let consumptionRate = Double(newest.consumed - oldest.consumed) / dt

        if _autoAdjust && productionRate > 0 && consumptionRate > 0 {
            // 1. Estimate per-step overhead from the measured
            //    consumption rate minus the known delay we set.
            //    Floor at the EMA of GPU step time so the estimate
            //    never collapses to zero during a transition where
            //    the delay exceeds the stale cycle measurement.
            let actualCycleSec = Double(batchSize) / consumptionRate
            let currentDelaySec = Double(_computedDelayMs) / 1000.0
            let emaFloorSec = _emaStepTimeMs / 1000.0
            let overheadSec = max(emaFloorSec, actualCycleSec - currentDelaySec)

            // 2. Desired cycle time from the target ratio.
            let desiredCycleSec = Double(batchSize) / (_targetRatio * productionRate)

            // 3. Target delay = desired cycle minus overhead.
            let targetDelayMs = max(0, (desiredCycleSec - overheadSec) * 1000.0)

            // 4. Damped move toward the target.
            let newDelayMs = Double(_computedDelayMs) + damping * (targetDelayMs - Double(_computedDelayMs))
            _computedDelayMs = max(0, min(maxDelayMs, Int(newDelayMs)))
        }

        return _autoAdjust ? _computedDelayMs : _manualDelayMs
    }

    // MARK: - UI snapshot

    struct RatioSnapshot: Sendable {
        let productionRate: Double
        let consumptionRate: Double
        let currentRatio: Double
        let targetRatio: Double
        let autoAdjust: Bool
        let computedDelayMs: Int
    }

    func snapshot() -> RatioSnapshot {
        lock.lock()
        defer { lock.unlock() }

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
            computedDelayMs: _computedDelayMs
        )
    }

    // MARK: - UI setters

    var targetRatio: Double {
        get { lock.lock(); defer { lock.unlock() }; return _targetRatio }
        set { lock.lock(); _targetRatio = newValue; lock.unlock() }
    }

    var autoAdjust: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _autoAdjust }
        set { lock.lock(); _autoAdjust = newValue; lock.unlock() }
    }

    var manualDelayMs: Int {
        get { lock.lock(); defer { lock.unlock() }; return _manualDelayMs }
        set { lock.lock(); _manualDelayMs = max(0, min(maxDelayMs, newValue)); lock.unlock() }
    }

    var computedDelayMs: Int {
        get { lock.lock(); defer { lock.unlock() }; return _computedDelayMs }
        set { lock.lock(); _computedDelayMs = max(0, min(maxDelayMs, newValue)); lock.unlock() }
    }
}
