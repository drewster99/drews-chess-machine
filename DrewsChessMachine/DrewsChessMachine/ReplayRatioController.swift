import Foundation

/// Tracks the 1-minute rolling ratio of training consumption to
/// self-play production and optionally auto-adjusts the training
/// step delay so the two sides stay in balance.
///
/// **Production rate** — positions added to the replay buffer per
/// second. The training worker passes the buffer's cumulative
/// `totalPositionsAdded` on each step; the controller diffs
/// successive readings to compute the rolling rate.
///
/// **Consumption rate** — positions consumed by training per second.
/// Each `recordStep` call increments by `batchSize`.
///
/// **Auto-adjustment** — when enabled, directly computes the delay
/// from an exponential moving average of GPU step time and the
/// 1-minute production rate:
///
///     desiredCycleSec = batchSize / (targetRatio × productionRate)
///     delayMs = max(0, desiredCycleSec - emaGpuTimeSec) × 1000
///
/// The EMA of GPU step time gives a stable estimate that adapts
/// over ~20 steps without the oscillation that accumulating deltas
/// against a stale 60-second window caused.
///
/// Thread-safe via `NSLock` — the training worker writes on each
/// step and the UI heartbeat reads for display.
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

    /// Exponential moving average of the GPU training step time
    /// (milliseconds). Updated on every step with alpha=0.05,
    /// giving an effective smoothing window of ~20 steps. Used as
    /// the "how long does the GPU need" estimate in the delay
    /// calculation.
    private var _emaStepTimeMs: Double = 0
    private var _emaInitialized: Bool = false
    private let emaAlpha: Double = 0.05

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
    /// Updates the GPU-time EMA, appends a rate sample, and
    /// optionally recomputes the delay.
    ///
    /// - Parameters:
    ///   - currentBufferTotal: `buffer.totalPositionsAdded`.
    ///   - stepTimeMs: Wall-clock duration of the just-completed
    ///     training step (`TrainStepTiming.totalMs`). Fed into an
    ///     EMA to produce a smooth GPU-time estimate.
    /// - Returns: The delay in ms the training worker should
    ///   `Task.sleep` before the next step.
    func recordStepAndGetDelay(
        currentBufferTotal: Int,
        stepTimeMs: Double
    ) -> Int {
        lock.lock()
        defer { lock.unlock() }

        // Update the GPU step-time EMA.
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

        // Prune samples older than the window.
        let cutoff = now.addingTimeInterval(-windowSeconds)
        while let first = samples.first, first.time < cutoff {
            samples.removeFirst()
        }

        // Wait for enough data before auto-adjusting.
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

        if _autoAdjust && productionRate > 0 {
            // Direct computation: how long should the total cycle
            // (GPU work + delay) be to hit the target ratio?
            //
            //   desiredCycleSec = batchSize / (targetRatio × productionRate)
            //   delay = desiredCycle - gpuTime
            //
            // The GPU time comes from the EMA, which is measured
            // independently of the delay and smoothed over ~20 steps.
            // This avoids the oscillation that delta-accumulation
            // against a stale 60-second window caused: the EMA
            // responds in seconds, the production rate is smooth
            // over 60s, and the delay is computed fresh each step
            // without any feedback loop through the measurement
            // window.
            let desiredCycleSec = Double(batchSize) / (_targetRatio * productionRate)
            let gpuTimeSec = _emaStepTimeMs / 1000.0
            let rawDelayMs = (desiredCycleSec - gpuTimeSec) * 1000.0
            _computedDelayMs = max(0, min(maxDelayMs, Int(rawDelayMs)))
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
