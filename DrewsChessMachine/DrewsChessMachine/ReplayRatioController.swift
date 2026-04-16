import Foundation

/// Tracks the 1-minute rolling ratio of training consumption to
/// self-play production and optionally auto-adjusts the training
/// step delay so the two sides stay in balance.
///
/// **Production rate** — positions added to the replay buffer per
/// second. The training worker passes the buffer's cumulative
/// `totalPositionsAdded` on each step; the controller diffs
/// successive readings to compute the rolling rate. No changes
/// to self-play workers are needed.
///
/// **Consumption rate** — positions consumed by training per second.
/// Each `recordStep` call increments by `batchSize`.
///
/// **Auto-adjustment** — when enabled, computes the training-step
/// delay (in ms) that would bring the ratio to `targetRatio`:
///
///     targetConsumptionRate = targetRatio × productionRate
///     desiredCycleSec = batchSize / targetConsumptionRate
///     delayMs = max(0, desiredCycleSec - stepTimeSec) × 1000
///
/// The result is clamped to `[0, maxDelayMs]` and returned to the
/// training worker, which uses it as its `Task.sleep` duration
/// for the next step. When auto-adjust is off, the most-recently-
/// set manual delay is returned unchanged.
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

    private struct Sample {
        let time: Date
        let produced: Int
        let consumed: Int
    }
    private var samples: [Sample] = []
    private let windowSeconds: Double = 60.0

    // MARK: - Init

    /// - Parameters:
    ///   - batchSize: Positions consumed per training step.
    ///   - targetRatio: Desired consumed/produced ratio (default 1.0).
    ///   - autoAdjust: Whether the controller manages the delay.
    ///   - initialDelayMs: Starting delay before any rate data arrives.
    ///   - maxDelayMs: Hard ceiling on computed delay.
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
    /// Appends a snapshot, prunes old ones, optionally recomputes
    /// the delay, and returns the delay to use for the next step.
    ///
    /// - Parameter currentBufferTotal: `buffer.totalPositionsAdded`
    ///   — the monotonically-increasing count of all positions ever
    ///   appended to the replay buffer.
    /// - Returns: The delay in ms the training worker should
    ///   `Task.sleep` before the next step.
    func recordStepAndGetDelay(
        currentBufferTotal: Int
    ) -> Int {
        lock.lock()
        defer { lock.unlock() }

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

        // Need enough samples spanning a meaningful window before the
        // auto-adjuster kicks in. During the first ~30 s of a session,
        // self-play is filling the buffer while training hasn't started
        // yet (or just started), so the ratio is meaninglessly skewed.
        // We use half the window as the threshold rather than the full
        // window because pruning removes samples older than windowSeconds,
        // so the oldest-to-newest span asymptotically approaches but
        // never reaches the full window.
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
            // Feedback controller: measure the current cycle time
            // (wall time per training step including delay), compute
            // the desired cycle time from the target ratio, and
            // adjust the delay by the difference.
            //
            //   currentCycle = dt / stepsInWindow     (measured)
            //   desiredCycle = batchSize / (target × productionRate)
            //   newDelay = currentDelay + (desiredCycle - currentCycle)
            //
            // This avoids the circular dependency of trying to
            // separate GPU time from delay time — we simply nudge the
            // delay in the direction that reduces the error between
            // actual and desired cycle time.
            let stepsInWindow = Double(newest.consumed - oldest.consumed) / Double(batchSize)
            guard stepsInWindow > 0 else {
                return _computedDelayMs
            }
            let currentCycleSec = dt / stepsInWindow
            let targetConsumptionRate = _targetRatio * productionRate
            let desiredCycleSec = Double(batchSize) / targetConsumptionRate
            let currentDelaySec = Double(_computedDelayMs) / 1000.0
            let newDelaySec = currentDelaySec + (desiredCycleSec - currentCycleSec)
            _computedDelayMs = max(0, min(maxDelayMs, Int(newDelaySec * 1000.0)))
        }

        return _autoAdjust ? _computedDelayMs : _manualDelayMs
    }

    // MARK: - UI snapshot

    /// Immutable read for the heartbeat / display layer.
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

        // Show rates as soon as we have a few seconds of data so the
        // display isn't stuck on dashes for the entire warmup period.
        // The auto-adjust guard in recordStepAndGetDelay still waits
        // for the full 60s window before touching the delay.
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
}
