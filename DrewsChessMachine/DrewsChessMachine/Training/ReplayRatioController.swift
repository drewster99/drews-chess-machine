import Foundation
import os

/// Event-driven replay-ratio controller. Three recording hooks —
/// two that drive control, one that feeds metadata:
///
///   • `recordSelfPlayBarrierTick(positionsProduced:currentDelaySettingMs:workerCount:)`
///     — fires every batched-evaluator barrier tick on the self-play
///     side. The CONTROLLER owns the inter-tick wall measurement —
///     it stamps `_lastSelfPlayTickAt` on every call and computes
///     `elapsedMs = (now - _lastSelfPlayTickAt) * 1000` for the
///     subsequent call. The caller passes only
///     `currentDelaySettingMs`: the per-game self-play delay value
///     it is currently configured to sleep (i.e. what the driver
///     just looked up from `computedSelfPlayDelayMs`). The
///     controller subtracts that from its own measured wall to
///     recover the underlying production rate at zero injected
///     delay. First call: stamp only, no measurement. Ticks fire at
///     hundreds-to-thousands per second (every ply across all N
///     slots), giving the controller much finer measurement
///     granularity than the old once-per-game cadence.
///
///   • `recordTrainingBatchAndGetDelay(currentDelaySettingMs:)` —
///     fires after each SGD step. Same contract as the self-play
///     side: controller owns the wall-clock with `_lastTrainingTickAt`,
///     caller passes only the current training-side delay setting
///     (what it is currently configured to sleep between batches).
///     First call: stamp only, return seeded delay. Returns the
///     sleep the worker should apply before the next SGD step.
///     Symmetric with sp side: both APIs are "report your current
///     delay setting"; the controller does ALL timing and ALL
///     subtraction in one place. Before this convention, the
///     training side passed only the SGD batch's work time, which
///     excluded its own sleep and any inter-call work — producing
///     an asymmetric sp/tr measurement that pulled the equilibrium
///     ratio away from `targetRatio`.
///
///   • `recordSelfPlayGameLength(_:)` — metadata-only. The driver
///     calls this once per completed game so the controller knows
///     `positionsPerGame` (`G` in the aggregate formula), needed
///     ONLY for converting the self-play-side sleep magnitude from
///     per-ply to per-game. Not a measurement event — doesn't touch
///     `selfPlayMsPerMove` or trigger a meaningful recompute.
///
/// Aggregate per-move times used by the formula:
///     selfPlayMsPerMove = elapsedMs / positionsProduced    (per barrier tick)
///     trainingMsPerMove = elapsedMs / trainingBatchSize
///
/// Closed-form delay (user-derived, model-inverting):
///     signedDelayPerTrainingBatchMs =
///         (selfPlayMsPerMove / targetRatio − trainingMsPerMove)
///         × trainingBatchSize
///
///   • signed ≥ 0 → training too fast for target. Raw training
///     delay = signed, raw self-play delay = 0.
///   • signed < 0 → training too slow for target / self-play too
///     fast. Raw training delay = 0. Raw self-play per-game sleep
///     derived from the aggregate-rate model: to raise aggregate
///     `selfPlayMsPerMove` by `gap` ms/move, each of `N` workers
///     playing games of `G` plies must sleep `D = gap × N × G` ms
///     per game (both factors come from aggregate
///     `(T_work + D) / (N × G)` = target).
///
/// The raw `signedDelay` value from each event is NOT applied
/// directly. Each event appends `(timestamp, signed)` to a
/// rolling history; the publicly-visible applied delays come
/// from an SMA of that history.
///
/// Smoothing over a fixed 20 s wall-clock window (not a fixed
/// sample count). To keep this efficient at high barrier-tick
/// rates (~2700 Hz), we use a ring of 20 buckets (1 second each).
/// Each bucket maintains its own sum and count.
private struct Bucket {
    var startTime: Double = 0
    var sum: Double = 0
    var count: Int = 0

    mutating func reset(startTime: Double) {
        self.startTime = startTime
        self.sum = 0
        self.count = 0
    }
}
private var _buckets: [Bucket] = (0..<20).map { _ in Bucket() }
private var _currentBucketIndex: Int = 0
/// Global running sum and count across all buckets. Updated on
/// every append and every bucket-wrap (eviction), so the SMA
/// projection is always O(1).
private var _historySum: Double = 0
private var _historyCount: Int = 0

// MARK: - 1-minute rolling rate estimator

/// Separate rolling estimator for the UI/log/JSON production and
/// consumption rates.
private let rateWindowSec: Double = 60.0
private let rateSampleIntervalSec: Double = 0.25
private var _totalSelfPlayPositions: Int = 0
private var _totalTrainingPositions: Int = 0
private struct RateSample {
    let at: Date
    let selfPlayTotal: Int
    let trainingTotal: Int
}
/// Small history of total-count snapshots (one every 0.25s).
/// Max size is ~240 entries; O(N) pruning is acceptable here.
private var _rateSamples: [RateSample] = []
private var _lastRateSampleAt: Date? = nil

// MARK: - Init

init(
    batchSize: Int,
    targetRatio: Double = 1.0,
    autoAdjust: Bool = true,
    initialDelayMs: Int = 50,
    maxTrainingStepDelayMs: Int = 3000,
    maxSelfPlayDelayMs: Int = 3000
) {
    self.trainingBatchSize = batchSize
    self._targetRatio = targetRatio
    self._autoAdjust = autoAdjust
    self._manualDelayMs = initialDelayMs
    // Seed the raw SIGNED delay from the passed initial value.
    let seedSigned = Double(max(0, min(maxTrainingStepDelayMs, initialDelayMs)))
    self._rawSignedDelayMs = seedSigned
    self.maxTrainingStepDelayMs = maxTrainingStepDelayMs
    self.maxSelfPlayDelayMs = maxSelfPlayDelayMs

    // Seed the first bucket.
    let now = CFAbsoluteTimeGetCurrent()
    _buckets[0].reset(startTime: now)
}

// MARK: - Recording hooks
    /// Called once per self-play barrier tick. A barrier tick is one
    /// fire of the shared batched evaluator: all currently-submitting
    /// slots submit one board each, the GPU runs one batched forward
    /// pass, and each slot resumes with its result. Exactly
    /// `positionsProduced` plies come out per tick.
    /// - Parameters:
    ///   - positionsProduced: plies emitted by this tick. Under
    ///     normal operation equals `workerCount`, but the batcher can
    ///     fire smaller batches during slot-count transitions, so we
    ///     accept both separately.
    ///   - currentDelaySettingMs: the **per-game** self-play delay
    ///     the caller is currently configured to sleep. Note the
    ///     unit: the caller's sleep is per game-end (a few hundred
    ///     ms), but a barrier tick is per-ply across the pool
    ///     (tens of ms). The controller can NOT subtract this value
    ///     directly from the per-tick wall — it has to convert from
    ///     per-game units to per-tick overhead first. With G =
    ///     `_lastSelfPlayPositionsPerGame` and `positionsProduced`
    ///     ply-submissions in this tick, the expected number of
    ///     game-ends across the worker pool in this tick is
    ///     `positionsProduced / G`. Each game-end contributes a
    ///     full per-game sleep to the slowest worker's tick (the
    ///     barrier waits for it), so the expected per-tick wall
    ///     inflation from injected delay is approximately
    ///     `(positionsProduced / G) × currentDelaySettingMs`. The
    ///     controller subtracts that per-tick estimate, NOT the
    ///     raw per-game value, before computing per-move rate.
    ///     Without this conversion the controller would think it
    ///     was injecting ~10× more delay than it actually does,
    ///     drive the smoothed self-play sleep way up, and settle on
    ///     a steady-state ratio far above target (observed ~2.5 vs
    ///     target 1.10 before this fix). Pass 0 if the caller
    ///     knows it is applying no throttling. Negative effective
    ///     elapsed (small-window noise) is not clamped — the 20s SMA
    ///     over `_delayHistory` absorbs transients.
    ///   - workerCount: currently-active slot count (`N` in the
    ///     aggregate sp-slowdown formula). Piped in each tick so the
    ///     per-game sleep calculation stays correct when the user
    ///     changes the worker stepper mid-session.
    func recordSelfPlayBarrierTick(
        positionsProduced: Int,
        currentDelaySettingMs: Double,
        workerCount: Int
    ) {
        lock.withLock {
            let now = CFAbsoluteTimeGetCurrent()
            if let prior = _lastSelfPlayTickAt, positionsProduced > 0 {
                let elapsedMs = (now - prior) * 1000.0
                let g = Double(max(1, _lastSelfPlayPositionsPerGame))
                let perTickOverheadMs =
                    currentDelaySettingMs * Double(positionsProduced) / g
                let effectiveElapsedMs = elapsedMs - perTickOverheadMs
                _selfPlayMsPerMove = effectiveElapsedMs / Double(positionsProduced)
                _totalSelfPlayPositions += positionsProduced
                maybeAppendRateSample(now: Date())
            }
            _lastSelfPlayTickAt = now
            _lastSelfPlayWorkerCount = max(1, workerCount)
            recomputeDelays()
        }
    }

    /// Called once per completed self-play game to refresh the
    /// positions-per-game metadata used only in the sp-slowdown
    /// sleep calculation. Not a measurement event — the aggregate
    /// ms-per-move comes from `recordSelfPlayBarrierTick`. We still
    /// recompute here so a freshly-updated `G` is reflected in the
    /// self-play sleep the next time signed < 0.
    func recordSelfPlayGameLength(_ plies: Int) {
        lock.withLock {
            _lastSelfPlayPositionsPerGame = max(1, plies)
            recomputeDelays()
        }
    }

    /// Called by the training worker after each SGD step. Symmetric
    /// API with `recordSelfPlayBarrierTick`: the controller owns the
    /// inter-call wall measurement directly (stamps
    /// `_lastTrainingTickAt` on every call), and the caller passes
    /// only the per-batch training-side delay it is currently
    /// configured to sleep. The controller subtracts that from its
    /// own measured wall to get tr_per_move at zero injected delay.
    /// First call (no prior stamp): just stamp and return the
    /// seeded delay; the first usable measurement arrives on call
    /// #2. Negative effective elapsed from small-window noise is
    /// not clamped — the 20s SMA absorbs transients.
    /// Returns the sleep the worker should apply before the next SGD
    /// step (training-side projection of the SMA signed delay over
    /// the last `historyWindowSec` seconds).
    func recordTrainingBatchAndGetDelay(
        currentDelaySettingMs: Double
    ) -> Int {
        lock.withLock {
            let now = CFAbsoluteTimeGetCurrent()
            if let prior = _lastTrainingTickAt {
                let elapsedMs = (now - prior) * 1000.0
                let effectiveElapsedMs = elapsedMs - currentDelaySettingMs
                _trainingMsPerMove = effectiveElapsedMs / Double(trainingBatchSize)
                _totalTrainingPositions += trainingBatchSize
                maybeAppendRateSample(now: Date())
            }
            _lastTrainingTickAt = now
            recomputeDelays()
            return _autoAdjust ? smoothedTrainingStepDelayMs() : _manualDelayMs
        }
    }

    /// Closed-form recompute of the RAW signed delay from the latest
    /// per-move estimates, followed by an append to the rolling
    /// history.
    private func recomputeDelays() {
        guard _autoAdjust,
              let sp = _selfPlayMsPerMove,
              let tr = _trainingMsPerMove,
              sp > 0, tr > 0,
              _targetRatio > 0 else {
            return
        }

        let signedPerTrainingBatchMs =
            (sp / _targetRatio - tr) * Double(trainingBatchSize)
        _rawSignedDelayMs = signedPerTrainingBatchMs

        let now = CFAbsoluteTimeGetCurrent()
        pruneDelayHistory(now: now)

        _buckets[_currentBucketIndex].sum += signedPerTrainingBatchMs
        _buckets[_currentBucketIndex].count += 1
        _historySum += signedPerTrainingBatchMs
        _historyCount += 1
    }

    /// Advance the bucket ring if `now` has crossed a 1-second
    /// boundary. When wrapping, subtract the old bucket's data from
    /// the running totals to maintain the 20-second window.
    private func pruneDelayHistory(now: Double) {
        let elapsedSinceBucketStart = now - _buckets[_currentBucketIndex].startTime
        if elapsedSinceBucketStart < 1.0 {
            return
        }

        // Cross as many 1-second boundaries as needed (handles
        // training pauses).
        let bucketsToMove = Int(elapsedSinceBucketStart.rounded(.down))
        for i in 1...min(bucketsToMove, 20) {
            let nextIndex = (_currentBucketIndex + i) % 20
            // Evict the old bucket's contributions from the global sum.
            _historySum -= _buckets[nextIndex].sum
            _historyCount -= _buckets[nextIndex].count
            // Reset for the new second.
            _buckets[nextIndex].reset(startTime: _buckets[_currentBucketIndex].startTime + Double(i))
        }
        _currentBucketIndex = (_currentBucketIndex + bucketsToMove) % 20
    }

    /// Append a rolling-rate sample if at least `rateSampleIntervalSec`
    /// has passed since the last one.
    private func maybeAppendRateSample(now: Date) {
        if let last = _lastRateSampleAt,
           now.timeIntervalSince(last) < rateSampleIntervalSec {
            return
        }
        _rateSamples.append(RateSample(
            at: now,
            selfPlayTotal: _totalSelfPlayPositions,
            trainingTotal: _totalTrainingPositions
        ))
        _lastRateSampleAt = now
        pruneRateSamples(now: now)
    }

    private func pruneRateSamples(now: Date) {
        let cutoff = now.addingTimeInterval(-rateWindowSec)
        while let first = _rateSamples.first, first.at < cutoff {
            _rateSamples.removeFirst()
        }
    }

    /// Rolling production and consumption rates (positions/sec).
    private func rollingRates() -> (production: Double, consumption: Double) {
        pruneRateSamples(now: Date())
        guard let oldest = _rateSamples.first,
              let newest = _rateSamples.last else {
            return (0, 0)
        }
        let dt = newest.at.timeIntervalSince(oldest.at)
        guard dt >= 1.0 else {
            return (0, 0)
        }
        let prod = Double(newest.selfPlayTotal - oldest.selfPlayTotal) / dt
        let cons = Double(newest.trainingTotal - oldest.trainingTotal) / dt
        return (prod, cons)
    }

    /// SMA of the raw signed delay. O(1).
    private func smoothedSignedDelayMs() -> Double {
        pruneDelayHistory(now: CFAbsoluteTimeGetCurrent())
        guard _historyCount > 0 else {
            return _rawSignedDelayMs
        }
        return _historySum / Double(_historyCount)
    }

    /// Training-side projection: if the smoothed signed value is
    /// non-negative, training is the side being throttled. Clamped to
    /// `maxTrainingStepDelayMs`.
    private func smoothedTrainingStepDelayMs() -> Int {
        let signed = smoothedSignedDelayMs()
        guard signed >= 0 else { return 0 }
        return Int(min(Double(maxTrainingStepDelayMs), signed))
    }

    /// Self-play-side projection: if the smoothed signed value is
    /// negative, self-play is the side being throttled. Convert
    /// the signed-per-training-batch quantity into a per-worker
    /// per-game sleep using the aggregate-rate model.
    ///
    /// Derivation — the actual per-move gap we need sp to make up
    /// is `target × tr − sp` (in ms/move), so aggregate sp grows
    /// from its observed value to the equilibrium `target × tr`.
    /// Reconstructing from the signed formula:
    ///     signed = (sp/target − tr) × batch
    ///     −signed/batch = tr − sp/target
    ///     target × (−signed/batch) = target×tr − sp     ← the gap
    /// The extra `target` factor IS REQUIRED — without it we'd
    /// undershoot (or overshoot, for target<1) by exactly the
    /// target ratio itself. Target=1 happens to make the factor
    /// disappear, which is a convenient trap.
    ///
    /// Then the standard aggregate-to-per-worker conversion: with N
    /// workers each playing games of G plies in `T_work` ms of work
    /// and sleeping D ms per game, aggregate ms-per-move is
    /// `(T_work + D) / (N × G)`. To raise it by `gap`, solve for D:
    ///     D = gap × N × G
    /// So the final per-worker per-game sleep is:
    ///     gap × workerCount × positionsPerGame
    /// Clamped to `maxSelfPlayDelayMs`. Mutual exclusivity with
    /// `smoothedTrainingStepDelayMs()` holds because both test the
    /// sign of the same smoothed scalar.
    private func smoothedSelfPlayPerGameDelayMs() -> Int {
        let signed = smoothedSignedDelayMs()
        guard signed < 0 else { return 0 }
        let gapPerMoveMs = _targetRatio * (-signed) / Double(trainingBatchSize)
        let perGameSleepMs =
            gapPerMoveMs
            * Double(_lastSelfPlayWorkerCount)
            * Double(_lastSelfPlayPositionsPerGame)
        let clamped = min(Double(maxSelfPlayDelayMs), max(0, perGameSleepMs))
        return Int(clamped)
    }

    // MARK: - UI snapshot

    struct RatioSnapshot: Sendable {
        /// Aggregate self-play production rate in positions/sec.
        /// Zero before the first self-play event.
        let productionRate: Double
        /// Aggregate training consumption rate in positions/sec.
        /// Zero before the first training event.
        let consumptionRate: Double
        /// `consumptionRate / productionRate`, or 0 if production is 0.
        let currentRatio: Double
        let targetRatio: Double
        let autoAdjust: Bool
        /// Training-side delay (ms per SGD step). Zero when self-play
        /// is the side being throttled.
        let computedDelayMs: Int
        /// Self-play-side delay (ms per game per slot). Zero when
        /// training is the side being throttled.
        let computedSelfPlayDelayMs: Int
        /// Latest observed self-play ms-per-produced-move (aggregate
        /// across all active workers). Nil until the first event.
        let selfPlayMsPerMove: Double?
        /// Latest observed training ms-per-consumed-move. Nil until
        /// the first SGD step.
        let trainingMsPerMove: Double?
        /// Most recent worker count piped in with a self-play event.
        let workerCount: Int
    }

    /// Off-main async variant of `snapshot()`. Lock acquisition runs
    /// on a global executor so the awaiter (typically the main actor)
    /// is never synchronously blocked on `lock.withLock`.
    func asyncSnapshot() async -> RatioSnapshot {
        let start = Date()
        return await withCheckedContinuation { (cont: CheckedContinuation<RatioSnapshot, Never>) in
            let inContinuation = Date()
            DispatchQueue.global(qos: .userInitiated).async {
                let dispatched = Date()
                let result = self.snapshot()
                let now = Date()
                let total = now.timeIntervalSince(start)
                if total > 0.05 {
                    let pre = inContinuation.timeIntervalSince(start)
                    let queue = dispatched.timeIntervalSince(inContinuation)
                    let work = now.timeIntervalSince(dispatched)
                    print(String(format: "[DISPATCH-LATENCY] ReplayRatioController.asyncSnapshot: total=%.2fms (pre=%.2fms queue=%.2fms work=%.2fms)", total*1000, pre*1000, queue*1000, work*1000))
                }
                cont.resume(returning: result)
            }
        }
    }

    func snapshot() -> RatioSnapshot {
        lock.withLock {
            // Production / consumption rates use the 60-s rolling
            // window — what the UI, [STATS] log line, and JSON
            // output all call the "1m" rates. Derived from cumulative
            // position counters, not from the instantaneous per-event
            // ms/move (those stay available separately in the snapshot
            // for Diag-row debug display).
            let (prodRate, consRate) = rollingRates()
            let ratio = prodRate > 0 ? consRate / prodRate : 0
            // Applied delays are the 20s SMA — the same values the
            // training worker and self-play slots actually use. The
            // UI should never show a different number from what is
            // being applied, or the reader can't diagnose behavior.
            let trSmoothed = smoothedTrainingStepDelayMs()
            let spSmoothed = smoothedSelfPlayPerGameDelayMs()
            return RatioSnapshot(
                productionRate: prodRate,
                consumptionRate: consRate,
                currentRatio: ratio,
                targetRatio: _targetRatio,
                autoAdjust: _autoAdjust,
                computedDelayMs: trSmoothed,
                // Gate the external sp delay on auto-adjust so the UI
                // matches what self-play workers actually see from
                // `computedSelfPlayDelayMs`. Internal state is kept so
                // re-enabling auto resumes from the last equilibrium.
                computedSelfPlayDelayMs: _autoAdjust ? spSmoothed : _manualSelfPlayDelayMs,
                selfPlayMsPerMove: _selfPlayMsPerMove,
                trainingMsPerMove: _trainingMsPerMove,
                workerCount: _lastSelfPlayWorkerCount
            )
        }
    }

    // MARK: - UI setters

    var targetRatio: Double {
        get { lock.withLock { _targetRatio } }
        set {
            lock.withLock {
                self._targetRatio = newValue
                self.recomputeDelays()
            }
        }
    }

    var autoAdjust: Bool {
        get { lock.withLock { _autoAdjust } }
        set {
            lock.withLock {
                self._autoAdjust = newValue
                self.recomputeDelays()
            }
        }
    }

    var manualDelayMs: Int {
        get { lock.withLock { _manualDelayMs } }
        set { lock.withLock { self._manualDelayMs = max(0, min(self.maxTrainingStepDelayMs, newValue)) } }
    }

    /// Per-game-per-worker self-play sleep used when `_autoAdjust ==
    /// false`. Symmetric with `manualDelayMs` on the training side.
    /// Clamped to `[0, maxSelfPlayDelayMs]`.
    var manualSelfPlayDelayMs: Int {
        get { lock.withLock { _manualSelfPlayDelayMs } }
        set { lock.withLock { self._manualSelfPlayDelayMs = max(0, min(self.maxSelfPlayDelayMs, newValue)) } }
    }

    /// Training-side delay projection. Always returns the smoothed
    /// (SMA) training sleep, regardless of `_autoAdjust` state — the
    /// Auto-Adjust toggle off-transition reads this to inherit the
    /// most recent auto-computed value into the manual slot, and
    /// gating on `_autoAdjust` would defeat that by returning the
    /// manual value we are about to overwrite. The return value
    /// consulted by the training worker each SGD step is the return
    /// value of `recordTrainingBatchAndGetDelay`, which DOES gate on
    /// `_autoAdjust` — this getter is for UI / mode-transition
    /// bookkeeping.
    var computedDelayMs: Int {
        get {
            lock.withLock { smoothedTrainingStepDelayMs() }
        }
        set {
            lock.withLock {
                let clamped = max(0, min(self.maxTrainingStepDelayMs, newValue))
                let now = CFAbsoluteTimeGetCurrent()
                self._rawSignedDelayMs = Double(clamped)
                self._historySum = Double(clamped)
                self._historyCount = 1
                for i in 0..<20 {
                    _buckets[i].reset(startTime: now - Double(19 - i))
                }
                _currentBucketIndex = 19
                _buckets[19].sum = Double(clamped)
                _buckets[19].count = 1
            }
        }
    }

    /// Clear the self-play wall-clock stamp so the next
    /// `recordSelfPlayBarrierTick` is treated as a fresh first-tick
    /// (stamp only, no measurement). Used by the batched evaluator
    /// when its slot count drops to 0 (drain mode / arena pause)
    /// and on reattach, so a multi-minute pause window doesn't
    /// produce one garbage ms/move sample on resume.
    func resetSelfPlayClock() {
        lock.withLock { _lastSelfPlayTickAt = nil }
    }

    /// Clear the training-side wall-clock stamp. Symmetric with
    /// `resetSelfPlayClock`; not currently called from the in-tree
    /// caller, but exposed for symmetry and future use (e.g. an
    /// arena pause that the training worker observes).
    func resetTrainingClock() {
        lock.withLock { _lastTrainingTickAt = nil }
    }

    /// Self-play-side delay projection. When auto-adjust is on,
    /// returns the smoothed controller-computed sleep. When off,
    /// returns the user-set `_manualSelfPlayDelayMs` so the SP delay
    /// stepper has the same manual override authority the training-
    /// step delay stepper has on its side. The internal smoothing
    /// history is preserved across the toggle so flipping auto back
    /// on resumes from the previous equilibrium point.
    var computedSelfPlayDelayMs: Int {
        lock.withLock { _autoAdjust ? smoothedSelfPlayPerGameDelayMs() : _manualSelfPlayDelayMs }
    }

    /// Always returns the smoothed (SMA) self-play sleep regardless
    /// of `_autoAdjust`, symmetric with `computedDelayMs` on the
    /// training side. Used by the auto-OFF transition to inherit the
    /// last auto-computed value into the manual slot (where reading
    /// `computedSelfPlayDelayMs` after the toggle would already see
    /// the new manual value and defeat the inherit). Not for routine
    /// worker use — the worker reads `computedSelfPlayDelayMs`.
    var smoothedSelfPlayDelayMs: Int {
        lock.withLock { smoothedSelfPlayPerGameDelayMs() }
    }
}
