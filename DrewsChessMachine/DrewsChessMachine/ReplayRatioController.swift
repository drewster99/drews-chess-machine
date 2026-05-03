import Foundation

/// Event-driven replay-ratio controller. Three recording hooks —
/// two that drive control, one that feeds metadata:
///
///   • `recordSelfPlayBarrierTick(positionsProduced:elapsedMs:workerCount:)`
///     — fires every batched-evaluator barrier tick on the self-play
///     side. One tick produces `positionsProduced` plies (equal to
///     the number of slots that submitted) in `elapsedMs` of wall
///     clock since the previous tick. That ratio IS the aggregate
///     self-play ms-per-move — no per-worker division needed, the
///     measurement is already aggregate by construction. Ticks fire
///     at hundreds-to-thousands per second (every ply across all N
///     slots), giving the controller much finer measurement
///     granularity than the old once-per-game cadence.
///
///   • `recordTrainingBatchAndGetDelay(elapsedMs:)` — fires after
///     each SGD step. `elapsedMs` is the EFFECTIVE inter-call wall
///     time — the controller-perspective tick-to-tick period for
///     the training side, with the caller's previously-applied
///     sleep ALREADY SUBTRACTED. The caller is responsible for
///     tracking its own `lastCallTime` and `lastSleptMs` and
///     passing `(now - lastCallTime) - lastSleptMs`. Symmetric to
///     the self-play barrier tick which is also tick-to-tick wall;
///     before this convention the training side passed only the
///     pure SGD batch work time, which excluded both the just-
///     applied sleep and any inter-call non-batch work, producing
///     an asymmetric sp/tr measurement and pushing the controller
///     to drift the equilibrium ratio away from `targetRatio`.
///     Returns the sleep the worker should apply before the next step.
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
/// directly. Each event appends `(timestamp, signed)` to a time-
/// stamped history; the publicly-visible applied delays come from
/// an SMA of that pre-branched signed history, with the sign branch
/// applied AFTER averaging. This preserves the mutual-exclusivity
/// invariant "at most one of training-delay / sp-delay is non-zero
/// at any moment" — averaging the two branched delays separately
/// would mix zeros-from-one-branch with nonzeros-from-the-other and
/// produce both sides sleeping simultaneously during transitions.
/// Smoothing over a fixed 7 s wall-clock window (not a fixed sample
/// count) — batch cadence varies with batch size, so a count window
/// would span wildly different durations.
///
/// There is no measurement window on the input side, no Kp, no
/// damping, no divergence boost, no deadband, no PID — only the
/// model-inverting formula plus the 7-second output smoothing.
///
/// Thread-safe via a private serial `DispatchQueue`. All accessors
/// hop through `queue.sync`.
final class ReplayRatioController: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.replayratiocontroller.serial")

    // MARK: - Configuration

    private var _targetRatio: Double
    private var _autoAdjust: Bool
    private var _manualDelayMs: Int
    /// Size of one SGD training batch, in positions. Used to convert
    /// per-move deltas into per-batch sleep durations.
    let trainingBatchSize: Int
    /// Hard ceiling on the training-side (positive) delay per SGD step.
    let maxTrainingStepDelayMs: Int
    /// Hard ceiling on the self-play-side per-game-per-slot sleep.
    let maxSelfPlayDelayMs: Int

    /// Publicly-visible alias preserved so existing call sites that
    /// read `batchSize` keep compiling. Same value as `trainingBatchSize`.
    var batchSize: Int { trainingBatchSize }

    // MARK: - Internal state

    /// Latest aggregate per-move time on each side, in milliseconds.
    /// `nil` until at least one batch has been recorded on that side.
    /// Both are WORK-TIME-only — no sleep is included in either.
    private var _selfPlayMsPerMove: Double? = nil
    private var _trainingMsPerMove: Double? = nil

    /// Positions flushed in the most recent self-play event. Treated
    /// as `positionsPerGame` when converting the self-play-side
    /// deficit into a per-game-per-worker sleep. Seeded to 100 as a
    /// safe non-zero default for use before the first event arrives.
    private var _lastSelfPlayPositionsPerGame: Int = 100

    /// Active self-play worker count (`N` in the aggregate formula).
    /// Updated on every self-play batch by the driver so the per-game
    /// delay stays correct when the user changes the worker count
    /// stepper mid-session. Seeded to 1 — safe floor until the first
    /// event sets the real value.
    private var _lastSelfPlayWorkerCount: Int = 1

    /// Latest raw signed delay (ms per training batch) as produced by
    /// the most recent `recomputeDelays()`. Positive = training slow;
    /// negative = sp slow. NOT the applied value — the workers read
    /// SMA-smoothed projections via the helpers below. Seeded from
    /// `initialDelayMs` (positive) in init so the first return from
    /// `recordTrainingBatchAndGetDelay` is sensible before any history
    /// has accumulated.
    private var _rawSignedDelayMs: Double

    /// Window (seconds) used for the SMA on the signed delay. Long
    /// enough that a single-batch outlier can't swing the delay hard,
    /// short enough that the controller still tracks real rate changes.
    /// Pinned to 7 s per the user's specification; also covers the
    /// long-tail case where a single SGD batch takes more than one
    /// second of wall clock, so the average still has ≥ several entries.
    private let historyWindowSec: Double = 7.0

    /// Time-stamped history of raw signed-delay values, appended on
    /// every `recomputeDelays()` run. Entries older than
    /// `historyWindowSec` are pruned on both append and read. Typical
    /// steady-state occupancy under 32 self-play slots + 3 SGD
    /// steps/sec is ~5-20k entries in the 7 s window — an array with
    /// leading-prune is still O(1) amortized because ticks arrive in
    /// time order and we only drop from the front. Reduction is
    /// linear in the window size but runs rarely (only on snapshot /
    /// explicit reads), not on every tick append.
    private struct DelayHistoryEntry {
        let at: Date
        let signedMs: Double
    }
    private var _delayHistory: [DelayHistoryEntry] = []

    // MARK: - 1-minute rolling rate estimator

    /// Separate rolling estimator for the UI/log/JSON production and
    /// consumption rates. The control loop uses instantaneous per-
    /// event measurements (`_selfPlayMsPerMove` / `_trainingMsPerMove`)
    /// because control needs the latest information, but display
    /// values should be smoothed over a human time scale so the
    /// reader isn't watching numbers flicker at 2700 Hz.
    ///
    /// Cumulative total positions counters, sampled at ~4 Hz into
    /// `_rateSamples`. Rate = (newest.total − oldest_in_window.total)
    /// / (newest.at − oldest_in_window.at). Window = 60 s.
    ///
    /// Subsampling keeps the buffer bounded at ~240 entries instead
    /// of the ~162k we'd accumulate sampling every barrier tick.
    private let rateWindowSec: Double = 60.0
    private let rateSampleIntervalSec: Double = 0.25
    private var _totalSelfPlayPositions: Int = 0
    private var _totalTrainingPositions: Int = 0
    private struct RateSample {
        let at: Date
        let selfPlayTotal: Int
        let trainingTotal: Int
    }
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
        // Seed the raw SIGNED delay from the passed initial value
        // (positive = training delay) so the first post-construction
        // read returns a sensible training-side number before any
        // batch has been recorded. Self-play per-game delay stays at
        // 0 — there is no UI concept of "initial self-play delay,"
        // and a positive seed keeps the applied projections in the
        // training-slow branch until real measurements arrive.
        let seedSigned = Double(max(0, min(maxTrainingStepDelayMs, initialDelayMs)))
        self._rawSignedDelayMs = seedSigned
        self.maxTrainingStepDelayMs = maxTrainingStepDelayMs
        self.maxSelfPlayDelayMs = maxSelfPlayDelayMs
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
    ///   - elapsedMs: wall clock since the previous barrier tick.
    ///     The full tick-to-tick period — includes one `graph.run`'s
    ///     duration, any barrier-assembly idle time, AND the per-game
    ///     spDelay sleep absorbed by any worker that game-ended in
    ///     the window (the barrier waits for the slowest worker, so
    ///     the wall reflects the longest nap). The controller
    ///     subtracts an estimated overhead — the per-game spDelay it
    ///     is currently asking workers to sleep, multiplied by the
    ///     expected number of game-ends across the pool this tick
    ///     (`positionsProduced / G`) — so what's stored as
    ///     `_selfPlayMsPerMove` is the underlying production rate at
    ///     ZERO injected delay. The signed-delay formula then
    ///     converges on the configured `targetRatio` rather than
    ///     drifting toward the current applied ratio. Negative
    ///     effective elapsed (small-window estimator overshoots) is
    ///     not clamped — the 7s SMA over `_delayHistory` absorbs
    ///     transient noise.
    ///   - workerCount: currently-active slot count (`N` in the
    ///     aggregate sp-slowdown formula). Piped in each tick so the
    ///     per-game sleep calculation stays correct when the user
    ///     changes the worker stepper mid-session.
    func recordSelfPlayBarrierTick(
        positionsProduced: Int,
        elapsedMs: Double,
        workerCount: Int
    ) {
        queue.sync {
            if positionsProduced > 0, elapsedMs > 0 {
                // Estimate sp-side overhead absorbed by this tick's
                // wall: every G plies across the pool, one worker
                // game-ends and naps for `smoothedSelfPlayPerGameDelayMs`.
                // Expected naps in a tick window = positionsProduced
                // / G. Each nap inflates the barrier wall by ~spDelay
                // (since the barrier waits for the slowest). The
                // controller has both numbers without caller help —
                // `smoothedSelfPlayPerGameDelayMs()` returns exactly
                // the value the caller is currently sleeping in
                // BatchedSelfPlayDriver, and `_lastSelfPlayPositionsPerGame`
                // is refreshed by every `recordSelfPlayGameLength`.
                // Subtract this estimate before computing per-move
                // rate so the formula sees the underlying production
                // rate, not the throttled rate.
                let g = Double(max(1, _lastSelfPlayPositionsPerGame))
                let spDelay = _autoAdjust ? Double(smoothedSelfPlayPerGameDelayMs()) : 0
                let estimatedOverheadMs = spDelay * Double(positionsProduced) / g
                let effectiveElapsedMs = elapsedMs - estimatedOverheadMs
                _selfPlayMsPerMove = effectiveElapsedMs / Double(positionsProduced)
                _totalSelfPlayPositions += positionsProduced
                maybeAppendRateSample(now: Date())
            }
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
        queue.sync {
            _lastSelfPlayPositionsPerGame = max(1, plies)
            recomputeDelays()
        }
    }

    /// Called by the training worker after each SGD step. `elapsedMs`
    /// is the EFFECTIVE inter-call wall: caller computes
    /// `(now - lastCallTime) - lastSleptMs` and passes the
    /// difference. This makes the training side symmetric with the
    /// self-play side (both ultimately reflect the underlying
    /// production rate at zero injected delay), so the signed-delay
    /// formula converges on the configured `targetRatio` rather than
    /// the current applied ratio. The caller is responsible for
    /// tracking `lastCallTime` and `lastSleptMs`. On the first call
    /// (no prior, no prior sleep) the caller can pass the SGD
    /// batch's own work time as a close-enough first sample.
    /// Negative `elapsedMs` from small-window estimator overshoot is
    /// not clamped — the 7s SMA absorbs transient noise.
    /// Returns the sleep the worker should apply before the next SGD
    /// step (training-side projection of the SMA signed delay over
    /// the last `historyWindowSec` seconds).
    func recordTrainingBatchAndGetDelay(elapsedMs: Double) -> Int {
        queue.sync {
            if elapsedMs > 0 {
                _trainingMsPerMove = elapsedMs / Double(trainingBatchSize)
                _totalTrainingPositions += trainingBatchSize
                maybeAppendRateSample(now: Date())
            }
            recomputeDelays()
            return _autoAdjust ? smoothedTrainingStepDelayMs() : _manualDelayMs
        }
    }

    /// Closed-form recompute of the RAW signed delay from the latest
    /// per-move estimates, followed by an append to the rolling
    /// history. Only runs once both sides have reported at least one
    /// batch — before that the seeded `_rawSignedDelayMs` from init
    /// stays in effect. The applied (smoothed, then branched) values
    /// are computed on read in the `smoothed*` helpers below.
    private func recomputeDelays() {
        guard _autoAdjust,
              let sp = _selfPlayMsPerMove,
              let tr = _trainingMsPerMove,
              sp > 0, tr > 0,
              _targetRatio > 0 else {
            return
        }

        // User-derived model-inverting formula (names match the spec):
        //   signed = (selfPlayMsPerMove / targetRatio − trainingMsPerMove)
        //            × trainingBatchSize
        //
        // Store ONLY the raw signed value. Branching into training-
        // side vs self-play-side delays happens AFTER the SMA in the
        // `smoothed*` helpers so the mutual-exclusivity invariant
        // "only one side is non-zero at a time" holds even during
        // sign transitions. Averaging the two branched values
        // separately would mix zeros from one branch with nonzeros
        // from the other and produce both sides sleeping simultaneously.
        let signedPerTrainingBatchMs =
            (sp / _targetRatio - tr) * Double(trainingBatchSize)
        _rawSignedDelayMs = signedPerTrainingBatchMs

        let now = Date()
        _delayHistory.append(
            DelayHistoryEntry(at: now, signedMs: signedPerTrainingBatchMs)
        )
        pruneDelayHistory(now: now)
    }

    /// Drop entries older than `historyWindowSec` from the front of
    /// the history. Called on every append and also on every read so
    /// the window stays honest even if the controller hasn't logged
    /// an event in a while (e.g. the training task paused).
    private func pruneDelayHistory(now: Date) {
        let cutoff = now.addingTimeInterval(-historyWindowSec)
        while let first = _delayHistory.first, first.at < cutoff {
            _delayHistory.removeFirst()
        }
    }

    /// Append a rolling-rate sample if at least `rateSampleIntervalSec`
    /// has passed since the last one. Always runs the 60-s window
    /// prune so the buffer stays bounded. Called from both recording
    /// hooks — training and self-play both update the cumulative
    /// counters, and either one crossing the subsample interval is
    /// reason to snapshot both counters together.
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

    /// Rolling production and consumption rates (positions/sec),
    /// averaged over up to `rateWindowSec` of real samples. Returns
    /// `(0, 0)` if we haven't accumulated at least one second of
    /// span in the window yet — short spans produce unstable rates.
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

    /// SMA of the last `historyWindowSec` seconds of raw signed delay.
    /// Falls back to `_rawSignedDelayMs` (the seeded or latest raw
    /// value) when no history has accumulated yet — e.g. startup
    /// before the first event pair, or immediately after an auto-off
    /// drain where all history entries have aged out.
    private func smoothedSignedDelayMs() -> Double {
        pruneDelayHistory(now: Date())
        guard !_delayHistory.isEmpty else {
            return _rawSignedDelayMs
        }
        let sum = _delayHistory.reduce(0.0) { $0 + $1.signedMs }
        return sum / Double(_delayHistory.count)
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

    func snapshot() -> RatioSnapshot {
        queue.sync {
            // Production / consumption rates use the 60-s rolling
            // window — what the UI, [STATS] log line, and JSON
            // output all call the "1m" rates. Derived from cumulative
            // position counters, not from the instantaneous per-event
            // ms/move (those stay available separately in the snapshot
            // for Diag-row debug display).
            let (prodRate, consRate) = rollingRates()
            let ratio = prodRate > 0 ? consRate / prodRate : 0
            // Applied delays are the 7s SMA — the same values the
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
                computedSelfPlayDelayMs: _autoAdjust ? spSmoothed : 0,
                selfPlayMsPerMove: _selfPlayMsPerMove,
                trainingMsPerMove: _trainingMsPerMove,
                workerCount: _lastSelfPlayWorkerCount
            )
        }
    }

    // MARK: - UI setters

    var targetRatio: Double {
        get { queue.sync { _targetRatio } }
        set {
            queue.async {
                self._targetRatio = newValue
                self.recomputeDelays()
            }
        }
    }

    var autoAdjust: Bool {
        get { queue.sync { _autoAdjust } }
        set {
            queue.async {
                self._autoAdjust = newValue
                self.recomputeDelays()
            }
        }
    }

    var manualDelayMs: Int {
        get { queue.sync { _manualDelayMs } }
        set { queue.async { self._manualDelayMs = max(0, min(self.maxTrainingStepDelayMs, newValue)) } }
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
            queue.sync { smoothedTrainingStepDelayMs() }
        }
        set {
            queue.async {
                let clamped = max(0, min(self.maxTrainingStepDelayMs, newValue))
                // Seed BOTH the raw signed latch AND the history so
                // an immediate read returns this value (not an
                // average still dominated by stale pre-toggle
                // entries). Positive signed = training-side delay,
                // which is exactly what this setter is meant to
                // represent.
                self._rawSignedDelayMs = Double(clamped)
                self._delayHistory.removeAll(keepingCapacity: true)
                self._delayHistory.append(
                    DelayHistoryEntry(at: Date(), signedMs: Double(clamped))
                )
            }
        }
    }

    /// Self-play-side delay projection. Read-only; the controller is
    /// the only writer via `recomputeDelays()`. When auto-adjust is
    /// off, returns 0 — the user is managing training delay manually
    /// in that mode and self-play is expected to run unthrottled. The
    /// underlying history still holds the last auto-computed values,
    /// so flipping auto back on resumes from the previous equilibrium
    /// point without a jump.
    var computedSelfPlayDelayMs: Int {
        queue.sync { _autoAdjust ? smoothedSelfPlayPerGameDelayMs() : 0 }
    }
}
