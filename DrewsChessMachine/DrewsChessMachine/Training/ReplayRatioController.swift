import Foundation
import os

/// Event-driven replay-ratio controller. Four recording hooks —
/// three that drive control, one that feeds metadata:
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
///     granularity than the old once-per-game cadence. This event
///     drives the RAW-produced-position counters; emitted positions
///     come from `recordSelfPlayEmittedGame` (see below).
///
///   • `recordSelfPlayEmittedGame(positions:)` — fires once per
///     completed self-play game that the driver decided to KEEP
///     (i.e. survived the per-game `selfPlayDrawKeepFraction`
///     filter). Increments the cumulative emitted-position counter
///     and feeds the rolling-window emitted-rate sample buffer.
///     This is the rate the controller targets: the closed-form
///     formula divides `_selfPlayMsPerMove` by the smoothed
///     emitted-fraction so the trainer slows proportionally when
///     draws are being filtered, keeping `cons / emittedProd` at
///     `targetRatio`. With `selfPlayDrawKeepFraction = 1.0`
///     (default), emittedFraction = 1.0 and behaviour matches
///     the pre-filter design exactly.
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
/// directly. Each event appends `(timestamp, signed)` to a time-
/// stamped history; the publicly-visible applied delays come from
/// an SMA of that pre-branched signed history, with the sign branch
/// applied AFTER averaging. This preserves the mutual-exclusivity
/// invariant "at most one of training-delay / sp-delay is non-zero
/// at any moment" — averaging the two branched delays separately
/// would mix zeros-from-one-branch with nonzeros-from-the-other and
/// produce both sides sleeping simultaneously during transitions.
/// Smoothing over a fixed 20 s wall-clock window (not a fixed sample
/// count) — batch cadence varies with batch size, so a count window
/// would span wildly different durations.
///
/// There is no measurement window on the input side, no Kp, no
/// damping, no divergence boost, no deadband, no PID — only the
/// model-inverting formula plus the 20-second output smoothing.
///
/// Thread-safe via a private `OSAllocatedUnfairLock`. All accessors
/// hop through `lock.withLock`. The lock is never held across an
/// `await`, and inner helpers like `recomputeDelays`,
/// `pruneDelayHistory`, `pruneRateSamples`, `maybeAppendRateSample`,
/// and the `smoothed*` projections assume the caller already holds
/// the lock — they must never re-enter `withLock` (that would trap;
/// `OSAllocatedUnfairLock` isn't recursive).
final class ReplayRatioController: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock()

    // MARK: - Configuration

    private var _targetRatio: Double
    private var _autoAdjust: Bool
    private var _manualDelayMs: Int
    /// User-set per-game-per-worker self-play delay used when
    /// `_autoAdjust == false`. Mirrors `_manualDelayMs` for the
    /// training side. When auto is on this value is dormant; the
    /// controller's own SP-side smoothing drives the worker sleep.
    /// Seeded to 0 so a fresh controller in manual mode runs SP
    /// unthrottled until the user dials in a non-zero value.
    private var _manualSelfPlayDelayMs: Int = 0
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

    /// Wall-clock stamp of the previous self-play barrier-tick callback.
    /// The controller owns the inter-tick wall measurement directly so
    /// the basis is symmetric with the training side and so the reading
    /// captures EVERYTHING that happens between callbacks (probes, BN
    /// syncs, queue waits, our own injected sleeps), not just whatever
    /// slice the caller chose to time. nil before the first tick — that
    /// first call only stamps and returns; the first usable measurement
    /// arrives on tick #2.
    private var _lastSelfPlayTickAt: CFAbsoluteTime? = nil

    /// Wall-clock stamp of the previous `recordTrainingBatchAndGetDelay`
    /// call. Same rationale as `_lastSelfPlayTickAt` — the controller's
    /// own clock is the only honest accounting for one full training
    /// "cycle" (batch work + post-step sleep + any inter-call work).
    private var _lastTrainingTickAt: CFAbsoluteTime? = nil

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
    /// Pinned to 20 s per the user's specification; also covers the
    /// long-tail case where a single SGD batch takes more than one
    /// second of wall clock, so the average still has ≥ several entries.
    private let historyWindowSec: Double = 20.0

    /// Time-stamped history of raw signed-delay values, appended on
    /// every `recomputeDelays()` run. Entries older than
    /// `historyWindowSec` are pruned on both append and read. Typical
    /// steady-state occupancy under 32 self-play slots + 3 SGD
    /// steps/sec is ~14-57k entries in the 20 s window — an array with
    /// leading-prune is still O(1) amortized because ticks arrive in
    /// time order and we only drop from the front. Reduction is
    /// linear in the window size but runs rarely (only on snapshot /
    /// explicit reads), not on every tick append.
    private struct DelayHistoryEntry {
        let at: Date
        let signedMs: Double
    }
    private var _delayHistory: [DelayHistoryEntry] = []
    private var _delayHistoryHead: Int = 0
    private var _delayHistoryRunningSum: Double = 0.0

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
    ///
    /// We track TWO self-play position counters: the raw produced
    /// count (every ply that came off the GPU, fed by
    /// `recordSelfPlayBarrierTick`) and the emitted count (plies
    /// that survived the per-game keep/drop filter, fed by
    /// `recordSelfPlayEmittedGame`). The replay-ratio target applies
    /// to the EMITTED rate; the raw count is surfaced separately for
    /// telemetry / debugging only.
    private let rateWindowSec: Double = 60.0
    private let rateSampleIntervalSec: Double = 0.25
    private var _totalSelfPlayPositionsProduced: Int = 0
    private var _totalSelfPlayPositionsEmitted: Int = 0
    private var _totalTrainingPositions: Int = 0
    private struct RateSample {
        let at: Date
        let selfPlayProducedTotal: Int
        let selfPlayEmittedTotal: Int
        let trainingTotal: Int
    }
    private var _rateSamples: [RateSample] = []
    private var _rateSamplesHead: Int = 0
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
                // Raw per-produced-ply ms. The closed-form formula
                // wants per-EMITTED-ply ms (so the trainer slows when
                // draws are being filtered out of the buffer) — apply
                // the emittedFraction correction in `recomputeDelays`.
                _selfPlayMsPerMove = effectiveElapsedMs / Double(positionsProduced)
                _totalSelfPlayPositionsProduced += positionsProduced
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

    /// Called once per completed self-play game that the driver
    /// flushed into the replay buffer (i.e. survived the per-game
    /// keep/drop filter). Drives the cumulative emitted-positions
    /// counter and the rolling 60-s emitted-rate display, which is
    /// what `[STATS]`'s `prod=` value reports and what the replay-
    /// ratio target gets compared against. Dropped games don't fire
    /// this hook — the implicit emitted-fraction signal is therefore
    /// observed end-to-end, not configured, so dynamic things
    /// (worker-count changes affecting draw rate, weight evolution
    /// affecting draw rate) get tracked correctly.
    func recordSelfPlayEmittedGame(positions: Int) {
        guard positions > 0 else { return }
        lock.withLock {
            _totalSelfPlayPositionsEmitted += positions
            maybeAppendRateSample(now: Date())
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
        // `sp` is measured as ms / produced-ply; the replay-ratio
        // target applies to EMITTED-ply throughput. Scale up by
        // `1 / emittedFraction` to get ms / emitted-ply. With default
        // `selfPlayDrawKeepFraction = 1.0` this is a no-op; with
        // filtering active the trainer slows proportionally so the
        // consumed/emitted ratio stays at `targetRatio`.
        //
        // Store ONLY the raw signed value. Branching into training-
        // side vs self-play-side delays happens AFTER the SMA in the
        // `smoothed*` helpers so the mutual-exclusivity invariant
        // "only one side is non-zero at a time" holds even during
        // sign transitions. Averaging the two branched values
        // separately would mix zeros from one branch with nonzeros
        // from the other and produce both sides sleeping simultaneously.
        let emittedFraction = smoothedEmittedFraction()
        let spEmittedMsPerMove = sp / emittedFraction
        let signedPerTrainingBatchMs =
            (spEmittedMsPerMove / _targetRatio - tr) * Double(trainingBatchSize)
        _rawSignedDelayMs = signedPerTrainingBatchMs

        let now = Date()
        _delayHistory.append(
            DelayHistoryEntry(at: now, signedMs: signedPerTrainingBatchMs)
        )
        _delayHistoryRunningSum += signedPerTrainingBatchMs
        pruneDelayHistory(now: now)
    }

    /// Drop entries older than `historyWindowSec` from the front of
    /// the history. Called on every append and also on every read so
    /// the window stays honest even if the controller hasn't logged
    /// an event in a while (e.g. the training task paused).
    ///
    /// `_delayHistoryRunningSum` (Double) drifts by a tiny round-off
    /// each add/subtract, so it's re-derived exactly from the
    /// surviving window whenever the backing array is compacted.
    /// That bounds the drift to "since the last compaction" rather
    /// than "since the session started".
    private func pruneDelayHistory(now: Date) {
        let cutoff = now.addingTimeInterval(-historyWindowSec)
        while _delayHistoryHead < _delayHistory.count, _delayHistory[_delayHistoryHead].at < cutoff {
            _delayHistoryRunningSum -= _delayHistory[_delayHistoryHead].signedMs
            _delayHistoryHead += 1
        }

        // Amortized O(1) compaction to avoid memory growth and O(N) shifts.
        if _delayHistoryHead > _delayHistory.count / 2 && _delayHistoryHead > 1000 {
            _delayHistory.removeSubrange(0..<_delayHistoryHead)
            _delayHistoryHead = 0
            _delayHistoryRunningSum = _delayHistory.reduce(0.0) { $0 + $1.signedMs }
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
            selfPlayProducedTotal: _totalSelfPlayPositionsProduced,
            selfPlayEmittedTotal: _totalSelfPlayPositionsEmitted,
            trainingTotal: _totalTrainingPositions
        ))
        _lastRateSampleAt = now
        pruneRateSamples(now: now)
    }

    private func pruneRateSamples(now: Date) {
        let cutoff = now.addingTimeInterval(-rateWindowSec)
        while _rateSamplesHead < _rateSamples.count, _rateSamples[_rateSamplesHead].at < cutoff {
            _rateSamplesHead += 1
        }

        if _rateSamplesHead > _rateSamples.count / 2 && _rateSamplesHead > 100 {
            _rateSamples.removeSubrange(0..<_rateSamplesHead)
            _rateSamplesHead = 0
        }
    }

    /// Rolling production and consumption rates (positions/sec),
    /// averaged over up to `rateWindowSec` of real samples. Returns
    /// all-zero if we haven't accumulated at least one second of
    /// span in the window yet — short spans produce unstable rates.
    ///
    /// `produced` is the raw GPU-side ply rate (every ply that came
    /// off the network, kept or dropped). `emitted` is the rate of
    /// plies that actually landed in the replay buffer. The
    /// replay-ratio target compares `consumption / emitted`.
    private func rollingRates() -> (produced: Double, emitted: Double, consumption: Double) {
        pruneRateSamples(now: Date())
        guard _rateSamplesHead < _rateSamples.count,
              let newest = _rateSamples.last else {
            return (0, 0, 0)
        }
        let oldest = _rateSamples[_rateSamplesHead]
        let dt = newest.at.timeIntervalSince(oldest.at)
        guard dt >= 1.0 else {
            return (0, 0, 0)
        }
        let prod = Double(newest.selfPlayProducedTotal - oldest.selfPlayProducedTotal) / dt
        let emit = Double(newest.selfPlayEmittedTotal - oldest.selfPlayEmittedTotal) / dt
        let cons = Double(newest.trainingTotal - oldest.trainingTotal) / dt
        return (prod, emit, cons)
    }

    /// Smoothed emitted-fraction over the rolling rate window:
    /// `recent_emitted / recent_produced`. Clamped to (0, 1].
    /// Defaults to 1.0 when:
    ///   • no rate samples have accumulated yet, or
    ///   • the window contains no produced positions, or
    ///   • the window contains no emitted positions (would push the
    ///     correction factor toward infinity and oscillate the
    ///     controller — at startup we want stability, not extreme
    ///     correction).
    /// Caller must hold `lock`. Used by `recomputeDelays` to convert
    /// `_selfPlayMsPerMove` (per-PRODUCED-ply) into per-EMITTED-ply
    /// before plugging into the closed-form delay formula.
    private func smoothedEmittedFraction() -> Double {
        pruneRateSamples(now: Date())
        guard _rateSamplesHead < _rateSamples.count,
              let newest = _rateSamples.last else {
            return 1.0
        }
        let oldest = _rateSamples[_rateSamplesHead]
        let producedDelta = Double(newest.selfPlayProducedTotal - oldest.selfPlayProducedTotal)
        let emittedDelta = Double(newest.selfPlayEmittedTotal - oldest.selfPlayEmittedTotal)
        guard producedDelta > 0, emittedDelta > 0 else {
            return 1.0
        }
        let raw = emittedDelta / producedDelta
        return min(1.0, max(0.0001, raw))
    }

    /// SMA of the last `historyWindowSec` seconds of raw signed delay.
    /// Falls back to `_rawSignedDelayMs` (the seeded or latest raw
    /// value) when no history has accumulated yet — e.g. startup
    /// before the first event pair, or immediately after an auto-off
    /// drain where all history entries have aged out.
    private func smoothedSignedDelayMs() -> Double {
        pruneDelayHistory(now: Date())
        let count = _delayHistory.count - _delayHistoryHead
        guard count > 0 else {
            return _rawSignedDelayMs
        }
        return _delayHistoryRunningSum / Double(count)
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
        // `gapPerMoveMs` is the emitted-ply ms/move we need to add.
        // The per-game sleep slows raw production; convert from
        // per-emitted-ply to per-produced-ply via `/ emittedFraction`
        // so the actual emitted slowdown matches the target.
        // When emittedFraction = 1.0 (default), this is a no-op.
        let emittedFraction = smoothedEmittedFraction()
        let gapPerMoveMs = _targetRatio * (-signed) / Double(trainingBatchSize)
        let perGameSleepMs =
            gapPerMoveMs
            * Double(_lastSelfPlayWorkerCount)
            * Double(_lastSelfPlayPositionsPerGame)
            / emittedFraction
        let clamped = min(Double(maxSelfPlayDelayMs), max(0, perGameSleepMs))
        return Int(clamped)
    }

    // MARK: - UI snapshot

    struct RatioSnapshot: Sendable {
        /// Aggregate self-play EMITTED-positions rate in positions/sec
        /// — i.e. plies that survived the per-game keep/drop filter
        /// and were flushed into the replay buffer. Zero before the
        /// first emitted-game event. This is the rate the replay-
        /// ratio controller targets; `currentRatio = consumptionRate
        /// / productionRate`.
        let productionRate: Double
        /// Aggregate self-play RAW-produced rate in positions/sec
        /// (every ply that came off the GPU, kept or dropped). Equal
        /// to `productionRate` when `selfPlayDrawKeepFraction = 1.0`
        /// (default — nothing is being filtered). Surfaced separately
        /// so the operator can read off the effective keep-fraction
        /// as `productionRate / producedRate`.
        let producedRate: Double
        /// Aggregate training consumption rate in positions/sec.
        /// Zero before the first training event.
        let consumptionRate: Double
        /// `consumptionRate / productionRate` (emitted), or 0 if
        /// emitted production is 0.
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
    ///
    /// Emits a `[DISPATCH-LATENCY]` line to stdout (Xcode console only,
    /// not the session log) when the round-trip exceeds 50 ms, broken
    /// into pre-continuation / queue-wait / lock-work components so a
    /// multi-second heartbeat tick can be attributed to a specific
    /// off-main snapshot read. Matches the instrumentation on
    /// `TournamentLiveBox.asyncSnapshot` / `GameDiversityTracker.asyncSnapshot`.
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
                    print(String(format: "[DISPATCH-LATENCY] ReplayRatioController.asyncSnapshot: total=%.2fms (pre=%.2fms queue=%.2fms work=%.2fms)", total * 1000, pre * 1000, queue * 1000, work * 1000))
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
            //
            // `prodRate` (telemetry) is the raw GPU output. `emitRate`
            // is what's actually landing in the replay buffer — that's
            // the rate the controller targets, so it's reported as
            // `productionRate` in the snapshot. The ratio = cons /
            // emitted.
            let (producedRateValue, emittedRate, consRate) = rollingRates()
            let ratio = emittedRate > 0 ? consRate / emittedRate : 0
            // Applied delays are the 20s SMA — the same values the
            // training worker and self-play slots actually use. The
            // UI should never show a different number from what is
            // being applied, or the reader can't diagnose behavior.
            let trSmoothed = smoothedTrainingStepDelayMs()
            let spSmoothed = smoothedSelfPlayPerGameDelayMs()
            return RatioSnapshot(
                productionRate: emittedRate,
                producedRate: producedRateValue,
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
                // Seed BOTH the raw signed latch AND the history so
                // an immediate read returns this value (not an
                // average still dominated by stale pre-toggle
                // entries). Positive signed = training-side delay,
                // which is exactly what this setter is meant to
                // represent.
                self._rawSignedDelayMs = Double(clamped)
                self._delayHistory.removeAll(keepingCapacity: true)
                self._delayHistoryHead = 0
                self._delayHistoryRunningSum = Double(clamped)
                self._delayHistory.append(
                    DelayHistoryEntry(at: Date(), signedMs: Double(clamped))
                )
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
