import Darwin
import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - Errors

enum ChessTrainerError: LocalizedError {
    case lossOutputMissing
    case gradientMissing(String)
    case nonFiniteLoss(total: Float, policy: Float, value: Float, gradNorm: Float)

    var errorDescription: String? {
        switch self {
        case .lossOutputMissing:
            return "Training step ran but loss tensor was not in the result map"
        case .gradientMissing(let name):
            return "Gradient missing for variable: \(name)"
        case .nonFiniteLoss(let total, let policy, let value, let gradNorm):
            return "Non-finite loss detected: total=\(total) policy=\(policy) value=\(value) gradNorm=\(gradNorm). Weights from this step are likely poisoned; training halted."
        }
    }
}

// MARK: - Training Step Timing

/// Per-step timing breakdown. All values in milliseconds.
struct TrainStepTiming: Sendable {
    /// CPU work to synthesize random inputs and pack them into MPSGraphTensorData.
    let dataPrepMs: Double
    /// GPU graph.run() — forward + backward + SGD weight updates.
    let gpuRunMs: Double
    /// CPU work to read the loss scalars back from the tensor results.
    let readbackMs: Double
    /// Total wall-clock time for the whole step.
    let totalMs: Double
    /// Total loss (policy + value) reported by the graph — what SGD minimizes.
    /// Lets us spot NaNs / explosions at a glance.
    let loss: Float
    /// Policy-only component of the loss. Outcome-weighted cross-entropy; can
    /// be negative when the played move already has high probability under a
    /// winning outcome, so it's unbounded on both sides and expected to be
    /// noisier than the value term.
    let policyLoss: Float
    /// Value-only component of the loss. Mean-squared error of (z − v), so
    /// bounded in [0, 4] — if it oscillates, training is genuinely unstable.
    let valueLoss: Float
    /// Mean Shannon entropy (in nats) of the trainee's policy softmax over
    /// this batch. Diagnostic only — not part of `loss`. Range is
    /// [0, log(policySize)] ≈ [0, 8.49] for the current 4864-logit head.
    /// Random init sits near the ceiling; a collapsed policy heads toward 0.
    /// Watch for monotonic drift to either extreme — that's the signature
    /// of policy collapse or a stuck-at-uniform learning failure.
    let policyEntropy: Float
    let policyNonNegligibleCount: Float
    /// Mean per-position count of ILLEGAL cells whose unmasked
    /// softmax probability exceeds 1/policySize. Healthy networks
    /// (legal mask doing its job) push this toward 0; a rising
    /// value signals mass leaking onto illegal cells. Pairs with
    /// `policyNonNegligibleCount` (which counts legal cells) on the
    /// "Above-uniform policy count" chart.
    let policyNonNegligibleIllegalCount: Float
    /// Global L2 norm of the flattened gradient vector across every
    /// trainable variable, computed on the GPU before clipping. When
    /// the value exceeds `ChessTrainer.gradClipMaxNorm`, the update
    /// step scales all gradients by `maxNorm / norm` so the effective
    /// step size is capped. Diagnostic only — the clip is already
    /// applied inside the graph. A value above `gradClipMaxNorm` is a
    /// clip event; steady values above it signal persistent overshoot
    /// that warrants a lower LR.
    let gradGlobalNorm: Float
    /// Mean of the value-head output across the batch. Value head is
    /// tanh-squashed so the output lives in [-1, +1]; a healthy batch
    /// of self-play positions should sit near 0 (most positions are
    /// drawn at early training). Drifting away from 0 signals the
    /// value head is learning a bias — if it sticks at ±1 the head
    /// has saturated and gradients through tanh are vanishing.
    let valueMean: Float
    /// Mean of |value-head output| across the batch. Together with
    /// `valueMean` this is the cheapest saturation probe available:
    /// if `valueAbsMean` ≈ 1.0 the head is saturated everywhere
    /// regardless of position, even if `valueMean` happens to sit
    /// near 0. Range [0, 1].
    let valueAbsMean: Float

    /// Mean absolute delta between the fresh per-position v(s)
    /// computed by the trainer's CURRENT value head and the stale
    /// vBaseline that was stored in the replay buffer at play-time
    /// (the champion's frozen-at-random-init value head). Nil on the
    /// random-data sweep path which has no meaningful play-time
    /// vBaseline. A rising delta shows the trainer's value head has
    /// shifted away from the play-time champion's view — i.e., the
    /// trainer is genuinely diverging.
    let vBaselineDelta: Float?

    /// Wall-clock cost of the fresh-baseline forward pass added to
    /// each real-data training step. Nil on paths that skip it
    /// (random-data sweep). Diagnostic only — included in `totalMs`.
    let freshBaselineMs: Double?

    /// L2 norm (sqrt of sum of squares) of the policy head's final
    /// 1×1 conv weight tensor (128 → 76). Tracks whether the L2
    /// weight decay is actually holding the weights that produce the
    /// policy logits in check. Monotonic growth here, especially if
    /// logit gaps on the Candidate Test panel look extreme, means the
    /// weight-decay coefficient is too small relative to the learning
    /// rate to balance the pull from pLoss. Read via a graph targetTensor
    /// rather than computed on CPU so the host never pulls the 9.8K
    /// float weight tensor back from the GPU just to measure it.
    let policyHeadWeightNorm: Float

    /// Batch mean of `max_i |logits[i]|` — the typical largest raw
    /// logit magnitude in absolute value. Pairs with `policyEntropy`:
    /// entropy can look healthy while a single runaway logit is
    /// already pre-saturating the softmax. Watch for monotonic growth
    /// much faster than `policyHeadWeightNorm`.
    let policyLogitAbsMax: Float

    /// Batch mean of softmax probability on the actually-played move.
    /// **Direction is undefined under the advantage-normalized policy
    /// loss** in this trainer: adv_normalized has zero batch-mean by
    /// construction, so ~half the positions push `p(a*)` up and ~half
    /// push it down. The unconditional mean can stay near
    /// `1/policySize ≈ 0.0002` even when training is perfectly healthy,
    /// and can rise spuriously on outcome-skewed batches where the
    /// `/σ[A]` normalization amplifies tail updates. Keep for backward
    /// compatibility with prior logs and as a coarse index-mismatch
    /// probe (both conditionals flat near `1/policySize` is strong
    /// evidence of action-index misalignment), but read
    /// `playedMoveProbPosAdv` / `playedMoveProbNegAdv` for the real
    /// direction-of-learning signal. Computed graph-side as
    /// `sum(softmax * oneHot)` along the class axis, then mean over
    /// batch. Zero extra readback.
    let playedMoveProb: Float

    /// Batch-conditional mean of `p(a*)` restricted to positions where
    /// the raw advantage `A = z − vBaseline > 0`. These are the
    /// positions whose REINFORCE update pushes `p(a*)` upward, so
    /// under a correctly-wired loop with a live action-index encoding
    /// this rises monotonically from `~1/policySize` as the policy
    /// sharpens on moves that led to better-than-baseline outcomes.
    /// A plateau near `1/policySize` while `pLoss` moves is the strong
    /// action-index-mismatch signature. NaN when no batch row has
    /// `A > 0` (rare — requires a fully negative-advantage batch).
    let playedMoveProbPosAdv: Float

    /// Batch-conditional mean of `p(a*)` restricted to positions where
    /// the raw advantage `A = z − vBaseline < 0`. Complement of
    /// `playedMoveProbPosAdv`. Under a working loop this *falls* from
    /// `~1/policySize` as the policy learns to place less mass on
    /// moves that led to worse-than-baseline outcomes. Combined with
    /// the `PosAdv` conditional, the two move in opposite directions
    /// — that divergence is the actual health signal, not the
    /// unconditional mean. NaN when no batch row has `A < 0`.
    let playedMoveProbNegAdv: Float

    /// Advantage summary for this batch. `advantageMean` is the batch
    /// mean of `A = z − v_baseline`; with a perfect baseline it sits
    /// near zero. `advantageStd` is the batch stdev — large values
    /// mean high-variance policy-gradient updates. `advantageMin` /
    /// `advantageMax` capture the tails. `advantageFracPositive` is
    /// the fraction of positions with A > 0 (positions where REINFORCE
    /// pushes p(a*) up); `advantageFracSmall` is the fraction with
    /// `|A| < 0.05` — "near-zero-signal" positions whose gradient
    /// contribution is tiny. Computed graph-side as scalar reductions.
    let advantageMean: Float
    let advantageStd: Float
    let advantageMin: Float
    let advantageMax: Float
    let advantageFracPositive: Float
    let advantageFracSmall: Float

    /// Per-position advantage values for this step, one float per
    /// batch row. Used by `TrainingLiveStatsBox` to maintain a rolling
    /// window of raw values for p05/p50/p95 percentile computation.
    /// Readback cost is a single `[batch, 1]` tensor (~2 KB at
    /// batch=512) — trivial against the existing per-step readback
    /// budget. Nil on the random-data sweep path (percentile view is
    /// meaningless there).
    let advantageRaw: [Float]?

    /// Mean policy loss over the batch positions where outcome z > 0.5
    /// (the position came from a winning game). Splitting the
    /// classic `policyLoss` by outcome makes the curve unambiguous:
    /// `policyLossWin` should drift negative as the network learns to
    /// favor moves played in winning games. Nil only when no batch
    /// position satisfies the predicate (rare — a batch with zero
    /// wins).
    let policyLossWin: Float?
    /// Mean policy loss over the batch positions where z < -0.5.
    /// Should drift negative as well, since the advantage-weighted
    /// CE term flips sign for losing positions: low p(a*) on a loss
    /// is rewarded ("don't repeat that move"). Nil when no batch
    /// position has a loss outcome.
    let policyLossLoss: Float?
}

// MARK: - Sweep Result

/// Either a measured row or a row we refused to run because it would
/// have blown past the device's working-set or single-buffer caps.
/// Once a sweep skips one batch size, every larger size is also skipped
/// (memory only grows from there) so the table still has one entry per
/// requested batch size — the skipped ones just carry our estimates
/// instead of timings.
enum SweepRow: Sendable {
    case completed(SweepResult)
    case skipped(SkippedRow)

    var batchSize: Int {
        switch self {
        case .completed(let r): return r.batchSize
        case .skipped(let r): return r.batchSize
        }
    }
}

/// A batch size we declined to actually run because our footprint estimate
/// would exceed the device caps. Carries the estimate so the caller can
/// show *why* it was skipped.
struct SkippedRow: Sendable {
    let batchSize: Int
    /// Estimated total working-set bytes for one training step at this batch size.
    let estimatedBytes: UInt64
    /// Estimated size in bytes of the largest single MTLBuffer we'd allocate.
    let largestBufferBytes: UInt64
    /// Which cap we tripped (or both).
    let exceededWorkingSet: Bool
    let exceededBufferLength: Bool
}

/// Snapshot of the Metal device's memory caps. Captured once at sweep
/// start so the UI can show "here's the ceiling and how close we are".
struct DeviceMemoryCaps: Sendable {
    let recommendedMaxWorkingSet: UInt64
    let currentAllocated: UInt64
    let maxBufferLength: UInt64
}

/// Cumulative CPU and GPU time for the current process at a single
/// wall-clock instant. Subtract two samples to compute %CPU / %GPU
/// over the interval between them:
///
/// ```
/// let wallS = cur.timestamp.timeIntervalSince(prev.timestamp)
/// let cpuPct = Double(cur.cpuNs - prev.cpuNs) / (wallS * 1e9) * 100
/// ```
///
/// Percentages follow the `top` / Activity Monitor convention — they
/// are relative to one core / one GPU engine, so a fully loaded
/// multi-core CPU can report well over 100%, and a multi-engine GPU
/// can too. `cpuNs` sums user + system time; `gpuNs` sums across all
/// GPU engines for this process.
struct ProcessUsageSample: Sendable {
    /// Wall-clock instant this sample was taken. Serves as the
    /// denominator when converting nanosecond counters into a
    /// percentage over an interval.
    let timestamp: Date
    /// Cumulative user + system CPU time for this process, in
    /// nanoseconds. Read from `proc_pid_rusage(RUSAGE_INFO_V4)`,
    /// which documents both fields as nanoseconds and accumulates
    /// across every thread the process has ever spawned.
    let cpuNs: UInt64
    /// Cumulative GPU execution time for this process, in
    /// nanoseconds. Read from `task_info(TASK_POWER_INFO_V2)` —
    /// `gpu_energy.task_gpu_utilisation`, which the kernel
    /// populates from each thread's `gpu_ns` counter summed
    /// across all GPU engines.
    let gpuNs: UInt64
}

/// One row of a batch-size sweep — what we measured at one fixed batch size.
struct SweepResult: Sendable {
    let batchSize: Int
    /// Wall-clock time of the very first step at this batch size. Includes
    /// MPSGraph kernel compilation; useful to see when the JIT recompiles.
    let warmupMs: Double
    /// Number of post-warmup steps timed.
    let steps: Int
    /// Wall-clock seconds for those `steps` steps.
    let elapsedSec: Double
    /// Mean per-step total wall time across the timed steps.
    let avgStepMs: Double
    /// Mean GPU run time (subset of avgStepMs) across the timed steps.
    let avgGpuMs: Double
    /// Effective per-second training throughput. The headline number — this
    /// is what the user actually wants to compare across batch sizes.
    let positionsPerSec: Double
    /// Last loss value at this batch size, for sanity checking.
    let lastLoss: Float
    /// Peak `phys_footprint` (process-wide resident memory, including
    /// everything Metal pulled into the unified-memory pool) sampled
    /// across this row's run. Sampled by the UI heartbeat ~10× per second
    /// while the row is in flight, plus once at row start and once at
    /// row end so even very fast rows get at least two readings. This is
    /// what we feed into the linear fit that predicts subsequent rows.
    let peakResidentBytes: UInt64
}

// MARK: - Continuous Training Stats

/// Aggregated stats over a continuous training run. Updated after every step.
///
/// All time-based fields measure **training wall time only** — i.e. the sum
/// of `TrainStepTiming.totalMs` across recorded steps. This excludes
/// self-play and any idle gaps between steps, so in the real-training
/// driver (which alternates play with train) these numbers reflect
/// trainer throughput rather than session wall clock. In pure-training
/// modes they're essentially identical to session elapsed.
struct TrainingRunStats: Sendable {
    var steps: Int = 0
    var totalGpuMs: Double = 0
    var totalStepMs: Double = 0
    var minStepMs: Double = .infinity
    var maxStepMs: Double = 0
    var lastTiming: TrainStepTiming?

    mutating func record(_ t: TrainStepTiming) {
        steps += 1
        totalGpuMs += t.gpuRunMs
        totalStepMs += t.totalMs
        if t.totalMs < minStepMs { minStepMs = t.totalMs }
        if t.totalMs > maxStepMs { maxStepMs = t.totalMs }
        lastTiming = t
    }

    var avgStepMs: Double { steps > 0 ? totalStepMs / Double(steps) : 0 }
    var avgGpuMs: Double { steps > 0 ? totalGpuMs / Double(steps) : 0 }
    /// Wall-clock seconds actually spent inside `trainStep` calls.
    var trainingSeconds: Double { totalStepMs / 1000 }
    /// Training throughput in steps per second of real training time.
    var stepsPerSecond: Double {
        totalStepMs > 0 ? Double(steps) * 1000 / totalStepMs : 0
    }

    /// Training throughput in positions per second of real training time,
    /// for a given batch size. Callers pass the batch size rather than
    /// storing it on the stats struct so the same type works across the
    /// random-data path, the real-data path, and any future variable-
    /// batch paths.
    func positionsPerSecond(batchSize: Int) -> Double {
        stepsPerSecond * Double(batchSize)
    }

    /// Projected wall time for one "epoch" of 250 batches, based on average step time.
    var projectedSecPer250Steps: Double { avgStepMs * 250 / 1000 }
}

// MARK: - Training Live Stats Box

/// Lock-protected holder for live training stats, shared between a
/// background training task (writer) and the UI heartbeat (reader).
///
/// Same design as `CancelBox` for the sweep: the worker calls
/// `recordStep(_:)` after each `trainStep`, which takes the lock briefly,
/// updates the running `TrainingRunStats`, and returns — no main-actor
/// hop per step. The SwiftUI `snapshotTimer` polls `snapshot()` at ~10 Hz
/// and mirrors the current values into `@State`, which is what actually
/// triggers view redraws. This decouples view-update frequency from
/// training-step rate: a 20 ms/step training loop used to fire 50
/// `MainActor.run` hops per second, now it fires zero.
///
/// The rolling-loss windows live here rather than on the view so the
/// worker can maintain them without any main-actor round-trips. Policy
/// and value losses are tracked in separate windows so the UI can show
/// which head is oscillating — a bounded value MSE moving 5× means
/// genuinely unstable training, while a noisy policy term alone is
/// usually just metric noise from outcome-weighted CE.
///
/// Marked `@unchecked Sendable` for the same reason as `CancelBox` and
/// `ReplayBuffer`: a private serial `DispatchQueue` serializes all
/// state mutation and snapshot reads. Writers (`recordStep`, `seed`,
/// `recordError`, `resetRollingWindows`) dispatch asynchronously so
/// the training worker never blocks on the UI heartbeat's snapshot
/// read, and vice-versa.
final class TrainingLiveStatsBox: @unchecked Sendable {
    private struct RollingDoubleWindow: Sendable {
        private var storage: [Double]
        private var head: Int = 0
        private var count: Int = 0
        private var sum: Double = 0

        init(limit: Int) {
            precondition(limit > 0, "Rolling window must be positive")
            self.storage = [Double](repeating: 0, count: limit)
        }

        mutating func append(_ value: Double) {
            if count < storage.count {
                storage[count] = value
                sum += value
                count += 1
                return
            }
            sum -= storage[head]
            storage[head] = value
            sum += value
            head += 1
            if head == storage.count { head = 0 }
        }

        mutating func removeAll() {
            head = 0
            count = 0
            sum = 0
        }

        var mean: Double? {
            guard count > 0 else { return nil }
            return sum / Double(count)
        }

        /// Running sum of every value currently in the window. Paired
        /// with `size` this lets callers expose exact counts for
        /// 0/1-valued windows (e.g. the per-step skip markers for the
        /// advantage-conditional played-move probabilities, where each
        /// step appends either 0 or 1 and the total is a skip count).
        var total: Double { sum }

        /// Number of values currently in the window (capped at the
        /// window limit). Useful as the denominator when presenting a
        /// skip count scoped to the live window span.
        var size: Int { count }
    }

    /// Immutable snapshot the UI reads. All fields are value types so
    /// the snapshot is independent of further worker writes.
    struct Snapshot: Sendable {
        let stats: TrainingRunStats
        let lastTiming: TrainStepTiming?
        let rollingPolicyLoss: Double?
        let rollingValueLoss: Double?
        let rollingPolicyEntropy: Double?
        let rollingPolicyNonNegCount: Double?
        let rollingPolicyNonNegIllegalCount: Double?
        let rollingGradGlobalNorm: Double?
        let rollingValueMean: Double?
        let rollingValueAbsMean: Double?
        /// Rolling-window mean of `TrainStepTiming.vBaselineDelta` —
        /// the per-step mean absolute distance between the trainer's
        /// CURRENT v(s) and the play-time-frozen vBaseline that was
        /// stored in the replay buffer when the move was originally
        /// played. Higher = the trainer's value head has drifted
        /// further from the random-init champion's view = the trainer
        /// is genuinely diverging. nil while the random-data sweep
        /// path is the source (no real vBaselines).
        let rollingVBaselineDelta: Double?
        /// Rolling-window mean of `TrainStepTiming.policyHeadWeightNorm`.
        /// Growing over a long run alongside extreme logit concentration
        /// signals weight decay is too weak for the current learning rate.
        let rollingPolicyHeadWeightNorm: Double?
        /// Rolling-window mean of `TrainStepTiming.policyLogitAbsMax`.
        /// Batch-averaged magnitude of the largest raw logit — rises
        /// before entropy collapses, so a sharper pre-saturation signal.
        let rollingPolicyLogitAbsMax: Double?
        /// Rolling-window mean of `TrainStepTiming.playedMoveProb`.
        /// Coarse action-index probe — see the field's docstring for why
        /// the unconditional mean is directionally ambiguous under the
        /// advantage-normalized policy loss.
        let rollingPlayedMoveProb: Double?
        /// Rolling-window means of the advantage-conditional played-move
        /// probabilities. `Pos` should rise and `Neg` should fall under
        /// a working loop; both staying flat near `1/policySize` is the
        /// action-index-mismatch signature.
        let rollingPlayedMoveProbPosAdv: Double?
        let rollingPlayedMoveProbNegAdv: Double?
        /// Count of steps in the current rolling window that skipped
        /// the respective conditional mean because the batch had zero
        /// positions on that side of the advantage sign (pure 0/0 →
        /// NaN). Readers should interpret the rolling conditional means
        /// as averages over `rollingPlayedMoveCondWindowSize − skipped`
        /// samples rather than the full window. Zero before any
        /// training steps have been observed — there is no "unknown"
        /// state for a skip counter.
        let rollingPlayedMoveProbPosAdvSkipped: Int
        let rollingPlayedMoveProbNegAdvSkipped: Int
        /// Total step count the skip counters are scoped to — the
        /// denominator for the "skipped K of N" presentation. Equals
        /// `min(stepsSinceReset, rollingWindow)`. Shared across the
        /// two sign-conditional skip counters since they advance in
        /// lockstep (every step appends to both). Zero before any
        /// training step has been observed.
        let rollingPlayedMoveCondWindowSize: Int
        /// Rolling-window mean of the advantage distribution
        /// summaries. `advMean` near zero and a stable `advStd` is
        /// the expected signature of a working baseline.
        let rollingAdvMean: Double?
        let rollingAdvStd: Double?
        let rollingAdvMin: Double?
        let rollingAdvMax: Double?
        let rollingAdvFracPositive: Double?
        let rollingAdvFracSmall: Double?
        /// Percentiles (p05, p50, p95) of raw per-position advantage
        /// values over the rolling window of recent training steps.
        /// Each call to `snapshot()` copies out the raw-value ring
        /// (~ window × batch floats), sorts, and reads the percentile
        /// positions. Nil while the ring is empty.
        let advantageP05: Double?
        let advantageP50: Double?
        let advantageP95: Double?
        /// Rolling-window mean policy loss restricted to win-outcome
        /// batch positions (z > 0.5). Pairs with `rollingPolicyLossLoss`
        /// to disambiguate the standard `rollingPolicyLoss` curve.
        let rollingPolicyLossWin: Double?
        /// Rolling-window mean policy loss restricted to loss-outcome
        /// batch positions (z < -0.5).
        let rollingPolicyLossLoss: Double?
        let error: String?
    }

    private let queue = DispatchQueue(label: "drewschess.traininglivestatsbox.serial")
    private var _stats = TrainingRunStats()
    private var _lastTiming: TrainStepTiming?
    private var _policyLossWindow: RollingDoubleWindow
    private var _valueLossWindow: RollingDoubleWindow
    private var _policyEntropyWindow: RollingDoubleWindow
    private var _policyNonNegWindow: RollingDoubleWindow
    private var _policyNonNegIllegalWindow: RollingDoubleWindow
    private var _gradNormWindow: RollingDoubleWindow
    private var _valueMeanWindow: RollingDoubleWindow
    private var _valueAbsMeanWindow: RollingDoubleWindow
    private var _vBaselineDeltaWindow: RollingDoubleWindow
    private var _policyHeadWeightNormWindow: RollingDoubleWindow
    private var _policyLogitAbsMaxWindow: RollingDoubleWindow
    private var _playedMoveProbWindow: RollingDoubleWindow
    private var _playedMoveProbPosAdvWindow: RollingDoubleWindow
    private var _playedMoveProbNegAdvWindow: RollingDoubleWindow
    /// Per-step 0/1 skip markers for the advantage-conditional played-
    /// move probabilities. Appended on every training step — 1.0 when
    /// the conditional mean was NaN (zero batch rows on that sign of
    /// the advantage), 0.0 otherwise. `sum` of the window is the
    /// skip count in the window's span; `size` is the denominator.
    /// Having this pair lets callers surface "mean over K/N samples"
    /// rather than silently advertising the conditional mean as if
    /// it reflected every step.
    private var _playedMoveProbPosAdvSkipWindow: RollingDoubleWindow
    private var _playedMoveProbNegAdvSkipWindow: RollingDoubleWindow
    private var _advMeanWindow: RollingDoubleWindow
    private var _advStdWindow: RollingDoubleWindow
    private var _advMinWindow: RollingDoubleWindow
    private var _advMaxWindow: RollingDoubleWindow
    private var _advFracPosWindow: RollingDoubleWindow
    private var _advFracSmallWindow: RollingDoubleWindow
    private var _policyLossWinWindow: RollingDoubleWindow
    private var _policyLossLossWindow: RollingDoubleWindow
    /// Ring of raw per-position advantage values across the rolling
    /// window of recent steps. Capped at `advRawRingMaxCapacity`
    /// floats (see the constant for the full rationale). `snapshot()`
    /// sorts the live set for percentile extraction, and that sort
    /// runs on main via the UI heartbeat's `queue.sync` — a larger
    /// ring was blocking main for ~150 ms per snapshot at 10 Hz,
    /// starving `fireCandidateProbeIfNeeded`'s MainActor hop and
    /// collapsing training throughput to ~300 moves/sec from a
    /// normal 2300 moves/sec.
    private var _advRawRing: [Float] = []
    private var _advRawRingHead: Int = 0
    private var _advRawRingFilled: Int = 0
    private var _advRawRingCapacity: Int = 0
    private var _error: String?
    private let rollingWindow: Int

    /// Hard cap on `_advRawRing` capacity in Float entries. At 32 K
    /// floats the copy + sort inside `percentiles()` runs in ~1 ms
    /// (vs. ~150 ms at the prior 2 M-entry ceiling), yet 32 K samples
    /// already pin the empirical p05/p50/p95 to within ~0.5% of the
    /// true distribution — more than tight enough for a diagnostic
    /// that's eyeballed in logs. Sized so that at the default batch
    /// of 4096 the ring still holds 8 full batches' worth of raw
    /// advantages; at smaller batches the ring is effectively the
    /// `rollingWindow` * batchSize product anyway.
    private static let advRawRingMaxCapacity: Int = 32_768

    init(rollingWindow: Int) {
        precondition(rollingWindow > 0, "Rolling window must be positive")
        self.rollingWindow = rollingWindow
        self._policyLossWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueLossWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyEntropyWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyNonNegWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyNonNegIllegalWindow = RollingDoubleWindow(limit: rollingWindow)
        self._gradNormWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueMeanWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueAbsMeanWindow = RollingDoubleWindow(limit: rollingWindow)
        self._vBaselineDeltaWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyHeadWeightNormWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyLogitAbsMaxWindow = RollingDoubleWindow(limit: rollingWindow)
        self._playedMoveProbWindow = RollingDoubleWindow(limit: rollingWindow)
        self._playedMoveProbPosAdvWindow = RollingDoubleWindow(limit: rollingWindow)
        self._playedMoveProbNegAdvWindow = RollingDoubleWindow(limit: rollingWindow)
        self._playedMoveProbPosAdvSkipWindow = RollingDoubleWindow(limit: rollingWindow)
        self._playedMoveProbNegAdvSkipWindow = RollingDoubleWindow(limit: rollingWindow)
        self._advMeanWindow = RollingDoubleWindow(limit: rollingWindow)
        self._advStdWindow = RollingDoubleWindow(limit: rollingWindow)
        self._advMinWindow = RollingDoubleWindow(limit: rollingWindow)
        self._advMaxWindow = RollingDoubleWindow(limit: rollingWindow)
        self._advFracPosWindow = RollingDoubleWindow(limit: rollingWindow)
        self._advFracSmallWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyLossWinWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyLossLossWindow = RollingDoubleWindow(limit: rollingWindow)
    }

    /// Seed the stats with values from a resumed session so the
    /// step counter and other totals don't restart from zero.
    func seed(_ stats: TrainingRunStats) {
        queue.async {
            self._stats = stats
        }
    }

    /// Record one completed training step. Called from the background
    /// training task. Dispatches the rolling-window bookkeeping to the
    /// serial queue asynchronously so the training worker never waits
    /// on the UI heartbeat's `snapshot()` read.
    func recordStep(_ timing: TrainStepTiming) {
        queue.async {
            self._stats.record(timing)
            self._lastTiming = timing
            self._policyLossWindow.append(Double(timing.policyLoss))
            self._valueLossWindow.append(Double(timing.valueLoss))
            self._policyEntropyWindow.append(Double(timing.policyEntropy))
            self._policyNonNegWindow.append(Double(timing.policyNonNegligibleCount))
            self._policyNonNegIllegalWindow.append(Double(timing.policyNonNegligibleIllegalCount))
            self._gradNormWindow.append(Double(timing.gradGlobalNorm))
            self._valueMeanWindow.append(Double(timing.valueMean))
            self._valueAbsMeanWindow.append(Double(timing.valueAbsMean))
            // Optional — only the real-data path supplies vBaselineDelta.
            if let delta = timing.vBaselineDelta {
                self._vBaselineDeltaWindow.append(Double(delta))
            }
            self._policyHeadWeightNormWindow.append(Double(timing.policyHeadWeightNorm))
            self._policyLogitAbsMaxWindow.append(Double(timing.policyLogitAbsMax))
            self._playedMoveProbWindow.append(Double(timing.playedMoveProb))
            // NaN protection: the conditional means are NaN when the
            // batch has zero positions on one side of the sign — skip
            // those entries so the rolling mean stays well-defined.
            // The parallel `…SkipWindow` ring is appended on every step
            // (1.0 on skip, 0.0 on contribution) so consumers can
            // present the conditional mean as "mean over K/N samples"
            // rather than silently losing the skipped-batch count.
            if timing.playedMoveProbPosAdv.isFinite {
                self._playedMoveProbPosAdvWindow.append(Double(timing.playedMoveProbPosAdv))
                self._playedMoveProbPosAdvSkipWindow.append(0.0)
            } else {
                self._playedMoveProbPosAdvSkipWindow.append(1.0)
            }
            if timing.playedMoveProbNegAdv.isFinite {
                self._playedMoveProbNegAdvWindow.append(Double(timing.playedMoveProbNegAdv))
                self._playedMoveProbNegAdvSkipWindow.append(0.0)
            } else {
                self._playedMoveProbNegAdvSkipWindow.append(1.0)
            }
            self._advMeanWindow.append(Double(timing.advantageMean))
            self._advStdWindow.append(Double(timing.advantageStd))
            self._advMinWindow.append(Double(timing.advantageMin))
            self._advMaxWindow.append(Double(timing.advantageMax))
            self._advFracPosWindow.append(Double(timing.advantageFracPositive))
            self._advFracSmallWindow.append(Double(timing.advantageFracSmall))
            // Outcome-partitioned policy losses — appended only when
            // finite. The graph emits NaN for batches with zero
            // win/loss positions; skipping NaN means the rolling mean
            // stays well-defined rather than getting poisoned.
            if let pwin = timing.policyLossWin, pwin.isFinite {
                self._policyLossWinWindow.append(Double(pwin))
            }
            if let plos = timing.policyLossLoss, plos.isFinite {
                self._policyLossLossWindow.append(Double(plos))
            }
            if let raw = timing.advantageRaw, !raw.isEmpty {
                self.pushAdvRaw(raw)
            }
        }
    }

    /// Push a batch's raw advantage values into the percentile ring.
    /// Called from within `recordStep`'s queue-async closure, so no
    /// additional serialization is needed. Capacity is grown on first
    /// use (and whenever `rollingWindow * batchSize` changes) to avoid
    /// reallocating in steady state.
    private func pushAdvRaw(_ batch: [Float]) {
        let desiredCapacity = min(
            self.rollingWindow * batch.count,
            Self.advRawRingMaxCapacity
        )
        if self._advRawRing.count != desiredCapacity {
            // First push (or batch-size change) — resize. Losing any
            // currently-held samples is fine since they were collected
            // under a different batch size and would otherwise skew
            // the distribution weighting.
            self._advRawRing = [Float](repeating: 0, count: desiredCapacity)
            self._advRawRingCapacity = desiredCapacity
            self._advRawRingHead = 0
            self._advRawRingFilled = 0
        }
        guard self._advRawRingCapacity > 0 else { return }
        for value in batch {
            self._advRawRing[self._advRawRingHead] = value
            self._advRawRingHead += 1
            if self._advRawRingHead >= self._advRawRingCapacity {
                self._advRawRingHead = 0
            }
            if self._advRawRingFilled < self._advRawRingCapacity {
                self._advRawRingFilled += 1
            }
        }
    }

    /// Record a terminal training error. Also called from the worker.
    /// The first error wins — subsequent calls are ignored so a
    /// follow-on error doesn't clobber the original cause.
    func recordError(_ message: String) {
        queue.async {
            if self._error == nil { self._error = message }
        }
    }

    /// Clear the rolling diagnostic windows while keeping cumulative
    /// training stats and the most recent error intact. Used after a
    /// promotion so post-promotion alarms and charts reflect the new
    /// aligned trainer/champion regime instead of inheriting the
    /// pre-promotion averages.
    func resetRollingWindows() {
        queue.async {
            self._lastTiming = nil
            self._policyLossWindow.removeAll()
            self._valueLossWindow.removeAll()
            self._policyEntropyWindow.removeAll()
            self._policyNonNegWindow.removeAll()
            self._policyNonNegIllegalWindow.removeAll()
            self._gradNormWindow.removeAll()
            self._valueMeanWindow.removeAll()
            self._valueAbsMeanWindow.removeAll()
            self._vBaselineDeltaWindow.removeAll()
            self._policyHeadWeightNormWindow.removeAll()
            self._policyLogitAbsMaxWindow.removeAll()
            self._playedMoveProbWindow.removeAll()
            self._playedMoveProbPosAdvWindow.removeAll()
            self._playedMoveProbNegAdvWindow.removeAll()
            self._playedMoveProbPosAdvSkipWindow.removeAll()
            self._playedMoveProbNegAdvSkipWindow.removeAll()
            self._advMeanWindow.removeAll()
            self._advStdWindow.removeAll()
            self._advMinWindow.removeAll()
            self._advMaxWindow.removeAll()
            self._advFracPosWindow.removeAll()
            self._advFracSmallWindow.removeAll()
            self._policyLossWinWindow.removeAll()
            self._policyLossLossWindow.removeAll()
            self._advRawRingHead = 0
            self._advRawRingFilled = 0
            // Keep _advRawRing capacity allocated — next push reuses it.
        }
    }

    /// Snapshot all fields atomically for the UI poller.
    func snapshot() -> Snapshot {
        queue.sync {
            let rollingPolicy = _policyLossWindow.mean
            let rollingValue = _valueLossWindow.mean
            let rollingEntropy = _policyEntropyWindow.mean
            let rollingNonNeg = _policyNonNegWindow.mean
            let rollingNonNegIllegal = _policyNonNegIllegalWindow.mean
            let rollingGradNorm = _gradNormWindow.mean
            let rollingVMean = _valueMeanWindow.mean
            let rollingVAbs = _valueAbsMeanWindow.mean
            let rollingVBaseDelta = _vBaselineDeltaWindow.mean
            let rollingPolicyHeadWNorm = _policyHeadWeightNormWindow.mean
            let rollingPLogitAbsMax = _policyLogitAbsMaxWindow.mean
            let rollingPlayedMoveP = _playedMoveProbWindow.mean
            let rollingPlayedMovePosAdv = _playedMoveProbPosAdvWindow.mean
            let rollingPlayedMoveNegAdv = _playedMoveProbNegAdvWindow.mean
            // Skip-window size is the same on the pos and neg rings
            // since both are appended on every step, so either one
            // can supply the shared denominator.
            let rollingCondWindowSize = _playedMoveProbPosAdvSkipWindow.size
            let rollingPlayedMovePosAdvSkipped = Int(_playedMoveProbPosAdvSkipWindow.total.rounded())
            let rollingPlayedMoveNegAdvSkipped = Int(_playedMoveProbNegAdvSkipWindow.total.rounded())
            let rollingAdvMean = _advMeanWindow.mean
            let rollingAdvStd = _advStdWindow.mean
            let rollingAdvMin = _advMinWindow.mean
            let rollingAdvMax = _advMaxWindow.mean
            let rollingAdvFracPos = _advFracPosWindow.mean
            let rollingAdvFracSmall = _advFracSmallWindow.mean
            let rollingPLossWin = _policyLossWinWindow.mean
            let rollingPLossLoss = _policyLossLossWindow.mean
            let (advP05, advP50, advP95) = Self.percentiles(
                ring: _advRawRing,
                filled: _advRawRingFilled
            )
            return Snapshot(
                stats: _stats,
                lastTiming: _lastTiming,
                rollingPolicyLoss: rollingPolicy,
                rollingValueLoss: rollingValue,
                rollingPolicyEntropy: rollingEntropy,
                rollingPolicyNonNegCount: rollingNonNeg,
                rollingPolicyNonNegIllegalCount: rollingNonNegIllegal,
                rollingGradGlobalNorm: rollingGradNorm,
                rollingValueMean: rollingVMean,
                rollingValueAbsMean: rollingVAbs,
                rollingVBaselineDelta: rollingVBaseDelta,
                rollingPolicyHeadWeightNorm: rollingPolicyHeadWNorm,
                rollingPolicyLogitAbsMax: rollingPLogitAbsMax,
                rollingPlayedMoveProb: rollingPlayedMoveP,
                rollingPlayedMoveProbPosAdv: rollingPlayedMovePosAdv,
                rollingPlayedMoveProbNegAdv: rollingPlayedMoveNegAdv,
                rollingPlayedMoveProbPosAdvSkipped: rollingPlayedMovePosAdvSkipped,
                rollingPlayedMoveProbNegAdvSkipped: rollingPlayedMoveNegAdvSkipped,
                rollingPlayedMoveCondWindowSize: rollingCondWindowSize,
                rollingAdvMean: rollingAdvMean,
                rollingAdvStd: rollingAdvStd,
                rollingAdvMin: rollingAdvMin,
                rollingAdvMax: rollingAdvMax,
                rollingAdvFracPositive: rollingAdvFracPos,
                rollingAdvFracSmall: rollingAdvFracSmall,
                advantageP05: advP05,
                advantageP50: advP50,
                advantageP95: advP95,
                rollingPolicyLossWin: rollingPLossWin,
                rollingPolicyLossLoss: rollingPLossLoss,
                error: _error
            )
        }
    }

    /// Compute (p05, p50, p95) from the live portion of a raw-value
    /// ring. Sorts a copy of the first `filled` elements (so the
    /// caller's ring storage isn't reordered) and indexes by fraction.
    /// Returns (nil, nil, nil) when the ring is empty.
    private static func percentiles(
        ring: [Float],
        filled: Int
    ) -> (Double?, Double?, Double?) {
        guard filled > 0, filled <= ring.count else { return (nil, nil, nil) }
        var sorted = Array(ring.prefix(filled))
        sorted.sort()
        let n = sorted.count
        func pct(_ p: Double) -> Double {
            let idx = Int((p * Double(n - 1)).rounded())
            return Double(sorted[max(0, min(n - 1, idx))])
        }
        return (pct(0.05), pct(0.50), pct(0.95))
    }
}

// MARK: - Chess Trainer

/// Builds a separate training-mode copy of the chess network and runs
/// benchmark training steps against it. The trainer owns its own
/// ChessNetwork instance (with `bnMode = .training`), distinct from the
/// inference network used by Play Game / Forward Pass — that way the
/// inference network keeps its frozen-stats BN for fast play, while the
/// trainer measures realistic training-step costs through batch-stats BN
/// and the full backward graph.
///
/// Repeated trainStep() calls actually update the trainer's internal
/// weights via SGD (this is how we verified the training pipeline is
/// mechanically correct: random data, random labels, but loss still drops).
///
/// Marked @unchecked Sendable for the same reason as ChessNetwork — Metal
/// objects aren't Sendable but access is serialized externally (UI gates
/// training and inference to never overlap).
final class ChessTrainer: @unchecked Sendable {

    // MARK: Configuration

    /// L2 weight-decay coefficient applied per training step. The
    /// optimizer here is plain SGD (no momentum, no Adam state), so
    /// the update rule for decay-eligible variables is
    /// `v_new = v - lr * (clipped_grad + weightDecayC * v)`,
    /// equivalent to `(1 - lr*c) * v - lr * clipped_grad`. With plain
    /// SGD, "decoupled" weight decay and ordinary L2 regularization
    /// are mathematically identical — the AdamW-vs-Adam-with-L2
    /// distinction only matters for adaptive optimizers, so this is
    /// just L2. Decay is applied only to conv and FC weight matrices;
    /// BN gamma/beta and FC biases are excluded, matching the standard
    /// PyTorch / AdamW recipe for which params to decay. (Decaying BN
    /// gamma toward zero zeros out a channel and reduces effective
    /// capacity — the prior "L2 on all params" decision was reverted
    /// after the deep ML review.)
    ///
    /// The actual value applied by the graph is read from
    /// `weightDecayC` and fed as a per-step scalar so the user can
    /// tune it live.
    static let weightDecayCDefault: Float = 1e-4

    /// Default global L2-norm gradient clipping threshold. If the L2
    /// norm of the concatenated gradient vector over every trainable
    /// variable exceeds this value, every gradient is scaled by
    /// `maxNorm / globalNorm` so the effective step is capped. 5.0 is
    /// a conservative value that sits well above steady-state norms
    /// under healthy training but cuts off the single-step blowups
    /// (see 2026-04-15 incident). Under heavy policy-collapse
    /// pressure the natural gradient norm can vastly exceed this,
    /// nullifying effective learning rate — live-tunable to let the
    /// user widen the valve when that happens.
    static let gradClipMaxNormDefault: Float = 30.0

    /// Default policy-loss coefficient K. Policy loss is REINFORCE
    /// on the played move over a `policySize`-way softmax, so its
    /// gradient is naturally much weaker than the value head's
    /// (z−v)² gradient. The K coefficient rescales the policy loss
    /// term in the total loss so both heads get meaningful gradient
    /// during early bootstrap. Lowered from 50 → 5 after review —
    /// the prior 50× multiplier was an amplifier on the gradient
    /// magnitude that the clip was eating nearly all of.
    static let policyScaleKDefault: Float = 5.0

    var learningRate: Float
    /// Base batch size at which `learningRate` and `weightDecayC`
    /// are taken as-is when sqrt-batch scaling is enabled. A step
    /// with `batchSize == sqrtScaleBaseBatchSize` multiplies each
    /// scaled value by exactly 1.0, so 4096 is the no-op pivot and
    /// the UI-displayed values are the "base" values at that pivot.
    /// Smaller batches scale down by `sqrt(batchSize/4096)` and
    /// larger batches scale up by the same rule — the standard
    /// Adam-family LR rule that preserves effective per-sample
    /// update magnitude across batch-size changes.
    static let sqrtScaleBaseBatchSize: Int = 4096
    /// When true, `learningRate` as fed to the optimizer each step
    /// is `learningRate * sqrt(batchSize / sqrtScaleBaseBatchSize)`.
    /// When false, LR is fed verbatim. The property itself always
    /// stores the user-facing base value regardless of this flag —
    /// scaling is applied at write time, never persisted back. Live-
    /// editable; a flip takes effect on the next training step.
    ///
    /// Weight decay is intentionally NOT sqrt-scaled: the standard
    /// AdamW convention is to scale LR with batch size and keep
    /// weight decay fixed at the user-configured value. Scaling
    /// both would compound to a linear-in-batch effect on the
    /// combined `lr × wd` decay term per step, which is not the
    /// Adam-family rule the user asked for.
    var sqrtBatchScalingForLR: Bool
    /// Linear warmup length for the learning rate. The LR fed to
    /// the optimizer each step is multiplied by
    /// `min(1, completedTrainSteps / lrWarmupSteps)` — so step 0
    /// uses zero LR (pure warmup) and step `lrWarmupSteps` (and
    /// later) uses the full configured LR. Composes multiplicatively
    /// with `sqrtBatchScalingForLR`. Zero disables warmup entirely
    /// (multiplier is a constant 1.0). Live-editable; a change takes
    /// effect on the next training step and is evaluated against
    /// the current step count, so lowering it mid-session can
    /// instantly end warmup, while raising it re-engages warmup for
    /// the remaining `lrWarmupSteps - completedTrainSteps` steps.
    var lrWarmupSteps: Int
    var entropyRegularizationCoeff: Float
    /// Live weight-decay coefficient. Fed into the training graph
    /// every step via a scalar placeholder, so edits take effect on
    /// the next step without graph rebuild.
    var weightDecayC: Float
    /// Live gradient-clip max norm. Fed via scalar placeholder each
    /// step.
    var gradClipMaxNorm: Float
    /// Live policy-loss coefficient K. Multiplied into `policyLoss`
    /// before it joins `valueLoss` and `−entropyCoeff·policyEntropy`
    /// in `total_loss`. Fed via scalar placeholder each step (live-
    /// tunable). Behaviorally a per-head weighting on shared-trunk
    /// gradients — bigger K = trunk pulled toward minimizing policy
    /// CE, smaller K = balanced trunk that lets the value head
    /// converge in parallel. NOT a multiplier on the policy LOGITS;
    /// see `weightedPolicy` in `buildTrainingOps`. Without MCTS-
    /// quality policy targets, values above ~3 tend to amplify
    /// noise on the policy head before the value baseline becomes
    /// useful. AlphaZero canonical is K=1 (equal weighting).
    var policyScaleK: Float

    /// Bootstrap-phase knob that rewrites drawn-game `z` values from 0
    /// to `-drawPenalty` before they reach the graph. Applied CPU-side
    /// in `trainStep` after the replay-buffer sample returns and
    /// before `buildFeeds`. Zero = no change (default). The transform
    /// treats all draw types the same (stalemate, 50-move, threefold,
    /// insufficient material); anything with z=0.0 exactly becomes
    /// `-drawPenalty`.
    ///
    /// What it actually does, math-wise: the value head's MSE target
    /// for drawn positions becomes `-drawPenalty` instead of 0, so v
    /// learns to predict slightly negative values for draw-prone
    /// positions. The advantage `z − v` then sees a lower baseline
    /// for non-draw positions, so winning-game gradients get a small
    /// extra positive lift relative to the prior baseline — and
    /// drawn-game gradients are roughly unchanged in expectation.
    ///
    /// Caveat — the docstring USED to claim this "turns REINFORCE-
    /// silent drawn games into a mild negative signal." That framing
    /// is misleading: drawn games are NOT REINFORCE-silent under
    /// `drawPenalty=0`. They produce signal *relative to the value
    /// baseline* — a draw with vBaseline > 0 (model thought you were
    /// winning) gives negative advantage; a draw with vBaseline < 0
    /// gives positive advantage. drawPenalty just shifts the
    /// "neutral-draw" threshold by `drawPenalty` units. The
    /// "self-limiting" property still holds: as v converges toward
    /// `-drawPenalty` for draw-prone positions, the threshold-shift
    /// effect washes out. So this is most defensible as an early-
    /// bootstrap nudge that anneals naturally; for steady-state
    /// runs `drawPenalty=0` is a reasonable default.
    var drawPenalty: Float
    /// Count of successfully-completed SGD steps this trainer has
    /// run since construction (or since a session-resume `seed`).
    /// Read by `buildFeeds` to compute the warmup multiplier before
    /// each step is fed and incremented by `runPreparedStep` after a
    /// graph run returns without throwing. Exposed to callers via
    /// `completedTrainSteps` so a session resume can pre-seed it to
    /// the persisted `trainingSteps` value — warmup then picks up
    /// mid-session instead of restarting from zero.
    ///
    /// Stored in a `SyncBox` (os_unfair_lock) rather than as a plain
    /// `Int` guarded by `executionQueue` so UI readers
    /// (`__processSnapshotTimerTick` at 10 Hz) don't have to `.sync`
    /// onto the trainer's worker queue and wait for an in-flight
    /// SGD step — that pattern was producing 1–3 s main-thread
    /// stalls. The lock is held only across a scalar read/RMW so
    /// contention between the trainer thread and UI thread is
    /// effectively free. The lock-protected `+= 1` in
    /// `runPreparedStep` keeps the read-modify-write atomic on its
    /// own, no longer dependent on the queue invariant.
    private let _completedTrainSteps = SyncBox<Int>(0)
    private let executionQueue = DispatchQueue(label: "drewschess.chesstrainer.serial")

    /// Optional stable identity for the trainer's internal network.
    /// Assigned by the UI layer at Play-and-Train start (after loading
    /// champion weights) and then kept stable for the lifetime of the
    /// Play-and-Train session — it represents the "current training
    /// lineage" rather than a specific byte-exact weight snapshot.
    /// See `sampling-parameters.md` for the full rule set.
    var identifier: ModelID?

    // MARK: Graph Tensors

    private(set) var network: ChessNetwork
    private var movePlayedPlaceholder: MPSGraphTensor   // [batch] int32
    private var zPlaceholder: MPSGraphTensor            // [batch, 1] float
    private var vBaselinePlaceholder: MPSGraphTensor    // [batch, 1] float
    private var legalMaskPlaceholder: MPSGraphTensor
    private var lrPlaceholder: MPSGraphTensor           // [] scalar float
    private var entropyCoeffPlaceholder: MPSGraphTensor // [] scalar float
    private var weightDecayPlaceholder: MPSGraphTensor  // [] scalar float
    private var gradClipMaxNormPlaceholder: MPSGraphTensor // [] scalar float
    private var policyScaleKPlaceholder: MPSGraphTensor // [] scalar float
    private var totalLoss: MPSGraphTensor               // scalar
    private var policyLossTensor: MPSGraphTensor        // scalar
    private var valueLossTensor: MPSGraphTensor         // scalar
    private var policyEntropyTensor: MPSGraphTensor     // scalar (diagnostic)
    private var policyNonNegCountTensor: MPSGraphTensor // scalar (diagnostic, legal cells)
    private var policyNonNegIllegalCountTensor: MPSGraphTensor // scalar (diagnostic, illegal cells)
    private var gradGlobalNormTensor: MPSGraphTensor    // scalar (diagnostic)
    private var valueMeanTensor: MPSGraphTensor         // scalar (diagnostic)
    private var valueAbsMeanTensor: MPSGraphTensor      // scalar (diagnostic)
    private var policyHeadWeightNormTensor: MPSGraphTensor // scalar (diagnostic)
    private var policyLogitAbsMaxTensor: MPSGraphTensor // scalar (diagnostic)
    private var playedMoveProbTensor: MPSGraphTensor    // scalar (diagnostic)
    private var playedMoveProbPosAdvTensor: MPSGraphTensor // scalar (diagnostic)
    private var playedMoveProbNegAdvTensor: MPSGraphTensor // scalar (diagnostic)
    private var advantageMeanTensor: MPSGraphTensor     // scalar (diagnostic)
    private var advantageStdTensor: MPSGraphTensor      // scalar (diagnostic)
    private var advantageMinTensor: MPSGraphTensor      // scalar (diagnostic)
    private var advantageMaxTensor: MPSGraphTensor      // scalar (diagnostic)
    private var advantageFracPosTensor: MPSGraphTensor  // scalar (diagnostic)
    private var advantageFracSmallTensor: MPSGraphTensor // scalar (diagnostic)
    /// [batch, 1] raw advantage tensor — read back per step so the
    /// stats box can maintain a rolling percentile window.
    private var advantageRawTensor: MPSGraphTensor
    /// Scalar mean policy loss restricted to batch positions where
    /// outcome z > 0.5. NaN when no win positions are in the batch.
    private var policyLossWinTensor: MPSGraphTensor
    /// Scalar mean policy loss restricted to batch positions where
    /// outcome z < -0.5. NaN when no loss positions are in the batch.
    private var policyLossLossTensor: MPSGraphTensor
    private var assignOps: [MPSGraphOperation]

    /// Pre-allocated scalar ND array for the learning-rate feed.
    /// Written with the current `learningRate` on each step so
    /// the value can change between steps without rebuilding the
    /// graph. Recreated in `resetNetwork()` alongside the feed
    /// cache so the new graph's placeholder maps to a fresh
    /// tensor-data wrapper.
    private var lrNDArray: MPSNDArray
    private var lrTensorData: MPSGraphTensorData
    private var entropyCoeffNDArray: MPSNDArray
    private var entropyCoeffTensorData: MPSGraphTensorData
    private var weightDecayNDArray: MPSNDArray
    private var weightDecayTensorData: MPSGraphTensorData
    private var gradClipMaxNormNDArray: MPSNDArray
    private var gradClipMaxNormTensorData: MPSGraphTensorData
    private var policyScaleKNDArray: MPSNDArray
    private var policyScaleKTensorData: MPSGraphTensorData

    /// Pre-allocated ND-array-backed tensor data for the three training
    /// placeholders at a given batch size, plus the pre-built
    /// `[MPSGraphTensor: MPSGraphTensorData]` feed dict the trainer
    /// hands to `graph.run`. `buildFeeds(...)` looks one of these up
    /// (or lazily creates it on the first call for each batch size)
    /// and writes new Swift-array values into the ND arrays in place,
    /// so steady-state training and the timed portion of the batch-size
    /// sweep allocate no MPS objects and no Swift dictionaries per
    /// step. The warmup step of a new batch size pays the allocation
    /// exactly once.
    private struct BatchFeeds {
        let boardND: MPSNDArray
        let boardTD: MPSGraphTensorData
        let moveND: MPSNDArray
        let moveTD: MPSGraphTensorData
        let zND: MPSNDArray
        let zTD: MPSGraphTensorData
        let vBaselineND: MPSNDArray
        let vBaselineTD: MPSGraphTensorData
        let legalMaskND: MPSNDArray
        let legalMaskTD: MPSGraphTensorData
        let feedsDict: [MPSGraphTensor: MPSGraphTensorData]
    }
    private var feedCache: [Int: BatchFeeds] = [:]

    /// Readback scratch for the per-step scalar outputs (`totalLoss`,
    /// `policyLoss`, `valueLoss`, and the diagnostic `policyEntropy`).
    /// `runPreparedStep` asks MPSGraph to write each scalar directly
    /// into its slot here so the hot path does not allocate a fresh
    /// `[Float](1)` per output per step. Allocated once in `init` and
    /// freed in `deinit`; `resetNetwork` does not touch it (the scalar
    /// type is network-independent).
    private let lossReadbackScratchPtr: UnsafeMutablePointer<Float>
    private static let lossReadbackSlotTotal: Int = 0
    private static let lossReadbackSlotPolicy: Int = 1
    private static let lossReadbackSlotValue: Int = 2
    private static let lossReadbackSlotEntropy: Int = 3
    private static let lossReadbackSlotNonNeg: Int = 4
    private static let lossReadbackSlotGradNorm: Int = 5
    private static let lossReadbackSlotValueMean: Int = 6
    private static let lossReadbackSlotValueAbsMean: Int = 7
    private static let lossReadbackSlotPolicyHeadWNorm: Int = 8
    private static let lossReadbackSlotPLogitAbsMax: Int = 9
    private static let lossReadbackSlotPlayedMoveProb: Int = 10
    private static let lossReadbackSlotAdvMean: Int = 11
    private static let lossReadbackSlotAdvStd: Int = 12
    private static let lossReadbackSlotAdvMin: Int = 13
    private static let lossReadbackSlotAdvMax: Int = 14
    private static let lossReadbackSlotAdvFracPos: Int = 15
    private static let lossReadbackSlotAdvFracSmall: Int = 16
    private static let lossReadbackSlotPlayedMoveProbPosAdv: Int = 17
    private static let lossReadbackSlotPlayedMoveProbNegAdv: Int = 18
    private static let lossReadbackSlotPolicyLossWin: Int = 19
    private static let lossReadbackSlotPolicyLossLoss: Int = 20
    private static let lossReadbackSlotNonNegIllegal: Int = 21
    private static let lossReadbackSlotCount: Int = 22

    /// Reusable host-side staging buffers for replay-buffer samples.
    /// The trainer owns these buffers so real-data training can hop
    /// onto `executionQueue`, sample directly into stable storage, and
    /// feed MPSGraph without any additional ownership-transfer copy.
    private var replayBatchCapacity: Int = 0
    private var replayBatchBoards: UnsafeMutablePointer<Float>?
    private var replayBatchMoves: UnsafeMutablePointer<Int32>?
    private var replayBatchZs: UnsafeMutablePointer<Float>?
    private var replayBatchVBaselines: UnsafeMutablePointer<Float>?
    private var replayBatchLegalMasks: UnsafeMutablePointer<Float>?

    // Per-position observability metadata staging buffers — populated
    // only on stats-collection batches (every Nth, when
    // `batchStatsInterval > 0`). Allocated alongside the training
    // buffers so they're sized in lock-step with batchSize.
    private var replayBatchPlies: UnsafeMutablePointer<UInt16>?
    private var replayBatchGameLengths: UnsafeMutablePointer<UInt16>?
    private var replayBatchTaus: UnsafeMutablePointer<Float>?
    private var replayBatchHashes: UnsafeMutablePointer<UInt64>?
    private var replayBatchWorkerGameIds: UnsafeMutablePointer<UInt32>?
    private var replayBatchMaterialCounts: UnsafeMutablePointer<UInt8>?

    /// How often (in training steps) to compute and emit a
    /// `[BATCH-STATS]` log line. 0 disables. Live-tunable from the UI
    /// or from `TrainingParameters.batchStatsInterval`.
    var batchStatsInterval: Int = 10
    /// Last computed unique-position percent (0..1) for surfacing in
    /// the regular `[STATS]` line. Defaults to NaN until the first
    /// stats-collection batch lands.
    private(set) var lastBatchStatsUniquePct: Double = .nan
    /// Last full batch-stats summary so the CLI recorder can ship
    /// every result.json's stats tick with the most-recent
    /// observability snapshot. Nil until the first stats batch lands.
    /// Reads/writes are unsynchronized scalar pointer assignments
    /// (the struct is small, but Swift atomicity isn't guaranteed) —
    /// acceptable for diagnostic purposes; readers may briefly see
    /// the prior value during update.
    private(set) var lastBatchStatsSummary: ReplayBuffer.BatchStatsSummary?

    // MARK: Init

    init(
        learningRate: Float = 5e-5,
        entropyRegularizationCoeff: Float = 0.0,
        drawPenalty: Float = 0.1,
        weightDecayC: Float = ChessTrainer.weightDecayCDefault,
        gradClipMaxNorm: Float = ChessTrainer.gradClipMaxNormDefault,
        policyScaleK: Float = ChessTrainer.policyScaleKDefault,
        sqrtBatchScalingForLR: Bool = true,
        lrWarmupSteps: Int = 100
    ) throws {
        self.learningRate = learningRate
        self.entropyRegularizationCoeff = entropyRegularizationCoeff
        self.drawPenalty = drawPenalty
        self.weightDecayC = weightDecayC
        self.gradClipMaxNorm = gradClipMaxNorm
        self.policyScaleK = policyScaleK
        self.sqrtBatchScalingForLR = sqrtBatchScalingForLR
        self.lrWarmupSteps = lrWarmupSteps
        let net = try ChessNetwork(bnMode: .training)
        self.network = net
        let built = try Self.buildTrainingOps(network: net)
        self.movePlayedPlaceholder = built.movePlayed
        self.zPlaceholder = built.z
        self.vBaselinePlaceholder = built.vBaseline
        self.legalMaskPlaceholder = built.legalMask
        self.lrPlaceholder = built.lr
        self.entropyCoeffPlaceholder = built.entropyCoeff
        self.weightDecayPlaceholder = built.weightDecay
        self.gradClipMaxNormPlaceholder = built.gradClipMaxNorm
        self.policyScaleKPlaceholder = built.policyScaleK
        self.totalLoss = built.totalLoss
        self.policyLossTensor = built.policyLoss
        self.valueLossTensor = built.valueLoss
        self.policyEntropyTensor = built.policyEntropy
        self.policyNonNegCountTensor = built.policyNonNegCount
        self.policyNonNegIllegalCountTensor = built.policyNonNegIllegalCount
        self.gradGlobalNormTensor = built.gradGlobalNorm
        self.valueMeanTensor = built.valueMean
        self.valueAbsMeanTensor = built.valueAbsMean
        self.policyHeadWeightNormTensor = built.policyHeadWeightNorm
        self.policyLogitAbsMaxTensor = built.policyLogitAbsMax
        self.playedMoveProbTensor = built.playedMoveProb
        self.playedMoveProbPosAdvTensor = built.playedMoveProbPosAdv
        self.playedMoveProbNegAdvTensor = built.playedMoveProbNegAdv
        self.advantageMeanTensor = built.advantageMean
        self.advantageStdTensor = built.advantageStd
        self.advantageMinTensor = built.advantageMin
        self.advantageMaxTensor = built.advantageMax
        self.advantageFracPosTensor = built.advantageFracPos
        self.advantageFracSmallTensor = built.advantageFracSmall
        self.advantageRawTensor = built.advantageRaw
        self.policyLossWinTensor = built.policyLossWin
        self.policyLossLossTensor = built.policyLossLoss
        self.assignOps = built.assignOps

        // Scalar ND array for the learning rate feed, reused every step.
        let lrDesc = MPSNDArrayDescriptor(
            dataType: ChessNetwork.dataType,
            shape: [1]
        )
        let lrND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.lrNDArray = lrND
        self.lrTensorData = MPSGraphTensorData(lrND)
        let entropyND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.entropyCoeffNDArray = entropyND
        self.entropyCoeffTensorData = MPSGraphTensorData(entropyND)
        let weightDecayND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.weightDecayNDArray = weightDecayND
        self.weightDecayTensorData = MPSGraphTensorData(weightDecayND)
        let gradClipND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.gradClipMaxNormNDArray = gradClipND
        self.gradClipMaxNormTensorData = MPSGraphTensorData(gradClipND)
        let policyScaleKND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.policyScaleKNDArray = policyScaleKND
        self.policyScaleKTensorData = MPSGraphTensorData(policyScaleKND)

        let lossPtr = UnsafeMutablePointer<Float>.allocate(
            capacity: Self.lossReadbackSlotCount
        )
        lossPtr.initialize(repeating: 0, count: Self.lossReadbackSlotCount)
        self.lossReadbackScratchPtr = lossPtr
    }

    deinit {
        lossReadbackScratchPtr.deinitialize(count: Self.lossReadbackSlotCount)
        lossReadbackScratchPtr.deallocate()
        if let ptr = replayBatchBoards {
            ptr.deinitialize(count: replayBatchCapacity * ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize)
            ptr.deallocate()
        }
        if let ptr = replayBatchMoves {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchZs {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchVBaselines {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchLegalMasks {                              // <-- add
            ptr.deinitialize(count: replayBatchCapacity * ChessNetwork.policySize)
            ptr.deallocate()
        }
        if let ptr = replayBatchPlies {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchGameLengths {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchTaus {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchHashes {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchWorkerGameIds {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchMaterialCounts {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
    }

    /// Tear down the current training-mode network and build a fresh one.
    /// Used at the start of a sweep so each run starts from random weights
    /// rather than whatever the previous run left behind. Throws if the
    /// underlying ChessNetwork init fails (Metal/device problems) or if
    /// gradient lookup fails for any trainable variable.
    func resetNetwork() async throws {
        try await enqueue {
            try self.internalResetNetwork()
        }
    }

    private func internalResetNetwork() throws {
        let net = try ChessNetwork(bnMode: .training)
        self.network = net
        let built = try Self.buildTrainingOps(network: net)
        self.movePlayedPlaceholder = built.movePlayed
        self.zPlaceholder = built.z
        self.vBaselinePlaceholder = built.vBaseline
        self.legalMaskPlaceholder = built.legalMask
        self.lrPlaceholder = built.lr
        self.entropyCoeffPlaceholder = built.entropyCoeff
        self.weightDecayPlaceholder = built.weightDecay
        self.gradClipMaxNormPlaceholder = built.gradClipMaxNorm
        self.policyScaleKPlaceholder = built.policyScaleK
        self.totalLoss = built.totalLoss
        self.policyLossTensor = built.policyLoss
        self.valueLossTensor = built.valueLoss
        self.policyEntropyTensor = built.policyEntropy
        self.policyNonNegCountTensor = built.policyNonNegCount
        self.policyNonNegIllegalCountTensor = built.policyNonNegIllegalCount
        self.gradGlobalNormTensor = built.gradGlobalNorm
        self.valueMeanTensor = built.valueMean
        self.valueAbsMeanTensor = built.valueAbsMean
        self.policyHeadWeightNormTensor = built.policyHeadWeightNorm
        self.policyLogitAbsMaxTensor = built.policyLogitAbsMax
        self.playedMoveProbTensor = built.playedMoveProb
        self.playedMoveProbPosAdvTensor = built.playedMoveProbPosAdv
        self.playedMoveProbNegAdvTensor = built.playedMoveProbNegAdv
        self.advantageMeanTensor = built.advantageMean
        self.advantageStdTensor = built.advantageStd
        self.advantageMinTensor = built.advantageMin
        self.advantageMaxTensor = built.advantageMax
        self.advantageFracPosTensor = built.advantageFracPos
        self.advantageFracSmallTensor = built.advantageFracSmall
        self.advantageRawTensor = built.advantageRaw
        self.policyLossWinTensor = built.policyLossWin
        self.policyLossLossTensor = built.policyLossLoss
        self.assignOps = built.assignOps
        // Rebuild the LR scalar feed against the new network's device
        // so the new graph's placeholder maps to a fresh wrapper.
        let lrDesc = MPSNDArrayDescriptor(
            dataType: ChessNetwork.dataType,
            shape: [1]
        )
        self.lrNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.lrTensorData = MPSGraphTensorData(lrNDArray)
        self.entropyCoeffNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.entropyCoeffTensorData = MPSGraphTensorData(entropyCoeffNDArray)
        self.weightDecayNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.weightDecayTensorData = MPSGraphTensorData(weightDecayNDArray)
        self.gradClipMaxNormNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.gradClipMaxNormTensorData = MPSGraphTensorData(gradClipMaxNormNDArray)
        self.policyScaleKNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.policyScaleKTensorData = MPSGraphTensorData(policyScaleKNDArray)
        // The cached ND arrays were allocated against the old network's
        // device and are keyed by batch size against the old graph's
        // placeholders. Drop the cache so the first trainStep after
        // reset rebuilds against the fresh network.
        feedCache.removeAll()
    }

    /// Build the training subgraph (loss + gradients + SGD assigns) on top
    /// of the given network's forward graph. Returns the placeholders, loss
    /// tensor, and assign ops the caller needs to run a training step.
    /// Throws `ChessTrainerError.gradientMissing` if any trainable variable
    /// fails gradient lookup — that would mean the autodiff couldn't reach
    /// it from the loss, which is a network-construction bug we want to
    /// surface immediately rather than silently train without it.
    private static func buildTrainingOps(
        network: ChessNetwork
    ) throws -> (
        movePlayed: MPSGraphTensor,
        z: MPSGraphTensor,
        vBaseline: MPSGraphTensor,
        legalMask: MPSGraphTensor, // legal moves mask
        lr: MPSGraphTensor,
        entropyCoeff: MPSGraphTensor,
        weightDecay: MPSGraphTensor,
        gradClipMaxNorm: MPSGraphTensor,
        policyScaleK: MPSGraphTensor,
        totalLoss: MPSGraphTensor,
        policyLoss: MPSGraphTensor,
        valueLoss: MPSGraphTensor,
        policyEntropy: MPSGraphTensor,
        policyNonNegCount: MPSGraphTensor,
        policyNonNegIllegalCount: MPSGraphTensor,
        gradGlobalNorm: MPSGraphTensor,
        valueMean: MPSGraphTensor,
        valueAbsMean: MPSGraphTensor,
        policyHeadWeightNorm: MPSGraphTensor,
        policyLogitAbsMax: MPSGraphTensor,
        playedMoveProb: MPSGraphTensor,
        playedMoveProbPosAdv: MPSGraphTensor,
        playedMoveProbNegAdv: MPSGraphTensor,
        advantageMean: MPSGraphTensor,
        advantageStd: MPSGraphTensor,
        advantageMin: MPSGraphTensor,
        advantageMax: MPSGraphTensor,
        advantageFracPos: MPSGraphTensor,
        advantageFracSmall: MPSGraphTensor,
        advantageRaw: MPSGraphTensor,
        policyLossWin: MPSGraphTensor,
        policyLossLoss: MPSGraphTensor,
        assignOps: [MPSGraphOperation]
    ) {
        let graph = network.graph
        let dtype = ChessNetwork.dataType

        // --- Placeholders for training targets ---

        let movePlayed = graph.placeholder(
            shape: [-1],
            dataType: .int32,
            name: "move_played"
        )
        let z = graph.placeholder(
            shape: [-1, 1],
            dataType: dtype,
            name: "z_outcome"
        )
        // vBaseline: the value-head's own prediction of this position
        // captured at play time, fed as a placeholder so autodiff can't
        // walk back into the value head from the policy loss. MPSGraph
        // has no stopGradient op, so feeding the baseline in externally
        // is how we get detach semantics.
        let vBaseline = graph.placeholder(
            shape: [-1, 1],
            dataType: dtype,
            name: "v_baseline"
        )

        let legalMask = graph.placeholder(
            shape: [-1, NSNumber(value: ChessNetwork.policySize)],
            dataType: dtype,
            name: "legal_move_mask"
        )

        // Build masked logits: illegal positions get a huge negative bias.
        let oneConst = graph.constant(1.0, dataType: dtype)
        let illegalMask = graph.subtraction(oneConst, legalMask, name: "illegal_mask")
        let largeNeg = graph.constant(-1e9, dataType: dtype)
        let additiveMask = graph.multiplication(illegalMask, largeNeg, name: "additive_mask")
        let maskedLogits = graph.addition(network.policyOutput, additiveMask, name: "masked_logits")

        // --- Policy loss: L = mean( z * -log_softmax(logits)[a*] ) ---
        //
        // Standard outcome-weighted cross entropy. We one-hot the played
        // move and feed the (logits, one-hot labels) pair to MPSGraph's
        // fused softMaxCrossEntropy, which ships its own autodiff
        // implementation. That matters because MPSGraph's autodiff has no
        // gradient for reductionMaximum — a manual stable log-softmax
        // built with max-subtraction would compile but crash inside
        // gradientForPrimaryTensor. The fused op sidesteps the issue and
        // is numerically stable by construction.
        //
        // Multiplying by z applies the outcome weighting from
        // chess-engine-design.md:
        //   z=+1 → push p(a*) up, z=-1 → push it down, z=0 → no contribution.

        let oneHot = graph.oneHot(
            withIndicesTensor: movePlayed,
            depth: ChessNetwork.policySize,
            axis: 1,
            dataType: dtype,
            onValue: 1.0,
            offValue: 0.0,
            name: "move_onehot"
        )

        let ceLossRaw = graph.softMaxCrossEntropy(
            maskedLogits,        // <-- changed from network.policyOutput
            labels: oneHot,
            axis: 1,
            reuctionType: .none,
            name: "policy_ce_raw"
        )

        // softMaxCrossEntropy with .none reduces the class axis, leaving
        // one loss per batch element. Reshape to [batch, 1] so it lines up
        // with z for the outcome-weighted multiply.
        let negLogProb = graph.reshape(
            ceLossRaw,
            shape: [-1, 1],
            name: "policy_ce_per_pos"
        )
        // --- Advantage baseline: (z − vBaseline) · −log p(a*) ---
        //
        // `vBaseline` is a placeholder — the inference-time v(position)
        // captured during self-play and stored alongside each position
        // in the ReplayBuffer. Feeding it back through a placeholder
        // is the MPSGraph-compatible way to "detach" (MPSGraph has no
        // stopGradient op, verified empirically in the 22:35 CDT
        // gradient-stop experiment: `variableFromTensor` + `read` does
        // not block backward flow). The advantage formulation reduces
        // policy-gradient variance by 5–20× per standard
        // REINFORCE-with-baseline literature, with zero bias — the
        // baseline only has to be a function of state, not the
        // current network's prediction.
        let advantage = graph.subtraction(z, vBaseline, name: "advantage")
        // Per-batch advantage standardization: `(A − E[A]) / (σ[A] + ε)`
        // before multiplying into the policy loss. Stabilizes the
        // policy-gradient magnitude batch-to-batch and removes the
        // bias-from-uncentered-baseline failure mode: when the value
        // head has a global offset (e.g. `E[v] ≈ 0.45` because 80 % of
        // self-play games are draws and the head has collapsed to the
        // draw-penalty average), raw advantages are systematically
        // skewed positive for wins and skewed negative for losses, so
        // the shared trunk receives a biased gradient in one direction.
        // Centering cancels the bias; std-dividing keeps step sizes
        // comparable even when the batch happens to span extreme z.
        //
        // MPSGraph has no stopGradient — but advantage is a pure
        // function of placeholders (`z`, `vBaseline`), so no gradient
        // path flows through `mean` / `std` back into trainable
        // variables. That's what makes this the "safe" standardization:
        // it adjusts the forward value used as the REINFORCE weight,
        // never touches the autograd path.
        //
        // ε of 1e-6 is conservative — the batch std is comfortably
        // above that under any outcome mix we actually train on (z in
        // {−1, 0, +1}, `drawPenalty ≈ 0.3`, batch=4096 → std ≈ 0.5+).
        let advantageMeanForNorm = graph.mean(
            of: advantage,
            axes: [0, 1],
            name: "advantage_mean_for_norm"
        )
        let advantageCentered = graph.subtraction(
            advantage,
            advantageMeanForNorm,
            name: "advantage_centered"
        )
        let advantageCenteredSq = graph.square(
            with: advantageCentered,
            name: "advantage_centered_sq"
        )
        let advantageVarForNorm = graph.mean(
            of: advantageCenteredSq,
            axes: [0, 1],
            name: "advantage_var_for_norm"
        )
        let advantageNormEps = graph.constant(1e-6, dataType: dtype)
        let advantageVarPlusEps = graph.addition(
            advantageVarForNorm,
            advantageNormEps,
            name: "advantage_var_plus_eps"
        )
        let advantageStdForNorm = graph.squareRoot(
            with: advantageVarPlusEps,
            name: "advantage_std_for_norm"
        )
        let advantageNormalized = graph.division(
            advantageCentered,
            advantageStdForNorm,
            name: "advantage_normalized"
        )
        let weightedCE = graph.multiplication(
            advantageNormalized,
            negLogProb,
            name: "adv_weighted_ce"
        )
        let policyLoss = graph.mean(
            of: weightedCE,
            axes: [0, 1],
            name: "policy_loss"
        )

        // --- Outcome-partitioned policy loss (diagnostic only) ---
        //
        // Split the batch policy loss by the sign of `z` (the
        // play-time outcome) so the curve can be read unambiguously:
        //   `policyLossWin` = mean over z > +0.5
        //   `policyLossLoss` = mean over z < -0.5
        // The mean computation is `sum(weightedCE * mask) / sum(mask)`.
        // We add a tiny epsilon to the denominator so a batch with
        // zero wins (or zero losses) returns 0 instead of NaN; that
        // case is rare but does happen near the start of a session.
        // These tensors are diagnostic-only — they're fetched via
        // `targetTensors`, never feed back into `totalLoss`, so
        // autodiff doesn't walk into them.
        let zPosThreshold = graph.constant(0.5, dataType: dtype)
        let zNegThreshold = graph.constant(-0.5, dataType: dtype)
        let maskWin = graph.cast(
            graph.greaterThan(z, zPosThreshold, name: "z_gt_pos_thresh"),
            to: dtype,
            name: "mask_win"
        )
        let maskLoss = graph.cast(
            graph.lessThan(z, zNegThreshold, name: "z_lt_neg_thresh"),
            to: dtype,
            name: "mask_loss"
        )
        let weightedCEWin = graph.multiplication(weightedCE, maskWin, name: "weighted_ce_win")
        let weightedCELoss = graph.multiplication(weightedCE, maskLoss, name: "weighted_ce_loss")
        let winSum = graph.reductionSum(with: weightedCEWin, axes: [0, 1], name: "weighted_ce_win_sum")
        let lossSum = graph.reductionSum(with: weightedCELoss, axes: [0, 1], name: "weighted_ce_loss_sum")
        let winMaskSum = graph.reductionSum(with: maskWin, axes: [0, 1], name: "mask_win_sum")
        let lossMaskSum = graph.reductionSum(with: maskLoss, axes: [0, 1], name: "mask_loss_sum")
        let denomEps = graph.constant(1e-6, dataType: dtype)
        let policyLossWin = graph.division(
            winSum,
            graph.addition(winMaskSum, denomEps, name: "mask_win_sum_eps"),
            name: "policy_loss_win"
        )
        let policyLossLoss = graph.division(
            lossSum,
            graph.addition(lossMaskSum, denomEps, name: "mask_loss_sum_eps"),
            name: "policy_loss_loss"
        )

        // --- Value loss: L = mean( (z - v)^2 ) ---

        let diff = graph.subtraction(z, network.valueOutput, name: "value_diff")
        let sq = graph.square(with: diff, name: "value_sq")
        let valueLoss = graph.mean(of: sq, axes: [0, 1], name: "value_loss")

        // --- Value-head output mean + abs-mean (diagnostic) ---
        //
        // Tanh saturation probe. `valueMean` near 0 is healthy (most
        // early-training positions are drawn, so batch mean should
        // sit near z's mean, which is ~0). `valueAbsMean` close to 1
        // means the tanh is saturated — every position reads as near
        // ±1 regardless of content, which kills the (1−v²) factor in
        // the MSE gradient and stalls value-head learning. Both are
        // fetched via `targetTensors`, never feed into totalLoss, so
        // autodiff never walks into them.
        let valueMean = graph.mean(
            of: network.valueOutput,
            axes: [0, 1],
            name: "value_mean"
        )
        let valueAbs = graph.absolute(with: network.valueOutput, name: "value_abs")
        let valueAbsMean = graph.mean(
            of: valueAbs,
            axes: [0, 1],
            name: "value_abs_mean"
        )

        // --- Policy entropy ---
        //
        // H(p) = −Σ p · log p, per position, then mean across batch.
        // Range is [0, log(policySize)] ≈ [0, 8.49] nats for the current
        // 4864-logit head; random init sits near the ceiling, a collapsed
        // policy heads toward 0.
        //
        // This tensor serves two roles: a diagnostic read via run-time
        // fetch AND a predecessor of totalLoss via the entropy
        // regularization term below. Because it flows into totalLoss,
        // every op on this path must have an MPSGraph autograd rule —
        // that rules out the max-subtracted logsumexp construction
        // (reductionMaximum has no gradient implementation). Built
        // here from graph.softMax (has gradient) plus log(p+ε) so
        // autodiff can walk the whole path cleanly. softMax is
        // numerically stable internally, and the ε clamp keeps
        // log/log-gradient finite on moves where p underflows to 0.
        //
        // ε = 1e-7 rather than 1e-10: in exact math the chain-rule
        // p factor on the outer multiply cancels the 1/x blowup of
        // log's local gradient (upstream grad for log(p+ε) is p, so
        // the composed contribution is p/(p+ε) ∈ [0,1]). That bound
        // holds regardless of ε, so a looser floor is free insurance
        // against FP32 edge cases without meaningfully biasing the
        // entropy estimate — at uniform init each p ≈ 1/4864 ≈ 2e-4,
        // well above 1e-7, so the clamp only bites once the policy
        // is already near-collapsed (at which point the pEnt alarm
        // has fired anyway). MPSGraph as of macOS 26.4 SDK still
        // exposes no stopGradient/detach, so feeding labels through
        // a placeholder or rebuilding as log-softmax-from-logits
        // (max-subtract needs reductionMaximum → no gradient) remain
        // closed off; the ε-bumped form is the available mitigation.
        let softmax = graph.softMax(
            with: maskedLogits,
            axis: 1,
            name: "policy_softmax"
        )
        let logEpsConst = graph.constant(1e-7, dataType: dtype)
        let softmaxClamped = graph.addition(
            softmax,
            logEpsConst,
            name: "policy_softmax_clamped"
        )
        let logSoftmax = graph.logarithm(
            with: softmaxClamped,
            name: "policy_log_softmax"
        )
        let pLogP = graph.multiplication(softmax, logSoftmax, name: "p_log_p")
        let negEntropyPerPos = graph.reductionSum(
            with: pLogP,
            axis: 1,
            name: "neg_entropy_per_pos"
        )
        let entropyPerPos = graph.negative(
            with: negEntropyPerPos,
            name: "entropy_per_pos"
        )
        let policyEntropy = graph.mean(
            of: entropyPerPos,
            axes: [0, 1],
            name: "policy_entropy"
        )

        // --- Policy non-negligible count (diagnostic) ---
        //
        // Count of softmax entries above 1/policySize (the uniform
        // probability), averaged across the batch. Starts near
        // ~policySize/2 with random init and drops as the policy
        // concentrates on promising moves. Like entropy, this is
        // diagnostic-only and not in totalLoss.
        let nonNegThreshold = graph.constant(
            1.0 / Double(ChessNetwork.policySize),
            dataType: dtype
        )
        // Legal-cell count: cells whose MASKED softmax is above
        // 1/policySize. The masked softmax is renormalized over legal
        // cells (illegals get ~0 after the -1e9 bias), so anything
        // above 1/policySize here is necessarily a legal cell with
        // meaningful mass.
        let aboveThreshold = graph.greaterThan(
            softmax,
            nonNegThreshold,
            name: "policy_above_thresh"
        )
        let aboveFloat = graph.cast(
            aboveThreshold,
            to: dtype,
            name: "policy_above_float"
        )
        let countPerPos = graph.reductionSum(
            with: aboveFloat,
            axis: 1,
            name: "policy_nonneg_per_pos"
        )
        let policyNonNegCount = graph.mean(
            of: countPerPos,
            axes: [0, 1],
            name: "policy_nonneg_count"
        )

        // Illegal-cell count: cells whose UNMASKED softmax is above
        // 1/policySize, restricted to illegal positions via the mask.
        // A healthy network with the legal mask doing its job sees
        // illegal mass approach 0, so this count should trend toward
        // 0 over training. A rising illegal-above-uniform count is a
        // direct signal that mass is leaking onto illegal cells.
        let unmaskedSoftmax = graph.softMax(
            with: network.policyOutput,
            axis: 1,
            name: "policy_softmax_unmasked"
        )
        let unmaskedAboveThreshold = graph.greaterThan(
            unmaskedSoftmax,
            nonNegThreshold,
            name: "policy_above_thresh_unmasked"
        )
        let unmaskedAboveFloat = graph.cast(
            unmaskedAboveThreshold,
            to: dtype,
            name: "policy_above_float_unmasked"
        )
        // Multiply by the illegal mask (per-position vector with 1.0
        // at illegal indices, 0.0 at legal). Sum gives per-position
        // count of illegal cells above uniform.
        let illegalAboveFloat = graph.multiplication(
            unmaskedAboveFloat,
            illegalMask,
            name: "policy_above_illegal_per_cell"
        )
        let illegalCountPerPos = graph.reductionSum(
            with: illegalAboveFloat,
            axis: 1,
            name: "policy_nonneg_illegal_per_pos"
        )
        let policyNonNegIllegalCount = graph.mean(
            of: illegalCountPerPos,
            axes: [0, 1],
            name: "policy_nonneg_illegal_count"
        )

        // --- Policy logit-magnitude probe (diagnostic) ---
        //
        // Batch mean of `max_i |logits[i]|`. Pre-saturation early
        // warning: entropy alone can look healthy while a single
        // runaway logit is already pulling the softmax toward a
        // one-hot, so a direct measurement of the largest logit
        // magnitude complements `policyEntropy`. Diagnostic only —
        // not on the totalLoss autograd path, so the lack of a
        // gradient for `reductionMaximum` is fine.
        let policyLogitAbs = graph.absolute(
            with: network.policyOutput,
            name: "policy_logit_abs"
        )
        let policyLogitAbsMaxPerPos = graph.reductionMaximum(
            with: policyLogitAbs,
            axis: 1,
            name: "policy_logit_abs_max_per_pos"
        )
        let policyLogitAbsMax = graph.mean(
            of: policyLogitAbsMaxPerPos,
            axes: [0, 1],
            name: "policy_logit_abs_max"
        )

        // --- Played-move probability (diagnostic) ---
        //
        // Per-position probability the softmax assigns to the actually
        // played move: `softmax[movePlayed]`, computed as
        // `sum(softmax * oneHot)` along the class axis. Reuses the
        // existing `oneHot` and `softmax` tensors so no new materialization
        // is needed.
        //
        // The **unconditional** batch mean of this quantity is directionally
        // ambiguous under this trainer's advantage-normalized policy loss.
        // `advantage_normalized` has zero batch-mean by construction, so
        // ~half the positions pull `p(a*)` up and ~half pull it down —
        // the unconditional mean can sit near `1/policySize` even when
        // learning is healthy. We keep it as a coarse index-mismatch
        // probe (both conditionals flat near `1/policySize` is strong
        // evidence of action-index misalignment) and emit two
        // **advantage-sign-conditional** means as the real direction-of-
        // learning signal: `playedMoveProbPosAdv` should rise and
        // `playedMoveProbNegAdv` should fall as training progresses. The
        // divergence between the two is the health signal, not the raw
        // mean.
        let playedSoftmaxMasked = graph.multiplication(
            softmax,
            oneHot,
            name: "played_softmax_masked"
        )
        let playedProbPerPos = graph.reductionSum(
            with: playedSoftmaxMasked,
            axis: 1,
            name: "played_prob_per_pos"
        )
        let playedMoveProbTensor = graph.mean(
            of: playedProbPerPos,
            axes: [0, 1],
            name: "played_move_prob"
        )

        // Advantage-sign masks on the raw advantage `A = z - vBaseline`
        // (not the batch-normalized form — we want the intrinsic sign
        // of the REINFORCE weight for the diagnostic, not a post-
        // centering reclassification). Shape [batch, 1], same as
        // `advantage` and `playedProbPerPos`.
        let zeroConstPlayedProb = graph.constant(0.0, dataType: dtype)
        let playedPosMaskBool = graph.greaterThan(
            advantage,
            zeroConstPlayedProb,
            name: "played_prob_pos_mask_bool"
        )
        let playedPosMask = graph.cast(
            playedPosMaskBool,
            to: dtype,
            name: "played_prob_pos_mask"
        )
        let playedNegMaskBool = graph.lessThan(
            advantage,
            zeroConstPlayedProb,
            name: "played_prob_neg_mask_bool"
        )
        let playedNegMask = graph.cast(
            playedNegMaskBool,
            to: dtype,
            name: "played_prob_neg_mask"
        )
        // Conditional mean = E[p(a*) · 1[A>0]] / E[1[A>0]]. Using batch
        // means rather than raw sums keeps the scale identical to the
        // unconditional `played_move_prob` so the three metrics are
        // directly comparable in logs. Division by zero (no batch row
        // has A>0) yields NaN; the Swift-side `recordStep` guards on
        // `isFinite` before pushing into the rolling window.
        let playedProbPosTensor = graph.multiplication(
            playedProbPerPos,
            playedPosMask,
            name: "played_prob_pos"
        )
        let playedProbNegTensor = graph.multiplication(
            playedProbPerPos,
            playedNegMask,
            name: "played_prob_neg"
        )
        let playedPosProductMean = graph.mean(
            of: playedProbPosTensor,
            axes: [0, 1],
            name: "played_prob_pos_product_mean"
        )
        let playedNegProductMean = graph.mean(
            of: playedProbNegTensor,
            axes: [0, 1],
            name: "played_prob_neg_product_mean"
        )
        let playedPosFrac = graph.mean(
            of: playedPosMask,
            axes: [0, 1],
            name: "played_prob_pos_frac"
        )
        let playedNegFrac = graph.mean(
            of: playedNegMask,
            axes: [0, 1],
            name: "played_prob_neg_frac"
        )
        let playedMoveProbPosAdvTensor = graph.division(
            playedPosProductMean,
            playedPosFrac,
            name: "played_move_prob_pos_adv"
        )
        let playedMoveProbNegAdvTensor = graph.division(
            playedNegProductMean,
            playedNegFrac,
            name: "played_move_prob_neg_adv"
        )

        // --- Advantage-distribution scalars (diagnostic) ---
        //
        // `advantage` is the [batch, 1] per-position `z − vBaseline`
        // term that weights the policy loss. Its distribution tells
        // us whether the baseline is absorbing outcome variance
        // (advantageMean near 0, small std = good) or biasing
        // updates in one direction (mean far from 0), and whether
        // updates have the right dynamic range (std, min, max).
        //
        // None of these scalars flow into totalLoss — they are
        // diagnostic-only `targetTensors`, so `reductionMinimum` /
        // `reductionMaximum`'s missing autograd rules aren't an
        // issue.
        let advantageMeanTensor = graph.mean(
            of: advantage,
            axes: [0, 1],
            name: "advantage_mean"
        )
        let advantageSqForStd = graph.square(
            with: advantage,
            name: "advantage_sq"
        )
        let advantageMeanSq = graph.mean(
            of: advantageSqForStd,
            axes: [0, 1],
            name: "advantage_mean_sq"
        )
        // Var = E[A²] − (E[A])². Use unbiased? No — the batch is not
        // a sample of an unknown population; it's just this batch.
        // Biased (population) variance is the natural descriptor.
        let advantageMeanSquared = graph.multiplication(
            advantageMeanTensor,
            advantageMeanTensor,
            name: "advantage_mean_squared"
        )
        let advantageVar = graph.subtraction(
            advantageMeanSq,
            advantageMeanSquared,
            name: "advantage_var"
        )
        // Clamp to zero before sqrt — E[A²] − E[A]² is nonnegative
        // in exact arithmetic but can go slightly negative under
        // float rounding when the batch is extremely homogeneous.
        let zeroConst = graph.constant(0.0, dataType: dtype)
        let advantageVarClamped = graph.maximum(
            advantageVar,
            zeroConst,
            name: "advantage_var_clamped"
        )
        let advantageStdTensor = graph.squareRoot(
            with: advantageVarClamped,
            name: "advantage_std"
        )
        let advantageMinTensor = graph.reductionMinimum(
            with: advantage,
            axes: [0, 1],
            name: "advantage_min"
        )
        let advantageMaxTensor = graph.reductionMaximum(
            with: advantage,
            axes: [0, 1],
            name: "advantage_max"
        )
        // frac(A > 0): cast comparison to float, mean over batch.
        let advantageGreaterZero = graph.greaterThan(
            advantage,
            zeroConst,
            name: "advantage_pos_mask"
        )
        let advantageGreaterZeroFloat = graph.cast(
            advantageGreaterZero,
            to: dtype,
            name: "advantage_pos_mask_float"
        )
        let advantageFracPosTensor = graph.mean(
            of: advantageGreaterZeroFloat,
            axes: [0, 1],
            name: "advantage_frac_pos"
        )
        // frac(|A| < 0.05): "near-zero-signal" positions whose
        // policy-gradient contribution is tiny. Threshold 0.05
        // picked to match the default `drawPenalty` — positions
        // where the fresh baseline already predicts z closely are
        // "well-learned" and shouldn't update much.
        let advantageAbs = graph.absolute(with: advantage, name: "advantage_abs")
        let smallThreshold = graph.constant(0.05, dataType: dtype)
        let advantageSmallMask = graph.lessThan(
            advantageAbs,
            smallThreshold,
            name: "advantage_small_mask"
        )
        let advantageSmallMaskFloat = graph.cast(
            advantageSmallMask,
            to: dtype,
            name: "advantage_small_mask_float"
        )
        let advantageFracSmallTensor = graph.mean(
            of: advantageSmallMaskFloat,
            axes: [0, 1],
            name: "advantage_frac_small"
        )

        // --- Total loss ---
        //
        // Policy loss is REINFORCE on the played move over a `policySize`-way
        // softmax, so its gradient is naturally much weaker than the
        // value head's (z−v)² gradient. Scale the policy term up by K
        // so both heads get meaningful gradient during the pre-MCTS
        // bootstrap phase of training.
        //
        // K is applied as a true coefficient on policyLoss only — no
        // global normalizer, because dividing the sum divides every
        // term and cancels the relative boost. If the larger effective
        // learning rate on the shared trunk causes instability, lower
        // the LR rather than adding a normalizer. Live-tunable via
        // the `policyScaleK` placeholder so the user can dial it
        // down if the amplified policy gradient is the source of
        // gradient-clip saturation.
        let lrTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "learning_rate"
        )
        let entropyCoeffTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "entropy_regularization_coeff"
        )
        let weightDecayTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "weight_decay_coeff"
        )
        let gradClipMaxNormTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "grad_clip_max_norm"
        )
        let policyScaleKTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "policy_scale_k"
        )
        let weightedPolicy = graph.multiplication(
            policyScaleKTensor,
            policyLoss,
            name: "weighted_policy_loss"
        )
        let entropyPenalty = graph.multiplication(
            entropyCoeffTensor,
            policyEntropy,
            name: "entropy_regularization_term"
        )
        let lossWithoutEntropy = graph.addition(
            valueLoss,
            weightedPolicy,
            name: "loss_without_entropy_regularization"
        )
        let totalLossTensor = graph.subtraction(
            lossWithoutEntropy,
            entropyPenalty,
            name: "total_loss"
        )

        // --- Gradients w.r.t. trainable variables ---

        let grads = graph.gradients(
            of: totalLossTensor,
            with: network.trainableVariables,
            name: "gradients"
        )

        // --- Global L2 norm across all gradients ---
        //
        // Compute once, reused in (a) the clip-scale denominator and
        // (b) the readback path so the UI can see the pre-clip norm
        // on every step.
        //
        // Per-variable: flatten → square → reduce-sum to a scalar.
        // Then sum all per-variable scalars and take sqrt to get the
        // global L2 norm.
        var gradSumOfSquares: MPSGraphTensor?
        var firstGradVariableName: String?
        for (i, variable) in network.trainableVariables.enumerated() {
            guard let grad = grads[variable] else {
                throw ChessTrainerError.gradientMissing(
                    variable.operation.name.isEmpty ? "trainable[\(i)]" : variable.operation.name
                )
            }
            if firstGradVariableName == nil {
                firstGradVariableName = variable.operation.name
            }
            let flat = graph.reshape(grad, shape: [-1], name: nil)
            let sq = graph.square(with: flat, name: nil)
            let scalar = graph.reductionSum(with: sq, axis: 0, name: nil)
            if let accum = gradSumOfSquares {
                gradSumOfSquares = graph.addition(accum, scalar, name: nil)
            } else {
                gradSumOfSquares = scalar
            }
        }
        // Non-empty `trainableVariables` is a precondition — every
        // network built by `ChessNetwork` exposes its weights. If it
        // is somehow empty, training is meaningless; surface the
        // first-variable mismatch rather than hand back a graph with
        // no update ops.
        guard let gradSumOfSquaresTensor = gradSumOfSquares else {
            throw ChessTrainerError.gradientMissing(
                firstGradVariableName ?? "(no trainable variables)"
            )
        }
        // `shape: [-1]` on a rank-0 scalar would fail — but every
        // gradient tensor has at least one element, so `sq` is at
        // least shape `[1]` after flatten-then-square, and
        // reductionSum over axis 0 gives shape `[1]`. The global
        // accumulator has the same shape.
        let gradGlobalNorm = graph.squareRoot(
            with: gradSumOfSquaresTensor,
            name: "grad_global_norm"
        )

        // --- Policy head final-conv weight L2 norm (diagnostic) ---
        //
        // Tracks the magnitude of the specific tensor whose logit-scale
        // growth is the mechanism behind extreme policy concentration:
        // large ||W||₂ means at least one row can produce outsized
        // logits and saturate the softmax on one move. Read via
        // targetTensor alongside losses so the host never pulls the
        // full 9.8K-float weight buffer back each step.
        //
        // `graph.read(variable)` explicitly materializes the variable's
        // current value as a tensor before the reshape. Reshaping a
        // variable reference directly works in most MPSGraph paths, but
        // has been observed to cause `mps.placeholder` lowering issues
        // in some runtime configurations. The read is zero-cost at
        // runtime (variables are already resident) and keeps the
        // downstream op chain fully tensor-valued.
        let policyWeightsRead = graph.read(
            network.policyHeadFinalWeights,
            name: "policy_weights_read"
        )
        let policyWeightFlat = graph.reshape(
            policyWeightsRead,
            shape: [-1],
            name: "policy_weight_flat"
        )
        let policyWeightSq = graph.square(
            with: policyWeightFlat,
            name: "policy_weight_sq"
        )
        let policyWeightSqSum = graph.reductionSum(
            with: policyWeightSq,
            axis: 0,
            name: "policy_weight_sq_sum"
        )
        let policyHeadWeightNormTensor = graph.squareRoot(
            with: policyWeightSqSum,
            name: "policy_weight_norm"
        )

        // --- Gradient clip scale: maxNorm / max(norm, maxNorm) ---
        //
        // Equivalent to `min(1, maxNorm / norm)`. When `norm ≤ maxNorm`
        // the scale is 1 (no-op); above the threshold the scale
        // shrinks so the resulting update has L2 norm exactly
        // `maxNorm`. No epsilon needed because `max(norm, maxNorm)`
        // is always ≥ maxNorm > 0.
        let clipDenom = graph.maximum(
            gradGlobalNorm,
            gradClipMaxNormTensor,
            name: "grad_clip_denom"
        )
        let clipScale = graph.division(
            gradClipMaxNormTensor,
            clipDenom,
            name: "grad_clip_scale"
        )

        // --- SGD updates with weight decay + clipped gradients ---
        //
        // v_new = v - lr * (clipped_grad + weightDecayC * v)
        //       = (1 - lr*weightDecayC) * v - lr * clipped_grad
        //
        // Plain SGD with L2 weight decay (no momentum, no Adam state).
        // Decay is applied only to variables flagged in
        // `network.trainableShouldDecay` — conv and FC weight matrices
        // — and skipped for BN gamma/beta and biases per the standard
        // PyTorch / AdamW recipe.
        //
        // The learning rate is a placeholder (not a constant) so it
        // can be changed between steps without rebuilding the graph.
        // Each training step feeds the current `self.learningRate`
        // via the pre-allocated `lrNDArray`.

        var ops: [MPSGraphOperation] = []
        ops.reserveCapacity(network.trainableVariables.count)
        precondition(
            network.trainableShouldDecay.count == network.trainableVariables.count,
            "ChessNetwork.trainableShouldDecay must align 1:1 with trainableVariables"
        )
        for (i, variable) in network.trainableVariables.enumerated() {
            guard let grad = grads[variable] else {
                // Already checked in the norm-accumulation loop above,
                // but re-guard here so a future refactor that splits
                // the two loops can't silently drop a variable.
                throw ChessTrainerError.gradientMissing(
                    variable.operation.name.isEmpty ? "trainable[\(i)]" : variable.operation.name
                )
            }
            // Apply the global L2 clip scale to this gradient.
            let clippedGrad = graph.multiplication(grad, clipScale, name: nil)
            // L2 weight decay term: c*v. Skipped for BN gamma/beta
            // and FC biases per the standard no-decay recipe.
            let combinedUpdate: MPSGraphTensor
            if network.trainableShouldDecay[i] {
                let decayTerm = graph.multiplication(
                    variable,
                    weightDecayTensor,
                    name: nil
                )
                combinedUpdate = graph.addition(clippedGrad, decayTerm, name: nil)
            } else {
                combinedUpdate = clippedGrad
            }
            let scaled = graph.multiplication(lrTensor, combinedUpdate, name: nil)
            let updated = graph.subtraction(variable, scaled, name: nil)
            let assignOp = graph.assign(variable, tensor: updated, name: nil)
            ops.append(assignOp)
        }

        // Include BN running-stat EMA updates from ChessNetwork's
        // training-mode BN layers. These run as targetOperations on
        // every trainStep alongside the SGD assigns, so the running
        // stats converge toward typical per-channel activation
        // statistics as training progresses — giving a sibling
        // inference network the calibration data it needs to produce
        // outputs matching training-time forward passes after
        // loadWeights().
        ops.append(contentsOf: network.bnRunningStatsAssignOps)

        return (
            movePlayed, z, vBaseline, legalMask,
            lrTensor, entropyCoeffTensor, weightDecayTensor, gradClipMaxNormTensor, policyScaleKTensor,
            totalLossTensor, policyLoss, valueLoss,
            policyEntropy, policyNonNegCount, policyNonNegIllegalCount, gradGlobalNorm, valueMean, valueAbsMean, policyHeadWeightNormTensor,
            policyLogitAbsMax, playedMoveProbTensor,
            playedMoveProbPosAdvTensor, playedMoveProbNegAdvTensor,
            advantageMeanTensor, advantageStdTensor, advantageMinTensor, advantageMaxTensor,
            advantageFracPosTensor, advantageFracSmallTensor,
            advantage,
            policyLossWin, policyLossLoss,
            ops
        )
    }

    // MARK: - Training Step

    /// Run a single training step on a batch of randomly synthesized data.
    /// Returns timing breakdown and the loss scalar. Repeated calls update
    /// this trainer's internal network weights via SGD — that's how we
    /// verified the training pipeline is mechanically correct (random data
    /// + random labels + monotonically decreasing loss). The trainer's
    /// internal network is **not** the inference network, so these updates
    /// don't affect Play Game or Forward Pass.
    func trainStep(batchSize: Int) async throws -> TrainStepTiming {
        try await enqueue {
            try self.internalTrainStep(batchSize: batchSize)
        }
    }

    /// Observed / seeded completed-step count. Getter and setter
    /// both go through the underlying `SyncBox` (os_unfair_lock):
    /// the read returns whatever the most recent training-step
    /// increment published; the setter is the session-resume path,
    /// where assigning a non-negative value overwrites the counter
    /// so warmup scaling resumes mid-session. Reading does NOT
    /// touch `executionQueue`, so a UI poll never blocks on an
    /// in-flight SGD step.
    var completedTrainSteps: Int {
        get { _completedTrainSteps.value }
        set { _completedTrainSteps.value = max(0, newValue) }
    }

    /// Effective learning rate that the optimizer is currently being
    /// fed, given the active warmup multiplier and (optionally) the
    /// sqrt-batch scaling rule. Mirrors the in-graph math at
    /// `buildFeeds` step time so a status-bar readout matches what the
    /// training step is actually applying. Reads the step count from
    /// the `SyncBox`, not the `executionQueue`, so a status-bar
    /// readout never blocks on an in-flight SGD step.
    func effectiveLearningRate(forBatchSize batchSize: Int) -> Float {
        let steps = _completedTrainSteps.value
        let warmupMul: Float
        if lrWarmupSteps > 0 {
            warmupMul = Float(min(1.0, Double(steps) / Double(lrWarmupSteps)))
        } else {
            warmupMul = 1.0
        }
        var lr: Float
        if sqrtBatchScalingForLR {
            let sqrtBatchScale: Float = Float(
                sqrt(Double(batchSize) / Double(Self.sqrtScaleBaseBatchSize))
            )
            lr = learningRate * sqrtBatchScale
        } else {
            lr = learningRate
        }
        return lr * warmupMul
    }

    private func internalTrainStep(batchSize: Int) throws -> TrainStepTiming {
        let totalStart = CFAbsoluteTimeGetCurrent()

        // --- Data prep: synthesize random boards, moves, outcomes ---

        let prepStart = CFAbsoluteTimeGetCurrent()
        let floatsPerBoard = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
        let totalBoardFloats = batchSize * floatsPerBoard

        var boardFloats = [Float](repeating: 0, count: totalBoardFloats)
        Self.fillRandomFloats(&boardFloats)

        var moveIndices = [Int32](repeating: 0, count: batchSize)
        // Random move indices in [0, policySize). One per batch row.
        for i in 0..<batchSize {
            moveIndices[i] = Int32.random(in: 0..<Int32(ChessNetwork.policySize))
        }

        var zValues = [Float](repeating: 0, count: batchSize)
        // Random outcomes from {-1, 0, +1} so the loss includes all three
        // signed regimes (push up, push down, no contribution).
        for i in 0..<batchSize {
            zValues[i] = Float(Int.random(in: 0..<3) - 1)
        }

        // vBaselines: all zeros for the random-data sweep. An all-zero
        // baseline degrades the advantage formulation to `z * negLogProb`,
        // which is exactly what the random-data smoke test measured
        // historically — so losses stay comparable to prior sweep runs.
        let vBaselineValues = [Float](repeating: 0, count: batchSize)

        // All-ones (no masking) for the synthetic-data path; the additive
        // mask term then evaluates to 0 everywhere and the graph behaves
        // identically to the pre-masking version. The real per-position
        // legal mask is built in `trainStepFromReplay`.
        let legalMaskValues = [Float](repeating: 1.0, count: batchSize * ChessNetwork.policySize)

        // Unbox the four Swift arrays into raw pointers and feed
        // them through the shared pointer-based `buildFeeds` /
        // `runPreparedStep` pipeline.
        return try boardFloats.withUnsafeBufferPointer { boardsBuf in
            try moveIndices.withUnsafeBufferPointer { movesBuf in
                try zValues.withUnsafeBufferPointer { zsBuf in
                    try vBaselineValues.withUnsafeBufferPointer { vBaseBuf in
                        try legalMaskValues.withUnsafeBufferPointer { legalMasksBuf in
                            // The arrays were allocated just above
                            // with positive batch size, so their
                            // `baseAddress`es are guaranteed non-nil.
                            guard
                                let boardsBase = boardsBuf.baseAddress,
                                let movesBase = movesBuf.baseAddress,
                                let zsBase = zsBuf.baseAddress,
                                let vBaseBase = vBaseBuf.baseAddress,
                                let legalMasksBase = legalMasksBuf.baseAddress
                            else {
                                preconditionFailure(
                                    "ChessTrainer.trainStep(batchSize:): non-empty inputs should have baseAddress"
                                )
                            }
                            let feeds = buildFeeds(
                                batchSize: batchSize,
                                boards: boardsBase,
                                moves: movesBase,
                                zs: zsBase,
                                vBaselines: vBaseBase,
                                legalMasks: legalMasksBase
                            )
                            let prepMs = (CFAbsoluteTimeGetCurrent() - prepStart) * 1000
                            return try runPreparedStep(
                                feeds: feeds,
                                prepMs: prepMs,
                                totalStart: totalStart
                            )
                        }
                    }
                }
            }
        }
    }

    /// Run a single training step on a batch sampled directly from the
    /// replay buffer into trainer-owned staging storage. Returns `nil`
    /// when the buffer has not yet accumulated `batchSize` positions.
    ///
    /// **Fresh-baseline forward pass:** before the actual training
    /// step runs, this method does a forward-only pass on the
    /// trainer's CURRENT network to compute v(s) for every position
    /// in the batch. Those fresh v values overwrite the play-time-
    /// frozen `vBaseline` values stored in the replay buffer, so the
    /// policy-gradient advantage `(z - vBaseline)` reflects the
    /// trainer's current belief instead of the random-init champion's
    /// belief from when the move was played.
    ///
    /// Why this is necessary: MPSGraph has no `stop_gradient` op, so
    /// computing v(s) inside the same training graph as both the
    /// value-loss target AND the policy-baseline causes gradient to
    /// leak back through the baseline path into the tower (verified
    /// empirically — see `MPSGraphGradientSemanticsTests`). The
    /// `vBaseline` placeholder mechanism that's already in the
    /// training graph IS a stop-gradient boundary; we just feed
    /// fresher values into it via this extra forward pass.
    ///
    /// Cost: ~33% extra forward FLOPs per training step. Worth it
    /// because the previous behavior used random-init values forever
    /// (until promotion happened), causing seed-dependent training
    /// dynamics and biasing draw advantages toward positive (which
    /// reinforced shuffle moves).
    func trainStep(
        replayBuffer: ReplayBuffer,
        batchSize: Int
    ) async throws -> TrainStepTiming? {
        // Phase 1 (trainer queue): sample into the staging buffers and
        // copy the boards out as a Sendable [Float] for the cross-queue
        // hop into the network's evaluate. Also copy the stale
        // vBaselines so we can compute the diagnostic delta at the end.
        struct Phase1: Sendable {
            let boardsCopy: [Float]
            let staleVBaselines: [Float]
        }
        let phase1: Phase1? = try await enqueue { [batchSize] in
            self.ensureReplayBatchCapacity(batchSize)
            guard
                let boards = self.replayBatchBoards,
                let moves = self.replayBatchMoves,
                let zs = self.replayBatchZs,
                let vBaselines = self.replayBatchVBaselines,
                let plies = self.replayBatchPlies,
                let gameLengths = self.replayBatchGameLengths,
                let taus = self.replayBatchTaus,
                let hashes = self.replayBatchHashes,
                let workerGameIds = self.replayBatchWorkerGameIds,
                let materials = self.replayBatchMaterialCounts
            else {
                preconditionFailure("ChessTrainer.ensureReplayBatchCapacity should populate replay staging buffers")
            }
            // Whether THIS step computes batch-stats — gates whether
            // the metadata buffers get filled. The interval is read
            // here on the trainer queue so toggling 0->N from the UI
            // takes effect immediately on the next step without
            // racing the in-flight one.
            let interval = self.batchStatsInterval
            let nextStep = self._completedTrainSteps.value + 1
            let isStatsStep = interval > 0 && nextStep % interval == 0
            let didSample = replayBuffer.sample(
                count: batchSize,
                intoBoards: boards,
                moves: moves,
                zs: zs,
                vBaselines: vBaselines,
                plies: isStatsStep ? plies : nil,
                gameLengths: isStatsStep ? gameLengths : nil,
                taus: isStatsStep ? taus : nil,
                hashes: isStatsStep ? hashes : nil,
                workerGameIds: isStatsStep ? workerGameIds : nil,
                materialCounts: isStatsStep ? materials : nil
            )
            guard didSample else { return nil }
            // Compute batch stats up-front (cheap, ~1 ms) and emit the
            // line BEFORE the heavy GPU work fires. Doing it here keeps
            // it on the trainer queue (no cross-queue ownership of the
            // metadata pointers) and means a stats failure can't
            // interrupt training.
            if isStatsStep {
                let summary = replayBuffer.computeBatchStats(
                    step: nextStep,
                    batchSize: batchSize,
                    plies: plies,
                    gameLengths: gameLengths,
                    taus: taus,
                    hashes: hashes,
                    workerGameIds: workerGameIds,
                    materialCounts: materials,
                    zs: zs
                )
                self.lastBatchStatsUniquePct = summary.uniquePct
                self.lastBatchStatsSummary = summary
                SessionLogger.shared.log("[BATCH-STATS] " + summary.jsonLine())
            }
            let floatsPerBoard = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
            let totalFloats = batchSize * floatsPerBoard
            let boardsCopy = Array(UnsafeBufferPointer(start: boards, count: totalFloats))
            let staleCopy = Array(UnsafeBufferPointer(start: vBaselines, count: batchSize))
            return Phase1(boardsCopy: boardsCopy, staleVBaselines: staleCopy)
        }
        guard let phase1 else { return nil }

        // Phase 2 (network queue, async): forward-only pass on the
        // trainer's network to compute v(s) for every position. We
        // discard the policy output and keep only the value scalars.
        let freshBaselineStart = CFAbsoluteTimeGetCurrent()
        let (freshPolicy, freshValues) = try await network.evaluate(
            batchBoards: phase1.boardsCopy,
            count: batchSize
        )
        let freshBaselineMs = (CFAbsoluteTimeGetCurrent() - freshBaselineStart) * 1000

        // Compute the diagnostic mean-absolute-delta between the
        // trainer's current value head and the play-time champion's
        // frozen value head. If this stays near zero, the trainer
        // isn't diverging from the champion — bad. If it grows over
        // time, the trainer is genuinely learning something different.
        var sumAbsDelta: Float = 0
        for i in 0..<batchSize {
            sumAbsDelta += abs(freshValues[i] - phase1.staleVBaselines[i])
        }
        let meanAbsDelta = sumAbsDelta / Float(batchSize)

        // Phase 3 (trainer queue): overwrite vBaselines with the fresh
        // values, apply draw penalty, build feeds, run the training
        // graph. Same flow as the pre-fresh-baseline implementation,
        // just with vBaselines now containing current-trainer values
        // instead of replay-buffer-frozen values.
        return try await enqueue { [batchSize, freshValues, freshPolicy, freshBaselineMs, meanAbsDelta] in
            let totalStart = CFAbsoluteTimeGetCurrent()
            let prepStart = CFAbsoluteTimeGetCurrent()

            guard
                let boards = self.replayBatchBoards,
                let moves = self.replayBatchMoves,
                let zs = self.replayBatchZs,
                let vBaselines = self.replayBatchVBaselines,
                let masks = self.replayBatchLegalMasks
            else {
                preconditionFailure("ChessTrainer staging buffers vanished between phases")
            }

            // Overwrite the play-time-frozen vBaselines with the
            // fresh current-trainer values from phase 2.
            for i in 0..<batchSize {
                vBaselines[i] = freshValues[i]
            }

            // Draw-penalty rewrite: draws arrive with z=0.0 exactly
            // (see `MPSChessPlayer.onGameEnded` — the four draw
            // results all assign `0.0` with no float arithmetic in
            // between). When `drawPenalty > 0`, substitute
            // `-drawPenalty` for every drawn position in this batch.
            // Mutating the replay staging buffer in place is safe —
            // it's private to the trainer and is fully overwritten by
            // the next `sample()` call.
            if self.drawPenalty > 0 {
                let penalty = -self.drawPenalty
                for i in 0..<batchSize where zs[i] == 0.0 {
                    zs[i] = penalty
                }
            }

            // NEW: populate the legal-move mask for each position in the batch.
            let policySize = ChessNetwork.policySize
            let floatsPerBoard = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize

            // Zero the entire mask buffer first — cheaper than zeroing per-row inside
            // the loop, and the legal-move generator will overwrite the legal indices.
            let totalMaskFloats = batchSize * policySize
            masks.update(repeating: 0.0, count: totalMaskFloats)

            for pos in 0..<batchSize {
                let boardPtr = boards.advanced(by: pos * floatsPerBoard)
                let state = BoardEncoder.decodeSynthetic(from: boardPtr)
                let legalMoves = MoveGenerator.legalMoves(for: state)
                let maskBase = pos * policySize
                for move in legalMoves {
                    let idx = PolicyEncoding.policyIndex(move, currentPlayer: .white)
                    precondition(
                        idx >= 0 && idx < policySize,
                        "PolicyEncoding.policyIndex returned out-of-range index \(idx) for legal move \(move); policySize=\(policySize)"
                    )
                    masks[maskBase + idx] = 1.0
                }
            }

            if self._completedTrainSteps.value == 0 {
                for pos in 0..<min(8, batchSize) {
                    let movedIdx = Int(moves[pos])
                    let inLegalMask = masks[pos * ChessNetwork.policySize + movedIdx] == 1.0
                    var legalCount: Int = 0
                    for i in 0..<ChessNetwork.policySize {
                        if masks[pos * ChessNetwork.policySize + i] == 1.0 { legalCount += 1 }
                    }
                    SessionLogger.shared.log(
                        "[MASK CHECK] pos=\(pos) movedIdx=\(movedIdx) inLegalMask=\(inLegalMask) legalCount=\(legalCount)"
                    )
                }
            }
//
//            // One-shot at step 200: confirm the additive -1e9 mask is
//            // actually wiping illegal-cell mass. Computes
//            // `softmax(maskedLogits)` for the first batch position
//            // (using the same -1e9 constant the graph uses) and reports
//            // the summed mass over legal vs. illegal indices. A healthy
//            // run shows legal_sum ≈ 1.0 and illegal_sum ≈ 0.0; a non-
//            // zero illegal_sum here would mean the mask isn't reaching
//            // the loss path. Uses `freshPolicy` (raw logits captured
//            // from the phase-2 forward pass) so no extra GPU work is
//            // spent on the probe.
//            if self._completedTrainSteps == 200 {
//                let policySize = ChessNetwork.policySize
//                let largeNeg: Float = -1e9
//                let logitsBase = 0 // first batch position
//                let maskBase = 0
//                var maxMaskedLogit: Float = -.infinity
//                for i in 0..<policySize {
//                    let mask = masks[maskBase + i]
//                    let masked = freshPolicy[logitsBase + i] + (1 - mask) * largeNeg
//                    if masked > maxMaskedLogit { maxMaskedLogit = masked }
//                }
//                var expSum: Double = 0
//                var legalExpSum: Double = 0
//                var illegalExpSum: Double = 0
//                var legalCount: Int = 0
//                for i in 0..<policySize {
//                    let mask = masks[maskBase + i]
//                    let masked = freshPolicy[logitsBase + i] + (1 - mask) * largeNeg
//                    let e = Double(expf(masked - maxMaskedLogit))
//                    expSum += e
//                    if mask == 1.0 {
//                        legalExpSum += e
//                        legalCount += 1
//                    } else {
//                        illegalExpSum += e
//                    }
//                }
//                let legalSum = expSum > 0 ? legalExpSum / expSum : 0
//                let illegalSum = expSum > 0 ? illegalExpSum / expSum : 0
//                SessionLogger.shared.log(
//                    String(
//                        format: "[MASKED-SOFTMAX] step=200 pos=0 legalCount=%d legal_sum=%.6e illegal_sum=%.6e",
//                        legalCount, legalSum, illegalSum
//                    )
//                )
//            }

            let feeds = self.buildFeeds(
                batchSize: batchSize,
                boards: UnsafePointer(boards),
                moves: UnsafePointer(moves),
                zs: UnsafePointer(zs),
                vBaselines: UnsafePointer(vBaselines),
                legalMasks: UnsafePointer(masks)
            )
            let prepMs = (CFAbsoluteTimeGetCurrent() - prepStart) * 1000

            // Run the training step. The returned timing has nil
            // fresh-baseline fields; we patch them in below.
            let baseTiming = try self.runPreparedStep(
                feeds: feeds,
                prepMs: prepMs,
                totalStart: totalStart
            )

            // Count a successfully-completed real-data SGD step, for
            // the LR-warmup multiplier in `buildFeeds`. Only real-
            // data steps advance warmup — the random-data
            // `trainStep(batchSize:)` path (used by GPU-sweep
            // diagnostics and continuous-training smoke tests) runs
            // `runPreparedStep` too, but warmup there would consume
            // the ramp-up against meaningless random labels, leaving
            // real Play-and-Train starting with a post-warmup LR.
            //
            // `modify` makes the read-modify-write atomic under the
            // SyncBox's os_unfair_lock. Even though this site only
            // runs from inside an `enqueue { ... }` block today, an
            // off-queue `+= 1` would race with the public setter
            // (which writes through the same SyncBox) — the lock
            // protects the increment in its own right rather than
            // relying on the queue invariant.
            self._completedTrainSteps.modify { $0 += 1 }

            return TrainStepTiming(
                dataPrepMs: baseTiming.dataPrepMs,
                gpuRunMs: baseTiming.gpuRunMs,
                readbackMs: baseTiming.readbackMs,
                // Include the fresh-baseline forward-pass time so the
                // replay-ratio controller (and any downstream throughput
                // calculation) sees the true wall-clock cost of one
                // training step. `freshBaselineMs` is also kept as a
                // separate diagnostic field for visibility, but
                // `totalMs` is the user-facing "this step took N ms"
                // figure that controllers throttle against.
                totalMs: baseTiming.totalMs + freshBaselineMs,
                loss: baseTiming.loss,
                policyLoss: baseTiming.policyLoss,
                valueLoss: baseTiming.valueLoss,
                policyEntropy: baseTiming.policyEntropy,
                policyNonNegligibleCount: baseTiming.policyNonNegligibleCount,
                policyNonNegligibleIllegalCount: baseTiming.policyNonNegligibleIllegalCount,
                gradGlobalNorm: baseTiming.gradGlobalNorm,
                valueMean: baseTiming.valueMean,
                valueAbsMean: baseTiming.valueAbsMean,
                vBaselineDelta: meanAbsDelta,
                freshBaselineMs: freshBaselineMs,
                policyHeadWeightNorm: baseTiming.policyHeadWeightNorm,
                policyLogitAbsMax: baseTiming.policyLogitAbsMax,
                playedMoveProb: baseTiming.playedMoveProb,
                playedMoveProbPosAdv: baseTiming.playedMoveProbPosAdv,
                playedMoveProbNegAdv: baseTiming.playedMoveProbNegAdv,
                advantageMean: baseTiming.advantageMean,
                advantageStd: baseTiming.advantageStd,
                advantageMin: baseTiming.advantageMin,
                advantageMax: baseTiming.advantageMax,
                advantageFracPositive: baseTiming.advantageFracPositive,
                advantageFracSmall: baseTiming.advantageFracSmall,
                advantageRaw: baseTiming.advantageRaw,
                policyLossWin: baseTiming.policyLossWin,
                policyLossLoss: baseTiming.policyLossLoss
            )
        }
    }

    /// Snapshot of the current policy's mass distribution over legal
    /// moves, computed over a fresh sample of `sampleSize` positions
    /// from `replayBuffer`. Used by the periodic STATS emit to log a
    /// number that is robust to policy sharpening: as training
    /// progresses, the softmax mass the network places on the legal
    /// move set rises from `~n_legal/policySize` (random init, most
    /// mass on illegal cells) toward 1.0. An index-mismatch bug would
    /// pin this near the random-init value even as other losses move.
    ///
    /// Returns nil when the replay buffer hasn't accumulated
    /// `sampleSize` positions yet.
    ///
    /// **Pass `inferenceNetwork` in production.** When provided, the
    /// probe copies the trainer's current weights into it and runs
    /// the forward pass on the inference-mode network — this keeps
    /// the probe from triggering the training-mode BN's running-mean /
    /// running-variance assign ops, which would otherwise drift the
    /// trainer's running statistics every time a probe fires. Callers
    /// should hand in the app-level `probeInferenceNetwork` (the same
    /// one used by candidate-test probes). The nil path runs the pass
    /// directly against `self.network` and IS affected by BN-stat
    /// pollution — retained only so the function remains callable in
    /// contexts that haven't been migrated (tests, exploratory code).
    /// Production call sites must always pass `inferenceNetwork`.
    func legalMassSnapshot(
        replayBuffer: ReplayBuffer,
        sampleSize: Int,
        inferenceNetwork: ChessMPSNetwork? = nil
    ) async throws -> LegalMassSnapshot? {
        // Sample boards on the trainer queue so we reuse the same
        // replay-buffer concurrency guards as trainStep. We only
        // need the boards — moves/zs/vBaselines are ignored for
        // this probe.
        struct Sampled: Sendable {
            let boards: [Float]
            let count: Int
        }
        let sampled: Sampled? = try await enqueue { [sampleSize] in
            self.ensureReplayBatchCapacity(sampleSize)
            guard
                let boards = self.replayBatchBoards,
                let moves = self.replayBatchMoves,
                let zs = self.replayBatchZs,
                let vBaselines = self.replayBatchVBaselines
            else {
                preconditionFailure("ChessTrainer staging buffers missing")
            }
            let ok = replayBuffer.sample(
                count: sampleSize,
                intoBoards: boards,
                moves: moves,
                zs: zs,
                vBaselines: vBaselines
            )
            guard ok else { return nil }
            let floatsPerBoard = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
            let total = sampleSize * floatsPerBoard
            return Sampled(
                boards: Array(UnsafeBufferPointer(start: boards, count: total)),
                count: sampleSize
            )
        }
        guard let sampled else { return nil }

        // Forward-only pass. Returns raw logits (not softmaxed) in
        // position-major layout.
        //
        // When an inference network is provided, mirror the trainer's
        // current weights into it first (same pattern candidate-test
        // probes use — see `fireCandidateProbeIfNeeded`). The forward
        // pass then runs on the inference-mode network, which does
        // NOT append running-stat assign ops. Without this redirect,
        // every probe call would subtly mutate the trainer's own BN
        // running statistics via the training-mode graph's assigns,
        // and multiple probe callers firing at different cadences
        // (STATS logger 60 s, collapse detector 15 s) compound into a
        // stall where SGD batches see a BN distribution that has
        // drifted away from the one they're trying to normalize —
        // the policy head's legal-mass signal flatlines near 1.0.
        let policy: [Float]
        if let inferenceNetwork {
            let weights = try await network.exportWeights()
            try await inferenceNetwork.loadWeights(weights)
            let (p, _) = try await inferenceNetwork.evaluate(
                batchBoards: sampled.boards,
                count: sampled.count
            )
            policy = p
        } else {
            let (p, _) = try await network.evaluate(
                batchBoards: sampled.boards,
                count: sampled.count
            )
            policy = p
        }

        let floatsPerBoard = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
        let policySize = ChessNetwork.policySize

        var legalMassSum: Double = 0
        var top1LegalCount: Int = 0
        var positionsWithLegal: Int = 0
        // Sum of per-position legal-masked Shannon entropies (in nats).
        // Distinct from the full-policy `pEnt`: it's computed over the
        // legal-only renormalized softmax, so a high value means
        // "policy is diffuse across legal moves" (early-training,
        // healthy) and a low value means "policy is concentrating
        // among legal moves" (mid-training, healthy goal). When the
        // network has placed essentially all mass on illegal cells,
        // legalEntropy still reads a finite value because we
        // renormalize — but the legalMass denominator is tiny, so
        // pEntLegal alone shouldn't be treated as a collapse signal;
        // pair it with legalMass to interpret.
        var legalEntropySum: Double = 0

        // CPU decode + softmax + mask on legal indices. ~2 ms total
        // for sampleSize=128 in micro-benchmarks.
        sampled.boards.withUnsafeBufferPointer { boardsBuf in
            guard let boardsBase = boardsBuf.baseAddress else { return }
            for pos in 0..<sampled.count {
                let boardPtr = boardsBase.advanced(by: pos * floatsPerBoard)
                let state = BoardEncoder.decodeSynthetic(from: boardPtr)
                let legalMoves = MoveGenerator.legalMoves(for: state)
                guard !legalMoves.isEmpty else { continue }
                positionsWithLegal += 1

                // Stable softmax over the slot's policy slice:
                // max-subtract, exp, normalize.
                let base = pos * policySize
                var maxLogit: Float = -.infinity
                for i in 0..<policySize {
                    let v = policy[base + i]
                    if v > maxLogit { maxLogit = v }
                }
                var expSum: Double = 0
                for i in 0..<policySize {
                    expSum += Double(expf(policy[base + i] - maxLogit))
                }
                // Argmax over the full policy — for top1Legal check.
                var argmax = 0
                var argmaxLogit = policy[base]
                for i in 1..<policySize {
                    let v = policy[base + i]
                    if v > argmaxLogit {
                        argmaxLogit = v
                        argmax = i
                    }
                }

                // Sum softmax mass over the legal set.
                var legalExpSum: Double = 0
                var legalIndexSet = Set<Int>()
                legalIndexSet.reserveCapacity(legalMoves.count)
                for move in legalMoves {
                    let idx = PolicyEncoding.policyIndex(move, currentPlayer: .white)
                    guard idx >= 0, idx < policySize else { continue }
                    if legalIndexSet.insert(idx).inserted {
                        legalExpSum += Double(expf(policy[base + idx] - maxLogit))
                    }
                }
                let legalMass = expSum > 0 ? legalExpSum / expSum : 0
                legalMassSum += legalMass
                if legalIndexSet.contains(argmax) {
                    top1LegalCount += 1
                }

                // Legal-masked Shannon entropy. Renormalize the legal-
                // only softmax mass to sum to 1 by dividing each
                // legal-cell exp by `legalExpSum`, then compute
                // -Σ p · log p in nats. Skip when legalExpSum is zero
                // (network has no probability on any legal cell — rare
                // numerical edge case).
                if legalExpSum > 0 {
                    var ent: Double = 0
                    for idx in legalIndexSet {
                        let pUn = Double(expf(policy[base + idx] - maxLogit))
                        let p = pUn / legalExpSum
                        if p > 0 { ent -= p * log(p) }
                    }
                    legalEntropySum += ent
                }
            }
        }

        guard positionsWithLegal > 0 else { return nil }
        return LegalMassSnapshot(
            sampleSize: positionsWithLegal,
            legalMass: legalMassSum / Double(positionsWithLegal),
            top1LegalFraction: Double(top1LegalCount) / Double(positionsWithLegal),
            legalEntropy: legalEntropySum / Double(positionsWithLegal)
        )
    }

    /// Result of `legalMassSnapshot`: batch-averaged softmax mass on
    /// the legal move set and batch fraction where the full-policy
    /// argmax corresponds to a legal move.
    struct LegalMassSnapshot: Sendable {
        let sampleSize: Int
        /// Batch-mean softmax probability mass placed on legal cells.
        /// Range [0, 1]. `legalMoves.count / policySize` at random init
        /// (~0.006 with ~30 legal moves), rising toward 1.0 as the
        /// policy sharpens on the rules.
        let legalMass: Double
        /// Fraction of positions where the full-4864-way argmax
        /// corresponds to a legal move. Rank-based sanity signal.
        let top1LegalFraction: Double
        /// Batch-mean Shannon entropy (in nats) of the legal-only
        /// renormalized softmax. log(N_legal) at random init for a
        /// position with N_legal legal moves (~3.4 nats for 30 legal
        /// moves), shrinking toward 0 as the policy concentrates on
        /// preferred legal moves. Distinguishes "diffuse across
        /// legal moves" (early-training, fine) from "concentrating
        /// onto a single legal move" (mid-training, the goal) — the
        /// full-policy `pEnt` cannot make this distinction because
        /// it conflates legal vs illegal mass.
        let legalEntropy: Double
    }

    private func ensureReplayBatchCapacity(_ needed: Int) {
        guard needed > replayBatchCapacity else { return }

        if let ptr = replayBatchBoards {
            ptr.deinitialize(count: replayBatchCapacity * ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize)
            ptr.deallocate()
        }
        if let ptr = replayBatchMoves {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchZs {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchVBaselines {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchLegalMasks {                              // <-- add
            ptr.deinitialize(count: replayBatchCapacity * ChessNetwork.policySize)
            ptr.deallocate()
        }

        if let ptr = replayBatchPlies {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchGameLengths {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchTaus {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchHashes {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchWorkerGameIds {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = replayBatchMaterialCounts {
            ptr.deinitialize(count: replayBatchCapacity)
            ptr.deallocate()
        }

        let floatsPerBoard = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
        let boardSlots = needed * floatsPerBoard
        let newBoards = UnsafeMutablePointer<Float>.allocate(capacity: boardSlots)
        newBoards.initialize(repeating: 0, count: boardSlots)
        replayBatchBoards = newBoards

        let newMoves = UnsafeMutablePointer<Int32>.allocate(capacity: needed)
        newMoves.initialize(repeating: 0, count: needed)
        replayBatchMoves = newMoves

        let newZs = UnsafeMutablePointer<Float>.allocate(capacity: needed)
        newZs.initialize(repeating: 0, count: needed)
        replayBatchZs = newZs

        let newVBaselines = UnsafeMutablePointer<Float>.allocate(capacity: needed)
        newVBaselines.initialize(repeating: 0, count: needed)
        replayBatchVBaselines = newVBaselines

        let maskFloats = needed * ChessNetwork.policySize                 // <-- add
        let newMasks = UnsafeMutablePointer<Float>.allocate(capacity: maskFloats)
        newMasks.initialize(repeating: 0, count: maskFloats)
        replayBatchLegalMasks = newMasks

        let newPlies = UnsafeMutablePointer<UInt16>.allocate(capacity: needed)
        newPlies.initialize(repeating: 0, count: needed)
        replayBatchPlies = newPlies

        let newGameLengths = UnsafeMutablePointer<UInt16>.allocate(capacity: needed)
        newGameLengths.initialize(repeating: 0, count: needed)
        replayBatchGameLengths = newGameLengths

        let newTaus = UnsafeMutablePointer<Float>.allocate(capacity: needed)
        newTaus.initialize(repeating: 0, count: needed)
        replayBatchTaus = newTaus

        let newHashes = UnsafeMutablePointer<UInt64>.allocate(capacity: needed)
        newHashes.initialize(repeating: 0, count: needed)
        replayBatchHashes = newHashes

        let newWorkerGameIds = UnsafeMutablePointer<UInt32>.allocate(capacity: needed)
        newWorkerGameIds.initialize(repeating: 0, count: needed)
        replayBatchWorkerGameIds = newWorkerGameIds

        let newMaterialCounts = UnsafeMutablePointer<UInt8>.allocate(capacity: needed)
        newMaterialCounts.initialize(repeating: 0, count: needed)
        replayBatchMaterialCounts = newMaterialCounts

        replayBatchCapacity = needed
    }

    private func enqueue<T: Sendable>(_ work: @Sendable @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            executionQueue.async {
                do {
                    continuation.resume(returning: try work())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Pack one training step's raw float/int32 buffers into the feed
    /// dictionary the graph expects. Shared by the random-data and
    /// real-data paths so they can't drift out of sync.
    ///
    /// The ND-array wrappers *and* the feeds dictionary for this batch
    /// size are allocated once on first use and cached in `feedCache`;
    /// every subsequent call at the same batch size reuses them by
    /// writing new values into the ND-array storage in place via
    /// `writeBytes` and returning the cached dict unchanged. The
    /// batch-size sweep's warmup step covers the first allocation;
    /// the timed window then runs allocation-free.
    ///
    /// Takes raw pointers so both the `[Float]`-backed random-data
    /// path and the `ReplayBuffer`-backed real-data path can feed
    /// through without any Swift Array CoW concerns.
    private func buildFeeds(
        batchSize: Int,
        boards: UnsafePointer<Float>,
        moves: UnsafePointer<Int32>,
        zs: UnsafePointer<Float>,
        vBaselines: UnsafePointer<Float>,
        legalMasks: UnsafePointer<Float>
    ) -> [MPSGraphTensor: MPSGraphTensorData] {
        let cached = feedsForBatch(batchSize)

        // Float32-only hot path. The ND array's element type matches
        // ChessNetwork.dataType, so on .float32 we can hand it the raw
        // bytes directly. A .float16 flip would need a reused
        // [UInt16] scratch buffer here and in ChessNetwork's
        // inference writer — fail loud until that exists.
        guard ChessNetwork.dataType == .float32 else {
            fatalError("ChessTrainer.buildFeeds: only .float32 is currently supported; got \(ChessNetwork.dataType)")
        }

        cached.boardND.writeBytes(
            UnsafeMutableRawPointer(mutating: boards),
            strideBytes: nil
        )
        cached.moveND.writeBytes(
            UnsafeMutableRawPointer(mutating: moves),
            strideBytes: nil
        )
        cached.zND.writeBytes(
            UnsafeMutableRawPointer(mutating: zs),
            strideBytes: nil
        )
        cached.vBaselineND.writeBytes(
            UnsafeMutableRawPointer(mutating: vBaselines),
            strideBytes: nil
        )
        cached.legalMaskND.writeBytes(                     // <-- add
            UnsafeMutableRawPointer(mutating: legalMasks),
            strideBytes: nil
        )
        // Write the current learning rate and weight decay into the
        // scalar feeds. Two independent multipliers can kick in on
        // the LR side: sqrt-batch scaling (matches Adam-family
        // batch-size rules around the 4096 pivot) and linear warmup
        // over `lrWarmupSteps` steps (LR lerps from 0 → base over
        // that count, evaluated against `_completedTrainSteps.value`).
        // They compose multiplicatively — e.g. step 250 with a 500-
        // step warmup at batch=2048 is `lr * 0.707 * 0.5 = lr * 0.354`.
        //
        // Weight decay is intentionally NOT batch-scaled — the
        // standard AdamW convention keeps the configured weight
        // decay fixed across batch sizes. The user-visible base LR
        // stays authoritative; scaling and warmup are applied here
        // at write time only, never persisted back.
        let warmupMul: Float
        if lrWarmupSteps > 0 {
            warmupMul = Float(min(1.0, Double(_completedTrainSteps.value) / Double(lrWarmupSteps)))
        } else {
            warmupMul = 1.0
        }
        var lr: Float
        if sqrtBatchScalingForLR {
            let sqrtBatchScale: Float = Float(
                sqrt(Double(batchSize) / Double(Self.sqrtScaleBaseBatchSize))
            )
            lr = learningRate * sqrtBatchScale
        } else {
            lr = learningRate
        }
        lr *= warmupMul
        lrNDArray.writeBytes(&lr, strideBytes: nil)
        var entropyCoeff = entropyRegularizationCoeff
        entropyCoeffNDArray.writeBytes(&entropyCoeff, strideBytes: nil)
        var weightDecay = weightDecayC
        weightDecayNDArray.writeBytes(&weightDecay, strideBytes: nil)
        var gradClip = gradClipMaxNorm
        gradClipMaxNormNDArray.writeBytes(&gradClip, strideBytes: nil)
        var kScale = policyScaleK
        policyScaleKNDArray.writeBytes(&kScale, strideBytes: nil)

        return cached.feedsDict
    }

    /// Return the cached `BatchFeeds` for `batchSize`, allocating it
    /// lazily on first use. The three ND arrays are sized exactly for
    /// this batch size; the wrappers and the feeds dict are built
    /// once per size and kept for the trainer's lifetime (or until
    /// `resetNetwork()` clears the cache).
    private func feedsForBatch(_ batchSize: Int) -> BatchFeeds {
        if let existing = feedCache[batchSize] {
            return existing
        }
        let mtlDevice = network.metalDevice
        let dtype = ChessNetwork.dataType

        let boardDesc = MPSNDArrayDescriptor(
            dataType: dtype,
            shape: [
                NSNumber(value: batchSize),
                NSNumber(value: ChessNetwork.inputPlanes),
                NSNumber(value: ChessNetwork.boardSize),
                NSNumber(value: ChessNetwork.boardSize)
            ]
        )
        let boardND = MPSNDArray(device: mtlDevice, descriptor: boardDesc)
        let boardTD = MPSGraphTensorData(boardND)

        let moveDesc = MPSNDArrayDescriptor(
            dataType: .int32,
            shape: [NSNumber(value: batchSize)]
        )
        let moveND = MPSNDArray(device: mtlDevice, descriptor: moveDesc)
        let moveTD = MPSGraphTensorData(moveND)

        let zDesc = MPSNDArrayDescriptor(
            dataType: dtype,
            shape: [NSNumber(value: batchSize), 1]
        )
        let zND = MPSNDArray(device: mtlDevice, descriptor: zDesc)
        let zTD = MPSGraphTensorData(zND)

        // vBaseline ND array — same shape as z, one scalar per row.
        let vBaselineDesc = MPSNDArrayDescriptor(
            dataType: dtype,
            shape: [NSNumber(value: batchSize), 1]
        )
        let vBaselineND = MPSNDArray(device: mtlDevice, descriptor: vBaselineDesc)
        let vBaselineTD = MPSGraphTensorData(vBaselineND)

        let legalMaskDesc = MPSNDArrayDescriptor(
            dataType: dtype,
            shape: [NSNumber(value: batchSize), NSNumber(value: ChessNetwork.policySize)]
        )
        let legalMaskND = MPSNDArray(device: mtlDevice, descriptor: legalMaskDesc)
        let legalMaskTD = MPSGraphTensorData(legalMaskND)

        // Pre-build the feeds dictionary so `buildFeeds` can return it
        // unchanged on every subsequent call at this batch size. The
        // keys (graph placeholders) and values (tensor data wrappers)
        // are all stable for the lifetime of the trainer network;
        // `resetNetwork` clears `feedCache` so a new trainer network
        // rebuilds fresh entries against its own placeholders.
        let feedsDict: [MPSGraphTensor: MPSGraphTensorData] = [
            network.inputPlaceholder: boardTD,
            movePlayedPlaceholder: moveTD,
            zPlaceholder: zTD,
            vBaselinePlaceholder: vBaselineTD,
            legalMaskPlaceholder: legalMaskTD,
            lrPlaceholder: lrTensorData,
            entropyCoeffPlaceholder: entropyCoeffTensorData,
            weightDecayPlaceholder: weightDecayTensorData,
            gradClipMaxNormPlaceholder: gradClipMaxNormTensorData,
            policyScaleKPlaceholder: policyScaleKTensorData
        ]

        let feeds = BatchFeeds(
            boardND: boardND,
            boardTD: boardTD,
            moveND: moveND,
            moveTD: moveTD,
            zND: zND,
            zTD: zTD,
            vBaselineND: vBaselineND,
            vBaselineTD: vBaselineTD,
            legalMaskND: legalMaskND,
            legalMaskTD: legalMaskTD,
            feedsDict: feedsDict
        )
        feedCache[batchSize] = feeds
        return feeds
    }

    /// Run the forward + backward + SGD update graph with the given feeds
    /// and read the loss scalar back. The two public `trainStep` entry
    /// points share this so they produce identical timing breakdowns.
    private func runPreparedStep(
        feeds: [MPSGraphTensor: MPSGraphTensorData],
        prepMs: Double,
        totalStart: CFAbsoluteTime
    ) throws -> TrainStepTiming {
        // Wrap the graph.run + readback in an autoreleasepool so the
        // results dictionary and its MPSGraphTensorData values — which
        // are returned autoreleased by MPSGraph — drain each step
        // instead of piling up until the enclosing long-lived training
        // Task returns. Without this, multi-hour sessions accumulate
        // massive VM-range allocations (seen as ~420 GB virtual vs
        // ~5 GB resident) and the main thread spends progressively
        // more time in deferred Obj-C releases.
//        return try autoreleasepool {
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let results = network.graph.run(
            with: network.commandQueue,
            feeds: feeds,
            targetTensors: [
                totalLoss, policyLossTensor, valueLossTensor,
                policyEntropyTensor, policyNonNegCountTensor, policyNonNegIllegalCountTensor, gradGlobalNormTensor,
                valueMeanTensor, valueAbsMeanTensor, policyHeadWeightNormTensor,
                policyLogitAbsMaxTensor, playedMoveProbTensor,
                playedMoveProbPosAdvTensor, playedMoveProbNegAdvTensor,
                advantageMeanTensor, advantageStdTensor, advantageMinTensor, advantageMaxTensor,
                advantageFracPosTensor, advantageFracSmallTensor,
                advantageRawTensor,
                policyLossWinTensor, policyLossLossTensor
            ],
            targetOperations: assignOps
        )
        let gpuMs = (CFAbsoluteTimeGetCurrent() - gpuStart) * 1000

        let readbackStart = CFAbsoluteTimeGetCurrent()
        guard
            let totalData = results[totalLoss],
            let policyData = results[policyLossTensor],
            let valueData = results[valueLossTensor],
            let entropyData = results[policyEntropyTensor],
            let nonNegData = results[policyNonNegCountTensor],
            let nonNegIllegalData = results[policyNonNegIllegalCountTensor],
            let gradNormData = results[gradGlobalNormTensor],
            let valueMeanData = results[valueMeanTensor],
            let valueAbsMeanData = results[valueAbsMeanTensor],
            let policyHeadWNormData = results[policyHeadWeightNormTensor],
            let pLogitAbsMaxData = results[policyLogitAbsMaxTensor],
            let playedMoveProbData = results[playedMoveProbTensor],
            let playedMoveProbPosAdvData = results[playedMoveProbPosAdvTensor],
            let playedMoveProbNegAdvData = results[playedMoveProbNegAdvTensor],
            let advMeanData = results[advantageMeanTensor],
            let advStdData = results[advantageStdTensor],
            let advMinData = results[advantageMinTensor],
            let advMaxData = results[advantageMaxTensor],
            let advFracPosData = results[advantageFracPosTensor],
            let advFracSmallData = results[advantageFracSmallTensor],
            let advRawData = results[advantageRawTensor],
            let policyLossWinData = results[policyLossWinTensor],
            let policyLossLossData = results[policyLossLossTensor]
        else {
            throw ChessTrainerError.lossOutputMissing
        }
        ChessNetwork.readFloats(
            from: totalData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotTotal),
            count: 1
        )
        ChessNetwork.readFloats(
            from: policyData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPolicy),
            count: 1
        )
        ChessNetwork.readFloats(
            from: valueData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValue),
            count: 1
        )
        ChessNetwork.readFloats(
            from: entropyData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotEntropy),
            count: 1
        )
        ChessNetwork.readFloats(
            from: nonNegData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotNonNeg),
            count: 1
        )
        ChessNetwork.readFloats(
            from: nonNegIllegalData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotNonNegIllegal),
            count: 1
        )
        ChessNetwork.readFloats(
            from: gradNormData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotGradNorm),
            count: 1
        )
        ChessNetwork.readFloats(
            from: valueMeanData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValueMean),
            count: 1
        )
        ChessNetwork.readFloats(
            from: valueAbsMeanData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValueAbsMean),
            count: 1
        )
        ChessNetwork.readFloats(
            from: policyHeadWNormData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPolicyHeadWNorm),
            count: 1
        )
        ChessNetwork.readFloats(
            from: pLogitAbsMaxData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPLogitAbsMax),
            count: 1
        )
        ChessNetwork.readFloats(
            from: playedMoveProbData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPlayedMoveProb),
            count: 1
        )
        ChessNetwork.readFloats(
            from: playedMoveProbPosAdvData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPlayedMoveProbPosAdv),
            count: 1
        )
        ChessNetwork.readFloats(
            from: playedMoveProbNegAdvData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPlayedMoveProbNegAdv),
            count: 1
        )
        ChessNetwork.readFloats(
            from: advMeanData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvMean),
            count: 1
        )
        ChessNetwork.readFloats(
            from: advStdData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvStd),
            count: 1
        )
        ChessNetwork.readFloats(
            from: advMinData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvMin),
            count: 1
        )
        ChessNetwork.readFloats(
            from: advMaxData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvMax),
            count: 1
        )
        ChessNetwork.readFloats(
            from: advFracPosData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvFracPos),
            count: 1
        )
        ChessNetwork.readFloats(
            from: advFracSmallData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvFracSmall),
            count: 1
        )
        ChessNetwork.readFloats(
            from: policyLossWinData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPolicyLossWin),
            count: 1
        )
        ChessNetwork.readFloats(
            from: policyLossLossData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPolicyLossLoss),
            count: 1
        )
        // Raw per-position advantage — batch-sized vector. Read into
        // a fresh [Float] since the size depends on the runtime batch
        // and we don't want to resize the scratch every time.
        let advRawBatchSize: Int = advRawData.shape.reduce(1) { acc, dim in
            acc * Int(truncating: dim)
        }
        var advRawValues = [Float](repeating: 0, count: advRawBatchSize)
        if advRawBatchSize > 0 {
            advRawValues.withUnsafeMutableBufferPointer { buf in
                if let base = buf.baseAddress {
                    ChessNetwork.readFloats(from: advRawData, into: base, count: advRawBatchSize)
                }
            }
        }
        let totalBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotTotal]
        let policyBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPolicy]
        let valueBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotValue]
        let entropyBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotEntropy]
        let nonNegBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotNonNeg]
        let nonNegIllegalBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotNonNegIllegal]
        let gradNormBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotGradNorm]
        let valueMeanBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotValueMean]
        let valueAbsMeanBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotValueAbsMean]
        let policyHeadWNormBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPolicyHeadWNorm]
        let pLogitAbsMaxBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPLogitAbsMax]
        let playedMoveProbBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPlayedMoveProb]
        let playedMoveProbPosAdvBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPlayedMoveProbPosAdv]
        let playedMoveProbNegAdvBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPlayedMoveProbNegAdv]
        let advMeanBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotAdvMean]
        let advStdBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotAdvStd]
        let advMinBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotAdvMin]
        let advMaxBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotAdvMax]
        let advFracPosBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotAdvFracPos]
        let advFracSmallBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotAdvFracSmall]
        let policyLossWinBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPolicyLossWin]
        let policyLossLossBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPolicyLossLoss]
        let readbackMs = (CFAbsoluteTimeGetCurrent() - readbackStart) * 1000

        // Health check: any NaN/Inf in the headline loss or grad scalars means
        // this step's weight update has already corrupted the network (the
        // optimizer assignOps ran inside the same graph.run). We can't undo
        // that, but we can stop compounding the damage by halting right here.
        // The six checked values are the ones printed in [STATS] and used to
        // drive alarms; checking valueMean + entropy too catches broader
        // corruption signatures that leave the top-level losses oddly finite.
        if !totalBufValue.isFinite
            || !policyBufValue.isFinite
            || !valueBufValue.isFinite
            || !gradNormBufValue.isFinite
            || !valueMeanBufValue.isFinite
            || !entropyBufValue.isFinite {
            SessionLogger.shared.log(
                "[ALARM] loss non-finite: total=\(totalBufValue) policy=\(policyBufValue) value=\(valueBufValue) grad=\(gradNormBufValue) vMean=\(valueMeanBufValue) pEnt=\(entropyBufValue)"
            )
            throw ChessTrainerError.nonFiniteLoss(
                total: totalBufValue,
                policy: policyBufValue,
                value: valueBufValue,
                gradNorm: gradNormBufValue
            )
        }

        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000

        return TrainStepTiming(
            dataPrepMs: prepMs,
            gpuRunMs: gpuMs,
            readbackMs: readbackMs,
            totalMs: totalMs,
            loss: totalBufValue,
            policyLoss: policyBufValue,
            valueLoss: valueBufValue,
            policyEntropy: entropyBufValue,
            policyNonNegligibleCount: nonNegBufValue,
            policyNonNegligibleIllegalCount: nonNegIllegalBufValue,
            gradGlobalNorm: gradNormBufValue,
            valueMean: valueMeanBufValue,
            valueAbsMean: valueAbsMeanBufValue,
            // Defaults — the real-data path overwrites these at the
            // outer trainStep level once it has computed the fresh
            // baseline. The random-data sweep path leaves them nil.
            vBaselineDelta: nil,
            freshBaselineMs: nil,
            policyHeadWeightNorm: policyHeadWNormBufValue,
            policyLogitAbsMax: pLogitAbsMaxBufValue,
            playedMoveProb: playedMoveProbBufValue,
            playedMoveProbPosAdv: playedMoveProbPosAdvBufValue,
            playedMoveProbNegAdv: playedMoveProbNegAdvBufValue,
            advantageMean: advMeanBufValue,
            advantageStd: advStdBufValue,
            advantageMin: advMinBufValue,
            advantageMax: advMaxBufValue,
            advantageFracPositive: advFracPosBufValue,
            advantageFracSmall: advFracSmallBufValue,
            advantageRaw: advRawValues,
            policyLossWin: policyLossWinBufValue.isFinite ? policyLossWinBufValue : nil,
            policyLossLoss: policyLossLossBufValue.isFinite ? policyLossLossBufValue : nil
        )
//        }  // autoreleasepool
    }

    // MARK: - Batch Size Sweep

    /// Run a batch-size sweep. For each size in `sizes`:
    ///   1. Run one warmup step (which pays MPSGraph kernel-compile cost
    ///      the first time a new batch shape is seen — measured separately
    ///      so it doesn't pollute the throughput number).
    ///   2. Loop trainStep until `targetSecondsPerSize` elapsed (or a step
    ///      cap is hit, whichever comes first).
    ///   3. Compute average per-step time and positions/sec from the timed
    ///      window only.
    ///
    /// `progress` is called from the worker thread before each step so the
    /// UI can show "currently sweeping batch=X, step Y, elapsed Z". Pass
    /// `cancelled` from the UI to stop a sweep early — checked between steps.
    ///
    /// The trainer's network is **not** reset by this method. Callers that
    /// want fresh weights should call `resetNetwork()` first. Loss across a
    /// long sweep will drift downward as SGD overfits the random inputs;
    /// that's harmless for timing purposes.
    func runSweep(
        sizes: [Int],
        targetSecondsPerSize: Double,
        maxStepsPerSize: Int = 10_000,
        cancelled: @Sendable () -> Bool = { false },
        progress: @Sendable (Int, Int, Double) -> Void = { _, _, _ in },
        recordPeakSampleNow: @Sendable () -> Void = {},
        consumeRowPeak: @Sendable () -> UInt64 = { 0 },
        onRowCompleted: @Sendable (SweepRow) -> Void = { _ in }
    ) async throws -> [SweepRow] {
        var results: [SweepRow] = []
        results.reserveCapacity(sizes.count)

        // Read device caps once. They're fixed for the lifetime of the
        // process so it's safe to cache for the whole sweep.
        let device = network.metalDevice
        let workingSetCap = device.recommendedMaxWorkingSetSize
        let bufferCap = UInt64(device.maxBufferLength)
        // Skip threshold: 75% of the smaller of the two caps. The "lesser"
        // bit is deliberately conservative — on this hardware
        // maxBufferLength is well under recommendedMaxWorkingSetSize, so
        // capping the *total* estimate against the smaller of the two
        // gives a safety margin even though the comparison mixes
        // different things (total vs. single-buffer). Better to skip a
        // borderline batch than to take down the machine.
        let safetyFraction = 0.75
        let estimateThreshold = UInt64(Double(min(workingSetCap, bufferCap)) * safetyFraction)
        // Once we cross either threshold, every larger batch size will too —
        // latch this so we stop trying instead of crashing the machine.
        var skipFromHere = false
        // Empirically observed (batch, currentAllocatedSize) pairs from rows
        // we've already run. We fit a line through these to predict the
        // next batch's working-set footprint instead of guessing from the
        // network architecture — the architectural estimate was wildly
        // pessimistic compared to what MPSGraph actually allocates.
        var allocSamples: [(batch: Int, bytes: UInt64)] = []

        for batchSize in sizes {
            if cancelled() { break }

            // Largest single MTLBuffer we'll ask Metal for. Exact, not
            // estimated: the trainer literally uploads a [batch, 128, 8, 8]
            // float32 activation tensor and that's the biggest buffer in
            // the graph (beats the [batch, policySize] policy tensors and
            // the [batch, inputPlanes, 8, 8] input).
            let largestBufferBytes = Self.largestBufferBytes(forBatchSize: batchSize)
            // Working-set prediction comes from a least-squares fit over
            // the rows we've already run. Returns nil before we have any
            // data to fit, in which case we don't skip on this criterion.
            let predictedBytes = Self.predictAllocatedBytes(
                forBatchSize: batchSize,
                from: allocSamples
            )

            let exceedsBuffer = largestBufferBytes > bufferCap
            let exceedsWorkingSet: Bool
            if let predictedBytes {
                exceedsWorkingSet = predictedBytes > estimateThreshold
            } else {
                exceedsWorkingSet = false
            }
            if exceedsWorkingSet || exceedsBuffer {
                skipFromHere = true
            }

            if skipFromHere {
                let skipped = SkippedRow(
                    batchSize: batchSize,
                    estimatedBytes: predictedBytes ?? 0,
                    largestBufferBytes: largestBufferBytes,
                    exceededWorkingSet: exceedsWorkingSet,
                    exceededBufferLength: exceedsBuffer
                )
                let row = SweepRow.skipped(skipped)
                results.append(row)
                onRowCompleted(row)
                continue
            }

            // Drop a peak sample right before warmup so even rows that
            // finish between heartbeats get a baseline reading.
            recordPeakSampleNow()

            // Warmup: first call at this batch size pays whatever per-shape
            // compile cost MPSGraph charges. Time it but don't count it
            // toward the throughput number.
            let warmup = try await trainStep(batchSize: batchSize)
            if cancelled() { break }
            recordPeakSampleNow()

            var timedSteps = 0
            var totalStepMs: Double = 0
            var totalGpuMs: Double = 0
            var lastLoss: Float = warmup.loss
            let runStart = CFAbsoluteTimeGetCurrent()

            while !cancelled() && timedSteps < maxStepsPerSize {
                let elapsed = CFAbsoluteTimeGetCurrent() - runStart
                if elapsed >= targetSecondsPerSize { break }
                progress(batchSize, timedSteps, elapsed)

                let timing = try await trainStep(batchSize: batchSize)
                timedSteps += 1
                totalStepMs += timing.totalMs
                totalGpuMs += timing.gpuRunMs
                lastLoss = timing.loss
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - runStart
            let avgStepMs = timedSteps > 0 ? totalStepMs / Double(timedSteps) : 0
            let avgGpuMs = timedSteps > 0 ? totalGpuMs / Double(timedSteps) : 0
            let positions = timedSteps * batchSize
            let positionsPerSec = elapsed > 0 ? Double(positions) / elapsed : 0

            // Final sample before we read the peak — captures whatever
            // resident memory the just-finished steps left behind.
            recordPeakSampleNow()
            let peakResident = consumeRowPeak()
            // Feed the empirical linear fit that predicts the next row's
            // footprint. The fit only sees what we actually observed, no
            // architectural fudge factors.
            allocSamples.append((batch: batchSize, bytes: peakResident))

            let row = SweepRow.completed(
                SweepResult(
                    batchSize: batchSize,
                    warmupMs: warmup.totalMs,
                    steps: timedSteps,
                    elapsedSec: elapsed,
                    avgStepMs: avgStepMs,
                    avgGpuMs: avgGpuMs,
                    positionsPerSec: positionsPerSec,
                    lastLoss: lastLoss,
                    peakResidentBytes: peakResident
                )
            )
            results.append(row)
            // Fire after the row is complete so the UI can show partial
            // results as the sweep advances rather than waiting for the
            // whole sweep to finish.
            onRowCompleted(row)
        }

        // The sweep ran training-mode forward passes against random
        // (non-chess) inputs, so the BN running-stat EMA variables now
        // reflect the per-channel statistics of noise. Leaving them in
        // that state would silently miscalibrate any subsequent
        // `loadWeights` into an inference network. Reset back to fresh
        // random weights + factory BN stats (zero mean, unit var) so the
        // trainer is in a clean state for whatever runs next.
        try await self.resetNetwork()

        return results
    }

    // MARK: - Footprint Helpers

    /// Exact size of the largest single MTLBuffer the trainer requests at
    /// this batch size — one [batch, 128, 8, 8] float32 activation tensor.
    /// That's larger than the [batch, policySize] policy tensors and the
    /// [batch, inputPlanes, 8, 8] input, so it's the buffer that would first hit
    /// `maxBufferLength`. This is an architectural fact, not a guess.
    static func largestBufferBytes(forBatchSize batchSize: Int) -> UInt64 {
        let floatBytes = MemoryLayout<Float>.size
        let spatial = ChessNetwork.boardSize * ChessNetwork.boardSize
        let channels = ChessNetwork.channels
        return UInt64(channels * spatial * floatBytes) * UInt64(batchSize)
    }

    /// Predict `currentAllocatedSize` for `batchSize` from the
    /// (batch, allocated) pairs already observed during this sweep.
    /// Returns nil before we have any samples to fit.
    ///
    /// With a single sample we draw a line from the origin through it
    /// (slope-only). With two or more samples we use ordinary least
    /// squares on (batch, bytes), which automatically captures both the
    /// per-sample slope and any fixed overhead. No fudge factors — what
    /// MPSGraph actually allocated is what we extrapolate from.
    static func predictAllocatedBytes(
        forBatchSize batchSize: Int,
        from samples: [(batch: Int, bytes: UInt64)]
    ) -> UInt64? {
        if samples.isEmpty { return nil }

        let target = Double(batchSize)
        if samples.count == 1 {
            let only = samples[0]
            let perSample = Double(only.bytes) / Double(only.batch)
            return UInt64(max(0, perSample * target))
        }

        let n = Double(samples.count)
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumXX = 0.0
        for s in samples {
            let x = Double(s.batch)
            let y = Double(s.bytes)
            sumX += x
            sumY += y
            sumXY += x * y
            sumXX += x * x
        }
        let denom = n * sumXX - sumX * sumX
        // denom is zero only if all sample batch sizes are identical —
        // which can't happen here since the sweep monotonically increases
        // batch size — but fall back to the slope-from-origin rule rather
        // than dividing by zero.
        guard denom != 0 else {
            let perSample = sumY / sumX
            return UInt64(max(0, perSample * target))
        }
        let slope = (n * sumXY - sumX * sumY) / denom
        let intercept = (sumY - slope * sumX) / n
        let predicted = slope * target + intercept
        return UInt64(max(0, predicted))
    }

    /// Read the process-wide `phys_footprint` from `task_info`. On Apple
    /// Silicon's unified memory architecture this captures everything the
    /// process is holding onto — CPU buffers and Metal-managed GPU memory
    /// alike — so it's a strictly better high-water-mark proxy than
    /// `MTLDevice.currentAllocatedSize`, which only sees memory that's
    /// still live at the moment you query it. Returns 0 on failure rather
    /// than throwing — the caller is sampling on a hot path and a missed
    /// reading is recoverable, while throwing would force exception
    /// handling around every UI tick.
    static func currentPhysFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    intPtr,
                    &count
                )
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }

    /// Sample cumulative CPU and GPU time for the current process.
    /// Two kernel reads: `proc_pid_rusage` for CPU time (documented
    /// to return nanoseconds in `ri_user_time` / `ri_system_time`),
    /// and `task_info(TASK_POWER_INFO_V2)` for
    /// `gpu_energy.task_gpu_utilisation` (also nanoseconds, summed
    /// across all GPU engines). Returns `nil` if either call fails —
    /// the caller polls out-of-band, so a dropped sample just skips
    /// one update tick.
    /// Cached Mach timebase for converting `mach_absolute_time`
    /// ticks to nanoseconds. Constant for the lifetime of the
    /// process, so one init + atomic read from then on. The
    /// `mach_timebase_info` API is documented to always succeed
    /// on Apple hardware and return non-zero numer/denom (typical
    /// values: Intel 1/1, Apple Silicon 125/3), but we still
    /// precondition both fields non-zero — a zero denom would
    /// produce a misleading integer-division-by-zero trap deeper
    /// in `sampleCurrentProcessUsage` rather than a clear failure.
    private static let machTimebase: mach_timebase_info_data_t = {
        var t = mach_timebase_info_data_t()
        let rc = mach_timebase_info(&t)
        precondition(
            rc == KERN_SUCCESS && t.numer > 0 && t.denom > 0,
            "mach_timebase_info failed or returned zero numer/denom; "
                + "rc=\(rc), numer=\(t.numer), denom=\(t.denom)"
        )
        return t
    }()

    static func sampleCurrentProcessUsage() -> ProcessUsageSample? {
        // CPU time: use TASK_ABSOLUTETIME_INFO, which exposes
        // `total_user` + `total_system` summed across BOTH live
        // and terminated threads. The previous TASK_THREAD_TIMES_INFO
        // flavor only reported LIVE-thread time, so every time a
        // self-play worker, save task, or arena game-runner thread
        // exited its accumulated time disappeared from the counter.
        // If the drop between two polls exceeded the live threads'
        // newly-accumulated time, `sample.cpuNs < prev.cpuNs` and
        // the caller clamped `cpuDelta` to 0 → CPU% blipped to 0 %
        // until live threads accumulated enough time to cover the
        // loss. ABSOLUTETIME_INFO is monotonic and fixes that.
        //
        // Values are in `mach_absolute_time` ticks; convert to
        // nanoseconds via the cached timebase (numer / denom).
        var abstime = task_absolutetime_info_data_t()
        var abstimeCount = mach_msg_type_number_t(
            MemoryLayout<task_absolutetime_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let abstimeRC = withUnsafeMutablePointer(to: &abstime) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(abstimeCount)) { intPtr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_ABSOLUTETIME_INFO),
                    intPtr,
                    &abstimeCount
                )
            }
        }
        guard abstimeRC == KERN_SUCCESS else { return nil }

        let tb = machTimebase
        let totalTicks = abstime.total_user &+ abstime.total_system
        // ticks * numer / denom → nanoseconds. Intermediate product
        // fits in UInt64 comfortably: numer/denom on Apple Silicon
        // is 125/3, so total_ticks * 125 overflows at ~1.47e17 ns =
        // ~4.6 years of continuous runtime — not a real concern.
        let cpuNs = totalTicks &* UInt64(tb.numer) / UInt64(tb.denom)

        // GPU time: task_info(TASK_POWER_INFO_V2) → gpu_energy.
        var power = task_power_info_v2_data_t()
        var powerCount = mach_msg_type_number_t(
            MemoryLayout<task_power_info_v2_data_t>.size / MemoryLayout<natural_t>.size
        )
        let powerRC = withUnsafeMutablePointer(to: &power) { infoPtr -> kern_return_t in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(powerCount)) { intPtr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_POWER_INFO_V2),
                    intPtr,
                    &powerCount
                )
            }
        }
        guard powerRC == KERN_SUCCESS else { return nil }

        return ProcessUsageSample(
            timestamp: Date(),
            cpuNs: cpuNs,
            gpuNs: power.gpu_energy.task_gpu_utilisation
        )
    }

    /// Snapshot the device's memory caps right now. Read once at the start
    /// of a sweep so the UI header has a stable reference point.
    func deviceMemoryCaps() -> DeviceMemoryCaps {
        let device = network.metalDevice
        return DeviceMemoryCaps(
            recommendedMaxWorkingSet: device.recommendedMaxWorkingSetSize,
            currentAllocated: UInt64(device.currentAllocatedSize),
            maxBufferLength: UInt64(device.maxBufferLength)
        )
    }

    // MARK: - Random Fill

    /// Fill a float buffer with pseudo-random values in [0, 1) using a fast
    /// inline LCG. Avoids the cost of arc4random_buf + conversion for the
    /// ~1.15M floats per batch we need. Quality doesn't matter — we only need
    /// non-zero, non-uniform values to exercise the same compute paths real
    /// data would.
    private static func fillRandomFloats(_ buffer: inout [Float]) {
        var rng: UInt64 = UInt64.random(in: 0...UInt64.max) | 1
        let scale: Float = Float(1.0 / 4294967296.0)
        buffer.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for i in 0..<buf.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let high = UInt32(truncatingIfNeeded: rng >> 32)
                base[i] = Float(high) * scale
            }
        }
    }
}
