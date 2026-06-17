import Accelerate
import Darwin
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import os

// MARK: - Errors

enum ChessTrainerError: LocalizedError {
    case lossOutputMissing
    case gradientMissing(String)
    case nonFiniteLoss(total: Float, policy: Float, value: Float, gradNorm: Float)
    case trainerWeightCountMismatch(expected: String, got: Int)
    case velocityReadbackMissing(String)
    case velocityLoadGraphFailed(String)
    /// The tower is deep enough that building its gradient graph would risk
    /// overflowing even the enlarged graph-build stack. Raised *before* the
    /// build runs so it surfaces as a catchable error instead of a SIGBUS.
    case towerTooDeepToBuild(numBlocks: Int, estimatedKB: Int, limitKB: Int)
    /// The GPU command buffer for a training step finished in a non-`completed`
    /// state (out-of-memory, timeout, kernel fault). Surfaced instead of reading
    /// back garbage result tensors and training on them.
    case gpuCommandFailed(stage: String, status: MTLCommandBufferStatus, error: String?)

    var errorDescription: String? {
        switch self {
        case .lossOutputMissing:
            return "Training step ran but loss tensor was not in the result map"
        case .gradientMissing(let name):
            return "Gradient missing for variable: \(name)"
        case .nonFiniteLoss(let total, let policy, let value, let gradNorm):
            return "Non-finite loss detected: total=\(total) policy=\(policy) value=\(value) gradNorm=\(gradNorm). Weights from this step are likely poisoned; training halted."
        case .trainerWeightCountMismatch(let expected, let got):
            return "Trainer weight count mismatch: expected \(expected), got \(got)"
        case .velocityReadbackMissing(let name):
            return "Velocity tensor missing from graph.run results: \(name)"
        case .velocityLoadGraphFailed(let name):
            return "Velocity load graph.run returned empty/missing result for tensor: \(name)"
        case .towerTooDeepToBuild(let numBlocks, let estimatedKB, let limitKB):
            return "Tower too deep to build gradients: \(numBlocks) residual blocks need ~\(estimatedKB) KB of build stack, over the \(limitKB) KB budget. Depth is the limit, not width or parameter count — reduce the block count."
        case .gpuCommandFailed(let stage, let status, let error):
            return "GPU command buffer failed during \(stage): status=\(status), error=\(error ?? "none"). Results from this step are unreliable; training halted."
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
    /// Wall time the public `trainStep(...)` call sat between
    /// "caller invoked it" and "the work began running on
    /// `executionQueue`". Captures backlog on the trainer's serial
    /// dispatch queue. Healthy sessions sit near 0; a growing value
    /// means something else is dispatching to the same queue and the
    /// trainer is queueing behind it. Diagnostic only.
    let queueWaitMs: Double
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
    /// Value-only component of the loss. Categorical cross-entropy of the
    /// W/D/L head against the smoothed one-hot target `oneHot(1 − z)`,
    /// per position then mean over the batch — so bounded below by 0 and
    /// roughly in [0, ln 3 ≈ 1.10] at convergence (more transiently while
    /// the head is still learning the class boundaries). NOTE: this is no
    /// longer the old MSE scale (~0.1–0.4); a value-loss curve compared
    /// across the scalar→WDL switch is not apples-to-apples.
    let valueLoss: Float
    /// Mean Shannon entropy (in nats) of the trainee's policy softmax over
    /// this batch. Diagnostic only — not part of `loss`. Range is
    /// [0, log(ChessNetwork.policySize)] nats.
    /// Random init sits near the ceiling; a collapsed policy heads toward 0.
    /// Watch for monotonic drift to either extreme — that's the signature
    /// of policy collapse or a stuck-at-uniform learning failure.
    let policyEntropy: Float
    let illegalMassPenalty: Float
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
    /// Mean of the derived scalar value `p_win − p_loss` across the
    /// batch. No tanh — the scalar is a difference of two softmax
    /// probabilities, naturally in [-1, +1]. A healthy batch of
    /// early-training self-play positions should sit near 0 (most
    /// positions ≈ draw → p_win ≈ p_loss). Drifting away from 0 means
    /// the head is learning a systematic bias toward win or loss.
    let valueMean: Float
    /// Mean of |p_win − p_loss| across the batch. Together with
    /// `valueMean` and the W/D/L probability means below, this is the
    /// cheapest "is the value head doing anything?" probe: ≈ 0
    /// everywhere = the head is calling every position a draw (the
    /// failure WDL is meant to break out of), ≈ 1 = the head is
    /// confidently classifying win-or-loss everywhere. Range [0, 1].
    let valueAbsMean: Float
    /// Batch-mean of the value head's softmax `p_win`. Pairs with
    /// `valueProbDraw` / `valueProbLoss` (they sum to 1). `valueProbDraw`
    /// → 1.0 is the new representation's "everything is a draw" — watch
    /// it the way the old code watched `vAbs → 0`.
    let valueProbWin: Float
    let valueProbDraw: Float
    let valueProbLoss: Float

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
    /// `1/policySize` even when training is perfectly healthy,
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
    /// Readback cost is a single `[batch, 1]` tensor — trivial against
    /// the existing per-step readback budget. Nil on the random-data
    /// sweep path (percentile view is meaningless there).
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

    /// Global L2 norm of the optimizer's velocity buffer ‖v‖ AFTER
    /// the SGD update was applied this step. With Polyak momentum,
    /// the steady-state velocity-norm under independent gradients
    /// approaches roughly ‖g‖/√(1−μ²), so reading this against
    /// `gradGlobalNorm` and `momentumCoeff` gives a direct view of
    /// how much momentum is amplifying per-step updates. Watch for
    /// monotonic growth without a corresponding ‖g‖ rise — that's a
    /// runaway-velocity event, typically caused by setting μ too
    /// high relative to LR for the current loss landscape.
    let velocityNorm: Float

    /// True when this step's diagnostic outputs (policy entropy, advantage
    /// stats, played-move / value-head probabilities, weight & velocity norms,
    /// non-negligible-cell counts, outcome-split policy losses) were actually
    /// computed and read back. They are gated to stats steps — on a non-stats
    /// step the trainer omits those extra graph reductions from the target
    /// list so MPSGraph never encodes them, and the corresponding fields above
    /// are `.nan` / `nil`. `recordStep` must not fold those placeholders into
    /// the rolling means. The loss components, grad-norm, and all timing fields
    /// are valid on every step. See GPU_UTILIZATION_PLAN.md (Phase 1).
    let hasDiagnostics: Bool
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
struct MetalDeviceMemoryLimits: Sendable {
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
/// hop per step. The SwiftUI `snapshotTimer` polls `snapshot()` on the
/// heartbeat and mirrors the current values into `@State`, which is what actually
/// triggers view redraws. This decouples view-update frequency from
/// training-step rate: a 20 ms/step training loop used to fire 50
/// `MainActor.run` hops per second, now it fires zero.
///
/// The rolling-loss windows live here rather than on the view so the
/// worker can maintain them without any main-actor round-trips. Policy
/// and value losses are tracked in separate windows so the UI can show
/// which head is oscillating — the value term is now a categorical
/// cross-entropy over the W/D/L head (roughly `[0, ln 3]` at
/// convergence, transiently larger while the head is learning the
/// class boundaries), so a sustained 5× swing there means genuinely
/// unstable training, while a noisy policy term alone is usually just
/// metric noise from outcome-weighted CE.
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
        let rollingIllegalMassPenalty: Double?
        let rollingPolicyNonNegCount: Double?
        let rollingPolicyNonNegIllegalCount: Double?
        let rollingGradGlobalNorm: Double?
        let rollingValueMean: Double?
        let rollingValueAbsMean: Double?
        /// Rolling-window means of the value head's softmax W/D/L
        /// probabilities (`TrainStepTiming.valueProbWin/Draw/Loss`).
        /// They sum to ≈1. `rollingValueProbDraw → 1.0` is the new
        /// representation's "everything is a draw" collapse — surfaced
        /// on `[STATS]` as `pW=/pD=/pL=`.
        let rollingValueProbWin: Double?
        let rollingValueProbDraw: Double?
        let rollingValueProbLoss: Double?
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
        /// Rolling-window mean of `TrainStepTiming.velocityNorm` —
        /// post-update velocity-buffer L2 norm over the recent window.
        /// Surfaced on the `[STATS]` line as `vNorm=…` next to `gNorm=`
        /// so velocity-vs-gradient magnitude can be compared at a
        /// glance when raising μ. Nil before any step has executed.
        let rollingVelocityNorm: Double?
        /// Rolling-window means of `TrainStepTiming` timing fields,
        /// over the last `rollingTimingWindow` steps. These intentionally
        /// shadow `TrainingRunStats.avgGpuMs` (which is cumulative across
        /// the whole session): the rolling values will track per-step
        /// degradation in real time, while the cumulative mean smears
        /// any later slowdown into the early-session average. Nil
        /// before the first step.
        let recentDataPrepMs: Double?
        let recentGpuRunMs: Double?
        let recentReadbackMs: Double?
        let recentQueueWaitMs: Double?
        let recentStepMs: Double?
        /// Number of training-step samples currently in the timing
        /// rolling window — denominator the reader should mentally
        /// divide by when interpreting the means above.
        let recentTimingSamples: Int
        let error: String?
    }

    private let lock = OSAllocatedUnfairLock()
    private var _stats = TrainingRunStats()
    private var _lastTiming: TrainStepTiming?
    private var _policyLossWindow: RollingDoubleWindow
    private var _valueLossWindow: RollingDoubleWindow
    private var _policyEntropyWindow: RollingDoubleWindow
    private var _illegalMassPenaltyWindow: RollingDoubleWindow
    private var _policyNonNegWindow: RollingDoubleWindow
    private var _policyNonNegIllegalWindow: RollingDoubleWindow
    private var _gradNormWindow: RollingDoubleWindow
    private var _valueMeanWindow: RollingDoubleWindow
    private var _valueAbsMeanWindow: RollingDoubleWindow
    private var _valueProbWinWindow: RollingDoubleWindow
    private var _valueProbDrawWindow: RollingDoubleWindow
    private var _valueProbLossWindow: RollingDoubleWindow
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
    private var _velocityNormWindow: RollingDoubleWindow
    /// Rolling per-step timing windows. Sized independently of
    /// `rollingWindow` because the right horizon for "is the trainer
    /// slowing down?" is hundreds of steps, not the much smaller window
    /// the diagnostic-loss windows use. See `Self.rollingTimingWindow`.
    private var _dataPrepMsWindow: RollingDoubleWindow
    private var _gpuRunMsWindow: RollingDoubleWindow
    private var _readbackMsWindow: RollingDoubleWindow
    private var _queueWaitMsWindow: RollingDoubleWindow
    private var _stepMsWindow: RollingDoubleWindow
    /// Ring of raw per-position advantage values across the rolling
    /// window of recent steps. Capped at `advRawRingMaxCapacity`
    /// floats (see the constant for the full rationale). `snapshot()`
    /// sorts the live set for percentile extraction, and that sort
    /// runs on main via the UI heartbeat's `lock.withLock` — a larger
    /// ring was blocking main for ~150 ms per snapshot, every heartbeat
    /// tick, starving `fireCandidateProbeIfNeeded`'s MainActor hop and
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
    /// that's eyeballed in logs. Sized so that at the configured
    /// `TrainingBatchSize` default the ring holds
    /// `advRawRingMaxCapacity / batchSize` full batches' worth of raw
    /// advantages; at smaller batches the ring is effectively the
    /// `rollingWindow` * batchSize product anyway.
    private static let advRawRingMaxCapacity: Int = 32_768

    /// Rolling-window length for per-step timing means
    /// (`recentDataPrepMs`, `recentGpuRunMs`, etc.). The
    /// `rollingTimingWindow`-step window at typical Play-and-Train
    /// throughput (~30 steps/min on the M-series dev machine) is
    /// ~17 minutes of history — long enough to smooth
    /// out per-step jitter, short enough that a slowdown beginning at
    /// the most recent arena boundary is visible within one [STATS]
    /// emit (60 s) rather than being washed out by hours of fast
    /// early-session steps.
    static let rollingTimingWindow: Int = 512

    init(rollingWindow: Int) {
        precondition(rollingWindow > 0, "Rolling window must be positive")
        self.rollingWindow = rollingWindow
        self._policyLossWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueLossWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyEntropyWindow = RollingDoubleWindow(limit: rollingWindow)
        self._illegalMassPenaltyWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyNonNegWindow = RollingDoubleWindow(limit: rollingWindow)
        self._policyNonNegIllegalWindow = RollingDoubleWindow(limit: rollingWindow)
        self._gradNormWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueMeanWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueAbsMeanWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueProbWinWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueProbDrawWindow = RollingDoubleWindow(limit: rollingWindow)
        self._valueProbLossWindow = RollingDoubleWindow(limit: rollingWindow)
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
        self._velocityNormWindow = RollingDoubleWindow(limit: rollingWindow)
        self._dataPrepMsWindow = RollingDoubleWindow(limit: Self.rollingTimingWindow)
        self._gpuRunMsWindow = RollingDoubleWindow(limit: Self.rollingTimingWindow)
        self._readbackMsWindow = RollingDoubleWindow(limit: Self.rollingTimingWindow)
        self._queueWaitMsWindow = RollingDoubleWindow(limit: Self.rollingTimingWindow)
        self._stepMsWindow = RollingDoubleWindow(limit: Self.rollingTimingWindow)
    }

    /// Seed the stats with values from a resumed session so the
    /// step counter and other totals don't restart from zero.
    func seed(_ stats: TrainingRunStats) {
        lock.withLock {
            self._stats = stats
        }
    }

    /// Rewind or reseed the cumulative step counter without touching
    /// timing/error fields. Used after arena promotion when the
    /// trainer weights are restored to the arena-start snapshot, so
    /// future checkpoint state stays aligned with
    /// `ChessTrainer.completedTrainSteps`.
    func setStepCount(_ steps: Int) {
        lock.withLock {
            self._stats.steps = max(0, steps)
        }
    }

    /// Record one completed training step. Called from the background
    /// training task. The lock acquisition is sub-microsecond and the
    /// rolling-window bookkeeping runs synchronously under it; the UI
    /// heartbeat's `snapshot()` read takes the same lock briefly.
    func recordStep(_ timing: TrainStepTiming) {
        lock.withLock {
            self._stats.record(timing)
            // Loss components, grad-norm, and the per-step timing windows are
            // valid on EVERY step — they're weight-update-path outputs (or pure
            // timing), always read back. Append them unconditionally.
            self._policyLossWindow.append(Double(timing.policyLoss))
            self._valueLossWindow.append(Double(timing.valueLoss))
            self._illegalMassPenaltyWindow.append(Double(timing.illegalMassPenalty))
            self._gradNormWindow.append(Double(timing.gradGlobalNorm))
            self._dataPrepMsWindow.append(timing.dataPrepMs)
            self._gpuRunMsWindow.append(timing.gpuRunMs)
            self._readbackMsWindow.append(timing.readbackMs)
            self._queueWaitMsWindow.append(timing.queueWaitMs)
            self._stepMsWindow.append(timing.totalMs)

            // The diagnostic reductions are computed only on stats steps
            // (`hasDiagnostics`); non-stats steps carry `.nan` / `nil`
            // placeholders that must not enter the rolling means. `_lastTiming`
            // — the source for the "Last Step" UI readout — is also updated only
            // here so it never surfaces a NaN entropy/advantage.
            guard timing.hasDiagnostics else { return }
            self._lastTiming = timing
            self._policyEntropyWindow.append(Double(timing.policyEntropy))
            self._policyNonNegWindow.append(Double(timing.policyNonNegligibleCount))
            self._policyNonNegIllegalWindow.append(Double(timing.policyNonNegligibleIllegalCount))
            self._valueMeanWindow.append(Double(timing.valueMean))
            self._valueAbsMeanWindow.append(Double(timing.valueAbsMean))
            self._valueProbWinWindow.append(Double(timing.valueProbWin))
            self._valueProbDrawWindow.append(Double(timing.valueProbDraw))
            self._valueProbLossWindow.append(Double(timing.valueProbLoss))
            self._policyHeadWeightNormWindow.append(Double(timing.policyHeadWeightNorm))
            self._policyLogitAbsMaxWindow.append(Double(timing.policyLogitAbsMax))
            self._playedMoveProbWindow.append(Double(timing.playedMoveProb))
            // NaN protection: the conditional means are NaN when the
            // batch has zero positions on one side of the sign — skip
            // those entries so the rolling mean stays well-defined.
            // The parallel `…SkipWindow` ring is appended on every stats
            // step (1.0 on skip, 0.0 on contribution) so consumers can
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
            if timing.velocityNorm.isFinite {
                self._velocityNormWindow.append(Double(timing.velocityNorm))
            }
            if let raw = timing.advantageRaw, !raw.isEmpty {
                self.pushAdvRaw(raw)
            }
        }
    }

    /// Push a batch's raw advantage values into the percentile ring.
    /// Called from within `recordStep`'s `lock.withLock` closure, so
    /// no additional serialization is needed. Capacity is grown on
    /// first use (and whenever `rollingWindow * batchSize` changes)
    /// to avoid reallocating in steady state.
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
        lock.withLock {
            if self._error == nil { self._error = message }
        }
    }

    /// Clear the rolling diagnostic windows while keeping cumulative
    /// training stats and the most recent error intact. Used after a
    /// promotion so post-promotion alarms and charts reflect the new
    /// aligned trainer/champion regime instead of inheriting the
    /// pre-promotion averages.
    func resetRollingWindows() {
        lock.withLock {
            self._lastTiming = nil
            self._policyLossWindow.removeAll()
            self._valueLossWindow.removeAll()
            self._policyEntropyWindow.removeAll()
            self._illegalMassPenaltyWindow.removeAll()
            self._policyNonNegWindow.removeAll()
            self._policyNonNegIllegalWindow.removeAll()
            self._gradNormWindow.removeAll()
            self._valueMeanWindow.removeAll()
            self._valueAbsMeanWindow.removeAll()
            self._valueProbWinWindow.removeAll()
            self._valueProbDrawWindow.removeAll()
            self._valueProbLossWindow.removeAll()
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
            self._velocityNormWindow.removeAll()
            self._dataPrepMsWindow.removeAll()
            self._gpuRunMsWindow.removeAll()
            self._readbackMsWindow.removeAll()
            self._queueWaitMsWindow.removeAll()
            self._stepMsWindow.removeAll()
            self._advRawRingHead = 0
            self._advRawRingFilled = 0
            // Keep _advRawRing capacity allocated — next push reuses it.
        }
    }

    /// Gets a snapshot of live stats
    func snapshot() async -> Snapshot {
        await withCheckedContinuation { (cont: CheckedContinuation<Snapshot, Never>) in
            DispatchQueue.global(qos: .default).async {
                cont.resume(returning: self.snapshot())
            }
        }
    }

    /// Snapshot all fields atomically for the UI poller.
    func snapshot() -> Snapshot {
        lock.withLock {
            let rollingPolicy = _policyLossWindow.mean
            let rollingValue = _valueLossWindow.mean
            let rollingEntropy = _policyEntropyWindow.mean
            let rollingIllegalPenalty = _illegalMassPenaltyWindow.mean
            let rollingNonNeg = _policyNonNegWindow.mean
            let rollingNonNegIllegal = _policyNonNegIllegalWindow.mean
            let rollingGradNorm = _gradNormWindow.mean
            let rollingVMean = _valueMeanWindow.mean
            let rollingVAbs = _valueAbsMeanWindow.mean
            let rollingVProbWin = _valueProbWinWindow.mean
            let rollingVProbDraw = _valueProbDrawWindow.mean
            let rollingVProbLoss = _valueProbLossWindow.mean
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
            let rollingVNorm = _velocityNormWindow.mean
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
                rollingIllegalMassPenalty: rollingIllegalPenalty,
                rollingPolicyNonNegCount: rollingNonNeg,
                rollingPolicyNonNegIllegalCount: rollingNonNegIllegal,
                rollingGradGlobalNorm: rollingGradNorm,
                rollingValueMean: rollingVMean,
                rollingValueAbsMean: rollingVAbs,
                rollingValueProbWin: rollingVProbWin,
                rollingValueProbDraw: rollingVProbDraw,
                rollingValueProbLoss: rollingVProbLoss,
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
                rollingVelocityNorm: rollingVNorm,
                recentDataPrepMs: _dataPrepMsWindow.mean,
                recentGpuRunMs: _gpuRunMsWindow.mean,
                recentReadbackMs: _readbackMsWindow.mean,
                recentQueueWaitMs: _queueWaitMsWindow.mean,
                recentStepMs: _stepMsWindow.mean,
                recentTimingSamples: _stepMsWindow.size,
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

    /// Weight-decay coefficient applied per training step. The optimizer here
    /// is SGD with Polyak momentum (`momentumCoeff` μ, 0 by default) plus
    /// decoupled (AdamW-style) weight decay: the decay is applied to the
    /// weights directly rather than folded into the gradient, so μ and
    /// `weightDecayC` tune independently (raising μ does not amplify decay).
    /// With μ=0 the velocity term vanishes and the per-step update for
    /// decay-eligible variables is
    /// `weight_new = (1 − lr·weightDecayC) · weight − lr · clipped_grad`.
    /// Decay is applied only to conv and FC weight matrices; BN gamma/beta
    /// and FC biases are excluded, matching the standard PyTorch / AdamW
    /// recipe for which params to decay.
    ///
    /// The actual value applied by the graph is read from
    /// `weightDecayC` and fed as a per-step scalar so the user can
    /// tune it live.
    static let weightDecayCDefault: Float = 1e-4

    /// Default global L2-norm gradient clipping threshold. If the L2
    /// norm of the concatenated gradient vector over every trainable
    /// variable exceeds this value, every gradient is scaled by
    /// `maxNorm / globalNorm` so the effective step is capped. 30.0 is
    /// a conservative value that sits well above steady-state norms
    /// under healthy training but cuts off the single-step blowups.
    /// Under heavy policy-collapse
    /// pressure the natural gradient norm can vastly exceed this,
    /// nullifying effective learning rate — live-tunable to let the
    /// user widen the valve when that happens.
    static let gradClipMaxNormDefault: Float = 30.0

    /// Fallback cadence (in training steps) for computing the graph diagnostic
    /// reductions when `batchStatsInterval` is 0 ([BATCH-STATS] disabled). The
    /// diagnostics feed the [STATS] line and the entropy / draw-collapse alarms
    /// — not just [BATCH-STATS] — so they must keep flowing regardless of the
    /// batch-stats setting. See GPU_UTILIZATION_PLAN.md (Phase 1).
    static let diagnosticsFallbackInterval = 10

    /// Default per-head loss coefficients in `total_loss =
    /// valueLossWeight · valueLoss + policyLossWeight · policyLoss
    /// − entropyCoeff · policyEntropy
    /// + illegalMassWeight · illegalMassPenalty`. AlphaZero canonical is
    /// 1.0 / 1.0 (both heads weighted equally); Lc0 / KataGo expose
    /// these as `policy_loss_weight` / `value_loss_weight` and tune
    /// the ratio per training stage.
    ///
    /// Note: this engine's policy loss is REINFORCE on the played
    /// move over a `policySize`-way softmax, so its raw gradient is
    /// naturally weaker than the value head's (z−v)². If the policy
    /// head is starving for signal early, raise `policyLossWeight`
    /// (or lower `valueLossWeight`) — the prior `K=5` default did
    /// the former implicitly.
    static let policyLossWeightDefault: Float = 1.0
    static let valueLossWeightDefault: Float = 1.0

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
    /// Live channel-dropout rate (drop probability, 0 = off). Unlike the
    /// per-step scalar feeds above, the rate lives in a GRAPH VARIABLE
    /// (`dropout_rate`) read by the training-mode forward pass — a
    /// placeholder there would impose a feed requirement on every forward
    /// run site. Setting this property therefore pushes the value into the
    /// graph via a tiny out-of-band assign on `executionQueue` (the
    /// loadWeights pattern); it takes effect on the next training step.
    /// Reads return the last value pushed. Clamped to the parameter's
    /// [0, 0.95] range defensively — a rate of 1.0 would divide by zero in
    /// the inverted-dropout scale.
    var dropoutRate: Float {
        get { _dropoutRate.value }
        set { pushDropoutRateToGraph(newValue) }
    }
    private let _dropoutRate = SyncBox<Float>(0)
    /// Live gradient-clip max norm. Fed via scalar placeholder each
    /// step.
    var gradClipMaxNorm: Float
    /// Live policy-loss coefficient. Multiplied into `policyLoss`
    /// before it joins the (similarly weighted) `valueLoss` term
    /// and `−entropyCoeff·policyEntropy` in `total_loss`. Fed via
    /// scalar placeholder each step (live-tunable). Behaviorally a
    /// per-head weighting on shared-trunk gradients — bigger value
    /// = trunk pulled toward minimizing policy CE, smaller = trunk
    /// follows the value-loss gradient instead. NOT a multiplier
    /// on the policy LOGITS; see `weightedPolicy` in
    /// `buildTrainingOps`.
    var policyLossWeight: Float
    /// Live value-loss coefficient. Multiplied into `valueLoss`
    /// before it joins `weightedPolicy` in `total_loss`. Pairs
    /// with `policyLossWeight`; the absolute scale of the two
    /// together is equivalent to a learning-rate multiplier, so
    /// only the ratio matters for shaping shared-trunk gradients.
    /// Mirror of Lc0 / KataGo's `value_loss_weight`.
    var valueLossWeight: Float

    /// Live illegal-mass penalty coefficient. Multiplied into
    /// `illegalMassPenalty` before it joins `total_loss` (minimizing
    /// total loss minimizes illegal mass). Start at 1.0; increase
    /// if the legal-mass diagnostic starts to walk down.
    var illegalMassPenaltyWeight: Float

    /// Label-smoothing coefficient ε for the policy CE target. Fed
    /// to the training graph each step as a scalar placeholder, so
    /// it's live-tunable.
    ///
    /// The policy CE target is built in-graph as
    ///   target = (1 − ε) · one_hot(played) + ε · uniform(legal)
    /// where `uniform(legal)` is `legalMask / |legal|` per position.
    /// At ε=0 the target collapses to one-hot (legacy behavior). At
    /// ε=0.1 the trainer reaches equilibrium with `p(played) ≈ 1 − ε`
    /// instead of the unreachable `p(played) = 1`, capping per-
    /// position concentration at a fixed level and converting the
    /// unbounded `−log p` gradient drive into a stable fixed point.
    /// The same ε also parameterizes the complement target used by
    /// the negative-advantage branch (`useSignedAdvantageComplementCE`),
    /// so the bounded-below equilibrium holds symmetrically on both
    /// signs of the advantage.
    var policyLabelSmoothingEpsilon: Float

    /// Label-smoothing coefficient ε for the value-head W/D/L CE
    /// target. Fed to the training graph each step as a scalar
    /// placeholder, so it's live-tunable.
    ///
    /// The value CE target is built in-graph as
    ///   target = (1 − ε) · one_hot(1 − z) + ε · (1/3)
    /// where `1 − z` maps the outcome z ∈ {+1, 0, −1} to the
    /// `[win, draw, loss]` slot. At ε=0 the target is a hard one-hot
    /// (the W/D/L equivalent of the legacy MSE-on-z target). At ε>0
    /// the loss equilibrium becomes a finite, reachable logit gap
    /// instead of `±∞`, exactly as for the policy label smoothing.
    /// Default 0 for the first WDL run; the parameter exists so a
    /// small positive value can be dialed in without rebuilding.
    var valueLabelSmoothingEpsilon: Float

    /// Polyak momentum coefficient μ. Fed into the training graph as
    /// a scalar placeholder each step (live-tunable). The optimizer
    /// update is decoupled-decay SGD with momentum:
    ///   `v_new      = μ·v_old + clipped_grad`
    ///   `weight_new = weight_old − lr·v_new − lr·decayC·weight_old`
    /// (decay term is skipped for variables flagged in
    /// `network.trainableShouldDecay`). μ=0 is equivalent to plain
    /// SGD with weight decay (the velocity term zeros out and the
    /// formula reduces to the classic update). Higher μ amplifies
    /// the effective velocity contribution to each weight step by
    /// ~1/(1−μ) in steady state under correlated gradients, so
    /// μ=0.9 still behaves like ~10× the LR on the gradient term —
    /// known to push this network into the one-hot-illegal collapse
    /// mode at the default `learningRate`. Decay is now
    /// independent of μ (decoupled form), so changing μ no longer
    /// silently amplifies decay. Start low and watch `legalMass` /
    /// `pEntLegal` before raising further.
    var momentumCoeff: Float

    /// Engage the complementary-CE branch for negative-advantage
    /// samples. Fed into the training graph each step as a 1.0/0.0
    /// scalar (live-tunable).
    ///
    /// When true, the policy gradient is driven by two cross-entropy
    /// terms, each bounded below by zero:
    ///   weightedCE = max(0, advNorm) * positiveCE
    ///              + max(0, -advNorm) * complementCE
    /// where `positiveCE` targets the smoothed one-hot on the played
    /// move (existing label smoothing) and `complementCE` targets a
    /// mirror-smoothed distribution that puts the (1 − ε) main mass
    /// on the *other* legal moves and ε across all legals. Negative-
    /// advantage samples push the played-move mass toward the other
    /// legals; the loss is bounded below by zero on both signs and the
    /// equilibria are reachable (p(played) → 1 − ε + ε/|legal| for
    /// positives, p(played) → ε/|legal| for negatives).
    ///
    /// When false, only the positive branch is active and negative-
    /// advantage samples contribute zero policy gradient — the legacy
    /// `max(0, advNorm) * positiveCE` clamp regime. Provided as the
    /// kill switch in case the complement branch destabilizes training.
    var useSignedAdvantageComplementCE: Bool

    /// Bootstrap-phase contempt knob. When `drawPenalty > 0`, every
    /// drawn game's outcome `z` is rewritten from `0.0` to
    /// `-drawPenalty` before the batch reaches the training graph —
    /// applied CPU-side in `trainStepFromReplay` (phase 3), after the
    /// replay sample is staged. `0` (default) is a no-op. All four
    /// draw types (stalemate, 50-move, threefold, insufficient
    /// material) arrive with `z == 0.0` exactly and are treated alike.
    ///
    /// The rewritten `z` reaches two consumers:
    ///
    /// (a) The policy gradient. The per-position weight is
    /// `max(0, normalize(z − vBaseline))` — REINFORCE-with-baseline,
    /// advantage RMS-normalized, then the negative branch dropped (see
    /// `buildTrainingOps`: `−A·log p` is unbounded below for `A < 0`).
    /// Because negatives are clamped to zero, `drawPenalty` can never
    /// produce a *punishing* gradient on a draw — it only shifts draws
    /// toward (or into) the dropped zone. A drawn position reinforces
    /// its played moves only when `z − vBaseline > 0`, i.e. when
    /// `vBaseline < −drawPenalty`: the network must have expected to
    /// lose by more than `drawPenalty` for clawing back the draw to
    /// still count as a positive sample. So `drawPenalty` is a
    /// *threshold on salvaged draws*, not a penalty term.
    ///
    /// (b) The value head W/D/L slot, `clamp(int(1 − z), 0, 2)`. For
    /// `drawPenalty ∈ (0, 1)`, `int(1 + drawPenalty)` truncates back
    /// to slot 1 (draw) — the value target is unchanged. Only the full
    /// `drawPenalty = 1` lands on slot 2 (loss), relabeling draws as
    /// losses and corrupting the W/D/L calibration the head exists to
    /// provide. So for any `drawPenalty < 1`, consumer (b) is inert and
    /// the entire effect is the policy-gradient threshold in (a).
    ///
    /// Why "bootstrap-phase": early in a fresh run the value head is
    /// near-random (`vBaseline ≈ 0` everywhere), so a positive
    /// `drawPenalty` keeps essentially every draw below the
    /// reinforcement threshold and the policy learns predominantly
    /// from decisive wins. As the value head calibrates, confidently
    /// drawn positions converge to `vBaseline ≈ 0` and sit just under
    /// the threshold regardless of `drawPenalty` — so the knob's effect
    /// concentrates on the transient and on misjudged draws, and `0`
    /// is a reasonable steady-state default.
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
    /// (`__processSnapshotTimerTick`) don't have to `.sync`
    /// onto the trainer's worker queue and wait for an in-flight
    /// SGD step — that pattern was producing 1–3 s main-thread
    /// stalls. The lock is held only across a scalar read/RMW so
    /// contention between the trainer thread and UI thread is
    /// effectively free. The lock-protected `+= 1` in
    /// `runPreparedStep` keeps the read-modify-write atomic on its
    /// own, no longer dependent on the queue invariant.
    private let _completedTrainSteps = SyncBox<Int>(0)

    /// Live LR/momentum cycling configuration (see `LRMomentumCycle` /
    /// TRAINING_DYNAMICS_PLAN.md §3). Read once per training step in
    /// `buildFeeds` — which runs off-main on `executionQueue` — and written
    /// from the main actor at session start, on each cycling edit, and on
    /// resume. Stored in a `SyncBox` (not a bare `var` like `learningRate`)
    /// because it is a multi-field struct: a bare `var` could tear a half-
    /// applied edit across the main/off-main boundary and feed a single step
    /// a mismatched (enabled / min / max) combination. The lock is held only
    /// across a struct copy, so the per-step read cost is negligible.
    /// `.disabled` until a session pushes the real config, so absent any
    /// configuration the static `learningRate` / `momentumCoeff` are used.
    private let _lrMomentumCycle = SyncBox<LRMomentumCycle>(.disabled)
    var lrMomentumCycle: LRMomentumCycle {
        get { _lrMomentumCycle.value }
        set { _lrMomentumCycle.value = newValue }
    }

    private let executionQueue = DispatchQueue(label: "drewschess.chesstrainer.serial")

    /// Optional stable identity for the trainer's internal network.
    /// Assigned by the UI layer at Play-and-Train start (after loading
    /// champion weights) and then kept stable for the lifetime of the
    /// Play-and-Train session — it represents the "current training
    /// lineage" rather than a specific byte-exact weight snapshot.
    /// See `sampling-parameters.md` for the full rule set.
    var identifier: ModelID?

    // MARK: Graph Tensors

    /// Architecture the trainer's network is built to — must match the champion
    /// it forks from. Captured at init and reused by `internalResetNetwork`.
    let arch: NetworkArchitecture
    /// MPSGraph compilation optimization level for the precompiled training
    /// executable. Defaults to `.level1` (the production setting). Exposed as an
    /// init parameter purely so the macOS-27 NaN-isolation tests can compile a
    /// `.level0` trainer and check whether the level-1 codegen path is what
    /// turns bf16 multi-step gradients non-finite.
    let executableOptimizationLevel: MPSGraphOptimization
    /// Experimental "config D" mixed-precision mode (see
    /// `ChessNetwork.bf16CastInForward`). When true, the trainer builds its
    /// training-mode network with fp32-stored weights cast to bf16 in the
    /// forward, and the optimizer runs the plain fp32 path (no masters, no
    /// working-sync). Threaded to every `ChessNetwork` the trainer builds.
    /// Default false keeps the canonical bf16-working-var / fp32-master path.
    let bf16CastInForward: Bool
    /// A/B knob for the macOS-27 NaN-isolation matrix: when true, every
    /// `ChessNetwork` this trainer builds calls `disableAutoLayoutConversion()`
    /// on its `MPSGraph`, opting out of the new (Xcode 27 b1 / macOS 27 beta)
    /// default that auto-converts conv layouts on the GPU. Default false leaves
    /// the production behavior unchanged. See `ChessNetwork.init`.
    let disableAutoLayoutConversion: Bool
    /// A/B knob for the macOS-27 NaN-isolation matrix: when non-nil, every
    /// `MPSGraphCompilationDescriptor` this trainer builds has its
    /// `reducedPrecisionFastMath` set to this value. `.none` forbids reduced-
    /// precision conv shortcuts (FP16 winograd intermediates, FP32->FP19/TF32
    /// operand narrowing). The documented default is already `.none`, so this is a
    /// force/verify lever, not a behavior change. Stored as the enum's raw `UInt`
    /// (the macOS-26 enum can't be named in a property on the app's deployment
    /// target); reconstructed under `#available` at the compile site. nil leaves
    /// the descriptor default.
    let reducedPrecisionFastMathRaw: UInt?
    /// Workaround for a bf16 mixed-precision GPU buffer stomp first seen under
    /// **Xcode 27 beta 1 + macOS 27 beta** (2026-06). When true (the default),
    /// the bf16 working-weight sync `working = cast(master)` is split OUT of the
    /// fused training executable and run as a SEPARATE `graph.run` after the
    /// master update's command buffer has fully completed. No-op under
    /// `.float32` (there is no second write to split).
    ///
    /// THE BUG. Under bf16, the per-trainable optimizer tail emits TWO target
    /// writes into one compiled executable: the fp32 master assign and the bf16
    /// working assign through an (unnamed) `cast` temporary (see the update loop
    /// in `buildTrainingOps`). On this beta stack that fused dual-write corrupts
    /// trainable weight buffers: the bf16 working weights read back NaN and the
    /// fp32 masters read back garbage (an exact `1.0` sentinel) from step 2 on,
    /// poisoning the whole net within a few steps. It is:
    ///   - bf16-ONLY — fp32 (single write, no master, no cast temp) is clean;
    ///   - NOT the cast op — a standalone `cast([128,32] fp32 -> bf16)` through
    ///     graph.run AND the compiled-executable path is bit-exact clean;
    ///   - intermittent / layout-sensitive — it favors the rank-2 SE fc1
    ///     `[128,32]` and fc2-bias `[1,256]` tensors, with the corrupted element
    ///     count growing with tower size (consistent with a buffer-aliasing /
    ///     liveness-planner fault around the fused dual-write, not a value bug);
    ///   - NEW with the toolchain/OS upgrade only — byte-identical source
    ///     trained bf16 for hundreds of thousands of steps before it. A stale
    ///     Metal toolchain was ruled out (reinstalling the matching one did not
    ///     help; `xcode-select` already pointed at the beta).
    ///
    /// THE FIX. Splitting the working sync into its own pass (different
    /// allocation/liveness scope, its own command buffer) takes the single-block
    /// repro from ~1,010,433 non-finite working elements to 0 across 16 steps,
    /// reproducibly. Only the *trainable* working writes are split; the BN
    /// running-stat master+working write is left fused (it never corrupted — the
    /// split A/B's full-net export, BN included, was 0 non-finite).
    ///
    /// ANE NOTE (recorded in case it matters). Throughout the bf16 runs the
    /// console spews, many times per step:
    ///     Error: ANE cannot handle intermediate tensor type fp32
    ///     Failed to create unit plist.
    /// The Apple Neural Engine is fp16-only, so it correctly refuses the fp32
    /// intermediates the bf16 path uniquely carries (masters / grad-widening
    /// casts) — MPSGraph attempts to partition onto the ANE, the ANE rejects the
    /// fp32 work, and it should fall back to GPU. That makes this most likely
    /// benign fallback noise, NOT the cause. It is not fully excluded, though:
    /// the message is about an *intermediate*, and the stomp is about the fp32
    /// master / cast-temp intermediate, so a bad ANE<->GPU partition/fallback
    /// handoff that the split happens to move off the offending boundary remains
    /// possible. The split fixes the stomp without disabling the ANE, so the two
    /// aren't distinguished; an ANE-disable test would settle it.
    ///
    /// Set false to restore the original fused single-executable path (one
    /// `graph.run` per step, faster) — do that once a future toolchain/OS fixes
    /// the underlying stomp, or to reproduce the bug for an Apple Feedback / the
    /// `MacOS27NaNIsolationTests` A/B.
    let splitWorkingWeightSync: Bool
    /// Separate `working = cast(master)` assigns, built when
    /// `splitWorkingWeightSync` is true; run as their own pass in
    /// `runPreparedStep`. Empty otherwise.
    private var workingSyncOps: [MPSGraphOperation] = []
    private(set) var network: ChessNetwork
    private var movePlayedPlaceholder: MPSGraphTensor   // [batch] int32
    private var zPlaceholder: MPSGraphTensor            // [batch, 1] float
    private var vBaselinePlaceholder: MPSGraphTensor    // [batch, 1] float
    private var legalMaskPlaceholder: MPSGraphTensor
    private var lrPlaceholder: MPSGraphTensor           // [] scalar float
    private var entropyCoeffPlaceholder: MPSGraphTensor // [] scalar float
    private var weightDecayPlaceholder: MPSGraphTensor  // [] scalar float
    private var gradClipMaxNormPlaceholder: MPSGraphTensor // [] scalar float
    private var policyLossWeightPlaceholder: MPSGraphTensor // [] scalar float
    private var valueLossWeightPlaceholder: MPSGraphTensor  // [] scalar float
    private var illegalMassWeightPlaceholder: MPSGraphTensor // [] scalar float
    private var labelSmoothingEpsilonPlaceholder: MPSGraphTensor // [] scalar float
    private var valueLabelSmoothingEpsilonPlaceholder: MPSGraphTensor // [] scalar float
    private var momentumPlaceholder: MPSGraphTensor     // [] scalar float — Polyak μ
    private var complementCEEnablePlaceholder: MPSGraphTensor // [] scalar float (1.0/0.0)
    /// Per-trainable-variable momentum velocity buffers, allocated parallel
    /// to `network.trainableVariables`. Each step's update is
    /// `v_new = μ·v_old + clipped_grad` — weight decay does NOT enter the
    /// velocity; it is applied decoupled at the weight-update site (see the
    /// optimizer comment on the SGD-update loop). This list
    /// holds the `v` for each variable. Initialized to zero on graph build
    /// (so μ=0.0 reduces to plain SGD bit-exact). Persisted across
    /// `Stop`/`Continue` and session save/load via the trainer's
    /// `.dcmmodel` file (format v2+).
    private var velocityVariables: [MPSGraphTensor] = []
    /// Per-velocity load infrastructure, parallel to `velocityVariables`.
    /// Used by `loadTrainerWeights(_:)` and `resetVelocitiesToZero()`
    /// to overwrite the velocity buffers from a saved-session snapshot
    /// or to clear them on promotion. Each entry is bound to the
    /// matching velocity variable via a single assign op:
    ///   `assign(velocityVariables[i], tensor: velocityLoadPlaceholders[i])`
    /// The caller writes new bytes into `velocityLoadNDArrays[i]` and
    /// runs the assign as a target op.
    private var velocityLoadPlaceholders: [MPSGraphTensor] = []
    private var velocityLoadAssignOps: [MPSGraphOperation] = []
    private var velocityLoadNDArrays: [MPSNDArray] = []
    private var velocityLoadTensorData: [MPSGraphTensorData] = []
    /// fp32 master weights + master running stats (canonical mixed-precision
    /// path; empty under `.float32`). Parallel to `trainableVariables +
    /// bnRunningStatsVariables`. Persisted in the trainer session; the bf16
    /// working copies are re-derived as `cast(master)` each step.
    private var masterVariables: [MPSGraphTensor] = []
    /// Per-master load infra (fp32), parallel to `masterVariables`, for
    /// restoring persisted masters on resume.
    private var masterLoadPlaceholders: [MPSGraphTensor] = []
    private var masterLoadAssignOps: [MPSGraphOperation] = []
    private var masterLoadNDArrays: [MPSNDArray] = []
    private var masterLoadTensorData: [MPSGraphTensorData] = []
    /// `master = cast(working, fp32)` assigns. Run once at construction and
    /// after any wholesale working-weight replacement (fresh fork / promotion)
    /// to seed the masters from the working copies.
    private var syncMastersOps: [MPSGraphOperation] = []
    private var totalLoss: MPSGraphTensor               // scalar
    private var policyLossTensor: MPSGraphTensor        // scalar
    private var valueLossTensor: MPSGraphTensor         // scalar
    private var policyEntropyTensor: MPSGraphTensor     // scalar (diagnostic)
    private var illegalMassPenaltyTensor: MPSGraphTensor // scalar (diagnostic)
    private var policyNonNegCountTensor: MPSGraphTensor // scalar (diagnostic)
    private var policyNonNegIllegalCountTensor: MPSGraphTensor // scalar (diagnostic, illegal cells)
    private var gradGlobalNormTensor: MPSGraphTensor    // scalar (diagnostic)
    private var valueMeanTensor: MPSGraphTensor         // scalar (diagnostic) — mean of p_win − p_loss
    private var valueAbsMeanTensor: MPSGraphTensor      // scalar (diagnostic) — mean |p_win − p_loss|
    private var valueProbWinTensor: MPSGraphTensor      // scalar (diagnostic) — batch-mean p_win
    private var valueProbDrawTensor: MPSGraphTensor     // scalar (diagnostic) — batch-mean p_draw
    private var valueProbLossTensor: MPSGraphTensor     // scalar (diagnostic) — batch-mean p_loss
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
    /// Scalar global L2 norm of the post-step velocity buffer ‖v_new‖.
    /// Reported on the [STATS] line so velocity magnitude growth is
    /// observable when raising μ.
    private var velocityGlobalNormTensor: MPSGraphTensor
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
    private var policyLossWeightNDArray: MPSNDArray
    private var policyLossWeightTensorData: MPSGraphTensorData
    private var valueLossWeightNDArray: MPSNDArray
    private var valueLossWeightTensorData: MPSGraphTensorData
    private var illegalMassWeightNDArray: MPSNDArray
    private var illegalMassWeightTensorData: MPSGraphTensorData
    private var labelSmoothingEpsilonNDArray: MPSNDArray
    private var labelSmoothingEpsilonTensorData: MPSGraphTensorData
    private var valueLabelSmoothingEpsilonNDArray: MPSNDArray
    private var valueLabelSmoothingEpsilonTensorData: MPSGraphTensorData
    private var momentumNDArray: MPSNDArray
    private var momentumTensorData: MPSGraphTensorData
    private var complementCEEnableNDArray: MPSNDArray
    private var complementCEEnableTensorData: MPSGraphTensorData

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

        /// Reusable host-side bf16 staging for the four real-valued
        /// feeds (board, z, vBaseline, legalMask) when the network dtype
        /// is narrower than Float32 (bf16/fp16). The per-ply self-play
        /// and replay paths always produce these as Float32, but every
        /// one of their graph placeholders is declared at
        /// `the net's compute dtype` — so the matching ND array storage is
        /// that same width, and each step must narrow Float32 → dtype
        /// bits before `writeBytes`. Each buffer is allocated once per
        /// batch size, sized to its tensor's element count, and reused
        /// every step so the timed training loop stays allocation-free,
        /// exactly like the ND arrays themselves.
        ///
        /// All four are `nil` when `the net's compute dtype == .float32`:
        /// there each ND array is Float32 too, so the host hands its raw
        /// Float32 bytes straight through with no conversion or scratch.
        ///
        /// The move feed never needs staging — it is int32 on both the
        /// host and the placeholder, so its Int32 bytes are always fed
        /// raw regardless of `the net's compute dtype`.
        let boardStaging: UnsafeMutableBufferPointer<UInt16>?
        let zStaging: UnsafeMutableBufferPointer<UInt16>?
        let vBaselineStaging: UnsafeMutableBufferPointer<UInt16>?
        let legalMaskStaging: UnsafeMutableBufferPointer<UInt16>?
    }

    /// Deallocate the four bf16 staging buffers a `BatchFeeds` owns.
    /// Each `UnsafeMutableBufferPointer` was hand-allocated in
    /// `feedsForBatch` (and is nil on `.float32`), so it must be
    /// explicitly freed when the owning `BatchFeeds` leaves the cache —
    /// on `resetNetwork`'s cache drop and on `deinit`. The ND arrays /
    /// tensor-data wrappers are ARC-managed and need no manual free.
    private static func freeBatchFeedsStaging(_ feeds: BatchFeeds) {
        for staging in [feeds.boardStaging, feeds.zStaging, feeds.vBaselineStaging, feeds.legalMaskStaging] {
            if let staging {
                staging.deinitialize()
                staging.deallocate()
            }
        }
    }
    private var feedCache: [Int: BatchFeeds] = [:]

    /// Compiled training-step executables, keyed by batch size and whether the
    /// diagnostic targets are included (Phase 1's two target sets → up to two
    /// executables per batch size). Built lazily on first use for each key and
    /// reused for the trainer network's lifetime; `resetNetwork` clears them
    /// alongside `feedCache`. Accessed only on `executionQueue` (same as
    /// `feedCache`), so no extra locking. See GPU_UTILIZATION_PLAN.md (Phase 2).
    private struct TrainingExecutableKey: Hashable {
        let batchSize: Int
        let includeDiagnostics: Bool
    }
    private var trainingExecutables: [TrainingExecutableKey: MPSGraphExecutable] = [:]

    /// Mirror of `feedCache.count` updated on every cache mutation so a
    /// reader on a different queue can observe the size without
    /// touching the dictionary itself. Exposed via `feedCacheCount`
    /// for the periodic `[STATS]` line — a stable count of 1 across a
    /// session means the trainer is calling MPSGraph with one feed
    /// shape (i.e. shape-variance is not the cause of any pipeline
    /// re-specialization MPSGraph is doing).
    private let _feedCacheCount = SyncBox<Int>(0)
    var feedCacheCount: Int { _feedCacheCount.value }

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
    private static let lossReadbackSlotIllegalMassPenalty: Int = 4
    private static let lossReadbackSlotNonNeg: Int = 5
    private static let lossReadbackSlotGradNorm: Int = 6
    private static let lossReadbackSlotValueMean: Int = 7
    private static let lossReadbackSlotValueAbsMean: Int = 8
    private static let lossReadbackSlotPolicyHeadWNorm: Int = 9
    private static let lossReadbackSlotPLogitAbsMax: Int = 10
    private static let lossReadbackSlotPlayedMoveProb: Int = 11
    private static let lossReadbackSlotAdvMean: Int = 12
    private static let lossReadbackSlotAdvStd: Int = 13
    private static let lossReadbackSlotAdvMin: Int = 14
    private static let lossReadbackSlotAdvMax: Int = 15
    private static let lossReadbackSlotAdvFracPos: Int = 16
    private static let lossReadbackSlotAdvFracSmall: Int = 17
    private static let lossReadbackSlotPlayedMoveProbPosAdv: Int = 18
    private static let lossReadbackSlotPlayedMoveProbNegAdv: Int = 19
    private static let lossReadbackSlotPolicyLossWin: Int = 20
    private static let lossReadbackSlotPolicyLossLoss: Int = 21
    private static let lossReadbackSlotNonNegIllegal: Int = 22
    private static let lossReadbackSlotVelocityNorm: Int = 23
    private static let lossReadbackSlotValueProbWin: Int = 24
    private static let lossReadbackSlotValueProbDraw: Int = 25
    private static let lossReadbackSlotValueProbLoss: Int = 26
    private static let lossReadbackSlotCount: Int = 27

    /// Reusable host-side staging buffers for replay-buffer samples.
    /// The trainer owns these buffers so real-data training can hop
    /// onto `executionQueue`, sample directly into stable storage, and
    /// feed MPSGraph without any additional ownership-transfer copy.
    private var replayBatchCapacity: Int = 0
    private var replayBatchBoards: UnsafeMutablePointer<Float>?
    private var replayBatchMoves: UnsafeMutablePointer<Int32>?
    private var replayBatchZs: UnsafeMutablePointer<Float>?
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

    // Per-step phase timings, accumulated within the current
    // batchStatsInterval window. Reset on every emit. Touched only
    // from inside `enqueue { ... }` closures (the trainer's serial
    // executor), so plain `var` is sufficient.
    private var phase1WallTimesMs: [Double] = []
    private var phase2WallTimesMs: [Double] = []
    private var phase3WallTimesMs: [Double] = []
    private var legalMaskLoopMsTimes: [Double] = []
    /// Pipeline-feasibility probe (GPU_UTILIZATION_PLAN Phase 3): pure CPU
    /// `executable.encode(...)` time vs the `commit + waitUntilCompleted` GPU
    /// wall, per training step. The encode call is documented async (returns
    /// after encoding, before the GPU runs), so these cleanly separate
    /// CPU-encode from GPU-execution — the number that decides whether a
    /// single-encoder pipeline can outpace the GPU (encode ≪ gpuWait) or whether
    /// we need parallel encoders / it's GPU-bound. Accumulated per step, emitted
    /// as `[ENCODE-COST]` on diagnostics steps.
    private var encodeMsTimes: [Double] = []
    private var gpuWaitMsTimes: [Double] = []
    /// Inter-step wall time: gap from the moment one training batch
    /// completes (after phase 3) to the same moment of the next.
    /// Captures everything `p1ms + p2ms + p3ms` doesn't — caller-side
    /// dispatch latency, awaits between phases, replay-ratio-controller
    /// throttle sleeps, any idle time before the next call lands. The
    /// first step has no prior reference so its delta is skipped.
    private var interStepWallTimesMs: [Double] = []
    /// Wall-clock time at which the most recent training step completed
    /// (end of phase 3). 0 means "no prior step yet" — used to skip
    /// the first delta after session start.
    private var lastTrainStepCompletedAt: CFAbsoluteTime = 0

    // Cached scratch for the synthetic-data sweep path
    // (`internalTrainStep(batchSize:)`). Reused across calls instead
    // of being freshly allocated each step.
    private var syntheticBoards: [Float] = []
    private var syntheticMoves: [Int32] = []
    private var syntheticZs: [Float] = []
    private var syntheticVBaselines: [Float] = []
    private var syntheticLegalMasks: [Float] = []

    // MARK: Init

    init(
        // Mirrors the `LearningRate` TrainingParameter default; 1e-3 is a
        // bf16-appropriate floor (smaller LRs produce sub-ULP, no-op weight
        // updates under the bf16 weight path). The live session always
        // passes an explicit value from TrainingParameters, so this is only
        // a test / fallback default.
        learningRate: Float = 1e-3,
        entropyRegularizationCoeff: Float = 0.0,
        drawPenalty: Float = 0.1,
        weightDecayC: Float = ChessTrainer.weightDecayCDefault,
        gradClipMaxNorm: Float = ChessTrainer.gradClipMaxNormDefault,
        policyLossWeight: Float = ChessTrainer.policyLossWeightDefault,
        valueLossWeight: Float = ChessTrainer.valueLossWeightDefault,
        illegalMassPenaltyWeight: Float = 1.0,
        policyLabelSmoothingEpsilon: Float = 0.1,
        valueLabelSmoothingEpsilon: Float = 0.0,
        momentumCoeff: Float = 0.0,
        useSignedAdvantageComplementCE: Bool = true,
        sqrtBatchScalingForLR: Bool = true,
        lrWarmupSteps: Int = 100,
        arch: NetworkArchitecture = .current,
        executableOptimizationLevel: MPSGraphOptimization = .level1,
        splitWorkingWeightSync: Bool = true,
        bf16CastInForward: Bool = false,
        disableAutoLayoutConversion: Bool = false,
        reducedPrecisionFastMathRaw: UInt? = nil
    ) throws {
        self.learningRate = learningRate
        self.entropyRegularizationCoeff = entropyRegularizationCoeff
        self.drawPenalty = drawPenalty
        self.weightDecayC = weightDecayC
        self.gradClipMaxNorm = gradClipMaxNorm
        self.policyLossWeight = policyLossWeight
        self.valueLossWeight = valueLossWeight
        self.illegalMassPenaltyWeight = illegalMassPenaltyWeight
        self.policyLabelSmoothingEpsilon = policyLabelSmoothingEpsilon
        self.valueLabelSmoothingEpsilon = valueLabelSmoothingEpsilon
        self.momentumCoeff = momentumCoeff
        self.useSignedAdvantageComplementCE = useSignedAdvantageComplementCE
        self.sqrtBatchScalingForLR = sqrtBatchScalingForLR
        self.lrWarmupSteps = lrWarmupSteps
        self.arch = arch
        self.executableOptimizationLevel = executableOptimizationLevel
        self.splitWorkingWeightSync = splitWorkingWeightSync
        self.bf16CastInForward = bf16CastInForward
        self.disableAutoLayoutConversion = disableAutoLayoutConversion
        self.reducedPrecisionFastMathRaw = reducedPrecisionFastMathRaw
        let net = try ChessNetwork(arch: arch, bnMode: .training, bf16CastInForward: bf16CastInForward,
                                   disableAutoLayoutConversion: disableAutoLayoutConversion,
                                   reducedPrecisionFastMathRaw: reducedPrecisionFastMathRaw)
        net.commandQueue.label = "ChessTrainer.net(init)"
        self.network = net
        let built = try withLargeBuildStack {
            try Self.buildTrainingOps(network: net, splitTrainableWorkingSync: splitWorkingWeightSync)
        }
        self.movePlayedPlaceholder = built.movePlayed
        self.zPlaceholder = built.z
        self.vBaselinePlaceholder = built.vBaseline
        self.legalMaskPlaceholder = built.legalMask
        self.lrPlaceholder = built.lr
        self.entropyCoeffPlaceholder = built.entropyCoeff
        self.weightDecayPlaceholder = built.weightDecay
        self.gradClipMaxNormPlaceholder = built.gradClipMaxNorm
        self.policyLossWeightPlaceholder = built.policyLossWeight
        self.valueLossWeightPlaceholder = built.valueLossWeight
        self.illegalMassWeightPlaceholder = built.illegalMassWeight
        self.labelSmoothingEpsilonPlaceholder = built.labelSmoothingEpsilon
        self.valueLabelSmoothingEpsilonPlaceholder = built.valueLabelSmoothingEpsilon
        self.momentumPlaceholder = built.momentum
        self.complementCEEnablePlaceholder = built.complementCEEnable
        self.velocityVariables = built.velocityVariables
        self.velocityLoadPlaceholders = built.velocityLoadPlaceholders
        self.velocityLoadAssignOps = built.velocityLoadAssignOps
        self.velocityLoadNDArrays = built.velocityLoadNDArrays
        self.velocityLoadTensorData = built.velocityLoadTensorData
        self.masterVariables = built.masterVariables
        self.masterLoadPlaceholders = built.masterLoadPlaceholders
        self.masterLoadAssignOps = built.masterLoadAssignOps
        self.masterLoadNDArrays = built.masterLoadNDArrays
        self.masterLoadTensorData = built.masterLoadTensorData
        self.syncMastersOps = built.syncMastersOps
        self.totalLoss = built.totalLoss
        self.policyLossTensor = built.policyLoss
        self.valueLossTensor = built.valueLoss
        self.policyEntropyTensor = built.policyEntropy
        self.illegalMassPenaltyTensor = built.illegalMassPenalty
        self.policyNonNegCountTensor = built.policyNonNegCount
        self.policyNonNegIllegalCountTensor = built.policyNonNegIllegalCount
        self.gradGlobalNormTensor = built.gradGlobalNorm
        self.valueMeanTensor = built.valueMean
        self.valueAbsMeanTensor = built.valueAbsMean
        self.valueProbWinTensor = built.valueProbWin
        self.valueProbDrawTensor = built.valueProbDraw
        self.valueProbLossTensor = built.valueProbLoss
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
        self.velocityGlobalNormTensor = built.velocityGlobalNorm
        self.assignOps = built.assignOps
        // Advance the dropout RNG stream exactly once per training step,
        // compiled into the same executable as the SGD assigns.
        if let advance = network.dropoutRngAdvanceOp {
            self.assignOps.append(advance)
        }
        // Experiment: when the working-weight sync is split out of the fused
        // executable, build the separate `working = cast(master)` assigns here
        // (no-op / empty unless splitWorkingWeightSync && bf16). Config D has
        // no working vars and no masters at all, so no working-sync is built.
        if splitWorkingWeightSync && !bf16CastInForward {
            self.workingSyncOps = Self.buildWorkingSyncOps(
                net: net, masterVariables: built.masterVariables, arch: arch)
        }

        // Scalar ND array for the learning rate feed, reused every step.
        // Every scalar hyperparameter placeholder (lr, entropyCoeff,
        // weightDecay, gradClipMaxNorm, the policy/value/illegal loss
        // weights, the label-smoothing epsilons, momentum,
        // complementCEEnable) is declared `dataType: dtype` in the graph
        // build — i.e. the network dtype (bf16 here) — so the ND array
        // storage they feed must be the same width. The host narrows
        // the Swift `Float` value into bf16 bits before each
        // `writeBytes` in `buildFeeds`; on `.float32` it writes the raw
        // `Float`. A `.float32` descriptor here would byte-mismatch the
        // bf16 placeholder under bf16.
        let lrDesc = MPSNDArrayDescriptor(
            dataType: ChessNetwork.mpsDataType(for: arch),
            shape: [1]
        )
        // The four optimizer-update scalars feed fp32 placeholders (see
        // buildTrainingOps) regardless of `dataType`, so their ND arrays are
        // fp32. `writeScalarFeed` branches on the ND array's own dtype, so it
        // writes the raw `Float` into these and narrows the rest.
        let optScalarDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [1]
        )
        let lrND = MPSNDArray(device: net.metalDevice, descriptor: optScalarDesc)
        lrND.label = "lrND"
        self.lrNDArray = lrND
        self.lrTensorData = MPSGraphTensorData(lrND)
        let entropyND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        entropyND.label = "entropyND"
        self.entropyCoeffNDArray = entropyND
        self.entropyCoeffTensorData = MPSGraphTensorData(entropyND)
        let weightDecayND = MPSNDArray(device: net.metalDevice, descriptor: optScalarDesc)
        weightDecayND.label = "weightDecayND"
        self.weightDecayNDArray = weightDecayND
        self.weightDecayTensorData = MPSGraphTensorData(weightDecayND)
        let gradClipND = MPSNDArray(device: net.metalDevice, descriptor: optScalarDesc)
        gradClipND.label = "gradClipND"
        self.gradClipMaxNormNDArray = gradClipND
        self.gradClipMaxNormTensorData = MPSGraphTensorData(gradClipND)
        let policyLossWeightND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        policyLossWeightND.label = "policyLossWeightND"
        self.policyLossWeightNDArray = policyLossWeightND
        self.policyLossWeightTensorData = MPSGraphTensorData(policyLossWeightND)
        let valueLossWeightND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        valueLossWeightND.label = "valueLossWeightND"
        self.valueLossWeightNDArray = valueLossWeightND
        self.valueLossWeightTensorData = MPSGraphTensorData(valueLossWeightND)
        let illegalMassWeightND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        illegalMassWeightND.label = "illegalMassWeightND"
        self.illegalMassWeightNDArray = illegalMassWeightND
        self.illegalMassWeightTensorData = MPSGraphTensorData(illegalMassWeightND)
        let labelSmoothingND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        labelSmoothingND.label = "labelSmoothingEpsilonND"
        self.labelSmoothingEpsilonNDArray = labelSmoothingND
        self.labelSmoothingEpsilonTensorData = MPSGraphTensorData(labelSmoothingND)
        let valueLabelSmoothingND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        valueLabelSmoothingND.label = "valueLabelSmoothingEpsilonND"
        self.valueLabelSmoothingEpsilonNDArray = valueLabelSmoothingND
        self.valueLabelSmoothingEpsilonTensorData = MPSGraphTensorData(valueLabelSmoothingND)
        let momentumND = MPSNDArray(device: net.metalDevice, descriptor: optScalarDesc)
        momentumND.label = "momentumND"
        self.momentumNDArray = momentumND
        self.momentumTensorData = MPSGraphTensorData(momentumND)
        let complementCEEnableND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        complementCEEnableND.label = "complementCEEnableND"
        self.complementCEEnableNDArray = complementCEEnableND
        self.complementCEEnableTensorData = MPSGraphTensorData(complementCEEnableND)

        let lossPtr = UnsafeMutablePointer<Float>.allocate(
            capacity: Self.lossReadbackSlotCount
        )
        lossPtr.initialize(repeating: 0, count: Self.lossReadbackSlotCount)
        self.lossReadbackScratchPtr = lossPtr

        // Seed the fp32 masters from the freshly-built (He/Glorot-init) bf16
        // working weights so the first trainStep accumulates from the real
        // init, not from zero. No-op under `.float32`. Not yet on
        // `executionQueue`, so wrap in a sync hop. The dropout RNG state is
        // seeded in the same hop (no-op on graphs without dropout nodes).
        executionQueue.sync {
            self.runSyncMastersOnQueue()
            self.runDropoutSeedOnQueue()
        }
    }

    deinit {
        // Free the bf16 staging each cached BatchFeeds owns (nil on
        // .float32). The ND arrays themselves are ARC-managed.
        for (_, feeds) in feedCache {
            Self.freeBatchFeedsStaging(feeds)
        }
        lossReadbackScratchPtr.deinitialize(count: Self.lossReadbackSlotCount)
        lossReadbackScratchPtr.deallocate()
        if let ptr = replayBatchBoards {
            ptr.deinitialize(count: replayBatchCapacity * arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize)
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
        let net = try ChessNetwork(arch: arch, bnMode: .training, bf16CastInForward: bf16CastInForward,
                                   disableAutoLayoutConversion: disableAutoLayoutConversion,
                                   reducedPrecisionFastMathRaw: reducedPrecisionFastMathRaw)
        net.commandQueue.label = "ChessTrainer.net(reset)"
        self.network = net
        let built = try withLargeBuildStack {
            try Self.buildTrainingOps(network: net, splitTrainableWorkingSync: self.splitWorkingWeightSync)
        }
        self.movePlayedPlaceholder = built.movePlayed
        self.zPlaceholder = built.z
        self.vBaselinePlaceholder = built.vBaseline
        self.legalMaskPlaceholder = built.legalMask
        self.lrPlaceholder = built.lr
        self.entropyCoeffPlaceholder = built.entropyCoeff
        self.weightDecayPlaceholder = built.weightDecay
        self.gradClipMaxNormPlaceholder = built.gradClipMaxNorm
        self.policyLossWeightPlaceholder = built.policyLossWeight
        self.valueLossWeightPlaceholder = built.valueLossWeight
        self.illegalMassWeightPlaceholder = built.illegalMassWeight
        self.labelSmoothingEpsilonPlaceholder = built.labelSmoothingEpsilon
        self.valueLabelSmoothingEpsilonPlaceholder = built.valueLabelSmoothingEpsilon
        self.momentumPlaceholder = built.momentum
        self.complementCEEnablePlaceholder = built.complementCEEnable
        self.velocityVariables = built.velocityVariables
        self.velocityLoadPlaceholders = built.velocityLoadPlaceholders
        self.velocityLoadAssignOps = built.velocityLoadAssignOps
        self.velocityLoadNDArrays = built.velocityLoadNDArrays
        self.velocityLoadTensorData = built.velocityLoadTensorData
        self.masterVariables = built.masterVariables
        self.masterLoadPlaceholders = built.masterLoadPlaceholders
        self.masterLoadAssignOps = built.masterLoadAssignOps
        self.masterLoadNDArrays = built.masterLoadNDArrays
        self.masterLoadTensorData = built.masterLoadTensorData
        self.syncMastersOps = built.syncMastersOps
        self.totalLoss = built.totalLoss
        self.policyLossTensor = built.policyLoss
        self.valueLossTensor = built.valueLoss
        self.policyEntropyTensor = built.policyEntropy
        self.illegalMassPenaltyTensor = built.illegalMassPenalty
        self.policyNonNegCountTensor = built.policyNonNegCount
        self.policyNonNegIllegalCountTensor = built.policyNonNegIllegalCount
        self.gradGlobalNormTensor = built.gradGlobalNorm
        self.valueMeanTensor = built.valueMean
        self.valueAbsMeanTensor = built.valueAbsMean
        self.valueProbWinTensor = built.valueProbWin
        self.valueProbDrawTensor = built.valueProbDraw
        self.valueProbLossTensor = built.valueProbLoss
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
        self.velocityGlobalNormTensor = built.velocityGlobalNorm
        self.assignOps = built.assignOps
        // Advance the dropout RNG stream exactly once per training step
        // (same as the designated init).
        if let advance = network.dropoutRngAdvanceOp {
            self.assignOps.append(advance)
        }
        // Rebuild the split working-sync ops against the fresh network (stale
        // ops from the previous net must not be reused). Config D has no
        // working vars / masters, so no working-sync is built.
        self.workingSyncOps = (splitWorkingWeightSync && !bf16CastInForward)
            ? Self.buildWorkingSyncOps(net: net, masterVariables: built.masterVariables, arch: arch)
            : []
        // Rebuild the LR scalar feed against the new network's device
        // so the new graph's placeholder maps to a fresh wrapper. As in
        // the designated init, these scalar feeds match the network
        // dtype (`dtype` placeholders in the graph build); the host
        // narrows each scalar to bf16 before `writeBytes` in
        // `buildFeeds` (raw `Float` on `.float32`).
        let lrDesc = MPSNDArrayDescriptor(
            dataType: ChessNetwork.mpsDataType(for: arch),
            shape: [1]
        )
        // fp32 ND arrays for the four optimizer-update scalars (see the
        // designated init for the rationale).
        let optScalarDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [1]
        )
        self.lrNDArray = MPSNDArray(device: net.metalDevice, descriptor: optScalarDesc)
        self.lrNDArray.label = "trainer.scalar.lr (reset)"
        self.lrTensorData = MPSGraphTensorData(lrNDArray)
        self.entropyCoeffNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.entropyCoeffNDArray.label = "trainer.scalar.entropyCoeff (reset)"
        self.entropyCoeffTensorData = MPSGraphTensorData(entropyCoeffNDArray)
        self.weightDecayNDArray = MPSNDArray(device: net.metalDevice, descriptor: optScalarDesc)
        self.weightDecayNDArray.label = "trainer.scalar.weightDecay (reset)"
        self.weightDecayTensorData = MPSGraphTensorData(weightDecayNDArray)
        self.gradClipMaxNormNDArray = MPSNDArray(device: net.metalDevice, descriptor: optScalarDesc)
        self.gradClipMaxNormNDArray.label = "trainer.scalar.gradClipMaxNorm (reset)"
        self.gradClipMaxNormTensorData = MPSGraphTensorData(gradClipMaxNormNDArray)
        self.policyLossWeightNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.policyLossWeightNDArray.label = "trainer.scalar.policyLossWeight (reset)"
        self.policyLossWeightTensorData = MPSGraphTensorData(policyLossWeightNDArray)
        self.valueLossWeightNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.valueLossWeightNDArray.label = "trainer.scalar.valueLossWeight (reset)"
        self.valueLossWeightTensorData = MPSGraphTensorData(valueLossWeightNDArray)
        self.illegalMassWeightNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.illegalMassWeightNDArray.label = "trainer.scalar.illegalMassWeight (reset)"
        self.illegalMassWeightTensorData = MPSGraphTensorData(illegalMassWeightNDArray)
        self.labelSmoothingEpsilonNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.labelSmoothingEpsilonNDArray.label = "trainer.scalar.labelSmoothingEpsilon (reset)"
        self.labelSmoothingEpsilonTensorData = MPSGraphTensorData(labelSmoothingEpsilonNDArray)
        self.valueLabelSmoothingEpsilonNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.valueLabelSmoothingEpsilonNDArray.label = "trainer.scalar.valueLabelSmoothingEpsilon (reset)"
        self.valueLabelSmoothingEpsilonTensorData = MPSGraphTensorData(valueLabelSmoothingEpsilonNDArray)
        self.momentumNDArray = MPSNDArray(device: net.metalDevice, descriptor: optScalarDesc)
        self.momentumNDArray.label = "trainer.scalar.momentum (reset)"
        self.momentumTensorData = MPSGraphTensorData(momentumNDArray)
        self.complementCEEnableNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.complementCEEnableNDArray.label = "trainer.scalar.complementCEEnable (reset)"
        self.complementCEEnableTensorData = MPSGraphTensorData(complementCEEnableNDArray)
        // The cached ND arrays were allocated against the old network's
        // device and are keyed by batch size against the old graph's
        // placeholders. Drop the cache so the first trainStep after
        // reset rebuilds against the fresh network. Free the manually-
        // allocated bf16 staging buffers each cached BatchFeeds owns
        // first, since `removeAll` only drops the struct references and
        // would otherwise leak the raw allocations.
        for (_, feeds) in feedCache {
            Self.freeBatchFeedsStaging(feeds)
        }
        feedCache.removeAll()
        _feedCacheCount.value = 0
        // The compiled executables reference this trainer network's graph and
        // variables; a rebuilt network invalidates them. Drop them so the next
        // step recompiles against the fresh graph.
        trainingExecutables.removeAll()
        // Fresh weights ⇒ the LR-warmup schedule must restart from
        // step 0. Without this, the warmup multiplier (a function of
        // `_completedTrainSteps / lrWarmupSteps`) jumps ahead of the
        // immature network and drives oversized first-step updates.
        _completedTrainSteps.value = 0

        // Seed the fp32 masters from the new network's freshly-built working
        // weights. Already on `executionQueue` (via `enqueue`), so run
        // directly. No-op under `.float32`. Re-seed the dropout RNG state
        // for the new graph as well.
        runSyncMastersOnQueue()
        runDropoutSeedOnQueue()
    }

    /// Build the training subgraph (loss + gradients + SGD assigns) on top
    /// of the given network's forward graph. Returns the placeholders, loss
    /// tensor, and assign ops the caller needs to run a training step.
    /// Throws `ChessTrainerError.gradientMissing` if any trainable variable
    /// fails gradient lookup — that would mean the autodiff couldn't reach
    /// it from the loss, which is a network-construction bug we want to
    /// surface immediately rather than silently train without it.
    /// Build the standalone `working = cast(master)` assign for each TRAINABLE
    /// (indices [0, trainableCount) of `masterVariables`), used by the
    /// split-working-sync experiment. Reads the master variable (which the main
    /// step already updated) and casts it into the bf16 working variable, in a
    /// separate graph pass. Empty on `.float32` (no masters).
    private static func buildWorkingSyncOps(
        net: ChessNetwork, masterVariables: [MPSGraphTensor], arch: NetworkArchitecture
    ) -> [MPSGraphOperation] {
        let dtype = ChessNetwork.mpsDataType(for: arch)
        guard dtype != .float32 else { return [] }
        let g = net.graph
        // ALL persistent working vars — trainables THEN bn running stats — in the
        // same order as `masterVariables`, so master[i] feeds working[i]. The bn
        // running-stat sync is included (its fused dual-write is the same stomp
        // pattern as the trainable one — see the macOS-27 repro); covering only
        // trainables left a corrupting path uncovered.
        let working = net.trainableVariables + net.bnRunningStatsVariables
        var ops: [MPSGraphOperation] = []
        ops.reserveCapacity(working.count)
        for i in 0..<working.count {
            let synced = g.cast(masterVariables[i], to: dtype, name: "split_working_sync_\(i)")
            ops.append(g.assign(working[i], tensor: synced, name: "split_working_assign_\(i)"))
        }
        return ops
    }

    private static func buildTrainingOps(
        network: ChessNetwork,
        splitTrainableWorkingSync: Bool = false
    ) throws -> (
        movePlayed: MPSGraphTensor,
        z: MPSGraphTensor,
        vBaseline: MPSGraphTensor,
        legalMask: MPSGraphTensor, // legal moves mask
        lr: MPSGraphTensor,
        entropyCoeff: MPSGraphTensor,
        weightDecay: MPSGraphTensor,
        gradClipMaxNorm: MPSGraphTensor,
        policyLossWeight: MPSGraphTensor,
        valueLossWeight: MPSGraphTensor,
        illegalMassWeight: MPSGraphTensor,
        labelSmoothingEpsilon: MPSGraphTensor,
        valueLabelSmoothingEpsilon: MPSGraphTensor,
        momentum: MPSGraphTensor,
        complementCEEnable: MPSGraphTensor,
        velocityVariables: [MPSGraphTensor],
        velocityLoadPlaceholders: [MPSGraphTensor],
        velocityLoadAssignOps: [MPSGraphOperation],
        velocityLoadNDArrays: [MPSNDArray],
        velocityLoadTensorData: [MPSGraphTensorData],
        masterVariables: [MPSGraphTensor],
        masterLoadPlaceholders: [MPSGraphTensor],
        masterLoadAssignOps: [MPSGraphOperation],
        masterLoadNDArrays: [MPSNDArray],
        masterLoadTensorData: [MPSGraphTensorData],
        syncMastersOps: [MPSGraphOperation],
        totalLoss: MPSGraphTensor,
        policyLoss: MPSGraphTensor,
        valueLoss: MPSGraphTensor,
        policyEntropy: MPSGraphTensor,
        illegalMassPenalty: MPSGraphTensor,
        policyNonNegCount: MPSGraphTensor,
        policyNonNegIllegalCount: MPSGraphTensor,
        gradGlobalNorm: MPSGraphTensor,
        valueMean: MPSGraphTensor,
        valueAbsMean: MPSGraphTensor,
        valueProbWin: MPSGraphTensor,
        valueProbDraw: MPSGraphTensor,
        valueProbLoss: MPSGraphTensor,
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
        velocityGlobalNorm: MPSGraphTensor,
        assignOps: [MPSGraphOperation]
    ) {
        // Stack-overflow backstop for pathologically deep towers — see
        // `graphBuildStackBytes`. Autodiff stack appetite scales with depth:
        // empirically a 100-block tower already overflowed the ~512 KB default
        // stack (so > ~5 KB/block), and 150 blocks recursed ~1,419 frames deep.
        // We build on a 64 MB stack, but it is still finite, so refuse — with a
        // catchable error rather than a SIGBUS — any tower whose estimated
        // appetite exceeds 75% of that budget. The per-block figure is
        // deliberately pessimistic (8 KB).
        let estimatedBytesPerBlock = 8 << 10
        let fixedBuildOverheadBytes = 512 << 10
        let estimatedBuildBytes =
            network.arch.numBlocks * estimatedBytesPerBlock + fixedBuildOverheadBytes
        let buildBudgetBytes = graphBuildStackBytes * 3 / 4
        if estimatedBuildBytes > buildBudgetBytes {
            throw ChessTrainerError.towerTooDeepToBuild(
                numBlocks: network.arch.numBlocks,
                estimatedKB: estimatedBuildBytes / 1024,
                limitKB: buildBudgetBytes / 1024
            )
        }

        let graph = network.graph
        let dtype = ChessNetwork.mpsDataType(for: network.arch)

        // --- fp32-accumulation guards for batch reductions ---
        //
        // Under a narrow `dtype` (bf16) every tensor in this graph,
        // including gradients and per-position losses, carries an 8-bit
        // mantissa. A `reductionSum` / `mean` over thousands of such
        // elements accumulates in that same narrow type, and once the
        // running total dwarfs an individual addend by more than half a
        // bf16 ULP the addend is silently dropped — so a sum over the
        // batch (or over a 147K-element weight gradient) loses its tail
        // and comes out biased low. That directly corrupts the global
        // gradient norm that gates clipping, and skews every reported
        // batch-mean loss / health scalar.
        //
        // `widenForReduction` lifts a tensor to fp32 *before* the reduce
        // so the accumulator is fp32; `narrowReductionResult` brings the
        // resulting scalar back to `dtype` so it rejoins the bf16 graph
        // and matches the dtype the host readback path assumes. The
        // matmuls/convs themselves stay bf16 — their hardware
        // accumulators are already fp32, so only these CPU-visible
        // reductions need the guard. Both are identity on `.float32`, so
        // flipping `dtype` back stays byte-identical to the fp32 graph.
        let widenForReduction: (MPSGraphTensor) -> MPSGraphTensor = { t in
            dtype == .float32 ? t : graph.cast(t, to: .float32, name: nil)
        }
        let narrowReductionResult: (MPSGraphTensor, String) -> MPSGraphTensor = { t, name in
            dtype == .float32 ? t : graph.cast(t, to: dtype, name: name)
        }

        // --- Placeholders for training targets ---

        let movePlayed = graph.placeholder(
            shape: [-1],
            dataType: .int32,
            name: "move_played"
        )
        // z / vBaseline / legalMask are fed as fp32 and narrowed to the
        // compute dtype by an in-graph `cast` — the input boundary for
        // these feeds, mirroring the fp32 board input (`board_input_cast`).
        // The host write is then a raw memcpy: no per-element
        // `float32ToBFloat16Bits` loop and no staging buffer (see
        // `feedsForBatch`). The cast is bit-exact against that CPU loop
        // (round-to-nearest-even, ties included — `BF16CastEquivalenceTests`),
        // so every downstream op sees the same bf16 bits it saw before; on a
        // `.float32` build the cast is the identity and is elided. The
        // returned tuple field (the trainer's feed key) is the fp32
        // placeholder `*Feed`; the bare `z` / `vBaseline` / `legalMask`
        // bound below are the cast outputs that the loss graph consumes.
        let zFeed = graph.placeholder(
            shape: [-1, 1],
            dataType: .float32,
            name: "z_outcome"
        )
        let z = (dtype == .float32) ? zFeed : graph.cast(zFeed, to: dtype, name: "z_cast")
        // vBaseline: the value-head's own prediction of this position
        // captured at play time, fed as a placeholder so autodiff can't
        // walk back into the value head from the policy loss. MPSGraph
        // has no stopGradient op, so feeding the baseline in externally
        // is how we get detach semantics. The fp32→cast indirection adds no
        // gradient path into any trainable (the placeholder is a leaf), so
        // the detach property is preserved.
        let vBaselineFeed = graph.placeholder(
            shape: [-1, 1],
            dataType: .float32,
            name: "v_baseline"
        )
        let vBaseline = (dtype == .float32) ? vBaselineFeed : graph.cast(vBaselineFeed, to: dtype, name: "v_baseline_cast")

        let legalMaskFeed = graph.placeholder(
            shape: [-1, NSNumber(value: ChessNetwork.policySize)],
            dataType: .float32,
            name: "legal_move_mask"
        )
        let legalMask = (dtype == .float32) ? legalMaskFeed : graph.cast(legalMaskFeed, to: dtype, name: "legal_move_mask_cast")

        // Build masked logits: illegal positions get a huge negative bias.
        let oneConst = graph.constant(1.0, dataType: dtype)
        let illegalMask = graph.subtraction(oneConst, legalMask, name: "illegal_mask")
        let largeNeg = graph.constant(-1e9, dataType: dtype)
        let additiveMask = graph.multiplication(illegalMask, largeNeg, name: "additive_mask")
        let maskedLogits = graph.addition(network.policyOutput, additiveMask, name: "masked_logits")

        // --- Policy loss: signed-advantage CE with positive + complement targets ---
        //
        //   L = mean(
        //          max(0,  advNorm) · CE(smoothedTarget,    p)
        //        + max(0, -advNorm) · CE(complementTarget,  p) · complementEnable
        //   )
        //
        // Three structural pieces, each motivated by a distinct past
        // failure mode (the first two by the divergence analysed in
        // CHECK_NEXT.md, the third by the *opposite* failure — illegal
        // mass saturating at ~0.997):
        //
        // 1. Label-smoothed target. The CE labels are no longer a hard
        //    one-hot at the played move; they're a smoothed distribution
        //          target = (1 − ε) · oneHot(played) + ε · uniform(legal)
        //    where `uniform(legal)` puts equal mass on every legal cell
        //    and zero on illegal cells. At ε = 0 this is bit-exact the
        //    legacy one-hot. At ε > 0 the loss equilibrium becomes
        //    `p(played) = 1 − ε + ε/|legal|` per position — a finite,
        //    reachable fixed point. With a one-hot target the equilibrium
        //    sits at `p(played) = 1`, which requires `logit(played) = +∞`;
        //    that's the unreachable-attractor that drives `pLogitAbsMax`
        //    into the tens of thousands and the run into divergence. The
        //    smoothed target replaces it with an attainable fixed point
        //    whose gradient self-corrects (passing `p(played) > 1 − ε`
        //    flips the sign of the played-cell gradient).
        //
        // 2. Signed-advantage split with complementary CE on the
        //    negative branch (built further down). The naïve form
        //    `advNorm · −log p(played)` is **unbounded below** when
        //    `advNorm < 0` (as p → 0, log p → −∞, so the whole
        //    expression → −∞ with a non-vanishing gradient on the
        //    played logit — no stopping condition; observed earlier
        //    in `dcm_log_20260509-155952.txt`, pLoss = −64868 before
        //    abort, full analysis in CHECK_NEXT.md). The replacement
        //    drives the negative branch with a *complementary* CE
        //    against a mirrored smoothed target — main mass on the
        //    OTHER legal moves, ε share spread across all legals —
        //    weighted by `max(0, −advNorm)`. Both terms are
        //    non-negative; total loss is bounded below by zero
        //    regardless of advantage sign. Equilibria: positive
        //    branch attracts `p(played) → 1 − ε + ε/|legal|`,
        //    negative branch attracts `p(played) → ε/|legal|`.
        //    The `useSignedAdvantageComplementCE` knob multiplies the
        //    negative branch by a runtime 1.0/0.0 scalar so the legacy
        //    positive-only clamp regime can be re-engaged at runtime
        //    if needed.
        //
        // 3. The CE softmax is over the **raw** policy logits
        //    (`network.policyOutput`), NOT the legal-masked logits.
        //    Commit `acc5340` had fed `maskedLogits` here, reasoning
        //    that masking would stop mass accumulating on illegal cells.
        //    It did the opposite: with `maskedLogits`, the softmax over
        //    illegal cells is ≈0 *by the −1e9 bias*, so
        //    `∂CE/∂(illegal logit) ≈ softmax_masked − target ≈ 0 − 0 = 0`
        //    — the CE became blind to illegal logits and they drifted up
        //    unopposed (the entropy bonus is also masked, and the
        //    softmax-mass `illegalMassPenalty` has gradient ∝ p, which
        //    → 0 once illegal mass ≈ 1, so there was *no* effective
        //    restoring force; `pIllM` parked at ~0.997 for entire runs).
        //    Over the raw logits the legal-only target is still zero on
        //    illegal cells — the CE can never *reward* illegal mass —
        //    but now `∂CE/∂(illegal logit) = softmax_raw(illegal) − 0
        //    = softmax_raw(illegal) ∈ [0, 1]`, a bounded gradient that
        //    always pushes illegal logits down and vanishes only once
        //    illegal mass is genuinely ≈ 0 (the correct fixed point).
        //    `maskedLogits` is still built — the entropy bonus, move
        //    selection, and the legal-cell diagnostics all legitimately
        //    need it — only the CE input changed back.
        //
        // Together: the policy loss is now structurally identical to a
        // supervised classifier with label smoothing on the positive-
        // advantage subset of positions, training over the full (legal +
        // illegal) logit vector. The value head still trains on all
        // positions via its categorical-CE term (built below from
        // `network.valueLogits`, independent of the policy path) — only
        // the policy gradient is gated.
        //
        // The graph still uses MPSGraph's fused `softMaxCrossEntropy`
        // because (a) it has an autodiff rule (manual stable
        // log-softmax with max-subtract would compile but blow up in
        // `gradientForPrimaryTensor` since `reductionMaximum` has no
        // gradient), and (b) it accepts arbitrary label tensors, not
        // just one-hots — passing a smoothed target shape-matches the
        // op without any other changes.

        // Label-smoothing ε — scalar placeholder so it's live-tunable.
        // Declared next to its only consumer (the smoothed-target build
        // below) rather than in the bottom-of-function placeholder
        // block, because we need the value to construct the target
        // before the CE op runs.
        let labelSmoothingEpsilonTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "policy_label_smoothing_epsilon"
        )

        let oneHot = graph.oneHot(
            withIndicesTensor: movePlayed,
            depth: ChessNetwork.policySize,
            axis: 1,
            dataType: dtype,
            onValue: 1.0,
            offValue: 0.0,
            name: "move_onehot"
        )

        // uniform(legal) = legalMask / |legal|, per position.
        //   legalMask has shape [batch, policySize] with 1.0 at legal,
        //   0.0 at illegal cells.
        //   |legal| (per row) is reductionSum over the class axis with
        //   keepdims; clamp at 1.0 to defend against the theoretically-
        //   impossible zero-legal-moves case (terminal positions never
        //   make it into the training buffer, but a divide-by-zero
        //   here would NaN the entire batch).
        let legalCountKeepDims = graph.reductionSum(
            with: legalMask,
            axis: 1,
            name: "legal_count_per_pos"
        )
        let oneFloatForLegal = graph.constant(1.0, dataType: dtype)
        let legalCountSafe = graph.maximum(
            legalCountKeepDims,
            oneFloatForLegal,
            name: "legal_count_safe"
        )
        let uniformOverLegal = graph.division(
            legalMask,
            legalCountSafe,
            name: "uniform_over_legal"
        )
        // smoothed = (1 − ε) · oneHot + ε · uniformOverLegal
        let oneMinusEpsilon = graph.subtraction(
            oneFloatForLegal,
            labelSmoothingEpsilonTensor,
            name: "label_smoothing_one_minus_eps"
        )
        let smoothedTargetOneHotComponent = graph.multiplication(
            oneHot,
            oneMinusEpsilon,
            name: "smoothed_target_onehot_part"
        )
        let smoothedTargetUniformComponent = graph.multiplication(
            uniformOverLegal,
            labelSmoothingEpsilonTensor,
            name: "smoothed_target_uniform_part"
        )
        let smoothedTarget = graph.addition(
            smoothedTargetOneHotComponent,
            smoothedTargetUniformComponent,
            name: "policy_smoothed_target"
        )

        // Complement smoothed target — mirror of `smoothedTarget` with
        // the (1 − ε) "main" mass spread over the OTHER legal moves
        // and the ε "smoothing" mass shared across all legals.
        //   complement = (1 − ε) · uniform(otherLegals) + ε · uniform(legal)
        // where uniform(otherLegals) = (legalMask − oneHot) / max(1, |legal| − 1).
        // Used by the negative-advantage branch (see the signed-split
        // construction further down). Teaches "the played move was bad
        // here; mass should sit on any other legal move." The (1 − ε)
        // numerator is all-zeros when |legal| = 1 (played is the only
        // option), so the main component vanishes and the target reduces
        // to ε · oneHot(played) — a small loss; one-legal positions are
        // a structural edge case (mostly forced replies in check) and
        // not worth a separate masking branch.
        let otherLegalMaskRaw = graph.subtraction(
            legalMask,
            oneHot,
            name: "other_legal_mask"
        )
        let legalCountMinusOne = graph.subtraction(
            legalCountSafe,
            oneFloatForLegal,
            name: "legal_count_minus_one"
        )
        let otherLegalCountSafe = graph.maximum(
            legalCountMinusOne,
            oneFloatForLegal,
            name: "other_legal_count_safe"
        )
        let uniformOverOtherLegal = graph.division(
            otherLegalMaskRaw,
            otherLegalCountSafe,
            name: "uniform_over_other_legal"
        )
        let complementMainComponent = graph.multiplication(
            uniformOverOtherLegal,
            oneMinusEpsilon,
            name: "complement_target_main_part"
        )
        // `smoothedTargetUniformComponent` (= ε · uniformOverLegal) is
        // already built above — the complement target shares it.
        let complementTarget = graph.addition(
            complementMainComponent,
            smoothedTargetUniformComponent,
            name: "policy_complement_target"
        )

        // Raw policy logits, NOT `maskedLogits` — the CE must see the
        // illegal cells in order to drive their softmax mass to zero.
        // See structural piece 3 in the comment block above (and the
        // `acc5340` backfire it describes).
        let ceLossRaw = graph.softMaxCrossEntropy(
            network.policyOutput,
            labels: smoothedTarget,
            axis: 1,
            reuctionType: .none,
            name: "policy_ce_raw"
        )

        // Parallel CE against the complement target — same raw-logits
        // softmax base as the positive-target CE so the illegal-mass
        // pressure stays in effect on both branches. Combined with the
        // positive branch via the signed-advantage split below.
        let ceLossComplementRaw = graph.softMaxCrossEntropy(
            network.policyOutput,
            labels: complementTarget,
            axis: 1,
            reuctionType: .none,
            name: "policy_ce_complement_raw"
        )

        // softMaxCrossEntropy with .none reduces the class axis, leaving
        // one loss per batch element. Reshape to [batch, 1] so it lines up
        // with z for the outcome-weighted multiply.
        let negLogProb = graph.reshape(
            ceLossRaw,
            shape: [-1, 1],
            name: "policy_ce_per_pos"
        )
        let negLogProbComplement = graph.reshape(
            ceLossComplementRaw,
            shape: [-1, 1],
            name: "policy_ce_complement_per_pos"
        )
        // No per-position CE clamp here. Two prior approaches and why
        // they were both rejected, kept for reference because the
        // tradeoffs are non-obvious:
        //
        //   (a) Hard `graph.clamp(CE, upper: log(policySize))` had zero
        //       gradient outside its bounds, so the high-magnitude
        //       `−log p(a*)` terms that should teach the policy to
        //       broaden produced no gradient at all. Removed in the
        //       19d8ab6 commit on this hypothesis.
        //   (b) No clamp at all (post-19d8ab6, pre-this-change). The
        //       reasoning was that the global gNorm clip plus
        //       `softMaxCrossEntropy`'s internal log-sum-exp stability
        //       would bound things. That reasoning was wrong: with
        //       signed-advantage multiplication, the loss is unbounded
        //       below, the gradient on the played logit does not
        //       vanish as `p(played) → 0`, gNorm clip preserves
        //       direction so every clipped step pushes the same way,
        //       and momentum integrates the consistent direction into
        //       a 5+ orders-of-magnitude logit blowup over ~1 hour of
        //       training (observed in dcm_log_20260509-155952.txt;
        //       full analysis in CHECK_NEXT.md).
        //
        // The current fix bounds the loss at a structural level
        // instead — label smoothing (above) gives both CE branches a
        // reachable fixed point, and the signed-advantage split with
        // complementary CE on the negative branch (below) keeps both
        // signs of advantage bounded below by zero without dropping
        // any samples. At convergence the per-position positive CE
        // settles near the positive-target entropy (`H(positive) ≈
        // 0.64 nats` for the current ε=`policyLabelSmoothingEpsilon`
        // default, |legal|≈30) and the complement CE settles near the
        // complement-target entropy, which is *higher* than the
        // positive-target entropy because the (1−ε) main mass spreads
        // over `(|legal|−1)` cells in the complement vs 1 cell in the
        // positive target (e.g. `H(complement) ≈ 3.4 nats` at the same
        // ε default, |legal|≈30). Different equilibrium *magnitudes*, but
        // both equilibria are reachable and bounded — that's what
        // matters for stability; absolute magnitude isn't a divergence
        // signal here.
        // Either can be transiently *large* — up to ≈ log(ChessNetwork.policySize)
        // nats while raw softmax mass is still mostly on illegal
        // cells, the post-`acc5340` recovery regime — but a large CE
        // *value* is not a large *update*: the CE's gradient on every
        // logit (legal or illegal) is `softmax_raw − target ∈ [−1, 1]`,
        // a hard bound that holds regardless of how concentrated the raw
        // softmax is. So no graph-level CE clamp is needed, and feeding
        // raw rather than masked logits adds no divergence vector — if
        // anything it's stabilizing, since the illegal-logit gradient
        // (`= softmax_raw(illegal) ≥ 0`) actively drains the mass that
        // would otherwise inflate `pLogitAbsMax`.
        // --- Advantage baseline: (z − vBaseline) · −log p(a*) ---
        //
        // `vBaseline` is a placeholder fed, each step, with a fresh
        // forward pass of the current trainer network over the batch
        // positions (staged in `trainStepFromReplay` phase 3 — the
        // ReplayBuffer no longer carries a per-position baseline
        // column; the WDL rewrite made the play-time-frozen baseline
        // obsolete, on-disk format v6→v7). Feeding it through a
        // placeholder is the MPSGraph-compatible way to "detach"
        // (MPSGraph has no stopGradient op, verified empirically in
        // the 22:35 CDT gradient-stop experiment: `variableFromTensor`
        // + `read` does not block backward flow), so no gradient flows
        // from the policy loss back through the baseline. The advantage
        // formulation reduces policy-gradient variance by 5–20× per
        // standard REINFORCE-with-baseline literature, with zero bias:
        // the baseline only has to be a function of state, which the
        // network's own value estimate is.
        let advantage = graph.subtraction(z, vBaseline, name: "advantage")
        // Per-batch advantage standardization: `A / RMS(A)`
        // before multiplying into the policy loss. Stabilizes the
        // policy-gradient magnitude batch-to-batch.
        //
        // DEPRECATED: Centering (subtracting E[A]) was found to cause
        // policy collapse in draw-heavy regimes by inverting the
        // signal for draws. We now preserve the absolute sign of the
        // advantage while still dividing by the magnitude to keep
        // step sizes comparable batch-to-batch.
        //
        // REFINED: We use the Root Mean Square (RMS) for normalization
        // rather than Standard Deviation (σ). Standard Deviation
        // subtracts the mean, which acts as a "Bias Amplifier" after
        // a weight rewind (where E[A] is large and σ is small, leading
        // to a massive multiplier). RMS measures total power and
        // correctly scales the update regardless of systematic bias.
        //
        // MPSGraph has no stopGradient — but advantage is a pure
        // function of placeholders (`z`, `vBaseline`), so no gradient
        // path flows through `rms` back into trainable variables.
        // That's what makes this the "safe" standardization:
        // it adjusts the forward value used as the REINFORCE weight,
        // never touches the autograd path.
        //
        // ε of 1e-6 is conservative. We floor the power at 0.04
        // (rms 0.2) via `graph.maximum` so that a homogeneous batch
        // (e.g. all draws) doesn't produce an infinite gradient.
        let advantageSq = graph.square(
            with: advantage,
            name: "advantage_sq"
        )
        let advantageMS = graph.mean(
            of: advantageSq,
            axes: [0, 1],
            name: "advantage_mean_square"
        )
        let advantagePowerFloor = graph.constant(0.04, dataType: dtype)
        let advantagePowerForNorm = graph.maximum(
            advantageMS,
            advantagePowerFloor,
            name: "advantage_power_for_norm"
        )
        let advantageNormEps = graph.constant(1e-6, dataType: dtype)
        let advantageRMSForNorm = graph.squareRoot(
            with: graph.addition(
                advantagePowerForNorm,
                advantageNormEps,
                name: "advantage_rms_sum"
            ),
            name: "advantage_rms_for_norm"
        )
        let advantageNormalized = graph.division(
            advantage,
            advantageRMSForNorm,
            name: "advantage_normalized"
        )
        // Signed-advantage policy gradient with complementary CE on
        // the negative branch.
        //
        //   weightedCE = max(0, advNorm) · positiveCE
        //              + max(0, −advNorm) · complementCE · complementEnable
        //
        // Both products are non-negative by construction, so the total
        // policy loss is bounded below by 0 regardless of advantage
        // sign. Equilibria:
        //   • Positive branch (advNorm > 0): p(played) → (1 − ε) + ε/|legal|
        //     — the legacy smoothed-CE attractor.
        //   • Negative branch (advNorm < 0): p(played) → ε/|legal|
        //     — structural mirror of the positive attractor; the
        //     (1 − ε) main mass redistributes over `(|legal| − 1)`
        //     other legal moves. Per-other-legal target mass therefore
        //     `(1 − ε)/(|legal| − 1) + ε/|legal|`, NOT the same as the
        //     positive branch's per-played mass — the structural form
        //     mirrors but the per-cell magnitudes (and hence the
        //     complement-target entropy) don't.
        //
        // `complementEnable` is a runtime scalar (1.0 / 0.0) so the
        // user can A/B between this signed-split formulation and the
        // legacy positive-only clamp regime without recompiling.
        // When the scalar is 0, the negative branch's contribution
        // collapses to zero and we recover `max(0, advNorm) · positiveCE`
        // bit-exact — the historical "wins teach, losses contribute
        // nothing" form.
        //
        // The diagnostic `policyLossWin` / `policyLossLoss` split
        // below (z-sign masked means of `weightedCE`) now reports the
        // per-sign loss CONTRIBUTION rather than only the positive
        // branch's. With the complement branch live, `policyLossLoss`
        // becomes the rolling negative-A signal — under the legacy
        // clamp-on regime (recoverable via complementEnable = 0) it
        // would be identically zero since loss positions clamp out;
        // with complement-CE active it's the headline "is the network
        // learning from losses and draws as well as wins" diagnostic.
        let zeroForAdvantage = graph.constant(0.0, dataType: dtype)
        let advantageNegated = graph.negative(
            with: advantageNormalized,
            name: "advantage_negated"
        )
        let advantagePositivePart = graph.maximum(
            advantageNormalized,
            zeroForAdvantage,
            name: "advantage_positive_part"
        )
        let advantageNegativePart = graph.maximum(
            advantageNegated,
            zeroForAdvantage,
            name: "advantage_negative_part"
        )
        let complementCEEnableTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "complement_ce_enable"
        )
        let policyTermPositive = graph.multiplication(
            advantagePositivePart,
            negLogProb,
            name: "policy_term_positive"
        )
        let policyTermNegativeUngated = graph.multiplication(
            advantageNegativePart,
            negLogProbComplement,
            name: "policy_term_negative_ungated"
        )
        let policyTermNegative = graph.multiplication(
            policyTermNegativeUngated,
            complementCEEnableTensor,
            name: "policy_term_negative"
        )
        let weightedCE = graph.addition(
            policyTermPositive,
            policyTermNegative,
            name: "adv_weighted_ce"
        )
        let policyLoss = narrowReductionResult(
            graph.mean(
                of: widenForReduction(weightedCE),
                axes: [0, 1],
                name: "policy_loss_f32"
            ),
            "policy_loss"
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

        // --- Value loss: categorical cross-entropy over W/D/L logits ---
        //
        // The value head emits 3 raw logits per position
        // (`network.valueLogits`, shape [batch, 3], slot order
        // [win, draw, loss]). The training target is a one-hot over the
        // play-time outcome z ∈ {+1, 0, −1}, mapped to a slot by
        // `idx = 1 − z` (z=+1→0 win, z=0→1 draw, z=−1→2 loss),
        // optionally label-smoothed by `value_label_smoothing_epsilon`:
        //      target = (1 − ε)·oneHot(1−z) + ε·(1/3)
        // Loss = mean over batch of −Σ_c target_c · logSoftmax(logits)_c
        // (i.e. `softMaxCrossEntropy` with reductionType .none, then mean).
        //
        // Why CE-over-WDL instead of MSE-on-a-tanh-scalar: on a
        // draw-heavy self-play buffer the scalar tanh head collapsed to
        // ≈0 ("everything is a draw") and stopped producing useful
        // gradient — the arena plateau-at-parity in the build-893 run.
        // A 3-way classifier keeps a usable gradient because the draw
        // class is just one of three competing logits. See
        // wdl-value-head.md. `network.valueLogits` is referenced only
        // here (the policy loss never touches it) and this CE never
        // references the policy logits, so the two losses keep disjoint
        // autograd subgraphs joined only at the shared trunk — same as
        // the old MSE term.
        //
        // NOTE on `draw_penalty`: by the time `z` reaches the graph it
        // may have been rewritten from 0.0 to `-drawPenalty` for drawn
        // positions (see `trainStepFromReplay` phase 3). With the
        // default `drawPenalty = 0` that's a no-op. For any
        // `drawPenalty ∈ (0, 1)`, `int(1 − z)` truncates back to slot 1
        // (draw), so the value target here is unchanged — the contempt
        // effect lives entirely in the policy-gradient path, not this
        // CE. Only the full `drawPenalty = 1` lands on slot 2 (loss),
        // relabeling drawn positions as losses; that is the one setting
        // where the value target stops matching the true game result.
        // Documented so it isn't mistaken for a bug.

        // value_label_smoothing_epsilon — scalar placeholder, live-tunable;
        // declared here next to its only consumer (the smoothed target).
        let valueLabelSmoothingEpsilonTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "value_label_smoothing_epsilon"
        )
        // The value loss depends on the head style. The W/D/L softmax head
        // trains with categorical cross-entropy against a smoothed one-hot
        // on the outcome slot; the scalar tanh head trains with MSE between
        // its `tanh` scalar and the outcome `z` directly. Both still feed
        // the same `valueLoss` scalar downstream.
        let valueLoss: MPSGraphTensor
        switch network.arch.valueHeadStyle {
        case .wdlSoftmax:
            // idx = 1 − z. z is [batch, 1] float in {−1, 0, +1} (exact in
            // FP32), so `1 − z ∈ {2, 1, 0}` is exact; casting to int32
            // truncates toward zero, which is identity on those values and
            // maps a `-drawPenalty` rewrite as described above. With the
            // current drawPenalty range ([0, 1]) the rewritten z stays in
            // [-1, 0], so `1 − z ∈ [1, 2]` and the truncated index is
            // always in {1, 2} ⊂ {0, 1, 2} — but clamp to [0, 2] anyway
            // (same defensive stance as the policy path's `max(|legal|, 1)`
            // guard): an out-of-range oneHot index would silently produce
            // an all-zero, gradient-free target row, not an error. oneHot
            // adds the class axis, so reshape the indices to rank-1 first.
            let valueSlotOneFloat = graph.constant(1.0, dataType: dtype)
            let valueSlotIndexFloat = graph.subtraction(valueSlotOneFloat, z, name: "value_slot_index_float")
            let valueSlotIndexLow = graph.constant(0.0, dataType: dtype)
            let valueSlotIndexHigh = graph.constant(Double(network.arch.valueHeadClasses - 1), dataType: dtype)
            let valueSlotIndexClamped = graph.minimum(
                graph.maximum(valueSlotIndexFloat, valueSlotIndexLow, name: "value_slot_index_lo"),
                valueSlotIndexHigh,
                name: "value_slot_index_clamped"
            )
            let valueSlotIndexInt = graph.cast(valueSlotIndexClamped, to: .int32, name: "value_slot_index")
            let valueSlotIndexFlat = graph.reshape(valueSlotIndexInt, shape: [-1], name: "value_slot_index_flat")
            let valueOneHot = graph.oneHot(
                withIndicesTensor: valueSlotIndexFlat,
                depth: 3,
                axis: 1,
                dataType: dtype,
                onValue: 1.0,
                offValue: 0.0,
                name: "value_onehot"
            )
            // smoothed = (1 − ε)·oneHot + ε·(1/3). At ε = 0 this is
            // bit-exact the hard one-hot.
            let valueOneMinusEps = graph.subtraction(
                valueSlotOneFloat,
                valueLabelSmoothingEpsilonTensor,
                name: "value_label_smoothing_one_minus_eps"
            )
            let valueUniformConst = graph.constant(1.0 / 3.0, shape: [1, 3], dataType: dtype)
            let valueSmoothedTarget = graph.addition(
                graph.multiplication(valueOneHot, valueOneMinusEps, name: "value_smoothed_target_onehot_part"),
                graph.multiplication(valueUniformConst, valueLabelSmoothingEpsilonTensor, name: "value_smoothed_target_uniform_part"),
                name: "value_smoothed_target"
            )
            // softMaxCrossEntropy has an autodiff rule and accepts an
            // arbitrary (here: smoothed) label tensor — same reasoning as
            // the policy CE above.
            let valueCEPerPos = graph.softMaxCrossEntropy(
                network.valueLogits,
                labels: valueSmoothedTarget,
                axis: 1,
                reuctionType: .none,
                name: "value_ce_raw"
            )
            // .none reduces the class axis → one loss per batch element
            // ([batch]); reshape to [batch, 1] so the mean lines up with
            // the rest of the scalar reductions.
            let valueCEPerPosReshaped = graph.reshape(valueCEPerPos, shape: [-1, 1], name: "value_ce_per_pos")
            valueLoss = narrowReductionResult(
                graph.mean(of: widenForReduction(valueCEPerPosReshaped), axes: [0, 1], name: "value_loss_f32"),
                "value_loss"
            )

        case .scalarTanh:
            // MSE between the tanh value scalar (`network.valueOutput`, which
            // is the raw tanh for this head style) and the outcome target z.
            // Label smoothing is a W/D/L-distribution concept and does not
            // apply here, so `valueLabelSmoothingEpsilonTensor` is unused on
            // this path (still created + fed so the graph/feed shape is
            // stable across head styles).
            let valueDiff = graph.subtraction(network.valueOutput, z, name: "value_tanh_diff")
            let valueSq = graph.multiplication(valueDiff, valueDiff, name: "value_tanh_sq")
            valueLoss = narrowReductionResult(
                graph.mean(of: widenForReduction(valueSq), axes: [0, 1], name: "value_loss_f32"),
                "value_loss"
            )
        }

        // --- Value-head output diagnostics ---
        //
        // `network.valueOutput` is now the derived scalar p_win − p_loss
        // (no tanh). `valueMean` near 0 = healthy on an early-training,
        // draw-heavy buffer (most positions ≈ draw → p_win ≈ p_loss).
        // `valueAbsMean` near 1 = the head confidently classifies
        // win-or-loss everywhere. The W/D/L probability means
        // (`valueProbWin/Draw/Loss` = batch means of the three softmax
        // columns of `network.valueProbs`) are the direct collapse
        // probe: `valueProbDraw → 1.0` is the new representation's
        // "everything is a draw" — the exact failure the WDL head is
        // meant to break out of, watch it like the old `vAbs`. All are
        // diagnostic-only `targetTensors`, never on the totalLoss
        // autograd path.
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
        // W/D/L probability means exist only for the 3-column softmax head;
        // the scalar tanh head's `valueProbs` is a single column, so slicing
        // columns 1/2 would be out of bounds. Report zeros there so the
        // stats readback slots stay well-formed (the UI's W/D/L row simply
        // reads flat for a tanh net).
        let valueProbWin: MPSGraphTensor
        let valueProbDraw: MPSGraphTensor
        let valueProbLoss: MPSGraphTensor
        switch network.arch.valueHeadStyle {
        case .wdlSoftmax:
            let valueProbWinCol = graph.sliceTensor(network.valueProbs, dimension: 1, start: 0, length: 1, name: "value_prob_win_col")
            let valueProbDrawCol = graph.sliceTensor(network.valueProbs, dimension: 1, start: 1, length: 1, name: "value_prob_draw_col")
            let valueProbLossCol = graph.sliceTensor(network.valueProbs, dimension: 1, start: 2, length: 1, name: "value_prob_loss_col")
            valueProbWin = graph.mean(of: valueProbWinCol, axes: [0, 1], name: "value_prob_win")
            valueProbDraw = graph.mean(of: valueProbDrawCol, axes: [0, 1], name: "value_prob_draw")
            valueProbLoss = graph.mean(of: valueProbLossCol, axes: [0, 1], name: "value_prob_loss")
        case .scalarTanh:
            // No W/D/L distribution for a 1-logit tanh head; report zeros. These
            // MUST be three DISTINCT graph tensors: the diagnostic readback keys
            // its results dict by tensor, and an anonymous `graph.constant(0)`
            // can be interned by MPSGraph into a single shared tensor — which
            // would duplicate-key the dictionary and trap on every stats step.
            // Three separate multiplication ops (real tensor × 0) are guaranteed
            // distinct tensor objects that all evaluate to 0.
            let zeroScale = graph.constant(0.0, dataType: dtype)
            valueProbWin = graph.multiplication(valueMean, zeroScale, name: "value_prob_win_inactive")
            valueProbDraw = graph.multiplication(valueAbsMean, zeroScale, name: "value_prob_draw_inactive")
            valueProbLoss = graph.multiplication(valueMean, zeroScale, name: "value_prob_loss_inactive")
        }

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
        //
        // NOTE: We use MASKED logits for the entropy bonus. This
        // encourages exploration within the set of legal moves without
        // perversely rewarding mass placement on illegal moves (the
        // prior unmasked form acted as an attractor for the ~4834
        // illegal cells). This is the *opposite* of the policy CE,
        // which is over the RAW logits (structural piece 3 above): the
        // CE has a legal-only *target* so it can never reward illegal
        // mass and it *must* see the raw logits to push them down; the
        // entropy bonus has no target — it just rewards spreading — so
        // over raw logits it would pay to spread onto the illegal cells.
        // CE → raw, entropy bonus → masked.
        let softmaxLegal = graph.softMax(
            with: maskedLogits,
            axis: 1,
            name: "policy_softmax_legal"
        )
        let logEpsConst = graph.constant(1e-7, dataType: dtype)
        let softmaxClampedLegal = graph.addition(
            softmaxLegal,
            logEpsConst,
            name: "policy_softmax_legal_clamped"
        )
        let logSoftmaxLegal = graph.logarithm(
            with: softmaxClampedLegal,
            name: "policy_log_softmax_legal"
        )
        let pLogPLegal = graph.multiplication(
            softmaxLegal,
            logSoftmaxLegal,
            name: "p_log_p_legal"
        )
        // Accumulate the per-position entropy (a sum over 4864 p·log p
        // terms) and the batch mean in fp32 — under bf16 the small tail
        // terms would otherwise be lost. Narrow the final scalar back to
        // `dtype` so it rejoins the bf16 graph (it feeds `total_loss` via
        // the entropy regularizer and is read back as `pEnt`).
        let negEntropyPerPos = graph.reductionSum(
            with: widenForReduction(pLogPLegal),
            axis: 1,
            name: "neg_entropy_per_pos_f32"
        )
        let entropyPerPos = graph.negative(
            with: negEntropyPerPos,
            name: "entropy_per_pos_f32"
        )
        let policyEntropy = narrowReductionResult(
            graph.mean(
                of: entropyPerPos,
                axes: [0, 1],
                name: "policy_entropy_f32"
            ),
            "policy_entropy"
        )

        // --- Illegal mass penalty ---
        //
        // Directly penalizes probability mass that leaks past the mask.
        // Unlike the entropy bonus, this has a stable attractor at
        // 100% legal mass. Minimizing total loss minimizes this term.
        let unmaskedSoftmax = graph.softMax(
            with: network.policyOutput,
            axis: 1,
            name: "policy_softmax_unmasked"
        )
        // fp32-accumulate the per-position illegal-mass sum (over 4864
        // classes) and the batch mean; this term joins `total_loss` and
        // is the `[STATS]` illegal-mass signal, so its tail must survive
        // the bf16 narrowing. Narrow the final scalar back to `dtype`.
        let illegalMassPerPos = graph.reductionSum(
            with: widenForReduction(graph.multiplication(
                unmaskedSoftmax,
                illegalMask,
                name: "policy_illegal_mass_per_pos_masked"
            )),
            axis: 1,
            name: "policy_illegal_mass_per_pos_f32"
        )
        let illegalMassPenalty = narrowReductionResult(
            graph.mean(
                of: illegalMassPerPos,
                axes: [0, 1],
                name: "illegal_mass_penalty_f32"
            ),
            "illegal_mass_penalty"
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
            softmaxLegal,
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
        // `advantage_normalized = advantage / RMS(advantage)` preserves the
        // sign of `advantage`, so positions with `z > vBaseline` pull
        // `p(a*)` up and positions with `z < vBaseline` pull it down.
        // Whether the batch's unconditional `p(a*)` mean trends up,
        // down, or stays flat depends on the win/draw/loss mix and the
        // value-head calibration — it can sit near `1/policySize` even
        // when learning is healthy, especially in the draw-heavy regime
        // where most positions have `z ≈ 0`. We keep it as a coarse
        // index-mismatch probe (both conditionals flat near `1/policySize`
        // is strong evidence of action-index misalignment) and emit two
        // **advantage-sign-conditional** means as the real direction-of-
        // learning signal: `playedMoveProbPosAdv` (conditioned on
        // `advantage > 0`) should rise and `playedMoveProbNegAdv`
        // (conditioned on `advantage < 0`) should fall as training
        // progresses. The divergence between the two is the health
        // signal, not the raw mean.
        let playedSoftmaxMasked = graph.multiplication(
            softmaxLegal,
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
        // picked as a small fixed cutoff — positions
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
        // the `policyLossWeight` placeholder so the user can dial it
        // down if the amplified policy gradient is the source of
        // gradient-clip saturation.
        // The four optimizer-update scalars (lr, weight decay, grad-clip
        // max-norm, momentum below) are declared **fp32 regardless of
        // `dtype`**. They are consumed only by the optimizer update, which
        // runs in fp32 in the canonical bf16 path (fp32 master weights — see
        // the SGD/EMA construction below). Feeding them as bf16 and casting
        // up would only recover the bf16-narrowed value (e.g. lr=1e-3 →
        // 0.0009766); an fp32 placeholder fed the raw `Float` keeps them
        // exact. Their feed NDArrays are sized fp32 in `init` and
        // `buildFeeds` writes the raw `Float`. Under `dataType == .float32`
        // this matches the rest of the graph; the loss-side scalars stay
        // `dtype` (they shape the bf16 loss, whose gradient is bf16 anyway).
        let lrTensor = graph.placeholder(
            shape: [1],
            dataType: .float32,
            name: "learning_rate"
        )
        let entropyCoeffTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "entropy_regularization_coeff"
        )
        let weightDecayTensor = graph.placeholder(
            shape: [1],
            dataType: .float32,
            name: "weight_decay_coeff"
        )
        let gradClipMaxNormTensor = graph.placeholder(
            shape: [1],
            dataType: .float32,
            name: "grad_clip_max_norm"
        )
        let policyLossWeightTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "policy_loss_weight"
        )
        let valueLossWeightTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "value_loss_weight"
        )
        let illegalMassWeightTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "illegal_mass_weight"
        )
        // Polyak momentum coefficient μ. μ=0 reduces the velocity term
        // to zero (μ·v_old = 0), so the update collapses to plain
        // SGD-with-decoupled-decay bit-exact. μ=0.9 still amplifies
        // the gradient term by ~1/(1−μ) = 10× in steady state under
        // correlated gradients — pair with a proportional LR drop or
        // expect collapse. Decay is decoupled from velocity so
        // changing μ does NOT amplify weight decay. Live-tunable via
        // the placeholder so the user can dial it without rebuilding
        // the graph.
        let momentumTensor = graph.placeholder(
            shape: [1],
            dataType: .float32,
            name: "momentum_coeff"
        )
        let weightedPolicy = graph.multiplication(
            policyLossWeightTensor,
            policyLoss,
            name: "weighted_policy_loss"
        )
        let weightedValue = graph.multiplication(
            valueLossWeightTensor,
            valueLoss,
            name: "weighted_value_loss"
        )
        let entropyPenalty = graph.multiplication(
            entropyCoeffTensor,
            policyEntropy,
            name: "entropy_regularization_term"
        )
        let illegalPenalty = graph.multiplication(
            illegalMassWeightTensor,
            illegalMassPenalty,
            name: "illegal_mass_penalty_term"
        )
        let lossWithoutRegularization = graph.addition(
            weightedValue,
            weightedPolicy,
            name: "loss_without_regularization"
        )
        let lossWithEntropy = graph.subtraction(
            lossWithoutRegularization,
            entropyPenalty,
            name: "loss_minus_entropy"
        )
        let totalLossTensor = graph.addition(
            lossWithEntropy,
            illegalPenalty,
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
            // Square + sum-of-squares in fp32: a single conv gradient is
            // ~147K elements, and bf16 accumulation drops every addend
            // that falls below half an ULP of the running total — biasing
            // the global norm low and weakening the clip it gates.
            let flat = graph.reshape(grad, shape: [-1], name: nil)
            let sq = graph.square(with: widenForReduction(flat), name: nil)
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
        // Narrow back to `dtype` so the norm rejoins the bf16 clip math
        // (`maximum`/`division` with the bf16 `gradClipMaxNorm`) and the
        // host readback, which both assume `the net's compute dtype`. The
        // fp32 accumulation above is what mattered; the final scalar's
        // bf16 rounding is negligible against a clip threshold.
        // Keep the fp32 norm for the clip math (the clip scalars are fp32),
        // and narrow a separate copy to `dtype` only for the host readback
        // (`readFloats` assumes `the net's compute dtype`).
        let gradGlobalNormF32 = graph.squareRoot(
            with: gradSumOfSquaresTensor,
            name: "grad_global_norm_f32"
        )
        let gradGlobalNorm = narrowReductionResult(gradGlobalNormF32, "grad_global_norm")

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
        // fp32-accumulate the sum-of-squares (same bf16-tail rationale as
        // the gradient norm) and narrow the scalar back to `dtype` for the
        // host readback.
        let policyWeightSq = graph.square(
            with: widenForReduction(policyWeightFlat),
            name: "policy_weight_sq"
        )
        let policyWeightSqSum = graph.reductionSum(
            with: policyWeightSq,
            axis: 0,
            name: "policy_weight_sq_sum"
        )
        let policyHeadWeightNormTensor = narrowReductionResult(
            graph.squareRoot(
                with: policyWeightSqSum,
                name: "policy_weight_norm_f32"
            ),
            "policy_weight_norm"
        )

        // --- Gradient clip scale: maxNorm / max(norm, maxNorm) ---
        //
        // Equivalent to `min(1, maxNorm / norm)`. When `norm ≤ maxNorm`
        // the scale is 1 (no-op); above the threshold the scale
        // shrinks so the resulting update has L2 norm exactly
        // `maxNorm`. No epsilon needed because `max(norm, maxNorm)`
        // is always ≥ maxNorm > 0.
        // Computed in fp32 (both operands fp32): the fp32 grad norm and the
        // fp32 `gradClipMaxNorm` scalar. `clipScale` is therefore fp32 and is
        // applied to the fp32 gradient in the update loop below.
        let clipDenom = graph.maximum(
            gradGlobalNormF32,
            gradClipMaxNormTensor,
            name: "grad_clip_denom"
        )
        let clipScale = graph.division(
            gradClipMaxNormTensor,
            clipDenom,
            name: "grad_clip_scale"
        )

        // --- SGD updates with decoupled weight decay, clipped gradients, Polyak momentum ---
        //
        // v_new       = μ · v_old + clipped_grad
        // weight_new  = weight − lr · v_new − lr · decayC · weight        (if shouldDecay)
        // weight_new  = weight − lr · v_new                                (otherwise)
        //
        // Decoupled weight decay (Loshchilov & Hutter 2017, the "AdamW"
        // paper). Decay is applied directly to the weight at update
        // time, NOT folded into the gradient before momentum. This
        // means the effective decay strength does NOT scale with
        // 1/(1−μ) the way the legacy PyTorch-default coupled L2 form
        // did — μ and decayC tune independently. At μ = 0 this
        // reduces bit-exact to plain SGD with weight decay (since
        // both forms collapse to the same `lr · (grad + decayC · weight)`
        // expression when the velocity term zeros out).
        //
        // Decay is applied only to variables flagged in
        // `network.trainableShouldDecay` — conv and FC weight matrices
        // — and skipped for BN gamma/beta and biases per the standard
        // PyTorch / AdamW recipe.
        //
        // Velocity variables are allocated here as MPSGraph variables on
        // the same graph as the trainable weights, with shape matching
        // each weight and zero-initial values. They are mutable state —
        // assigned every step. Persisted across save/load via
        // `exportTrainerWeights()` / `loadTrainerWeights(_:)`
        // (file format `ModelCheckpointFile` v2). On promotion the
        // velocities are zeroed via `resetVelocitiesToZero()` because
        // the trainer's weights are entirely replaced, so the
        // previously-accumulated velocity vector points against a
        // weight surface that no longer exists.
        //
        // The learning rate, weight decay, and momentum are all scalar
        // placeholders (not constants), so they can be changed between
        // steps without rebuilding the graph. Each training step feeds
        // the current values via the pre-allocated NDArrays.

        // Optimizer state is fp32 (the canonical mixed-precision path). Under
        // a reduced-precision `dtype` (bf16) the working weights the forward
        // graph multiplies stay bf16, but the optimizer keeps an fp32 *master*
        // of every persistent tensor and accumulates updates into it, so a
        // per-step step below a bf16 ULP isn't rounded away; the bf16 working
        // copy is re-derived each step as `cast(master)`. Under `.float32`
        // there is no separate master (working weights are the master) and
        // this collapses to the prior plain path.
        // Config D (`network.bf16CastActive`): the persistent weight/stat
        // variables are stored fp32 (the variable IS the master; the forward
        // casts each to bf16 at point of use). The optimizer therefore runs
        // exactly the fp32 path — SGD on the fp32 variable directly, fp32
        // velocity, NO master, NO working-sync — so `useMaster` is forced
        // off even though `dtype` is bf16.
        let useMaster = (dtype != .float32) && !network.bf16CastActive
        // fp32 zero-init bytes for an fp32 variable of `count` elements.
        func fp32Zeros(_ count: Int) -> Data { Data(count: count * MemoryLayout<Float>.size) }

        // Velocity buffers — always fp32 (momentum accumulates gradients; bf16
        // here would suffer the same ULP loss as the weight update). One per
        // trainable, parallel to `network.trainableVariables`.
        var velocities: [MPSGraphTensor] = []
        velocities.reserveCapacity(network.trainableVariables.count)
        // Per-velocity load infrastructure built in lockstep with
        // velocity allocation so the index spaces align.
        var velLoadPlaceholders: [MPSGraphTensor] = []
        var velLoadAssignOps: [MPSGraphOperation] = []
        var velLoadNDArrays: [MPSNDArray] = []
        var velLoadTensorData: [MPSGraphTensorData] = []
        velLoadPlaceholders.reserveCapacity(network.trainableVariables.count)
        velLoadAssignOps.reserveCapacity(network.trainableVariables.count)
        velLoadNDArrays.reserveCapacity(network.trainableVariables.count)
        velLoadTensorData.reserveCapacity(network.trainableVariables.count)
        for (i, variable) in network.trainableVariables.enumerated() {
            guard let shape = variable.shape else {
                throw ChessTrainerError.gradientMissing(
                    "trainable[\(i)] has no static shape; cannot allocate velocity"
                )
            }
            let elementCount = shape.reduce(1) { $0 * $1.intValue }
            let varName = variable.operation.name.isEmpty
                ? "trainable_\(i)"
                : variable.operation.name
            let velocity = graph.variable(
                with: fp32Zeros(elementCount),
                shape: shape,
                dataType: .float32,
                name: "\(varName)_velocity"
            )
            velocities.append(velocity)

            // Build the velocity-load placeholder + assign op for this
            // variable. Used by `loadTrainerWeights` and
            // `resetVelocitiesToZero` to overwrite the velocity buffer
            // out-of-band from the SGD update path.
            let loadPh = graph.placeholder(
                shape: shape,
                dataType: .float32,
                name: "\(varName)_velocity_load"
            )
            let loadAssign = graph.assign(velocity, tensor: loadPh, name: "\(varName)_velocity_load_assign")
            velLoadPlaceholders.append(loadPh)
            velLoadAssignOps.append(loadAssign)

            let desc = MPSNDArrayDescriptor(dataType: .float32, shape: shape)
            let nda = MPSNDArray(device: network.metalDevice, descriptor: desc)
            velLoadNDArrays.append(nda)
            velLoadTensorData.append(MPSGraphTensorData(nda))
        }

        // fp32 master weights + master running stats (only under reduced
        // precision). Built parallel to the network's full persistent state —
        // `trainableVariables` first, then `bnRunningStatsVariables` — so the
        // index space matches `exportTrainerWeights`'s base ordering and one
        // load-infra set covers both. Indices [0, nTrainable) are SGD-updated
        // weight masters; [nTrainable, …) are EMA-updated running-stat
        // masters. `syncMastersOps` seeds each master from its current working
        // value (`master = cast(working, fp32)`); run once at construction and
        // after any wholesale weight replacement.
        let allPersistent = network.trainableVariables + network.bnRunningStatsVariables
        var masterVariables: [MPSGraphTensor] = []
        var masterLoadPlaceholders: [MPSGraphTensor] = []
        var masterLoadAssignOps: [MPSGraphOperation] = []
        var masterLoadNDArrays: [MPSNDArray] = []
        var masterLoadTensorData: [MPSGraphTensorData] = []
        var syncMastersOps: [MPSGraphOperation] = []
        if useMaster {
            masterVariables.reserveCapacity(allPersistent.count)
            for (i, working) in allPersistent.enumerated() {
                guard let shape = working.shape else {
                    throw ChessTrainerError.gradientMissing(
                        "persistent[\(i)] has no static shape; cannot allocate fp32 master"
                    )
                }
                let elementCount = shape.reduce(1) { $0 * $1.intValue }
                let baseName = working.operation.name.isEmpty ? "persistent_\(i)" : working.operation.name
                let master = graph.variable(
                    with: fp32Zeros(elementCount),
                    shape: shape,
                    dataType: .float32,
                    name: "\(baseName)_master"
                )
                masterVariables.append(master)
                // master <- cast(working, fp32): seeds the master from the
                // working copy at init / after a weight replacement.
                let synced = graph.cast(working, to: .float32, name: nil)
                syncMastersOps.append(graph.assign(master, tensor: synced, name: "\(baseName)_master_sync"))
                // fp32 load infra for restoring a persisted master.
                let loadPh = graph.placeholder(shape: shape, dataType: .float32, name: "\(baseName)_master_load")
                masterLoadPlaceholders.append(loadPh)
                masterLoadAssignOps.append(graph.assign(master, tensor: loadPh, name: "\(baseName)_master_load_assign"))
                let mdesc = MPSNDArrayDescriptor(dataType: .float32, shape: shape)
                let mnda = MPSNDArray(device: network.metalDevice, descriptor: mdesc)
                masterLoadNDArrays.append(mnda)
                masterLoadTensorData.append(MPSGraphTensorData(mnda))
            }
        }

        var ops: [MPSGraphOperation] = []
        // Two assigns per variable now (velocity + weight) plus the BN
        // running-stats appended below.
        ops.reserveCapacity(network.trainableVariables.count * 2)
        precondition(
            network.trainableShouldDecay.count == network.trainableVariables.count,
            "ChessNetwork.trainableShouldDecay must align 1:1 with trainableVariables"
        )
        // Diagnostic: global L2 norm of the post-step velocity buffer
        // ‖v_new‖. Built up as the sum of per-velocity squared sums
        // alongside the SGD update, so MPSGraph fuses it into the same
        // training pass without a second graph.run. Reported on the
        // [STATS] line so the user can see velocity magnitude growth
        // directly when raising μ.
        var velSumOfSquares: MPSGraphTensor?
        for (i, variable) in network.trainableVariables.enumerated() {
            guard let grad = grads[variable] else {
                // Already checked in the norm-accumulation loop above,
                // but re-guard here so a future refactor that splits
                // the two loops can't silently drop a variable.
                throw ChessTrainerError.gradientMissing(
                    variable.operation.name.isEmpty ? "trainable[\(i)]" : variable.operation.name
                )
            }
            let velocity = velocities[i]
            // All optimizer math runs in fp32. Under bf16 the gradient is
            // bf16; cast it up. `clipScale` / `lrTensor` / `momentumTensor` /
            // `weightDecayTensor` are fp32 already (see above).
            let gradF = useMaster ? graph.cast(grad, to: .float32, name: nil) : grad
            // Apply the global L2 clip scale to this gradient.
            let clippedGrad = graph.multiplication(gradF, clipScale, name: nil)
            // Polyak momentum: v_new = μ · v_old + clippedGrad.
            // Weight decay does NOT enter the velocity (decoupled form).
            let scaledOldVelocity = graph.multiplication(velocity, momentumTensor, name: nil)
            let newVelocity = graph.addition(scaledOldVelocity, clippedGrad, name: nil)
            let velocityAssign = graph.assign(velocity, tensor: newVelocity, name: nil)
            ops.append(velocityAssign)
            // Accumulate ‖v_new‖² for the global velocity-norm diagnostic.
            // Uses the symbolic newVelocity (matching the reasoning at the
            // weight-update site below: assigning a variable does not
            // invalidate value-typed references to the assigned tensor).
            // newVelocity is already fp32 (velocity is fp32), so no widen.
            let velFlat = graph.reshape(newVelocity, shape: [-1], name: nil)
            let velSq = graph.square(with: velFlat, name: nil)
            let velScalar = graph.reductionSum(with: velSq, axis: 0, name: nil)
            if let accum = velSumOfSquares {
                velSumOfSquares = graph.addition(accum, velScalar, name: nil)
            } else {
                velSumOfSquares = velScalar
            }
            // Update target: the fp32 master under bf16, else the (fp32)
            // weight variable itself. The step (lr · v_new + decoupled decay)
            // is computed in fp32 and accumulated into it; under bf16 the
            // working weight is then re-synced as cast(master).
            let optWeight = useMaster ? masterVariables[i] : variable
            let momentumStep = graph.multiplication(lrTensor, newVelocity, name: nil)
            let totalStep: MPSGraphTensor
            if network.trainableShouldDecay[i] {
                let decayScaled = graph.multiplication(
                    optWeight,
                    weightDecayTensor,
                    name: nil
                )
                let decayStep = graph.multiplication(lrTensor, decayScaled, name: nil)
                totalStep = graph.addition(momentumStep, decayStep, name: nil)
            } else {
                totalStep = momentumStep
            }
            let updated = graph.subtraction(optWeight, totalStep, name: nil)
            ops.append(graph.assign(optWeight, tensor: updated, name: nil))
            if useMaster && !splitTrainableWorkingSync {
                // Re-derive the bf16 working weight the forward graph reads.
                // Skipped under the split experiment: the working sync runs as a
                // separate pass (`workingSyncOps`) after the master update, so
                // this fused dual-write executable isn't built.
                ops.append(graph.assign(variable, tensor: graph.cast(updated, to: dtype, name: nil), name: nil))
            }
        }

        // BN running-stat EMA. Under bf16 the trainer owns it in fp32 — an
        // fp32 master running stat per BN mean/var, EMA-accumulated from the
        // freshly-computed batch stats (`network.bnBatchMean/VarTensors`,
        // available in training mode), then re-synced to the bf16 working
        // running-stat variable the inference-mode normalize / champions read.
        // The network's own bf16 EMA assigns are skipped in this mode. Under
        // `.float32` we use the network's EMA assigns directly (already fp32).
        if useMaster {
            let nTrainable = network.trainableVariables.count
            let layerCount = network.bnBatchMeanTensors.count
            precondition(
                network.bnRunningStatsVariables.count == layerCount * 2,
                "bnRunningStatsVariables must be mean,var-interleaved (2 per BN layer)"
            )
            // EMA: running = 0.99 · running + 0.01 · batch (matches
            // ChessNetwork.batchNorm), accumulated in fp32.
            let emaMomentum = graph.constant(0.99, dataType: .float32)
            let emaComplement = graph.constant(0.01, dataType: .float32)
            for layer in 0..<layerCount {
                let batchStat = [network.bnBatchMeanTensors[layer], network.bnBatchVarTensors[layer]]
                for half in 0..<2 {
                    let runIdx = layer * 2 + half
                    let workingStat = network.bnRunningStatsVariables[runIdx]
                    let statMaster = masterVariables[nTrainable + runIdx]
                    let batchF = graph.cast(batchStat[half], to: .float32, name: nil)
                    let scaledOld = graph.multiplication(statMaster, emaMomentum, name: nil)
                    let scaledNew = graph.multiplication(batchF, emaComplement, name: nil)
                    let updatedStat = graph.addition(scaledOld, scaledNew, name: nil)
                    ops.append(graph.assign(statMaster, tensor: updatedStat, name: nil))
                    if !splitTrainableWorkingSync {
                        // Fused bf16 working-stat sync. Skipped under the split:
                        // the bn working stat is re-derived from its master in the
                        // separate `workingSyncOps` pass (see `buildWorkingSyncOps`),
                        // same as the trainable weights — its fused dual-write is the
                        // same macOS-27 stomp pattern.
                        ops.append(graph.assign(workingStat, tensor: graph.cast(updatedStat, to: dtype, name: nil), name: nil))
                    }
                }
            }
        } else {
            ops.append(contentsOf: network.bnRunningStatsAssignOps)
        }

        // Finalize the velocity-norm scalar. Same precondition as
        // gradGlobalNorm: trainableVariables is non-empty so the
        // accumulator is always populated.
        guard let velSumOfSquaresTensor = velSumOfSquares else {
            throw ChessTrainerError.gradientMissing(
                "(no trainable variables for velocity-norm)"
            )
        }
        let velocityGlobalNormTensor = narrowReductionResult(
            graph.squareRoot(
                with: velSumOfSquaresTensor,
                name: "velocity_global_norm_f32"
            ),
            "velocity_global_norm"
        )

        return (
            movePlayed, zFeed, vBaselineFeed, legalMaskFeed,
            lrTensor, entropyCoeffTensor, weightDecayTensor, gradClipMaxNormTensor, policyLossWeightTensor,
            valueLossWeightTensor,
            illegalMassWeightTensor,
            labelSmoothingEpsilonTensor,
            valueLabelSmoothingEpsilonTensor,
            momentumTensor,
            complementCEEnableTensor,
            velocities,
            velLoadPlaceholders, velLoadAssignOps, velLoadNDArrays, velLoadTensorData,
            masterVariables, masterLoadPlaceholders, masterLoadAssignOps, masterLoadNDArrays, masterLoadTensorData, syncMastersOps,
            totalLossTensor, policyLoss, valueLoss,
            policyEntropy, illegalMassPenalty, policyNonNegCount, policyNonNegIllegalCount, gradGlobalNorm, valueMean, valueAbsMean,
            valueProbWin, valueProbDraw, valueProbLoss,
            policyHeadWeightNormTensor,
            policyLogitAbsMax, playedMoveProbTensor,
            playedMoveProbPosAdvTensor, playedMoveProbNegAdvTensor,
            advantageMeanTensor, advantageStdTensor, advantageMinTensor, advantageMaxTensor,
            advantageFracPosTensor, advantageFracSmallTensor,
            advantage,
            policyLossWin, policyLossLoss,
            velocityGlobalNormTensor,
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
        let dispatchedAt = CFAbsoluteTimeGetCurrent()
        return try await enqueue {
            let queueWaitMs = (CFAbsoluteTimeGetCurrent() - dispatchedAt) * 1000
            return try self.internalTrainStep(batchSize: batchSize, queueWaitMs: queueWaitMs)
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

    /// Off-main async getter for `completedTrainSteps`. The lock read
    /// runs on a global executor so the awaiter (typically the main
    /// actor) is never synchronously blocked.
    func asyncCompletedTrainSteps() async -> Int {
        let start = Date()
        return await withCheckedContinuation { (cont: CheckedContinuation<Int, Never>) in
            let inContinuation = Date()
            DispatchQueue.global(qos: .userInitiated).async {
                let dispatched = Date()
                let result = self._completedTrainSteps.value
                let now = Date()
                let total = now.timeIntervalSince(start)
                if total > 0.05 {
                    let pre = inContinuation.timeIntervalSince(start)
                    let queue = dispatched.timeIntervalSince(inContinuation)
                    let work = now.timeIntervalSince(dispatched)
                    print(String(format: "[DISPATCH-LATENCY] asyncCompletedTrainSteps: total=%.2fms (pre=%.2fms queue=%.2fms work=%.2fms)", total*1000, pre*1000, queue*1000, work*1000))
                }
                cont.resume(returning: result)
            }
        }
    }

    /// Off-main async variant of `effectiveLearningRate(forBatchSize:completedSteps:)`.
    /// The (potential) lock read runs on a global executor so the
    /// awaiter is never synchronously blocked.
    func asyncEffectiveLearningRate(forBatchSize batchSize: Int, completedSteps: Int? = nil) async -> Float {
        let start = Date()
        return await withCheckedContinuation { (cont: CheckedContinuation<Float, Never>) in
            let inContinuation = Date()
            DispatchQueue.global(qos: .userInitiated).async {
                let dispatched = Date()
                let result = self.effectiveLearningRate(
                    forBatchSize: batchSize,
                    completedSteps: completedSteps
                )
                let now = Date()
                let total = now.timeIntervalSince(start)
                if total > 0.05 {
                    let pre = inContinuation.timeIntervalSince(start)
                    let queue = dispatched.timeIntervalSince(inContinuation)
                    let work = now.timeIntervalSince(dispatched)
                    print(String(format: "[DISPATCH-LATENCY] asyncEffectiveLearningRate: total=%.2fms (pre=%.2fms queue=%.2fms work=%.2fms)", total*1000, pre*1000, queue*1000, work*1000))
                }
                cont.resume(returning: result)
            }
        }
    }

    /// Effective learning rate that the optimizer is currently being
    /// fed, given the active warmup multiplier and (optionally) the
    /// sqrt-batch scaling rule. Mirrors the in-graph math at
    /// `buildFeeds` step time so a status-bar readout matches what the
    /// training step is actually applying. When `completedSteps` is
    /// nil the function reads the step count from the `SyncBox` (still
    /// not from `executionQueue`, so a status-bar readout never blocks
    /// on an in-flight SGD step). Pass an explicit value when the
    /// caller already has a snapshot of the step count and wants both
    /// the count and the LR to reflect the same observation — e.g.
    /// the UI heartbeat that publishes `TrainerWarmupSnapshot`, where
    /// reading the SyncBox twice would otherwise let the count and LR
    /// disagree by one training step.
    func effectiveLearningRate(forBatchSize batchSize: Int, completedSteps: Int? = nil) -> Float {
        let steps = completedSteps ?? _completedTrainSteps.value
        let warmupMul: Float
        if lrWarmupSteps > 0 {
            warmupMul = Float(min(1.0, Double(steps) / Double(lrWarmupSteps)))
        } else {
            warmupMul = 1.0
        }
        // Base LR: the cycle's geometric value when LR cycling is active,
        // otherwise the static `learningRate`. Identical resolution to
        // `buildFeeds` so this readout matches the LR the SGD step actually
        // applies — sqrt-batch scaling and warmup compose on top.
        let baseLR: Float = _lrMomentumCycle.value.learningRate(forStep: steps).map { Float($0) } ?? learningRate
        var lr: Float
        if sqrtBatchScalingForLR {
            let sqrtBatchScale: Float = Float(
                sqrt(Double(batchSize) / Double(Self.sqrtScaleBaseBatchSize))
            )
            lr = baseLR * sqrtBatchScale
        } else {
            lr = baseLR
        }
        return lr * warmupMul
    }

    /// Effective Polyak momentum the optimizer is currently being fed:
    /// the cycle's linear value when momentum cycling is active, otherwise
    /// the static `momentumCoeff`. Mirrors the `buildFeeds` resolution so a
    /// status-bar readout matches what the SGD step applies. Like
    /// `effectiveLearningRate`, reads the step count from the `SyncBox`
    /// (never `executionQueue`), so a UI readout never blocks on an
    /// in-flight step; pass `completedSteps` to pin it to the same
    /// observation as a co-published LR.
    func effectiveMomentum(completedSteps: Int? = nil) -> Float {
        let steps = completedSteps ?? _completedTrainSteps.value
        return _lrMomentumCycle.value.momentum(forStep: steps).map { Float($0) } ?? momentumCoeff
    }

    private func internalTrainStep(batchSize: Int, queueWaitMs: Double = 0) throws -> TrainStepTiming {
        let totalStart = CFAbsoluteTimeGetCurrent()

        // --- Data prep: synthesize random boards, moves, outcomes ---

        let prepStart = CFAbsoluteTimeGetCurrent()
        let floatsPerBoard = arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
        let totalBoardFloats = batchSize * floatsPerBoard
        let totalMaskFloats = batchSize * ChessNetwork.policySize

        // Reuse trainer-owned synthetic scratch across sweep calls.
        // Reallocate only when batchSize changes.
        if syntheticBoards.count != totalBoardFloats {
            syntheticBoards = [Float](repeating: 0, count: totalBoardFloats)
        }
        if syntheticMoves.count != batchSize {
            syntheticMoves = [Int32](repeating: 0, count: batchSize)
        }
        if syntheticZs.count != batchSize {
            syntheticZs = [Float](repeating: 0, count: batchSize)
        }
        if syntheticVBaselines.count != batchSize {
            // vBaselines: all zeros for the random-data sweep. An
            // all-zero baseline degrades the advantage formulation to
            // `z * negLogProb`, which is exactly what the random-data
            // smoke test measured historically — so losses stay
            // comparable to prior sweep runs.
            syntheticVBaselines = [Float](repeating: 0, count: batchSize)
        }
        if syntheticLegalMasks.count != totalMaskFloats {
            // All-ones (no masking) for the synthetic-data path; the
            // additive mask term then evaluates to 0 everywhere and
            // the graph behaves identically to the pre-masking
            // version. The real per-position legal mask is built in
            // `trainStepFromReplay`.
            syntheticLegalMasks = [Float](repeating: 1.0, count: totalMaskFloats)
        }
        Self.fillRandomFloats(&syntheticBoards)
        // Random move indices in [0, policySize). One per batch row.
        for i in 0..<batchSize {
            syntheticMoves[i] = Int32.random(in: 0..<Int32(ChessNetwork.policySize))
        }
        // Random outcomes from {-1, 0, +1} so the loss includes all three
        // signed regimes (push up, push down, no contribution).
        for i in 0..<batchSize {
            syntheticZs[i] = Float(Int.random(in: 0..<3) - 1)
        }

        // Unbox the cached arrays into raw pointers and feed
        // them through the shared pointer-based `buildFeeds` /
        // `runPreparedStep` pipeline.
        return try syntheticBoards.withUnsafeBufferPointer { boardsBuf in
            try syntheticMoves.withUnsafeBufferPointer { movesBuf in
                try syntheticZs.withUnsafeBufferPointer { zsBuf in
                    try syntheticVBaselines.withUnsafeBufferPointer { vBaseBuf in
                        try syntheticLegalMasks.withUnsafeBufferPointer { legalMasksBuf in
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
                            let feeds = buildFeeds(BatchFeedsInput(
                                batchSize: batchSize,
                                boards: boardsBase,
                                moves: movesBase,
                                zs: zsBase,
                                vBaselines: vBaseBase,
                                legalMasks: legalMasksBase
                            ))
                            let prepMs = (CFAbsoluteTimeGetCurrent() - prepStart) * 1000
                            return try runPreparedStep(
                                feeds: feeds,
                                prepMs: prepMs,
                                queueWaitMs: queueWaitMs,
                                totalStart: totalStart,
                                batchSize: batchSize,
                                // Random-data sweep: keep the full readback so
                                // its measured step matches historical sweeps.
                                includeDiagnostics: true
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
    /// in the batch. Those fresh v values fill the trainer's
    /// `vBaseline` staging buffer, so the policy-gradient advantage
    /// `(z - vBaseline)` reflects the trainer's current belief. (The
    /// replay buffer no longer stores any play-time baseline — the
    /// W/D/L value-head rewrite made it dead, since the derived value
    /// scalar is now recomputed here every step.)
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
        // hop into the network's evaluate.
        struct Phase1: Sendable {
            let boardsCopy: [Float]
            let isStatsStep: Bool
            let includeDiagnostics: Bool
        }
        let phase1: Phase1? = try await enqueue { [batchSize] in
            let phase1Start = CFAbsoluteTimeGetCurrent()
            self.ensureReplayBatchCapacity(batchSize)
            guard
                let boards = self.replayBatchBoards,
                let moves = self.replayBatchMoves,
                let zs = self.replayBatchZs,
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
            // Graph diagnostics are gated separately from [BATCH-STATS]: they
            // coincide with stats steps when `batchStatsInterval` is set, but
            // fall back to a fixed cadence when it's 0 so the [STATS] line and
            // the entropy/draw-collapse alarms never lose their inputs.
            let diagnosticsInterval = interval > 0 ? interval : Self.diagnosticsFallbackInterval
            let includeDiagnostics = nextStep % diagnosticsInterval == 0
            let didSample = replayBuffer.sample(
                count: batchSize,
                intoBoards: boards,
                moves: moves,
                zs: zs,
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

                // Surface composition-constraint deviations: stratum
                // clamps on the draw cap (in either direction), length
                // targets below the shortest resident game, and the
                // attempt-budget fallback. Gated on `isStatsStep` so log
                // volume tracks `batch_stats_interval`; only fires when
                // there's something to report.
                let sr = replayBuffer.lastSamplingResult()
                if sr.wasDegraded {
                    let reqD = String(format: "%.1f", sr.requestedDrawPercent)
                    let gotD = String(format: "%.1f", sr.achievedDrawPercent)
                    let mlen = String(format: "%.1f", sr.achievedMeanGameLength)
                    let infeasible = sr.lengthTargetInfeasible ? "Y" : "N"
                    let budget = sr.attemptBudgetHit ? "Y" : "N"
                    let line = "[SAMPLER] step=\(nextStep) batch=\(sr.batchSize)"
                        + " req=(K=\(sr.constraints.maxPerGame) D=\(reqD)% T=\(sr.constraints.targetMeanGameLengthPlies))"
                        + " got=(D=\(gotD)% maxG=\(sr.achievedMaxPerGame) mlen=\(mlen))"
                        + " flags=(infeasible_target=\(infeasible) shortest_resident=\(sr.shortestResidentLength) budget_hit=\(budget))"
                    SessionLogger.shared.log(line)
                }
            }
            let floatsPerBoard = self.arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
            let totalFloats = batchSize * floatsPerBoard
            let boardsCopy = Array(UnsafeBufferPointer(start: boards, count: totalFloats))
            self.phase1WallTimesMs.append((CFAbsoluteTimeGetCurrent() - phase1Start) * 1000)
            return Phase1(boardsCopy: boardsCopy, isStatsStep: isStatsStep, includeDiagnostics: includeDiagnostics)
        }
        guard let phase1 else { return nil }

        // Phase 2 (network queue, async): forward-only pass on the
        // trainer's network to compute v(s) for every position. We
        // discard the policy output and keep only the value scalars.
        let freshBaselineStart = CFAbsoluteTimeGetCurrent()
        // GPU→GPU baseline handoff: run a value-only forward on the trainer's
        // current network and KEEP the per-position v(s) in a network-owned GPU
        // buffer, handed back here without a CPU readback. Phase 3 feeds that
        // buffer straight into the training step's vBaseline placeholder (see
        // `runPreparedStep`), eliminating the old `Array(valuesBuf)` copy +
        // staging re-write. Numerically identical to the old path — same v(s),
        // just no CPU round-trip. `nonisolated(unsafe)` mirrors the prior
        // `freshValues` pattern: the await fully completes the forward (its `run`
        // waits) before phase 3 reads the buffer, so the single assignment is
        // safe; it is then boxed (`VBaselineHandoff`) to cross the enqueue hop.
        nonisolated(unsafe) var vBaselineResult: MPSGraphTensorData? = nil
        try await network.computeValueBaselineGPU(
            batchBoards: phase1.boardsCopy,
            count: batchSize
        ) { td in
            vBaselineResult = td
        }
        let freshBaselineMs = (CFAbsoluteTimeGetCurrent() - freshBaselineStart) * 1000
        guard let vBaselineResult else {
            preconditionFailure("ChessTrainer phase 2: value-baseline forward did not yield a result buffer")
        }
        let vBaselineHandoff = VBaselineHandoff(tensorData: vBaselineResult)

        // Phase 3 (trainer queue): apply draw penalty, build feeds, run
        // the training graph. The vBaseline is no longer staged from the
        // host — it rides in `vBaselineHandoff.tensorData`, the GPU buffer
        // phase 2's value-only forward wrote, bound directly to the
        // vBaseline placeholder in `runPreparedStep` (GPU→GPU handoff).
        let dispatchedAtPhase3 = CFAbsoluteTimeGetCurrent()
        let isStatsStep = phase1.isStatsStep
        let includeDiagnostics = phase1.includeDiagnostics
        return try await enqueue { [batchSize, vBaselineHandoff, freshBaselineMs, dispatchedAtPhase3, isStatsStep, includeDiagnostics] in
            let phase3Start = CFAbsoluteTimeGetCurrent()
            let phase3QueueWaitMs = (phase3Start - dispatchedAtPhase3) * 1000
            let totalStart = phase3Start
            let prepStart = phase3Start

            guard
                let boards = self.replayBatchBoards,
                let moves = self.replayBatchMoves,
                let zs = self.replayBatchZs,
                let masks = self.replayBatchLegalMasks
            else {
                preconditionFailure("ChessTrainer staging buffers vanished between phases")
            }

            // The fresh per-position v(s) lives in `vBaselineHandoff.tensorData`
            // (a GPU buffer produced by phase 2's value-only forward) and is
            // bound straight into the training step below — no CPU staging copy.

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
            let floatsPerBoard = self.arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize

            // Zero the entire mask buffer first — cheaper than zeroing per-row inside
            // the loop, and the legal-move generator will overwrite the legal indices.
            let totalMaskFloats = batchSize * policySize
            masks.update(repeating: 0.0, count: totalMaskFloats)

            // Time the legal-mask construction loop (decode-board +
            // MoveGenerator.legalMoves + scatter) to inform the Part 2
            // decision on whether to cache legal-move indices in the
            // replay buffer. Aggregated across the current stats window
            // and emitted as `[LEGAL-COST]` on each `isStatsStep`.
            let legalMaskLoopStart = CFAbsoluteTimeGetCurrent()
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
            self.legalMaskLoopMsTimes.append((CFAbsoluteTimeGetCurrent() - legalMaskLoopStart) * 1000)

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

            let feeds = self.buildFeeds(BatchFeedsInput(
                batchSize: batchSize,
                boards: UnsafePointer(boards),
                moves: UnsafePointer(moves),
                zs: UnsafePointer(zs),
                vBaselines: nil,   // fed via GPU→GPU handoff; bound in runPreparedStep
                legalMasks: UnsafePointer(masks)
            ))
            let prepMs = (CFAbsoluteTimeGetCurrent() - prepStart) * 1000

            // Run the training step. The returned timing has nil
            // fresh-baseline fields; we patch them in below.
            let baseTiming = try self.runPreparedStep(
                feeds: feeds,
                prepMs: prepMs,
                queueWaitMs: phase3QueueWaitMs,
                totalStart: totalStart,
                batchSize: batchSize,
                vBaselineOverride: vBaselineHandoff.tensorData,
                // Diagnostic graph reductions on the diagnostics cadence
                // (== stats steps when batchStatsInterval > 0; a fixed
                // fallback when it's 0) — see GPU_UTILIZATION_PLAN.md (Phase 1).
                includeDiagnostics: includeDiagnostics
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

            // Accumulate phase timings for the current stats window.
            let completionTime = CFAbsoluteTimeGetCurrent()
            self.phase2WallTimesMs.append(freshBaselineMs)
            self.phase3WallTimesMs.append((completionTime - phase3Start) * 1000)
            // Inter-step delta: wall time from the previous step's
            // completion to this step's completion. Skips the first
            // step (no prior reference) so the accumulator only holds
            // genuine inter-step gaps. Window is reset alongside the
            // phase timings on each isStatsStep emit below.
            if self.lastTrainStepCompletedAt > 0 {
                self.interStepWallTimesMs.append(
                    (completionTime - self.lastTrainStepCompletedAt) * 1000
                )
            }
            self.lastTrainStepCompletedAt = completionTime

            // On every isStatsStep, emit one [LEGAL-COST] line with
            // P50/P99 of each accumulator and clear them. Gates the
            // future replay-buffer legal-mask caching decision.
            if isStatsStep {
                let p1Count = self.phase1WallTimesMs.count
                let p2Count = self.phase2WallTimesMs.count
                let p3Count = self.phase3WallTimesMs.count
                let lmCount = self.legalMaskLoopMsTimes.count
                let isCount = self.interStepWallTimesMs.count
                let p1p50 = Self.percentile(self.phase1WallTimesMs, 0.50)
                let p1p99 = Self.percentile(self.phase1WallTimesMs, 0.99)
                let p2p50 = Self.percentile(self.phase2WallTimesMs, 0.50)
                let p2p99 = Self.percentile(self.phase2WallTimesMs, 0.99)
                let p3p50 = Self.percentile(self.phase3WallTimesMs, 0.50)
                let p3p99 = Self.percentile(self.phase3WallTimesMs, 0.99)
                let lmp50 = Self.percentile(self.legalMaskLoopMsTimes, 0.50)
                let lmp99 = Self.percentile(self.legalMaskLoopMsTimes, 0.99)
                let isp50 = Self.percentile(self.interStepWallTimesMs, 0.50)
                let isp99 = Self.percentile(self.interStepWallTimesMs, 0.99)
                // Wall-time accounting: interStep is the end-to-end
                // measurement (step completion → next step completion).
                // `phaseSum = p1+p2+p3`. `gap = interStep − phaseSum`
                // surfaces dispatch latency / await overhead / replay-
                // ratio sleeps — i.e. everything we'd miss if we only
                // summed the phases. Computed at P50 for the headline
                // number; consumers can inspect the raw distributions
                // above for tail behavior.
                let phaseSumP50 = p1p50 + p2p50 + p3p50
                let gapP50 = isp50.isFinite ? isp50 - phaseSumP50 : .nan
                let line = "[LEGAL-COST]"
                    + " step=\(self._completedTrainSteps.value)"
                    + " batch=\(batchSize)"
                    + " window=(p1=\(p1Count) p2=\(p2Count) p3=\(p3Count) lm=\(lmCount) is=\(isCount))"
                    + String(format: " p1ms=(p50=%.2f p99=%.2f)", p1p50, p1p99)
                    + String(format: " p2ms=(p50=%.2f p99=%.2f)", p2p50, p2p99)
                    + String(format: " p3ms=(p50=%.2f p99=%.2f)", p3p50, p3p99)
                    + String(format: " legalMaskMs=(p50=%.2f p99=%.2f)", lmp50, lmp99)
                    + String(format: " legalMaskPerPosUs=(p50=%.1f p99=%.1f)",
                        (lmp50 / Double(batchSize)) * 1000.0,
                        (lmp99 / Double(batchSize)) * 1000.0)
                    + String(format: " interStepMs=(p50=%.2f p99=%.2f)", isp50, isp99)
                    + String(format: " gapMs=(p50=%.2f)", gapP50)
                SessionLogger.shared.log(line)
                self.phase1WallTimesMs.removeAll(keepingCapacity: true)
                self.phase2WallTimesMs.removeAll(keepingCapacity: true)
                self.phase3WallTimesMs.removeAll(keepingCapacity: true)
                self.legalMaskLoopMsTimes.removeAll(keepingCapacity: true)
                self.interStepWallTimesMs.removeAll(keepingCapacity: true)
            }

            return TrainStepTiming(
                dataPrepMs: baseTiming.dataPrepMs,
                gpuRunMs: baseTiming.gpuRunMs,
                readbackMs: baseTiming.readbackMs,
                queueWaitMs: baseTiming.queueWaitMs,
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
                illegalMassPenalty: baseTiming.illegalMassPenalty,
                policyNonNegligibleCount: baseTiming.policyNonNegligibleCount,
                policyNonNegligibleIllegalCount: baseTiming.policyNonNegligibleIllegalCount,
                gradGlobalNorm: baseTiming.gradGlobalNorm,
                valueMean: baseTiming.valueMean,
                valueAbsMean: baseTiming.valueAbsMean,
                valueProbWin: baseTiming.valueProbWin,
                valueProbDraw: baseTiming.valueProbDraw,
                valueProbLoss: baseTiming.valueProbLoss,
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
                policyLossLoss: baseTiming.policyLossLoss,
                velocityNorm: baseTiming.velocityNorm,
                hasDiagnostics: baseTiming.hasDiagnostics
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
        // need the boards — moves/zs are ignored for this probe.
        struct Sampled: Sendable {
            let boards: [Float]
            let count: Int
        }
        let sampled: Sampled? = try await enqueue { [sampleSize] in
            // Probe-private scratch — deliberately NOT the trainStep staging
            // buffers (`replayBatchBoards/Moves/Zs`). A probe and an in-flight
            // `trainStep` share this serial queue, but `trainStep` frees the
            // queue during its Phase-2 cross-queue await; sampling into the
            // shared staging in that window would overwrite the live batch that
            // Phase 3 then trains against a now-stale vBaseline (wrong
            // advantages). Dedicated buffers, freed on return, eliminate it.
            // `Array(...)` copies the data out before the `defer` deallocates
            // (defer runs after the return value is built), so the copy is safe.
            let floatsPerBoard = self.arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
            let total = sampleSize * floatsPerBoard
            let boards = UnsafeMutablePointer<Float>.allocate(capacity: total)
            let moves = UnsafeMutablePointer<Int32>.allocate(capacity: sampleSize)
            let zs = UnsafeMutablePointer<Float>.allocate(capacity: sampleSize)
            defer { boards.deallocate(); moves.deallocate(); zs.deallocate() }
            let ok = replayBuffer.sample(
                count: sampleSize,
                intoBoards: boards,
                moves: moves,
                zs: zs
            )
            guard ok else { return nil }
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
        // `nonisolated(unsafe)` so the `var` can be mutated from
        // inside the `@Sendable` consume closure. Safe because the
        // await suspends this task for the closure window.
        nonisolated(unsafe) var policy: [Float] = []
        if let inferenceNetwork {
            let weights = try await network.exportWeights()
            try await inferenceNetwork.loadWeights(weights)
            try await inferenceNetwork.evaluateBatched(
                batchBoards: sampled.boards,
                count: sampled.count
            ) { policyBuf, _, _ in
                policy = Array(policyBuf)
            }
        } else {
            try await network.evaluateBatched(
                batchBoards: sampled.boards,
                count: sampled.count
            ) { policyBuf, _, _ in
                policy = Array(policyBuf)
            }
        }

        let floatsPerBoard = arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
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

        // CPU decode + softmax + mask on legal indices.
        // Pre-allocated `expScratch` (policySize floats) is reused per
        // position to hold the exp(logit - max) values, fed to vDSP
        // for the sum/legal-sum and to the entropy loop for legal-cell
        // probabilities. Max/argmax via `vDSP_maxvi`, exp via
        // `vvexpf`, sums via `vDSP_sve`.
        var expScratch = [Float](repeating: 0, count: policySize)
        sampled.boards.withUnsafeBufferPointer { boardsBuf in
            guard let boardsBase = boardsBuf.baseAddress else { return }
            policy.withUnsafeBufferPointer { policyBuf in
                guard let policyBase = policyBuf.baseAddress else { return }
                expScratch.withUnsafeMutableBufferPointer { eBuf in
                    guard let eBase = eBuf.baseAddress else { return }
                    let policyLen = vDSP_Length(policySize)
                    for pos in 0..<sampled.count {
                        let boardPtr = boardsBase.advanced(by: pos * floatsPerBoard)
                        let state = BoardEncoder.decodeSynthetic(from: boardPtr)
                        let legalMoves = MoveGenerator.legalMoves(for: state)
                        guard !legalMoves.isEmpty else { continue }
                        positionsWithLegal += 1

                        let policyRow = policyBase.advanced(by: pos * policySize)

                        // Max + argmax in one pass.
                        var maxLogit: Float = 0
                        var argmaxIdx: vDSP_Length = 0
                        vDSP_maxvi(policyRow, 1, &maxLogit, &argmaxIdx, policyLen)
                        let argmax = Int(argmaxIdx)

                        // expScratch[i] = exp(policyRow[i] - maxLogit)
                        var negMax = -maxLogit
                        vDSP_vsadd(policyRow, 1, &negMax, eBase, 1, policyLen)
                        var expCount = Int32(policySize)
                        vvexpf(eBase, eBase, &expCount)

                        var expSumF: Float = 0
                        vDSP_sve(eBase, 1, &expSumF, policyLen)
                        let expSum: Double = Double(expSumF)

                        // Sum softmax mass over the legal set, using
                        // already-computed exp values in expScratch.
                        var legalExpSum: Double = 0
                        var legalIndexSet = Set<Int>()
                        legalIndexSet.reserveCapacity(legalMoves.count)
                        for move in legalMoves {
                            let idx = PolicyEncoding.policyIndex(move, currentPlayer: .white)
                            guard idx >= 0, idx < policySize else { continue }
                            if legalIndexSet.insert(idx).inserted {
                                legalExpSum += Double(eBase[idx])
                            }
                        }
                        let legalMass = expSum > 0 ? legalExpSum / expSum : 0
                        legalMassSum += legalMass
                        if legalIndexSet.contains(argmax) {
                            top1LegalCount += 1
                        }

                        // Legal-masked Shannon entropy. Renormalize
                        // the legal-only softmax mass to sum to 1 by
                        // dividing each legal-cell exp by
                        // `legalExpSum`, then compute -Σ p · log p in
                        // nats. Skip when legalExpSum is zero (network
                        // has no probability on any legal cell — rare
                        // numerical edge case).
                        if legalExpSum > 0 {
                            var ent: Double = 0
                            for idx in legalIndexSet {
                                let pUn = Double(eBase[idx])
                                let p = pUn / legalExpSum
                                if p > 0 { ent -= p * log(p) }
                            }
                            legalEntropySum += ent
                        }
                    }
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
            ptr.deinitialize(count: replayBatchCapacity * arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize)
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

        let floatsPerBoard = arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
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

    // MARK: - Trainer State Persistence (weights + BN running stats + momentum velocity)
    //
    // The trainer's persistent state for a session save is:
    //   - all `network.trainableVariables` weights
    //   - all `network.bnRunningStatsVariables` running stats
    //   - all `velocityVariables` (one per trainable, momentum velocity)
    //
    // Saved as a single flat `[[Float]]` of length
    //   trainables.count + bnRunningStats.count + velocityVariables.count
    // ordered exactly as listed above. The file format
    // (`ModelCheckpointFile` v2) records this as the same flat tensor
    // list — no schema change beyond version bump.
    //
    // Thread-safety: callers MUST have paused both selfPlayGate and
    // trainingGate before invoking these methods. They drive
    // `network.graph.run` directly (bypassing the trainer's
    // executionQueue) and rely on no concurrent SGD step or self-play
    // evaluator touching the same variables.

    /// Total expected weight count for a v2 trainer state.
    private var trainerWeightCountV2: Int {
        network.trainableVariables.count + network.bnRunningStatsVariables.count + velocityVariables.count
    }

    /// Total expected weight count for base network state: trainables
    /// plus BN running stats, with no optimizer velocity.
    private var trainerWeightCountV1: Int {
        network.trainableVariables.count + network.bnRunningStatsVariables.count
    }

    /// Read all trainer state (weights + bn + velocities) into a flat
    /// `[[Float]]` array suitable for `ModelCheckpointFile.weights`.
    /// Caller MUST have paused training (and ideally self-play) before
    /// calling. See class comment for thread-safety contract.
    func exportTrainerWeights() async throws -> [[Float]] {
        // Base portion: under the canonical mixed-precision path emit the
        // fp32 *masters* (full precision — that's the whole point of
        // persisting them) in place of the bf16 working weights; they're
        // parallel to `trainableVariables + bnRunningStatsVariables`, so the
        // count and on-disk layout are identical to the bf16-native export.
        // Under `.float32` (no masters) fall back to the working weights.
        let baseWeights: [[Float]]
        if masterVariables.isEmpty {
            baseWeights = try await network.exportWeights()
        } else {
            baseWeights = try await readMasterValues()
        }
        // Velocities via a separate small graph.run on the same graph.
        // No race because the caller has paused training.
        let velocityWeights = try await readVelocityValues()
        return baseWeights + velocityWeights
    }

    /// Load exact trainer state (weights + bn + velocities) from a
    /// flat `[[Float]]` array previously produced by
    /// `exportTrainerWeights()`.
    /// Caller MUST have paused training before calling.
    func loadTrainerWeights(_ weights: [[Float]]) async throws {
        let v1Count = trainerWeightCountV1
        let v2Count = trainerWeightCountV2
        guard weights.count == v2Count else {
            throw ChessTrainerError.trainerWeightCountMismatch(
                expected: "\(v2Count) (full trainer state: \(v1Count) base + \(velocityVariables.count) velocity)",
                got: weights.count
            )
        }

        let baseWeights = Array(weights.prefix(v1Count))
        let velocityWeights = Array(weights.suffix(velocityVariables.count))
        // The base portion is the fp32 masters under the mixed-precision path.
        // Restore them, then load the bf16 working weights as the rounded
        // masters (`loadWeights` narrows fp32→bf16), so the working/master
        // invariant holds without a separate master→working sync. Under
        // `.float32` (no masters) this is just the working-weight load.
        if !masterVariables.isEmpty {
            try await writeMasterValues(baseWeights)
        }
        try await network.loadWeights(baseWeights)
        try await writeVelocityValues(velocityWeights)
    }

    /// Initialize a trainer from base/champion network weights and
    /// intentionally discard optimizer velocity. This is for fresh
    /// forks and explicit reset-from-champion flows only; session
    /// resume must use `loadTrainerWeights(_:)` so missing velocity
    /// fails loudly instead of being invented as zero.
    func loadBaseWeightsResetVelocity(_ weights: [[Float]]) async throws {
        let expected = trainerWeightCountV1
        guard weights.count == expected else {
            throw ChessTrainerError.trainerWeightCountMismatch(
                expected: "\(expected) (base network state)",
                got: weights.count
            )
        }
        try await network.loadWeights(weights)
        // Seed the fp32 masters directly from the loaded values, not by
        // re-deriving from the bf16-rounded working copy — lossless, so any
        // fp32 precision in `weights` survives (for a bf16 champion fork the
        // input is already bf16-aligned, so it's identical). The working copy
        // is the bf16-rounded `weights`; the master is `weights` exactly.
        // No-op under `.float32` (no masters).
        if !masterVariables.isEmpty {
            try await writeMasterValues(weights)
        }
        try await resetVelocitiesToZero()
    }

    /// Overwrite all velocity buffers with zeros. Retained as an
    /// escape hatch for callers that explicitly want to discard
    /// momentum (e.g. tests). Promotion no longer uses this path —
    /// it now snapshots velocity at arena-start and restores that
    /// snapshot on promotion (`exportVelocitySnapshot()` /
    /// `loadVelocitySnapshot(_:)`), which keeps the optimizer's
    /// accumulated gradient signal aligned with the validated
    /// candidate weights instead of throwing it away.
    func resetVelocitiesToZero() async throws {
        // Build a zeros payload sized per velocity tensor.
        var zeroed: [[Float]] = []
        zeroed.reserveCapacity(velocityVariables.count)
        for v in velocityVariables {
            let count = try ChessNetwork.elementCount(of: v)
            zeroed.append([Float](repeating: 0, count: count))
        }
        try await writeVelocityValues(zeroed)
    }

    /// Read the optimizer's per-trainable velocity buffers into a flat
    /// `[[Float]]` (one sub-array per trainable variable, parallel to
    /// `network.trainableVariables`). Used at arena start to snapshot
    /// the velocity that built the candidate weights, then restored
    /// on promotion via `loadVelocitySnapshot(_:)`. Caller must have
    /// paused training (the readback drives `network.graph.run`
    /// directly and races against concurrent SGD steps).
    func exportVelocitySnapshot() async throws -> [[Float]] {
        try await readVelocityValues()
    }

    /// Overwrite all velocity buffers from a previously-captured
    /// snapshot produced by `exportVelocitySnapshot()`. Caller must
    /// have paused training. Throws if the snapshot's per-tensor
    /// element counts don't match the current trainer's velocity
    /// shapes — protects against loading a snapshot taken from a
    /// trainer with a different architecture.
    func loadVelocitySnapshot(_ snapshot: [[Float]]) async throws {
        try await writeVelocityValues(snapshot)
    }

    /// Read current velocity values via a single graph.run targeting
    /// all velocity variables. Internal helper for `exportTrainerWeights`.
    /// Runs synchronously on the network's command queue (caller must
    /// have paused training).
    private func readVelocityValues() async throws -> [[Float]] {
        guard !velocityVariables.isEmpty else { return [] }
        return try await withCheckedThrowingContinuation { continuation in
            executionQueue.async { [self] in
                do {
                    let result: [[Float]] = try autoreleasepool {
                        let results = network.graph.run(
                            with: network.commandQueue,
                            feeds: [network.inputPlaceholder: network.dummyInferenceInputTensorData],
                            targetTensors: velocityVariables,
                            targetOperations: nil
                        )
                        var out: [[Float]] = []
                        out.reserveCapacity(velocityVariables.count)
                        for v in velocityVariables {
                            guard let data = results[v] else {
                                throw ChessTrainerError.velocityReadbackMissing(v.operation.name)
                            }
                            let count = try ChessNetwork.elementCount(of: v)
                            // Velocity is fp32 (canonical mixed-precision path).
                            out.append(ChessNetwork.readFloatsFP32(from: data, count: count))
                        }
                        return out
                    }
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Write velocity values via a single graph.run that drives all
    /// velocity-load assign ops. Internal helper for both
    /// `loadTrainerWeights` and `resetVelocitiesToZero`. Caller must
    /// have paused training.
    private func writeVelocityValues(_ weights: [[Float]]) async throws {
        guard weights.count == velocityVariables.count else {
            throw ChessTrainerError.trainerWeightCountMismatch(
                expected: "\(velocityVariables.count) velocity tensors",
                got: weights.count
            )
        }
        // Validate sizes up-front so a mismatch fails before we start
        // writing into shared NDArrays.
        for (i, v) in velocityVariables.enumerated() {
            let expected = try ChessNetwork.elementCount(of: v)
            guard weights[i].count == expected else {
                throw ChessTrainerError.trainerWeightCountMismatch(
                    expected: "velocity[\(i)] (\(v.operation.name)): \(expected) floats",
                    got: weights[i].count
                )
            }
        }
        return try await withCheckedThrowingContinuation { continuation in
            executionQueue.async { [self] in
                do {
                    try autoreleasepool {
                        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
                            network.inputPlaceholder: network.dummyInferenceInputTensorData
                        ]
                        feeds.reserveCapacity(velocityVariables.count + 1)
                        for i in 0..<velocityVariables.count {
                            // Velocity is fp32 (canonical mixed-precision path).
                            ChessNetwork.writeFloatsFP32(weights[i], into: velocityLoadNDArrays[i])
                            feeds[velocityLoadPlaceholders[i]] = velocityLoadTensorData[i]
                        }
                        // graph.run requires at least one target tensor.
                        // Use the first velocity variable as a dummy read —
                        // its post-assign value is whatever we just wrote,
                        // which we discard. A missing entry in the result
                        // dict is MPSGraph's CPU-side signal of a failed
                        // GPU run; surface it as a thrown error so a
                        // poisoned velocity load can't pass silently.
                        let results = network.graph.run(
                            with: network.commandQueue,
                            feeds: feeds,
                            targetTensors: [velocityVariables[0]],
                            targetOperations: velocityLoadAssignOps
                        )
                        guard results[velocityVariables[0]] != nil else {
                            throw ChessTrainerError.velocityLoadGraphFailed(
                                velocityVariables[0].operation.name
                            )
                        }
                    }
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Read the fp32 master values (weights + running stats), parallel to
    /// `trainableVariables + bnRunningStatsVariables`. Empty under `.float32`
    /// (no masters). Caller must have paused training.
    /// Internal (not private) so the macOS-27 NaN-isolation tests can compare
    /// master-vs-working weight norms to localize the bf16 divergence.
    func readMasterValues() async throws -> [[Float]] {
        guard !masterVariables.isEmpty else { return [] }
        return try await withCheckedThrowingContinuation { continuation in
            executionQueue.async { [self] in
                do {
                    let result: [[Float]] = try autoreleasepool {
                        let results = network.graph.run(
                            with: network.commandQueue,
                            feeds: [network.inputPlaceholder: network.dummyInferenceInputTensorData],
                            targetTensors: masterVariables,
                            targetOperations: nil
                        )
                        var out: [[Float]] = []
                        out.reserveCapacity(masterVariables.count)
                        for v in masterVariables {
                            guard let data = results[v] else {
                                throw ChessTrainerError.velocityReadbackMissing(v.operation.name)
                            }
                            let count = try ChessNetwork.elementCount(of: v)
                            out.append(ChessNetwork.readFloatsFP32(from: data, count: count))
                        }
                        return out
                    }
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Write fp32 master values via the master-load assign ops. Caller must
    /// have paused training. After this, sync the working weights with
    /// `syncWorkingFromMasters` is NOT needed — `loadTrainerWeights` loads the
    /// base working weights separately; this only restores the masters.
    private func writeMasterValues(_ weights: [[Float]]) async throws {
        guard weights.count == masterVariables.count else {
            throw ChessTrainerError.trainerWeightCountMismatch(
                expected: "\(masterVariables.count) master tensors",
                got: weights.count
            )
        }
        for (i, v) in masterVariables.enumerated() {
            let expected = try ChessNetwork.elementCount(of: v)
            guard weights[i].count == expected else {
                throw ChessTrainerError.trainerWeightCountMismatch(
                    expected: "master[\(i)] (\(v.operation.name)): \(expected) floats",
                    got: weights[i].count
                )
            }
        }
        return try await withCheckedThrowingContinuation { continuation in
            executionQueue.async { [self] in
                do {
                    try autoreleasepool {
                        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
                            network.inputPlaceholder: network.dummyInferenceInputTensorData
                        ]
                        feeds.reserveCapacity(masterVariables.count + 1)
                        for i in 0..<masterVariables.count {
                            ChessNetwork.writeFloatsFP32(weights[i], into: masterLoadNDArrays[i])
                            feeds[masterLoadPlaceholders[i]] = masterLoadTensorData[i]
                        }
                        let results = network.graph.run(
                            with: network.commandQueue,
                            feeds: feeds,
                            targetTensors: [masterVariables[0]],
                            targetOperations: masterLoadAssignOps
                        )
                        guard results[masterVariables[0]] != nil else {
                            throw ChessTrainerError.velocityLoadGraphFailed(masterVariables[0].operation.name)
                        }
                    }
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Seed the fp32 masters from the current bf16 working weights/stats
    /// (`master = cast(working, fp32)`). No-op under `.float32`. Run after any
    /// wholesale working-weight replacement (fresh fork / promotion) so the
    /// masters don't keep training from stale values. Caller must have paused
    /// training. (Init uses the synchronous `runSyncMastersAtInit`.)
    func syncMastersFromWorking() async throws {
        guard !syncMastersOps.isEmpty, !masterVariables.isEmpty else { return }
        return try await withCheckedThrowingContinuation { continuation in
            executionQueue.async { [self] in
                do {
                    try autoreleasepool {
                        let results = network.graph.run(
                            with: network.commandQueue,
                            feeds: [network.inputPlaceholder: network.dummyInferenceInputTensorData],
                            targetTensors: [masterVariables[0]],
                            targetOperations: syncMastersOps
                        )
                        guard results[masterVariables[0]] != nil else {
                            throw ChessTrainerError.velocityLoadGraphFailed("master_sync")
                        }
                    }
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Run the master-seed ops on the **current** thread (caller owns the
    /// queue): masters are built zero-initialized, so they must equal the
    /// working weights before the first trainStep or the first update would
    /// overwrite the He-init weights with `cast(0 − lr·v)`. No-op under
    /// `.float32`. `init` wraps this in `executionQueue.sync`;
    /// `internalResetNetwork` (already on `executionQueue`) calls it directly.
    private func runSyncMastersOnQueue() {
        guard !syncMastersOps.isEmpty, let first = masterVariables.first else { return }
        autoreleasepool {
            _ = network.graph.run(
                with: network.commandQueue,
                feeds: [network.inputPlaceholder: network.dummyInferenceInputTensorData],
                targetTensors: [first],
                targetOperations: syncMastersOps
            )
        }
    }

    /// Push a new dropout rate into the training graph's `dropout_rate`
    /// variable. Stores the clamped value immediately (so readers see it)
    /// and runs the assign asynchronously on `executionQueue`, serialized
    /// behind any in-flight training step. Logs the application so a
    /// mid-run change is visible in the session log next to its effects.
    private func pushDropoutRateToGraph(_ rate: Float) {
        let clamped = min(max(rate, 0.0), 0.95)
        _dropoutRate.value = clamped
        guard let ph = network.dropoutRateLoadPlaceholder,
              let assign = network.dropoutRateAssignOp,
              let nda = network.dropoutRateLoadNDArray,
              let td = network.dropoutRateLoadTensorData,
              let rateVar = network.dropoutRateVariable else {
            SessionLogger.shared.log(
                "[PARAM] dropoutRate set on a network without dropout scaffolding — ignored (inference-mode graph?)"
            )
            return
        }
        executionQueue.async {
            var v = clamped
            nda.writeBytes(&v, strideBytes: nil)
            autoreleasepool {
                _ = self.network.graph.run(
                    with: self.network.commandQueue,
                    feeds: [
                        self.network.inputPlaceholder: self.network.dummyInferenceInputTensorData,
                        ph: td
                    ],
                    targetTensors: [rateVar],
                    targetOperations: [assign]
                )
            }
            SessionLogger.shared.log(String(format: "[PARAM] dropoutRate applied to training graph: %.4f", clamped))
        }
    }

    /// Run the one-time dropout RNG seed assign on the **current** thread
    /// (caller owns the queue) — `dropout_rng_state <- philoxState(seed)`.
    /// The state variable is built zero-filled; without this the per-block
    /// random draws would all start from the degenerate zero state. No-op
    /// on graphs without dropout scaffolding (inference networks). Same
    /// dummy-input feed pattern as `runSyncMastersOnQueue`.
    private func runDropoutSeedOnQueue() {
        guard let seedOp = network.dropoutRngSeedOp,
              let stateVar = network.dropoutRngStateVariable else { return }
        autoreleasepool {
            _ = network.graph.run(
                with: network.commandQueue,
                feeds: [network.inputPlaceholder: network.dummyInferenceInputTensorData],
                targetTensors: [stateVar],
                targetOperations: [seedOp]
            )
        }
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
    ///
    /// Wrapped in `BatchFeedsInput` (rather than six positional args, four of
    /// which are same-typed `UnsafePointer<Float>`) so the compiler binds the
    /// batch's board / z / value-baseline / legal-mask buffers by name — a
    /// future refactor can't silently swap two of them and still produce a
    /// shaped batch. The struct is constructed at the call site and consumed
    /// synchronously; the pointers it holds must outlive the call (the
    /// caller's `withUnsafeBufferPointer` scope), exactly as before.
    struct BatchFeedsInput {
        let batchSize: Int
        let boards: UnsafePointer<Float>
        let moves: UnsafePointer<Int32>
        let zs: UnsafePointer<Float>
        /// `nil` on the real-data path: the vBaseline is supplied as a GPU buffer
        /// (the value-only forward's output) and bound directly in
        /// `runPreparedStep`, so there is no CPU vBaseline to stage. Non-nil only
        /// on the random-data sweep path, which has no baseline forward.
        let vBaselines: UnsafePointer<Float>?
        let legalMasks: UnsafePointer<Float>
    }

    /// `@unchecked Sendable` wrapper so the value-only forward's GPU result
    /// buffer (an `MPSGraphTensorData`, not `Sendable`) can cross the phase-2 →
    /// phase-3 `enqueue` hop. Safe because the await fully completes the forward
    /// (its `run` waits) before phase 3 reads the buffer. Same pattern as
    /// `ChessNetwork.BatchBoardSource`.
    private struct VBaselineHandoff: @unchecked Sendable {
        let tensorData: MPSGraphTensorData
    }

    private func buildFeeds(_ input: BatchFeedsInput) -> [MPSGraphTensor: MPSGraphTensorData] {
        let cached = feedsForBatch(input.batchSize)

        // The four real-valued feeds (board, z, vBaseline, legalMask)
        // each have a graph placeholder declared at
        // `the net's compute dtype`, so their ND-array storage is that
        // width. On `.float32` the host's Float32 source bytes go
        // straight through, zero-copy. On a narrower dtype (bf16) the
        // Float32 source is the wrong width, so narrow it into the
        // per-batch-size staging buffer first and feed that. The staging
        // width matches the ND array's dtype, so the `writeBytes` byte
        // count lines up. `writeRealValuedFeed` branches once on dtype.
        let boardElementCount = input.batchSize
            * arch.inputPlanes
            * ChessNetwork.boardSize
            * ChessNetwork.boardSize
        writeRealValuedFeed(cached.boardND, from: input.boards, count: boardElementCount, staging: cached.boardStaging)
        writeRealValuedFeed(cached.zND, from: input.zs, count: input.batchSize, staging: cached.zStaging)
        // vBaseline: only the random-data sweep path stages it from the host. The
        // real-data path passes `nil` and binds the value-only forward's GPU
        // buffer directly in `runPreparedStep` (GPU→GPU handoff), so there is
        // nothing to write here and `cached.vBaselineND` is left untouched (its
        // dict entry is overridden at bind time).
        if let vBaselines = input.vBaselines {
            writeRealValuedFeed(cached.vBaselineND, from: vBaselines, count: input.batchSize, staging: cached.vBaselineStaging)
        }
        writeRealValuedFeed(cached.legalMaskND, from: input.legalMasks, count: input.batchSize * ChessNetwork.policySize, staging: cached.legalMaskStaging)
        // move is int32 on both host and placeholder — always raw.
        cached.moveND.writeBytes(
            UnsafeMutableRawPointer(mutating: input.moves),
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
        // Snapshot the step count and the cycling config once, so warmup,
        // the LR cycle, and the momentum cycle all key off the same step and
        // a single consistent (untorn) config read.
        let currentStep = _completedTrainSteps.value
        let cycle = _lrMomentumCycle.value

        let warmupMul: Float
        if lrWarmupSteps > 0 {
            warmupMul = Float(min(1.0, Double(currentStep) / Double(lrWarmupSteps)))
        } else {
            warmupMul = 1.0
        }
        // Base LR: the cycle's geometric value when LR cycling is active,
        // otherwise the static configured learning rate. sqrt-batch scaling
        // and warmup then compose multiplicatively on top, exactly as before —
        // enabling LR cycling overrides the static base, not the multipliers.
        let baseLR: Float = cycle.learningRate(forStep: currentStep).map { Float($0) } ?? learningRate
        var lr: Float
        if sqrtBatchScalingForLR {
            let sqrtBatchScale: Float = Float(
                sqrt(Double(input.batchSize) / Double(Self.sqrtScaleBaseBatchSize))
            )
            lr = baseLR * sqrtBatchScale
        } else {
            lr = baseLR
        }
        lr *= warmupMul
        // Each scalar hyperparameter ND array is declared at
        // `the net's compute dtype` (its graph placeholder is `dtype`), so
        // `writeScalarFeed` narrows the Swift `Float` to bf16 before
        // `writeBytes` on a narrow dtype, or writes the raw `Float` on
        // `.float32`. A raw `writeBytes(&lr, …)` of a 4-byte Float into
        // a 2-byte bf16 ND array would otherwise byte-mismatch.
        writeScalarFeed(lrNDArray, value: lr)
        writeScalarFeed(entropyCoeffNDArray, value: entropyRegularizationCoeff)
        writeScalarFeed(weightDecayNDArray, value: weightDecayC)
        writeScalarFeed(gradClipMaxNormNDArray, value: gradClipMaxNorm)
        writeScalarFeed(policyLossWeightNDArray, value: policyLossWeight)
        writeScalarFeed(valueLossWeightNDArray, value: valueLossWeight)
        writeScalarFeed(illegalMassWeightNDArray, value: illegalMassPenaltyWeight)
        writeScalarFeed(labelSmoothingEpsilonNDArray, value: policyLabelSmoothingEpsilon)
        writeScalarFeed(valueLabelSmoothingEpsilonNDArray, value: valueLabelSmoothingEpsilon)
        // Momentum: the cycle's linear value when momentum cycling is active,
        // otherwise the static configured coefficient.
        let momentumToFeed: Float = cycle.momentum(forStep: currentStep).map { Float($0) } ?? momentumCoeff
        writeScalarFeed(momentumNDArray, value: momentumToFeed)
        writeScalarFeed(complementCEEnableNDArray, value: useSignedAdvantageComplementCE ? 1.0 : 0.0)

        return cached.feedsDict
    }

    /// Write a Float32 host buffer into an ND array whose storage is
    /// `the net's compute dtype`. On `.float32` this is a zero-copy raw
    /// `writeBytes` of the Float32 source. On a narrower dtype (bf16)
    /// the source is narrowed element-by-element into the supplied
    /// reusable `staging` buffer first, then `staging`'s bytes are fed —
    /// the staging width matches the ND array, so the byte count lines
    /// up. `staging` must be non-nil and sized to `count` on a narrow
    /// dtype (allocated once per batch size in `feedsForBatch`); it is
    /// `nil` on `.float32`.
    /// Branches on the **ND array's own** dtype, not `the net's compute dtype`.
    /// All four real-valued feeds (board, z, vBaseline, legalMask) are now
    /// fp32 — each feeds an fp32 placeholder narrowed to the compute dtype by
    /// an in-graph `cast` — so on the bf16 build every call takes the fp32
    /// raw-passthrough branch and `staging` is `nil`. The bf16 staging branch
    /// is retained (general against a future narrow-dtype feed) but is
    /// currently unexercised. Same dtype-of-the-array discipline as
    /// `writeScalarFeed`.
    private func writeRealValuedFeed(
        _ ndArray: MPSNDArray,
        from floatPtr: UnsafePointer<Float>,
        count: Int,
        staging: UnsafeMutableBufferPointer<UInt16>?
    ) {
        switch ndArray.dataType {
        case .float32:
            ndArray.writeBytes(UnsafeMutableRawPointer(mutating: floatPtr), strideBytes: nil)
        case .bFloat16:
            guard let staging, let stagingBase = staging.baseAddress, staging.count >= count else {
                fatalError("ChessTrainer.writeRealValuedFeed: bf16 staging missing or undersized (have \(staging?.count ?? -1), need \(count))")
            }
            for elementIndex in 0..<count {
                stagingBase[elementIndex] = ChessNetwork.float32ToBFloat16Bits(floatPtr[elementIndex])
            }
            ndArray.writeBytes(UnsafeMutableRawPointer(stagingBase), strideBytes: nil)
        case .float16:
            guard let staging, let stagingBase = staging.baseAddress, staging.count >= count else {
                fatalError("ChessTrainer.writeRealValuedFeed: fp16 staging missing or undersized (have \(staging?.count ?? -1), need \(count))")
            }
            var srcImg = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: floatPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float>.size
            )
            var dstImg = vImage_Buffer(
                data: stagingBase,
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<UInt16>.size
            )
            _ = vImageConvert_PlanarFtoPlanar16F(&srcImg, &dstImg, 0)
            ndArray.writeBytes(UnsafeMutableRawPointer(stagingBase), strideBytes: nil)
        default:
            fatalError("ChessTrainer.writeRealValuedFeed: no host-side converter for ND array dtype \(ndArray.dataType)")
        }
    }

    /// Write a single Float32 scalar into a 1-element ND array whose
    /// storage is `the net's compute dtype`. On `.float32` the raw `Float`
    /// bytes are written directly; on bf16 the value is narrowed to a
    /// single `UInt16` on the stack and that is written. No reusable
    /// staging is needed — one element fits in a local.
    private func writeScalarFeed(_ ndArray: MPSNDArray, value: Float) {
        // Branch on the ND array's *own* dtype, not the static network
        // dtype: the four optimizer-update scalars (lr / weight decay /
        // grad-clip / momentum) are fp32 arrays even under bf16 (they feed
        // the fp32 master update — see buildTrainingOps), so they must get
        // the raw 4-byte `Float`; the loss-side scalars are `dataType`.
        switch ndArray.dataType {
        case .float32:
            var v = value
            ndArray.writeBytes(&v, strideBytes: nil)
        case .bFloat16:
            var bits = ChessNetwork.float32ToBFloat16Bits(value)
            ndArray.writeBytes(&bits, strideBytes: nil)
        case .float16:
            var bits = Float16(value).bitPattern
            ndArray.writeBytes(&bits, strideBytes: nil)
        default:
            fatalError("ChessTrainer.writeScalarFeed: no host-side converter for dtype \(ndArray.dataType)")
        }
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
        let dtype = ChessNetwork.mpsDataType(for: arch)

        // The board feed is always fp32: it feeds the network's fp32
        // `inputPlaceholder`, which narrows to the compute dtype on the GPU
        // (`board_input_cast`). So the board ND is Float32 regardless of the
        // network dtype, and its host write is a raw passthrough — no bf16
        // staging (unlike z / vBaseline / legalMask below, whose placeholders
        // are the compute dtype).
        let boardDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [
                NSNumber(value: batchSize),
                NSNumber(value: arch.inputPlanes),
                NSNumber(value: ChessNetwork.boardSize),
                NSNumber(value: ChessNetwork.boardSize)
            ]
        )
        let boardND = MPSNDArray(device: mtlDevice, descriptor: boardDesc)
        // Label every NDArray we hand to MPSGraph so subsequent Metal-
        // trace captures can identify which buffer is which by name.
        // Without these, Xcode's trace UI just shows hex addresses
        // for each `setBuffer:` call, which is unreadable when chasing
        // "which feed is the source of the late MTLBuffer creation?"
        boardND.label = "trainer.feed.board[\(batchSize)] (reset)"
        let boardTD = MPSGraphTensorData(boardND)

        let moveDesc = MPSNDArrayDescriptor(
            dataType: .int32,
            shape: [NSNumber(value: batchSize)]
        )
        let moveND = MPSNDArray(device: mtlDevice, descriptor: moveDesc)
        moveND.label = "trainer.feed.movePlayed[\(batchSize)] (reset)"
        let moveTD = MPSGraphTensorData(moveND)

        // z / vBaseline / legalMask ND arrays are fp32, like the board:
        // their graph placeholders are now declared `dataType: .float32`
        // and narrowed to the compute dtype on the GPU by an in-graph
        // `cast` (see `buildTrainingOps`). So the ND storage is Float32
        // regardless of the network dtype, and the host write is a raw
        // Float32 passthrough — no per-batch-size bf16 staging.
        let zDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [NSNumber(value: batchSize), 1]
        )
        let zND = MPSNDArray(device: mtlDevice, descriptor: zDesc)
        zND.label = "trainer.feed.z[\(batchSize)] (reset)"
        let zTD = MPSGraphTensorData(zND)

        // vBaseline ND array — same shape as z, one scalar per row (fp32).
        let vBaselineDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [NSNumber(value: batchSize), 1]
        )
        let vBaselineND = MPSNDArray(device: mtlDevice, descriptor: vBaselineDesc)
        vBaselineND.label = "trainer.feed.vBaseline[\(batchSize)] (reset)"
        let vBaselineTD = MPSGraphTensorData(vBaselineND)

        let legalMaskDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [NSNumber(value: batchSize), NSNumber(value: ChessNetwork.policySize)]
        )
        let legalMaskND = MPSNDArray(device: mtlDevice, descriptor: legalMaskDesc)
        legalMaskND.label = "trainer.feed.legalMask[\(batchSize)] (reset)"
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
            policyLossWeightPlaceholder: policyLossWeightTensorData,
            valueLossWeightPlaceholder: valueLossWeightTensorData,
            illegalMassWeightPlaceholder: illegalMassWeightTensorData,
            labelSmoothingEpsilonPlaceholder: labelSmoothingEpsilonTensorData,
            valueLabelSmoothingEpsilonPlaceholder: valueLabelSmoothingEpsilonTensorData,
            momentumPlaceholder: momentumTensorData,
            complementCEEnablePlaceholder: complementCEEnableTensorData
        ]

        // All four real-valued feeds are fp32 and narrowed to the compute
        // dtype on the GPU by an in-graph `cast` (board via
        // `board_input_cast`; z / vBaseline / legalMask via the casts added
        // in `buildTrainingOps`). So none needs host-side bf16 staging — the
        // host write is a raw Float32 memcpy through the `.float32` branch of
        // `writeRealValuedFeed`, regardless of the network dtype.
        let boardStaging: UnsafeMutableBufferPointer<UInt16>? = nil
        let zStaging: UnsafeMutableBufferPointer<UInt16>? = nil
        let vBaselineStaging: UnsafeMutableBufferPointer<UInt16>? = nil
        let legalMaskStaging: UnsafeMutableBufferPointer<UInt16>? = nil

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
            feedsDict: feedsDict,
            boardStaging: boardStaging,
            zStaging: zStaging,
            vBaselineStaging: vBaselineStaging,
            legalMaskStaging: legalMaskStaging
        )
        feedCache[batchSize] = feeds
        _feedCacheCount.value = feedCache.count
        return feeds
    }

    /// Return the compiled training-step executable for `(batchSize,
    /// includeDiagnostics)`, compiling and caching it on first use. Must run on
    /// `executionQueue` (same discipline as `feedCache`/`trainingExecutables`).
    ///
    /// The concrete feed shapes are derived from the live feed tensor data
    /// (`feeds`) — the graph placeholders carry `-1` batch dims, but the
    /// executable is specialized to this batch size, which is the whole point
    /// (no per-call shape re-specialization). `assignOps` are compiled in as
    /// target operations so the SGD weight update runs as part of the
    /// executable, exactly as it did under `graph.run`.
    private func trainingExecutable(
        batchSize: Int,
        includeDiagnostics: Bool,
        targets: [MPSGraphTensor],
        feeds: [MPSGraphTensor: MPSGraphTensorData]
    ) throws -> MPSGraphExecutable {
        let key = TrainingExecutableKey(batchSize: batchSize, includeDiagnostics: includeDiagnostics)
        if let existing = trainingExecutables[key] {
            return existing
        }
        var feedShapes: [MPSGraphTensor: MPSGraphShapedType] = [:]
        feedShapes.reserveCapacity(feeds.count)
        for (placeholder, tensorData) in feeds {
            feedShapes[placeholder] = MPSGraphShapedType(
                shape: tensorData.shape,
                dataType: placeholder.dataType
            )
        }
        let des = MPSGraphCompilationDescriptor()
        des.optimizationLevel = self.executableOptimizationLevel
        if self.disableAutoLayoutConversion, #available(macOS 27.0, *) {
            des.disableAutoLayoutConversion()
        }
        // Record what MPSGraph defaults reducedPrecisionFastMath to (answers
        // "is the compiler silently allowing FP16 winograd / FP19 shortcuts?"),
        // then apply the A/B override if one was requested.
        if #available(macOS 26.0, *) {
            SessionLogger.shared.log(
                "[EXEC] reducedPrecisionFastMath default=\(des.reducedPrecisionFastMath.rawValue) override=\(self.reducedPrecisionFastMathRaw.map(String.init) ?? "nil")"
            )
            if let reducedPrecisionFastMathRaw = self.reducedPrecisionFastMathRaw {
                des.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMath(rawValue: reducedPrecisionFastMathRaw)
            }
        }
        // Compile on the large-stack thread too: like autodiff, MPSGraph's
        // compile traverses the full op DAG (now including the gradient ops),
        // so a deep tower can overflow the default dispatch-worker stack here as
        // well. See `withLargeBuildStack`.
        let executable = try withLargeBuildStack {
            self.network.graph.compile(
                with: MPSGraphDevice(mtlDevice: self.network.metalDevice),
                feeds: feedShapes,
                targetTensors: targets,
                targetOperations: self.assignOps,
                compilationDescriptor: des
            )
        }
        trainingExecutables[key] = executable
        SessionLogger.shared.log(
            "[EXEC] compiled training executable batch=\(batchSize) diagnostics=\(includeDiagnostics) targets=\(targets.count)"
        )
        return executable
    }

    /// Run the forward + backward + SGD update graph with the given feeds
    /// and read the loss scalar back. The two public `trainStep` entry
    /// points share this so they produce identical timing breakdowns.
    private func runPreparedStep(
        feeds: [MPSGraphTensor: MPSGraphTensorData],
        prepMs: Double,
        queueWaitMs: Double,
        totalStart: CFAbsoluteTime,
        batchSize: Int,
        vBaselineOverride: MPSGraphTensorData? = nil,
        includeDiagnostics: Bool
    ) throws -> TrainStepTiming {
        // Wrap the graph.run + readback in an autoreleasepool so the
        // results dictionary and its MPSGraphTensorData values — which
        // are returned autoreleased by MPSGraph — drain each step
        // instead of piling up until the enclosing long-lived training
        // Task returns. Without this, multi-hour sessions accumulate
        // massive VM-range allocations (seen as ~420 GB virtual vs
        // ~5 GB resident) and the main thread spends progressively
        // more time in deferred Obj-C releases.
        return try autoreleasepool {
        let gpuStart = CFAbsoluteTimeGetCurrent()
        // Lean outputs (loss components + grad-norm) are on the weight-update
        // path and read back every step. The diagnostic reductions — several
        // over the full [batch, policySize] logit tensor — are extra graph work
        // that only feeds the periodic [STATS] line and charts, so on non-stats
        // steps we omit them from the target list and MPSGraph never encodes
        // them. `assignOps` still forces the full forward+backward+optimizer,
        // so this trims diagnostic reductions, not the core step. See
        // GPU_UTILIZATION_PLAN.md (Phase 1).
        let leanTargets: [MPSGraphTensor] = [
            totalLoss, policyLossTensor, valueLossTensor,
            illegalMassPenaltyTensor, gradGlobalNormTensor
        ]
        let diagnosticTargets: [MPSGraphTensor] = [
            policyEntropyTensor, policyNonNegCountTensor, policyNonNegIllegalCountTensor,
            valueMeanTensor, valueAbsMeanTensor,
            valueProbWinTensor, valueProbDrawTensor, valueProbLossTensor,
            policyHeadWeightNormTensor,
            policyLogitAbsMaxTensor, playedMoveProbTensor,
            playedMoveProbPosAdvTensor, playedMoveProbNegAdvTensor,
            advantageMeanTensor, advantageStdTensor, advantageMinTensor, advantageMaxTensor,
            advantageFracPosTensor, advantageFracSmallTensor,
            advantageRawTensor,
            policyLossWinTensor, policyLossLossTensor,
            velocityGlobalNormTensor
        ]
        let targets = includeDiagnostics ? leanTargets + diagnosticTargets : leanTargets
        // Phase 2: run through a compiled MPSGraphExecutable (cached per
        // batchSize × target set) rather than `graph.run`, which re-derives the
        // execution plan each call. The executable shares the graph's weight
        // variables — bidirectional, proven in
        // MPSGraphExecutableVariableSemanticsTests — so loadWeights / promotion
        // / checkpoint reads still observe the trainer's in-place updates and
        // vice versa. The compiled step is numerically identical to graph.run
        // (MPSGraphExecutableTrainingEquivalenceTests). See
        // GPU_UTILIZATION_PLAN.md (Phase 2).
        let executable = try trainingExecutable(
            batchSize: batchSize,
            includeDiagnostics: includeDiagnostics,
            targets: targets,
            feeds: feeds
        )
        // Bind inputs in the executable's OWN feed order (ordering-safe: we map
        // each of the executable's feed tensors to its bound data rather than
        // guessing positions).
        guard let feedTensors = executable.feedTensors else {
            throw ChessTrainerError.lossOutputMissing
        }
        var inputs: [MPSGraphTensorData] = []
        inputs.reserveCapacity(feedTensors.count)
        for tensor in feedTensors {
            // GPU→GPU vBaseline handoff: when the caller supplies a baseline
            // buffer (the real-data path's value-only forward output), bind it
            // directly to the vBaseline placeholder instead of the cached
            // CPU-staged feed — no CPU round-trip. fp32 [batch,1], matching the
            // placeholder. See GPU_UTILIZATION_PLAN.md.
            if let vBaselineOverride, tensor === vBaselinePlaceholder {
                inputs.append(vBaselineOverride)
                continue
            }
            guard let data = feeds[tensor] else {
                preconditionFailure("ChessTrainer.runPreparedStep: executable feed tensor has no bound data")
            }
            inputs.append(data)
        }
        // Phase 3, Increment 1: encode into a command buffer we own, commit,
        // and wait — instead of the synchronous `executable.run`. Functionally
        // identical to `run` (which is encode+commit+wait internally) and the
        // same perf at 1-deep; the point is to establish the
        // encode/commit/completion plumbing so Increment 2 can stop waiting and
        // keep N command buffers in flight. `MPSCommandBuffer` wraps a raw
        // MTLCommandBuffer and conforms to MTLCommandBuffer, so commit/wait work
        // on it directly. Equivalence to `run` is locked by
        // testExecutableEncodeToCommandBufferMatchesRun.
        guard let mtlCommandBuffer = network.commandQueue.makeCommandBuffer() else {
            throw ChessTrainerError.lossOutputMissing
        }
        let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: mtlCommandBuffer)
        // Pipeline-feasibility probe: time the (async, CPU-only) encode call
        // separately from commit + GPU wait. encodeMs is pure CPU encode;
        // gpuWaitMs is commit + GPU execution + wait. Their ratio decides the
        // Phase 3 architecture. The executable was already compiled by
        // `trainingExecutable` above, so no compile cost is folded into encodeMs.
        let encodeStart = CFAbsoluteTimeGetCurrent()
        let resultArray = executable.encode(
            to: mpsCommandBuffer,
            inputs: inputs,
            results: nil,
            executionDescriptor: nil
        )
        let encodeMs = (CFAbsoluteTimeGetCurrent() - encodeStart) * 1000
        let gpuWaitStart = CFAbsoluteTimeGetCurrent()
        // Serialize this SGD weight-write against concurrent `exportWeights`
        // reads (probe paths run `graph.run` on the network queue). Held only
        // across commit→wait — the GPU section that writes the variables — and
        // released before the readback. Throw-free between wait/signal, so the
        // signal is always reached. See `ChessNetwork.weightAccessLock`.
        network.weightAccessLock.wait()
        mpsCommandBuffer.commit()
        mpsCommandBuffer.waitUntilCompleted()
        let stepStatus = mtlCommandBuffer.status
        network.weightAccessLock.signal()
        // `waitUntilCompleted` returns regardless of GPU success, leaving the
        // buffer in either `.completed` or `.error`. On `.error` (OOM / timeout /
        // kernel fault — e.g. an oversized network) the result tensors below hold
        // garbage; surface it instead of reading them back and training on
        // poisoned weights.
        if stepStatus == .error {
            throw ChessTrainerError.gpuCommandFailed(
                stage: "training step",
                status: stepStatus,
                error: mtlCommandBuffer.error?.localizedDescription
            )
        }
        // Split working-weight sync (bf16 stomp workaround; see
        // `splitWorkingWeightSync`): the main executable updated the fp32 masters
        // but did NOT re-derive the bf16 working weights. Do that now in a
        // SEPARATE graph.run — its own command buffer, after the master writes
        // have fully completed (waitUntilCompleted above) — so the cast
        // temporary lives in a different allocation scope than the fused
        // dual-write that stomps on the Xcode-27/macOS-27 beta stack. No feeds:
        // the sync ops depend only on the master variables, not on any
        // placeholder. Empty (skipped) under `.float32` and when the flag is off.
        if !workingSyncOps.isEmpty {
            // Non-empty targetTensors (read back one variable, ignored) avoids
            // any empty-target edge case in graph.run; the real work is the
            // targetOperations (the working = cast(master) assigns).
            let dummyTarget = network.trainableVariables[0]
            network.weightAccessLock.wait()
            _ = network.graph.run(
                with: network.commandQueue,
                feeds: [:],
                targetTensors: [dummyTarget],
                targetOperations: workingSyncOps
            )
            network.weightAccessLock.signal()
        }
        let gpuWaitMs = (CFAbsoluteTimeGetCurrent() - gpuWaitStart) * 1000
        encodeMsTimes.append(encodeMs)
        gpuWaitMsTimes.append(gpuWaitMs)
        if includeDiagnostics {
            SessionLogger.shared.log(
                "[ENCODE-COST] step=\(_completedTrainSteps.value) batch=\(batchSize)"
                + " n=\(encodeMsTimes.count)"
                + String(format: " encodeMs=(p50=%.2f p99=%.2f)",
                         Self.percentile(encodeMsTimes, 0.50), Self.percentile(encodeMsTimes, 0.99))
                + String(format: " gpuWaitMs=(p50=%.2f p99=%.2f)",
                         Self.percentile(gpuWaitMsTimes, 0.50), Self.percentile(gpuWaitMsTimes, 0.99))
            )
            encodeMsTimes.removeAll(keepingCapacity: true)
            gpuWaitMsTimes.removeAll(keepingCapacity: true)
        }
        // `encode` returns results in the compiled targetTensors order (same as
        // `run`), so zip restores the tensor→data dictionary the readback expects.
        let results = Dictionary(uniqueKeysWithValues: zip(targets, resultArray))
        let gpuMs = (CFAbsoluteTimeGetCurrent() - gpuStart) * 1000

        let readbackStart = CFAbsoluteTimeGetCurrent()
        // Lean outputs are present every step (on the weight-update path).
        guard
            let totalData = results[totalLoss],
            let policyData = results[policyLossTensor],
            let valueData = results[valueLossTensor],
            let illegalPenaltyData = results[illegalMassPenaltyTensor],
            let gradNormData = results[gradGlobalNormTensor]
        else {
            throw ChessTrainerError.lossOutputMissing
        }
        let dtype = ChessNetwork.mpsDataType(for: arch)
        ChessNetwork.readFloats(
            from: totalData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotTotal),
            count: 1,
            dataType: dtype
        )
        ChessNetwork.readFloats(
            from: policyData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPolicy),
            count: 1,
            dataType: dtype
        )
        ChessNetwork.readFloats(
            from: valueData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValue),
            count: 1,
            dataType: dtype
        )
        ChessNetwork.readFloats(
            from: illegalPenaltyData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotIllegalMassPenalty),
            count: 1,
            dataType: dtype
        )
        ChessNetwork.readFloats(
            from: gradNormData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotGradNorm),
            count: 1,
            dataType: dtype
        )

        // Diagnostic outputs are requested only on stats steps, so their
        // results are absent otherwise. Read them all here under one guard;
        // the per-field Buf values below default to `.nan` when skipped.
        var advRawValues: [Float] = []
        if includeDiagnostics {
            guard
                let entropyData = results[policyEntropyTensor],
                let nonNegData = results[policyNonNegCountTensor],
                let nonNegIllegalData = results[policyNonNegIllegalCountTensor],
                let valueMeanData = results[valueMeanTensor],
                let valueAbsMeanData = results[valueAbsMeanTensor],
                let valueProbWinData = results[valueProbWinTensor],
                let valueProbDrawData = results[valueProbDrawTensor],
                let valueProbLossData = results[valueProbLossTensor],
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
                let policyLossLossData = results[policyLossLossTensor],
                let velocityNormData = results[velocityGlobalNormTensor]
            else {
                throw ChessTrainerError.lossOutputMissing
            }
            ChessNetwork.readFloats(from: entropyData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotEntropy), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: nonNegData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotNonNeg), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: nonNegIllegalData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotNonNegIllegal), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: valueMeanData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValueMean), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: valueAbsMeanData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValueAbsMean), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: valueProbWinData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValueProbWin), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: valueProbDrawData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValueProbDraw), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: valueProbLossData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotValueProbLoss), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: policyHeadWNormData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPolicyHeadWNorm), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: pLogitAbsMaxData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPLogitAbsMax), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: playedMoveProbData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPlayedMoveProb), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: playedMoveProbPosAdvData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPlayedMoveProbPosAdv), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: playedMoveProbNegAdvData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPlayedMoveProbNegAdv), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: advMeanData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvMean), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: advStdData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvStd), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: advMinData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvMin), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: advMaxData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvMax), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: advFracPosData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvFracPos), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: advFracSmallData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotAdvFracSmall), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: policyLossWinData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPolicyLossWin), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: policyLossLossData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotPolicyLossLoss), count: 1, dataType: dtype)
            ChessNetwork.readFloats(from: velocityNormData, into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotVelocityNorm), count: 1, dataType: dtype)
            // Raw per-position advantage — batch-sized vector. Read into a
            // fresh [Float] since the size depends on the runtime batch.
            let advRawBatchSize: Int = advRawData.shape.reduce(1) { acc, dim in
                acc * Int(truncating: dim)
            }
            advRawValues = [Float](repeating: 0, count: advRawBatchSize)
            if advRawBatchSize > 0 {
                advRawValues.withUnsafeMutableBufferPointer { buf in
                    if let base = buf.baseAddress {
                        ChessNetwork.readFloats(from: advRawData, into: base, count: advRawBatchSize, dataType: dtype)
                    }
                }
            }
        }
        let totalBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotTotal]
        let policyBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPolicy]
        let valueBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotValue]
        let illegalPenaltyBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotIllegalMassPenalty]
        let gradNormBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotGradNorm]
        // Diagnostic Buf values: the live scratch slot on stats steps, `.nan`
        // otherwise (the reductions weren't encoded, so the slots are stale).
        let entropyBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotEntropy] : Float.nan
        let nonNegBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotNonNeg] : Float.nan
        let nonNegIllegalBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotNonNegIllegal] : Float.nan
        let valueMeanBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotValueMean] : Float.nan
        let valueAbsMeanBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotValueAbsMean] : Float.nan
        let valueProbWinBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotValueProbWin] : Float.nan
        let valueProbDrawBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotValueProbDraw] : Float.nan
        let valueProbLossBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotValueProbLoss] : Float.nan
        let policyHeadWNormBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotPolicyHeadWNorm] : Float.nan
        let pLogitAbsMaxBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotPLogitAbsMax] : Float.nan
        let playedMoveProbBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotPlayedMoveProb] : Float.nan
        let playedMoveProbPosAdvBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotPlayedMoveProbPosAdv] : Float.nan
        let playedMoveProbNegAdvBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotPlayedMoveProbNegAdv] : Float.nan
        let advMeanBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotAdvMean] : Float.nan
        let advStdBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotAdvStd] : Float.nan
        let advMinBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotAdvMin] : Float.nan
        let advMaxBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotAdvMax] : Float.nan
        let advFracPosBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotAdvFracPos] : Float.nan
        let advFracSmallBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotAdvFracSmall] : Float.nan
        let policyLossWinBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotPolicyLossWin] : Float.nan
        let policyLossLossBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotPolicyLossLoss] : Float.nan
        let velocityNormBufValue = includeDiagnostics ? lossReadbackScratchPtr[Self.lossReadbackSlotVelocityNorm] : Float.nan
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
            || (includeDiagnostics && (!valueMeanBufValue.isFinite || !entropyBufValue.isFinite)) {
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
            queueWaitMs: queueWaitMs,
            totalMs: totalMs,
            loss: totalBufValue,
            policyLoss: policyBufValue,
            valueLoss: valueBufValue,
            policyEntropy: entropyBufValue,
            illegalMassPenalty: illegalPenaltyBufValue,
            policyNonNegligibleCount: nonNegBufValue,
            policyNonNegligibleIllegalCount: nonNegIllegalBufValue,
            gradGlobalNorm: gradNormBufValue,
            valueMean: valueMeanBufValue,
            valueAbsMean: valueAbsMeanBufValue,
            valueProbWin: valueProbWinBufValue,
            valueProbDraw: valueProbDrawBufValue,
            valueProbLoss: valueProbLossBufValue,
            // Default — the real-data path overwrites this at the outer
            // trainStep level once it has computed the fresh baseline.
            // The random-data sweep path leaves it nil.
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
            advantageRaw: includeDiagnostics ? advRawValues : nil,
            policyLossWin: policyLossWinBufValue.isFinite ? policyLossWinBufValue : nil,
            policyLossLoss: policyLossLossBufValue.isFinite ? policyLossLossBufValue : nil,
            velocityNorm: velocityNormBufValue,
            hasDiagnostics: includeDiagnostics
        )
        }  // autoreleasepool
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
        // Skip threshold: 80% of the smaller of the two caps. The "lesser"
        // bit is deliberately conservative — on this hardware
        // maxBufferLength is well under recommendedMaxWorkingSetSize, so
        // capping the *total* estimate against the smaller of the two
        // gives a safety margin even though the comparison mixes
        // different things (total vs. single-buffer). Better to skip a
        // borderline batch than to take down the machine.
        let safetyFraction = 0.80
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
            // estimated: the trainer literally uploads a
            // [batch, arch.channels, ChessNetwork.boardSize, ChessNetwork.boardSize]
            // float32 activation tensor and that's the biggest buffer in
            // the graph (beats the [batch, policySize] policy tensors and
            // the [batch, inputPlanes, ChessNetwork.boardSize, ChessNetwork.boardSize] input).
            let largestBufferBytes = Self.largestBufferBytes(forBatchSize: batchSize, arch: arch)
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
    /// this batch size — one
    /// [batch, arch.channels, ChessNetwork.boardSize, ChessNetwork.boardSize]
    /// float32 activation tensor. That's larger than the [batch, policySize]
    /// policy tensors and the [batch, inputPlanes, ChessNetwork.boardSize,
    /// ChessNetwork.boardSize] input, so it's the buffer that would first hit
    /// `maxBufferLength`. This is an architectural fact, not a guess.
    static func largestBufferBytes(forBatchSize batchSize: Int, arch: NetworkArchitecture) -> UInt64 {
        let floatBytes = MemoryLayout<Float>.size
        let spatial = ChessNetwork.boardSize * ChessNetwork.boardSize
        let channels = arch.maxBlockChannels
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
    /// Off-main async variant of `currentPhysFootprintBytes()`. The
    /// `task_info` kernel call runs on a global executor so the
    /// awaiter (typically the main actor) is never synchronously
    /// blocked.
    static func getAppMemoryFootprintBytes() async -> UInt64 {
        await withCheckedContinuation { (cont: CheckedContinuation<UInt64, Never>) in
            DispatchQueue.global(qos: .default).async {
                cont.resume(returning: Self.currentPhysFootprintBytes())
            }
        }
    }

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

    /// Off-main async variant of `sampleCurrentProcessUsage()`. Both
    /// `task_info` kernel calls run on a global executor so the
    /// awaiter (typically the main actor) is never synchronously
    /// blocked.
    static func asyncSampleCurrentProcessUsage() async -> ProcessUsageSample? {
        await withCheckedContinuation { (cont: CheckedContinuation<ProcessUsageSample?, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(returning: Self.sampleCurrentProcessUsage())
            }
        }
    }

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

    /// Stats and limits for the current Metal device's memory
    func currentMetalMemoryLimits() async -> MetalDeviceMemoryLimits {
        await withCheckedContinuation { [network] (cont: CheckedContinuation<MetalDeviceMemoryLimits, Never>) in
            DispatchQueue.global(qos: .default).async {
                let device = network.metalDevice
                let deviceMemoryCaps = MetalDeviceMemoryLimits(
                    recommendedMaxWorkingSet: device.recommendedMaxWorkingSetSize,
                    currentAllocated: UInt64(device.currentAllocatedSize),
                    maxBufferLength: UInt64(device.maxBufferLength)
                )
                cont.resume(returning: deviceMemoryCaps)
            }
        }
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

    /// Single-percentile helper over a `[Double]` accumulator. Sorts
    /// a copy (caller's storage isn't reordered) and indexes by
    /// fraction. Returns `.nan` on empty input.
    fileprivate static func percentile(_ samples: [Double], _ p: Double) -> Double {
        guard !samples.isEmpty else { return .nan }
        let sorted = samples.sorted()
        let idx = Int((p * Double(sorted.count - 1)).rounded())
        return sorted[max(0, min(sorted.count - 1, idx))]
    }
}
