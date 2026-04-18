import Darwin
import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - Errors

enum ChessTrainerError: LocalizedError {
    case lossOutputMissing
    case gradientMissing(String)

    var errorDescription: String? {
        switch self {
        case .lossOutputMissing:
            return "Training step ran but loss tensor was not in the result map"
        case .gradientMissing(let name):
            return "Gradient missing for variable: \(name)"
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
    /// [0, log(4096)] ≈ [0, 8.32]. Random init sits near the ceiling;
    /// a collapsed policy heads toward 0. Watch for monotonic drift to
    /// either extreme — that's the signature of policy collapse or a
    /// stuck-at-uniform learning failure.
    let policyEntropy: Float
    let policyNonNegligibleCount: Float
    /// Global L2 norm of the flattened gradient vector across every
    /// trainable variable, computed on the GPU before clipping. When
    /// the value exceeds `ChessTrainer.gradClipMaxNorm`, the update
    /// step scales all gradients by `maxNorm / norm` so the effective
    /// step size is capped. Diagnostic only — the clip is already
    /// applied inside the graph. A value above `gradClipMaxNorm` is a
    /// clip event; steady values above it signal persistent overshoot
    /// that warrants a lower LR.
    let gradGlobalNorm: Float
}

// MARK: - Training Batch

/// A batch of labeled real-data training examples as non-owning views
/// over `ReplayBuffer`'s reusable sample storage. The pointers are
/// valid only until the next `ReplayBuffer.sample` call on the buffer
/// that produced them; the trainer must consume the batch synchronously
/// and not retain any of the pointers past `trainStep(batch:)`.
///
/// Marked `@unchecked Sendable` because the raw pointers make the
/// default Sendable inference fail, but the actual use is safe: the
/// training worker is a single task that samples, trains, and drops
/// the batch within one loop iteration. The batch is never shared
/// across tasks and never held past the next `sample()` call that
/// would invalidate its pointers.
struct TrainingBatch: @unchecked Sendable {
    /// Flat `[batchSize, 18, 8, 8]` float32 board planes, current-player
    /// relative (same encoding that `BoardEncoder` and the inference path
    /// already use). Aliases the replay buffer's reusable output array.
    let boards: UnsafePointer<Float>
    /// Policy-target indices in the network's coordinate system (0–4095),
    /// one per batch row. These are the already-flipped indices emitted by
    /// `MPSChessPlayer.networkPolicyIndex(for:flip:)` so they line up with
    /// the flipped board planes above.
    let moves: UnsafePointer<Int32>
    /// Game outcome from each position's current player's perspective:
    /// +1 win, 0 draw, −1 loss. One per batch row.
    let zs: UnsafePointer<Float>
    /// Inference-time value estimate `v(position)` captured at the
    /// moment this position was played, per batch row. Fed as the
    /// advantage baseline so the policy loss becomes
    /// `mean((z − vBaseline) · −log p(a*))`. Feeding it through a
    /// placeholder is what detaches it from the autodiff graph —
    /// MPSGraph has no stopGradient op, so the placeholder feed is
    /// the mechanism. Same ring source as `zs`.
    let vBaselines: UnsafePointer<Float>
    let batchSize: Int
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
/// hop per step. The SwiftUI `snapshotTimer` polls `snapshot()` at ~60 Hz
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
/// `ReplayBuffer`: an `NSLock` guards all state.
final class TrainingLiveStatsBox: @unchecked Sendable {
    /// Immutable snapshot the UI reads. All fields are value types so
    /// the snapshot is independent of further worker writes.
    struct Snapshot: Sendable {
        let stats: TrainingRunStats
        let lastTiming: TrainStepTiming?
        let rollingPolicyLoss: Double?
        let rollingValueLoss: Double?
        let rollingPolicyEntropy: Double?
        let rollingPolicyNonNegCount: Double?
        let rollingGradGlobalNorm: Double?
        let error: String?
    }

    private let lock = NSLock()
    private var _stats = TrainingRunStats()
    private var _lastTiming: TrainStepTiming?
    private var _policyLossWindow: [Double] = []
    private var _valueLossWindow: [Double] = []
    private var _policyEntropyWindow: [Double] = []
    private var _policyNonNegWindow: [Double] = []
    private var _gradNormWindow: [Double] = []
    private var _error: String?
    private let rollingWindow: Int

    init(rollingWindow: Int) {
        precondition(rollingWindow > 0, "Rolling window must be positive")
        self.rollingWindow = rollingWindow
    }

    /// Seed the stats with values from a resumed session so the
    /// step counter and other totals don't restart from zero.
    func seed(_ stats: TrainingRunStats) {
        lock.lock()
        defer { lock.unlock() }
        _stats = stats
    }

    /// Record one completed training step. Called from the background
    /// training task; takes the lock briefly and returns. No main-actor
    /// hop and no SwiftUI invalidation — the view picks the change up on
    /// the next heartbeat tick.
    func recordStep(_ timing: TrainStepTiming) {
        lock.lock()
        defer { lock.unlock() }
        _stats.record(timing)
        _lastTiming = timing
        _policyLossWindow.append(Double(timing.policyLoss))
        if _policyLossWindow.count > rollingWindow {
            _policyLossWindow.removeFirst(_policyLossWindow.count - rollingWindow)
        }
        _valueLossWindow.append(Double(timing.valueLoss))
        if _valueLossWindow.count > rollingWindow {
            _valueLossWindow.removeFirst(_valueLossWindow.count - rollingWindow)
        }
        _policyEntropyWindow.append(Double(timing.policyEntropy))
        if _policyEntropyWindow.count > rollingWindow {
            _policyEntropyWindow.removeFirst(_policyEntropyWindow.count - rollingWindow)
        }
        _policyNonNegWindow.append(Double(timing.policyNonNegligibleCount))
        if _policyNonNegWindow.count > rollingWindow {
            _policyNonNegWindow.removeFirst(_policyNonNegWindow.count - rollingWindow)
        }
        _gradNormWindow.append(Double(timing.gradGlobalNorm))
        if _gradNormWindow.count > rollingWindow {
            _gradNormWindow.removeFirst(_gradNormWindow.count - rollingWindow)
        }
    }

    /// Record a terminal training error. Also called from the worker.
    /// The first error wins — subsequent calls are ignored so a
    /// follow-on error doesn't clobber the original cause.
    func recordError(_ message: String) {
        lock.lock()
        defer { lock.unlock() }
        if _error == nil { _error = message }
    }

    /// Snapshot all fields atomically for the UI poller.
    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        let rollingPolicy = Self.mean(_policyLossWindow)
        let rollingValue = Self.mean(_valueLossWindow)
        let rollingEntropy = Self.mean(_policyEntropyWindow)
        let rollingNonNeg = Self.mean(_policyNonNegWindow)
        let rollingGradNorm = Self.mean(_gradNormWindow)
        return Snapshot(
            stats: _stats,
            lastTiming: _lastTiming,
            rollingPolicyLoss: rollingPolicy,
            rollingValueLoss: rollingValue,
            rollingPolicyEntropy: rollingEntropy,
            rollingPolicyNonNegCount: rollingNonNeg,
            rollingGradGlobalNorm: rollingGradNorm,
            error: _error
        )
    }

    private static func mean(_ window: [Double]) -> Double? {
        guard !window.isEmpty else { return nil }
        let sum = window.reduce(0, +)
        return sum / Double(window.count)
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

    /// L2 weight-decay coefficient applied to every trainable variable
    /// on every step (decoupled, AdamW-style). The update rule becomes
    /// `v_new = v - lr * (clipped_grad + weightDecayC * v)`, which is
    /// equivalent to `(1 - lr*c) * v - lr * clipped_grad`. Applied
    /// uniformly — biases and BN scale/shift included — matching the
    /// design doc's "L2 on all params" decision.
    static let weightDecayC: Float = 1e-4

    /// Global L2-norm gradient clipping threshold. If the L2 norm of
    /// the concatenated gradient vector over every trainable variable
    /// exceeds this value, every gradient is scaled by
    /// `maxNorm / globalNorm` so the effective step is capped. 5.0 is
    /// a conservative value that sits well above steady-state norms
    /// under healthy training but cuts off the single-step blowups
    /// (see 2026-04-15 incident).
    static let gradClipMaxNorm: Float = 5.0

    var learningRate: Float

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
    private var lrPlaceholder: MPSGraphTensor           // [] scalar float
    private var totalLoss: MPSGraphTensor               // scalar
    private var policyLossTensor: MPSGraphTensor        // scalar
    private var valueLossTensor: MPSGraphTensor         // scalar
    private var policyEntropyTensor: MPSGraphTensor     // scalar (diagnostic)
    private var policyNonNegCountTensor: MPSGraphTensor // scalar (diagnostic)
    private var gradGlobalNormTensor: MPSGraphTensor    // scalar (diagnostic)
    private var assignOps: [MPSGraphOperation]

    /// Pre-allocated scalar ND array for the learning-rate feed.
    /// Written with the current `learningRate` on each step so
    /// the value can change between steps without rebuilding the
    /// graph. Recreated in `resetNetwork()` alongside the feed
    /// cache so the new graph's placeholder maps to a fresh
    /// tensor-data wrapper.
    private var lrNDArray: MPSNDArray
    private var lrTensorData: MPSGraphTensorData

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
    private static let lossReadbackSlotCount: Int = 6

    // MARK: Init

    init(learningRate: Float = 1e-4) throws {
        self.learningRate = learningRate
        let net = try ChessNetwork(bnMode: .training)
        self.network = net
        let built = try Self.buildTrainingOps(network: net)
        self.movePlayedPlaceholder = built.movePlayed
        self.zPlaceholder = built.z
        self.vBaselinePlaceholder = built.vBaseline
        self.lrPlaceholder = built.lr
        self.totalLoss = built.totalLoss
        self.policyLossTensor = built.policyLoss
        self.valueLossTensor = built.valueLoss
        self.policyEntropyTensor = built.policyEntropy
        self.policyNonNegCountTensor = built.policyNonNegCount
        self.gradGlobalNormTensor = built.gradGlobalNorm
        self.assignOps = built.assignOps

        // Scalar ND array for the learning rate feed, reused every step.
        let lrDesc = MPSNDArrayDescriptor(
            dataType: ChessNetwork.dataType,
            shape: [1]
        )
        let lrND = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.lrNDArray = lrND
        self.lrTensorData = MPSGraphTensorData(lrND)

        let lossPtr = UnsafeMutablePointer<Float>.allocate(
            capacity: Self.lossReadbackSlotCount
        )
        lossPtr.initialize(repeating: 0, count: Self.lossReadbackSlotCount)
        self.lossReadbackScratchPtr = lossPtr
    }

    deinit {
        lossReadbackScratchPtr.deinitialize(count: Self.lossReadbackSlotCount)
        lossReadbackScratchPtr.deallocate()
    }

    /// Tear down the current training-mode network and build a fresh one.
    /// Used at the start of a sweep so each run starts from random weights
    /// rather than whatever the previous run left behind. Throws if the
    /// underlying ChessNetwork init fails (Metal/device problems) or if
    /// gradient lookup fails for any trainable variable.
    func resetNetwork() throws {
        let net = try ChessNetwork(bnMode: .training)
        self.network = net
        let built = try Self.buildTrainingOps(network: net)
        self.movePlayedPlaceholder = built.movePlayed
        self.zPlaceholder = built.z
        self.vBaselinePlaceholder = built.vBaseline
        self.lrPlaceholder = built.lr
        self.totalLoss = built.totalLoss
        self.policyLossTensor = built.policyLoss
        self.valueLossTensor = built.valueLoss
        self.policyEntropyTensor = built.policyEntropy
        self.policyNonNegCountTensor = built.policyNonNegCount
        self.gradGlobalNormTensor = built.gradGlobalNorm
        self.assignOps = built.assignOps
        // Rebuild the LR scalar feed against the new network's device
        // so the new graph's placeholder maps to a fresh wrapper.
        let lrDesc = MPSNDArrayDescriptor(
            dataType: ChessNetwork.dataType,
            shape: [1]
        )
        self.lrNDArray = MPSNDArray(device: net.metalDevice, descriptor: lrDesc)
        self.lrTensorData = MPSGraphTensorData(lrNDArray)
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
        lr: MPSGraphTensor,
        totalLoss: MPSGraphTensor,
        policyLoss: MPSGraphTensor,
        valueLoss: MPSGraphTensor,
        policyEntropy: MPSGraphTensor,
        policyNonNegCount: MPSGraphTensor,
        gradGlobalNorm: MPSGraphTensor,
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
            depth: 4096,
            axis: 1,
            dataType: dtype,
            onValue: 1.0,
            offValue: 0.0,
            name: "move_onehot"
        )
        let ceLossRaw = graph.softMaxCrossEntropy(
            network.policyOutput,
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
        let weightedCE = graph.multiplication(advantage, negLogProb, name: "adv_weighted_ce")
        let policyLoss = graph.mean(of: weightedCE, axes: [0, 1], name: "policy_loss")

        // --- Value loss: L = mean( (z - v)^2 ) ---

        let diff = graph.subtraction(z, network.valueOutput, name: "value_diff")
        let sq = graph.square(with: diff, name: "value_sq")
        let valueLoss = graph.mean(of: sq, axes: [0, 1], name: "value_loss")

        // --- Policy entropy (diagnostic; not in totalLoss) ---
        //
        // H(p) = −Σ p · log p, per position, then mean across batch.
        // Range is [0, log(4096)] ≈ [0, 8.32] nats; random init sits near
        // the ceiling, a collapsed policy heads toward 0.
        //
        // This path is read via run-time fetch but is NOT an input to
        // totalLoss, so the autodiff walk from totalLoss never enters it.
        // That lets us build a numerically stable log-softmax here using
        // max-subtraction (reductionMaximum has no gradient implementation
        // in MPSGraph, but we don't need one for a diagnostic tensor).
        let logitsMax = graph.reductionMaximum(
            with: network.policyOutput,
            axis: 1,
            name: "policy_logits_max"
        )
        let shiftedLogits = graph.subtraction(
            network.policyOutput,
            logitsMax,
            name: "policy_logits_shifted"
        )
        let expShifted = graph.exponent(
            with: shiftedLogits,
            name: "policy_exp_shifted"
        )
        let sumExpShifted = graph.reductionSum(
            with: expShifted,
            axis: 1,
            name: "policy_sum_exp_shifted"
        )
        let logSumExpShifted = graph.logarithm(
            with: sumExpShifted,
            name: "policy_log_sum_exp_shifted"
        )
        let logSoftmax = graph.subtraction(
            shiftedLogits,
            logSumExpShifted,
            name: "policy_log_softmax"
        )
        let softmax = graph.exponent(with: logSoftmax, name: "policy_softmax")
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
        // Count of softmax entries above 1/4096 (the uniform
        // probability), averaged across the batch. Starts near
        // ~2048 with random init and drops as the policy
        // concentrates on promising moves. Like entropy, this is
        // diagnostic-only and not in totalLoss.
        let nonNegThreshold = graph.constant(
            1.0 / Double(ChessNetwork.policySize),
            dataType: dtype
        )
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

        // --- Total loss ---
        //
        // Policy loss is REINFORCE on the played move over a 4096-way
        // softmax, so its gradient is naturally much weaker than the
        // value head's (z−v)² gradient. Scale the policy term up by K
        // so both heads get meaningful gradient during the pre-MCTS
        // bootstrap phase of training.
        //
        // K is applied as a true coefficient on policyLoss only — no
        // global normalizer, because dividing the sum divides every
        // term and cancels the relative boost. If the larger effective
        // learning rate on the shared trunk causes instability, lower
        // the LR rather than adding a normalizer.
        let policyWeight = graph.constant(50.0, dataType: dtype)
        let weightedPolicy = graph.multiplication(
            policyWeight,
            policyLoss,
            name: "weighted_policy_loss"
        )
        let totalLossTensor = graph.addition(
            valueLoss,
            weightedPolicy,
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

        // --- Gradient clip scale: maxNorm / max(norm, maxNorm) ---
        //
        // Equivalent to `min(1, maxNorm / norm)`. When `norm ≤ maxNorm`
        // the scale is 1 (no-op); above the threshold the scale
        // shrinks so the resulting update has L2 norm exactly
        // `maxNorm`. No epsilon needed because `max(norm, maxNorm)`
        // is always ≥ maxNorm > 0.
        let maxNormScalar = graph.constant(
            Double(Self.gradClipMaxNorm),
            dataType: dtype
        )
        let clipDenom = graph.maximum(
            gradGlobalNorm,
            maxNormScalar,
            name: "grad_clip_denom"
        )
        let clipScale = graph.division(
            maxNormScalar,
            clipDenom,
            name: "grad_clip_scale"
        )

        // --- SGD updates with weight decay + clipped gradients ---
        //
        // v_new = v - lr * (clipped_grad + weightDecayC * v)
        //       = (1 - lr*weightDecayC) * v - lr * clipped_grad
        //
        // Decoupled weight decay (AdamW-style) applied uniformly to
        // every trainable variable including biases and BN params.
        // Matches the design-doc decision to "L2 on all params".
        //
        // The learning rate is a placeholder (not a constant) so it
        // can be changed between steps without rebuilding the graph.
        // Each training step feeds the current `self.learningRate`
        // via the pre-allocated `lrNDArray`.

        let lrTensor = graph.placeholder(
            shape: [1],
            dataType: dtype,
            name: "learning_rate"
        )
        let weightDecayConstant = graph.constant(
            Double(Self.weightDecayC),
            dataType: dtype
        )

        var ops: [MPSGraphOperation] = []
        ops.reserveCapacity(network.trainableVariables.count)
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
            // Decoupled weight decay term: c*v.
            let decayTerm = graph.multiplication(
                variable,
                weightDecayConstant,
                name: nil
            )
            // Combined update direction before scaling by lr.
            let combinedUpdate = graph.addition(clippedGrad, decayTerm, name: nil)
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

        return (movePlayed, z, vBaseline, lrTensor, totalLossTensor, policyLoss, valueLoss, policyEntropy, policyNonNegCount, gradGlobalNorm, ops)
    }

    // MARK: - Training Step

    /// Run a single training step on a batch of randomly synthesized data.
    /// Returns timing breakdown and the loss scalar. Repeated calls update
    /// this trainer's internal network weights via SGD — that's how we
    /// verified the training pipeline is mechanically correct (random data
    /// + random labels + monotonically decreasing loss). The trainer's
    /// internal network is **not** the inference network, so these updates
    /// don't affect Play Game or Forward Pass.
    func trainStep(batchSize: Int) throws -> TrainStepTiming {
        let totalStart = CFAbsoluteTimeGetCurrent()

        // --- Data prep: synthesize random boards, moves, outcomes ---

        let prepStart = CFAbsoluteTimeGetCurrent()
        let floatsPerBoard = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
        let totalBoardFloats = batchSize * floatsPerBoard

        var boardFloats = [Float](repeating: 0, count: totalBoardFloats)
        Self.fillRandomFloats(&boardFloats)

        var moveIndices = [Int32](repeating: 0, count: batchSize)
        // Random move indices in [0, 4096). One per batch row.
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

        // Unbox the four Swift arrays into raw pointers and feed
        // them through the shared pointer-based `buildFeeds` /
        // `runPreparedStep` pipeline.
        return try boardFloats.withUnsafeBufferPointer { boardsBuf in
            try moveIndices.withUnsafeBufferPointer { movesBuf in
                try zValues.withUnsafeBufferPointer { zsBuf in
                    try vBaselineValues.withUnsafeBufferPointer { vBaseBuf in
                        // The four arrays were allocated just above
                        // with positive batch size, so their
                        // `baseAddress`es are guaranteed non-nil.
                        guard
                            let boardsBase = boardsBuf.baseAddress,
                            let movesBase = movesBuf.baseAddress,
                            let zsBase = zsBuf.baseAddress,
                            let vBaseBase = vBaseBuf.baseAddress
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
                            vBaselines: vBaseBase
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

    /// Run a single training step on a pre-built batch of real self-play
    /// data. Same SGD update rule, same loss output — only the data source
    /// differs from the random-data `trainStep(batchSize:)` path. Callers
    /// are expected to draw batches from a `ReplayBuffer` and feed them
    /// here; see `ContentView.startRealTraining` for the driver loop.
    ///
    /// The `batch` is a non-owning view over the replay buffer's reusable
    /// output storage — this function consumes the pointers synchronously
    /// inside `buildFeeds` (via `writeBytes` into the cached training ND
    /// arrays) and does not retain them past the call.
    func trainStep(batch: TrainingBatch) throws -> TrainStepTiming {
        let totalStart = CFAbsoluteTimeGetCurrent()
        let prepStart = CFAbsoluteTimeGetCurrent()

        let feeds = buildFeeds(
            batchSize: batch.batchSize,
            boards: batch.boards,
            moves: batch.moves,
            zs: batch.zs,
            vBaselines: batch.vBaselines
        )
        let prepMs = (CFAbsoluteTimeGetCurrent() - prepStart) * 1000

        return try runPreparedStep(
            feeds: feeds,
            prepMs: prepMs,
            totalStart: totalStart
        )
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
        vBaselines: UnsafePointer<Float>
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

        // Write the current learning rate into the scalar feed.
        var lr = learningRate
        lrNDArray.writeBytes(&lr, strideBytes: nil)

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
            lrPlaceholder: lrTensorData
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
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let results = network.graph.run(
            with: network.commandQueue,
            feeds: feeds,
            targetTensors: [totalLoss, policyLossTensor, valueLossTensor, policyEntropyTensor, policyNonNegCountTensor, gradGlobalNormTensor],
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
            let gradNormData = results[gradGlobalNormTensor]
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
            from: gradNormData,
            into: lossReadbackScratchPtr.advanced(by: Self.lossReadbackSlotGradNorm),
            count: 1
        )
        let totalBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotTotal]
        let policyBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotPolicy]
        let valueBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotValue]
        let entropyBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotEntropy]
        let nonNegBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotNonNeg]
        let gradNormBufValue = lossReadbackScratchPtr[Self.lossReadbackSlotGradNorm]
        let readbackMs = (CFAbsoluteTimeGetCurrent() - readbackStart) * 1000

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
            gradGlobalNorm: gradNormBufValue
        )
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
    ) throws -> [SweepRow] {
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
            // the graph (beats the [batch, 4096] policy tensors and the
            // [batch, 18, 8, 8] input).
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
            let warmup = try trainStep(batchSize: batchSize)
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

                let timing = try trainStep(batchSize: batchSize)
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

        return results
    }

    // MARK: - Footprint Helpers

    /// Exact size of the largest single MTLBuffer the trainer requests at
    /// this batch size — one [batch, 128, 8, 8] float32 activation tensor.
    /// That's larger than the [batch, 4096] policy tensors and the
    /// [batch, 18, 8, 8] input, so it's the buffer that would first hit
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
    static func sampleCurrentProcessUsage() -> ProcessUsageSample? {
        // CPU time: use task_info(TASK_THREAD_TIMES_INFO) which
        // reliably sums user + system time across ALL threads.
        // proc_pid_rusage(ri_user_time) was under-reporting on
        // macOS 26, giving ~14% instead of Activity Monitor's ~560%.
        var timesInfo = task_thread_times_info_data_t()
        var timesCount = mach_msg_type_number_t(
            MemoryLayout<task_thread_times_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let timesRC = withUnsafeMutablePointer(to: &timesInfo) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(timesCount)) { intPtr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_THREAD_TIMES_INFO),
                    intPtr,
                    &timesCount
                )
            }
        }
        guard timesRC == KERN_SUCCESS else { return nil }

        let userNs = UInt64(timesInfo.user_time.seconds) &* 1_000_000_000
            &+ UInt64(timesInfo.user_time.microseconds) &* 1000
        let sysNs = UInt64(timesInfo.system_time.seconds) &* 1_000_000_000
            &+ UInt64(timesInfo.system_time.microseconds) &* 1000

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
            cpuNs: userNs &+ sysNs,
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
