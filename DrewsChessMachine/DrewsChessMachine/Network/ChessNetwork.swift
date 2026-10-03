import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

// MARK: - Errors

enum ChessNetworkError: LocalizedError {
    case metalNotSupported
    case commandQueueCreationFailed
    case descriptorCreationFailed
    case randomDescriptorCreationFailed
    case outputMissing(String)
    case weightLoadMismatch(String)
    case variableShapeMissing(String)
    /// The graph's weight variables disagree with `weightTensorPlan()` — the
    /// contract that makes positional weight I/O safe. See
    /// `ChessNetwork.validateAgainstPlan`.
    case weightPlanMismatch(String)
    case boardSizeMismatch(expected: Int, got: Int)
    /// A GPU command buffer we committed finished in a non-`completed` state
    /// (out-of-memory, timeout, kernel fault). Surfaced instead of consuming
    /// garbage result tensors.
    case gpuCommandFailed(stage: String, status: MTLCommandBufferStatus, error: String?)
    /// The command queue returned no command buffer to encode into.
    case commandBufferCreationFailed(stage: String)
    /// The network was built `overwrittenByLoad` and asked to evaluate,
    /// export or train before `loadWeights` gave it real weights.
    case weightsNotLoaded(operation: String)

    var errorDescription: String? {
        switch self {
        case .metalNotSupported:
            return "Metal is not supported on this device"
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .descriptorCreationFailed:
            return "Failed to create convolution descriptor"
        case .randomDescriptorCreationFailed:
            return "Failed to create random-op descriptor for dropout"
        case .outputMissing(let name):
            return "Inference output missing: \(name)"
        case .weightLoadMismatch(let detail):
            return "Weight load mismatch: \(detail)"
        case .variableShapeMissing(let name):
            return "Variable '\(name)' has no shape — cannot size load placeholder"
        case .weightPlanMismatch(let detail):
            return "Weight plan mismatch: \(detail)"
        case .boardSizeMismatch(let expected, let got):
            return "Inference input size mismatch: expected \(expected) floats, got \(got)"
        case .gpuCommandFailed(let stage, let status, let error):
            return "GPU command buffer failed during \(stage): status=\(status), error=\(error ?? "none")."
        case .commandBufferCreationFailed(let stage):
            return "The Metal command queue returned no command buffer for \(stage)."
        case .weightsNotLoaded(let operation):
            return "\(operation) on a network built to receive loaded weights, before any weights were loaded"
        }
    }
}

// MARK: - BN Mode

/// How batch normalization is computed in the forward graph.
///
/// `inference` uses fixed running statistics (the existing behavior — fast,
/// stateless, but degenerate during training because the running stats are
/// frozen). `training` computes batch mean and variance from the input on
/// every forward pass, which is what real training does and which produces
/// meaningfully different gradient computations.
///
/// Used by ChessTrainer to build a separate copy of the network with
/// training-mode BN for benchmarking, while the inference network used by
/// Play Game stays in inference mode.
enum BNMode {
    case inference
    case training
}

// MARK: - Chess Neural Network

/// Chess engine neural network forward pass implemented with MPSGraph.
///
/// Architecture v4 (pre-activation / ResNet v2 tower):
/// - Input: 30x8x8 board tensor (NCHW layout). 20 baseline planes
///   (pieces + castling + EP + halfmove clock + 2 repetition-count
///   planes — planes 18/19 are ≥1× before / ≥2× before signals) plus
///   10 binary temporal-repetition-history planes (20–29). See
///   `BoardEncoder` and the `inputPlanes` doc below for the full
///   plane table.
/// - Stem: `towerConvKernelSize`-square conv (`inputPlanes` -> 128
///   channels) -> BN. **No stem
///   ReLU** — the first nonlinearity is deferred to block0's
///   pre-activation. The stem BN bounds `x_0`, the skip highway's
///   starting value.
/// - Tower: `numBlocks` pre-activation residual blocks. Each block is a
///   clean identity skip `out = input + α·F(input)` (**no activation on
///   the sum**), where the residual function is
///     BN -> ReLU -> conv -> BN -> ReLU -> conv -> [SE module]
///   and α is a per-block trainable ReZero scalar (init `1/√numBlocks`)
///   that bounds depth variance without a dead start. The SE module is a
///   *scale-and-bias* gate: squeeze (global avg pool) -> FC(128 -> 32)
///   -> ReLU -> FC(32 -> 256) -> split into (gammas, betas) ->
///   `sigmoid(gammas)·z + betas` (sigmoid on the scale half only; the
///   bias half is added linearly). FC1 is He-init; FC2 is Glorot-init
///   (it feeds the sigmoid). lc0-style per-position channel attention.
/// - Tower end: `BN -> ReLU` before the heads (pre-activation blocks end
///   in a bare conv-add, so the tower output is an un-normalized linear
///   accumulation — this normalizes + activates it for the heads).
/// - Policy head: 1x1 conv (128 -> 128) -> BN -> ReLU -> 1x1 conv
///   (128 -> 76) → reshape to [B, 4864] (logits). The intermediate
///   conv->BN->ReLU mirrors the value head / lc0 and renormalizes the
///   tower output so the deep tower's activation scale can't inflate the
///   logits. 76 channels = 56 queen-style + 8 knight + 9 underpromotion +
///   3 queen-promotion. See `PolicyEncoding` for the layout.
/// - Value head: 1x1 conv (128 -> 1) -> BN -> ReLU -> flatten -> FC(64 -> 64) -> ReLU -> FC(64 -> 3) -> 3 raw W/D/L logits.
///   Exposed three ways: `valueLogits` (the [B, 3] logits, for the
///   categorical-CE value loss + the W/D/L diagnostics), `valueProbs`
///   (their softmax), and `valueOutput` — the derived scalar
///   `Σ_c softmax(logits)_c·[+1, 0, −1]_c = p_win − p_loss ∈ [−1, +1]`,
///   which is what every inference consumer reads (no tanh).
///
/// Total parameters: ~2.47M (down from ~2.92M pre-refresh — the FC
/// policy head was the largest single component and has been replaced
/// with a fully-convolutional 1×1 conv that uses ~50× fewer params
/// while preserving spatial structure end-to-end).
///
/// Marked `@unchecked Sendable` because MPSGraph/Metal state is not
/// Sendable, but all public entry points serialize access through the
/// instance's private execution queue.

/// Sendable carrier for an externally-owned batched-input pointer.
/// `UnsafePointer<Float>` is not Sendable; this carrier crosses the
/// `enqueue` closure boundary so the pointer-flavored
/// `evaluateBatched(batchBoardsPointer:floatCount:count:consume:)`
/// can hand the buffer to the network's execution-queue work block
/// without boxing into a `[Float]`. Caller's lifetime responsibility:
/// the buffer must outlive the await (the awaiting task is suspended,
/// so any field-stored pointer on the caller stays alive).
struct BatchBoardSource: @unchecked Sendable {
    let pointer: UnsafePointer<Float>
    let floatCount: Int
}

final class ChessNetwork: @unchecked Sendable {

    // MARK: Configuration

    /// Numeric precision for all weights and activations is no longer a
    /// global static — it is per-architecture (`arch.computeDataType`,
    /// mapped to an `MPSDataType` by `mpsDataType(for:)`). The conversion
    /// core (`makeWeightData` Float32 → dtype, the two `readFloats`
    /// overloads dtype → Float32, the init-data builders, `onesData`/`zerosData`,
    /// `writeFloats`, `decodeWeightData`, `bytesPerWeightElement`,
    /// `weightRelativeEpsilon`) all take an explicit `dataType:` so a net
    /// built fp32 and a net built bf16 can coexist; on-disk weights always
    /// speak `Float`, converting only at the GPU boundary, so a model
    /// trained under one precision reloads cleanly under another (modulo
    /// rounding at the re-cast).
    ///
    /// The bf16 trap that motivated keeping fp32 the default: with weights
    /// stored in bf16 and the SGD step `w -= lr · grad` done in bf16 too,
    /// an update smaller than the bf16 ULP at `|w|` rounds to a no-op. For
    /// BN gamma at 1.0 the ULP is `2^-7 ≈ 7.8e-3`, so `lr=1e-3 · grad≈0.05`
    /// is orders of magnitude below the threshold and every gamma update
    /// rounds to zero (confirmed in the KXvb-1 snapshot: all BN gammas
    /// bit-exact at 1.0 after 49k steps). The fix is mixed precision with an
    /// fp32 master copy (LC0-style) or staying in fp32. `.float16` uses the
    /// vImage half path; `.bFloat16` uses `float32ToBFloat16Bits` /
    /// `bFloat16BitsToFloat32` (vImage has no bfloat16 primitive).

    /// Input plane count is per-architecture (`arch.inputPlanes`, derived
    /// from `arch.inputEncoding.planeCount`) — not a global static. It
    /// drives the stem weight shape `[channels, inputPlanes, k, k]`, the
    /// board-encoding stride (`ReplayBuffer.floatsPerBoard`), and the input
    /// placeholder shape; all read it off the built net's arch so a net
    /// using a different encoding builds and loads correctly.
    static let boardSize = 8
    /// Number of policy output channels: 56 queen-style (8 dirs × 7 dists)
    /// + 8 knight + 9 underpromotion (3 pieces × 3 dirs) + 3 queen-promotion
    /// (3 dirs) = 76. See `PolicyEncoding` for the full layout.
    static let policyChannels = 76
    /// Total raw policy logits emitted by the network: `policyChannels × 64`.
    static let policySize = policyChannels * boardSize * boardSize

    // Per-architecture identity constants (channels, numBlocks, kernel
    // sizes, SE reduction, value-head dims, version, parameterCount) used
    // to live here as static lets describing one hardcoded topology. They
    // were removed once the architecture became runtime-configurable: the
    // single source of truth is now the instance `arch:
    // NetworkArchitecture`. Read `arch.maxBlockChannels`, `arch.numBlocks`,
    // `arch.parameterCount`, etc. — never a global default — so every
    // consumer describes the ACTUAL built net. Only the genuinely fixed
    // engine constants (`boardSize`, `policyChannels`, `policySize`, and —
    // pending their own removal passes — `dataType` / `inputPlanes`) remain
    // static.

    // The one-line human-readable summary now lives on `NetworkArchitecture`
    // (`net.network.arch.architectureSummary`) so it always describes the
    // ACTUAL built net rather than the static defaults. The former static
    // `ChessNetwork.architectureSummary` was removed to avoid a second,
    // divergently-formatted source.

    /// Hand-maintained qualitative note about the current architecture
    /// experiment, surfaced as `architecture.notes` in analysis exports.
    /// Deliberately carries **no numbers** — every quantity lives in the
    /// structured arch constants and `architectureSummary`, so this
    /// string can't go stale when a constant changes. Edit it when
    /// starting a new architecture experiment; an empty string is
    /// omitted from the export.
    static let architectureNotes =
        "Shallow-wide kernel experiment: fewer residual blocks with larger "
        + "spatial convolutions, probing whether kernel width can substitute "
        + "for tower depth."

    // MARK: Graph Tensors

    let graph: MPSGraph
    let inputPlaceholder: MPSGraphTensor

    // MARK: Dropout (training-mode graphs only; all nil on inference graphs)

    /// Global dropout rate, a graph PLACEHOLDER fed per execution. fp32
    /// scalar; compared against fp32 uniforms regardless of compute dtype. At
    /// rate 0 every dropout node is an exact identity (mask all-ones, scale
    /// 1.0; ×1.0 is exact in bf16 too), so a rate-0 graph is numerically
    /// indistinguishable from one without the nodes — only the
    /// random-generation cost remains, which is precisely what the perf probe
    /// measures.
    ///
    /// This was a graph VARIABLE until the value baseline was found to be
    /// reading it. `valueBaselineExecutable` targets `valueOutput`, which
    /// sits downstream of every block's dropout node, so one shared mutable
    /// rate meant v(s) was computed through the *training step's* mask and the
    /// advantage `(z − vBaseline)` became a function of that step's random
    /// draw rather than of the position. A variable cannot be overridden
    /// per-executable, and zeroing it around the baseline is not an option
    /// either: the baseline deliberately commits WITHOUT waiting so its GPU
    /// work overlaps the training-step encode, so mutating a shared rate would
    /// race that step's own forward.
    ///
    /// As a placeholder each execution states its own rate. The training step
    /// binds the live rate; every other consumer — value baseline, BN warmup,
    /// and batched/single inference on a training-mode graph (the diagnostics
    /// readbacks) — binds 0 and is therefore exactly dropout-free. The cost is
    /// one extra `[1]` fp32 feed per run, which is why the feed is attached
    /// centrally (`batchInputEntry` / `inferenceFeeds` / `runPreparedStep`)
    /// rather than at each call site.
    let dropoutRateFeedPlaceholder: MPSGraphTensor?
    /// Preallocated `[1]` fp32 zero — the binding every dropout-free consumer
    /// uses. Immutable after construction, so it is safe to share across
    /// concurrently-encoding executions.
    let dropoutRateZeroTensorData: MPSGraphTensorData?
    /// Philox RNG state threaded through the per-block channel-mask draws.
    /// Variable (persists across executions); written through
    /// `dropoutRngSeedOp` and advanced once per training step via
    /// `dropoutRngAdvanceOp` so every step draws fresh masks. The trainer
    /// writes it from its `dropout` random stream at setup and can read it
    /// back and write a saved value, so a resumed run continues the same
    /// mask sequence (`DropoutPhiloxState`). Forward passes that skip the
    /// advance op (e.g. diagnostics `evaluate` on the trainer network)
    /// re-read the same stream position; harmless for noise, exact no-op at
    /// rate 0.
    let dropoutRngStateVariable: MPSGraphTensor?
    /// `[7]` Int32 value the state assign copies into the state variable. Fed,
    /// not baked in as a constant, so one compiled graph serves any seed and
    /// a saved state can be written back.
    let dropoutRngStateFeedPlaceholder: MPSGraphTensor?
    /// `stateVar <- dropoutRngStateFeedPlaceholder`; the trainer runs it with
    /// a seed-derived state right after graph construction (and after a
    /// network reset), and with a saved state on exact resume.
    let dropoutRngSeedOp: MPSGraphOperation?
    /// Per-step state-advance assign; the trainer appends this to its SGD
    /// `assignOps` so the compiled training executable advances the stream
    /// exactly once per step.
    let dropoutRngAdvanceOp: MPSGraphOperation?
    /// Preallocated `[1]` fp32 holding the LIVE rate — the binding used by the
    /// training step, and by nothing else. `ChessTrainer.dropoutRate` rewrites
    /// these bytes on the trainer's execution queue, which serializes the write
    /// against any in-flight step that binds the same buffer. Starts at 0, so a
    /// trainer that never sets a rate trains dropout-free.
    ///
    /// Replaces the old variable-assign plumbing (`dropoutRateLoadPlaceholder`
    /// + assign op + `graph.run`): with the rate fed per execution there is no
    /// graph state to update, so setting it is now a buffer write rather than a
    /// graph execution.
    let dropoutRateLiveNDArray: MPSNDArray?
    let dropoutRateLiveTensorData: MPSGraphTensorData?

    /// Raw policy logits, shape `[batch, policySize]`, always fp32
    /// (`headTailDataType`) whatever the compute dtype: the head's tail runs
    /// in fp32 (see `widenToHeadTail`). The inference readback target and
    /// the trainer's loss input alike.
    let policyOutput: MPSGraphTensor
    /// Derived scalar value, shape `[batch, 1]` = `p_win − p_loss`
    /// (= E[outcome] ∈ [−1, +1], no tanh). This is what every inference
    /// consumer reads and what the policy-gradient baseline is fed; the
    /// full W/D/L distribution stays available via `valueLogits` /
    /// `valueProbs` for the value loss and diagnostics. Always fp32, so it
    /// is also the direct target of the trainer's GPU→GPU value-baseline
    /// forward, which lands `v(s)` in the fp32 buffer the training step's
    /// fp32 `vBaseline` placeholder reads. See `computeValueBaselineGPU`.
    let valueOutput: MPSGraphTensor
    /// Raw W/D/L value-head logits, shape `[batch, 3]` in `[win, draw,
    /// loss]` slot order — matched to the training target `idx = 1 − z`
    /// with z ∈ {+1, 0, −1} (win→0, draw→1, loss→2). Consumed by
    /// `ChessTrainer.buildTrainingOps` for the categorical-cross-entropy
    /// value loss and by the W/D/L probability diagnostics. The
    /// inference path never reads this.
    let valueLogits: MPSGraphTensor
    /// Softmax of `valueLogits`, shape `[batch, 3]` — predicted
    /// (p_win, p_draw, p_loss). Exposed for the W/D/L diagnostics in
    /// the trainer; `valueOutput == Σ_c valueProbs_c · [+1, 0, −1]_c`.
    let valueProbs: MPSGraphTensor

    /// Numerics-audit readbacks (fp32), in graph build order: normalization
    /// inputs, block and tower outputs, the heads' last hidden activations,
    /// and the head outputs. Empty on every network built without
    /// `analysisTaps`, which is every production network.
    let analysisTapReadbacks: [(name: String, tensor: MPSGraphTensor)]
    /// The policy head's final 1×1 conv weight tensor (128 → 76 channels).
    /// Exposed so the trainer can compute diagnostic ||W||₂ per step — the
    /// sharpness of this tensor drives logit magnitudes, which directly
    /// controls how concentrated the temperature-scaled policy becomes.
    /// Growing unbounded is the signature of weight-decay not being
    /// strong enough relative to LR to hold logits in a usable range.
    /// Set in `init` from `policyHead`'s tuple return (non-optional — no IUO).
    private(set) var policyHeadFinalWeights: MPSGraphTensor

    /// All graph variables that should receive gradient updates during
    /// training: every conv weight, FC weight, FC bias, and BN gamma/beta.
    /// Excludes BN running mean/variance — those are EMA-updated (not
    /// gradient-updated) in training mode and loaded directly in
    /// inference mode. See `bnRunningStatsVariables`.
    private(set) var trainableVariables: [MPSGraphTensor] = []

    /// Parallel `[Bool]` flagging which entries of `trainableVariables`
    /// should receive L2 weight decay during training. `true` for conv
    /// and FC weight matrices (the proper "weights"); `false` for BN
    /// gamma/beta and FC biases (the no-decay group, matching the
    /// PyTorch / AdamW recipe). Decaying BN gamma toward zero zeros
    /// out a channel and reduces effective capacity, so those are
    /// explicitly excluded. Indices align 1:1 with `trainableVariables`.
    private(set) var trainableShouldDecay: [Bool] = []

    /// BN running statistics (per-channel mean and variance, shape
    /// `[1, C, 1, 1]` each). Ordered mean-then-variance for each BN
    /// layer, with layers appearing in build order. Used directly by
    /// inference-mode BN to normalize; EMA-updated by training-mode BN
    /// via `bnRunningStatsAssignOps`. `exportWeights` / `loadWeights`
    /// include these alongside trainables so a trained trainer network
    /// can be copied into an inference network as a self-consistent
    /// state snapshot.
    private(set) var bnRunningStatsVariables: [MPSGraphTensor] = []

    /// EMA-update assign operations for BN running statistics. Populated
    /// only in `.training` mode; empty in `.inference`. ChessTrainer
    /// appends these to its SGD assign ops so each training step
    /// updates the running stats in the same graph execution as the
    /// weight updates — after enough steps the running stats converge
    /// to the typical per-channel statistics the trained network's
    /// activations exhibit, which is what inference-mode BN needs.
    private(set) var bnRunningStatsAssignOps: [MPSGraphOperation] = []

    /// Per-BN-layer fresh batch-mean tensors, exposed only in `.training`
    /// mode (empty in `.inference`). Same order as
    /// `bnRunningStatsVariables` mean entries (i.e. mean[layer i] sits
    /// at index i of THIS list, matching index 2i of the running-stats
    /// list which interleaves mean-then-variance). Read out by the
    /// one-shot BN warmup path that primes a fresh inference network's
    /// running stats from one batched forward through a sibling
    /// training-mode network — see `loadBNRunningStatsFromBatchStats`.
    private(set) var bnBatchMeanTensors: [MPSGraphTensor] = []

    /// Per-BN-layer fresh batch-variance tensors. Same shape and
    /// ordering convention as `bnBatchMeanTensors`. Together they let
    /// the warmup path snapshot the population the inference network
    /// will actually see at run time, without waiting for the EMA to
    /// converge over hundreds of training steps.
    private(set) var bnBatchVarTensors: [MPSGraphTensor] = []

    /// Per-persistent-variable placeholder / assign pair used by
    /// `loadWeights(_:)` to write fresh float data into variables at
    /// runtime. Built once at init time so loading is a single graph
    /// execution. Ordered trainables-first then running stats, matching
    /// the output of `exportWeights()`.
    private var weightLoadPlaceholders: [MPSGraphTensor] = []
    private var weightLoadAssignOps: [MPSGraphOperation] = []

    /// One pre-allocated `MPSNDArray` + `MPSGraphTensorData` wrapper per
    /// persistent variable, ordered identically to
    /// `weightLoadPlaceholders`. `loadWeights(_:)` writes new values into
    /// each ND array in place via `writeBytes` and feeds the cached
    /// tensor data, so a weight transfer allocates no MPS objects.
    private let weightLoadNDArrays: [MPSNDArray]
    private let weightLoadTensorData: [MPSGraphTensorData]

    /// Pre-allocated `[1, inputPlanes, 8, 8]` input feed reused on every
    /// `evaluate(board:)` call. The ND array holds the board floats in
    /// `Self.mpsDataType(for: arch)`; the tensor data wrapper is built once and fed
    /// into `graph.run` unchanged. The per-move inference hot path
    /// writes directly into this ND array and allocates zero MPS
    /// objects or shape arrays.
    private let inferenceInputNDArray: MPSNDArray
    private let inferenceInputTensorData: MPSGraphTensorData

    /// Cached feeds dictionary and target tensor list for `evaluate(board:)`.
    /// Built once at init; every inference call feeds these unchanged so
    /// the hot path allocates no Swift `Dictionary` or `Array` on each
    /// call. The ND array backing `inferenceInputTensorData` has its
    /// bytes overwritten in place before each `graph.run`.
    private let inferenceFeeds: [MPSGraphTensor: MPSGraphTensorData]
    private let inferenceTargets: [MPSGraphTensor]

    /// Readback scratch for the policy logits. `evaluate(board:)` asks
    /// MPSGraph to write the policySize-element policy output directly into
    /// this buffer and returns an `UnsafeBufferPointer` over it to the
    /// caller. The buffer is reused across calls — **not re-entrant**;
    /// the returned pointer is valid only until the next `evaluate` call
    /// on this network. Allocated via `UnsafeMutablePointer` rather than
    /// a `[Float]` so we can return a stable pointer without hitting
    /// Swift array CoW.
    private let inferencePolicyScratchPtr: UnsafeMutablePointer<Float>

    /// Readback scratch for the value scalar. Same contract as the
    /// policy scratch; returned to the caller by value rather than as a
    /// pointer, so the aliasing concern does not apply there.
    private let inferenceValueScratchPtr: UnsafeMutablePointer<Float>
    /// Readback scratch for the 3-wide W/D/L softmax — used only by
    /// `evaluateValueDistribution(board:)` (a diagnostics path; the
    /// universal inference closures carry only the derived scalar).
    /// Capacity 3, returned to the caller by value.
    private let inferenceValueProbsScratchPtr: UnsafeMutablePointer<Float>

    /// Zero-filled `[1, inputPlanes, 8, 8]` feed shared by `exportWeights()` and
    /// `loadWeights(_:)` to satisfy MPSGraph's requirement that every
    /// graph placeholder be fed even when the target ops don't consume
    /// it. Filled once at init, never modified afterwards. Also exposed
    /// to `ChessTrainer` for its velocity-tensor read/write helpers,
    /// which need to satisfy the same input-placeholder requirement
    /// without doing any actual forward computation.
    let dummyInferenceInputTensorData: MPSGraphTensorData

    // MARK: Batched Inference Scratch

    /// Per-batch-size input feed cache for `evaluateBatched(batchBoards:count:consume:)`.
    /// Keyed by batch count. Each entry holds one `[count, inputPlanes, 8, 8]`
    /// MPSNDArray (bytes overwritten in place on every call) plus a
    /// pre-built feeds dict. Entries are added lazily the first time a
    /// given batch size is requested and retained for the life of the
    /// network.
    private struct BatchInputEntry {
        let ndArray: MPSNDArray
        let tensorData: MPSGraphTensorData
        let feeds: [MPSGraphTensor: MPSGraphTensorData]
    }
    private var batchInputCache: [Int: BatchInputEntry] = [:]

    /// Compiled inference executables, keyed by batch size (`count`). The batched
    /// forward pass is the highest-frequency GPU submission in the app (self-play
    /// workers, plus the trainer's fresh-baseline forward), so it runs through a
    /// compiled `MPSGraphExecutable` (`.level1` optimization) instead of
    /// `graph.run`, which re-derives execution bookkeeping each call. Compiled
    /// once per batch size and reused for the network's lifetime — the graph is
    /// never rebuilt, and `loadWeights` (champion snapshots at each arena) mutates
    /// the shared variables in place, which the executable observes (proven in
    /// MPSGraphExecutableVariableSemanticsTests). Accessed only on
    /// `executionQueue`, like `batchInputCache`. See GPU_UTILIZATION_PLAN.md.
    private var inferenceExecutables: [Int: MPSGraphExecutable] = [:]

    /// Compiled value-only executables for the trainer's GPU→GPU baseline
    /// forward, keyed by batch size. Targets just `valueOutput` (no policy /
    /// valueProbs, no assign target-ops, so it neither computes the discarded
    /// policy nor pollutes BN running stats — same read-only semantics as the
    /// inference executable). Shares the graph's weight variables like every
    /// other executable. Lives for the network's lifetime (the graph is never
    /// rebuilt), same as `inferenceExecutables`.
    private var valueBaselineExecutables: [Int: MPSGraphExecutable] = [:]
    /// Caller-owned (never `results: nil`) fp32 `[count, 1]` result buffers for
    /// the value-baseline forward, keyed by batch size. The trainer feeds the
    /// returned buffer straight into the training step's `vBaseline`; reused each
    /// call, safe under the trainer's serial phase-2 → phase-3 ordering.
    private var valueBaselineResultCache: [Int: MPSGraphTensorData] = [:]

    /// Readback scratch for batched policy logits. Grows on demand to
    /// the largest batch size ever requested. **Not re-entrant** — the
    /// `UnsafeBufferPointer` handed to the consume closure of
    /// `evaluateBatched(batchBoards:count:consume:)` aliases this storage
    /// and is valid only for the duration of that closure call.
    private var batchPolicyScratchPtr: UnsafeMutablePointer<Float>?
    private var batchPolicyScratchCapacity: Int = 0

    /// Readback scratch for batched value scalars. Same re-entrancy
    /// contract as `batchPolicyScratchPtr`.
    private var batchValueScratchPtr: UnsafeMutablePointer<Float>?
    private var batchValueScratchCapacity: Int = 0

    /// Readback scratch for the batched W/D/L softmax — `count * 3`
    /// floats in slot order `[win, draw, loss]`, position-major. Same
    /// re-entrancy contract as `batchPolicyScratchPtr`: the pointer
    /// handed to the third arg of `consume` aliases this storage and
    /// is valid only for the duration of that closure call. Consumed
    /// by the self-play `DrawWatchTracker` (per-ply pDraw monitoring);
    /// arena and tests pass a `_ in`-style closure and ignore it.
    private var batchValueProbsScratchPtr: UnsafeMutablePointer<Float>?
    private var batchValueProbsScratchCapacity: Int = 0

    // MARK: Metal

    let metalDevice: MTLDevice
    let commandQueue: MTLCommandQueue
    let graphDevice: MPSGraphDevice
    private let executionQueue = DispatchQueue(label: "drewschess.chessnetwork.serial")

    /// Mutex serializing **variable reads** (`exportWeights`, which runs
    /// `graph.run` on this network's queue) against the trainer's **SGD
    /// weight-update** (`ChessTrainer.runPreparedStep`, which writes these same
    /// MPSGraph variables from a *different* queue). Without it, a probe
    /// exporting weights (candidate probe, legal-mass / tactical probes,
    /// analyses) races the in-flight SGD step on the shared variables — torn
    /// reads and result contamination, violating the trainer's "pause before
    /// touching weights" contract. A `DispatchSemaphore` (not the project's
    /// usual `OSAllocatedUnfairLock`) because it's held across GPU work
    /// (commit→waitUntilCompleted / `graph.run`), which unfair locks aren't
    /// meant for. Held only across the variable-touching GPU section on each
    /// side; never nested (the SGD never exports, export never trains), so it
    /// cannot deadlock. Contention is rare (probes fire on multi-second timers).
    let weightAccessLock = DispatchSemaphore(value: 1)

    /// The architecture this instance was built to. Drives every layer shape
    /// in the build path (the static `channels`/`numBlocks`/… constants remain
    /// as the *default* arch for external callers; a guard test asserts they
    /// match `NetworkArchitecture.current`). Compute dtype stays on the static
    /// `dataType` for now — per-model precision is a later phase.
    let arch: NetworkArchitecture

    /// How this network's random-init tensors got their first values: a
    /// seeded draw (the init seed is readable here, for logging and
    /// recording), or none at all because loaded weights replace them.
    let initialization: WeightInitialization

    /// The distribution each conv / FC weight was drawn from in this build,
    /// by plan name (`TensorInitializer.randomTensorRoles`).
    let randomTensorRoles: [String: RandomTensorRole]

    /// True while an `overwrittenByLoad` network has not yet received
    /// weights. Cleared only by `internalLoadWeights`, on `executionQueue`,
    /// once the load's GPU work has completed. Read from more than one queue:
    /// the network's own gates run on `executionQueue`, so for them a load and
    /// a use are ordered by the queue; `ChessTrainer`'s gates run on the
    /// trainer's queue, where a use is ordered after a load only because the
    /// caller awaits `loadWeights` before it. The lock makes every read safe.
    private let awaitingWeightLoad: SyncBox<Bool>

    /// Throws unless the network holds real weights — refuses evaluating,
    /// exporting, training or computing statistics from the zero-filled
    /// variables of an `overwrittenByLoad` network that was never loaded.
    func requireLoadedWeights(_ operation: String) throws {
        if awaitingWeightLoad.value { throw ChessNetworkError.weightsNotLoaded(operation: operation) }
    }

    /// Where the policy head's fp32 tail begins (see `PolicyTailPrecision`).
    let policyTailPrecision: PolicyTailPrecision

    /// When true, every `graph.compile` site sets `disableAutoLayoutConversion`
    /// on its `MPSGraphCompilationDescriptor`, opting out of the new (Xcode 27 b1
    /// / macOS 27 beta) default that auto-converts conv layouts on the GPU. A/B
    /// knob for the macOS-27 NaN-isolation matrix; default false.
    private let disableAutoLayoutConversion: Bool

    /// When non-nil, the raw value forced onto every compile descriptor's
    /// `reducedPrecisionFastMath` (reconstructed under `#available(macOS 26.0)`).
    /// A/B knob for the macOS-27 NaN-isolation matrix; default nil leaves the
    /// descriptor default (`.none`). See `init`.
    private let reducedPrecisionFastMathRaw: UInt?

    /// A/B knob: when true, `computeValueBaselineGPU` blocks with
    /// `waitUntilCompleted` after committing the value-baseline forward, instead of
    /// the default non-blocking commit that overlaps it with the following training
    /// step. The default (false) relies on cross-command-buffer ordering on the
    /// shared queue + single-buffered (`resultTD`/board `entry`) caches — if that
    /// ordering is not actually honored on macOS 27, the training step reads a
    /// partially-written / next-step-clobbered vBaseline. Setting this true fully
    /// serializes the baseline so that hazard cannot occur, isolating it as a cause.
    var blockingValueBaseline: Bool = false

    // MARK: Initialization

    /// Build the network. Default `bnMode = .inference` keeps the existing
    /// behavior for play / forward-pass demos; pass `.training` to build a
    /// copy whose BN layers compute fresh batch stats on every forward pass
    /// (used by ChessTrainer for accurate training-step benchmarks).
    ///
    /// `policyTailPrecision` defaults to the process's value
    /// (`PolicyTailPrecision.process`).
    init(arch: NetworkArchitecture = .current, bnMode: BNMode = .inference,
         initialization: WeightInitialization,
         policyTailPrecision: PolicyTailPrecision = .process,
         disableAutoLayoutConversion: Bool = false,
         reducedPrecisionFastMathRaw: UInt? = nil,
         analysisTaps: Bool = false) throws {
        try arch.validate()
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            throw ChessNetworkError.metalNotSupported
        }
        guard let cmdQueue = mtlDevice.makeCommandQueue() else {
            throw ChessNetworkError.commandQueueCreationFailed
        }

        let infOrTrain = bnMode == .inference ? "inf" : "train"
        cmdQueue.label = "init - \(infOrTrain)"
        metalDevice = mtlDevice
        commandQueue = cmdQueue
        graphDevice = MPSGraphDevice(mtlDevice: mtlDevice)
        let g = MPSGraph()
        graph = g
        self.arch = arch
        self.initialization = initialization
        self.awaitingWeightLoad = SyncBox(initialization == .overwrittenByLoad)
        self.policyTailPrecision = policyTailPrecision
        // macOS 27 / Xcode 27 b1 made automatic NCHW->NHWC layout conversion for
        // GPU convolutions the default (`MPSGraphCompilationDescriptor.convertLayoutToNHWC`
        // is now a deprecated no-op; the opt-out is the new
        // `disableAutoLayoutConversion`). The lever is a *compilation-descriptor*
        // property, applied at every `graph.compile` site below — not on the graph
        // object. Stored here so each compile honors it; default false leaves the
        // production behavior unchanged. Used by the macOS-27 NaN-isolation A/B.
        self.disableAutoLayoutConversion = disableAutoLayoutConversion
        // When non-nil, every `graph.compile` site sets the descriptor's
        // `reducedPrecisionFastMath` to this raw value. `.none` (0) forbids MPSGraph
        // from taking reduced-precision shortcuts (FP16 winograd-transform
        // intermediates, FP32->FP19/TF32 operand narrowing) — its documented default
        // is already `.none`, so this is a *force/verify* knob for the macOS-27
        // NaN-isolation A/B, not a behavior change. Stored as the enum's raw `UInt`
        // (not the enum itself) so the property is declarable on the app target,
        // whose deployment minimum predates the macOS-26 enum. nil leaves the
        // descriptor default untouched.
        self.reducedPrecisionFastMathRaw = reducedPrecisionFastMathRaw

        // Every trainable and running-stat variable is created in the compute
        // dtype.
        let computeDType = Self.mpsDataType(for: arch)

        let conv1x1 = try Self.makeConv1x1Descriptor()
        let stemConvDescriptor = try Self.makeConvDescriptor(kernelSize: arch.stemConvKernelSize)

        // Numerics-audit taps: nil (the production default) records nothing and
        // adds no graph nodes, so every production graph is unchanged.
        let taps: AnalysisTapRecorder? = analysisTaps ? AnalysisTapRecorder() : nil

        // Hands every conv / FC weight its initial values by plan name and
        // checks the build covers the plan exactly (see `TensorInitializer`).
        let initializer = TensorInitializer(initialization: initialization, architecture: arch)


        // Input: [batch, inputPlanes, 8, 8]. The placeholder is fp32 — the
        // CPU feeds raw fp32 board planes (no host-side bf16 narrowing) and
        // the narrowing to the compute dtype runs on the GPU via the cast
        // below, the graph's first op. On a `.float32` build the cast is the
        // identity and is elided. This same placeholder is fed by the
        // self-play / arena inference paths *and* by the trainer's board
        // feed, so both write fp32 (see ChessTrainer.feedsForBatch).
        let input = g.placeholder(
            shape: [-1, NSNumber(value: arch.inputPlanes), 8, 8],
            dataType: .float32,
            name: "board_input"
        )
        inputPlaceholder = input
        let computeInput = (Self.mpsDataType(for: arch) == .float32)
            ? input
            : g.cast(input, to: Self.mpsDataType(for: arch), name: "board_input_cast")

        // Build the forward graph into local arrays and assign to
        // `self.*` after everything is set. We can't use `self` methods
        // until all stored properties are initialized, so the layer
        // builders are static and take the arrays as inout.
        //
        // - `trainables`: conv/FC weights + biases + BN gamma/beta.
        //   Gradient-updated by ChessTrainer's SGD assigns.
        // - `runningStats`: BN running mean/var variables. Used directly
        //   by inference-mode BN; EMA-updated by training-mode BN.
        // - `runningStatsAssigns`: EMA-update assign ops for training
        //   mode. Empty in inference mode.
        var trainables: [MPSGraphTensor] = []
        var shouldDecay: [Bool] = []
        var runningStats: [MPSGraphTensor] = []
        var runningStatsAssigns: [MPSGraphOperation] = []
        var batchMeans: [MPSGraphTensor] = []
        var batchVars: [MPSGraphTensor] = []

        // --- Stem: same-padded conv (inputPlanes -> first group's width) -> BN -> ReLU ---

        let stemOutC = arch.stemOutputChannels
        let stemWeights = g.variable(
            with: try initializer.weightData(
                "stem.conv.weight",
                nativeShape: [stemOutC, arch.inputPlanes, arch.stemConvKernelSize, arch.stemConvKernelSize],
                distribution: .heNormal,
                dataType: computeDType
            ),
            shape: [
                NSNumber(value: stemOutC),
                NSNumber(value: arch.inputPlanes),
                NSNumber(value: arch.stemConvKernelSize),
                NSNumber(value: arch.stemConvKernelSize)
            ],
            dataType: computeDType,
            name: "stem_conv_weights"
        )
        trainables.append(stemWeights)
        shouldDecay.append(true)
        var x = g.convolution2D(
            computeInput,
            weights: stemWeights,
            descriptor: stemConvDescriptor,
            name: "stem_conv"
        )
        x = Self.batchNorm(
            graph: g, input: x, channels: stemOutC, gammaInitValue: Self.standardBatchNormGamma, name: "stem_bn", taps: taps, bnMode: bnMode, dataType: Self.mpsDataType(for: arch),
            variableDataType: computeDType,
            trainables: &trainables,
            shouldDecay: &shouldDecay,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssigns,
            batchMeans: &batchMeans,
            batchVars: &batchVars
        )
        // Stem activation only for post-activation towers; the pre-activation
        // tower defers the first nonlinearity to block 0's `BN -> act` (the
        // stem BN still bounds x_0, the skip highway's starting value).
        if arch.hasStemActivation {
            x = Self.activation(g, x, arch, name: "stem_act")
        }

        // Hold the stem output for the optional feature skip (a single long concat
        // skip into the heads). This is the post-stem-BN[-act] tensor `x0`; the tower
        // loop is about to overwrite `x`, so the reference is captured now. Nil-cost
        // when the feature is off (just a retained graph-node reference).
        let stemOutputTensor = x
        taps?.record("stem_output", x)

        // --- Dropout scaffolding (training-mode graphs only) ---
        //
        // Built unconditionally into every training-mode block at the WRN
        // slot (after the conv2-side BN+activation, before conv2). The rate is a
        // fed placeholder, and every consumer other than the training step binds
        // the preallocated zero built below, so the node is an exact identity by
        // default; see the property docs.
        // The channel mask shape [N, C, 1, 1] is derived at runtime from the
        // stem output's shape (batch is dynamic in this graph).
        var dropoutRatePh: MPSGraphTensor?
        var dropoutStateVar: MPSGraphTensor?
        var dropoutStateFeed: MPSGraphTensor?
        var dropoutSeedOp: MPSGraphOperation?
        var dropoutAdvanceOp: MPSGraphOperation?
        var dropoutMaskShapes: [Int: MPSGraphTensor] = [:]
        var dropoutState: MPSGraphTensor?
        var dropoutRateZeroTD: MPSGraphTensorData?
        var dropoutRateLiveNDA: MPSNDArray?
        var dropoutRateLiveTD: MPSGraphTensorData?
        if bnMode == .training {
            // Rate is fed, not stored, so each execution picks its own — see
            // `dropoutRateFeedPlaceholder` for why a variable was unworkable.
            dropoutRatePh = g.placeholder(
                shape: [1], dataType: .float32, name: "dropout_rate"
            )
            let rateDesc = MPSNDArrayDescriptor(dataType: .float32, shape: [1])
            // Two distinct buffers: an immutable zero for the dropout-free
            // consumers, and a mutable one the trainer rewrites. Sharing one
            // would reintroduce exactly the coupling this change removes.
            let zeroNDA = MPSNDArray(device: mtlDevice, descriptor: rateDesc)
            var zero = Float(0)
            zeroNDA.writeBytes(&zero, strideBytes: nil)
            dropoutRateZeroTD = MPSGraphTensorData(zeroNDA)
            let liveNDA = MPSNDArray(device: mtlDevice, descriptor: rateDesc)
            var live = Float(0)
            liveNDA.writeBytes(&live, strideBytes: nil)
            dropoutRateLiveNDA = liveNDA
            dropoutRateLiveTD = MPSGraphTensorData(liveNDA)
            let stateVar = g.variable(
                with: Data(count: DropoutPhiloxState.wordCount * MemoryLayout<Int32>.size),
                shape: [NSNumber(value: DropoutPhiloxState.wordCount)], dataType: .int32, name: "dropout_rng_state"
            )
            dropoutStateVar = stateVar
            let stateFeed = g.placeholder(
                shape: [NSNumber(value: DropoutPhiloxState.wordCount)], dataType: .int32,
                name: "dropout_rng_seed_state"
            )
            dropoutStateFeed = stateFeed
            dropoutSeedOp = g.assign(stateVar, tensor: stateFeed, name: "dropout_rng_seed_assign")
            dropoutState = stateVar
            // Mask shape [N, C, 1, 1]: the dynamic batch dim is read from the
            // INPUT PLACEHOLDER's shape, never from a tensor downstream of
            // trainable weights. `shapeOf` has no gradient function, so
            // attaching it to e.g. the stem output adds a consumer autodiff
            // cannot sum over — MPSGraph then fails to produce gradients for
            // everything upstream ("Couldn't get gradient Tensor for tensor
            // of op: stem_conv_weights", hard assertion). No trainable is
            // upstream of the placeholder, so this branch is invisible to
            // the backward pass.
            let inputShape = g.shapeOf(input, name: "dropout_input_shape")
            let batchOnly = g.sliceTensor(
                inputShape, dimension: 0, start: 0, length: 1, name: "dropout_shape_n"
            )
            let spatialOnes = g.constant(1.0, shape: [2], dataType: .int32)
            // One mask shape tensor per distinct block width (the mask is
            // applied at the WRN slot, where the tensor already runs at the
            // block's OUTPUT width), shared by every block at that width.
            // Sorted for deterministic graph-construction order.
            for width in Set(arch.expandedBlocks.map(\.channels)).sorted() {
                let channelDim = g.constant(
                    Double(width), shape: [1], dataType: .int32
                )
                dropoutMaskShapes[width] = g.concatTensors(
                    [batchOnly, channelDim, spatialOnes], dimension: 0,
                    name: "dropout_mask_shape_c\(width)"
                )
            }
        }

        // --- Tower: residual blocks, walking the flat expanded list and ---
        // --- threading the incoming width (`expandedBlocks` is the      ---
        // --- engine's only view of the group structure)                 ---

        var towerInC = stemOutC
        for (i, spec) in arch.expandedBlocks.enumerated() {
            // Final-block feature skip (concatDirect): concat the stem source onto
            // this block's input; the block's own width-transition machinery (conv1
            // `inC` + the 1×1 skip projection that appears when inC != outC) absorbs
            // it. `blockSkipExtraInputChannels` is non-zero ONLY for the last block
            // under a routed skip-to-final-block, so every other block is unchanged.
            let blockSkipExtra = arch.blockSkipExtraInputChannels(blockIndex: i)
            let blockInput = blockSkipExtra > 0
                ? g.concatTensors([x, stemOutputTensor], dimension: 1, name: "block\(i)_skip_concat")
                : x
            x = try Self.residualBlock(
                graph: g,
                arch: arch,
                spec: spec,
                input: blockInput,
                inChannels: towerInC + blockSkipExtra,
                blockIndex: i,
                bnMode: bnMode,
                taps: taps,
                variableDataType: computeDType,
                initializer: initializer,
                dropoutRate: dropoutRatePh,
                dropoutMaskShape: dropoutMaskShapes[spec.channels],
                dropoutRngState: &dropoutState,
                trainables: &trainables,
                shouldDecay: &shouldDecay,
                runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssigns,
                batchMeans: &batchMeans,
                batchVars: &batchVars
            )
            towerInC = spec.channels
            taps?.record("block\(i)_output", x)
        }

        // Advance the RNG state variable to the post-tower stream position
        // once per training step (the trainer compiles this into its SGD
        // assign ops). Only meaningful when the blocks actually consumed
        // randomness — i.e. the chained state differs from the variable.
        if let stateVar = dropoutStateVar, let finalState = dropoutState, finalState !== stateVar {
            dropoutAdvanceOp = g.assign(stateVar, tensor: finalState, name: "dropout_rng_advance")
        }
        self.dropoutRateFeedPlaceholder = dropoutRatePh
        self.dropoutRateZeroTensorData = dropoutRateZeroTD
        self.dropoutRateLiveNDArray = dropoutRateLiveNDA
        self.dropoutRateLiveTensorData = dropoutRateLiveTD
        self.dropoutRngStateVariable = dropoutStateVar
        self.dropoutRngStateFeedPlaceholder = dropoutStateFeed
        self.dropoutRngSeedOp = dropoutSeedOp
        self.dropoutRngAdvanceOp = dropoutAdvanceOp

        // --- Tower-end normalization (pre-activation only) ---
        //
        // Each pre-activation block ends in a bare conv-add on a clean identity
        // skip, so the tower output is an un-normalized, never-activated linear
        // accumulation — normalize + activate it here for the heads. A
        // post-activation tower ends each block in an activation, so its output
        // is already conditioned and no tower-end BN exists (matches v3).
        if arch.hasTowerEndBN {
            x = Self.batchNorm(
                graph: g, input: x, channels: arch.towerOutputChannels, gammaInitValue: Self.standardBatchNormGamma, name: "tower_final_bn", taps: taps, bnMode: bnMode, dataType: Self.mpsDataType(for: arch),
                variableDataType: computeDType,
                trainables: &trainables,
                shouldDecay: &shouldDecay,
                runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssigns,
                batchMeans: &batchMeans,
                batchVars: &batchVars
            )
            x = Self.activation(g, x, arch, name: "tower_final_act")
        }
        taps?.record("tower_output", x)

        // --- Feature skip into the heads ---
        //
        // Concat along the channel axis (dim 1, NCHW). Two mutually-exclusive modes:
        //  • concatDirect — a routed head reads concat([tower_out, source]) and its
        //    own first conv (widened to towerC + sourceC) absorbs it. No new tensors;
        //    trainable/running-stat ordering is untouched.
        //  • compressConvBNReLU — ONE shared node f = act(BN(Conv1×1(concat → towerC)))
        //    feeds the routed heads at width towerC. Its conv + BN append HERE — after
        //    the tower-end BN, before the heads — exactly mirroring `weightTensorPlan`,
        //    so the index-aligned export/load contract holds.
        // The concat is built once and shared by both routed heads; unrouted heads
        // (and off configs) read the bare tower output `x`.
        let policyHeadInput: MPSGraphTensor
        let valueHeadInput: MPSGraphTensor
        if arch.featureSkipUsesCompressNode {
            let concat = g.concatTensors([x, stemOutputTensor], dimension: 1, name: "feature_skip_concat")
            let fusionInC = arch.featureSkipCompressInputChannels
            let towerC = arch.towerOutputChannels
            let fusionConvW = g.variable(
                with: try initializer.weightData(
                    "feature_skip.conv.weight", nativeShape: [towerC, fusionInC, 1, 1],
                    distribution: .heNormal, dataType: computeDType),
                shape: [NSNumber(value: towerC), NSNumber(value: fusionInC), 1, 1],
                dataType: computeDType, name: "feature_skip_conv_weights")
            trainables.append(fusionConvW)
            shouldDecay.append(true)
            var f = g.convolution2D(
                concat, weights: fusionConvW, descriptor: conv1x1, name: "feature_skip_conv")
            f = Self.batchNorm(
                graph: g, input: f, channels: towerC, gammaInitValue: Self.standardBatchNormGamma, name: "feature_skip_bn", taps: taps, bnMode: bnMode,
                dataType: Self.mpsDataType(for: arch), variableDataType: computeDType,
                trainables: &trainables, shouldDecay: &shouldDecay,
                runningStats: &runningStats, runningStatsAssignOps: &runningStatsAssigns,
                batchMeans: &batchMeans, batchVars: &batchVars)
            f = Self.activation(g, f, arch, name: "feature_skip_act")
            policyHeadInput = arch.featureSkipToPolicyHead ? f : x
            valueHeadInput = arch.featureSkipToValueHead ? f : x
        } else if arch.featureSkipEnabled, arch.featureSkipFusion == .concatDirect,
                  arch.featureSkipToPolicyHead || arch.featureSkipToValueHead {
            let concat = g.concatTensors([x, stemOutputTensor], dimension: 1, name: "feature_skip_concat")
            policyHeadInput = arch.featureSkipToPolicyHead ? concat : x
            valueHeadInput = arch.featureSkipToValueHead ? concat : x
        } else {
            policyHeadInput = x
            valueHeadInput = x
        }

        // --- Policy head ---

        let policy = try Self.policyHead(
            graph: g, arch: arch, input: policyHeadInput, inputChannels: arch.policyHeadInputChannels,
            descriptor: conv1x1, bnMode: bnMode, taps: taps, tailPrecision: policyTailPrecision,
            variableDataType: computeDType,
            initializer: initializer,
            trainables: &trainables,
            shouldDecay: &shouldDecay,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssigns,
            batchMeans: &batchMeans,
            batchVars: &batchVars
        )
        policyOutput = policy.output
        policyHeadFinalWeights = policy.finalWeights

        // --- Value head ---

        let valueHeadOut = try Self.valueHead(
            graph: g, arch: arch, input: valueHeadInput, inputChannels: arch.valueHeadInputChannels,
            descriptor: conv1x1, bnMode: bnMode, taps: taps,
            variableDataType: computeDType,
            initializer: initializer,
            trainables: &trainables,
            shouldDecay: &shouldDecay,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssigns,
            batchMeans: &batchMeans,
            batchVars: &batchVars
        )
        valueOutput = valueHeadOut.scalar
        valueLogits = valueHeadOut.logits
        valueProbs = valueHeadOut.probs

        // Audit readbacks: every tap plus the head outputs, each widened to fp32
        // so one reader serves every compute dtype. Empty unless taps are on.
        if let taps {
            taps.record("policy_logits", policy.output)
            taps.record("value_logits", valueHeadOut.logits)
            taps.record("value_probs", valueHeadOut.probs)
            analysisTapReadbacks = taps.taps.map { tap in
                let readback = tap.tensor.dataType == .float32
                    ? tap.tensor
                    : g.cast(tap.tensor, to: .float32, name: "analysis_tap_\(tap.name)_f32")
                return (name: tap.name, tensor: readback)
            }
        } else {
            analysisTapReadbacks = []
        }

        trainableVariables = trainables
        trainableShouldDecay = shouldDecay
        bnRunningStatsVariables = runningStats
        bnRunningStatsAssignOps = runningStatsAssigns

        // Enforce the plan↔builder contract, which everything downstream assumes
        // and nothing previously checked.
        //
        // `weightTensorPlan()` is the sole authority for what the tensor at each
        // position is CALLED and what layout transform it gets, but values are
        // paired to it purely by INDEX: `SafetensorsModelIO.encode` labels
        // `weights[i]` with `plan[i].name` and reshapes it per `plan[i].kind`,
        // and `loadWeights` assigns positionally. So if this builder's append
        // order ever drifts from the plan, every model written from that point
        // is mislabeled on disk and every load lands in the wrong variable —
        // with no error raised anywhere, because the only guard downstream is an
        // element count. Checking it here, once, at the one place both sides
        // exist, is what makes the naming honest.
        try Self.validateAgainstPlan(
            trainables: trainables, runningStats: runningStats, arch: arch
        )
        // And every conv / FC weight got its initial values under its plan
        // name, which is what the per-tensor init seeds are keyed on.
        try initializer.verifyEveryRandomTensorInitialized()
        randomTensorRoles = initializer.randomTensorRoles
        bnBatchMeanTensors = batchMeans
        bnBatchVarTensors = batchVars

        // Build per-variable weight-load infrastructure. For each
        // persistent variable (trainable + running stat), add one
        // placeholder with matching shape and one assign op that writes
        // the placeholder's value back into the variable. `loadWeights`
        // feeds these placeholders at runtime and runs all assigns as
        // a single graph execution — no new graph, no variable-by-
        // variable round trips.
        var loadPlaceholders: [MPSGraphTensor] = []
        var loadAssignOps: [MPSGraphOperation] = []
        var loadNDArrays: [MPSNDArray] = []
        var loadTensorData: [MPSGraphTensorData] = []
        let persistent = trainables + runningStats
        loadPlaceholders.reserveCapacity(persistent.count)
        loadAssignOps.reserveCapacity(persistent.count)
        loadNDArrays.reserveCapacity(persistent.count)
        loadTensorData.reserveCapacity(persistent.count)
        for v in persistent {
            guard let shape = v.shape else {
                throw ChessNetworkError.variableShapeMissing(v.operation.name)
            }
            // Persistent weight/stat variables are stored in the compute
            // dtype, so the load placeholder + NDArray use it too.
            let ph = g.placeholder(
                shape: shape,
                dataType: computeDType,
                name: "\(v.operation.name)_load"
            )
            let assignOp = g.assign(v, tensor: ph, name: "\(v.operation.name)_load_assign")
            loadPlaceholders.append(ph)
            loadAssignOps.append(assignOp)

            let desc = MPSNDArrayDescriptor(dataType: computeDType, shape: shape)
            let nda = MPSNDArray(device: mtlDevice, descriptor: desc)
            loadNDArrays.append(nda)
            loadTensorData.append(MPSGraphTensorData(nda))
        }
        weightLoadPlaceholders = loadPlaceholders
        weightLoadAssignOps = loadAssignOps
        weightLoadNDArrays = loadNDArrays
        weightLoadTensorData = loadTensorData

        // Reusable `[1, inputPlanes, 8, 8]` inference input ND array + wrapper.
        // `evaluate(board:)` writes new floats directly into this
        // array's storage each call and feeds the same wrapper — no
        // per-move MPS allocations.
        // fp32 storage — the input boundary is always fp32 (the GPU cast
        // narrows to the compute dtype). Feeds the fp32 `inputPlaceholder`.
        let inputDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [1, NSNumber(value: arch.inputPlanes), 8, 8]
        )
        let inputND = MPSNDArray(device: mtlDevice, descriptor: inputDesc)
        inputND.label = "inputND"
        inferenceInputNDArray = inputND
        inferenceInputTensorData = MPSGraphTensorData(inputND)

        // Zero-filled dummy input shared by exportWeights / loadWeights.
        // `inputDesc` is fp32 (the input boundary), so this must write fp32
        // bytes — `writeFloats` would narrow to the compute dtype via
        // `makeWeightData` and then over-read that narrower buffer when
        // `writeBytes` copies the fp32 array's full size.
        let dummyND = MPSNDArray(device: mtlDevice, descriptor: inputDesc)
        dummyND.label = "dummyND"
        Self.writeFloatsFP32(
            [Float](repeating: 0, count: 1 * arch.inputPlanes * Self.boardSize * Self.boardSize),
            into: dummyND
        )
        dummyInferenceInputTensorData = MPSGraphTensorData(dummyND)

        // Cache the feeds dict and target tensor list so the per-move
        // inference path doesn't rebuild them. Both are immutable — the
        // ND array backing `inferenceInputTensorData` is written
        // through `writeBytes` on the same underlying storage every
        // call.
        // On a training-mode graph the single-position `evaluate` path is a
        // DIAGNOSTIC read of the trainer's network, so it binds rate 0 and sees
        // a clean forward — previously it inherited whatever rate the training
        // step had set, quietly masking the diagnostics it exists to report.
        var builtInferenceFeeds: [MPSGraphTensor: MPSGraphTensorData] = [
            inputPlaceholder: inferenceInputTensorData
        ]
        if let ratePlaceholder = dropoutRatePh, let zeroRate = dropoutRateZeroTD {
            builtInferenceFeeds[ratePlaceholder] = zeroRate
        }
        inferenceFeeds = builtInferenceFeeds
        // Every head output is fp32 (the heads' fp32 tails), so all three are
        // read back as raw fp32 with no conversion.
        inferenceTargets = [policyOutput, valueOutput, valueProbs]

        // Raw-pointer readback scratches for the policy logits and
        // value scalar. UnsafeMutablePointer avoids Swift array CoW so
        // `evaluate(board:)` can hand a stable UnsafeBufferPointer back
        // to the caller without triggering an allocation.
        let policyScratch = UnsafeMutablePointer<Float>.allocate(capacity: Self.policySize)
        policyScratch.initialize(repeating: 0, count: Self.policySize)
        inferencePolicyScratchPtr = policyScratch
        let valueScratch = UnsafeMutablePointer<Float>.allocate(capacity: 1)
        valueScratch.initialize(repeating: 0, count: 1)
        inferenceValueScratchPtr = valueScratch
        let valueProbsScratch = UnsafeMutablePointer<Float>.allocate(capacity: arch.valueHeadClasses)
        valueProbsScratch.initialize(repeating: 0, count: arch.valueHeadClasses)
        inferenceValueProbsScratchPtr = valueProbsScratch
    }

    deinit {
        inferencePolicyScratchPtr.deinitialize(count: Self.policySize)
        inferencePolicyScratchPtr.deallocate()
        inferenceValueScratchPtr.deinitialize(count: 1)
        inferenceValueScratchPtr.deallocate()
        inferenceValueProbsScratchPtr.deinitialize(count: arch.valueHeadClasses)
        inferenceValueProbsScratchPtr.deallocate()
        if let ptr = batchPolicyScratchPtr {
            ptr.deinitialize(count: batchPolicyScratchCapacity)
            ptr.deallocate()
        }
        if let ptr = batchValueScratchPtr {
            ptr.deinitialize(count: batchValueScratchCapacity)
            ptr.deallocate()
        }
        if let ptr = batchValueProbsScratchPtr {
            ptr.deinitialize(count: batchValueProbsScratchCapacity)
            ptr.deallocate()
        }
    }

    // MARK: - Inference

    /// Evaluate a single board position and hand the policy/value
    /// readback to `consume` synchronously, inside the network's
    /// `executionQueue` work block and inside an `autoreleasepool`.
    ///
    /// `consume` receives an `UnsafeBufferPointer<Float>` of `policySize`
    /// policy logits plus the derived scalar value `p_win − p_loss ∈
    /// [−1, +1]` (the W/D/L head's softmax · `[+1, 0, −1]`). The buffer
    /// aliases the network's shared inference scratch and is valid only
    /// for the duration of the closure call — copy any bytes that need
    /// to outlive the closure (e.g. into a caller-owned destination)
    /// before returning.
    ///
    /// `consume` is non-throwing by contract. If `consume` is invoked,
    /// it runs to completion before this method returns; if the network
    /// itself throws (shape mismatch, output missing) before reaching
    /// the closure, `consume` is never invoked.
    ///
    /// Concurrent callers are safe: the forward pass and the `consume` call
    /// that reads the shared scratch run inside one work block on the serial
    /// `executionQueue`, so another caller's pass cannot overwrite the
    /// scratch until this caller's `consume` has returned.
    ///
    /// - Parameter board: `arch.inputPlanes` × `ChessNetwork.boardSize` ×
    ///   `ChessNetwork.boardSize` floats in NCHW order (planes, rows, cols).
    func evaluate(
        board: [Float],
        consume: @Sendable @escaping (UnsafeBufferPointer<Float>, Float) -> Void
    ) async throws {
        try await enqueue {
            try self.internalEvaluate(board: board, consume: consume)
        }
    }

    private func internalEvaluate(
        board: UnsafeBufferPointer<Float>,
        consume: (UnsafeBufferPointer<Float>, Float) -> Void
    ) throws {
        try internalEvaluateCore(board: board) { policy, value, _ in
            consume(policy, value)
        }
    }

    /// The single-position forward pass behind both `evaluate` variants. One
    /// `graph.run` produces the policy logits, the value scalar and the W/D/L
    /// softmax (`inferenceTargets` always includes `valueProbs`). The W/D/L
    /// triple is read back only if `consume` calls the `readValueDistribution`
    /// accessor it is handed, so the scalar-only path does exactly the
    /// readbacks it always did. The accessor is valid only during `consume`.
    private func internalEvaluateCore(
        board: UnsafeBufferPointer<Float>,
        consume: (
            UnsafeBufferPointer<Float>,
            Float,
            _ readValueDistribution: () throws -> (win: Float, draw: Float, loss: Float)
        ) throws -> Void
    ) throws {
        try requireLoadedWeights("evaluate")
        let expected = 1 * arch.inputPlanes * Self.boardSize * Self.boardSize
        guard board.count == expected else {
            throw ChessNetworkError.boardSizeMismatch(expected: expected, got: board.count)
        }

        // Wrap graph.run + readback + consume in an autoreleasepool so the
        // `[MPSGraphTensor: MPSGraphTensorData]` result dictionary,
        // the MPSNDArray handles reached through `.mpsndarray()`, and
        // any other autoreleased Obj-C objects allocated inside MPS
        // are released on the way out instead of piling up until the
        // enclosing Swift Task finishes. Without this, long-running
        // inference loops accumulate unbounded VM-range allocations
        // (observed as ~420 GB virtual against ~5 GB resident during
        // multi-hour Play-and-Train sessions) and the main thread
        // spends progressively more time in the deferred drain.
        try autoreleasepool {
            Self.writeInferenceInput(board, into: inferenceInputNDArray)

            let results = graph.run(
                with: commandQueue,
                feeds: inferenceFeeds,
                targetTensors: inferenceTargets,
                targetOperations: nil
            )

            guard let policyData = results[policyOutput] else {
                throw ChessNetworkError.outputMissing("policy")
            }
            guard let valueData = results[valueOutput] else {
                throw ChessNetworkError.outputMissing("value")
            }

            Self.readFloatsFP32(from: policyData, into: inferencePolicyScratchPtr, count: Self.policySize)
            Self.readFloatsFP32(from: valueData, into: inferenceValueScratchPtr, count: 1)

            try consume(
                UnsafeBufferPointer(start: inferencePolicyScratchPtr, count: Self.policySize),
                inferenceValueScratchPtr.pointee,
                {
                    guard let probsData = results[self.valueProbs] else {
                        throw ChessNetworkError.outputMissing("valueProbs")
                    }
                    Self.readFloatsFP32(
                        from: probsData,
                        into: self.inferenceValueProbsScratchPtr,
                        count: self.arch.valueHeadClasses
                    )
                    return Self.valueDistribution(
                        fromProbs: UnsafePointer(self.inferenceValueProbsScratchPtr),
                        style: self.arch.valueHeadStyle
                    )
                }
            )
        }
    }

    /// Evaluate a single board and hand `consume` the policy logits together
    /// with the value head's full `(p_win, p_draw, p_loss)` distribution —
    /// the same one forward pass as `evaluate(board:consume:)`, with one more
    /// small readback, so callers that want W/D/L on every move (the Lichess
    /// bot) never pay for the second pass `evaluateValueDistribution` runs.
    /// Same contract as `evaluate(board:consume:)`: the policy buffer aliases
    /// shared scratch and is valid only during the closure call.
    func evaluateWithValueDistribution(
        board: [Float],
        consume: @Sendable @escaping (UnsafeBufferPointer<Float>, (win: Float, draw: Float, loss: Float)) -> Void
    ) async throws {
        try await enqueue {
            try board.withUnsafeBufferPointer { buf in
                try self.internalEvaluateCore(board: buf) { policy, _, readValueDistribution in
                    consume(policy, try readValueDistribution())
                }
            }
        }
    }

    /// Project the value head's softmax readback onto `(p_win, p_draw,
    /// p_loss)`. The one place both W/D/L paths turn raw probabilities into
    /// the triple, so the scalar-head projection can't drift between them.
    ///
    /// - `.wdlSoftmax`: `probs` holds 3 elements in slot order win, draw, loss.
    /// - `.scalarTanh`: `probs` holds a single `v = p_win − p_loss ∈ [−1, +1]`.
    ///   A scalar head carries no separable draw mass, so the triple keeps
    ///   `win − loss = v` and reports draw as 0.
    static func valueDistribution(
        fromProbs probs: UnsafePointer<Float>,
        style: ValueHeadStyle
    ) -> (win: Float, draw: Float, loss: Float) {
        switch style {
        case .wdlSoftmax:
            return (win: probs[0], draw: probs[1], loss: probs[2])
        case .scalarTanh:
            let v = probs[0]
            return (win: max(v, 0), draw: 0, loss: max(-v, 0))
        }
    }

    private func internalEvaluate(
        board: [Float],
        consume: (UnsafeBufferPointer<Float>, Float) -> Void
    ) throws {
        try board.withUnsafeBufferPointer { buf in
            try internalEvaluate(board: buf, consume: consume)
        }
    }

    /// Forward-only pass returning the value head's W/D/L softmax
    /// `(p_win, p_draw, p_loss)` for a single position. Separate from
    /// `evaluate(board:consume:)` because the universal inference path
    /// returns only the *derived scalar* `p_win − p_loss`; this is for
    /// diagnostics (the candidate-test probe / Run Forward Pass panel)
    /// that want the full distribution. Runs on the network's
    /// `executionQueue`, inside an `autoreleasepool`, like `evaluate`.
    /// Returns immediately after the readback — does not invoke a
    /// closure (the three floats are cheap to return by value).
    func evaluateValueDistribution(board: [Float]) async throws -> (win: Float, draw: Float, loss: Float) {
        try await enqueue {
            try self.internalEvaluateValueDistribution(board: board)
        }
    }

    private func internalEvaluateValueDistribution(board: [Float]) throws -> (win: Float, draw: Float, loss: Float) {
        try requireLoadedWeights("evaluateValueDistribution")
        let expected = 1 * arch.inputPlanes * Self.boardSize * Self.boardSize
        guard board.count == expected else {
            throw ChessNetworkError.boardSizeMismatch(expected: expected, got: board.count)
        }
        return try board.withUnsafeBufferPointer { buf in
            try autoreleasepool {
                Self.writeInferenceInput(buf, into: inferenceInputNDArray)
                let results = graph.run(
                    with: commandQueue,
                    feeds: inferenceFeeds,
                    targetTensors: [valueProbs],
                    targetOperations: nil
                )
                guard let probsData = results[valueProbs] else {
                    throw ChessNetworkError.outputMissing("valueProbs")
                }
                Self.readFloatsFP32(from: probsData, into: inferenceValueProbsScratchPtr, count: arch.valueHeadClasses)
                return Self.valueDistribution(
                    fromProbs: UnsafePointer(inferenceValueProbsScratchPtr),
                    style: arch.valueHeadStyle
                )
            }
        }
    }

    /// Evaluate a batch of `count` board positions in one graph execution
    /// and hand the policy / value / W-D-L readback to `consume`
    /// synchronously, inside the network's `executionQueue` work block
    /// and inside an `autoreleasepool`.
    ///
    /// `consume` receives three `UnsafeBufferPointer<Float>`s that alias
    /// this network's batched readback scratch:
    /// - `policy` holds `count * policySize` raw logits laid out
    ///   position-major (slot `i` starts at `i * policySize`).
    /// - `values` holds `count` scalars in [-1, +1] (the derived
    ///   `p_win - p_loss`).
    /// - `wdlProbs` holds `count * valueHeadClasses` softmax
    ///   probabilities in slot order `[win, draw, loss]`, position-
    ///   major (slot `i` starts at `i * 3`). Consumed by the self-play
    ///   `DrawWatchTracker`; arena and tests ignore via `_ in`.
    /// All three buffers are valid only for the duration of the closure
    /// call. Callers that need any bytes past the closure must copy them
    /// out (typically into a caller-owned destination such as
    /// `MPSChessPlayer`'s policy scratch).
    ///
    /// `consume` is non-throwing by contract. If `consume` is invoked,
    /// it runs to completion before this method returns; if the network
    /// itself throws (shape mismatch, output missing) before reaching
    /// the closure, `consume` is never invoked.
    ///
    /// The first call at a given `count` lazily allocates a per-batch-
    /// size input `MPSNDArray` + feeds dict that is reused on all later
    /// calls at that size. Policy and value readback scratches grow to
    /// the largest batch size ever requested. This is the self-play
    /// hot path — steady-state batches allocate nothing.
    ///
    /// - Parameters:
    ///   - batchBoards: `count * inputPlanes * 8 * 8` floats in
    ///                  NCHW order, one position after another.
    ///   - count: number of positions in the batch; must be >= 1.
    ///   - consume: non-throwing closure invoked once with the policy
    ///              logits, the derived value scalars, and the per-
    ///              position W/D/L softmax probabilities (`count * 3`
    ///              floats, position-major, slot order
    ///              `[win, draw, loss]`). Callers that don't need the
    ///              W/D/L distribution can ignore the third argument
    ///              (`{ policy, values, _ in ... }`).
    func evaluateBatched(
        batchBoards: [Float],
        count: Int,
        consume: @Sendable @escaping (
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>
        ) -> Void
    ) async throws {
        try await enqueue {
            try self.internalEvaluate(batchBoards: batchBoards, count: count, consume: consume)
        }
    }

    /// Pointer-flavored batched evaluate. The caller owns
    /// `batchBoardsPointer` and is responsible for keeping the
    /// underlying buffer alive across the `await` (typically by
    /// holding it as an instance field of the caller's driver). This
    /// avoids the per-fire `[Float](repeating: …)` allocation the
    /// `[Float]`-flavored overload's `withUnsafeBufferPointer` would
    /// require if the caller had to convert pointer → Array.
    ///
    /// Sendable handling: `UnsafePointer<Float>` is not Sendable, so
    /// the pointer + count are wrapped in `BatchBoardSource` (an
    /// `@unchecked Sendable` carrier) to cross the `enqueue` closure
    /// boundary. The caller's lifetime contract above keeps the
    /// pointer valid for the entire `await`.
    ///
    /// - Parameters:
    ///   - batchBoardsPointer: base pointer to a contiguous run of
    ///     `floatCount = count * inputPlanes * 8 * 8` floats.
    ///   - floatCount: total element count addressed by the pointer.
    ///                 Must equal `count * inputPlanes * 8 * 8`;
    ///                 enforced inside `internalEvaluate`.
    ///   - count: number of positions in the batch; must be >= 1.
    ///   - consume: identical contract to the `[Float]` overload.
    func evaluateBatched(
        batchBoardsPointer: UnsafePointer<Float>,
        floatCount: Int,
        count: Int,
        consume: @Sendable @escaping (
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>
        ) -> Void
    ) async throws {
        let source = BatchBoardSource(pointer: batchBoardsPointer, floatCount: floatCount)
        try await enqueue {
            let buf = UnsafeBufferPointer<Float>(start: source.pointer, count: source.floatCount)
            try self.internalEvaluate(batchBoards: buf, count: count, consume: consume)
        }
    }

    private func internalEvaluate(
        batchBoards: UnsafeBufferPointer<Float>,
        count: Int,
        consume: (
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>
        ) -> Void
    ) throws {
        try requireLoadedWeights("evaluateBatched")
        // Validation runs synchronously on `executionQueue` after the
        // [Float] has been pinned via `withUnsafeBufferPointer`. `count`
        // and `batchBoards.count` are stable for the rest of the body
        // because Swift value-type semantics isolate our captured copy
        // from the caller's binding (COW), and the buffer pointer's
        // count is set at construction and never derived dynamically.
        guard count >= 1 else {
            throw ChessNetworkError.boardSizeMismatch(expected: arch.inputPlanes * Self.boardSize * Self.boardSize, got: 0)
        }
        let expected = count * arch.inputPlanes * Self.boardSize * Self.boardSize
        guard batchBoards.count == expected else {
            throw ChessNetworkError.boardSizeMismatch(expected: expected, got: batchBoards.count)
        }

        let entry = batchInputEntry(for: count)
        let policyPtr = ensureBatchPolicyScratch(count: count)
        let valuePtr = ensureBatchValueScratch(count: count)
        let valueProbsPtr = ensureBatchValueProbsScratch(count: count)

        // Same autoreleasepool discipline as `evaluate(board:)` — the
        // self-play batched path is the highest-frequency graph.run
        // site in the app (roughly once per barrier cycle at ~20-40
        // Hz across concurrent slots), so a missed pool drain here
        // dominates the long-session VM bloat.
        try autoreleasepool {
            Self.writeInferenceInput(batchBoards, into: entry.ndArray)

            // Compiled-executable path (highest-frequency forward pass). Bind
            // inputs in the executable's own feed order; the result array comes
            // back in compiled targetTensors order, so zip restores the
            // tensor→data dictionary the readback below expects. See
            // MPSGraphExecutableTrainingEquivalenceTests for the equivalence and
            // ordering proofs, and GPU_UTILIZATION_PLAN.md.
            let executable = inferenceExecutable(for: count, feeds: entry.feeds)
            guard let feedTensors = executable.feedTensors else {
                throw ChessNetworkError.outputMissing("inference feed tensors")
            }
            var inputs: [MPSGraphTensorData] = []
            inputs.reserveCapacity(feedTensors.count)
            for tensor in feedTensors {
                guard let data = entry.feeds[tensor] else {
                    throw ChessNetworkError.outputMissing("inference feed binding")
                }
                inputs.append(data)
            }
            let resultArray = executable.run(
                with: commandQueue,
                inputs: inputs,
                results: nil,
                executionDescriptor: nil
            )
            let results = Dictionary(uniqueKeysWithValues: zip(inferenceTargets, resultArray))

            guard let policyData = results[policyOutput] else {
                throw ChessNetworkError.outputMissing("policy")
            }
            guard let valueData = results[valueOutput] else {
                throw ChessNetworkError.outputMissing("value")
            }
            guard let valueProbsData = results[valueProbs] else {
                throw ChessNetworkError.outputMissing("valueProbs")
            }

            Self.readFloatsFP32(from: policyData, into: policyPtr, count: count * Self.policySize)
            Self.readFloatsFP32(from: valueData, into: valuePtr, count: count)
            Self.readFloatsFP32(from: valueProbsData, into: valueProbsPtr, count: count * arch.valueHeadClasses)

            consume(
                UnsafeBufferPointer(start: policyPtr, count: count * Self.policySize),
                UnsafeBufferPointer(start: valuePtr, count: count),
                UnsafeBufferPointer(start: valueProbsPtr, count: count * arch.valueHeadClasses)
            )
        }
    }

    private func internalEvaluate(
        batchBoards: [Float],
        count: Int,
        consume: (
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>
        ) -> Void
    ) throws {
        try batchBoards.withUnsafeBufferPointer { buf in
            try internalEvaluate(batchBoards: buf, count: count, consume: consume)
        }
    }

    private func batchInputEntry(for count: Int) -> BatchInputEntry {
        if let cached = batchInputCache[count] {
            return cached
        }
        // fp32 storage — feeds the fp32 `inputPlaceholder`; the GPU cast
        // narrows to the compute dtype.
        let desc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [NSNumber(value: count), NSNumber(value: arch.inputPlanes), 8, 8]
        )
        let nda = MPSNDArray(device: metalDevice, descriptor: desc)
        nda.label = "inference.input[\(count)]"
        let tensorData = MPSGraphTensorData(nda)
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [inputPlaceholder: tensorData]
        // Every consumer of this entry — value baseline, BN warmup, batched
        // inference — must run dropout-free, so they all bind rate 0. The
        // training step is the sole binder of the LIVE rate, and it does so by
        // overwriting this key on a copy of its own cached feed dict
        // (`ChessTrainer.runPreparedStep`) — not by building feeds from scratch.
        // That override is deliberate: it means a caller who assembles feeds by
        // copying the zero-rate pattern from here still trains at the live rate
        // rather than silently training dropout-free. Nil on inference graphs,
        // which have no dropout nodes to feed.
        if let ratePlaceholder = dropoutRateFeedPlaceholder,
           let zeroRate = dropoutRateZeroTensorData {
            feeds[ratePlaceholder] = zeroRate
        }
        let entry = BatchInputEntry(ndArray: nda, tensorData: tensorData, feeds: feeds)
        batchInputCache[count] = entry
        return entry
    }

    /// Compile + cache the inference executable for batch size `count`. Concrete
    /// feed shapes are taken from the live feed tensor data (the placeholder
    /// carries a `-1` batch dim; the executable is specialized to this count).
    /// Targets are the policy / value / valueProbs outputs; no target operations
    /// (inference is read-only). `.level1` trades compile time for execution
    /// time — paid once per batch size, amortized to nothing. Must run on
    /// `executionQueue`.
    private func inferenceExecutable(
        for count: Int,
        feeds: [MPSGraphTensor: MPSGraphTensorData]
    ) -> MPSGraphExecutable {
        if let cached = inferenceExecutables[count] {
            return cached
        }
        var feedShapes: [MPSGraphTensor: MPSGraphShapedType] = [:]
        feedShapes.reserveCapacity(feeds.count)
        for (placeholder, tensorData) in feeds {
            feedShapes[placeholder] = MPSGraphShapedType(
                shape: tensorData.shape,
                dataType: placeholder.dataType
            )
        }
        let desc = MPSGraphCompilationDescriptor()
        desc.optimizationLevel = .level1
        if disableAutoLayoutConversion, #available(macOS 27.0, *) {
            desc.disableAutoLayoutConversion()
        }
        if let reducedPrecisionFastMathRaw, #available(macOS 26.0, *) {
            desc.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMath(rawValue: reducedPrecisionFastMathRaw)
        }
        // Compile on a large stack: MPSGraph's compile traverses the full op DAG
        // depth-first, so a deep tower can overflow the default dispatch-worker
        // stack here just as the trainer's autodiff does. See
        // `withLargeBuildStack`. The wrapped compile cannot itself throw, so the
        // only possible error is the helper's unreachable thread-failure.
        let executable: MPSGraphExecutable
        do {
            executable = try withLargeBuildStack {
                self.graph.compile(
                    with: MPSGraphDevice(mtlDevice: self.metalDevice),
                    feeds: feedShapes,
                    targetTensors: self.inferenceTargets,
                    targetOperations: nil,
                    compilationDescriptor: desc
                )
            }
        } catch {
            preconditionFailure("ChessNetwork inference compile failed on large-stack thread: \(error)")
        }
        inferenceExecutables[count] = executable
        return executable
    }

    /// GPU→GPU value baseline. Runs a value-only forward on this network's
    /// current weights and leaves the per-position `v(s)` (fp32, shape
    /// `[count, 1]`) in a network-owned GPU buffer, handed to `consume` WITHOUT a
    /// CPU readback. The trainer feeds that buffer straight into the training
    /// step's `vBaseline` placeholder, eliminating the `Array(valuesBuf)` copy +
    /// staging re-write the old `evaluateBatched` baseline path did. Numerically
    /// identical to that path — the same fp32 `valueOutput` — just no CPU round
    /// trip. The result buffer is reused per call; safe because the trainer
    /// drives this serially (phase 2 fully completes before phase 3 reads it).
    /// The most recent value-baseline command buffer. It is committed WITHOUT a
    /// host wait (it overlaps the trainer's step), so we cannot inspect its
    /// status synchronously at commit time. Instead we hold the reference and
    /// check its (by-then settled) status on the *next* baseline call — a failed
    /// baseline would otherwise silently feed garbage `v(s)` into training.
    /// Accessed only on `executionQueue`.
    private var lastBaselineCommandBuffer: MTLCommandBuffer?

    func computeValueBaselineGPU(
        batchBoards: [Float],
        count: Int,
        consume: @Sendable @escaping (MPSGraphTensorData) -> Void
    ) async throws {
        try await enqueue {
            try self.internalComputeValueBaseline(batchBoards: batchBoards, count: count, consume: consume)
        }
    }

    private func internalComputeValueBaseline(
        batchBoards: [Float],
        count: Int,
        consume: (MPSGraphTensorData) -> Void
    ) throws {
        try requireLoadedWeights("computeValueBaseline")
        guard count >= 1 else {
            throw ChessNetworkError.boardSizeMismatch(expected: arch.inputPlanes * Self.boardSize * Self.boardSize, got: 0)
        }
        let expected = count * arch.inputPlanes * Self.boardSize * Self.boardSize
        guard batchBoards.count == expected else {
            throw ChessNetworkError.boardSizeMismatch(expected: expected, got: batchBoards.count)
        }
        // The previous baseline was committed without a host wait; by now it has
        // settled (the trainer's dependent step waited on it). If it faulted,
        // surface that before issuing another step on top of poisoned state.
        if let previous = lastBaselineCommandBuffer, previous.status == .error {
            lastBaselineCommandBuffer = nil
            throw ChessNetworkError.gpuCommandFailed(
                stage: "value baseline",
                status: previous.status,
                error: previous.error?.localizedDescription
            )
        }
        let entry = batchInputEntry(for: count)
        let resultTD = valueBaselineResultTD(for: count)
        try autoreleasepool {
            batchBoards.withUnsafeBufferPointer { buf in
                Self.writeInferenceInput(buf, into: entry.ndArray)
            }
            let executable = valueBaselineExecutable(for: count, feeds: entry.feeds)
            guard let feedTensors = executable.feedTensors else {
                throw ChessNetworkError.outputMissing("value-baseline feed tensors")
            }
            var inputs: [MPSGraphTensorData] = []
            inputs.reserveCapacity(feedTensors.count)
            for tensor in feedTensors {
                guard let data = entry.feeds[tensor] else {
                    throw ChessNetworkError.outputMissing("value-baseline feed binding")
                }
                inputs.append(data)
            }
            // Non-blocking baseline forward (GPU_UTILIZATION_PLAN.md Phase 3,
            // step 1). Encode into our own command buffer and `commit` WITHOUT
            // `waitUntilCompleted`, so the baseline's GPU work overlaps the
            // training-step encode that follows on the trainer queue instead of
            // stalling the CPU here. Correctness rests on enqueue ordering +
            // tracked-resource hazard tracking: the training step reads `resultTD`
            // (the vBaseline) from a command buffer committed *after* this one on
            // the same `commandQueue`, so Metal serializes the read behind this
            // write — the trainer never touches `resultTD` on the CPU, so no host
            // wait is needed. Caller-owned result buffer (never `results: nil`)
            // per the proven concurrent-encode contract; `consume` only hands the
            // buffer reference onward (the data fills on the GPU, in order).
            guard let mtlCommandBuffer = commandQueue.makeCommandBuffer() else {
                throw ChessNetworkError.outputMissing("value-baseline command buffer")
            }
            let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: mtlCommandBuffer)
            _ = executable.encode(
                to: mpsCommandBuffer,
                inputs: inputs,
                results: [resultTD],
                executionDescriptor: nil
            )
            mpsCommandBuffer.commit()
            // A/B: fully serialize the baseline before the training step reads its
            // output. Defeats any cross-command-buffer read-before-write hazard on
            // the single-buffered `resultTD` if the default overlap is unsafe on
            // macOS 27. Off by default (keeps the non-blocking overlap).
            if blockingValueBaseline {
                mpsCommandBuffer.waitUntilCompleted()
            }
            // Retain for the next call's status check (see `lastBaselineCommandBuffer`).
            lastBaselineCommandBuffer = mtlCommandBuffer
            consume(resultTD)
        }
    }

    /// Compile + cache the value-only baseline executable for batch size `count`.
    /// Feed shapes are taken from the live board feed (the placeholder carries a
    /// `-1` batch dim). Target is just `valueOutput`; no target operations
    /// (read-only). Must run on `executionQueue`.
    private func valueBaselineExecutable(
        for count: Int,
        feeds: [MPSGraphTensor: MPSGraphTensorData]
    ) -> MPSGraphExecutable {
        if let cached = valueBaselineExecutables[count] {
            return cached
        }
        var feedShapes: [MPSGraphTensor: MPSGraphShapedType] = [:]
        feedShapes.reserveCapacity(feeds.count)
        for (placeholder, tensorData) in feeds {
            feedShapes[placeholder] = MPSGraphShapedType(
                shape: tensorData.shape,
                dataType: placeholder.dataType
            )
        }
        let desc = MPSGraphCompilationDescriptor()
        desc.optimizationLevel = .level1
        if disableAutoLayoutConversion, #available(macOS 27.0, *) {
            desc.disableAutoLayoutConversion()
        }
        if let reducedPrecisionFastMathRaw, #available(macOS 26.0, *) {
            desc.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMath(rawValue: reducedPrecisionFastMathRaw)
        }

        // Large-stack compile — see `inferenceExecutable` / `withLargeBuildStack`.
        let executable: MPSGraphExecutable
        do {
            executable = try withLargeBuildStack {
                self.graph.compile(
                    with: MPSGraphDevice(mtlDevice: self.metalDevice),
                    feeds: feedShapes,
                    targetTensors: [self.valueOutput],
                    targetOperations: nil,
                    compilationDescriptor: desc
                )
            }
        } catch {
            preconditionFailure("ChessNetwork value-baseline compile failed on large-stack thread: \(error)")
        }
        valueBaselineExecutables[count] = executable
        return executable
    }

    /// Network-owned fp32 `[count, 1]` result buffer for the value baseline,
    /// matching `valueOutput`'s shape/dtype (fp32). Cached per batch size.
    private func valueBaselineResultTD(for count: Int) -> MPSGraphTensorData {
        if let cached = valueBaselineResultCache[count] {
            return cached
        }
        let desc = MPSNDArrayDescriptor(dataType: .float32, shape: [NSNumber(value: count), 1])
        let nda = MPSNDArray(device: metalDevice, descriptor: desc)
        nda.label = "vbaseline.result[\(count)]"
        let td = MPSGraphTensorData(nda)
        valueBaselineResultCache[count] = td
        return td
    }

    private func ensureBatchPolicyScratch(count: Int) -> UnsafeMutablePointer<Float> {
        let needed = count * Self.policySize
        if let ptr = batchPolicyScratchPtr, batchPolicyScratchCapacity >= needed {
            return ptr
        }
        if let old = batchPolicyScratchPtr {
            old.deinitialize(count: batchPolicyScratchCapacity)
            old.deallocate()
        }
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: needed)
        ptr.initialize(repeating: 0, count: needed)
        batchPolicyScratchPtr = ptr
        batchPolicyScratchCapacity = needed
        return ptr
    }

    private func ensureBatchValueScratch(count: Int) -> UnsafeMutablePointer<Float> {
        if let ptr = batchValueScratchPtr, batchValueScratchCapacity >= count {
            return ptr
        }
        if let old = batchValueScratchPtr {
            old.deinitialize(count: batchValueScratchCapacity)
            old.deallocate()
        }
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: count)
        ptr.initialize(repeating: 0, count: count)
        batchValueScratchPtr = ptr
        batchValueScratchCapacity = count
        return ptr
    }

    private func ensureBatchValueProbsScratch(count: Int) -> UnsafeMutablePointer<Float> {
        let needed = count * arch.valueHeadClasses
        if let ptr = batchValueProbsScratchPtr, batchValueProbsScratchCapacity >= needed {
            return ptr
        }
        if let old = batchValueProbsScratchPtr {
            old.deinitialize(count: batchValueProbsScratchCapacity)
            old.deallocate()
        }
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: needed)
        ptr.initialize(repeating: 0, count: needed)
        batchValueProbsScratchPtr = ptr
        batchValueProbsScratchCapacity = needed
        return ptr
    }

    // MARK: - Weight Transfer

    /// Snapshot all persistent network state as flat float arrays, one
    /// per variable. Ordered trainables-first (conv/FC weights + biases
    /// + BN gamma/beta) then BN running stats (mean + variance per BN
    /// layer). Element order within each array is the variable's stored
    /// row-major order. Feed directly into `loadWeights(_:)` on a
    /// sibling network of identical architecture to copy state across.
    ///
    /// This is how ChessTrainer's internal network's learned weights +
    /// EMA running stats make their way into the inference network
    /// during Play and Train. No gradient, no forward pass, just a read
    /// of the current variable state.
    func exportWeights() async throws -> [[Float]] {
        try await enqueue {
            try self.internalExportWeights()
        }
    }

    /// `exportWeights()` for a caller on another serial work queue that must
    /// read the weights and its own state with nothing in between (the
    /// trainer pairing weights with its step count). Blocks the calling
    /// thread until this network's queue has run the export, so it must
    /// never be called from the cooperative pool or from this network's own
    /// queue.
    func exportWeightsBlocking() throws -> [[Float]] {
        try executionQueue.sync {
            try internalExportWeights()
        }
    }

    private func internalExportWeights() throws -> [[Float]] {
        try requireLoadedWeights("exportWeights")
        let allVars = trainableVariables + bnRunningStatsVariables

        // Serialize this variable read against the trainer's concurrent SGD
        // weight-write (it runs from the trainer queue, not this one). See
        // `weightAccessLock`. Held across the `graph.run` read below; the
        // subsequent host-side readback only touches the result snapshots.
        weightAccessLock.wait()
        defer { weightAccessLock.signal() }

        // MPSGraph requires feeds for every placeholder in the graph,
        // even ones unreachable from the target tensors. We feed the
        // board_input placeholder with a pre-built zero-filled dummy
        // (and nothing for the weight-load placeholders, which are safe
        // to omit because no run-time target reaches them). targetTensors
        // are the variables themselves — reading them doesn't require
        // any compute ancestor, so no forward pass actually runs.
        //
        // Autoreleasepool-wrapped for the same reason as
        // `evaluate(board:)` — the results dictionary and its
        // MPSGraphTensorData values are autoreleased and should drain
        // before we return to the caller, which may itself be invoked
        // from a long-lived background Task (arena start / promotion
        // flows, checkpoint autosave) without a natural pool boundary.
        return try autoreleasepool {
            let results = graph.run(
                with: commandQueue,
                feeds: [inputPlaceholder: dummyInferenceInputTensorData],
                targetTensors: allVars,
                targetOperations: nil
            )

            var out: [[Float]] = []
            out.reserveCapacity(allVars.count)
            for v in allVars {
                guard let data = results[v] else {
                    throw ChessNetworkError.outputMissing(v.operation.name)
                }
                let count = try Self.elementCount(of: v)
                // Persistent weight/stat variables live in the compute
                // dtype.
                out.append(Self.readFloats(from: data, count: count, dataType: Self.mpsDataType(for: arch)))
            }
            return out
        }
    }

    /// Overwrite all persistent network state from a snapshot produced
    /// by `exportWeights()` on a network of the same architecture. The
    /// input must contain exactly one float array per variable (in the
    /// same order `exportWeights` uses), with correct element counts.
    /// Mismatches throw.
    ///
    /// Runs a single graph execution: feeds each variable's new values
    /// through its per-variable load placeholder (built once at init
    /// time) and runs the corresponding assign ops as target
    /// operations. After return, the network's variables hold the new
    /// values; subsequent `evaluate(board:)` calls see the loaded state.
    /// A load whose GPU work does not complete throws
    /// `ChessNetworkError.gpuCommandFailed`, and a network built
    /// `overwrittenByLoad` then still refuses to be used.
    func loadWeights(_ weights: [[Float]]) async throws {
        try await enqueue {
            try self.internalLoadWeights(weights)
        }
    }

    private func internalLoadWeights(_ weights: [[Float]]) throws {
        let allVars = trainableVariables + bnRunningStatsVariables
        guard weights.count == allVars.count else {
            throw ChessNetworkError.weightLoadMismatch(
                "expected \(allVars.count) tensors, got \(weights.count)"
            )
        }

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        feeds.reserveCapacity(allVars.count + 1)

        // Dummy feed for the board_input placeholder — MPSGraph wants
        // every graph placeholder fed even though the target operations
        // below never consume board_input.
        feeds[inputPlaceholder] = dummyInferenceInputTensorData

        for (i, v) in allVars.enumerated() {
            let expectedCount = try Self.elementCount(of: v)
            guard weights[i].count == expectedCount else {
                throw ChessNetworkError.weightLoadMismatch(
                    "variable \(v.operation.name): expected \(expectedCount) floats, got \(weights[i].count)"
                )
            }
            // Persistent variables live in the compute dtype; the load
            // NDArrays were sized to match.
            Self.writeFloats(weights[i], into: weightLoadNDArrays[i], dataType: Self.mpsDataType(for: arch))
            feeds[weightLoadPlaceholders[i]] = weightLoadTensorData[i]
        }

        // Encode into a command buffer we own, instead of the high-level
        // `graph.run`, which hides its buffer and reports no status: a load
        // whose GPU work faults (out of memory, timeout, kernel error) would
        // otherwise return normally and clear the load gate below, marking a
        // network that still holds its zero-filled variables as loaded —
        // the silent garbage the gate exists to refuse. Encode + commit +
        // wait is what `graph.run` does internally.
        //
        // The encode needs at least one target tensor. Use the first
        // persistent variable as a dummy read — its value after the
        // assigns run is whatever we just wrote in, which we ignore.
        // Autoreleasepool-wrapped for the same reason as the other
        // graph.run sites in this file.
        guard let rootCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw ChessNetworkError.commandBufferCreationFailed(stage: Self.weightLoadStage)
        }
        let commandBuffer = MPSCommandBuffer(commandBuffer: rootCommandBuffer)
        autoreleasepool {
            _ = graph.encode(
                to: commandBuffer,
                feeds: feeds,
                targetTensors: [allVars[0]],
                targetOperations: weightLoadAssignOps,
                executionDescriptor: nil
            )
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        // MPS may have committed the root buffer early and continued in a
        // new one (`commitAndContinue`); the load completed only if both did.
        rootCommandBuffer.waitUntilCompleted()
        try Self.requireCompleted(
            status: rootCommandBuffer.status, error: rootCommandBuffer.error, stage: Self.weightLoadStage)
        try Self.requireCompleted(
            status: commandBuffer.commandBuffer.status, error: commandBuffer.commandBuffer.error,
            stage: Self.weightLoadStage)
        awaitingWeightLoad.value = false
    }

    /// The `gpuCommandFailed` stage a failed weight load reports.
    static let weightLoadStage = "weight load"

    /// Throws `gpuCommandFailed` unless a waited-for command buffer finished
    /// `.completed`. `waitUntilCompleted` returns whatever happened on the
    /// GPU, so a caller about to treat the buffer's work as done checks its
    /// status here first.
    static func requireCompleted(status: MTLCommandBufferStatus, error: Error?, stage: String) throws {
        guard status == .completed else {
            throw ChessNetworkError.gpuCommandFailed(stage: stage, status: status, error: error?.localizedDescription)
        }
    }

    // MARK: - BN Warmup

    /// Run one batched forward pass on `boards` and return the per-BN-
    /// layer batch_mean and batch_var, one entry per BN layer in build
    /// order (matching `bnBatchMeanTensors` / `bnBatchVarTensors` and
    /// the mean-then-variance interleaved order of
    /// `bnRunningStatsVariables`).
    ///
    /// Only meaningful on a `.training`-mode network — that's the mode
    /// in which `bnBatchMeanTensors` / `bnBatchVarTensors` are populated.
    /// Calling this on an `.inference`-mode network throws because the
    /// batch-stat tensors don't exist there. Pair this method with
    /// `loadBNRunningStats` on a sibling inference-mode network to
    /// prime its running stats from one real-distribution forward pass
    /// without waiting for the EMA to converge over hundreds of steps.
    ///
    /// Each returned `[Float]` has shape `[1, C, 1, 1]` flattened —
    /// element count equals the channel count of that BN layer.
    func computeBatchStats(
        boards: [Float],
        count: Int
    ) async throws -> (means: [[Float]], vars: [[Float]]) {
        try await enqueue {
            try self.internalComputeBatchStats(boards: boards, count: count)
        }
    }

    private func internalComputeBatchStats(
        boards: [Float],
        count: Int
    ) throws -> (means: [[Float]], vars: [[Float]]) {
        try requireLoadedWeights("computeBatchStats")
        guard !bnBatchMeanTensors.isEmpty else {
            throw ChessNetworkError.outputMissing(
                "computeBatchStats: bnBatchMeanTensors is empty — this method requires bnMode = .training"
            )
        }
        guard count >= 1 else {
            throw ChessNetworkError.boardSizeMismatch(
                expected: arch.inputPlanes * Self.boardSize * Self.boardSize, got: 0
            )
        }
        let expected = count * arch.inputPlanes * Self.boardSize * Self.boardSize
        guard boards.count == expected else {
            throw ChessNetworkError.boardSizeMismatch(expected: expected, got: boards.count)
        }

        let entry = batchInputEntry(for: count)

        return try autoreleasepool {
            Self.writeInferenceInput(boards, into: entry.ndArray)
            // Targets: every BN layer's batch_mean and batch_var.
            // Order: all means first, then all vars — caller splits.
            let targets = bnBatchMeanTensors + bnBatchVarTensors
            let results = graph.run(
                with: commandQueue,
                feeds: entry.feeds,
                targetTensors: targets,
                targetOperations: nil
            )
            var means: [[Float]] = []
            var vars_: [[Float]] = []
            means.reserveCapacity(bnBatchMeanTensors.count)
            vars_.reserveCapacity(bnBatchVarTensors.count)
            for t in bnBatchMeanTensors {
                guard let data = results[t] else {
                    throw ChessNetworkError.outputMissing(t.operation.name)
                }
                let n = try Self.elementCount(of: t)
                // Read by the tensor's own dtype: a head-tail BN (the policy
                // pre-BN) normalizes fp32 activations, so its batch stats are
                // fp32 while the tower's are the compute dtype.
                means.append(Self.readFloats(from: data, count: n, dataType: t.dataType))
            }
            for t in bnBatchVarTensors {
                guard let data = results[t] else {
                    throw ChessNetworkError.outputMissing(t.operation.name)
                }
                let n = try Self.elementCount(of: t)
                vars_.append(Self.readFloats(from: data, count: n, dataType: t.dataType))
            }
            return (means: means, vars: vars_)
        }
    }

    /// One analysis tap's values for a batch: fp32, flattened in the tensor's
    /// own layout (`shape`, batch first).
    struct AnalysisTapValues: Sendable {
        let name: String
        let shape: [Int]
        let values: [Float]
    }

    /// Run `count` boards through an audit network and read back every
    /// analysis tap (numerics audit). Throws on a network built without
    /// `analysisTaps`.
    func evaluateAnalysisTaps(boards: [Float], count: Int) async throws -> [AnalysisTapValues] {
        try await enqueue {
            try self.internalEvaluateAnalysisTaps(boards: boards, count: count)
        }
    }

    private func internalEvaluateAnalysisTaps(boards: [Float], count: Int) throws -> [AnalysisTapValues] {
        try requireLoadedWeights("evaluateAnalysisTaps")
        guard !analysisTapReadbacks.isEmpty else {
            throw ChessNetworkError.outputMissing("evaluateAnalysisTaps: this network was built without analysis taps")
        }
        guard count >= 1 else {
            throw ChessNetworkError.boardSizeMismatch(
                expected: arch.inputPlanes * Self.boardSize * Self.boardSize, got: 0
            )
        }
        let expected = count * arch.inputPlanes * Self.boardSize * Self.boardSize
        guard boards.count == expected else {
            throw ChessNetworkError.boardSizeMismatch(expected: expected, got: boards.count)
        }
        let entry = batchInputEntry(for: count)
        return try autoreleasepool {
            Self.writeInferenceInput(boards, into: entry.ndArray)
            let results = graph.run(
                with: commandQueue,
                feeds: entry.feeds,
                targetTensors: analysisTapReadbacks.map(\.tensor),
                targetOperations: nil
            )
            var out: [AnalysisTapValues] = []
            out.reserveCapacity(analysisTapReadbacks.count)
            for tap in analysisTapReadbacks {
                guard let data = results[tap.tensor] else {
                    throw ChessNetworkError.outputMissing("analysis tap \(tap.name)")
                }
                let shape = data.shape.map { $0.intValue }
                let elementCount = shape.reduce(1, *)
                out.append(AnalysisTapValues(
                    name: tap.name,
                    shape: shape,
                    values: Self.readFloatsFP32(from: data, count: elementCount)
                ))
            }
            return out
        }
    }

    /// Overwrite this network's BN running_mean and running_var
    /// variables from caller-supplied per-layer batch stats. Used by
    /// the construction-time warmup path: a fresh `.inference` network
    /// has its (0, 1) defaults replaced with stats computed by a
    /// sibling `.training` network's `computeBatchStats`. After this
    /// call returns, inference-mode forward passes through the deep
    /// residual tower see properly-normalized BN output instead of the
    /// effectively-identity normalization the (0, 1) defaults produce.
    ///
    /// `means.count` and `vars.count` must each equal the BN layer
    /// count; per-layer element counts must match the corresponding
    /// running-stat variable's shape. Mismatches throw.
    func loadBNRunningStats(
        means: [[Float]],
        vars: [[Float]]
    ) async throws {
        try await enqueue {
            try self.internalLoadBNRunningStats(means: means, vars: vars)
        }
    }

    private func internalLoadBNRunningStats(
        means: [[Float]],
        vars: [[Float]]
    ) throws {
        // Running-stat variables are stored interleaved mean-then-var
        // per layer. Validate counts before any feed work so an off-by-
        // one fails loudly.
        let layerCount = bnRunningStatsVariables.count / 2
        guard means.count == layerCount, vars.count == layerCount else {
            throw ChessNetworkError.weightLoadMismatch(
                "loadBNRunningStats: expected \(layerCount) mean+\(layerCount) var arrays, " +
                "got \(means.count) mean + \(vars.count) var"
            )
        }

        // Reuse the existing weight-load machinery. weightLoadPlaceholders
        // is ordered trainables-first then running-stats; the running
        // stats start at index trainableVariables.count and follow the
        // same mean-then-var interleaving as bnRunningStatsVariables.
        let nTrain = trainableVariables.count
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        feeds[inputPlaceholder] = dummyInferenceInputTensorData

        var assignOpsToRun: [MPSGraphOperation] = []
        assignOpsToRun.reserveCapacity(layerCount * 2)

        for layer in 0..<layerCount {
            let meanIdx = nTrain + 2 * layer
            let varIdx = meanIdx + 1
            let meanVar = bnRunningStatsVariables[2 * layer]
            let varVar = bnRunningStatsVariables[2 * layer + 1]
            let expectedMeanCount = try Self.elementCount(of: meanVar)
            let expectedVarCount = try Self.elementCount(of: varVar)
            guard means[layer].count == expectedMeanCount else {
                throw ChessNetworkError.weightLoadMismatch(
                    "loadBNRunningStats: layer \(layer) mean expected \(expectedMeanCount) floats, got \(means[layer].count)"
                )
            }
            guard vars[layer].count == expectedVarCount else {
                throw ChessNetworkError.weightLoadMismatch(
                    "loadBNRunningStats: layer \(layer) var expected \(expectedVarCount) floats, got \(vars[layer].count)"
                )
            }
            // Running-stat variables live in the compute dtype; their load
            // NDArrays match.
            Self.writeFloats(means[layer], into: weightLoadNDArrays[meanIdx], dataType: Self.mpsDataType(for: arch))
            Self.writeFloats(vars[layer], into: weightLoadNDArrays[varIdx], dataType: Self.mpsDataType(for: arch))
            feeds[weightLoadPlaceholders[meanIdx]] = weightLoadTensorData[meanIdx]
            feeds[weightLoadPlaceholders[varIdx]] = weightLoadTensorData[varIdx]
            assignOpsToRun.append(weightLoadAssignOps[meanIdx])
            assignOpsToRun.append(weightLoadAssignOps[varIdx])
        }

        autoreleasepool {
            _ = graph.run(
                with: commandQueue,
                feeds: feeds,
                targetTensors: [bnRunningStatsVariables[0]],
                targetOperations: assignOpsToRun
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

    /// Total scalar count in a tensor's statically-known shape.
    /// Throws if the tensor's shape is missing — which shouldn't happen
    /// for variables (they have concrete shapes at creation time).
    /// Exposed `internal` so `ChessTrainer` can size its velocity-tensor
    /// readback buffers identically.
    /// Verify the graph's weight variables line up, position for position, with
    /// `arch.weightTensorPlan()`.
    ///
    /// Compares SQUEEZED shapes: the plan records logical shapes (a BN gamma is
    /// `[C]`, torch convention) while the builder declares the same tensor
    /// broadcast-ready (`[1, C, 1, 1]`) so it applies across NCHW without a
    /// reshape. Those describe identical values in identical order, so raw
    /// shape equality would reject every correct network — but element-count
    /// equality is too weak, since it cannot distinguish `[in, out]` from
    /// `[out, in]`. Squeezing keeps the check sensitive to a transposed or
    /// re-factored tensor while tolerating the degenerate axes.
    ///
    /// What this still cannot catch: two tensors of the SAME squeezed shape
    /// swapping positions (a BN `weight`/`bias` pair is `[C]` either way).
    /// Closing that needs the builder and the plan to share one naming scheme;
    /// they currently do not (`block0_bn1_gamma` vs `blocks.0.bn1.weight`).
    private static func validateAgainstPlan(
        trainables: [MPSGraphTensor],
        runningStats: [MPSGraphTensor],
        arch: NetworkArchitecture
    ) throws {
        let plan = arch.weightTensorPlan()
        let allVars = trainables + runningStats
        guard allVars.count == plan.count else {
            throw ChessNetworkError.weightPlanMismatch(
                "graph built \(allVars.count) weight variables "
                + "(\(trainables.count) trainable + \(runningStats.count) BN running stat) "
                + "but the plan for this architecture describes \(plan.count). "
                + "The builder and NetworkArchitecture.weightTensorPlan() have diverged; "
                + "positional weight I/O (safetensors naming, loadWeights) is unsafe until they agree."
            )
        }
        for (i, spec) in plan.enumerated() {
            guard let rawShape = allVars[i].shape else {
                throw ChessNetworkError.variableShapeMissing(allVars[i].operation.name)
            }
            let actual = WeightTensorSpec.squeeze(rawShape.map(\.intValue))
            guard actual == spec.squeezedShape else {
                throw ChessNetworkError.weightPlanMismatch(
                    "position \(i): plan calls this '\(spec.name)' with shape \(spec.shape) "
                    + "but the graph variable '\(allVars[i].operation.name)' has shape "
                    + "\(rawShape.map(\.intValue)) (squeezed \(actual) vs \(spec.squeezedShape)). "
                    + "Every saved model labels values by plan position, so this tensor would be "
                    + "written under the wrong name and reloaded into the wrong variable."
                )
            }
        }
    }

    static func elementCount(of tensor: MPSGraphTensor) throws -> Int {
        guard let shape = tensor.shape else {
            throw ChessNetworkError.variableShapeMissing(tensor.operation.name)
        }
        return shape.reduce(1) { $0 * $1.intValue }
    }

    // MARK: - Convolution Descriptors

    /// Stem / residual-tower convolution: a `towerConvKernelSize`-square
    /// kernel, "same"-padded so it preserves the 8×8 board. Stride 1 with
    /// padding `(towerConvKernelSize - 1) / 2` on every side yields
    /// `out = in + 2·pad − kernel + 1 = in`. The padding is computed from
    /// the kernel constant, so this stays correct if the kernel changes —
    /// but only odd kernels give an integer symmetric pad (an even kernel
    /// needs `kernel − 1` total padding split unevenly, which this asserts
    /// against rather than silently mis-pad).
    /// Same-padded conv descriptor for an odd `kernelSize` (stride 1, symmetric pad).
    /// Per-conv now that conv1/conv2/stem can each carry a different kernel size.
    private static func makeConvDescriptor(kernelSize: Int) throws -> MPSGraphConvolution2DOpDescriptor {
        precondition(
            kernelSize % 2 == 1,
            "conv kernelSize must be odd for symmetric same-padding (got \(kernelSize))"
        )
        guard let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1,
            dilationRateInX: 1, dilationRateInY: 1,
            groups: 1,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else {
            throw ChessNetworkError.descriptorCreationFailed
        }
        let pad = (kernelSize - 1) / 2
        desc.paddingLeft = pad
        desc.paddingRight = pad
        desc.paddingTop = pad
        desc.paddingBottom = pad
        return desc
    }

    /// The tower-LEVEL hidden activation (stem act when post-act, tower-end,
    /// heads), selected by `arch.activationFunction`. Block main paths and SE
    /// FC1 use the overload below with their group's own function
    /// (`activationFunction` and `seActivation` respectively).
    private static func activation(
        _ graph: MPSGraph, _ x: MPSGraphTensor, _ arch: NetworkArchitecture, name: String
    ) -> MPSGraphTensor {
        activation(graph, x, arch.activationFunction, name: name)
    }

    /// SiLU = `x*sigmoid(x)`; GELU exact (erf-based); leaky ReLU with the
    /// fixed `ActivationFunction.leakyReLUNegativeSlope`. The SE gate
    /// (sigmoid) and value output (tanh/softmax) are structural and call their
    /// own ops directly. Internal (not private) so tests can check each
    /// function's values and gradients on a bare graph.
    static func activation(
        _ graph: MPSGraph, _ x: MPSGraphTensor, _ fn: ActivationFunction, name: String
    ) -> MPSGraphTensor {
        switch fn {
        case .relu:
            return graph.reLU(with: x, name: name)
        case .leakyRelu:
            return graph.leakyReLU(with: x, alpha: ActivationFunction.leakyReLUNegativeSlope, name: name)
        case .silu:
            let s = graph.sigmoid(with: x, name: "\(name)_sig")
            return graph.multiplication(x, s, name: name)
        case .gelu:
            // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
            let dt = x.dataType
            let invSqrt2 = graph.constant(0.7071067811865476, dataType: dt)
            let half = graph.constant(0.5, dataType: dt)
            let one = graph.constant(1.0, dataType: dt)
            let scaled = graph.multiplication(x, invSqrt2, name: "\(name)_scaled")
            let erf = graph.erf(with: scaled, name: "\(name)_erf")
            let onePlus = graph.addition(erf, one, name: "\(name)_1plus")
            let hx = graph.multiplication(half, x, name: "\(name)_halfx")
            return graph.multiplication(hx, onePlus, name: name)
        }
    }

    /// Map the model's compute precision to an `MPSDataType`.
    static func mpsDataType(for arch: NetworkArchitecture) -> MPSDataType {
        switch arch.computeDataType {
        case .float32: return .float32
        case .bFloat16: return .bFloat16
        case .float16: return .float16
        }
    }

    /// 1x1 convolution with no padding (used in policy and value heads).
    private static func makeConv1x1Descriptor() throws -> MPSGraphConvolution2DOpDescriptor {
        guard let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1,
            dilationRateInX: 1, dilationRateInY: 1,
            groups: 1,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else {
            throw ChessNetworkError.descriptorCreationFailed
        }
        desc.paddingLeft = 0
        desc.paddingRight = 0
        desc.paddingTop = 0
        desc.paddingBottom = 0
        return desc
    }

    // MARK: - Layer Builders

    /// The γ every BatchNorm starts at unless an init option says otherwise.
    static let standardBatchNormGamma: Float = 1

    /// The γ init of the last BN of a branch of `group` (post-activation
    /// `bn2`): 0 under `branch_output_init: zero_last_bn_gamma`, else the
    /// standard 1. The single rule the builder reads.
    static func lastBranchBatchNormGamma(_ group: BlockGroup) -> Float {
        switch group.branchOutputInit {
        case .standard: return standardBatchNormGamma
        case .zeroLastBNGamma: return 0
        }
    }

    /// Batch normalization. Behavior depends on `bnMode`:
    ///
    /// - `.inference`: uses the stored running statistics
    ///   (`running_mean`, `running_var`) to normalize. Initialized to
    ///   (0, 1) so a freshly-built inference network behaves as near-
    ///   identity until `loadWeights` populates the running stats with
    ///   EMA values from a trained sibling network.
    ///
    /// - `.training`: computes per-batch mean and variance over
    ///   (batch, height, width) on every forward pass and normalizes by
    ///   those, the standard BN training path. Also EMA-updates the
    ///   stored `running_mean` / `running_var` variables on each step
    ///   via assign ops appended to `runningStatsAssignOps`, so that
    ///   after enough training the running stats converge to typical
    ///   per-channel activation statistics — exactly what a sibling
    ///   inference network needs to produce results matching the
    ///   training-time forward pass. EMA momentum = 0.99 (i.e. tracks
    ///   roughly the last ~100 batches).
    ///
    /// gamma and beta are appended to `trainables` in both modes; γ starts at
    /// `gammaInitValue`, β at 0. Running-stat variables are appended to
    /// `runningStats` in both modes. Only `.training` appends to
    /// `runningStatsAssignOps`.
    private static func batchNorm(
        graph: MPSGraph,
        input: MPSGraphTensor,
        channels: Int,
        gammaInitValue: Float,
        name: String,
        taps: AnalysisTapRecorder?,
        bnMode: BNMode,
        dataType: MPSDataType,
        variableDataType: MPSDataType,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation],
        batchMeans: inout [MPSGraphTensor],
        batchVars: inout [MPSGraphTensor]
    ) -> MPSGraphTensor {
        taps?.record("\(name)_input", input)
        let ch = NSNumber(value: channels)
        // gamma/beta/running-stat variables live in `variableDataType` (the
        // compute dtype), but `normalize()` runs in `dataType`, the input's
        // dtype: the compute dtype everywhere except the policy pre-block in
        // the fp32 head tail. Where the two differ each is widened at point of
        // use (`inNormalizeDataType`); widening a stored value is exact.

        // gamma and beta are trainable in both modes. Every BN inits β=0 and
        // γ = `gammaInitValue`: 1 (standard) everywhere except the last BN
        // of a post-activation branch whose group asks for
        // `branch_output_init: zero_last_bn_gamma` (`lastBranchBatchNormGamma`).
        // That zero-γ init is an explicit per-group choice now, not the
        // default: in the pre-activation tower the per-block ReZero scalar α
        // owns depth-variance control instead, and unlike zero-γ it lets
        // every block contribute signal *and* gradient from step 1. See
        // `residualBlock`.
        let gamma = graph.variable(
            with: gammaInitValue == Self.standardBatchNormGamma
                ? onesData(count: channels, dataType: variableDataType)
                : makeWeightData([Float](repeating: gammaInitValue, count: channels), dataType: variableDataType),
            shape: [1, ch, 1, 1],
            dataType: variableDataType,
            name: "\(name)_gamma"
        )
        let beta = graph.variable(
            with: zerosData(count: channels, dataType: variableDataType),
            shape: [1, ch, 1, 1],
            dataType: variableDataType,
            name: "\(name)_beta"
        )
        trainables.append(gamma)
        shouldDecay.append(false)
        trainables.append(beta)
        shouldDecay.append(false)

        // Running stats exist in both modes — used directly for
        // normalization in `.inference`, used as the EMA target in
        // `.training`. Init to (0, 1) so a random-weight inference
        // network is near-identity until real stats get loaded in.
        let runningMean = graph.variable(
            with: zerosData(count: channels, dataType: variableDataType),
            shape: [1, ch, 1, 1],
            dataType: variableDataType,
            name: "\(name)_running_mean"
        )
        let runningVar = graph.variable(
            with: onesData(count: channels, dataType: variableDataType),
            shape: [1, ch, 1, 1],
            dataType: variableDataType,
            name: "\(name)_running_var"
        )
        runningStats.append(runningMean)
        runningStats.append(runningVar)

        // `normalize()`-dtype views of gamma/beta/running-stats: the
        // variables themselves in the tower, an fp32 widen in the policy head
        // tail.
        func inNormalizeDataType(_ variable: MPSGraphTensor) -> MPSGraphTensor {
            variable.dataType == dataType ? variable : graph.cast(variable, to: dataType, name: nil)
        }
        let gammaC = inNormalizeDataType(gamma)
        let betaC = inNormalizeDataType(beta)

        let meanTensor: MPSGraphTensor
        let varianceTensor: MPSGraphTensor

        switch bnMode {
        case .inference:
            // Inference normalize uses the running stats.
            meanTensor = inNormalizeDataType(runningMean)
            varianceTensor = inNormalizeDataType(runningVar)

        case .training:
            // Compute fresh batch statistics over (batch, height, width)
            // for each channel — axes [0, 2, 3] keep the channel dim,
            // reduce everything else. MPSGraph reductions keep the
            // reduced dims at size 1, so `bMean` / `bVar` have shape
            // [1, C, 1, 1] — compatible with normalize() and with the
            // running-stat variables below.
            let bMean = graph.mean(of: input, axes: [0, 2, 3], name: "\(name)_batch_mean")
            let bVar = graph.variance(of: input, axes: [0, 2, 3], name: "\(name)_batch_var")
            meanTensor = bMean
            varianceTensor = bVar
            // Surface the batch-stat tensors so a one-shot warmup pass
            // can read them out and prime an inference network's
            // running stats from them. See `bnBatchMeanTensors` /
            // `bnBatchVarTensors` for the contract.
            batchMeans.append(bMean)
            batchVars.append(bVar)

            // EMA update: new_running = 0.99 * old_running + 0.01 * batch.
            // Emitted as assign ops that the trainer runs alongside SGD
            // assigns, so every training step advances both the weights
            // and the running-stat estimate.
            //
            // The running-stat variables are `variableDataType`, but the
            // batch stats (`bMean`/`bVar`) take the input's dtype, which is
            // fp32 for the policy pre-block in the head tail. The EMA math
            // therefore runs in the variables' dtype, casting the batch stats
            // where they differ, so the assign target dtype matches; in the
            // tower every cast below is the identity.
            let emaDType = variableDataType
            let bMeanEMA = (bMean.dataType == emaDType) ? bMean : graph.cast(bMean, to: emaDType, name: nil)
            let bVarEMA = (bVar.dataType == emaDType) ? bVar : graph.cast(bVar, to: emaDType, name: nil)
            let momentum = graph.constant(0.99, dataType: emaDType)
            let oneMinusMomentum = graph.constant(0.01, dataType: emaDType)

            let scaledOldMean = graph.multiplication(momentum, runningMean, name: nil)
            let scaledNewMean = graph.multiplication(oneMinusMomentum, bMeanEMA, name: nil)
            let updatedMean = graph.addition(
                scaledOldMean, scaledNewMean, name: "\(name)_running_mean_update"
            )
            let assignMean = graph.assign(
                runningMean, tensor: updatedMean, name: "\(name)_running_mean_assign"
            )
            runningStatsAssignOps.append(assignMean)

            let scaledOldVar = graph.multiplication(momentum, runningVar, name: nil)
            let scaledNewVar = graph.multiplication(oneMinusMomentum, bVarEMA, name: nil)
            let updatedVar = graph.addition(
                scaledOldVar, scaledNewVar, name: "\(name)_running_var_update"
            )
            let assignVar = graph.assign(
                runningVar, tensor: updatedVar, name: "\(name)_running_var_assign"
            )
            runningStatsAssignOps.append(assignVar)
        }

        // `normalize` runs in the input's dtype (the compute dtype, or fp32
        // for the policy pre-block): `meanTensor`/`varianceTensor` are
        // already in it (running stats cast in `.inference`; the batch stats
        // in `.training`), and gamma/beta are cast above.
        return graph.normalize(
            input,
            mean: meanTensor,
            variance: varianceTensor,
            gamma: gammaC,
            beta: betaC,
            epsilon: 1e-5,
            name: name
        )
    }

    /// Channel-wise LayerNorm over the C dimension at each board square
    /// (ConvNeXt convention), with per-channel learnable γ/β. Unlike
    /// `batchNorm` it keeps **no running stats** and has **no train/eval
    /// branch**: the mean/variance are recomputed every forward over axis [1],
    /// so the op is byte-identical at training and inference. That is the whole
    /// point — it re-centers the clean-add residual stream every block (killing
    /// the highway mean-drift that degraded v4) without reintroducing the
    /// running-stat train/eval gap that BatchNorm carries. γ (init 1) and β
    /// (init 0) append to `trainables`, un-decayed, mirroring the
    /// `\(name).weight` / `\(name).bias` order in `weightTensorPlan`.
    private static func layerNorm(
        graph: MPSGraph,
        input: MPSGraphTensor,
        channels: Int,
        name: String,
        taps: AnalysisTapRecorder?,
        variableDataType: MPSDataType,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool]
    ) -> MPSGraphTensor {
        taps?.record("\(name)_input", input)
        let ch = NSNumber(value: channels)
        let gamma = graph.variable(
            with: onesData(count: channels, dataType: variableDataType),
            shape: [1, ch, 1, 1], dataType: variableDataType, name: "\(name)_gamma"
        )
        let beta = graph.variable(
            with: zerosData(count: channels, dataType: variableDataType),
            shape: [1, ch, 1, 1], dataType: variableDataType, name: "\(name)_beta"
        )
        trainables.append(gamma); shouldDecay.append(false)
        trainables.append(beta); shouldDecay.append(false)

        // Stats over the channel axis, per (n, h, w). MPSGraph keeps reduced
        // dims at size 1, so mean/variance are [N, 1, H, W] — broadcastable
        // against the [1, C, 1, 1] affine and the [N, C, H, W] input. Runs in
        // the compute dtype, as do γ/β.
        let mean = graph.mean(of: input, axes: [1], name: "\(name)_mean")
        let variance = graph.variance(of: input, axes: [1], name: "\(name)_var")
        return graph.normalize(
            input, mean: mean, variance: variance,
            gamma: gamma, beta: beta,
            epsilon: 1e-5, name: name
        )
    }

    /// One pre-activation (ResNet v2) residual block with a scale-and-bias
    /// SE module and a ReZero branch scalar:
    ///   out = input + α · F(input),   F = BN→ReLU→conv→BN→ReLU→conv→SE
    /// The skip is a **clean identity** — no activation on the sum — so the
    /// tower is an additive highway with un-gated gradient flow to depth.
    ///
    /// SE is *scale-and-bias*: squeeze (global avg pool) → FC1 128→32
    /// (He, ReLU) → FC2 32→256 (Glorot) → split into `gammas` and `betas`
    /// → `SE_out = sigmoid(gammas)·z + betas`. The sigmoid gates only the
    /// scale half (so attention stays bounded); the bias half is added
    /// linearly, letting the globally-pooled signal also inject a learned
    /// per-channel offset, not just attenuate. `z` is the raw conv2 output.
    ///
    /// `α` (`*_res_scale`) is a per-block trainable scalar, init the group's
    /// `rezeroAlphaInit` (the presets use `1/√numBlocks`), applied through
    /// the soft bound `C·tanh(α/C)` with C = the group's `rezeroAlphaCap`.
    /// With L additive branches of ~unit variance the tower variance grows
    /// ~L; the `1/√L` init holds it ~O(1) while still letting every block
    /// contribute signal *and* gradient from step 1 (unlike the old zero-γ
    /// init, whose branch was dead until gradient woke it). An init of 0 —
    /// the ReZero paper's — starts every block as an exact identity instead
    /// and lets α grow from there (see the comment at the α variable). It is
    /// excluded from weight decay. Reduction ratio = `seReductionRatio`.
    private static func residualBlock(
        graph: MPSGraph,
        arch: NetworkArchitecture,
        spec: BlockGroup,
        input: MPSGraphTensor,
        inChannels: Int,
        blockIndex: Int,
        bnMode: BNMode,
        taps: AnalysisTapRecorder?,
        variableDataType: MPSDataType,
        initializer: TensorInitializer,
        dropoutRate: MPSGraphTensor?,
        dropoutMaskShape: MPSGraphTensor?,
        dropoutRngState: inout MPSGraphTensor?,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation],
        batchMeans: inout [MPSGraphTensor],
        batchVars: inout [MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        let prefix = "block\(blockIndex)"
        // The block's tensors in `weightTensorPlan()` (and on disk) — the names
        // the per-tensor init seeds are keyed on.
        let planPrefix = "blocks.\(blockIndex)"
        // `inC` is the previous expanded block's width (stem output for block
        // 0); `outC` is this block's own. They differ exactly at a width
        // transition, where conv1 carries the remap on the branch and the
        // skip gets the 1×1 projection (below).
        let inC = inChannels
        let outC = spec.channels
        let conv1Desc = try makeConvDescriptor(kernelSize: spec.conv1KernelSize)
        let conv2Desc = try makeConvDescriptor(kernelSize: spec.conv2KernelSize)

        // Channel (spatial) dropout with inverted scaling, present only on
        // training-mode graphs (the caller passes nil rate/shape/state
        // otherwise). One fresh [N, C, 1, 1] uniform draw per block per
        // step, chained through `dropoutRngState`; keep-mask = (u >= rate);
        // survivors scaled by 1/(1-rate) so train-time expectations match
        // the dropout-free inference graphs. The rate arrives as a fed
        // placeholder, so each execution chooses it: the training step binds
        // the live rate and every other consumer binds 0, at which the whole
        // node is an exact identity — see `dropoutRateFeedPlaceholder`.
        //
        // WHY CHANNEL GRANULARITY: the mask broadcasts one coin per
        // (sample, channel) across the whole board, so a dropped feature
        // map goes dark as a unit. Per-element ("unit") dropout was
        // rejected because adjacent squares within one feature map are
        // strongly correlated on a board — a pinhole at one square is
        // trivially reconstructed from its neighbors, so the effective
        // regularization is far weaker than the nominal rate (the
        // SpatialDropout argument, Tompson et al.). DropBlock-style
        // contiguous patches were rejected because their niche — patches
        // larger than the correlation length but smaller than the map —
        // barely exists on a board this small; the interesting granularity
        // endpoints here are unit and channel, and channel is the one that
        // forces cross-channel redundancy (no feature may rely on one
        // fragile channel coalition). Whole-branch dropping (stochastic
        // depth / DropPath, the modern transformer-era favorite) is a
        // separate future axis gated on the ReZero multiply, not this
        // node. Decision record with paper references:
        // ARCHITECTURE_EXPANSION_PLAN.md (Feature 1).
        //
        // PLACEMENT (WRN slot, between the conv2-side activation and
        // conv2): the block's own BN statistics are computed on clean,
        // un-dropped activations (the dropout/BN variance-shift
        // disharmony only propagates forward); conv2 is the directly
        // regularized consumer; the skip path is never touched, so a
        // heavily-masked branch degrades toward a clean no-op through the
        // identity add. Note our SE module sits BELOW the mask, so SE
        // attention is computed from masked activations — a deliberate
        // deviation from plain WRN (SE learns missing-channel robustness
        // too); flag this when comparing against WRN-style results.
        var dropoutStateLocal = dropoutRngState
        func applyChannelDropout(_ h: MPSGraphTensor) throws -> MPSGraphTensor {
            guard let baseRate = dropoutRate,
                  let maskShape = dropoutMaskShape,
                  let stateIn = dropoutStateLocal else { return h }
            // Per-group dropout multiplier: effective rate = min(rate ×
            // multiplier, 0.95) — the cap keeps the inverted 1/(1-rate)
            // scale finite. Multiplier 1 (the uniform/legacy semantic)
            // composes NO extra ops, so uniform towers keep today's exact
            // graph.
            let rate: MPSGraphTensor
            if spec.dropoutMultiplier == 1 {
                rate = baseRate
            } else {
                let mult = graph.constant(
                    Double(spec.dropoutMultiplier), shape: [1], dataType: .float32
                )
                let scaled = graph.multiplication(
                    baseRate, mult, name: "\(prefix)_dropout_rate_scaled"
                )
                let cap = graph.constant(0.95, shape: [1], dataType: .float32)
                rate = graph.minimum(scaled, cap, name: "\(prefix)_dropout_rate_capped")
            }
            guard let desc = MPSGraphRandomOpDescriptor(
                distribution: .uniform, dataType: .float32
            ) else {
                throw ChessNetworkError.randomDescriptorCreationFailed
            }
            let drawn = graph.randomTensor(
                withShapeTensor: maskShape, descriptor: desc,
                stateTensor: stateIn, name: "\(prefix)_dropout_rng"
            )
            dropoutStateLocal = drawn[1]
            let keep = graph.greaterThanOrEqualTo(drawn[0], rate, name: "\(prefix)_dropout_keep")
            let maskF = graph.cast(keep, to: .float32, name: "\(prefix)_dropout_mask")
            let one = graph.constant(1.0, shape: [1], dataType: .float32)
            let scale = graph.division(
                one, graph.subtraction(one, rate, name: "\(prefix)_dropout_keep_frac"),
                name: "\(prefix)_dropout_scale"
            )
            let scaledMask = graph.multiplication(maskF, scale, name: "\(prefix)_dropout_scaled_mask")
            let dtype = Self.mpsDataType(for: arch)
            let scaledMaskCast = (dtype == .float32)
                ? scaledMask
                : graph.cast(scaledMask, to: dtype, name: "\(prefix)_dropout_scaled_mask_cast")
            return graph.multiplication(h, scaledMaskCast, name: "\(prefix)_dropout")
        }

        // Bias-free, He-init conv weight (caller appends to `trainables`).
        // Fan-in derives from the tensor's OWN shape (inCh × k²), never from
        // tower-level fields — per-block widths make any global assumption
        // wrong, not just stale. `name` is the graph variable's stem,
        // `planName` the tensor's plan (safetensors) name.
        func makeConvWeight(_ name: String, planName: String, _ k: Int, _ inCh: Int, _ outCh: Int) throws -> MPSGraphTensor {
            graph.variable(
                with: try initializer.weightData(
                    planName, nativeShape: [outCh, inCh, k, k],
                    distribution: .heNormal, dataType: variableDataType),
                shape: [NSNumber(value: outCh), NSNumber(value: inCh), NSNumber(value: k), NSNumber(value: k)],
                dataType: variableDataType,
                name: "\(name)_weights"
            )
        }

        // Residual function F(input). `z` is the SE input: the raw conv2 output
        // in pre-activation, or the BN2 output in post-activation. The append
        // order here is the single source of truth that `weightTensorPlan`
        // mirrors (pre: bn1,conv1,bn2,conv2 ; post: conv1,bn1,conv2,bn2; then
        // SE, rezero, and — width transitions only — the skip projection LAST).
        //
        // `skipProjInput` is what a width-transition skip projection consumes:
        // for pre-activation blocks it is the SHARED pre-activation (the
        // BN1→act output the branch also reads — He et al. v2 convention, so
        // both paths see the same normalized input at a transition); for
        // post-activation blocks it is the raw block input (v1 convention).
        let z: MPSGraphTensor
        var skipProjInput = input
        switch spec.activationStyle {
        case .pre:
            var h = batchNorm(graph: graph, input: input, channels: inC, gammaInitValue: Self.standardBatchNormGamma, name: "\(prefix)_bn1", taps: taps, bnMode: bnMode, dataType: Self.mpsDataType(for: arch),
                variableDataType: variableDataType,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            h = activation(graph, h, spec.activationFunction, name: "\(prefix)_act1")
            skipProjInput = h
            let conv1W = try makeConvWeight("\(prefix)_conv1", planName: "\(planPrefix).conv1.weight", spec.conv1KernelSize, inC, outC)
            trainables.append(conv1W); shouldDecay.append(true)
            h = graph.convolution2D(h, weights: conv1W, descriptor: conv1Desc, name: "\(prefix)_conv1")
            h = batchNorm(graph: graph, input: h, channels: outC, gammaInitValue: Self.standardBatchNormGamma, name: "\(prefix)_bn2", taps: taps, bnMode: bnMode, dataType: Self.mpsDataType(for: arch),
                variableDataType: variableDataType,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            h = activation(graph, h, spec.activationFunction, name: "\(prefix)_act2")
            h = try applyChannelDropout(h)
            let conv2W = try makeConvWeight("\(prefix)_conv2", planName: "\(planPrefix).conv2.weight", spec.conv2KernelSize, outC, outC)
            trainables.append(conv2W); shouldDecay.append(true)
            z = graph.convolution2D(h, weights: conv2W, descriptor: conv2Desc, name: "\(prefix)_conv2")
        case .post:
            let conv1W = try makeConvWeight("\(prefix)_conv1", planName: "\(planPrefix).conv1.weight", spec.conv1KernelSize, inC, outC)
            trainables.append(conv1W); shouldDecay.append(true)
            var h = graph.convolution2D(input, weights: conv1W, descriptor: conv1Desc, name: "\(prefix)_conv1")
            h = batchNorm(graph: graph, input: h, channels: outC, gammaInitValue: Self.standardBatchNormGamma, name: "\(prefix)_bn1", taps: taps, bnMode: bnMode, dataType: Self.mpsDataType(for: arch),
                variableDataType: variableDataType,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            h = activation(graph, h, spec.activationFunction, name: "\(prefix)_act1")
            h = try applyChannelDropout(h)
            let conv2W = try makeConvWeight("\(prefix)_conv2", planName: "\(planPrefix).conv2.weight", spec.conv2KernelSize, outC, outC)
            trainables.append(conv2W); shouldDecay.append(true)
            h = graph.convolution2D(h, weights: conv2W, descriptor: conv2Desc, name: "\(prefix)_conv2")
            z = batchNorm(graph: graph, input: h, channels: outC, gammaInitValue: Self.lastBranchBatchNormGamma(spec), name: "\(prefix)_bn2", taps: taps, bnMode: bnMode, dataType: Self.mpsDataType(for: arch),
                variableDataType: variableDataType,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
        }

        // Hand the advanced RNG stream position back to the caller so the
        // next block draws fresh numbers (and the post-tower advance op
        // captures the final position).
        dropoutRngState = dropoutStateLocal

        // SE channel attention (style-dependent; identity when .none).
        let seOut = try applySE(graph: graph, arch: arch, spec: spec, z: z, prefix: prefix, planPrefix: planPrefix,
            variableDataType: variableDataType,
            initializer: initializer,
            trainables: &trainables, shouldDecay: &shouldDecay)

        // ReZero branch scalar (optional), init `rezeroAlphaInit`, no weight decay.
        // Stored in `variableDataType`, the compute dtype of the branch it
        // scales.
        //
        // α₀ = 0 is legal and is the published ReZero init: at α = 0 the
        // bounded scale C·tanh(0) is exactly 0, so the branch output is zeroed
        // and the identity path alone carries the signal (the block starts as
        // an exact identity). Nothing is dead, though: ∂L/∂α = ⟨∂L/∂branch_out,
        // F(x)⟩ · d(C·tanh(α/C))/dα, and that derivative is 1 − tanh²(α/C) = 1
        // at α = 0 — the soft bound passes α's gradient through unscaled
        // there — so α moves on step 1. The branch weights receive gradient
        // scaled by the effective α, so they are frozen only at step 0 and
        // start learning as soon as α leaves zero.
        var branch = seOut
        if spec.useRezero {
            let alpha = graph.variable(
                with: makeWeightData([spec.rezeroAlphaInit], dataType: variableDataType),
                shape: [1], dataType: variableDataType, name: "\(prefix)_res_scale")
            trainables.append(alpha); shouldDecay.append(false)
            // Soft-bound the ReZero scalar through `C·tanh(α/C)` in the forward.
            // α is a free, undecayed scalar and nothing opposes its growth — left
            // unbounded it ratchets (a runaway to ~30 once drowned the identity
            // skip and degenerated the whole tower). A *hard* clamp at C bounded
            // the magnitude but had zero gradient past C, so the stored α drifted
            // freely above the wall (we saw 2.0→2.27) while pinned at the ceiling,
            // and C=2.0 sat the tower in an already-degraded regime for ~17k steps.
            // tanh fixes both: it's a smooth saturating bound to ±C with a gradient
            // that is alive everywhere (never a dead zone), near-identity for small
            // α (C·tanh(α/C) ≈ α when α≪C, so it starts at the depth-aware 1/√N
            // init and behaves normally in the healthy range), and asymptotes to C
            // so α can never enter the runaway regime.
            //
            // C is explicit per group (`BlockGroup.rezeroAlphaCap`, read through
            // `rezeroTanhCeiling`). Files older than architecture format v6 carry
            // no cap and resolve it to α₀ (the old `rezeroTanhCeilingMultiple·α₀`
            // rule, mult 1.0), so every legacy model builds this exact graph. That
            // rule's reasoning still holds for a positive init: the raw α ratchets
            // up (its gradient is one-way), so effective α saturates AT the cap
            // across all blocks — pinning the cap at α₀ = 1/√N makes that
            // saturated state variance-preserving (Σα² ≈ N·(1/√N)² = 1). C=1.0
            // with α₀ = 1/√5 failed because saturating at ~0.95 per block gives
            // Σα² ≈ 4.5 and the stream mean still exploded (bn1Mean 43→1384 over
            // 1k steps, broke ~step 5800). Decoupling C from α₀ is what makes the
            // zero init possible at all (C = α₀ = 0 would divide by zero), and it
            // lets the cap be chosen on its own. Bounding in the forward means a
            // saved α is still bounded on reload, and the branch keeps full
            // gradient. See documentation/rezero-alpha-clamp.md.
            let cConst = graph.constant(spec.rezeroTanhCeiling, dataType: alpha.dataType)
            let alphaBounded = graph.multiplication(
                cConst,
                graph.tanh(with: graph.division(alpha, cConst, name: nil), name: nil),
                name: "\(prefix)_res_scale_tanh"
            )
            branch = graph.multiplication(seOut, alphaBounded, name: "\(prefix)_res_scaled")
        }

        // Skip path: clean identity everywhere widths match; at a width
        // transition (inC != outC) the add cannot typecheck, so the skip gets
        // the minimum repair — a bias-free 1×1 projection (a pure per-square
        // linear remap of the feature vector, zero spatial mixing), He-init
        // by its own fan-in or — under the group's `skip_projection_init:
        // identity_like` — a partial identity, weight-decayed. Its weight appends LAST within
        // the block (after the rezero α) per the tensor-order contract that
        // `weightTensorPlan` mirrors.
        var skipSource = input
        var skipProjWeight: MPSGraphTensor? = nil
        if inC != outC {
            let projW = graph.variable(
                with: try initializer.skipProjectionData(
                    "\(planPrefix).skip_proj.weight", nativeShape: [outC, inC, 1, 1],
                    initialization: spec.skipProjectionInit, dataType: variableDataType),
                shape: [NSNumber(value: outC), NSNumber(value: inC), 1, 1],
                dataType: variableDataType,
                name: "\(prefix)_skip_proj_weights")
            skipProjWeight = projW
            let projDesc = try makeConvDescriptor(kernelSize: 1)
            skipSource = graph.convolution2D(
                skipProjInput, weights: projW, descriptor: projDesc,
                name: "\(prefix)_skip_proj"
            )
        }

        // Merge with the skip.
        let merged: MPSGraphTensor
        switch spec.skipMerge {
        case .cleanAdd:
            // out = skip + [alpha .] F(input) — clean identity highway (no activation on the sum).
            merged = graph.addition(skipSource, branch, name: "\(prefix)_skip")
        case .activationGated:
            // out = activation(skip + F(input)) — the v3 gated merge.
            let sum = graph.addition(skipSource, branch, name: "\(prefix)_skip_sum")
            merged = activation(graph, sum, spec.activationFunction, name: "\(prefix)_skip")
        }
        if let skipProjWeight {
            trainables.append(skipProjWeight); shouldDecay.append(true)
        }

        // Optional output normalization — applied to the merged block output
        // AFTER the skip projection, so its γ/β are the LAST tensors appended
        // within this block (matching weightTensorPlan's `res_ln` placement).
        // Re-centers the residual stream every block; composes with either
        // skipMerge mode. (v5 = v4 clean-add highway + this LayerNorm.)
        guard spec.resolvedOutputNorm == .layerNorm else { return merged }
        return layerNorm(
            graph: graph, input: merged, channels: outC, name: "\(prefix)_res_ln", taps: taps,
            variableDataType: variableDataType,
            trainables: &trainables, shouldDecay: &shouldDecay
        )
    }

    /// Squeeze-and-Excitation channel attention applied to `z`. Appends SE weights
    /// to `trainables` (FC1 w/b then FC2 w/b). Identity (returns `z`) when
    /// `arch.blockSeStyle == .none`. `attenuateOnly`: FC2->C, `sigmoid(z)*x`.
    /// `scaleAndBias`: FC2->2C, `sigmoid(gamma)*x + beta`.
    private static func applySE(
        graph: MPSGraph, arch: NetworkArchitecture, spec: BlockGroup, z: MPSGraphTensor, prefix: String,
        planPrefix: String,
        variableDataType: MPSDataType,
        initializer: TensorInitializer,
        trainables: inout [MPSGraphTensor], shouldDecay: inout [Bool]
    ) throws -> MPSGraphTensor {
        guard spec.seStyle != .none else { return z }
        let channels = spec.channels
        let seReduced = channels / spec.seReductionRatio
        let seExpand = spec.seStyle == .scaleAndBias ? 2 * channels : channels
        // Mirrors `weightTensorPlan()`'s per-style module name.
        let sePlanPrefix = spec.seStyle == .scaleAndBias
            ? "\(planPrefix).se_scalebias"
            : "\(planPrefix).se_attenuate"

        // Squeeze: global average pool over [H, W] -> [B, C, 1, 1] -> [B, C].
        var s = graph.mean(of: z, axes: [2, 3], name: "\(prefix)_se_squeeze")
        s = graph.reshape(s, shape: [-1, NSNumber(value: channels)], name: "\(prefix)_se_squeeze_flatten")

        // Excite FC1: C -> C/r (He), + the group's SE activation
        // (`seActivation`, which may differ from the main path's — a leaky
        // FC1 keeps a gradient through units ReLU would leave dead).
        let fc1 = graph.variable(
            with: try initializer.weightData(
                "\(sePlanPrefix).fc1.weight", nativeShape: [channels, seReduced],
                distribution: .heNormal, dataType: variableDataType),
            shape: [NSNumber(value: channels), NSNumber(value: seReduced)],
            dataType: variableDataType, name: "\(prefix)_se_fc1_weights")
        let fc1b = graph.variable(
            with: zerosData(count: seReduced, dataType: variableDataType),
            shape: [1, NSNumber(value: seReduced)],
            dataType: variableDataType, name: "\(prefix)_se_fc1_bias")
        trainables.append(fc1);  shouldDecay.append(true)
        trainables.append(fc1b); shouldDecay.append(false)
        s = graph.matrixMultiplication(primary: s, secondary: fc1, name: "\(prefix)_se_fc1")
        s = graph.addition(s, fc1b, name: "\(prefix)_se_fc1_bias_add")
        s = activation(graph, s, spec.seActivation, name: "\(prefix)_se_act")

        // Excite FC2: C/r -> seExpand (Glorot, feeds the sigmoid gate). A
        // zero-β group draws the same Glorot matrix and then zeroes its β
        // columns, so the γ half is initialized exactly as a Glorot-β group's
        // is and only the additive half starts at "do nothing"
        // (`WeightInitScheme.seFC2NativeValues`). The FC2 bias starts at the
        // group's `se_gamma_bias_init` on the γ half (standard 0, so every
        // gate starts at sigmoid(0)) and zero on the β half
        // (`WeightInitScheme.seFC2BiasValues`).
        let fc2 = graph.variable(
            with: try initializer.seFC2Data(
                "\(sePlanPrefix).fc2.weight", nativeShape: [seReduced, seExpand],
                group: spec, dataType: variableDataType),
            shape: [NSNumber(value: seReduced), NSNumber(value: seExpand)],
            dataType: variableDataType, name: "\(prefix)_se_fc2_weights")
        let fc2b = graph.variable(
            with: makeWeightData(try WeightInitScheme.seFC2BiasValues(group: spec), dataType: variableDataType),
            shape: [1, NSNumber(value: seExpand)],
            dataType: variableDataType, name: "\(prefix)_se_fc2_bias")
        trainables.append(fc2);  shouldDecay.append(true)
        trainables.append(fc2b); shouldDecay.append(false)
        s = graph.matrixMultiplication(primary: s, secondary: fc2, name: "\(prefix)_se_fc2")
        s = graph.addition(s, fc2b, name: "\(prefix)_se_fc2_bias_add")

        switch spec.seStyle {
        case .none:
            return z
        case .attenuateOnly:
            var gate = graph.sigmoid(with: s, name: "\(prefix)_se_gate")
            gate = graph.reshape(gate, shape: [-1, NSNumber(value: channels), 1, 1], name: "\(prefix)_se_gate_reshape")
            return graph.multiplication(z, gate, name: "\(prefix)_se_scaled")
        case .scaleAndBias:
            let gammas = graph.sliceTensor(s, dimension: 1, start: 0, length: channels, name: "\(prefix)_se_gammas")
            let betas = graph.sliceTensor(s, dimension: 1, start: channels, length: channels, name: "\(prefix)_se_betas")
            var scale = graph.sigmoid(with: gammas, name: "\(prefix)_se_gate")
            scale = graph.reshape(scale, shape: [-1, NSNumber(value: channels), 1, 1], name: "\(prefix)_se_scale_reshape")
            let bias = graph.reshape(betas, shape: [-1, NSNumber(value: channels), 1, 1], name: "\(prefix)_se_bias_reshape")
            var seOut = graph.multiplication(z, scale, name: "\(prefix)_se_scaled")
            seOut = graph.addition(seOut, bias, name: "\(prefix)_se_biased")
            return seOut
        }
    }

    /// Policy head: 1×1 conv (128 → 128) → BN → ReLU → 1×1 conv
    /// (128 → policyChannels=76) → reshape to flat `[batch,
    /// policySize=4864]` logits.
    ///
    /// Fully convolutional. The intermediate `conv → BN → ReLU` mirrors
    /// the value head (and lc0's convolutional policy head): the BN
    /// renormalizes the residual tower's output before the logit
    /// projection, so the deep (16-block) tower's accumulated activation
    /// scale can't inflate the raw logits and collapse the init softmax.
    /// The final 1×1 conv emits logits directly — no BN/activation after
    /// it, since logits need free scale for the downstream softmax. Both
    /// convs' weights are shared across all 64 spatial positions
    /// (translation equivariance), so each output cell at
    /// `(channel, row, col)` is the logit for "move of type `channel`
    /// from square `(row, col)`" in the current player's encoder frame.
    /// See `PolicyEncoding` for the channel layout (76 = 56 queen-style
    /// + 8 knight + 9 underpromo + 3 queen-promo).
    ///
    /// `finalWeights` is the *final* logit-projection conv (128→76), not
    /// the intermediate one — that is what `policyHeadFinalWeights` feeds.
    private static func policyHead(
        graph: MPSGraph,
        arch: NetworkArchitecture,
        input: MPSGraphTensor,
        inputChannels: Int,
        descriptor: MPSGraphConvolution2DOpDescriptor,
        bnMode: BNMode,
        taps: AnalysisTapRecorder?,
        tailPrecision: PolicyTailPrecision,
        variableDataType: MPSDataType,
        initializer: TensorInitializer,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation],
        batchMeans: inout [MPSGraphTensor],
        batchVars: inout [MPSGraphTensor]
    ) throws -> (output: MPSGraphTensor, finalWeights: MPSGraphTensor) {
        // The first conv's input width. Equals `arch.towerOutputChannels` normally;
        // wider when a routed `concatDirect` feature skip feeds this head. The caller
        // passes `arch.policyHeadInputChannels` so paramCount/weightTensorPlan/builder
        // share one formula.
        let channels = inputChannels
        let pc = Self.policyChannels
        let pK = arch.policyPreConvChannels

        // All styles emit 4864 raw logits in the current PolicyEncoding (76x64);
        // masking + softmax happen CPU-side. `finalWeights` is the logit-projecting
        // weight (for the trainer's ||W|| diagnostic). NCHW row-major flatten matches
        // PolicyEncoding.policyIndex = channel*64 + row*8 + col.
        switch arch.policyHeadStyle {
        case .simpleConv:
            // Single 1x1 conv channels -> 76 (+bias) -> reshape. No pre-block, so
            // the fp32 tail is the final conv alone: its input, weights and bias
            // are widened and everything from the conv on runs in fp32.
            let convW = graph.variable(
                with: try initializer.headFinalData(
                    "policy.conv.weight", nativeShape: [pc, channels, 1, 1],
                    initialization: arch.policyHeadFinalInit, dataType: variableDataType),
                shape: [NSNumber(value: pc), NSNumber(value: channels), 1, 1],
                dataType: variableDataType, name: "policy_conv_weights")
            let convBias = graph.variable(
                with: zerosData(count: pc, dataType: variableDataType),
                shape: [1, NSNumber(value: pc), 1, 1],
                dataType: variableDataType, name: "policy_conv_bias")
            trainables.append(convW);    shouldDecay.append(true)
            trainables.append(convBias); shouldDecay.append(false)
            var x: MPSGraphTensor
            switch tailPrecision {
            case .float32FromPreBatchNorm:
                let tailInput = widenToHeadTail(input, graph: graph, name: "policy_tail_input_f32")
                x = graph.convolution2D(
                    tailInput, weights: widenToHeadTail(convW, graph: graph, name: nil),
                    descriptor: descriptor, name: "policy_conv")
            case .mixedFinalProjection:
                x = graph.convolution2D(input, weights: convW, descriptor: descriptor, name: "policy_conv")
                x = widenToHeadTail(x, graph: graph, name: "policy_conv_output_f32")
            }
            x = graph.addition(x, widenToHeadTail(convBias, graph: graph, name: nil), name: "policy_conv_bias_add")
            let flat = graph.reshape(x, shape: [-1, NSNumber(value: Self.policySize)], name: "policy_flatten")
            return (output: flat, finalWeights: convW)

        case .intermediateConv:
            // 1x1 conv channels -> K -> BN -> act -> 1x1 conv K -> 76 (+bias) -> reshape.
            // The pre-conv runs in the compute dtype; the fp32 tail starts at the
            // pre-BN normalize. Starting it only at the final conv leaves the
            // compute-dtype rounding of the K policy features multiplied by the
            // final conv's large shared row, a per-square error softmax does not
            // cancel; normalizing in fp32 removes most of it.
            let preConvW = graph.variable(
                with: try initializer.weightData(
                    "policy.pre_conv.weight", nativeShape: [pK, channels, 1, 1],
                    distribution: .heNormal, dataType: variableDataType),
                shape: [NSNumber(value: pK), NSNumber(value: channels), 1, 1],
                dataType: variableDataType, name: "policy_pre_conv_weights")
            trainables.append(preConvW); shouldDecay.append(true)
            var x = graph.convolution2D(input, weights: preConvW, descriptor: descriptor, name: "policy_pre_conv")
            switch tailPrecision {
            case .float32FromPreBatchNorm:
                x = widenToHeadTail(x, graph: graph, name: "policy_tail_input_f32")
                x = batchNorm(graph: graph, input: x, channels: pK, gammaInitValue: Self.standardBatchNormGamma, name: "policy_pre_bn", taps: taps, bnMode: bnMode, dataType: headTailDataType,
                    variableDataType: variableDataType,
                    trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                    runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            case .mixedFinalProjection:
                x = batchNorm(graph: graph, input: x, channels: pK, gammaInitValue: Self.standardBatchNormGamma, name: "policy_pre_bn", taps: taps, bnMode: bnMode,
                    dataType: Self.mpsDataType(for: arch),
                    variableDataType: variableDataType,
                    trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                    runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            }
            x = activation(graph, x, arch, name: "policy_pre_act")
            taps?.record("policy_pre_act", x)
            let convW = graph.variable(
                with: try initializer.headFinalData(
                    "policy.conv.weight", nativeShape: [pc, pK, 1, 1],
                    initialization: arch.policyHeadFinalInit, dataType: variableDataType),
                shape: [NSNumber(value: pc), NSNumber(value: pK), 1, 1],
                dataType: variableDataType, name: "policy_conv_weights")
            let convBias = graph.variable(
                with: zerosData(count: pc, dataType: variableDataType),
                shape: [1, NSNumber(value: pc), 1, 1],
                dataType: variableDataType, name: "policy_conv_bias")
            trainables.append(convW);    shouldDecay.append(true)
            trainables.append(convBias); shouldDecay.append(false)
            switch tailPrecision {
            case .float32FromPreBatchNorm:
                x = graph.convolution2D(
                    x, weights: widenToHeadTail(convW, graph: graph, name: nil),
                    descriptor: descriptor, name: "policy_conv")
            case .mixedFinalProjection:
                x = graph.convolution2D(x, weights: convW, descriptor: descriptor, name: "policy_conv")
                x = widenToHeadTail(x, graph: graph, name: "policy_conv_output_f32")
            }
            x = graph.addition(x, widenToHeadTail(convBias, graph: graph, name: nil), name: "policy_conv_bias_add")
            let flat = graph.reshape(x, shape: [-1, NSNumber(value: Self.policySize)], name: "policy_flatten")
            return (output: flat, finalWeights: convW)

        case .fcBottleneck:
            // 1x1 conv channels -> K -> BN -> act -> flatten(K*64) -> FC(K*64 -> 4864) (+bias).
            // Same fp32 tail boundary as `intermediateConv`: the pre-BN normalize.
            let preConvW = graph.variable(
                with: try initializer.weightData(
                    "policy.pre_conv.weight", nativeShape: [pK, channels, 1, 1],
                    distribution: .heNormal, dataType: variableDataType),
                shape: [NSNumber(value: pK), NSNumber(value: channels), 1, 1],
                dataType: variableDataType, name: "policy_pre_conv_weights")
            trainables.append(preConvW); shouldDecay.append(true)
            var x = graph.convolution2D(input, weights: preConvW, descriptor: descriptor, name: "policy_pre_conv")
            switch tailPrecision {
            case .float32FromPreBatchNorm:
                x = widenToHeadTail(x, graph: graph, name: "policy_tail_input_f32")
                x = batchNorm(graph: graph, input: x, channels: pK, gammaInitValue: Self.standardBatchNormGamma, name: "policy_pre_bn", taps: taps, bnMode: bnMode, dataType: headTailDataType,
                    variableDataType: variableDataType,
                    trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                    runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            case .mixedFinalProjection:
                x = batchNorm(graph: graph, input: x, channels: pK, gammaInitValue: Self.standardBatchNormGamma, name: "policy_pre_bn", taps: taps, bnMode: bnMode,
                    dataType: Self.mpsDataType(for: arch),
                    variableDataType: variableDataType,
                    trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                    runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            }
            x = activation(graph, x, arch, name: "policy_pre_act")
            taps?.record("policy_pre_act", x)
            let flatSize = pK * Self.boardSize * Self.boardSize
            x = graph.reshape(x, shape: [-1, NSNumber(value: flatSize)], name: "policy_flatten_pre")
            let fcW = graph.variable(
                with: try initializer.headFinalData(
                    "policy.fc.weight", nativeShape: [flatSize, Self.policySize],
                    initialization: arch.policyHeadFinalInit, dataType: variableDataType),
                shape: [NSNumber(value: flatSize), NSNumber(value: Self.policySize)],
                dataType: variableDataType, name: "policy_fc_weights")
            let fcBias = graph.variable(
                with: zerosData(count: Self.policySize, dataType: variableDataType),
                shape: [1, NSNumber(value: Self.policySize)],
                dataType: variableDataType, name: "policy_fc_bias")
            trainables.append(fcW);    shouldDecay.append(true)
            trainables.append(fcBias); shouldDecay.append(false)
            switch tailPrecision {
            case .float32FromPreBatchNorm:
                x = graph.matrixMultiplication(
                    primary: x, secondary: widenToHeadTail(fcW, graph: graph, name: nil), name: "policy_fc")
            case .mixedFinalProjection:
                x = graph.matrixMultiplication(primary: x, secondary: fcW, name: "policy_fc")
                x = widenToHeadTail(x, graph: graph, name: "policy_fc_output_f32")
            }
            let logits = graph.addition(x, widenToHeadTail(fcBias, graph: graph, name: nil), name: "policy_fc_bias_add")
            return (output: logits, finalWeights: fcW)
        }
    }

    /// Where the policy head leaves the compute dtype for fp32. Not an
    /// architecture field and not saved with a model: the weights are
    /// identical under both, only the graph's arithmetic differs.
    ///
    /// Why there are two: the head-numerics fix first started the fp32 tail
    /// at the policy pre-BN. That cost training throughput, because the final
    /// projection left the bf16-input conv kernels for generic fp32 ones and
    /// BN / ReLU ran in fp32 over a twice-as-large tensor.
    /// `mixedFinalProjection` keeps both in the compute dtype and widens only
    /// the projection's output, which recovers most of that cost. What it
    /// gives back: the policy features are rounded to the compute dtype before
    /// the projection. On models trained with the fix that adds little to the
    /// bf16-vs-fp32 policy divergence; on older weights whose projection
    /// carries a large shared row it adds noticeably more (the numerics audit
    /// measures it). `float32FromPreBatchNorm` stays selectable for that
    /// comparison.
    enum PolicyTailPrecision: String, CaseIterable, Sendable {
        /// The precision every network is built with unless a caller asks
        /// for another — the single source of the default.
        static let `default`: PolicyTailPrecision = .mixedFinalProjection

        /// fp32 from the pre-BN normalize on (from the final projection's
        /// input for `simple_conv`, which has no pre-block).
        case float32FromPreBatchNorm = "fp32_from_pre_bn"
        /// Pre-block and final projection in the compute dtype; the
        /// projection's output, its bias add and everything after are fp32.
        case mixedFinalProjection = "mixed_final_projection"

        /// The command-line flag that chooses the process's value.
        static let flag = "--policy-tail-precision"

        /// Where the process's value came from.
        enum Source: String, Sendable {
            case defaultValue = "default"
            case flag
        }

        /// The process's value and its source.
        struct Resolution: Sendable, Equatable {
            let value: PolicyTailPrecision
            let source: Source
        }

        enum ResolutionError: Error, Equatable, CustomStringConvertible {
            case missingValue
            case unknownValue(String)
            case repeated(count: Int)

            var description: String {
                switch self {
                case .missingValue:
                    return "\(PolicyTailPrecision.flag) requires a value"
                case .unknownValue(let raw):
                    let allowed = PolicyTailPrecision.allCases.map(\.rawValue).joined(separator: ", ")
                    return "\(PolicyTailPrecision.flag) expects one of \(allowed), got '\(raw)'"
                case .repeated(let count):
                    return "\(PolicyTailPrecision.flag) given \(count) times; give it at most once"
                }
            }
        }

        /// The one parser of `--policy-tail-precision`: the flag's value, or
        /// `default` when the flag is absent. Every mode — the GUI, every
        /// CLI, the numerics audit — resolves it here.
        static func resolve(arguments: [String]) throws -> Resolution {
            let positions = arguments.indices.filter { arguments[$0] == flag }
            guard let position = positions.first else {
                return Resolution(value: .default, source: .defaultValue)
            }
            guard positions.count == 1 else { throw ResolutionError.repeated(count: positions.count) }
            let valueIndex = position + 1
            guard valueIndex < arguments.count, !arguments[valueIndex].hasPrefix("--") else {
                throw ResolutionError.missingValue
            }
            let raw = arguments[valueIndex]
            guard let value = PolicyTailPrecision(rawValue: raw) else { throw ResolutionError.unknownValue(raw) }
            return Resolution(value: value, source: .flag)
        }

        /// The value every network and trainer in this process is built
        /// with unless a caller passes one explicitly — fixed for the life of
        /// the process, because the app keeps its inference and arena
        /// networks for the whole launch. `DrewsChessMachineApp.init` resolves
        /// the same arguments first and exits on a bad flag, so the trap here
        /// cannot be reached from a launch.
        static let processResolution: Resolution = {
            do {
                return try resolve(arguments: CommandLine.arguments)
            } catch {
                preconditionFailure("\(error) — the launch-time check should have refused this")
            }
        }()

        /// `processResolution.value`.
        static var process: PolicyTailPrecision { processResolution.value }

        /// One log line naming the process's value and where it came from.
        static var processLogLine: String {
            "[NUMERICS] policy_tail_precision=\(processResolution.value.rawValue) source=\(processResolution.source.rawValue) (affects bf16 / fp16 models only)"
        }
    }

    /// Compute dtype of both heads' tails (see `widenToHeadTail`). Every
    /// head output — `policyOutput`, `valueLogits`, `valueProbs`,
    /// `valueOutput` — is this dtype on every build, whatever the tower's
    /// compute dtype, so every reader of a head output reads fp32.
    static let headTailDataType: MPSDataType = .float32

    /// Widen a tensor entering a head's fp32 tail. A head's outputs sit on a
    /// shared per-position offset that softmax ignores; in a narrow dtype the
    /// rounding step at that offset's magnitude swallows the real differences
    /// between logits (tied moves, tied W/D/L classes). Computing the tail in
    /// fp32 keeps those differences.
    ///
    /// Identity when the tensor is already fp32 (a `.float32` build). Weights
    /// keep their stored dtype; widening a stored value is exact, so nothing
    /// is lost.
    private static func widenToHeadTail(
        _ tensor: MPSGraphTensor, graph: MPSGraph, name: String?
    ) -> MPSGraphTensor {
        tensor.dataType == headTailDataType ? tensor : graph.cast(tensor, to: headTailDataType, name: name)
    }

    /// Value head: 1x1 conv (128 -> 1) -> BN -> ReLU -> flatten -> FC(64 -> 64) -> ReLU -> FC(64 -> 3) -> W/D/L logits.
    ///
    /// Returns the raw 3-wide logits (`logits`, `[batch, 3]`, slot order
    /// `[win, draw, loss]`), their softmax (`probs`, the predicted
    /// `(p_win, p_draw, p_loss)`), and the derived scalar
    /// `scalar = Σ_c probs_c · [+1, 0, −1]_c = p_win − p_loss` — which
    /// is naturally in `[−1, +1]` (a difference of two probabilities),
    /// so there is no tanh. The scalar is what move-selection's value
    /// readback, the dashboard, and the policy-gradient baseline use;
    /// the logits/probs feed the value cross-entropy loss and the
    /// W/D/L diagnostics in `ChessTrainer`.
    private static func valueHead(
        graph: MPSGraph,
        arch: NetworkArchitecture,
        input: MPSGraphTensor,
        inputChannels: Int,
        descriptor: MPSGraphConvolution2DOpDescriptor,
        bnMode: BNMode,
        taps: AnalysisTapRecorder?,
        variableDataType: MPSDataType,
        initializer: TensorInitializer,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation],
        batchMeans: inout [MPSGraphTensor],
        batchVars: inout [MPSGraphTensor]
    ) throws -> (scalar: MPSGraphTensor, logits: MPSGraphTensor, probs: MPSGraphTensor) {
        // 1x1 conv: compress the trunk to `valueHeadConvChannels` scoring maps. The
        // input width equals `arch.towerOutputChannels` normally; wider when a routed
        // `concatDirect` feature skip feeds this head (caller passes
        // `arch.valueHeadInputChannels`).
        let convChannels = arch.valueHeadConvChannels
        let towerOut = inputChannels
        let convW = graph.variable(
            with: try initializer.weightData(
                "value.conv.weight", nativeShape: [convChannels, towerOut, 1, 1],
                distribution: .heNormal, dataType: variableDataType),
            shape: [NSNumber(value: convChannels), NSNumber(value: towerOut), 1, 1],
            dataType: variableDataType,
            name: "value_conv_weights"
        )
        trainables.append(convW)
        shouldDecay.append(true)
        var x = graph.convolution2D(
            input, weights: convW, descriptor: descriptor, name: "value_conv"
        )
        x = batchNorm(
            graph: graph, input: x, channels: convChannels, gammaInitValue: Self.standardBatchNormGamma, name: "value_bn", taps: taps, bnMode: bnMode, dataType: Self.mpsDataType(for: arch),
            variableDataType: variableDataType,
            trainables: &trainables,
            shouldDecay: &shouldDecay,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssignOps,
            batchMeans: &batchMeans,
            batchVars: &batchVars
        )
        x = activation(graph, x, arch, name: "value_act")

        // Flatten: [batch, convChannels, 8, 8] -> [batch, convChannels*64]
        let flattenSize = Self.boardSize * Self.boardSize * convChannels
        x = graph.reshape(x, shape: [-1, NSNumber(value: flattenSize)], name: "value_flatten")

        // FC1: flattenSize -> valueHeadHiddenUnits
        let hidden = arch.valueHeadHiddenUnits
        let fc1W = graph.variable(
            with: try initializer.weightData(
                "value.fc1.weight", nativeShape: [flattenSize, hidden],
                distribution: .heNormal, dataType: variableDataType),
            shape: [NSNumber(value: flattenSize), NSNumber(value: hidden)],
            dataType: variableDataType,
            name: "value_fc1_weights"
        )
        let fc1Bias = graph.variable(
            with: zerosData(count: hidden, dataType: variableDataType),
            shape: [1, NSNumber(value: hidden)],
            dataType: variableDataType,
            name: "value_fc1_bias"
        )
        trainables.append(fc1W)
        shouldDecay.append(true)
        trainables.append(fc1Bias)
        shouldDecay.append(false)
        x = graph.matrixMultiplication(primary: x, secondary: fc1W, name: "value_fc1")
        x = graph.addition(x, fc1Bias, name: "value_fc1_bias_add")
        x = activation(graph, x, arch, name: "value_fc1_act")
        taps?.record("value_fc1_act", x)

        // FC2: hidden -> valueHeadClasses (3 = W/D/L logits, or 1 = scalar pre-tanh).
        let classes = arch.valueHeadClasses
        let fc2Name = arch.valueHeadStyle == .wdlSoftmax ? "value_wdl_fc2" : "value_scalar_fc2"
        let fc2PlanName = arch.valueHeadStyle == .wdlSoftmax ? "value.wdl_fc2.weight" : "value.scalar_fc2.weight"
        let fc2W = graph.variable(
            with: try initializer.headFinalData(
                fc2PlanName, nativeShape: [hidden, classes],
                initialization: arch.valueHeadFinalInit, dataType: variableDataType),
            shape: [NSNumber(value: hidden), NSNumber(value: classes)],
            dataType: variableDataType,
            name: "\(fc2Name)_weights"
        )
        // Bias init: WDL -> `wdlBiasPrior(drawProbability:)` of the
        // architecture's `value_head_draw_prior` (standard 0.75 = [0, ln 6, 0],
        // initial softmax (0.125, 0.75, 0.125); the derived scalar starts at 0
        // for any prior). scalar-tanh -> [0] (tanh(0)=0). Slot order
        // [win, draw, loss] for WDL.
        let fc2BiasValues: [Float] = arch.valueHeadStyle == .wdlSoftmax
            ? NetworkArchitecture.wdlBiasPrior(drawProbability: arch.valueHeadDrawPrior)
            : [0.0]
        let fc2Bias = graph.variable(
            with: makeWeightData(fc2BiasValues, dataType: variableDataType),
            shape: [1, NSNumber(value: classes)],
            dataType: variableDataType,
            name: "\(fc2Name)_bias"
        )
        trainables.append(fc2W)
        shouldDecay.append(true)
        trainables.append(fc2Bias)
        shouldDecay.append(false)
        // fp32 tail: fc2 + bias -> logits -> softmax -> scalar. The logits carry
        // a shared offset softmax ignores, and in the compute dtype its rounding
        // step ties the W/D/L classes; see `widenToHeadTail`.
        let tailInput = widenToHeadTail(x, graph: graph, name: "value_tail_input_f32")
        x = graph.matrixMultiplication(
            primary: tailInput, secondary: widenToHeadTail(fc2W, graph: graph, name: nil), name: "value_fc2")
        let logits = graph.addition(x, widenToHeadTail(fc2Bias, graph: graph, name: nil), name: "value_fc2_bias_add")

        switch arch.valueHeadStyle {
        case .wdlSoftmax:
            // Derived scalar v = p_win - p_loss (no tanh): softmax . [+1, 0, -1].
            let probs = graph.softMax(with: logits, axis: 1, name: "value_probs")
            let scalarWeights = graph.constant(
                makeWeightData([1.0, 0.0, -1.0], dataType: headTailDataType), shape: [1, 3], dataType: headTailDataType)
            let scalarWeighted = graph.multiplication(probs, scalarWeights, name: "value_scalar_weighted")
            // reductionSum(axis:1) keeps the reduced dim -> [batch, 1].
            let scalar = graph.reductionSum(with: scalarWeighted, axis: 1, name: "value_scalar")
            return (scalar: scalar, logits: logits, probs: probs)
        case .scalarTanh:
            // scalar = tanh(raw logit) in [-1, 1]; trained with MSE vs z (Phase D).
            // `probs` mirrors `scalar` (W/D/L diagnostics only apply to wdl nets)
            // but MUST be a DISTINCT graph tensor: `inferenceTargets` lists both
            // `valueOutput` (= scalar) and `valueProbs` (= probs), and the
            // readback builds `Dictionary(uniqueKeysWithValues:)` keyed by
            // tensor — aliasing the two to one tensor traps on a duplicate key.
            // A no-op reshape (scalar is already [batch, 1]) yields a separate
            // tensor carrying the same values.
            let scalar = graph.tanh(with: logits, name: "value_scalar")
            // `probs` must be a DISTINCT graph tensor from `scalar`: both go into
            // `inferenceTargets` and the readback keys its results dict by tensor,
            // so aliasing them traps on a duplicate key. A multiply-by-1 op yields
            // a separate tensor object carrying the same values (a same-shape
            // reshape could in principle be elided; an op output cannot).
            let probsOne = graph.constant(1.0, dataType: headTailDataType)
            let probs = graph.multiplication(scalar, probsOne, name: "value_scalar_probs")
            return (scalar: scalar, logits: logits, probs: probs)
        }
    }

    // MARK: - Data Helpers

    /// He-normal values (`std = √(2 / fanIn)`) for a tensor of `shape`,
    /// from a freshly drawn seed, for checking the init distributions.
    /// Network builds never call this: they draw every tensor by plan name
    /// through `TensorInitializer`. It shares their one transform
    /// (`WeightInitScheme.standardNormals`, i.e. `DCMNormalMath`), so its
    /// statistics are the builds' statistics.
    ///
    /// `fanIn` depends on the weight layout: for an OIHW conv
    /// `[outC, inC, kH, kW]` it is `inC·kH·kW`; for an FC stored `[in, out]`
    /// (the layout `matrixMultiplication(primary: x, secondary: W)` uses) it
    /// is `in`, the first dimension — the opposite of the conv case.
    static func heInitData(shape: [Int], fanIn: Int, dataType: MPSDataType) -> Data {
        precondition(fanIn > 0, "He init: fanIn must be > 0 (got \(fanIn))")
        let std = (2.0 / Float(fanIn)).squareRoot()
        return makeWeightData(freshScaledNormals(count: shape.reduce(1, *), std: std), dataType: dataType)
    }

    /// Glorot-normal values (`std = √(2 / (fan_in + fan_out))`) for an FC
    /// weight stored `[in, out]`, from a freshly drawn seed — the
    /// distribution-check counterpart of `heInitData` (see there).
    static func glorotInitDataFCInOut(shape: [Int], dataType: MPSDataType) -> Data {
        precondition(shape.count == 2, "FC [in, out] shape must be 2D (got \(shape))")
        let std = (2.0 / Float(shape[0] + shape[1])).squareRoot()
        return makeWeightData(freshScaledNormals(count: shape.reduce(1, *), std: std), dataType: dataType)
    }

    private static func freshScaledNormals(count: Int, std: Float) -> [Float] {
        var system = SystemRandomNumberGenerator()
        var values = WeightInitScheme.standardNormals(seed: system.next(), count: count)
        for index in values.indices { values[index] = std * values[index] }
        return values
    }

    static func onesData(count: Int, dataType: MPSDataType) -> Data {
        makeWeightData([Float](repeating: 1.0, count: count), dataType: dataType)
    }

    static func zerosData(count: Int, dataType: MPSDataType) -> Data {
        makeWeightData([Float](repeating: 0.0, count: count), dataType: dataType)
    }

    /// Write raw fp32 board planes from `buffer` directly into `array`'s
    /// storage. Primary inference-hot-path writer: the caller passes a
    /// pre-encoded `UnsafeBufferPointer<Float>` (e.g. a slice of a per-game
    /// scratch) and the bytes flow straight into the MPSNDArray with zero
    /// intermediate copies and **no host-side conversion**.
    ///
    /// The inference input boundary is always fp32 — `inputPlaceholder` is an
    /// fp32 placeholder and the narrowing to the compute dtype runs on the GPU
    /// (the `board_input_cast` op). Before that GPU-cast offload this method
    /// narrowed fp32→bf16 in a profiled host-side hot loop; that work now
    /// happens in the graph, so this is an unconditional passthrough.
    static func writeInferenceInput(
        _ buffer: UnsafeBufferPointer<Float>,
        into array: MPSNDArray
    ) {
        precondition(
            !buffer.isEmpty,
            "writeInferenceInput: empty buffer would leave MPSNDArray with stale bytes"
        )
        precondition(
            array.dataType == .float32,
            "writeInferenceInput: input ND array must be fp32 (got \(array.dataType)); "
            + "the GPU board_input_cast handles narrowing to the compute dtype."
        )
        guard let base = buffer.baseAddress else {
            preconditionFailure(
                "writeInferenceInput: buffer baseAddress is nil (count=\(buffer.count)); "
                + "upstream invariant violated."
            )
        }
        array.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil)
    }

    /// `[Float]`-input overload for callers outside the hot path. Wraps
    /// `withUnsafeBufferPointer` and delegates — no copy on `.float32`.
    static func writeInferenceInput(_ floats: [Float], into array: MPSNDArray) {
        floats.withUnsafeBufferPointer { buf in
            writeInferenceInput(buf, into: array)
        }
    }

    /// Copy `floats` into `array`'s storage, going through
    /// `makeWeightData` for dtype conversion. Used by cold paths
    /// (`loadWeights`, init-time dummy fill) where the transient `Data`
    /// allocation is acceptable. Don't call from hot paths — use
    /// `writeInferenceInput` or the trainer's in-place writer instead.
    static func writeFloats(_ floats: [Float], into array: MPSNDArray, dataType: MPSDataType) {
        precondition(
            !floats.isEmpty,
            "writeFloats: empty input would silently skip the MPSNDArray write"
        )
        let data = makeWeightData(floats, dataType: dataType)
        data.withUnsafeBytes { buf in
            guard let base = buf.baseAddress else {
                preconditionFailure(
                    "writeFloats: data baseAddress is nil despite non-empty input "
                    + "(floats.count=\(floats.count), data.count=\(data.count))"
                )
            }
            array.writeBytes(
                UnsafeMutableRawPointer(mutating: base),
                strideBytes: nil
            )
        }
    }

    /// Read an MPSGraphTensorData backed by an **fp32** ND array as Float32
    /// (raw bytes, no dtype conversion). For optimizer state — the fp32
    /// velocity buffers and fp32 master weights — which are fp32 regardless
    /// of `dataType`; the dtype-branching `readFloats` would mis-decode them.
    static func readFloatsFP32(from data: MPSGraphTensorData, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBytes { buf in
            if let ptr = buf.baseAddress {
                data.mpsndarray().readBytes(ptr, strideBytes: nil)
            }
        }
        return out
    }

    /// Read an **already-fp32** graph output straight into the caller's Float
    /// buffer — raw `readBytes`, no conversion. Inference-hot-path policy
    /// readback: every head output is fp32 (the heads' fp32 tails), so the
    /// host side is a plain memcpy. The caller is responsible for
    /// `pointer` having capacity `count`.
    static func readFloatsFP32(
        from data: MPSGraphTensorData,
        into pointer: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        // A raw byte copy cannot tell fp32 from a narrower dtype, and a
        // mismatch reads garbage without any error. Every caller reads a
        // graph output whose dtype and size are fixed by construction, so a
        // violation is a graph-wiring bug: trap on it.
        precondition(
            data.dataType == .float32,
            "readFloatsFP32: tensor data is \(data.dataType), not fp32"
        )
        let elementCount = data.shape.reduce(1) { $0 * $1.intValue }
        precondition(
            elementCount == count,
            "readFloatsFP32: tensor data holds \(elementCount) elements, caller expects \(count)"
        )
        data.mpsndarray().readBytes(UnsafeMutableRawPointer(pointer), strideBytes: nil)
    }

    /// Write a Float32 array into an **fp32** ND array (raw bytes). Counterpart
    /// to `readFloatsFP32` for fp32 optimizer-state load.
    static func writeFloatsFP32(_ floats: [Float], into array: MPSNDArray) {
        precondition(
            !floats.isEmpty,
            "writeFloatsFP32: empty input would silently skip the MPSNDArray write"
        )
        var local = floats
        local.withUnsafeMutableBytes { buf in
            guard let base = buf.baseAddress else {
                preconditionFailure("writeFloatsFP32: baseAddress nil despite non-empty input")
            }
            array.writeBytes(base, strideBytes: nil)
        }
    }

    /// Narrow a Float32 to bfloat16, returned as raw 16 bits.
    ///
    /// bfloat16 is literally the high 16 bits of an IEEE float32 — same
    /// sign, same 8-bit exponent, top 7 of the 23 mantissa bits — so the
    /// conversion is a shift with round-to-nearest-ties-to-even on the
    /// discarded low 16 bits. NaN is preserved explicitly so a near-NaN
    /// payload can't round up into an infinity. (vImage has no bfloat16
    /// primitive, unlike IEEE half, so this is done by hand.)
    @inline(__always)
    static func float32ToBFloat16Bits(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let isNaN = (bits & 0x7F80_0000) == 0x7F80_0000
            && (bits & 0x007F_FFFF) != 0
        if isNaN {
            return UInt16(truncatingIfNeeded: (bits >> 16) | 0x0040)
        }
        let keptLSB = (bits >> 16) & 1
        let roundingBias = UInt32(0x7FFF) &+ keptLSB
        let rounded = bits &+ roundingBias          // wrapping add — never traps
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    /// Widen a raw bfloat16 (16 bits) back to Float32 — exact, just a left
    /// shift into the high half of the float32 bit pattern.
    @inline(__always)
    static func bFloat16BitsToFloat32(_ half: UInt16) -> Float {
        return Float(bitPattern: UInt32(half) << 16)
    }

    /// Bytes per weight/activation element in `dataType`: Float32 → 4,
    /// Float16 / bFloat16 → 2. Single source of truth for any byte ↔
    /// element-count conversion (so callers never hardcode `MemoryLayout
    /// <Float>.size` and silently halve the count under a 16-bit dtype).
    static func bytesPerWeightElement(for dataType: MPSDataType) -> Int {
        switch dataType {
        case .float32: return MemoryLayout<Float>.size
        case .float16, .bFloat16: return MemoryLayout<UInt16>.size
        default: fatalError("Unsupported compute dataType: \(dataType)")
        }
    }

    /// Relative machine epsilon (the ULP of 1.0) for `dataType` —
    /// `2^-mantissaBits`: Float32 ≈ 1.19e-7, Float16 ≈ 9.77e-4, bFloat16
    /// ≈ 7.81e-3. The correct scale for a *relative* numerical tolerance:
    /// an absolute tolerance of `weightRelativeEpsilon · |x|` is one ULP at
    /// magnitude `|x|`. Lets numeric tests derive accuracy bounds from the
    /// active dtype instead of hardcoding fp32-era constants.
    static func weightRelativeEpsilon(for dataType: MPSDataType) -> Float {
        switch dataType {
        case .float32: return Float.ulpOfOne   // 2^-23
        case .float16: return 0x1p-10          // 2^-10
        case .bFloat16: return 0x1p-7          // 2^-7
        default: fatalError("Unsupported compute dataType: \(dataType)")
        }
    }

    /// Decode raw `dataType` weight bytes (as produced by `makeWeightData`)
    /// back into Float32 — the exact inverse of `makeWeightData`. Element
    /// count is inferred as `data.count / bytesPerWeightElement`.
    static func decodeWeightData(_ data: Data, dataType: MPSDataType) -> [Float] {
        let count = data.count / bytesPerWeightElement(for: dataType)
        switch dataType {
        case .float32:
            return data.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Float.self).prefix(count))
            }
        case .bFloat16:
            return data.withUnsafeBytes { raw in
                let half = raw.bindMemory(to: UInt16.self)
                return (0..<count).map { bFloat16BitsToFloat32(half[$0]) }
            }
        case .float16:
            var floats = [Float](repeating: 0, count: count)
            data.withUnsafeBytes { raw in
                let srcBase = UnsafeMutableRawPointer(mutating: raw.baseAddress!)
                floats.withUnsafeMutableBufferPointer { dst in
                    var src = vImage_Buffer(
                        data: srcBase, height: 1, width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<UInt16>.size
                    )
                    var dstB = vImage_Buffer(
                        data: dst.baseAddress, height: 1, width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<Float>.size
                    )
                    _ = vImageConvert_Planar16FtoPlanarF(&src, &dstB, 0)
                }
            }
            return floats
        default:
            fatalError("Unsupported compute dataType: \(dataType)")
        }
    }

    /// Convert a Float32 array into bytes laid out in `Self.mpsDataType(for: arch)`.
    /// Float32 → passthrough; Float16 → conversion via vImage; bFloat16 →
    /// bit-shift narrowing.
    static func makeWeightData(_ floats: [Float], dataType: MPSDataType) -> Data {
        switch dataType {
        case .float32:
            return floats.withUnsafeBytes { Data($0) }

        case .bFloat16:
            var halfBuf = [UInt16](repeating: 0, count: floats.count)
            for i in 0..<floats.count {
                halfBuf[i] = float32ToBFloat16Bits(floats[i])
            }
            return halfBuf.withUnsafeBytes { Data($0) }

        case .float16:
            let count = floats.count
            var halfBuf = [UInt16](repeating: 0, count: count)
            floats.withUnsafeBufferPointer { srcBuf in
                halfBuf.withUnsafeMutableBufferPointer { dstBuf in
                    var src = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: srcBuf.baseAddress),
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<Float>.size
                    )
                    var dst = vImage_Buffer(
                        data: dstBuf.baseAddress,
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<UInt16>.size
                    )
                    _ = vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
                }
            }
            return halfBuf.withUnsafeBytes { Data($0) }

        default:
            fatalError("Unsupported compute dataType: \(dataType)")
        }
    }

    /// Read inference output as Float32, converting from `Self.mpsDataType(for: arch)`.
    static func readFloats(from data: MPSGraphTensorData, count: Int, dataType: MPSDataType) -> [Float] {
        switch dataType {
        case .float32:
            var out = [Float](repeating: 0, count: count)
            out.withUnsafeMutableBytes { buf in
                if let ptr = buf.baseAddress {
                    data.mpsndarray().readBytes(ptr, strideBytes: nil)
                }
            }
            return out

        case .bFloat16:
            var halfBuf = [UInt16](repeating: 0, count: count)
            var out = [Float](repeating: 0, count: count)
            halfBuf.withUnsafeMutableBufferPointer { hb in
                guard let src = hb.baseAddress else {
                    preconditionFailure("readFloats: bf16 staging baseAddress nil (count=\(count))")
                }
                data.mpsndarray().readBytes(UnsafeMutableRawPointer(src), strideBytes: nil)
                out.withUnsafeMutableBufferPointer { ob in
                    guard let dst = ob.baseAddress else {
                        preconditionFailure("readFloats: out baseAddress nil (count=\(count))")
                    }
                    // Bare-pointer `while` widen — bit-identical to the old
                    // `for i in 0..<count` over Array subscripts, minus the
                    // iterator/bounds-check overhead that dominated the path.
                    var i = 0
                    while i < count {
                        dst[i] = bFloat16BitsToFloat32(src[i])
                        i += 1
                    }
                }
            }
            return out

        case .float16:
            var halfBuf = [UInt16](repeating: 0, count: count)
            halfBuf.withUnsafeMutableBytes { buf in
                if let ptr = buf.baseAddress {
                    data.mpsndarray().readBytes(ptr, strideBytes: nil)
                }
            }
            var out = [Float](repeating: 0, count: count)
            halfBuf.withUnsafeMutableBufferPointer { srcBuf in
                out.withUnsafeMutableBufferPointer { dstBuf in
                    var src = vImage_Buffer(
                        data: srcBuf.baseAddress,
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<UInt16>.size
                    )
                    var dst = vImage_Buffer(
                        data: dstBuf.baseAddress,
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<Float>.size
                    )
                    _ = vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                }
            }
            return out

        default:
            fatalError("Unsupported compute dataType: \(dataType)")
        }
    }

    /// Read inference output into a caller-owned float buffer. Used by
    /// the hot inference and training paths so the readback doesn't
    /// allocate a fresh Swift array on every call. The `count` argument
    /// must match the underlying tensor's element count (it's validated
    /// only in debug via the MPSNDArray shape, not here).
    ///
    /// `.float16` reads the half bytes into a transient `[UInt16]` then
    /// widens into the caller's buffer with a single vImage planar pass
    /// (matching the array-returning `readFloats` overload); bf16 widens
    /// with a bare-pointer loop instead because no vImage bf16 converter
    /// exists.
    static func readFloats(
        from data: MPSGraphTensorData,
        into pointer: UnsafeMutablePointer<Float>,
        count: Int,
        dataType: MPSDataType
    ) {
        switch dataType {
        case .float32:
            data.mpsndarray().readBytes(
                UnsafeMutableRawPointer(pointer),
                strideBytes: nil
            )
        case .float16:
            var halfBuf = [UInt16](repeating: 0, count: count)
            halfBuf.withUnsafeMutableBufferPointer { hb in
                guard let src = hb.baseAddress else {
                    preconditionFailure("readFloats(into:): fp16 staging baseAddress nil (count=\(count))")
                }
                data.mpsndarray().readBytes(UnsafeMutableRawPointer(src), strideBytes: nil)
                var srcImg = vImage_Buffer(
                    data: src,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.size
                )
                var dstImg = vImage_Buffer(
                    data: UnsafeMutableRawPointer(pointer),
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size
                )
                _ = vImageConvert_Planar16FtoPlanarF(&srcImg, &dstImg, 0)
            }
        case .bFloat16:
            // Read the bf16 bytes into a transient [UInt16] then widen into
            // the caller's Float buffer. The widen runs as a bare-pointer
            // `while` loop: profiling this exact site showed the per-element
            // Array subscript + `IndexingIterator`/`Int.==` range machinery —
            // not the `bFloat16BitsToFloat32` shift — was the cost. Output is
            // bit-identical to the old `for` loop. The transient alloc is a
            // negligible fraction of the profile and is left as-is.
            var halfBuf = [UInt16](repeating: 0, count: count)
            halfBuf.withUnsafeMutableBufferPointer { hb in
                guard let src = hb.baseAddress else {
                    preconditionFailure("readFloats(into:): bf16 staging baseAddress nil (count=\(count))")
                }
                data.mpsndarray().readBytes(UnsafeMutableRawPointer(src), strideBytes: nil)
                var i = 0
                while i < count {
                    pointer[i] = bFloat16BitsToFloat32(src[i])
                    i += 1
                }
            }
        default:
            fatalError("readFloats(from:into:count:): unsupported dataType \(dataType).")
        }
    }
}
