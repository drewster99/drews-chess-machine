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
    case outputMissing(String)
    case weightLoadMismatch(String)
    case variableShapeMissing(String)
    case boardSizeMismatch(expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .metalNotSupported:
            return "Metal is not supported on this device"
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .descriptorCreationFailed:
            return "Failed to create convolution descriptor"
        case .outputMissing(let name):
            return "Inference output missing: \(name)"
        case .weightLoadMismatch(let detail):
            return "Weight load mismatch: \(detail)"
        case .variableShapeMissing(let name):
            return "Variable '\(name)' has no shape — cannot size load placeholder"
        case .boardSizeMismatch(let expected, let got):
            return "Inference input size mismatch: expected \(expected) floats, got \(got)"
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

    /// Numeric precision for all weights and activations.
    ///
    /// **Currently `.float32`** after a bf16 experiment (commit `bb70ab8`,
    /// branch `bf16-trainer`) measured that pure-bf16 weights+updates
    /// stall training within ~10k steps. The bf16 trap: with weights
    /// stored in bf16 and the SGD step `w -= lr · grad` done in bf16
    /// too, an update smaller than the bf16 ULP at `|w|` rounds to a
    /// no-op. For BN gamma at value 1.0 the ULP is `2^-7 ≈ 7.8e-3`,
    /// so `lr=1e-3 · grad≈0.05` is 3 orders of magnitude below the
    /// threshold and every gamma update rounds to zero. Confirmed
    /// in the KXvb-1 snapshot: all 3,712 BN gamma channels bit-exact
    /// at 1.0 after 49k steps; `value_fc2_bias[D]` stuck at one bf16
    /// ULP from `ln 6`; tower L2 norms within 1% of init. The fix is
    /// either (a) mixed precision with an fp32 master copy LC0-style,
    /// or (b) just stay in fp32. We're doing (b) for now; bf16 +
    /// mixed-precision is on the table if MPSGraph's bf16 matmuls
    /// turn out to be meaningfully faster than fp32 on Apple Silicon
    /// (initial measurements suggest the gap is small).
    ///
    /// All weight/activation byte conversion is centralized in
    /// `makeWeightData` (Float32 → dataType) and the two `readFloats`
    /// overloads (dataType → Float32), which branch on this. Saved
    /// `.dcmmodel` files always speak `Float` (conversion happens only at
    /// the GPU boundary), so a model trained under one dataType reloads
    /// cleanly under another (modulo rounding at the GPU-boundary
    /// re-cast). `.float16` uses the vImage half path; `.bFloat16` uses
    /// the bit-shift helpers `float32ToBFloat16Bits` /
    /// `bFloat16BitsToFloat32` (vImage has no bfloat16 primitive). The
    /// `.bFloat16` and `.float16` paths are preserved so the experiment
    /// can be re-run without re-implementing the conversion plumbing.
    ///
    /// `inputPlanes` etc. are independent of this.
    static let dataType: MPSDataType = .bFloat16

    /// Input plane count. v3 architecture: 20 baseline planes (pieces +
    /// castling + EP + halfmove clock + 2 repetition-count planes) plus
    /// 10 binary temporal-repetition-history planes (planes 20–29 in
    /// `BoardEncoder`). Changing this value automatically propagates
    /// through `BoardEncoder.tensorLength`, `ReplayBuffer.floatsPerBoard`,
    /// the stem's weight shape `[channels, inputPlanes, k, k]` (k =
    /// `towerConvKernelSize`), and the
    /// network's `arch_hash`, so old checkpoints with a different value
    /// fail to load with a clear shape mismatch at startup.
    static let inputPlanes = 30
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
    // NetworkArchitecture`. Read `arch.channels`, `arch.numBlocks`,
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
    let policyOutput: MPSGraphTensor
    /// fp32 cast of `policyOutput`, used only as the inference readback
    /// target so the CPU reads raw fp32 bytes (the bf16→fp32 widen of the
    /// wide policy output happens on the GPU, not in a host loop). Identical
    /// to `policyOutput` on a `.float32` build. The training graph never
    /// targets this tensor — the trainer reads `policyOutput` (compute
    /// dtype) for its loss — so it costs nothing during training.
    private let policyOutputReadback: MPSGraphTensor
    /// Derived scalar value, shape `[batch, 1]` = `p_win − p_loss`
    /// (= E[outcome] ∈ [−1, +1], no tanh). This is what every inference
    /// consumer reads and what the policy-gradient baseline is fed; the
    /// full W/D/L distribution stays available via `valueLogits` /
    /// `valueProbs` for the value loss and diagnostics.
    let valueOutput: MPSGraphTensor
    /// fp32 cast of `valueOutput` (or `valueOutput` itself on a `.float32`
    /// build), used only as the target of the trainer's GPU→GPU value-baseline
    /// forward so the per-position `v(s)` lands in an fp32 buffer that feeds the
    /// training step's fp32 `vBaseline` placeholder with no CPU round-trip.
    /// Mirrors `policyOutputReadback`. See `computeValueBaselineGPU`.
    let valueOutputFP32: MPSGraphTensor
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
    /// `Self.dataType`; the tensor data wrapper is built once and fed
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
    /// forward, keyed by batch size. Targets just `valueOutputFP32` (no policy /
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

    /// The architecture this instance was built to. Drives every layer shape
    /// in the build path (the static `channels`/`numBlocks`/… constants remain
    /// as the *default* arch for external callers; a guard test asserts they
    /// match `NetworkArchitecture.current`). Compute dtype stays on the static
    /// `dataType` for now — per-model precision is a later phase.
    let arch: NetworkArchitecture

    // MARK: Initialization

    /// Build the network. Default `bnMode = .inference` keeps the existing
    /// behavior for play / forward-pass demos; pass `.training` to build a
    /// copy whose BN layers compute fresh batch stats on every forward pass
    /// (used by ChessTrainer for accurate training-step benchmarks).
    init(arch: NetworkArchitecture = .current, bnMode: BNMode = .inference) throws {
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

        let conv1x1 = try Self.makeConv1x1Descriptor()
        let stemConvDescriptor = try Self.makeConvDescriptor(kernelSize: arch.stemConvKernelSize)


        // Input: [batch, inputPlanes, 8, 8]. The placeholder is fp32 — the
        // CPU feeds raw fp32 board planes (no host-side bf16 narrowing) and
        // the narrowing to the compute dtype runs on the GPU via the cast
        // below, the graph's first op. On a `.float32` build the cast is the
        // identity and is elided. This same placeholder is fed by the
        // self-play / arena inference paths *and* by the trainer's board
        // feed, so both write fp32 (see ChessTrainer.feedsForBatch).
        let input = g.placeholder(
            shape: [-1, NSNumber(value: Self.inputPlanes), 8, 8],
            dataType: .float32,
            name: "board_input"
        )
        inputPlaceholder = input
        let computeInput = (Self.dataType == .float32)
            ? input
            : g.cast(input, to: Self.dataType, name: "board_input_cast")

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

        // --- Stem: same-padded conv (inputPlanes -> 128) -> BN -> ReLU ---

        let stemWeights = g.variable(
            with: Self.heInitDataConvOIHW(
                shape: [arch.channels, Self.inputPlanes, arch.stemConvKernelSize, arch.stemConvKernelSize]
            ),
            shape: [
                NSNumber(value: arch.channels),
                NSNumber(value: Self.inputPlanes),
                NSNumber(value: arch.stemConvKernelSize),
                NSNumber(value: arch.stemConvKernelSize)
            ],
            dataType: Self.dataType,
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
            graph: g, input: x, channels: arch.channels, name: "stem_bn", bnMode: bnMode,
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

        // --- Tower: pre-activation residual blocks (count = `numBlocks`) ---

        for i in 0..<arch.numBlocks {
            x = try Self.residualBlock(
                graph: g,
                arch: arch,
                input: x,
                blockIndex: i,
                bnMode: bnMode,
                trainables: &trainables,
                shouldDecay: &shouldDecay,
                runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssigns,
                batchMeans: &batchMeans,
                batchVars: &batchVars
            )
        }

        // --- Tower-end normalization (pre-activation only) ---
        //
        // Each pre-activation block ends in a bare conv-add on a clean identity
        // skip, so the tower output is an un-normalized, never-activated linear
        // accumulation — normalize + activate it here for the heads. A
        // post-activation tower ends each block in an activation, so its output
        // is already conditioned and no tower-end BN exists (matches v3).
        if arch.hasTowerEndBN {
            x = Self.batchNorm(
                graph: g, input: x, channels: arch.channels, name: "tower_final_bn", bnMode: bnMode,
                trainables: &trainables,
                shouldDecay: &shouldDecay,
                runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssigns,
                batchMeans: &batchMeans,
                batchVars: &batchVars
            )
            x = Self.activation(g, x, arch, name: "tower_final_act")
        }

        // --- Policy head ---

        let policy = Self.policyHead(
            graph: g, arch: arch, input: x, descriptor: conv1x1, bnMode: bnMode,
            trainables: &trainables,
            shouldDecay: &shouldDecay,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssigns,
            batchMeans: &batchMeans,
            batchVars: &batchVars
        )
        policyOutput = policy.output
        policyOutputReadback = (Self.dataType == .float32)
            ? policy.output
            : g.cast(policy.output, to: .float32, name: "policy_output_f32")
        policyHeadFinalWeights = policy.finalWeights

        // --- Value head ---

        let valueHeadOut = Self.valueHead(
            graph: g, arch: arch, input: x, descriptor: conv1x1, bnMode: bnMode,
            trainables: &trainables,
            shouldDecay: &shouldDecay,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssigns,
            batchMeans: &batchMeans,
            batchVars: &batchVars
        )
        valueOutput = valueHeadOut.scalar
        valueOutputFP32 = (Self.dataType == .float32)
            ? valueHeadOut.scalar
            : g.cast(valueHeadOut.scalar, to: .float32, name: "value_scalar_f32")
        valueLogits = valueHeadOut.logits
        valueProbs = valueHeadOut.probs

        trainableVariables = trainables
        trainableShouldDecay = shouldDecay
        bnRunningStatsVariables = runningStats
        bnRunningStatsAssignOps = runningStatsAssigns
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
            let ph = g.placeholder(
                shape: shape,
                dataType: Self.dataType,
                name: "\(v.operation.name)_load"
            )
            let assignOp = g.assign(v, tensor: ph, name: "\(v.operation.name)_load_assign")
            loadPlaceholders.append(ph)
            loadAssignOps.append(assignOp)

            let desc = MPSNDArrayDescriptor(dataType: Self.dataType, shape: shape)
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
            shape: [1, NSNumber(value: Self.inputPlanes), 8, 8]
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
            [Float](repeating: 0, count: 1 * Self.inputPlanes * Self.boardSize * Self.boardSize),
            into: dummyND
        )
        dummyInferenceInputTensorData = MPSGraphTensorData(dummyND)

        // Cache the feeds dict and target tensor list so the per-move
        // inference path doesn't rebuild them. Both are immutable — the
        // ND array backing `inferenceInputTensorData` is written
        // through `writeBytes` on the same underlying storage every
        // call.
        inferenceFeeds = [inputPlaceholder: inferenceInputTensorData]
        // Policy is read back as fp32 (GPU-cast); the value scalar and WDL
        // probs stay compute-dtype and are widened on the host — far too
        // small to be worth a GPU cast + its readback dispatch.
        inferenceTargets = [policyOutputReadback, valueOutput, valueProbs]

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
    /// In self-play both `MPSChessPlayer` instances share one
    /// `ChessNetwork` but are driven sequentially inside a single
    /// `ChessMachine.runGameLoop`, so only one side evaluates at a
    /// time. Any future refactor that runs two games concurrently on
    /// one network must give each game its own `ChessNetwork` or add
    /// explicit serialization here.
    ///
    /// - Parameter board: `inputPlanes`×8×8 = 1,920 floats in NCHW order (planes, rows, cols).
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
        let expected = 1 * Self.inputPlanes * Self.boardSize * Self.boardSize
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

            guard let policyData = results[policyOutputReadback] else {
                throw ChessNetworkError.outputMissing("policy")
            }
            guard let valueData = results[valueOutput] else {
                throw ChessNetworkError.outputMissing("value")
            }

            Self.readFloatsFP32(from: policyData, into: inferencePolicyScratchPtr, count: Self.policySize)
            Self.readFloats(from: valueData, into: inferenceValueScratchPtr, count: 1)

            consume(
                UnsafeBufferPointer(start: inferencePolicyScratchPtr, count: Self.policySize),
                inferenceValueScratchPtr.pointee
            )
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
        let expected = 1 * Self.inputPlanes * Self.boardSize * Self.boardSize
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
                Self.readFloats(from: probsData, into: inferenceValueProbsScratchPtr, count: arch.valueHeadClasses)
                return (
                    win: inferenceValueProbsScratchPtr[0],
                    draw: inferenceValueProbsScratchPtr[1],
                    loss: inferenceValueProbsScratchPtr[2]
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
        // Validation runs synchronously on `executionQueue` after the
        // [Float] has been pinned via `withUnsafeBufferPointer`. `count`
        // and `batchBoards.count` are stable for the rest of the body
        // because Swift value-type semantics isolate our captured copy
        // from the caller's binding (COW), and the buffer pointer's
        // count is set at construction and never derived dynamically.
        guard count >= 1 else {
            throw ChessNetworkError.boardSizeMismatch(expected: Self.inputPlanes * Self.boardSize * Self.boardSize, got: 0)
        }
        let expected = count * Self.inputPlanes * Self.boardSize * Self.boardSize
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

            guard let policyData = results[policyOutputReadback] else {
                throw ChessNetworkError.outputMissing("policy")
            }
            guard let valueData = results[valueOutput] else {
                throw ChessNetworkError.outputMissing("value")
            }
            guard let valueProbsData = results[valueProbs] else {
                throw ChessNetworkError.outputMissing("valueProbs")
            }

            Self.readFloatsFP32(from: policyData, into: policyPtr, count: count * Self.policySize)
            Self.readFloats(from: valueData, into: valuePtr, count: count)
            Self.readFloats(from: valueProbsData, into: valueProbsPtr, count: count * arch.valueHeadClasses)

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
            shape: [NSNumber(value: count), NSNumber(value: Self.inputPlanes), 8, 8]
        )
        let nda = MPSNDArray(device: metalDevice, descriptor: desc)
        nda.label = "inference.input[\(count)]"
        let tensorData = MPSGraphTensorData(nda)
        let feeds: [MPSGraphTensor: MPSGraphTensorData] = [inputPlaceholder: tensorData]
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
        let executable = graph.compile(
            with: MPSGraphDevice(mtlDevice: metalDevice),
            feeds: feedShapes,
            targetTensors: inferenceTargets,
            targetOperations: nil,
            compilationDescriptor: desc
        )
        inferenceExecutables[count] = executable
        return executable
    }

    /// GPU→GPU value baseline. Runs a value-only forward on this network's
    /// current weights and leaves the per-position `v(s)` (fp32, shape
    /// `[count, 1]`) in a network-owned GPU buffer, handed to `consume` WITHOUT a
    /// CPU readback. The trainer feeds that buffer straight into the training
    /// step's `vBaseline` placeholder, eliminating the `Array(valuesBuf)` copy +
    /// staging re-write the old `evaluateBatched` baseline path did. Numerically
    /// identical to that path — same `v(s)` (the old path read `valueOutput`
    /// back as bf16→fp32; this casts bf16→fp32 on the GPU) — just no CPU round
    /// trip. The result buffer is reused per call; safe because the trainer
    /// drives this serially (phase 2 fully completes before phase 3 reads it).
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
        guard count >= 1 else {
            throw ChessNetworkError.boardSizeMismatch(expected: Self.inputPlanes * Self.boardSize * Self.boardSize, got: 0)
        }
        let expected = count * Self.inputPlanes * Self.boardSize * Self.boardSize
        guard batchBoards.count == expected else {
            throw ChessNetworkError.boardSizeMismatch(expected: expected, got: batchBoards.count)
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
            consume(resultTD)
        }
    }

    /// Compile + cache the value-only baseline executable for batch size `count`.
    /// Feed shapes are taken from the live board feed (the placeholder carries a
    /// `-1` batch dim). Target is just `valueOutputFP32`; no target operations
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
        let executable = graph.compile(
            with: MPSGraphDevice(mtlDevice: metalDevice),
            feeds: feedShapes,
            targetTensors: [valueOutputFP32],
            targetOperations: nil,
            compilationDescriptor: desc
        )
        valueBaselineExecutables[count] = executable
        return executable
    }

    /// Network-owned fp32 `[count, 1]` result buffer for the value baseline,
    /// matching `valueOutputFP32`'s shape/dtype. Cached per batch size.
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

    private func internalExportWeights() throws -> [[Float]] {
        let allVars = trainableVariables + bnRunningStatsVariables

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
                out.append(Self.readFloats(from: data, count: count))
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
            Self.writeFloats(weights[i], into: weightLoadNDArrays[i])
            feeds[weightLoadPlaceholders[i]] = weightLoadTensorData[i]
        }

        // graph.run requires at least one target tensor. Use the first
        // persistent variable as a dummy read — its value after the
        // assigns run is whatever we just wrote in, which we ignore.
        // Autoreleasepool-wrapped for the same reason as the other
        // graph.run sites in this file.
        autoreleasepool {
            _ = graph.run(
                with: commandQueue,
                feeds: feeds,
                targetTensors: [allVars[0]],
                targetOperations: weightLoadAssignOps
            )
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
        guard !bnBatchMeanTensors.isEmpty else {
            throw ChessNetworkError.outputMissing(
                "computeBatchStats: bnBatchMeanTensors is empty — this method requires bnMode = .training"
            )
        }
        guard count >= 1 else {
            throw ChessNetworkError.boardSizeMismatch(
                expected: Self.inputPlanes * Self.boardSize * Self.boardSize, got: 0
            )
        }
        let expected = count * Self.inputPlanes * Self.boardSize * Self.boardSize
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
                means.append(Self.readFloats(from: data, count: n))
            }
            for t in bnBatchVarTensors {
                guard let data = results[t] else {
                    throw ChessNetworkError.outputMissing(t.operation.name)
                }
                let n = try Self.elementCount(of: t)
                vars_.append(Self.readFloats(from: data, count: n))
            }
            return (means: means, vars: vars_)
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
            Self.writeFloats(means[layer], into: weightLoadNDArrays[meanIdx])
            Self.writeFloats(vars[layer], into: weightLoadNDArrays[varIdx])
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

    /// The single network-wide hidden activation, selected by `arch.activationFunction`.
    /// SiLU = `x*sigmoid(x)`; GELU exact (erf-based). Used at every hidden site (block
    /// main path, SE FC1, tower-end, heads). The SE gate (sigmoid) and value output
    /// (tanh/softmax) are structural and call their own ops directly.
    private static func activation(
        _ graph: MPSGraph, _ x: MPSGraphTensor, _ arch: NetworkArchitecture, name: String
    ) -> MPSGraphTensor {
        switch arch.activationFunction {
        case .relu:
            return graph.reLU(with: x, name: name)
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
        arch.computeDataType == .bFloat16 ? .bFloat16 : .float32
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
    /// gamma and beta are appended to `trainables` in both modes.
    /// Running-stat variables are appended to `runningStats` in both
    /// modes. Only `.training` appends to `runningStatsAssignOps`.
    private static func batchNorm(
        graph: MPSGraph,
        input: MPSGraphTensor,
        channels: Int,
        name: String,
        bnMode: BNMode,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation],
        batchMeans: inout [MPSGraphTensor],
        batchVars: inout [MPSGraphTensor]
    ) -> MPSGraphTensor {
        let ch = NSNumber(value: channels)

        // gamma and beta are trainable in both modes. All BN layers init
        // γ=1, β=0 (standard). The old zero-γ "identity block" init is
        // gone — in the pre-activation tower the per-block ReZero scalar α
        // (init `1/√numBlocks`) owns depth-variance control instead, and
        // unlike zero-γ it lets every block contribute signal *and*
        // gradient from step 1. See `residualBlock`.
        let gamma = graph.variable(
            with: onesData(count: channels),
            shape: [1, ch, 1, 1],
            dataType: Self.dataType,
            name: "\(name)_gamma"
        )
        let beta = graph.variable(
            with: zerosData(count: channels),
            shape: [1, ch, 1, 1],
            dataType: Self.dataType,
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
            with: zerosData(count: channels),
            shape: [1, ch, 1, 1],
            dataType: Self.dataType,
            name: "\(name)_running_mean"
        )
        let runningVar = graph.variable(
            with: onesData(count: channels),
            shape: [1, ch, 1, 1],
            dataType: Self.dataType,
            name: "\(name)_running_var"
        )
        runningStats.append(runningMean)
        runningStats.append(runningVar)

        let meanTensor: MPSGraphTensor
        let varianceTensor: MPSGraphTensor

        switch bnMode {
        case .inference:
            meanTensor = runningMean
            varianceTensor = runningVar

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
            let momentum = graph.constant(0.99, dataType: Self.dataType)
            let oneMinusMomentum = graph.constant(0.01, dataType: Self.dataType)

            let scaledOldMean = graph.multiplication(momentum, runningMean, name: nil)
            let scaledNewMean = graph.multiplication(oneMinusMomentum, bMean, name: nil)
            let updatedMean = graph.addition(
                scaledOldMean, scaledNewMean, name: "\(name)_running_mean_update"
            )
            let assignMean = graph.assign(
                runningMean, tensor: updatedMean, name: "\(name)_running_mean_assign"
            )
            runningStatsAssignOps.append(assignMean)

            let scaledOldVar = graph.multiplication(momentum, runningVar, name: nil)
            let scaledNewVar = graph.multiplication(oneMinusMomentum, bVar, name: nil)
            let updatedVar = graph.addition(
                scaledOldVar, scaledNewVar, name: "\(name)_running_var_update"
            )
            let assignVar = graph.assign(
                runningVar, tensor: updatedVar, name: "\(name)_running_var_assign"
            )
            runningStatsAssignOps.append(assignVar)
        }

        return graph.normalize(
            input,
            mean: meanTensor,
            variance: varianceTensor,
            gamma: gamma,
            beta: beta,
            epsilon: 1e-5,
            name: name
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
    /// `α` (`*_res_scale`) is a per-block trainable scalar, init
    /// `1/√numBlocks`. With L additive branches of ~unit variance the tower
    /// variance grows ~L; the `1/√L` init holds it ~O(1) while still letting
    /// every block contribute signal *and* gradient from step 1 (unlike the
    /// old zero-γ init, whose branch was dead until gradient woke it). It is
    /// excluded from weight decay. Reduction ratio = `seReductionRatio`.
    private static func residualBlock(
        graph: MPSGraph,
        arch: NetworkArchitecture,
        input: MPSGraphTensor,
        blockIndex: Int,
        bnMode: BNMode,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation],
        batchMeans: inout [MPSGraphTensor],
        batchVars: inout [MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        let prefix = "block\(blockIndex)"
        let channels = arch.channels
        let conv1Desc = try makeConvDescriptor(kernelSize: arch.blockConv1KernelSize)
        let conv2Desc = try makeConvDescriptor(kernelSize: arch.blockConv2KernelSize)

        // Bias-free, He-init conv weight (caller appends to `trainables`).
        func makeConvWeight(_ name: String, _ k: Int) -> MPSGraphTensor {
            graph.variable(
                with: heInitDataConvOIHW(shape: [channels, channels, k, k]),
                shape: [NSNumber(value: channels), NSNumber(value: channels), NSNumber(value: k), NSNumber(value: k)],
                dataType: Self.dataType,
                name: "\(name)_weights"
            )
        }

        // Residual function F(input). `z` is the SE input: the raw conv2 output
        // in pre-activation, or the BN2 output in post-activation. The append
        // order here is the single source of truth that `weightTensorPlan`
        // mirrors (pre: bn1,conv1,bn2,conv2 ; post: conv1,bn1,conv2,bn2).
        let z: MPSGraphTensor
        switch arch.blockActivationStyle {
        case .pre:
            var h = batchNorm(graph: graph, input: input, channels: channels, name: "\(prefix)_bn1", bnMode: bnMode,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            h = activation(graph, h, arch, name: "\(prefix)_act1")
            let conv1W = makeConvWeight("\(prefix)_conv1", arch.blockConv1KernelSize)
            trainables.append(conv1W); shouldDecay.append(true)
            h = graph.convolution2D(h, weights: conv1W, descriptor: conv1Desc, name: "\(prefix)_conv1")
            h = batchNorm(graph: graph, input: h, channels: channels, name: "\(prefix)_bn2", bnMode: bnMode,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            h = activation(graph, h, arch, name: "\(prefix)_act2")
            let conv2W = makeConvWeight("\(prefix)_conv2", arch.blockConv2KernelSize)
            trainables.append(conv2W); shouldDecay.append(true)
            z = graph.convolution2D(h, weights: conv2W, descriptor: conv2Desc, name: "\(prefix)_conv2")
        case .post:
            let conv1W = makeConvWeight("\(prefix)_conv1", arch.blockConv1KernelSize)
            trainables.append(conv1W); shouldDecay.append(true)
            var h = graph.convolution2D(input, weights: conv1W, descriptor: conv1Desc, name: "\(prefix)_conv1")
            h = batchNorm(graph: graph, input: h, channels: channels, name: "\(prefix)_bn1", bnMode: bnMode,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            h = activation(graph, h, arch, name: "\(prefix)_act1")
            let conv2W = makeConvWeight("\(prefix)_conv2", arch.blockConv2KernelSize)
            trainables.append(conv2W); shouldDecay.append(true)
            h = graph.convolution2D(h, weights: conv2W, descriptor: conv2Desc, name: "\(prefix)_conv2")
            z = batchNorm(graph: graph, input: h, channels: channels, name: "\(prefix)_bn2", bnMode: bnMode,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
        }

        // SE channel attention (style-dependent; identity when .none).
        let seOut = applySE(graph: graph, arch: arch, z: z, prefix: prefix,
            trainables: &trainables, shouldDecay: &shouldDecay)

        // ReZero branch scalar (optional), init `rezeroAlphaInit`, no weight decay.
        var branch = seOut
        if arch.blockUseRezero {
            let alpha = graph.variable(
                with: makeWeightData([arch.rezeroAlphaInit]),
                shape: [1], dataType: Self.dataType, name: "\(prefix)_res_scale")
            trainables.append(alpha); shouldDecay.append(false)
            branch = graph.multiplication(seOut, alpha, name: "\(prefix)_res_scaled")
        }

        // Merge with the skip.
        switch arch.blockSkipMerge {
        case .cleanAdd:
            // out = input + [alpha .] F(input) — clean identity highway (no activation on the sum).
            return graph.addition(input, branch, name: "\(prefix)_skip")
        case .activationGated:
            // out = activation(input + F(input)) — the v3 gated merge.
            let sum = graph.addition(input, branch, name: "\(prefix)_skip_sum")
            return activation(graph, sum, arch, name: "\(prefix)_skip")
        }
    }

    /// Squeeze-and-Excitation channel attention applied to `z`. Appends SE weights
    /// to `trainables` (FC1 w/b then FC2 w/b). Identity (returns `z`) when
    /// `arch.blockSeStyle == .none`. `attenuateOnly`: FC2->C, `sigmoid(z)*x`.
    /// `scaleAndBias`: FC2->2C, `sigmoid(gamma)*x + beta`.
    private static func applySE(
        graph: MPSGraph, arch: NetworkArchitecture, z: MPSGraphTensor, prefix: String,
        trainables: inout [MPSGraphTensor], shouldDecay: inout [Bool]
    ) -> MPSGraphTensor {
        guard arch.blockSeStyle != .none else { return z }
        let channels = arch.channels
        let seReduced = channels / arch.blockSeReductionRatio
        let seExpand = arch.blockSeStyle == .scaleAndBias ? 2 * channels : channels

        // Squeeze: global average pool over [H, W] -> [B, C, 1, 1] -> [B, C].
        var s = graph.mean(of: z, axes: [2, 3], name: "\(prefix)_se_squeeze")
        s = graph.reshape(s, shape: [-1, NSNumber(value: channels)], name: "\(prefix)_se_squeeze_flatten")

        // Excite FC1: C -> C/r (He), + activation.
        let fc1 = graph.variable(
            with: heInitDataFCInOut(shape: [channels, seReduced]),
            shape: [NSNumber(value: channels), NSNumber(value: seReduced)],
            dataType: Self.dataType, name: "\(prefix)_se_fc1_weights")
        let fc1b = graph.variable(
            with: zerosData(count: seReduced),
            shape: [1, NSNumber(value: seReduced)],
            dataType: Self.dataType, name: "\(prefix)_se_fc1_bias")
        trainables.append(fc1);  shouldDecay.append(true)
        trainables.append(fc1b); shouldDecay.append(false)
        s = graph.matrixMultiplication(primary: s, secondary: fc1, name: "\(prefix)_se_fc1")
        s = graph.addition(s, fc1b, name: "\(prefix)_se_fc1_bias_add")
        s = activation(graph, s, arch, name: "\(prefix)_se_act")

        // Excite FC2: C/r -> seExpand (Glorot, feeds the sigmoid gate).
        let fc2 = graph.variable(
            with: glorotInitDataFCInOut(shape: [seReduced, seExpand]),
            shape: [NSNumber(value: seReduced), NSNumber(value: seExpand)],
            dataType: Self.dataType, name: "\(prefix)_se_fc2_weights")
        let fc2b = graph.variable(
            with: zerosData(count: seExpand),
            shape: [1, NSNumber(value: seExpand)],
            dataType: Self.dataType, name: "\(prefix)_se_fc2_bias")
        trainables.append(fc2);  shouldDecay.append(true)
        trainables.append(fc2b); shouldDecay.append(false)
        s = graph.matrixMultiplication(primary: s, secondary: fc2, name: "\(prefix)_se_fc2")
        s = graph.addition(s, fc2b, name: "\(prefix)_se_fc2_bias_add")

        switch arch.blockSeStyle {
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
        descriptor: MPSGraphConvolution2DOpDescriptor,
        bnMode: BNMode,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation],
        batchMeans: inout [MPSGraphTensor],
        batchVars: inout [MPSGraphTensor]
    ) -> (output: MPSGraphTensor, finalWeights: MPSGraphTensor) {
        let channels = arch.channels
        let pc = Self.policyChannels
        let pK = arch.policyPreConvChannels

        // All styles emit 4864 raw logits in the current PolicyEncoding (76x64);
        // masking + softmax happen CPU-side. `finalWeights` is the logit-projecting
        // weight (for the trainer's ||W|| diagnostic). NCHW row-major flatten matches
        // PolicyEncoding.policyIndex = channel*64 + row*8 + col.
        switch arch.policyHeadStyle {
        case .simpleConv:
            // Single 1x1 conv channels -> 76 (+bias) -> reshape.
            let convW = graph.variable(
                with: heInitDataConvOIHW(shape: [pc, channels, 1, 1]),
                shape: [NSNumber(value: pc), NSNumber(value: channels), 1, 1],
                dataType: Self.dataType, name: "policy_conv_weights")
            let convBias = graph.variable(
                with: zerosData(count: pc),
                shape: [1, NSNumber(value: pc), 1, 1],
                dataType: Self.dataType, name: "policy_conv_bias")
            trainables.append(convW);    shouldDecay.append(true)
            trainables.append(convBias); shouldDecay.append(false)
            var x = graph.convolution2D(input, weights: convW, descriptor: descriptor, name: "policy_conv")
            x = graph.addition(x, convBias, name: "policy_conv_bias_add")
            let flat = graph.reshape(x, shape: [-1, NSNumber(value: Self.policySize)], name: "policy_flatten")
            return (output: flat, finalWeights: convW)

        case .intermediateConv:
            // 1x1 conv channels -> K -> BN -> act -> 1x1 conv K -> 76 (+bias) -> reshape.
            let preConvW = graph.variable(
                with: heInitDataConvOIHW(shape: [pK, channels, 1, 1]),
                shape: [NSNumber(value: pK), NSNumber(value: channels), 1, 1],
                dataType: Self.dataType, name: "policy_pre_conv_weights")
            trainables.append(preConvW); shouldDecay.append(true)
            var x = graph.convolution2D(input, weights: preConvW, descriptor: descriptor, name: "policy_pre_conv")
            x = batchNorm(graph: graph, input: x, channels: pK, name: "policy_pre_bn", bnMode: bnMode,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            x = activation(graph, x, arch, name: "policy_pre_act")
            let convW = graph.variable(
                with: heInitDataConvOIHW(shape: [pc, pK, 1, 1]),
                shape: [NSNumber(value: pc), NSNumber(value: pK), 1, 1],
                dataType: Self.dataType, name: "policy_conv_weights")
            let convBias = graph.variable(
                with: zerosData(count: pc),
                shape: [1, NSNumber(value: pc), 1, 1],
                dataType: Self.dataType, name: "policy_conv_bias")
            trainables.append(convW);    shouldDecay.append(true)
            trainables.append(convBias); shouldDecay.append(false)
            x = graph.convolution2D(x, weights: convW, descriptor: descriptor, name: "policy_conv")
            x = graph.addition(x, convBias, name: "policy_conv_bias_add")
            let flat = graph.reshape(x, shape: [-1, NSNumber(value: Self.policySize)], name: "policy_flatten")
            return (output: flat, finalWeights: convW)

        case .fcBottleneck:
            // 1x1 conv channels -> K -> BN -> act -> flatten(K*64) -> FC(K*64 -> 4864) (+bias).
            let preConvW = graph.variable(
                with: heInitDataConvOIHW(shape: [pK, channels, 1, 1]),
                shape: [NSNumber(value: pK), NSNumber(value: channels), 1, 1],
                dataType: Self.dataType, name: "policy_pre_conv_weights")
            trainables.append(preConvW); shouldDecay.append(true)
            var x = graph.convolution2D(input, weights: preConvW, descriptor: descriptor, name: "policy_pre_conv")
            x = batchNorm(graph: graph, input: x, channels: pK, name: "policy_pre_bn", bnMode: bnMode,
                trainables: &trainables, shouldDecay: &shouldDecay, runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssignOps, batchMeans: &batchMeans, batchVars: &batchVars)
            x = activation(graph, x, arch, name: "policy_pre_act")
            let flatSize = pK * Self.boardSize * Self.boardSize
            x = graph.reshape(x, shape: [-1, NSNumber(value: flatSize)], name: "policy_flatten_pre")
            let fcW = graph.variable(
                with: heInitDataFCInOut(shape: [flatSize, Self.policySize]),
                shape: [NSNumber(value: flatSize), NSNumber(value: Self.policySize)],
                dataType: Self.dataType, name: "policy_fc_weights")
            let fcBias = graph.variable(
                with: zerosData(count: Self.policySize),
                shape: [1, NSNumber(value: Self.policySize)],
                dataType: Self.dataType, name: "policy_fc_bias")
            trainables.append(fcW);    shouldDecay.append(true)
            trainables.append(fcBias); shouldDecay.append(false)
            x = graph.matrixMultiplication(primary: x, secondary: fcW, name: "policy_fc")
            let logits = graph.addition(x, fcBias, name: "policy_fc_bias_add")
            return (output: logits, finalWeights: fcW)
        }
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
        descriptor: MPSGraphConvolution2DOpDescriptor,
        bnMode: BNMode,
        trainables: inout [MPSGraphTensor],
        shouldDecay: inout [Bool],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation],
        batchMeans: inout [MPSGraphTensor],
        batchVars: inout [MPSGraphTensor]
    ) -> (scalar: MPSGraphTensor, logits: MPSGraphTensor, probs: MPSGraphTensor) {
        // 1x1 conv: compress the trunk to `valueHeadConvChannels` scoring maps.
        let convChannels = arch.valueHeadConvChannels
        let convW = graph.variable(
            with: heInitDataConvOIHW(shape: [convChannels, arch.channels, 1, 1]),
            shape: [NSNumber(value: convChannels), NSNumber(value: arch.channels), 1, 1],
            dataType: Self.dataType,
            name: "value_conv_weights"
        )
        trainables.append(convW)
        shouldDecay.append(true)
        var x = graph.convolution2D(
            input, weights: convW, descriptor: descriptor, name: "value_conv"
        )
        x = batchNorm(
            graph: graph, input: x, channels: convChannels, name: "value_bn", bnMode: bnMode,
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
            with: heInitDataFCInOut(shape: [flattenSize, hidden]),
            shape: [NSNumber(value: flattenSize), NSNumber(value: hidden)],
            dataType: Self.dataType,
            name: "value_fc1_weights"
        )
        let fc1Bias = graph.variable(
            with: zerosData(count: hidden),
            shape: [1, NSNumber(value: hidden)],
            dataType: Self.dataType,
            name: "value_fc1_bias"
        )
        trainables.append(fc1W)
        shouldDecay.append(true)
        trainables.append(fc1Bias)
        shouldDecay.append(false)
        x = graph.matrixMultiplication(primary: x, secondary: fc1W, name: "value_fc1")
        x = graph.addition(x, fc1Bias, name: "value_fc1_bias_add")
        x = activation(graph, x, arch, name: "value_fc1_act")

        // FC2: hidden -> valueHeadClasses (3 = W/D/L logits, or 1 = scalar pre-tanh).
        let classes = arch.valueHeadClasses
        let fc2Name = arch.valueHeadStyle == .wdlSoftmax ? "value_wdl_fc2" : "value_scalar_fc2"
        let fc2W = graph.variable(
            with: heInitDataFCInOut(shape: [hidden, classes]),
            shape: [NSNumber(value: hidden), NSNumber(value: classes)],
            dataType: Self.dataType,
            name: "\(fc2Name)_weights"
        )
        // Bias init: WDL -> [0, ln6, 0] (draw-heavy prior; initial softmax
        // (0.125, 0.75, 0.125), derived scalar starts at 0). scalar-tanh -> [0]
        // (tanh(0)=0). slot order [win, draw, loss] for WDL.
        let fc2BiasValues: [Float] = arch.valueHeadStyle == .wdlSoftmax
            ? [0.0, 1.791759469228055, 0.0]
            : [0.0]
        let fc2Bias = graph.variable(
            with: makeWeightData(fc2BiasValues),
            shape: [1, NSNumber(value: classes)],
            dataType: Self.dataType,
            name: "\(fc2Name)_bias"
        )
        trainables.append(fc2W)
        shouldDecay.append(true)
        trainables.append(fc2Bias)
        shouldDecay.append(false)
        x = graph.matrixMultiplication(primary: x, secondary: fc2W, name: "value_fc2")
        let logits = graph.addition(x, fc2Bias, name: "value_fc2_bias_add")

        switch arch.valueHeadStyle {
        case .wdlSoftmax:
            // Derived scalar v = p_win - p_loss (no tanh): softmax . [+1, 0, -1].
            let probs = graph.softMax(with: logits, axis: 1, name: "value_probs")
            let scalarWeights = graph.constant(
                makeWeightData([1.0, 0.0, -1.0]), shape: [1, 3], dataType: Self.dataType)
            let scalarWeighted = graph.multiplication(probs, scalarWeights, name: "value_scalar_weighted")
            // reductionSum(axis:1) keeps the reduced dim -> [batch, 1].
            let scalar = graph.reductionSum(with: scalarWeighted, axis: 1, name: "value_scalar")
            return (scalar: scalar, logits: logits, probs: probs)
        case .scalarTanh:
            // scalar = tanh(raw logit) in [-1, 1]; trained with MSE vs z (Phase D).
            // `probs` mirrors `scalar` so the tuple shape is uniform; W/D/L
            // diagnostics only apply to wdl nets.
            let scalar = graph.tanh(with: logits, name: "value_scalar")
            return (scalar: scalar, logits: logits, probs: scalar)
        }
    }

    // MARK: - Data Helpers

    /// He initialization: random normal with std = sqrt(2 / fanIn).
    ///
    /// `fanIn` must be supplied by the caller because it depends on the
    /// weight layout, which differs per layer kind. For OIHW conv weights
    /// `[outC, inC, kH, kW]`, fan_in = inC*kH*kW. For FC weights stored as
    /// `[in, out]` (the layout this codebase uses with
    /// `matrixMultiplication(primary: x, secondary: W)`), fan_in = in,
    /// i.e. the first dimension — the opposite of the conv case. A
    /// previous implementation that always used `shape.dropFirst()` was
    /// silently 8× too generous for the then-scalar `value_fc2` ([64, 1])
    /// and 5.7× too stingy for the prior FC policy head's `policy_fc`
    /// ([128, 4096]) — both shapes are historical (the value head is now
    /// W/D/L and the FC policy head became a 1×1 conv), but the fan_in fix
    /// remains correct for the value head's current FC layers.
    ///
    /// Implementation note: this used to be a per-element scalar Box-Muller
    /// loop. With ~2.9M weights to initialize, that dominated build time. The
    /// vectorized version below uses Accelerate (vDSP/vForce) on bulk arrays
    /// of uniform random Floats, which is roughly an order of magnitude
    /// faster on Apple silicon.
    static func heInitData(shape: [Int], fanIn: Int) -> Data {
        precondition(fanIn > 0, "He init: fanIn must be > 0 (got \(fanIn))")
        let std = sqrt(2.0 / Float(fanIn))
        let count = shape.reduce(1, *)
        let values = heInitFloats(count: count, std: std)
        return makeWeightData(values)
    }

    /// He init for an OIHW conv weight tensor `[outC, inC, kH, kW]`.
    /// Computes fan_in as inC * kH * kW.
    static func heInitDataConvOIHW(shape: [Int]) -> Data {
        precondition(shape.count == 4, "Conv OIHW shape must be 4D (got \(shape))")
        let fanIn = shape[1] * shape[2] * shape[3]
        return heInitData(shape: shape, fanIn: fanIn)
    }

    /// He init for an FC weight tensor stored as `[in, out]` to match
    /// `matrixMultiplication(primary: x, secondary: W)` where x has
    /// shape `[batch, in]`. Computes fan_in as the first dimension.
    static func heInitDataFCInOut(shape: [Int]) -> Data {
        precondition(shape.count == 2, "FC [in, out] shape must be 2D (got \(shape))")
        return heInitData(shape: shape, fanIn: shape[0])
    }

    /// Glorot (Xavier) normal init for an FC weight stored as `[in, out]`:
    /// random normal with `std = sqrt(2 / (fan_in + fan_out))`. Used for
    /// the SE FC2 weight, whose output feeds the sigmoid gate — Glorot
    /// targets the activation variance a symmetric saturating nonlinearity
    /// wants, unlike He (which compensates for ReLU's half-rectification).
    static func glorotInitDataFCInOut(shape: [Int]) -> Data {
        precondition(shape.count == 2, "FC [in, out] shape must be 2D (got \(shape))")
        let std = sqrt(2.0 / Float(shape[0] + shape[1]))
        let count = shape.reduce(1, *)
        let values = heInitFloats(count: count, std: std)
        return makeWeightData(values)
    }

    /// Vectorized He initialization producing `count` random normals with
    /// standard deviation `std`. Box-Muller form:
    ///   z = std * sqrt(-2 * ln(u1)) * cos(2π * u2)
    /// where u1, u2 are independent uniforms.
    private static func heInitFloats(count: Int, std: Float) -> [Float] {
        // Box-Muller produces values in pairs; we generate `count` outputs
        // using `count` u1 + `count` u2 (one z per pair of uniforms).
        var u1 = [Float](repeating: 0, count: count)
        var u2 = [Float](repeating: 0, count: count)

        // Bulk-fill with uniform Floats in [0, 1). arc4random_buf gives us
        // uniformly distributed UInt32s; divide by 2^32 for [0, 1).
        u1.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress.map { fillUniform01(buf: $0, count: count) }
        }
        u2.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress.map { fillUniform01(buf: $0, count: count) }
        }

        // Clamp u1 into [leastNormalMagnitude, 1.0] so the next-step
        // `vvlogf(u1)` is finite.
        //
        // The hazard we're defending against is `arc4random_buf` returning
        // exactly 0 (a legitimate UInt32 outcome at ~1-in-4-billion), which
        // would feed `vvlogf` a +0 and produce -inf, then `sqrt(-2 * -inf)`
        // → +inf, propagating an inf weight into the network. The lo bound
        // pulls those zeros up to the smallest representable normal float
        // (~1.18e-38), giving `log(lo) ≈ -87`, then `sqrt(2*87) ≈ 13.2`,
        // scaled by std to a finite weight. The hi bound is defensive —
        // our uniform draw is in [0, 1) so it never trips, but feeding
        // something > 1 to log would give a positive value and skew
        // Box-Muller's distribution.
        var lo: Float = .leastNormalMagnitude
        var hi: Float = 1.0
        vDSP_vclip(u1, 1, &lo, &hi, &u1, 1, vDSP_Length(count))

        // u1 = ln(u1)  →  u1 = -2 * u1  →  u1 = sqrt(u1)
        var n = Int32(count)
        vvlogf(&u1, u1, &n)
        var negTwo: Float = -2
        vDSP_vsmul(u1, 1, &negTwo, &u1, 1, vDSP_Length(count))
        vvsqrtf(&u1, u1, &n)

        // u2 = cos(2π * u2)
        var twoPi: Float = 2 * .pi
        vDSP_vsmul(u2, 1, &twoPi, &u2, 1, vDSP_Length(count))
        vvcosf(&u2, u2, &n)

        // z = u1 * u2
        var z = [Float](repeating: 0, count: count)
        vDSP_vmul(u1, 1, u2, 1, &z, 1, vDSP_Length(count))

        // Scale by std
        var stdVar = std
        vDSP_vsmul(z, 1, &stdVar, &z, 1, vDSP_Length(count))
        return z
    }

    /// Fill a Float buffer with uniformly distributed values in [0, 1).
    private static func fillUniform01(buf: UnsafeMutablePointer<Float>, count: Int) {
        // Generate UInt32s straight into a temporary buffer, then convert to
        // Float. 2^-32 maps the full UInt32 range to [0, 1).
        var raw = [UInt32](repeating: 0, count: count)
        raw.withUnsafeMutableBytes { rawBytes in
            if let base = rawBytes.baseAddress {
                arc4random_buf(base, rawBytes.count)
            }
        }
        let scale: Float = Float(1.0 / 4294967296.0) // 2^-32
        for i in 0..<count {
            buf[i] = Float(raw[i]) * scale
        }
    }

    static func onesData(count: Int) -> Data {
        makeWeightData([Float](repeating: 1.0, count: count))
    }

    static func zerosData(count: Int) -> Data {
        makeWeightData([Float](repeating: 0.0, count: count))
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
    static func writeFloats(_ floats: [Float], into array: MPSNDArray) {
        precondition(
            !floats.isEmpty,
            "writeFloats: empty input would silently skip the MPSNDArray write"
        )
        let data = makeWeightData(floats)
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
    /// readback: `policyOutputReadback` is cast to fp32 on the GPU, so the
    /// host side is a plain memcpy (the bf16→fp32 widen of the wide policy
    /// output no longer runs in a host loop). The caller is responsible for
    /// `pointer` having capacity `count`.
    static func readFloatsFP32(
        from data: MPSGraphTensorData,
        into pointer: UnsafeMutablePointer<Float>,
        count: Int
    ) {
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
    static var bytesPerWeightElement: Int {
        switch dataType {
        case .float32: return MemoryLayout<Float>.size
        case .float16, .bFloat16: return MemoryLayout<UInt16>.size
        default: fatalError("Unsupported ChessNetwork.dataType: \(dataType)")
        }
    }

    /// Relative machine epsilon (the ULP of 1.0) for `dataType` —
    /// `2^-mantissaBits`: Float32 ≈ 1.19e-7, Float16 ≈ 9.77e-4, bFloat16
    /// ≈ 7.81e-3. The correct scale for a *relative* numerical tolerance:
    /// an absolute tolerance of `weightRelativeEpsilon · |x|` is one ULP at
    /// magnitude `|x|`. Lets numeric tests derive accuracy bounds from the
    /// active dtype instead of hardcoding fp32-era constants.
    static var weightRelativeEpsilon: Float {
        switch dataType {
        case .float32: return Float.ulpOfOne   // 2^-23
        case .float16: return 0x1p-10          // 2^-10
        case .bFloat16: return 0x1p-7          // 2^-7
        default: fatalError("Unsupported ChessNetwork.dataType: \(dataType)")
        }
    }

    /// Decode raw `dataType` weight bytes (as produced by `makeWeightData`)
    /// back into Float32 — the exact inverse of `makeWeightData`. Element
    /// count is inferred as `data.count / bytesPerWeightElement`.
    static func decodeWeightData(_ data: Data) -> [Float] {
        let count = data.count / bytesPerWeightElement
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
            fatalError("Unsupported ChessNetwork.dataType: \(dataType)")
        }
    }

    /// Convert a Float32 array into bytes laid out in `Self.dataType`.
    /// Float32 → passthrough; Float16 → conversion via vImage; bFloat16 →
    /// bit-shift narrowing.
    static func makeWeightData(_ floats: [Float]) -> Data {
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
            fatalError("Unsupported ChessNetwork.dataType: \(dataType)")
        }
    }

    /// Read inference output as Float32, converting from `Self.dataType`.
    static func readFloats(from data: MPSGraphTensorData, count: Int) -> [Float] {
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
            fatalError("Unsupported ChessNetwork.dataType: \(dataType)")
        }
    }

    /// Read inference output into a caller-owned float buffer. Used by
    /// the hot inference and training paths so the readback doesn't
    /// allocate a fresh Swift array on every call. The `count` argument
    /// must match the underlying tensor's element count (it's validated
    /// only in debug via the MPSNDArray shape, not here).
    ///
    /// On `.float16` this would need a reused `[UInt16]` scratch; not
    /// yet implemented because `dataType` is currently `.float32`. The
    /// fatal matches the pattern used by `writeInferenceInput` so a
    /// future `.float16` flip fails loudly rather than silently.
    static func readFloats(
        from data: MPSGraphTensorData,
        into pointer: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        switch dataType {
        case .float32:
            data.mpsndarray().readBytes(
                UnsafeMutableRawPointer(pointer),
                strideBytes: nil
            )
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
            fatalError("readFloats(from:into:count:): unsupported dataType \(dataType). "
                + "Implement a reused half-scratch buffer before flipping to .float16.")
        }
    }
}
