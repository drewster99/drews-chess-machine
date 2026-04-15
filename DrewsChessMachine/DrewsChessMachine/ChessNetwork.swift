import Accelerate
import Foundation
import Metal
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
/// Architecture (from chess-engine-design.md):
/// - Input: 18x8x8 board tensor (NCHW layout)
/// - Stem: 3x3 conv (18 -> 128 channels), batch norm, ReLU
/// - Tower: 8 residual blocks (each: conv -> BN -> ReLU -> conv -> BN -> skip add -> ReLU)
/// - Policy head: 1x1 conv (128 -> 2) -> BN -> ReLU -> flatten -> FC(128 -> 4096) (logits)
/// - Value head: 1x1 conv (128 -> 1) -> BN -> ReLU -> flatten -> FC(64 -> 64) -> ReLU -> FC(64 -> 1) -> tanh
///
/// Total parameters: ~2,917,383 (~2.9M)
final class ChessNetwork {

    // MARK: Configuration

    /// Numeric precision for all weights and activations.
    ///
    /// Switching this between `.float32` and `.float16` should Just Work —
    /// the data helpers (`heInitData`, `onesData`, `zerosData`) and the
    /// inference output reader (`readFloats(from:count:)`) both consult
    /// `dataType` and convert via Accelerate as needed. The graph's
    /// `placeholder` and `variable` calls all pass `Self.dataType`, and
    /// `evaluate(board:)` writes the input tensor in the configured
    /// precision before passing it to the graph.
    static let dataType: MPSDataType = .float32

    static let channels = 128
    static let inputPlanes = 18
    static let boardSize = 8
    static let numBlocks = 8
    static let policySize = 4096

    // MARK: Graph Tensors

    let graph: MPSGraph
    let inputPlaceholder: MPSGraphTensor
    let policyOutput: MPSGraphTensor
    let valueOutput: MPSGraphTensor

    /// All graph variables that should receive gradient updates during
    /// training: every conv weight, FC weight, FC bias, and BN gamma/beta.
    /// Excludes BN running mean/variance — those are EMA-updated (not
    /// gradient-updated) in training mode and loaded directly in
    /// inference mode. See `bnRunningStatsVariables`.
    private(set) var trainableVariables: [MPSGraphTensor] = []

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

    /// Pre-allocated `[1, 18, 8, 8]` input feed reused on every
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
    /// MPSGraph to write the 4096-element policy output directly into
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

    /// Zero-filled `[1, 18, 8, 8]` feed shared by `exportWeights()` and
    /// `loadWeights(_:)` to satisfy MPSGraph's requirement that every
    /// graph placeholder be fed even when the target ops don't consume
    /// it. Filled once at init, never modified afterwards.
    private let dummyInferenceInputTensorData: MPSGraphTensorData

    // MARK: Metal

    let metalDevice: MTLDevice
    let commandQueue: MTLCommandQueue
    let graphDevice: MPSGraphDevice

    // MARK: Initialization

    /// Build the network. Default `bnMode = .inference` keeps the existing
    /// behavior for play / forward-pass demos; pass `.training` to build a
    /// copy whose BN layers compute fresh batch stats on every forward pass
    /// (used by ChessTrainer for accurate training-step benchmarks).
    init(bnMode: BNMode = .inference) throws {
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            throw ChessNetworkError.metalNotSupported
        }
        guard let cmdQueue = mtlDevice.makeCommandQueue() else {
            throw ChessNetworkError.commandQueueCreationFailed
        }

        metalDevice = mtlDevice
        commandQueue = cmdQueue
        graphDevice = MPSGraphDevice(mtlDevice: mtlDevice)
        let g = MPSGraph()
        graph = g

        let conv3x3 = try Self.makeConv3x3Descriptor()
        let conv1x1 = try Self.makeConv1x1Descriptor()

        // Input: [batch, 18, 8, 8]
        let input = g.placeholder(
            shape: [-1, 18, 8, 8],
            dataType: Self.dataType,
            name: "board_input"
        )
        inputPlaceholder = input

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
        var runningStats: [MPSGraphTensor] = []
        var runningStatsAssigns: [MPSGraphOperation] = []

        // --- Stem: 3x3 conv (18 -> 128) -> BN -> ReLU ---

        let stemWeights = g.variable(
            with: Self.heInitData(shape: [128, 18, 3, 3]),
            shape: [128, 18, 3, 3],
            dataType: Self.dataType,
            name: "stem_conv_weights"
        )
        trainables.append(stemWeights)
        var x = g.convolution2D(
            input,
            weights: stemWeights,
            descriptor: conv3x3,
            name: "stem_conv"
        )
        x = Self.batchNorm(
            graph: g, input: x, channels: 128, name: "stem_bn", bnMode: bnMode,
            trainables: &trainables,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssigns
        )
        x = g.reLU(with: x, name: "stem_relu")

        // --- Tower: 8 residual blocks ---

        for i in 0..<8 {
            x = Self.residualBlock(
                graph: g, input: x, descriptor: conv3x3, blockIndex: i, bnMode: bnMode,
                trainables: &trainables,
                runningStats: &runningStats,
                runningStatsAssignOps: &runningStatsAssigns
            )
        }

        // --- Policy head ---

        policyOutput = Self.policyHead(
            graph: g, input: x, descriptor: conv1x1, bnMode: bnMode,
            trainables: &trainables,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssigns
        )

        // --- Value head ---

        valueOutput = Self.valueHead(
            graph: g, input: x, descriptor: conv1x1, bnMode: bnMode,
            trainables: &trainables,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssigns
        )

        trainableVariables = trainables
        bnRunningStatsVariables = runningStats
        bnRunningStatsAssignOps = runningStatsAssigns

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

        // Reusable `[1, 18, 8, 8]` inference input ND array + wrapper.
        // `evaluate(board:)` writes new floats directly into this
        // array's storage each call and feeds the same wrapper — no
        // per-move MPS allocations.
        let inputDesc = MPSNDArrayDescriptor(
            dataType: Self.dataType,
            shape: [1, 18, 8, 8]
        )
        let inputND = MPSNDArray(device: mtlDevice, descriptor: inputDesc)
        inferenceInputNDArray = inputND
        inferenceInputTensorData = MPSGraphTensorData(inputND)

        // Zero-filled dummy input shared by exportWeights / loadWeights.
        let dummyND = MPSNDArray(device: mtlDevice, descriptor: inputDesc)
        Self.writeFloats(
            [Float](repeating: 0, count: 1 * 18 * 8 * 8),
            into: dummyND
        )
        dummyInferenceInputTensorData = MPSGraphTensorData(dummyND)

        // Cache the feeds dict and target tensor list so the per-move
        // inference path doesn't rebuild them. Both are immutable — the
        // ND array backing `inferenceInputTensorData` is written
        // through `writeBytes` on the same underlying storage every
        // call.
        inferenceFeeds = [inputPlaceholder: inferenceInputTensorData]
        inferenceTargets = [policyOutput, valueOutput]

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
    }

    deinit {
        inferencePolicyScratchPtr.deinitialize(count: Self.policySize)
        inferencePolicyScratchPtr.deallocate()
        inferenceValueScratchPtr.deinitialize(count: 1)
        inferenceValueScratchPtr.deallocate()
    }

    // MARK: - Inference

    /// Evaluate a single board position.
    ///
    /// **Not re-entrant.** The returned `policy` buffer aliases this
    /// network's shared readback scratch — it is valid only until the
    /// next `evaluate` call on this same network. Callers must consume
    /// the policy vector before issuing another `evaluate`. The `value`
    /// scalar is returned by copy and is not subject to this constraint.
    ///
    /// In self-play both `MPSChessPlayer` instances share one
    /// `ChessNetwork` but are driven sequentially inside a single
    /// `ChessMachine.runGameLoop`, so only one side evaluates at a
    /// time. Any future refactor that runs two games concurrently on
    /// one network must give each game its own `ChessNetwork` or add
    /// explicit serialization here.
    ///
    /// - Parameter board: 18×8×8 = 1,152 floats in NCHW order (planes, rows, cols).
    /// - Returns: `UnsafeBufferPointer` over 4,096 policy logits plus the scalar value in [-1, +1].
    func evaluate(
        board: UnsafeBufferPointer<Float>
    ) throws -> (policy: UnsafeBufferPointer<Float>, value: Float) {
        let expected = 1 * Self.inputPlanes * Self.boardSize * Self.boardSize
        guard board.count == expected else {
            throw ChessNetworkError.boardSizeMismatch(expected: expected, got: board.count)
        }

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

        Self.readFloats(from: policyData, into: inferencePolicyScratchPtr, count: Self.policySize)
        Self.readFloats(from: valueData, into: inferenceValueScratchPtr, count: 1)

        return (
            policy: UnsafeBufferPointer(start: inferencePolicyScratchPtr, count: Self.policySize),
            value: inferenceValueScratchPtr.pointee
        )
    }

    /// `[Float]`-input overload for callers outside the hot path (tests,
    /// the Forward Pass demo). Delegates to the pointer-based primary
    /// entry point via `withUnsafeBufferPointer` — no copy.
    func evaluate(
        board: [Float]
    ) throws -> (policy: UnsafeBufferPointer<Float>, value: Float) {
        try board.withUnsafeBufferPointer { buf in
            try evaluate(board: buf)
        }
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
    func exportWeights() throws -> [[Float]] {
        let allVars = trainableVariables + bnRunningStatsVariables

        // MPSGraph requires feeds for every placeholder in the graph,
        // even ones unreachable from the target tensors. We feed the
        // board_input placeholder with a pre-built zero-filled dummy
        // (and nothing for the weight-load placeholders, which are safe
        // to omit because no run-time target reaches them). targetTensors
        // are the variables themselves — reading them doesn't require
        // any compute ancestor, so no forward pass actually runs.
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
    func loadWeights(_ weights: [[Float]]) throws {
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
        _ = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: [allVars[0]],
            targetOperations: weightLoadAssignOps
        )
    }

    /// Total scalar count in a tensor's statically-known shape.
    /// Throws if the tensor's shape is missing — which shouldn't happen
    /// for variables (they have concrete shapes at creation time).
    private static func elementCount(of tensor: MPSGraphTensor) throws -> Int {
        guard let shape = tensor.shape else {
            throw ChessNetworkError.variableShapeMissing(tensor.operation.name)
        }
        return shape.reduce(1) { $0 * $1.intValue }
    }

    // MARK: - Convolution Descriptors

    /// 3x3 convolution with padding=1 (preserves 8x8 spatial dimensions).
    private static func makeConv3x3Descriptor() throws -> MPSGraphConvolution2DOpDescriptor {
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
        desc.paddingLeft = 1
        desc.paddingRight = 1
        desc.paddingTop = 1
        desc.paddingBottom = 1
        return desc
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
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation]
    ) -> MPSGraphTensor {
        let ch = NSNumber(value: channels)

        // gamma and beta are trainable in both modes.
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
        trainables.append(beta)

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

    /// One residual block: conv1 -> BN -> ReLU -> conv2 -> BN -> skip add -> ReLU
    private static func residualBlock(
        graph: MPSGraph,
        input: MPSGraphTensor,
        descriptor: MPSGraphConvolution2DOpDescriptor,
        blockIndex: Int,
        bnMode: BNMode,
        trainables: inout [MPSGraphTensor],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation]
    ) -> MPSGraphTensor {
        let prefix = "block\(blockIndex)"

        // First path: conv -> BN -> ReLU
        let conv1W = graph.variable(
            with: heInitData(shape: [128, 128, 3, 3]),
            shape: [128, 128, 3, 3],
            dataType: Self.dataType,
            name: "\(prefix)_conv1_weights"
        )
        trainables.append(conv1W)
        var x = graph.convolution2D(
            input, weights: conv1W, descriptor: descriptor, name: "\(prefix)_conv1"
        )
        x = batchNorm(
            graph: graph, input: x, channels: 128, name: "\(prefix)_bn1", bnMode: bnMode,
            trainables: &trainables,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssignOps
        )
        x = graph.reLU(with: x, name: "\(prefix)_relu1")

        // Second path: conv -> BN (no ReLU yet — applied after skip add)
        let conv2W = graph.variable(
            with: heInitData(shape: [128, 128, 3, 3]),
            shape: [128, 128, 3, 3],
            dataType: Self.dataType,
            name: "\(prefix)_conv2_weights"
        )
        trainables.append(conv2W)
        x = graph.convolution2D(
            x, weights: conv2W, descriptor: descriptor, name: "\(prefix)_conv2"
        )
        x = batchNorm(
            graph: graph, input: x, channels: 128, name: "\(prefix)_bn2", bnMode: bnMode,
            trainables: &trainables,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssignOps
        )

        // Skip connection: add original input, then ReLU
        x = graph.addition(input, x, name: "\(prefix)_skip")
        x = graph.reLU(with: x, name: "\(prefix)_relu2")

        return x
    }

    /// Policy head: 1x1 conv (128 -> 2) -> BN -> ReLU -> flatten -> FC (128 -> 4096) (logits)
    ///
    /// Note: this used to end with a 4096-way softmax. We dropped it because
    /// the CPU has to mask illegal moves anyway, and computing softmax over
    /// only the ~30 legal moves is far cheaper than softmax over 4096 slots
    /// followed by mask + renormalize. The CPU side (MPSChessPlayer.sampleMove)
    /// now does softmax-over-legal-moves directly from these logits.
    private static func policyHead(
        graph: MPSGraph,
        input: MPSGraphTensor,
        descriptor: MPSGraphConvolution2DOpDescriptor,
        bnMode: BNMode,
        trainables: inout [MPSGraphTensor],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation]
    ) -> MPSGraphTensor {
        // 1x1 conv: compress 128 channels to 2
        let convW = graph.variable(
            with: heInitData(shape: [2, 128, 1, 1]),
            shape: [2, 128, 1, 1],
            dataType: Self.dataType,
            name: "policy_conv_weights"
        )
        trainables.append(convW)
        var x = graph.convolution2D(
            input, weights: convW, descriptor: descriptor, name: "policy_conv"
        )
        x = batchNorm(
            graph: graph, input: x, channels: 2, name: "policy_bn", bnMode: bnMode,
            trainables: &trainables,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssignOps
        )
        x = graph.reLU(with: x, name: "policy_relu")

        // Flatten: [batch, 2, 8, 8] -> [batch, 128]
        x = graph.reshape(x, shape: [-1, 128], name: "policy_flatten")

        // FC: 128 -> 4096
        let fcW = graph.variable(
            with: heInitData(shape: [128, 4096]),
            shape: [128, 4096],
            dataType: Self.dataType,
            name: "policy_fc_weights"
        )
        let fcBias = graph.variable(
            with: zerosData(count: 4096),
            shape: [1, 4096],
            dataType: Self.dataType,
            name: "policy_fc_bias"
        )
        trainables.append(fcW)
        trainables.append(fcBias)
        x = graph.matrixMultiplication(primary: x, secondary: fcW, name: "policy_fc")
        x = graph.addition(x, fcBias, name: "policy_fc_bias_add")

        // Logits, not probabilities — softmax happens on the CPU over only
        // the legal moves (see MPSChessPlayer.sampleMove).
        return x
    }

    /// Value head: 1x1 conv (128 -> 1) -> BN -> ReLU -> flatten -> FC(64 -> 64) -> ReLU -> FC(64 -> 1) -> tanh
    private static func valueHead(
        graph: MPSGraph,
        input: MPSGraphTensor,
        descriptor: MPSGraphConvolution2DOpDescriptor,
        bnMode: BNMode,
        trainables: inout [MPSGraphTensor],
        runningStats: inout [MPSGraphTensor],
        runningStatsAssignOps: inout [MPSGraphOperation]
    ) -> MPSGraphTensor {
        // 1x1 conv: compress 128 channels to 1
        let convW = graph.variable(
            with: heInitData(shape: [1, 128, 1, 1]),
            shape: [1, 128, 1, 1],
            dataType: Self.dataType,
            name: "value_conv_weights"
        )
        trainables.append(convW)
        var x = graph.convolution2D(
            input, weights: convW, descriptor: descriptor, name: "value_conv"
        )
        x = batchNorm(
            graph: graph, input: x, channels: 1, name: "value_bn", bnMode: bnMode,
            trainables: &trainables,
            runningStats: &runningStats,
            runningStatsAssignOps: &runningStatsAssignOps
        )
        x = graph.reLU(with: x, name: "value_relu")

        // Flatten: [batch, 1, 8, 8] -> [batch, 64]
        x = graph.reshape(x, shape: [-1, 64], name: "value_flatten")

        // FC1: 64 -> 64
        let fc1W = graph.variable(
            with: heInitData(shape: [64, 64]),
            shape: [64, 64],
            dataType: Self.dataType,
            name: "value_fc1_weights"
        )
        let fc1Bias = graph.variable(
            with: zerosData(count: 64),
            shape: [1, 64],
            dataType: Self.dataType,
            name: "value_fc1_bias"
        )
        trainables.append(fc1W)
        trainables.append(fc1Bias)
        x = graph.matrixMultiplication(primary: x, secondary: fc1W, name: "value_fc1")
        x = graph.addition(x, fc1Bias, name: "value_fc1_bias_add")
        x = graph.reLU(with: x, name: "value_fc1_relu")

        // FC2: 64 -> 1
        let fc2W = graph.variable(
            with: heInitData(shape: [64, 1]),
            shape: [64, 1],
            dataType: Self.dataType,
            name: "value_fc2_weights"
        )
        let fc2Bias = graph.variable(
            with: zerosData(count: 1),
            shape: [1, 1],
            dataType: Self.dataType,
            name: "value_fc2_bias"
        )
        trainables.append(fc2W)
        trainables.append(fc2Bias)
        x = graph.matrixMultiplication(primary: x, secondary: fc2W, name: "value_fc2")
        x = graph.addition(x, fc2Bias, name: "value_fc2_bias_add")

        // Tanh: squash to [-1.0, +1.0]
        x = graph.tanh(with: x, name: "value_tanh")

        return x
    }

    // MARK: - Data Helpers

    /// He initialization: random normal with std = sqrt(2 / fan_in).
    /// Fan-in is the product of all dimensions except the first (output channels).
    ///
    /// Implementation note: this used to be a per-element scalar Box-Muller
    /// loop. With ~2.9M weights to initialize, that dominated build time. The
    /// vectorized version below uses Accelerate (vDSP/vForce) on bulk arrays
    /// of uniform random Floats, which is roughly an order of magnitude
    /// faster on Apple silicon.
    static func heInitData(shape: [Int]) -> Data {
        let fanIn = shape.dropFirst().reduce(1, *)
        let std = sqrt(2.0 / Float(fanIn))
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

        // Clamp u1 away from 0 so log(u1) is finite.
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

    /// Write raw float bytes from `buffer` directly into `array`'s
    /// storage. Primary inference-hot-path writer: the caller passes a
    /// pre-encoded `UnsafeBufferPointer<Float>` (e.g. a slice of a
    /// per-game scratch) and the bytes flow straight into the MPSNDArray
    /// with zero intermediate copies. On `.float32` the data is handed
    /// unchanged to `writeBytes`; `.float16` would need a reused
    /// `[UInt16]` scratch buffer (not yet implemented because dataType
    /// is currently `.float32`).
    static func writeInferenceInput(
        _ buffer: UnsafeBufferPointer<Float>,
        into array: MPSNDArray
    ) {
        switch dataType {
        case .float32:
            guard let base = buffer.baseAddress else { return }
            array.writeBytes(
                UnsafeMutableRawPointer(mutating: base),
                strideBytes: nil
            )
        default:
            fatalError("writeInferenceInput: unsupported dataType \(dataType). "
                + "Implement a reused half-scratch buffer before flipping to .float16.")
        }
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
        let data = makeWeightData(floats)
        data.withUnsafeBytes { buf in
            guard let base = buf.baseAddress else { return }
            array.writeBytes(
                UnsafeMutableRawPointer(mutating: base),
                strideBytes: nil
            )
        }
    }

    /// Convert a Float32 array into bytes laid out in `Self.dataType`.
    /// Float32 → passthrough; Float16 → conversion via vImage.
    static func makeWeightData(_ floats: [Float]) -> Data {
        switch dataType {
        case .float32:
            return floats.withUnsafeBytes { Data($0) }

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
        default:
            fatalError("readFloats(from:into:count:): unsupported dataType \(dataType). "
                + "Implement a reused half-scratch buffer before flipping to .float16.")
        }
    }
}
