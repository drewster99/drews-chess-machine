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
        let persistent = trainables + runningStats
        loadPlaceholders.reserveCapacity(persistent.count)
        loadAssignOps.reserveCapacity(persistent.count)
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
        }
        weightLoadPlaceholders = loadPlaceholders
        weightLoadAssignOps = loadAssignOps
    }

    // MARK: - Inference

    /// Evaluate a single board position.
    /// - Parameter board: 18x8x8 = 1,152 floats in NCHW order (planes, rows, cols)
    /// - Returns: Policy probabilities (4096 move slots) and position value in [-1, +1]
    func evaluate(board: [Float]) throws -> (policy: [Float], value: Float) {
        let inputBytes = Self.makeWeightData(board)
        let inputData = MPSGraphTensorData(
            device: graphDevice,
            data: inputBytes,
            shape: [1, 18, 8, 8],
            dataType: Self.dataType
        )

        let results = graph.run(
            with: commandQueue,
            feeds: [inputPlaceholder: inputData],
            targetTensors: [policyOutput, valueOutput],
            targetOperations: nil
        )

        guard let policyData = results[policyOutput] else {
            throw ChessNetworkError.outputMissing("policy")
        }
        guard let valueData = results[valueOutput] else {
            throw ChessNetworkError.outputMissing("value")
        }

        let policy = Self.readFloats(from: policyData, count: Self.policySize)
        let valueBuf = Self.readFloats(from: valueData, count: 1)
        return (policy, valueBuf[0])
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
        // board_input placeholder with a zero-filled batch-of-1 dummy
        // (and nothing for the weight-load placeholders, which are safe
        // to omit because no run-time target reaches them). targetTensors
        // are the variables themselves — reading them doesn't require
        // any compute ancestor, so no forward pass actually runs.
        let dummyBoard = [Float](repeating: 0, count: 1 * 18 * 8 * 8)
        let dummyInput = MPSGraphTensorData(
            device: graphDevice,
            data: Self.makeWeightData(dummyBoard),
            shape: [1, 18, 8, 8],
            dataType: Self.dataType
        )
        let results = graph.run(
            with: commandQueue,
            feeds: [inputPlaceholder: dummyInput],
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
        let dummyBoard = [Float](repeating: 0, count: 1 * 18 * 8 * 8)
        feeds[inputPlaceholder] = MPSGraphTensorData(
            device: graphDevice,
            data: Self.makeWeightData(dummyBoard),
            shape: [1, 18, 8, 8],
            dataType: Self.dataType
        )

        for (i, v) in allVars.enumerated() {
            let expectedCount = try Self.elementCount(of: v)
            guard weights[i].count == expectedCount else {
                throw ChessNetworkError.weightLoadMismatch(
                    "variable \(v.operation.name): expected \(expectedCount) floats, got \(weights[i].count)"
                )
            }
            let ph = weightLoadPlaceholders[i]
            guard let phShape = ph.shape else {
                throw ChessNetworkError.variableShapeMissing(ph.operation.name)
            }
            feeds[ph] = MPSGraphTensorData(
                device: graphDevice,
                data: Self.makeWeightData(weights[i]),
                shape: phShape,
                dataType: Self.dataType
            )
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
}
