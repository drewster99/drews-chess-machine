import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - Errors

enum ChessNetworkError: LocalizedError {
    case metalNotSupported
    case commandQueueCreationFailed
    case descriptorCreationFailed
    case outputMissing(String)

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
        }
    }
}

// MARK: - Chess Neural Network

/// Chess engine neural network forward pass implemented with MPSGraph.
///
/// Architecture (from chess-engine-design.md):
/// - Input: 18x8x8 board tensor (NCHW layout)
/// - Stem: 3x3 conv (18 -> 128 channels), batch norm, ReLU
/// - Tower: 8 residual blocks (each: conv -> BN -> ReLU -> conv -> BN -> skip add -> ReLU)
/// - Policy head: 1x1 conv (128 -> 2) -> BN -> ReLU -> flatten -> FC(128 -> 4096) -> softmax
/// - Value head: 1x1 conv (128 -> 1) -> BN -> ReLU -> flatten -> FC(64 -> 64) -> ReLU -> FC(64 -> 1) -> tanh
///
/// Total parameters: ~2,917,383 (~2.9M)
final class ChessNetwork {

    // MARK: Configuration

    /// Numeric precision for all weights and activations.
    /// Data helpers (heInitData, onesData, zerosData) produce Float bytes to match.
    static let dataType: MPSDataType = .float16

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

    // MARK: Metal

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let graphDevice: MPSGraphDevice

    // MARK: Initialization

    init() throws {
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            throw ChessNetworkError.metalNotSupported
        }
        guard let cmdQueue = mtlDevice.makeCommandQueue() else {
            throw ChessNetworkError.commandQueueCreationFailed
        }

        device = mtlDevice
        commandQueue = cmdQueue
        graphDevice = MPSGraphDevice(mtlDevice: mtlDevice)
        graph = MPSGraph()

        let conv3x3 = try Self.makeConv3x3Descriptor()
        let conv1x1 = try Self.makeConv1x1Descriptor()

        // Input: [batch, 18, 8, 8]
        inputPlaceholder = graph.placeholder(
            shape: [-1, 18, 8, 8],
            dataType: Self.dataType,
            name: "board_input"
        )

        // --- Stem: 3x3 conv (18 -> 128) -> BN -> ReLU ---

        let stemWeights = graph.variable(
            with: Self.heInitData(shape: [128, 18, 3, 3]),
            shape: [128, 18, 3, 3],
            dataType: Self.dataType,
            name: "stem_conv_weights"
        )
        var x = graph.convolution2D(
            inputPlaceholder,
            weights: stemWeights,
            descriptor: conv3x3,
            name: "stem_conv"
        )
        x = Self.batchNorm(graph: graph, input: x, channels: 128, name: "stem_bn")
        x = graph.reLU(with: x, name: "stem_relu")

        // --- Tower: 8 residual blocks ---

        for i in 0..<8 {
            x = Self.residualBlock(graph: graph, input: x, descriptor: conv3x3, blockIndex: i)
        }

        // --- Policy head ---

        policyOutput = Self.policyHead(graph: graph, input: x, descriptor: conv1x1)

        // --- Value head ---

        valueOutput = Self.valueHead(graph: graph, input: x, descriptor: conv1x1)
    }

    // MARK: - Inference

    /// Evaluate a single board position.
    /// - Parameter board: 18x8x8 = 1,152 floats in NCHW order (planes, rows, cols)
    /// - Returns: Policy probabilities (4096 move slots) and position value in [-1, +1]
    func evaluate(board: [Float]) throws -> (policy: [Float], value: Float) {
        let boardBytes = board.withUnsafeBytes { Data($0) }
        let inputData = MPSGraphTensorData(
            device: graphDevice,
            data: boardBytes,
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

        var policy = [Float](repeating: 0, count: Self.policySize)
        policy.withUnsafeMutableBytes { buffer in
            guard let ptr = buffer.baseAddress else { return }
            policyData.mpsndarray().readBytes(ptr, strideBytes: nil)
        }

        var value: Float = 0
        valueData.mpsndarray().readBytes(&value, strideBytes: nil)

        return (policy, value)
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

    /// Inference-time batch normalization using stored running statistics.
    ///
    /// Running mean initialized to 0, running variance initialized to 1. With gamma=1 and
    /// beta=0 this is an identity transform — correct for an untrained network. During
    /// training these would be updated via exponential moving average; at inference they're
    /// frozen constants.
    private static func batchNorm(
        graph: MPSGraph,
        input: MPSGraphTensor,
        channels: Int,
        name: String
    ) -> MPSGraphTensor {
        let ch = NSNumber(value: channels)

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

        return graph.normalize(
            input,
            mean: runningMean,
            variance: runningVar,
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
        blockIndex: Int
    ) -> MPSGraphTensor {
        let prefix = "block\(blockIndex)"

        // First path: conv -> BN -> ReLU
        let conv1W = graph.variable(
            with: heInitData(shape: [128, 128, 3, 3]),
            shape: [128, 128, 3, 3],
            dataType: Self.dataType,
            name: "\(prefix)_conv1_weights"
        )
        var x = graph.convolution2D(
            input, weights: conv1W, descriptor: descriptor, name: "\(prefix)_conv1"
        )
        x = batchNorm(graph: graph, input: x, channels: 128, name: "\(prefix)_bn1")
        x = graph.reLU(with: x, name: "\(prefix)_relu1")

        // Second path: conv -> BN (no ReLU yet — applied after skip add)
        let conv2W = graph.variable(
            with: heInitData(shape: [128, 128, 3, 3]),
            shape: [128, 128, 3, 3],
            dataType: Self.dataType,
            name: "\(prefix)_conv2_weights"
        )
        x = graph.convolution2D(
            x, weights: conv2W, descriptor: descriptor, name: "\(prefix)_conv2"
        )
        x = batchNorm(graph: graph, input: x, channels: 128, name: "\(prefix)_bn2")

        // Skip connection: add original input, then ReLU
        x = graph.addition(input, x, name: "\(prefix)_skip")
        x = graph.reLU(with: x, name: "\(prefix)_relu2")

        return x
    }

    /// Policy head: 1x1 conv (128 -> 2) -> BN -> ReLU -> flatten -> FC (128 -> 4096) -> softmax
    private static func policyHead(
        graph: MPSGraph,
        input: MPSGraphTensor,
        descriptor: MPSGraphConvolution2DOpDescriptor
    ) -> MPSGraphTensor {
        // 1x1 conv: compress 128 channels to 2
        let convW = graph.variable(
            with: heInitData(shape: [2, 128, 1, 1]),
            shape: [2, 128, 1, 1],
            dataType: Self.dataType,
            name: "policy_conv_weights"
        )
        var x = graph.convolution2D(
            input, weights: convW, descriptor: descriptor, name: "policy_conv"
        )
        x = batchNorm(graph: graph, input: x, channels: 2, name: "policy_bn")
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
        x = graph.matrixMultiplication(primary: x, secondary: fcW, name: "policy_fc")
        x = graph.addition(x, fcBias, name: "policy_fc_bias_add")

        // Softmax: convert logits to probabilities summing to 1.0
        x = graph.softMax(with: x, axis: 1, name: "policy_softmax")

        return x
    }

    /// Value head: 1x1 conv (128 -> 1) -> BN -> ReLU -> flatten -> FC(64 -> 64) -> ReLU -> FC(64 -> 1) -> tanh
    private static func valueHead(
        graph: MPSGraph,
        input: MPSGraphTensor,
        descriptor: MPSGraphConvolution2DOpDescriptor
    ) -> MPSGraphTensor {
        // 1x1 conv: compress 128 channels to 1
        let convW = graph.variable(
            with: heInitData(shape: [1, 128, 1, 1]),
            shape: [1, 128, 1, 1],
            dataType: Self.dataType,
            name: "value_conv_weights"
        )
        var x = graph.convolution2D(
            input, weights: convW, descriptor: descriptor, name: "value_conv"
        )
        x = batchNorm(graph: graph, input: x, channels: 1, name: "value_bn")
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
        x = graph.matrixMultiplication(primary: x, secondary: fc2W, name: "value_fc2")
        x = graph.addition(x, fc2Bias, name: "value_fc2_bias_add")

        // Tanh: squash to [-1.0, +1.0]
        x = graph.tanh(with: x, name: "value_tanh")

        return x
    }

    // MARK: - Data Helpers

    /// He initialization: random normal with std = sqrt(2 / fan_in).
    /// Fan-in is the product of all dimensions except the first (output channels).
    static func heInitData(shape: [Int]) -> Data {
        let fanIn = shape.dropFirst().reduce(1, *)
        let std = sqrt(2.0 / Float(fanIn))
        let count = shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        for i in 0..<count {
            // Box-Muller transform for normal distribution
            let u1 = Float.random(in: Float.ulpOfOne...1)
            let u2 = Float.random(in: 0...1)
            values[i] = std * sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
        }
        return values.withUnsafeBytes { Data($0) }
    }

    static func onesData(count: Int) -> Data {
        [Float](repeating: 1.0, count: count).withUnsafeBytes { Data($0) }
    }

    static func zerosData(count: Int) -> Data {
        [Float](repeating: 0.0, count: count).withUnsafeBytes { Data($0) }
    }
}
