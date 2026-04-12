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

        // Logits, not probabilities — softmax happens on the CPU over only
        // the legal moves (see MPSChessPlayer.sampleMove).
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
