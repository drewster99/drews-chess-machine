import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// Bug map: MPSGraph's fp32 convolution weight gradient is wrong past 2³¹
/// elements when the convolution's output has another consumer
/// (`ConvolutionWeightGradientWorkaround`; measured on macOS 27.2 26B123,
/// Apple M5 Max). Plain MPSGraph, no DCM graph code: Y = conv2d(X, W) at batch
/// 4096, 9×9, 128 channels, loss = Σ R·Y with Y also read as a target. dY is
/// exactly R, so dW's float64 reference is the correlation of X and R.
///
/// **The unsplit fp32 case asserts that the defect is still there.** When it
/// fails, MPSGraph changed: re-measure (`experiments/20261008-zlra-selfplay-lr/
/// analysis-fp32/repro/`) before relaxing or removing the workaround. The
/// split and bf16 cases are the workaround's and the reference's own checks.
///
/// Gated (`DCM_RUN_SLOW_TESTS`): the unsplit fp32 kernel allocates about
/// 20 GiB, and the CPU reference takes a while. Run with no trainer live.
final class MPSGraphFP32ConvWeightGradientBugMapTests: XCTestCase {

    override func setUpWithError() throws {
        try SlowTestGate.requireEnabled("MPSGraphFP32ConvWeightGradientBugMap")
    }

    private enum Form { case unsplit, inputChannelSplit }

    private static let batch = 4096
    private static let channels = 128
    private static let kernel = 9

    private func bf16Rounded(_ value: Float) -> Float {
        ChessNetwork.bFloat16BitsToFloat32(ChessNetwork.float32ToBFloat16Bits(value))
    }

    /// The relative error of MPSGraph's dW against the float64 reference.
    private func weightGradientRelativeError(dtype: MPSDataType, form: Form) throws -> Double {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let n = Self.batch, c = Self.channels, k = Self.kernel, pad = (Self.kernel - 1) / 2
        var random = DCMRandom(seed: 3)
        // bf16-representable values, so both dtypes see the same numbers.
        let x = (0..<(n * c * 64)).map { _ in bf16Rounded(Float.random(in: -2...2, using: &random)) }
        let w = (0..<(c * c * k * k)).map { _ in bf16Rounded(Float.random(in: -0.05...0.05, using: &random)) }
        let r = (0..<(n * c * 64)).map { _ in bf16Rounded(Float.random(in: -0.2...0.2, using: &random)) }
        func data(_ values: [Float]) -> Data {
            if dtype == .float32 { return values.withUnsafeBufferPointer { Data(buffer: $0) } }
            let bits = values.map { ChessNetwork.float32ToBFloat16Bits($0) }
            return bits.withUnsafeBufferPointer { Data(buffer: $0) }
        }
        let graph = MPSGraph()
        let shape: [NSNumber] = [NSNumber(value: n), NSNumber(value: c), 8, 8]
        let input = graph.constant(data(x), shape: shape, dataType: dtype)
        let weights = graph.variable(with: data(w), shape: [NSNumber(value: c), NSNumber(value: c), NSNumber(value: k), NSNumber(value: k)],
                                     dataType: dtype, name: nil)
        let noise = graph.constant(data(r), shape: shape, dataType: dtype)
        guard let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1,
            paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW) else {
            throw XCTSkip("convolution descriptor unavailable")
        }
        descriptor.paddingLeft = pad; descriptor.paddingRight = pad
        descriptor.paddingTop = pad; descriptor.paddingBottom = pad
        let output: MPSGraphTensor
        switch form {
        case .unsplit:
            output = graph.convolution2D(input, weights: weights, descriptor: descriptor, name: nil)
        case .inputChannelSplit:
            let half = c / 2
            let parts = (0..<2).map { g in
                graph.convolution2D(graph.sliceTensor(input, dimension: 1, start: g * half, length: half, name: nil),
                                    weights: graph.sliceTensor(weights, dimension: 1, start: g * half, length: half, name: nil),
                                    descriptor: descriptor, name: nil)
            }
            output = graph.addition(parts[0], parts[1], name: nil)
        }
        let product = graph.cast(graph.reshape(graph.multiplication(output, noise, name: nil), shape: [-1], name: nil),
                                 to: .float32, name: nil)
        let loss = graph.reductionSum(with: product, axis: 0, name: nil)
        let gradient = try XCTUnwrap(graph.gradients(of: loss, with: [weights], name: nil)[weights])
        // `output` as a second target: the extra consumer that selects the bad kernel.
        let results = graph.run(with: queue, feeds: [:], targetTensors: [gradient, output], targetOperations: nil)
        let gradientData = try XCTUnwrap(results[gradient])
        let dW: [Float]
        if dtype == .float32 {
            var values = [Float](repeating: 0, count: w.count)
            gradientData.mpsndarray().readBytes(&values, strideBytes: nil)
            dW = values
        } else {
            var bits = [UInt16](repeating: 0, count: w.count)
            gradientData.mpsndarray().readBytes(&bits, strideBytes: nil)
            dW = bits.map { ChessNetwork.bFloat16BitsToFloat32($0) }
        }

        var errorSquared = 0.0, referenceSquared = 0.0
        for t in 0..<4 {
            let o = (t * 37) % c, i = (t * 53 + 7) % c
            for ky in 0..<k { for kx in 0..<k {
                var sum = 0.0
                for b in 0..<n { for yy in 0..<8 {
                    let sy = yy + ky - pad
                    guard sy >= 0, sy < 8 else { continue }
                    for xx in 0..<8 {
                        let sx = xx + kx - pad
                        guard sx >= 0, sx < 8 else { continue }
                        sum += Double(r[((b * c + o) * 8 + yy) * 8 + xx]) * Double(x[((b * c + i) * 8 + sy) * 8 + sx])
                    }
                } }
                let difference = Double(dW[((o * c + i) * k + ky) * k + kx]) - sum
                errorSquared += difference * difference
                referenceSquared += sum * sum
            } }
        }
        return (errorSquared / referenceSquared).squareRoot()
    }

    /// Fails when MPSGraph's unsplit fp32 weight gradient at this shape becomes
    /// correct (or fails differently): the signal to re-measure the defect.
    func testUnsplitFloat32WeightGradientIsStillWrong() throws {
        let error = try weightGradientRelativeError(dtype: .float32, form: .unsplit)
        XCTAssertGreaterThan(error, 0.1, "MPSGraph's fp32 weight gradient at batch 4096 / 9x9 / 128 is now within "
                             + "\(error) of exact: the defect changed; re-measure before touching the workaround")
    }

    /// Protects: the input-channel split the app uses is exact at this shape.
    func testInputChannelSplitFloat32WeightGradientIsExact() throws {
        XCTAssertLessThan(try weightGradientRelativeError(dtype: .float32, form: .inputChannelSplit), 1e-4)
    }

    /// Protects: bf16 at the same shape is right to bf16 rounding (why bf16
    /// graphs are not split).
    func testUnsplitBFloat16WeightGradientIsRightToRounding() throws {
        XCTAssertLessThan(try weightGradientRelativeError(dtype: .bFloat16, form: .unsplit), 1e-2)
    }
}
