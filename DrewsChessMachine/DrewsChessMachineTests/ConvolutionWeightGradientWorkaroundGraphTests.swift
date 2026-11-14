import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// The app's own fp32 training graph gets the right weight gradient for a
/// BatchNorm-fed 9×9 / 128-channel convolution at batch 4096 — the shape and
/// batch where MPSGraph's unsplit fp32 weight gradient is 75% wrong
/// (`ConvolutionWeightGradientWorkaround`; R-elitecyc-fp32).
///
/// The reference needs no MPSGraph gradient math of the convolution: dW is the
/// correlation of the convolution's input X with the gradient arriving at its
/// output dY, computed here in float64. X is rebuilt from the block input the
/// graph taps (`block0_bn1_input`) through bn1's batch statistics, γ/β and ReLU;
/// dY is the graph's gradient at the tapped conv1 output (`block0_bn2_input`),
/// which the BatchNorm backward produces and the measurements show exact.
final class ConvolutionWeightGradientWorkaroundGraphTests: XCTestCase {

    private static let batch = 4096
    private static let channels = 128
    private static let kernel = 9

    private func architecture() throws -> NetworkArchitecture {
        var arch = NetworkArchitecture(
            inputEncoding: .basic24, channels: Self.channels, numBlocks: 1, stemConvKernelSize: 5,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: false, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: Self.kernel, blockConv2KernelSize: Self.kernel,
            blockSeStyle: .none, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 32,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32)
        arch.clearActivationSitesTheTopologyLacks()
        try arch.setActivationAtEveryExistingSite(.relu)
        try arch.validate()
        return arch
    }

    private func read(_ data: MPSGraphTensorData, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        data.mpsndarray().readBytes(&out, strideBytes: nil)
        return out
    }

    /// Protects: fp32 training of a BatchNorm-fed convolution past 2³¹
    /// elements gets the true weight gradient.
    /// Fails if: the training graph builds that convolution unsplit (MPSGraph's
    /// fp32 kernel gives a relative error ≈ 0.75 here), or the split is built
    /// wrong (wrong slices, a missing group, a group summed twice).
    func testFloat32TrainingGraphWeightGradientOfBatchNormFedConvolutionIsExact() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        let arch = try architecture()
        let network = try ChessNetwork(arch: arch, bnMode: .training, initialization: .seeded(initSeed: 1),
                                       analysisTaps: true)
        XCTAssertEqual(network.convolutionWeightGradientSplit.groups(forWeight: "blocks.0.conv1.weight"), 2,
                       "the shape under test must be one the workaround covers")
        let graph = network.graph
        let n = Self.batch, c = Self.channels, k = Self.kernel, pad = (Self.kernel - 1) / 2

        func variable(_ name: String) throws -> MPSGraphTensor {
            try XCTUnwrap(network.trainableVariables.first { $0.operation.name == name }, name)
        }
        func tap(_ name: String) throws -> MPSGraphTensor {
            try XCTUnwrap(network.analysisTapReadbacks.first { $0.name == name }?.tensor, name)
        }
        let conv1Weights = try variable("block0_conv1_weights")
        let bn1Gamma = try variable("block0_bn1_gamma")
        let bn1Beta = try variable("block0_bn1_beta")
        let blockInput = try tap("block0_bn1_input")
        let conv1Output = try tap("block0_bn2_input")

        // loss = Σ R · policy logits, R fixed noise: a gradient reaches every
        // weight through the whole tower.
        var random = DCMRandom(seed: 41)
        let policyCount = n * ChessNetwork.policySize
        let rValues = (0..<policyCount).map { _ in Float.random(in: -1...1, using: &random) }
        let r = graph.constant(rValues.withUnsafeBufferPointer { Data(buffer: $0) },
                               shape: [NSNumber(value: n), NSNumber(value: ChessNetwork.policySize)], dataType: .float32)
        let loss = graph.reductionSum(
            with: graph.reshape(graph.multiplication(network.policyOutput, r, name: nil), shape: [-1], name: nil),
            axis: 0, name: nil)
        let grads = graph.gradients(of: loss, with: [conv1Weights, conv1Output], name: nil)
        let dWTensor = try XCTUnwrap(grads[conv1Weights])
        let dYTensor = try XCTUnwrap(grads[conv1Output])

        // Sparse 0/1 board planes, like real encodings.
        let planes = arch.inputPlanes
        let boardValues = (0..<(n * planes * 64)).map { _ in Float.random(in: 0..<1, using: &random) < 0.15 ? Float(1) : 0 }
        let boardData = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: network.metalDevice),
                                           data: boardValues.withUnsafeBufferPointer { Data(buffer: $0) },
                                           shape: [NSNumber(value: n), NSNumber(value: planes), 8, 8], dataType: .float32)
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [network.inputPlaceholder: boardData]
        if let ratePlaceholder = network.dropoutRateFeedPlaceholder {
            feeds[ratePlaceholder] = try XCTUnwrap(network.dropoutRateZeroTensorData)
        }
        let results = graph.run(with: network.commandQueue, feeds: feeds,
                                targetTensors: [dWTensor, dYTensor, blockInput, bn1Gamma, bn1Beta, conv1Weights],
                                targetOperations: nil)
        let graphDW = read(try XCTUnwrap(results[dWTensor]), count: c * c * k * k)
        let dY = read(try XCTUnwrap(results[dYTensor]), count: n * c * 64)
        let input = read(try XCTUnwrap(results[blockInput]), count: n * c * 64)
        let gamma = read(try XCTUnwrap(results[bn1Gamma]), count: c)
        let beta = read(try XCTUnwrap(results[bn1Beta]), count: c)

        // X = ReLU(γ·(input − μ)/√(σ² + ε) + β), batch statistics over (n, h, w).
        var x = [Double](repeating: 0, count: n * c * 64)
        let m = Double(n * 64)
        for ch in 0..<c {
            var mean = 0.0
            for b in 0..<n { for j in 0..<64 { mean += Double(input[(b * c + ch) * 64 + j]) } }
            mean /= m
            var variance = 0.0
            for b in 0..<n { for j in 0..<64 { let d = Double(input[(b * c + ch) * 64 + j]) - mean; variance += d * d } }
            variance /= m
            let scale = Double(gamma[ch]) / (variance + 1e-5).squareRoot()
            for b in 0..<n { for j in 0..<64 {
                let i = (b * c + ch) * 64 + j
                x[i] = max(0, (Double(input[i]) - mean) * scale + Double(beta[ch]))
            } }
        }

        // dW[o, i, ky, kx] = Σ_{b,y,x} dY[b, o, y, x] · X[b, i, y + ky − pad, x + kx − pad]
        // on a spread of (o, i) pairs, all 81 taps each.
        var errorSquared = 0.0, referenceSquared = 0.0
        for t in 0..<6 {
            let o = (t * 37) % c, i = (t * 53 + 7) % c
            for ky in 0..<k { for kx in 0..<k {
                var sum = 0.0
                for b in 0..<n { for yy in 0..<8 {
                    let sy = yy + ky - pad
                    guard sy >= 0, sy < 8 else { continue }
                    for xx in 0..<8 {
                        let sx = xx + kx - pad
                        guard sx >= 0, sx < 8 else { continue }
                        sum += Double(dY[((b * c + o) * 8 + yy) * 8 + xx]) * x[((b * c + i) * 8 + sy) * 8 + sx]
                    }
                } }
                let difference = Double(graphDW[((o * c + i) * k + ky) * k + kx]) - sum
                errorSquared += difference * difference
                referenceSquared += sum * sum
            } }
        }
        XCTAssertGreaterThan(referenceSquared, 0, "the reference gradient must be non-zero")
        let relativeError = (errorSquared / referenceSquared).squareRoot()
        XCTAssertLessThan(relativeError, 1e-4, "graph dW vs float64 corr(X, dY): relative error \(relativeError)")
    }
}
