import XCTest
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// De-risking gate for "compile the training step to an `MPSGraphExecutable`"
/// (GPU_UTILIZATION_PLAN.md, Phase 2): proves that running a representative
/// SGD step through a compiled `MPSGraphExecutable` produces **the same loss and
/// the same weight update** as the current `graph.run` path, bit-comparably
/// within fp tolerance.
///
/// The mini-graph mirrors the real training step's shape: a trainable
/// `graph.variable`, several feed placeholders (input, target, learning rate),
/// an MSE loss, gradients via `graph.gradients(of:with:)`, and an SGD weight
/// update applied as a `graph.assign` **target operation** — the same
/// ingredients `ChessTrainer.buildTrainingOps` uses. It also exercises the
/// `executable.feedTensors`-ordered input binding, which is the one place a
/// silent feed-ordering bug could corrupt training in the real conversion.
///
/// Two independent graphs with identical initial weights are used (one per
/// path) so neither run's in-place weight mutation perturbs the other.
///
/// Run only with no training session live (Metal/MPSGraph).
final class MPSGraphExecutableTrainingEquivalenceTests: XCTestCase {

    private let n = 3
    private var shape: [NSNumber] { [NSNumber(value: n)] }
    private let wInit: [Float] = [0.5, -0.3, 1.0]
    private let xVals: [Float] = [1.0, 2.0, -1.0]
    private let yVals: [Float] = [0.25, -0.5, 0.75]
    private let lrVal: [Float] = [0.1]

    /// One SGD-step mini training graph: `loss = mean((W·x − y)²)`, update
    /// `W ← W − lr · dLoss/dW` applied via an assign target op.
    private struct MiniTrainer {
        let graph: MPSGraph
        let W: MPSGraphTensor
        let x: MPSGraphTensor
        let y: MPSGraphTensor
        let lr: MPSGraphTensor
        let loss: MPSGraphTensor
        let readW: MPSGraphTensor
        let assignW: MPSGraphOperation
    }

    private func buildMiniTrainer() throws -> MiniTrainer {
        let graph = MPSGraph()
        let W = graph.variable(with: dataOf(wInit), shape: shape, dataType: .float32, name: "W")
        let x = graph.placeholder(shape: shape, dataType: .float32, name: "x")
        let y = graph.placeholder(shape: shape, dataType: .float32, name: "y")
        let lr = graph.placeholder(shape: [1], dataType: .float32, name: "lr")

        let pred = graph.multiplication(W, x, name: "pred")
        let diff = graph.subtraction(pred, y, name: "diff")
        let sq = graph.multiplication(diff, diff, name: "sq")
        let loss = graph.mean(of: sq, axes: [0], name: "loss")  // shape [1]

        let grads = graph.gradients(of: loss, with: [W], name: "grads")
        guard let gW = grads[W] else { throw Failure.gradientMissing }
        let step = graph.multiplication(lr, gW, name: "step")     // [1]·[n] → [n]
        let wNew = graph.subtraction(W, step, name: "w_new")
        let assignW = graph.assign(W, tensor: wNew, name: "assign_w")

        let zero = graph.constant(0.0, shape: shape, dataType: .float32)
        let readW = graph.addition(W, zero, name: "read_w")
        return MiniTrainer(graph: graph, W: W, x: x, y: y, lr: lr, loss: loss, readW: readW, assignW: assignW)
    }

    private enum Failure: Error { case gradientMissing }

    private func dataOf(_ xs: [Float]) -> Data { xs.withUnsafeBytes { Data($0) } }

    private func tensorData(_ device: MTLDevice, _ xs: [Float], shape: [NSNumber]) -> MPSGraphTensorData {
        let nd = MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: shape))
        var local = xs
        local.withUnsafeMutableBytes { if let b = $0.baseAddress { nd.writeBytes(b, strideBytes: nil) } }
        return MPSGraphTensorData(nd)
    }

    private func readBack(_ td: MPSGraphTensorData, count: Int) -> [Float] {
        var out = [Float](repeating: .nan, count: count)
        out.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { td.mpsndarray().readBytes(base, strideBytes: nil) }
        }
        return out
    }

    func testExecutableTrainingStepMatchesGraphRun() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }

        // --- Path A: graph.run ---
        let a = try buildMiniTrainer()
        let feedsA: [MPSGraphTensor: MPSGraphTensorData] = [
            a.x: tensorData(device, xVals, shape: shape),
            a.y: tensorData(device, yVals, shape: shape),
            a.lr: tensorData(device, lrVal, shape: [1]),
        ]
        let resA = a.graph.run(with: queue, feeds: feedsA, targetTensors: [a.loss], targetOperations: [a.assignW])
        let lossA = readBack(try XCTUnwrap(resA[a.loss]), count: 1)[0]
        // Read the post-update weights in a SEPARATE run (assign→read ordering
        // is not guaranteed within one run; see the variable-semantics test).
        let readA = a.graph.run(with: queue, feeds: [:], targetTensors: [a.readW], targetOperations: nil)
        let wAfterA = readBack(try XCTUnwrap(readA[a.readW]), count: n)

        // --- Path B: compiled executable ---
        let b = try buildMiniTrainer()
        let executable = b.graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [
                b.x: MPSGraphShapedType(shape: shape, dataType: .float32),
                b.y: MPSGraphShapedType(shape: shape, dataType: .float32),
                b.lr: MPSGraphShapedType(shape: [1], dataType: .float32),
            ],
            targetTensors: [b.loss],
            targetOperations: [b.assignW],
            compilationDescriptor: MPSGraphCompilationDescriptor()
        )
        // Bind inputs in the executable's own feedTensors order — the exact
        // binding the ChessTrainer conversion will use.
        let dataByTensor: [MPSGraphTensor: MPSGraphTensorData] = [
            b.x: tensorData(device, xVals, shape: shape),
            b.y: tensorData(device, yVals, shape: shape),
            b.lr: tensorData(device, lrVal, shape: [1]),
        ]
        let feedOrder = try XCTUnwrap(executable.feedTensors, "executable should expose feedTensors")
        let inputs = try feedOrder.map { try XCTUnwrap(dataByTensor[$0], "missing input for a feed tensor") }
        let resB = executable.run(with: queue, inputs: inputs, results: nil, executionDescriptor: nil)
        let lossB = readBack(try XCTUnwrap(resB.first), count: 1)[0]
        let readB = b.graph.run(with: queue, feeds: [:], targetTensors: [b.readW], targetOperations: nil)
        let wAfterB = readBack(try XCTUnwrap(readB[b.readW]), count: n)

        // --- Sanity: the step actually did something non-degenerate ---
        XCTAssertTrue(lossA.isFinite && lossA > 0, "loss should be finite & positive, got \(lossA)")
        XCTAssertNotEqual(wAfterA, wInit, "the SGD step should have moved the weights")

        // --- Equivalence: executable == graph.run ---
        XCTAssertEqual(lossA, lossB, accuracy: 1e-6,
                       "executable loss must match graph.run loss")
        for i in 0..<n {
            XCTAssertEqual(wAfterA[i], wAfterB[i], accuracy: 1e-6,
                           "post-update weight[\(i)] mismatch: graph.run=\(wAfterA[i]) executable=\(wAfterB[i])")
        }
    }
}
