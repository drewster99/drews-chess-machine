import XCTest
import MetalPerformanceShaders
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

    /// `runPreparedStep` maps the executable's result array back to the loss /
    /// diagnostic tensors with `Dictionary(zip(targets, resultArray))`, which
    /// assumes `executable.run` returns results in the **same order** as the
    /// `targetTensors` passed at compile. That assumption is invisible with a
    /// single target, so this exercises it with TWO distinct-valued outputs:
    /// if `run` reordered them, the `zip` mapping would assign the wrong value
    /// to each tensor and the asserts would fail.
    func testExecutableMultiTargetResultsKeepCompileOrder() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let graph = MPSGraph()
        let W = graph.variable(with: dataOf([2.0, 3.0, 4.0]), shape: shape, dataType: .float32, name: "W")
        let x = graph.placeholder(shape: shape, dataType: .float32, name: "x")
        let prod = graph.multiplication(W, x, name: "prod")              // [3]
        let outSum = graph.reductionSum(with: prod, axes: [0], name: "outSum")        // 9
        let outMax = graph.reductionMaximum(with: prod, axes: [0], name: "outMax")    // 4

        // Same target order used by both paths and by the mapping under test.
        let targets = [outSum, outMax]
        let feedX = tensorData(device, [1.0, 1.0, 1.0], shape: shape)

        // Ground truth via graph.run (dictionary-keyed, order-independent).
        let gr = graph.run(with: queue, feeds: [x: feedX], targetTensors: targets, targetOperations: nil)
        let grSum = readBack(try XCTUnwrap(gr[outSum]), count: 1)[0]
        let grMax = readBack(try XCTUnwrap(gr[outMax]), count: 1)[0]
        // The two outputs must be distinct, else a swap would be undetectable.
        XCTAssertNotEqual(grSum, grMax, "test outputs must differ to detect a reorder")

        // Executable path, mapped exactly as runPreparedStep does.
        let exe = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [x: MPSGraphShapedType(shape: shape, dataType: .float32)],
            targetTensors: targets,
            targetOperations: nil,
            compilationDescriptor: MPSGraphCompilationDescriptor()
        )
        let feedOrder = try XCTUnwrap(exe.feedTensors)
        let dataByTensor: [MPSGraphTensor: MPSGraphTensorData] = [x: feedX]
        let inputs = try feedOrder.map { try XCTUnwrap(dataByTensor[$0]) }
        let resultArray = exe.run(with: queue, inputs: inputs, results: nil, executionDescriptor: nil)
        let mapped = Dictionary(uniqueKeysWithValues: zip(targets, resultArray))

        XCTAssertEqual(readBack(try XCTUnwrap(mapped[outSum]), count: 1)[0], grSum, accuracy: 1e-6,
                       "outSum must map to the sum, not the max — result order must follow compile order")
        XCTAssertEqual(readBack(try XCTUnwrap(mapped[outMax]), count: 1)[0], grMax, accuracy: 1e-6,
                       "outMax must map to the max — result order must follow compile order")
    }

    /// Phase 3, Increment 1: `runPreparedStep` switches from `executable.run`
    /// (synchronous) to `executable.encode(to: MPSCommandBuffer)` + commit +
    /// wait. This proves the two produce identical results, so the plumbing
    /// change is safe before Increment 2 stops waiting and goes N-deep.
    func testExecutableEncodeToCommandBufferMatchesRun() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let graph = MPSGraph()
        let W = graph.variable(with: dataOf([2.0, 3.0, 4.0]), shape: shape, dataType: .float32, name: "W")
        let x = graph.placeholder(shape: shape, dataType: .float32, name: "x")
        let prod = graph.multiplication(W, x, name: "prod")
        let outSum = graph.reductionSum(with: prod, axes: [0], name: "outSum")
        let outMax = graph.reductionMaximum(with: prod, axes: [0], name: "outMax")
        let targets = [outSum, outMax]
        let feedX = tensorData(device, [1.0, 1.0, 1.0], shape: shape)

        let exe = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [x: MPSGraphShapedType(shape: shape, dataType: .float32)],
            targetTensors: targets,
            targetOperations: nil,
            compilationDescriptor: MPSGraphCompilationDescriptor()
        )
        let inputs = try XCTUnwrap(exe.feedTensors).map { _ in feedX }

        // Path A — synchronous run (the old path).
        let runResults = exe.run(with: queue, inputs: inputs, results: nil, executionDescriptor: nil)

        // Path B — encode into a command buffer we own, commit, wait (Increment 1).
        let mtlCB = try XCTUnwrap(queue.makeCommandBuffer())
        let mpsCB = MPSCommandBuffer(commandBuffer: mtlCB)
        let encResults = exe.encode(to: mpsCB, inputs: inputs, results: nil, executionDescriptor: nil)
        mpsCB.commit()
        mpsCB.waitUntilCompleted()

        XCTAssertEqual(runResults.count, encResults.count)
        for i in runResults.indices {
            XCTAssertEqual(readBack(runResults[i], count: 1)[0],
                           readBack(encResults[i], count: 1)[0],
                           accuracy: 1e-6,
                           "encode(to:) result[\(i)] must match run() result")
        }
    }

    /// The load-bearing correctness for Increment 1: that `encode(to:)` actually
    /// executes the compiled **assign target-operations** (the SGD weight update
    /// in the real training executable), not just the result tensors. A silently
    /// skipped assign would stop the network learning. Build a graph whose target
    /// operation is an assign, drive it through encode + commit + wait, and
    /// confirm the variable was updated — read in a SEPARATE run, since
    /// assign→read within one run is unordered.
    func testExecutableEncodeAppliesAssignTargetOperation() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let cmdQueue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let graph = MPSGraph()
        let W = graph.variable(with: dataOf([1.0, 1.0, 1.0]), shape: shape, dataType: .float32, name: "W")
        let ph = graph.placeholder(shape: shape, dataType: .float32, name: "ph")
        let assign = graph.assign(W, tensor: ph, name: "assign")
        let zero = graph.constant(0.0, shape: shape, dataType: .float32)
        let readW = graph.addition(W, zero, name: "readW")

        let exe = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [ph: MPSGraphShapedType(shape: shape, dataType: .float32)],
            targetTensors: [readW],
            targetOperations: [assign],
            compilationDescriptor: MPSGraphCompilationDescriptor()
        )
        let inputs = try XCTUnwrap(exe.feedTensors).map { _ in tensorData(device, [7.0, 7.0, 7.0], shape: shape) }

        let mtlCB = try XCTUnwrap(cmdQueue.makeCommandBuffer())
        let mpsCB = MPSCommandBuffer(commandBuffer: mtlCB)
        _ = exe.encode(to: mpsCB, inputs: inputs, results: nil, executionDescriptor: nil)
        mpsCB.commit()
        mpsCB.waitUntilCompleted()

        // Separate run reads the variable; it must reflect the encode's assign.
        let read = graph.run(with: cmdQueue, feeds: [:], targetTensors: [readW], targetOperations: nil)
        XCTAssertEqual(readBack(try XCTUnwrap(read[readW]), count: n), [7.0, 7.0, 7.0],
                       "encode(to:) must execute the compiled assign target-operation (weight update)")
    }

    /// Guards the safe pattern the trainer actually uses: **one cached
    /// `MPSGraphExecutable`, re-encoded SERIALLY** into a fresh command buffer each
    /// step (distinct inputs), must give the correct result every time. This is
    /// what `runPreparedStep` relies on — compile once, reuse every step.
    ///
    /// Backstory (Phase-3 prerequisite probe, build 1620, 2026-06-03): a first
    /// concurrent-encode probe that passed `results: nil` cross-contaminated its
    /// outputs (iter #5 got #7's answer, etc.) — because with `nil` the executable
    /// allocates and *reuses* internal result storage, which is shared mutable
    /// state. The follow-up `testExecutableConcurrentEncodeWithCallerOwnedResultsIsSafe`
    /// then showed that with **caller-owned per-thread result buffers**, concurrent
    /// encode on one executable IS correct. So the contract is: one executable may
    /// be encoded serially (this test) or concurrently (that test) as long as the
    /// caller owns the input/result buffers. See GPU_UTILIZATION_PLAN.md Phase 3.
    func testExecutableSerialReuseAcrossCommandBuffersIsCorrect() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        queue.label = "serial-reuse-test"
        let graph = MPSGraph()
        let W = graph.variable(with: dataOf([2.0, 3.0, 4.0]), shape: shape, dataType: .float32, name: "W")
        let x = graph.placeholder(shape: shape, dataType: .float32, name: "x")
        let prod = graph.multiplication(W, x, name: "prod")
        let outSum = graph.reductionSum(with: prod, axes: [0], name: "outSum")   // = 9·xi
        let exe = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [x: MPSGraphShapedType(shape: shape, dataType: .float32)],
            targetTensors: [outSum],
            targetOperations: nil,
            compilationDescriptor: MPSGraphCompilationDescriptor()
        )

        let iterations = 16
        for i in 0..<iterations {
            let xi = Float(i + 1)
            let feed = tensorData(device, [xi, xi, xi], shape: shape)
            guard let mtlCB = queue.makeCommandBuffer() else {
                XCTFail("makeCommandBuffer returned nil at iteration \(i)")
                return
            }
            let mpsCB = MPSCommandBuffer(commandBuffer: mtlCB)
            let r = exe.encode(to: mpsCB, inputs: [feed], results: nil, executionDescriptor: nil)
            mpsCB.commit()
            mpsCB.waitUntilCompleted()
            let got = readBack(r[0], count: 1)[0]
            XCTAssertEqual(got, 9.0 * xi, accuracy: 1e-4,
                           "serial reuse #\(i): expected \(9.0 * xi), got \(got)")
        }
    }

    /// The doc-informed re-test of concurrent encode. The header documents
    /// `resultsArray` as "Tensors for which the caller wishes MPSGraphTensorData to
    /// be returned"; with `results: nil` the executable allocates and **reuses**
    /// internal result storage — shared mutable state that cross-contaminated the
    /// earlier nil-results concurrent probe. Here each thread passes its **own**
    /// pre-allocated result buffer, so concurrent encodes can't clobber each other's
    /// outputs. If this passes, `encode` IS usable concurrently on one executable
    /// provided the caller owns the result (and input) buffers — which would put the
    /// parallel-encoder pipeline back on the table (GPU_UTILIZATION_PLAN.md Phase 3).
    func testExecutableConcurrentEncodeWithCallerOwnedResultsIsSafe() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        queue.label = "concurrent-encode-owned-results"
        let graph = MPSGraph()
        let W = graph.variable(with: dataOf([2.0, 3.0, 4.0]), shape: shape, dataType: .float32, name: "W")
        let x = graph.placeholder(shape: shape, dataType: .float32, name: "x")
        let prod = graph.multiplication(W, x, name: "prod")
        let outSum = graph.reductionSum(with: prod, axes: [0], name: "outSum")   // shape [1], = 9·xi
        let exe = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [x: MPSGraphShapedType(shape: shape, dataType: .float32)],
            targetTensors: [outSum],
            targetOperations: nil,
            compilationDescriptor: MPSGraphCompilationDescriptor()
        )

        let iterations = 16
        let outShape: [NSNumber] = [1]
        // Per-iteration inputs AND result buffers, pre-created serially.
        let feeds: [MPSGraphTensorData] = (0..<iterations).map { i in
            let xi = Float(i + 1)
            return tensorData(device, [xi, xi, xi], shape: shape)
        }
        let outputs: [MPSGraphTensorData] = (0..<iterations).map { _ in
            MPSGraphTensorData(MPSNDArray(device: device,
                                          descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: outShape)))
        }
        var results = [Float](repeating: .nan, count: iterations)
        results.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: iterations) { i in
                guard let mtlCB = queue.makeCommandBuffer() else { return }
                let mpsCB = MPSCommandBuffer(commandBuffer: mtlCB)
                _ = exe.encode(to: mpsCB, inputs: [feeds[i]], results: [outputs[i]], executionDescriptor: nil)
                mpsCB.commit()
                mpsCB.waitUntilCompleted()
                buf[i] = self.readBack(outputs[i], count: 1)[0]
            }
        }

        for i in 0..<iterations {
            XCTAssertEqual(results[i], 9.0 * Float(i + 1), accuracy: 1e-4,
                           "owned-results concurrent encode #\(i): expected \(9.0 * Float(i + 1)), got \(results[i])")
        }
    }
}
