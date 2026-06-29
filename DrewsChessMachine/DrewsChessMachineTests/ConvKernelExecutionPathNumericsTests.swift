import XCTest
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// Discriminating measurement for the mid-run training-collapse investigation
/// (OVERNIGHT_INVESTIGATION.md Track 2, continued 2026-06-10).
///
/// The timeline evidence: every run whose tower/stem contains 3×3 convs and
/// that trained on the compiled-executable step path (`a1f1e7e` moved the
/// training step from `graph.run` onto a cached `MPSGraphExecutable`;
/// `3f02ac4` set `.level1` optimization; both 2026-06-02) eventually shows an
/// out-of-distribution policy blowup after tens of thousands of healthy
/// steps, while pure-7×7 towers on the same code are stable AND a 12-block
/// 3×3 tower trained to ~106k steps cleanly on the pre-`a1f1e7e` `graph.run`
/// path (build 1562, 2026-06-01). 3×3 is the kernel size eligible for
/// Winograd-style fast convolution — a transform whose intermediate values
/// are numerically fragile in bf16 — while 7×7 takes direct convolution, so
/// "compiled `.level1` executable selects a different (lower-precision or
/// biased) 3×3 conv/gradient kernel than `graph.run` did" fits every run.
/// The executable≡run equivalence test written alongside `a1f1e7e`
/// (`MPSGraphExecutableTrainingEquivalenceTests`) exercises element-wise
/// mini-graphs only, so it could not have caught a conv-kernel-selection
/// difference; this file closes that hole.
///
/// These tests MEASURE rather than assert equivalence: each arm's gradients
/// are compared against an fp32 `graph.run` reference and against the other
/// execution paths, and the numbers are written to a report file
/// (`reportPath`) plus the test log. Assertions cover only invariants that
/// must hold regardless of which way the evidence falls (finite, non-zero
/// gradients; report written), so the suite stays green and the verdict is
/// read from the report. If the hypothesis is right, the report shows the
/// bf16 3×3 arms diverging between `graph.run` and the `.level1` executable
/// (especially a non-zero signed bias, which rounding noise alone cannot
/// produce) while the 7×7 arms and the fp32 arms agree closely.
///
/// Caveat for interpreting a CLEAN result: MPS kernel selection depends on
/// problem size as well as kernel size. These towers use the live net's
/// channel width but a smaller batch than the live 4096, so agreement here
/// narrows the hypothesis without fully exonerating the production shapes —
/// the definitive falsifier remains a real training run with the step
/// executable compiled at `.level0`.
///
/// Run only with no training session live (Metal/MPSGraph).
final class ConvKernelExecutionPathNumericsTests: XCTestCase {

    // MARK: - Geometry

    /// Channel width matches the live net's tower so the conv shapes MPS sees
    /// are realistic; batch is smaller than the live 4096 to keep the test
    /// fast (see the class-doc caveat about size-dependent kernel selection).
    private static let channels = 128
    private static let gradientTestBatch = 512
    private static let side = ChessNetwork.boardSize

    /// Where the measurement report lands. /tmp so a failed-to-clean run
    /// can't pollute the repo; the contents get copied into the
    /// investigation notes once read.
    private static let reportPath = "/tmp/conv_kernel_path_numerics_report.txt"

    // MARK: - Deterministic data

    /// LCG matching the convention used by the other Metal numerics tests.
    private struct SeededRandom {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return (Float(state >> 40) / Float(1 << 24)) * 2 - 1   // [-1, 1)
        }
    }

    /// Values that are exactly representable in bf16, as fp32. Feeding these
    /// to the fp32 arms and the bf16 arms guarantees every arm starts from
    /// identical real-valued data — any later divergence is compute, not
    /// input rounding.
    private static func bf16RepresentableValues(count: Int, scale: Float, seed: UInt64) -> [Float] {
        var rng = SeededRandom(state: seed)
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let bits = ChessNetwork.float32ToBFloat16Bits(rng.next() * scale)
            out[i] = ChessNetwork.bFloat16BitsToFloat32(bits)
        }
        return out
    }

    /// Values deliberately placed a chosen fraction of a bf16 ULP away from
    /// a representable bf16 value, as fp32. These are NOT bf16-representable:
    /// the in-graph fp32→bf16 cast has to round them, and `halfUlpFraction`
    /// controls which side of the round-to-nearest boundary they sit on
    /// (just below ½ ULP rounds down, just above rounds up, exactly ½ ULP
    /// exercises ties-to-even). Every execution path receives identical fp32
    /// bits, so any cross-path difference means the paths round differently —
    /// e.g. a `.level1` cast→conv fusion that truncates instead of RNE.
    private static func bf16UlpBoundaryValues(
        count: Int, scale: Float, halfUlpFraction: Float, seed: UInt64
    ) -> [Float] {
        var rng = SeededRandom(state: seed)
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let raw = rng.next() * scale
            // Clamp the magnitude above bf16's min normal so `bits + 1` is
            // always the next representable value in the same binade
            // direction and the ULP below is well-defined.
            let magnitude = max(abs(raw), 1.5e-38)
            let bits = ChessNetwork.float32ToBFloat16Bits(magnitude)
            let base = ChessNetwork.bFloat16BitsToFloat32(bits)
            let ulp = ChessNetwork.bFloat16BitsToFloat32(bits + 1) - base
            let sign: Float = raw < 0 ? -1 : 1
            out[i] = sign * (base + halfUlpFraction * ulp)
        }
        return out
    }

    /// bf16-representable magnitudes spread across thirteen power-of-two
    /// binades with random signs. Accumulating products of these forces
    /// repeated catastrophic cancellation across ~four orders of magnitude,
    /// so any difference in summation ORDER between two conv implementations
    /// (direct vs transformed, different tiling) produces visibly different
    /// rounded results — bit-equality under this load is strong evidence the
    /// paths run the identical kernel.
    private static func mixedExponentValues(count: Int, scale: Float, seed: UInt64) -> [Float] {
        var rng = SeededRandom(state: seed)
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let bits = ChessNetwork.float32ToBFloat16Bits(rng.next() * scale)
            let binade = exp2(Float((i % 13) - 6))   // 2^-6 … 2^+6
            out[i] = ChessNetwork.bFloat16BitsToFloat32(bits) * binade
        }
        return out
    }

    private static func tensorData(
        _ device: MTLDevice, _ values: [Float], shape: [NSNumber]
    ) -> MPSGraphTensorData {
        let nd = MPSNDArray(
            device: device,
            descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: shape)
        )
        var local = values
        local.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { nd.writeBytes(base, strideBytes: nil) }
        }
        return MPSGraphTensorData(nd)
    }

    private static func readBack(_ td: MPSGraphTensorData, count: Int) -> [Float] {
        var out = [Float](repeating: .nan, count: count)
        out.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { td.mpsndarray().readBytes(base, strideBytes: nil) }
        }
        return out
    }

    // MARK: - Tower construction

    /// Mirrors the live trainer's weight regime: fp32 "master" variables, an
    /// in-graph cast to the compute dtype (the working copy), bf16 (or fp32)
    /// convs, gradients taken with respect to the fp32 masters through the
    /// cast — exactly the autodiff structure `ChessTrainer.buildTrainingOps`
    /// produces under bf16. The tower is conv(k)→relu→conv(k) with an
    /// MSE-to-target loss; loss and weight gradients are cast to fp32 for
    /// exact readback.
    private struct ConvTower {
        let graph: MPSGraph
        let inputPlaceholder: MPSGraphTensor
        let targetPlaceholder: MPSGraphTensor
        let lossF32: MPSGraphTensor
        let gradF32: [MPSGraphTensor]      // d loss / d master, per conv, fp32
        let readMastersF32: [MPSGraphTensor]
        let assignOps: [MPSGraphOperation] // SGD master update, drift test only
        let inputShape: [NSNumber]
    }

    private enum Failure: Error { case gradientMissing(String) }

    private static func buildConvTower(
        kernel: Int,
        computeDataType: MPSDataType,
        batch: Int,
        masterW1: [Float],
        masterW2: [Float],
        sgdLearningRate: Float?
    ) throws -> ConvTower {
        let c = channels
        let graph = MPSGraph()
        let inputShape: [NSNumber] = [
            NSNumber(value: batch), NSNumber(value: c),
            NSNumber(value: side), NSNumber(value: side),
        ]
        let weightShape: [NSNumber] = [
            NSNumber(value: c), NSNumber(value: c),
            NSNumber(value: kernel), NSNumber(value: kernel),
        ]
        let pad = (kernel - 1) / 2
        guard let convDesc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1,
            groups: 1,
            paddingLeft: pad, paddingRight: pad, paddingTop: pad, paddingBottom: pad,
            paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW
        ) else { throw Failure.gradientMissing("conv descriptor") }

        func masterVariable(_ values: [Float], name: String) -> MPSGraphTensor {
            let data = values.withUnsafeBytes { Data($0) }
            return graph.variable(with: data, shape: weightShape, dataType: .float32, name: name)
        }
        func working(_ master: MPSGraphTensor, name: String) -> MPSGraphTensor {
            computeDataType == .float32
                ? master
                : graph.cast(master, to: computeDataType, name: name)
        }

        let m1 = masterVariable(masterW1, name: "master_w1")
        let m2 = masterVariable(masterW2, name: "master_w2")
        let w1 = working(m1, name: "working_w1")
        let w2 = working(m2, name: "working_w2")

        let inputPH = graph.placeholder(shape: inputShape, dataType: .float32, name: "input")
        let targetPH = graph.placeholder(shape: inputShape, dataType: .float32, name: "target")
        let x = computeDataType == .float32
            ? inputPH
            : graph.cast(inputPH, to: computeDataType, name: "input_cast")
        let t = computeDataType == .float32
            ? targetPH
            : graph.cast(targetPH, to: computeDataType, name: "target_cast")

        let h1 = graph.convolution2D(x, weights: w1, descriptor: convDesc, name: "conv1")
        let a1 = graph.reLU(with: h1, name: "relu1")
        let out = graph.convolution2D(a1, weights: w2, descriptor: convDesc, name: "conv2")

        let diff = graph.subtraction(out, t, name: "diff")
        let sq = graph.multiplication(diff, diff, name: "sq")
        let meanLoss = graph.mean(of: sq, axes: [0, 1, 2, 3], name: "mse")
        // Lift the gradients out of the deep-subnormal range the all-axes
        // mean would put them in (the 1/N seed is tiny); identical scaling in
        // every arm so comparisons are unaffected.
        let lossScale = graph.constant(256.0, dataType: computeDataType)
        let loss = graph.multiplication(meanLoss, lossScale, name: "scaled_loss")
        let lossF32 = computeDataType == .float32
            ? loss
            : graph.cast(loss, to: .float32, name: "loss_f32")

        let grads = graph.gradients(of: loss, with: [m1, m2], name: "grads")
        guard let g1 = grads[m1] else { throw Failure.gradientMissing("d/d master_w1") }
        guard let g2 = grads[m2] else { throw Failure.gradientMissing("d/d master_w2") }
        // Master gradients are already fp32 (autodiff through the cast lands
        // back in the master's dtype); the identity-add gives them a stable
        // readable target name in both dtype arms.
        let zeroW = graph.constant(0.0, shape: weightShape, dataType: .float32)
        let g1F32 = graph.addition(g1, zeroW, name: "grad_w1_f32")
        let g2F32 = graph.addition(g2, zeroW, name: "grad_w2_f32")

        var assignOps: [MPSGraphOperation] = []
        if let lr = sgdLearningRate {
            let lrConst = graph.constant(Double(lr), dataType: .float32)
            for (master, grad) in [(m1, g1), (m2, g2)] {
                let step = graph.multiplication(lrConst, grad, name: nil)
                let updated = graph.subtraction(master, step, name: nil)
                assignOps.append(graph.assign(master, tensor: updated, name: nil))
            }
        }

        let readM1 = graph.addition(m1, zeroW, name: "read_master_w1")
        let readM2 = graph.addition(m2, zeroW, name: "read_master_w2")

        return ConvTower(
            graph: graph,
            inputPlaceholder: inputPH,
            targetPlaceholder: targetPH,
            lossF32: lossF32,
            gradF32: [g1F32, g2F32],
            readMastersF32: [readM1, readM2],
            assignOps: assignOps,
            inputShape: inputShape
        )
    }

    // MARK: - Execution paths

    /// The three step-execution mechanisms under comparison. `graphRun` is
    /// what the trainer used through 2026-06-01; `executable(.level1)` is
    /// what `runPreparedStep` does today; `executable(.level0)` isolates the
    /// optimization level from the run-vs-executable plumbing.
    private enum ExecutionPath: CustomStringConvertible {
        case graphRun
        case executable(MPSGraphOptimization)

        var description: String {
            switch self {
            case .graphRun: return "graph.run"
            case .executable(let level): return "executable(level\(level.rawValue))"
            }
        }
    }

    /// One forward+backward (and, when the tower has assign ops, one SGD
    /// update) through the requested path. Returns the fp32 readbacks of
    /// [loss, gradW1, gradW2] in that order. The executable variant mirrors
    /// `ChessTrainer.runPreparedStep`: compile once (the caller caches the
    /// returned closure's executable via `makeStepRunner`), bind inputs in
    /// `feedTensors` order, encode into an `MPSCommandBuffer`, commit, wait,
    /// and map results by compile-target order.
    private struct StepRunner {
        let run: (_ input: MPSGraphTensorData, _ target: MPSGraphTensorData) throws -> (
            loss: Float, grads: [[Float]]
        )
    }

    private static func makeStepRunner(
        tower: ConvTower,
        path: ExecutionPath,
        device: MTLDevice,
        queue: MTLCommandQueue,
        gradCount: Int
    ) throws -> StepRunner {
        let targets = [tower.lossF32] + tower.gradF32
        let targetOps = tower.assignOps.isEmpty ? nil : tower.assignOps

        func unpack(_ byTensor: [MPSGraphTensor: MPSGraphTensorData]) throws -> (Float, [[Float]]) {
            guard let lossTD = byTensor[tower.lossF32] else {
                throw Failure.gradientMissing("loss result")
            }
            let loss = readBack(lossTD, count: 1)[0]
            var grads: [[Float]] = []
            for g in tower.gradF32 {
                guard let gTD = byTensor[g] else { throw Failure.gradientMissing("grad result") }
                grads.append(readBack(gTD, count: gradCount))
            }
            return (loss, grads)
        }

        switch path {
        case .graphRun:
            return StepRunner(run: { input, target in
                let res = tower.graph.run(
                    with: queue,
                    feeds: [tower.inputPlaceholder: input, tower.targetPlaceholder: target],
                    targetTensors: targets,
                    targetOperations: targetOps
                )
                return try unpack(res)
            })
        case .executable(let level):
            let desc = MPSGraphCompilationDescriptor()
            desc.optimizationLevel = level
            let shapedType = MPSGraphShapedType(shape: tower.inputShape, dataType: .float32)
            let executable = tower.graph.compile(
                with: MPSGraphDevice(mtlDevice: device),
                feeds: [tower.inputPlaceholder: shapedType, tower.targetPlaceholder: shapedType],
                targetTensors: targets,
                targetOperations: targetOps,
                compilationDescriptor: desc
            )
            guard let feedOrder = executable.feedTensors else {
                throw Failure.gradientMissing("executable.feedTensors")
            }
            return StepRunner(run: { input, target in
                let dataByTensor: [MPSGraphTensor: MPSGraphTensorData] = [
                    tower.inputPlaceholder: input,
                    tower.targetPlaceholder: target,
                ]
                var inputs: [MPSGraphTensorData] = []
                inputs.reserveCapacity(feedOrder.count)
                for tensor in feedOrder {
                    guard let data = dataByTensor[tensor] else {
                        throw Failure.gradientMissing("feed binding")
                    }
                    inputs.append(data)
                }
                guard let mtlCB = queue.makeCommandBuffer() else {
                    throw Failure.gradientMissing("command buffer")
                }
                let mpsCB = MPSCommandBuffer(commandBuffer: mtlCB)
                let resultArray = executable.encode(
                    to: mpsCB, inputs: inputs, results: nil, executionDescriptor: nil
                )
                mpsCB.commit()
                mpsCB.waitUntilCompleted()
                let mapped = Dictionary(uniqueKeysWithValues: zip(targets, resultArray))
                return try unpack(mapped)
            })
        }
    }

    // MARK: - Comparison metrics

    private struct DiffStats: CustomStringConvertible {
        let maxAbsDiff: Float
        let meanSignedDiff: Float    // systematic bias — rounding noise alone is ~zero here
        let rmsDiff: Float
        let referenceRMS: Float
        let differingElements: Int
        let totalElements: Int
        /// Non-finite counts on each side. The ULP-adversarial regimes can
        /// legitimately overflow or flush; what matters is whether both
        /// execution paths do so IDENTICALLY.
        let candidateNonFinite: Int
        let referenceNonFinite: Int

        var description: String {
            String(
                format: "maxAbs=%.3e bias=%+.3e rms=%.3e (refRMS=%.3e) differing=%d/%d nonFinite=%d|%d",
                maxAbsDiff, meanSignedDiff, rmsDiff, referenceRMS,
                differingElements, totalElements,
                candidateNonFinite, referenceNonFinite
            )
        }
    }

    private static func compare(_ candidate: [Float], reference: [Float]) -> DiffStats {
        precondition(candidate.count == reference.count, "comparing mismatched tensors")
        var maxAbs: Float = 0
        var sumSigned: Double = 0
        var sumSq: Double = 0
        var refSumSq: Double = 0
        var differing = 0
        var candNonFinite = 0
        var refNonFinite = 0
        for i in 0..<candidate.count {
            if !candidate[i].isFinite { candNonFinite += 1 }
            if !reference[i].isFinite { refNonFinite += 1 }
            let d = candidate[i] - reference[i]
            if candidate[i].bitPattern != reference[i].bitPattern { differing += 1 }
            maxAbs = max(maxAbs, abs(d))
            sumSigned += Double(d)
            sumSq += Double(d) * Double(d)
            refSumSq += Double(reference[i]) * Double(reference[i])
        }
        let n = Double(candidate.count)
        return DiffStats(
            maxAbsDiff: maxAbs,
            meanSignedDiff: Float(sumSigned / n),
            rmsDiff: Float((sumSq / n).squareRoot()),
            referenceRMS: Float((refSumSq / n).squareRoot()),
            differingElements: differing,
            totalElements: candidate.count,
            candidateNonFinite: candNonFinite,
            referenceNonFinite: refNonFinite
        )
    }

    // MARK: - Report plumbing

    private static func writeReport(_ lines: [String]) throws {
        let body = lines.joined(separator: "\n") + "\n"
        print(body)
        try body.write(toFile: reportPath, atomically: true, encoding: .utf8)
    }

    private static func appendReport(_ lines: [String]) throws {
        let body = lines.joined(separator: "\n") + "\n"
        print(body)
        if FileManager.default.fileExists(atPath: reportPath) {
            let existing = try String(contentsOfFile: reportPath, encoding: .utf8)
            try (existing + body).write(toFile: reportPath, atomically: true, encoding: .utf8)
        } else {
            try body.write(toFile: reportPath, atomically: true, encoding: .utf8)
        }
    }

    // MARK: - Test 1: single-step gradients, all arms

    /// For each kernel size and compute dtype, runs ONE identical
    /// forward+backward through all three execution paths and reports each
    /// path's weight gradients against the fp32 `graph.run` reference, plus
    /// the head-to-head `graph.run`-vs-`.level1` comparison that brackets the
    /// 2026-06-02 trainer change.
    func testGradientNumericsAcrossExecutionPathsAndKernels() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let c = Self.channels
        let batch = Self.gradientTestBatch
        let inputCount = batch * c * Self.side * Self.side
        let input = Self.bf16RepresentableValues(count: inputCount, scale: 1.0, seed: 0x9E3779B97F4A7C15)
        let target = Self.bf16RepresentableValues(count: inputCount, scale: 0.5, seed: 0xD1B54A32D192ED03)

        var report: [String] = [
            "=== Conv kernel × execution path gradient numerics ===",
            "channels=\(c) batch=\(batch) board=\(Self.side)×\(Self.side) " +
            "tower=conv(k)→relu→conv(k), fp32 masters, MSE loss",
            "default MPSGraphCompilationDescriptor optimizationLevel raw=" +
            "\(MPSGraphCompilationDescriptor().optimizationLevel.rawValue)",
            "",
        ]

        for kernel in [3, 7] {
            let gradCount = c * c * kernel * kernel
            // He-scaled init (like the real builder) keeps each conv's gain
            // near 1 for BOTH kernel sizes, so the same SGD step size is
            // stable for both; a fixed scale made the 7x7 tower's far larger
            // fan-in blow past the MSE curvature limit and diverge.
            let heScale = (2.0 / Float(c * kernel * kernel)).squareRoot()
            let w1 = Self.bf16RepresentableValues(count: gradCount, scale: heScale, seed: 0xA5A5_0000 + UInt64(kernel))
            let w2 = Self.bf16RepresentableValues(count: gradCount, scale: heScale, seed: 0x5A5A_0000 + UInt64(kernel))

            // fp32 graph.run is the precision reference for every arm.
            var resultsByArm: [String: (loss: Float, grads: [[Float]])] = [:]
            for dtype in [MPSDataType.float32, .bFloat16] {
                for path in [ExecutionPath.graphRun, .executable(.level0), .executable(.level1)] {
                    let tower = try Self.buildConvTower(
                        kernel: kernel, computeDataType: dtype, batch: batch,
                        masterW1: w1, masterW2: w2, sgdLearningRate: nil
                    )
                    let runner = try Self.makeStepRunner(
                        tower: tower, path: path, device: device, queue: queue, gradCount: gradCount
                    )
                    let inputTD = Self.tensorData(device, input, shape: tower.inputShape)
                    let targetTD = Self.tensorData(device, target, shape: tower.inputShape)
                    let result = try runner.run(inputTD, targetTD)

                    XCTAssertTrue(result.loss.isFinite, "k=\(kernel) \(dtype) \(path): loss not finite")
                    for (gi, g) in result.grads.enumerated() {
                        XCTAssertTrue(g.allSatisfy { $0.isFinite },
                                      "k=\(kernel) \(dtype) \(path): grad[\(gi)] has non-finite values")
                        XCTAssertTrue(g.contains { $0 != 0 },
                                      "k=\(kernel) \(dtype) \(path): grad[\(gi)] is all zeros")
                    }
                    let dtypeName = dtype == .float32 ? "fp32" : "bf16"
                    resultsByArm["\(dtypeName)/\(path)"] = result
                }
            }

            guard let reference = resultsByArm["fp32/graph.run"] else {
                XCTFail("missing fp32 reference arm"); return
            }
            report.append("--- kernel \(kernel)×\(kernel) ---")
            report.append(String(format: "fp32/graph.run loss=%.8e (reference)", reference.loss))
            let armOrder = [
                "fp32/executable(level0)", "fp32/executable(level1)",
                "bf16/graph.run", "bf16/executable(level0)", "bf16/executable(level1)",
            ]
            for arm in armOrder {
                guard let r = resultsByArm[arm] else { continue }
                report.append(String(format: "%@ loss=%.8e", arm, r.loss))
                for gi in 0..<r.grads.count {
                    let stats = Self.compare(r.grads[gi], reference: reference.grads[gi])
                    report.append("  vs fp32 ref, gradW\(gi + 1): \(stats)")
                }
            }
            // The head-to-head that brackets the 2026-06-02 change.
            if let runArm = resultsByArm["bf16/graph.run"],
               let exeArm = resultsByArm["bf16/executable(level1)"] {
                report.append("  HEAD-TO-HEAD bf16 graph.run vs executable(level1):")
                for gi in 0..<runArm.grads.count {
                    let stats = Self.compare(exeArm.grads[gi], reference: runArm.grads[gi])
                    report.append("    gradW\(gi + 1): \(stats)")
                }
            }
            report.append("")
        }

        try Self.writeReport(report)
        XCTAssertTrue(FileManager.default.fileExists(atPath: Self.reportPath),
                      "measurement report missing at \(Self.reportPath)")
    }

    // MARK: - Test 2: repeated-step weight drift

    /// Runs the same SGD trajectory (fp32 master update, fixed learning
    /// rate, a deterministic cycle of batches) through `graph.run` and the
    /// `.level1` executable for each kernel size, reading the fp32 masters at
    /// checkpoints. A systematic per-step gradient difference compounds into
    /// monotonically growing weight divergence; symmetric rounding noise
    /// random-walks far more slowly. The fp32 `graph.run` trajectory is
    /// recorded alongside as the scale yardstick.
    func testRepeatedStepWeightDriftGraphRunVsLevel1Executable() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let c = Self.channels
        let batch = 128
        let steps = 100
        let checkpoints: Set<Int> = [1, 5, 25, 50, 100]
        let batchCycle = 8
        // Sized against the tower's ×256-scaled loss: gradient RMS lands
        // near the weight scale itself, so the rate must keep the cumulative
        // 100-step movement well inside the weights' magnitude or the
        // trajectory diverges (an early revision used a rate ~500× larger
        // and reached NaN within five steps).
        let learningRate: Float = 0.002

        let inputCount = batch * c * Self.side * Self.side
        let inputs: [[Float]] = (0..<batchCycle).map { i in
            Self.bf16RepresentableValues(count: inputCount, scale: 1.0, seed: 0xBEEF_0000 + UInt64(i))
        }
        let targets: [[Float]] = (0..<batchCycle).map { i in
            Self.bf16RepresentableValues(count: inputCount, scale: 0.5, seed: 0xFACE_0000 + UInt64(i))
        }

        var report: [String] = [
            "=== Repeated-step weight drift: graph.run vs executable(level1) ===",
            "channels=\(c) batch=\(batch) steps=\(steps) lr=\(learningRate) " +
            "batchCycle=\(batchCycle), fp32 master SGD, masters read at checkpoints",
            "",
        ]

        for kernel in [3, 7] {
            let gradCount = c * c * kernel * kernel
            // He-scaled init (like the real builder) keeps each conv's gain
            // near 1 for BOTH kernel sizes, so the same SGD step size is
            // stable for both; a fixed scale made the 7x7 tower's far larger
            // fan-in blow past the MSE curvature limit and diverge.
            let heScale = (2.0 / Float(c * kernel * kernel)).squareRoot()
            let w1 = Self.bf16RepresentableValues(count: gradCount, scale: heScale, seed: 0xA5A5_0000 + UInt64(kernel))
            let w2 = Self.bf16RepresentableValues(count: gradCount, scale: heScale, seed: 0x5A5A_0000 + UInt64(kernel))

            struct Arm {
                let name: String
                let tower: ConvTower
                let runner: StepRunner
                var masterTrajectory: [Int: [[Float]]] = [:]
            }
            func makeArm(name: String, dtype: MPSDataType, path: ExecutionPath) throws -> Arm {
                let tower = try Self.buildConvTower(
                    kernel: kernel, computeDataType: dtype, batch: batch,
                    masterW1: w1, masterW2: w2, sgdLearningRate: learningRate
                )
                let runner = try Self.makeStepRunner(
                    tower: tower, path: path, device: device, queue: queue, gradCount: gradCount
                )
                return Arm(name: name, tower: tower, runner: runner)
            }
            var arms: [Arm] = [
                try makeArm(name: "bf16/graph.run", dtype: .bFloat16, path: .graphRun),
                try makeArm(name: "bf16/executable(level1)", dtype: .bFloat16, path: .executable(.level1)),
                try makeArm(name: "fp32/graph.run", dtype: .float32, path: .graphRun),
            ]

            for step in 1...steps {
                let batchIndex = (step - 1) % batchCycle
                for armIndex in arms.indices {
                    let arm = arms[armIndex]
                    let inputTD = Self.tensorData(device, inputs[batchIndex], shape: arm.tower.inputShape)
                    let targetTD = Self.tensorData(device, targets[batchIndex], shape: arm.tower.inputShape)
                    let result = try arm.runner.run(inputTD, targetTD)
                    guard result.loss.isFinite else {
                        // Fail fast: once one arm diverges, every later
                        // checkpoint is NaN spam and the drift comparison is
                        // meaningless.
                        XCTFail("k=\(kernel) \(arm.name) step \(step): loss not finite (\(result.loss)) — trajectory diverged; lower the test learning rate")
                        return
                    }
                    if checkpoints.contains(step) {
                        // Read the masters in a separate run — assign→read
                        // within one run is unordered (see the
                        // variable-semantics tests).
                        let read = arm.tower.graph.run(
                            with: queue, feeds: [:],
                            targetTensors: arm.tower.readMastersF32, targetOperations: nil
                        )
                        var masters: [[Float]] = []
                        for tensor in arm.tower.readMastersF32 {
                            guard let td = read[tensor] else {
                                XCTFail("k=\(kernel) \(arm.name): master readback missing"); return
                            }
                            masters.append(Self.readBack(td, count: gradCount))
                        }
                        arms[armIndex].masterTrajectory[step] = masters
                    }
                }
            }

            report.append("--- kernel \(kernel)×\(kernel) ---")
            guard let runArm = arms.first(where: { $0.name == "bf16/graph.run" }),
                  let exeArm = arms.first(where: { $0.name == "bf16/executable(level1)" }),
                  let refArm = arms.first(where: { $0.name == "fp32/graph.run" }) else {
                XCTFail("drift arms missing"); return
            }
            for step in checkpoints.sorted() {
                guard let a = runArm.masterTrajectory[step],
                      let b = exeArm.masterTrajectory[step],
                      let ref = refArm.masterTrajectory[step] else { continue }
                for gi in 0..<a.count {
                    let pathDiff = Self.compare(b[gi], reference: a[gi])
                    let bf16VsFp32 = Self.compare(a[gi], reference: ref[gi])
                    report.append(
                        "step \(step) masterW\(gi + 1): run-vs-level1 \(pathDiff)"
                    )
                    report.append(
                        "                 bf16-vs-fp32 \(bf16VsFp32)   (precision-scale yardstick)"
                    )
                }
            }
            report.append("")
        }

        try Self.appendReport(report)
    }

    // MARK: - Test 3: production batch size

    /// Closes the size caveat from the class doc: MPS kernel selection
    /// depends on problem size, and the live trainer runs batch 4096. Same
    /// single-step comparison as the first test, at the production batch.
    /// Only the bf16 arms get all three paths (the live regime); fp32 is the
    /// `graph.run` precision reference.
    func testGradientNumericsAtProductionBatchSize() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let c = Self.channels
        let batch = 4096
        let inputCount = batch * c * Self.side * Self.side
        let input = Self.bf16RepresentableValues(count: inputCount, scale: 1.0, seed: 0x9E3779B97F4A7C15)
        let target = Self.bf16RepresentableValues(count: inputCount, scale: 0.5, seed: 0xD1B54A32D192ED03)

        var report: [String] = [
            "=== Production batch size: gradient numerics, batch=\(batch) ===",
            "channels=\(c) board=\(Self.side)×\(Self.side), same tower as the single-step test",
            "",
        ]

        for kernel in [3, 7] {
            let gradCount = c * c * kernel * kernel
            let heScale = (2.0 / Float(c * kernel * kernel)).squareRoot()
            let w1 = Self.bf16RepresentableValues(count: gradCount, scale: heScale, seed: 0xA5A5_0000 + UInt64(kernel))
            let w2 = Self.bf16RepresentableValues(count: gradCount, scale: heScale, seed: 0x5A5A_0000 + UInt64(kernel))

            var resultsByArm: [String: (loss: Float, grads: [[Float]])] = [:]
            let arms: [(dtype: MPSDataType, path: ExecutionPath)] = [
                (.float32, .graphRun),
                (.bFloat16, .graphRun),
                (.bFloat16, .executable(.level0)),
                (.bFloat16, .executable(.level1)),
            ]
            for (dtype, path) in arms {
                let tower = try Self.buildConvTower(
                    kernel: kernel, computeDataType: dtype, batch: batch,
                    masterW1: w1, masterW2: w2, sgdLearningRate: nil
                )
                let runner = try Self.makeStepRunner(
                    tower: tower, path: path, device: device, queue: queue, gradCount: gradCount
                )
                let inputTD = Self.tensorData(device, input, shape: tower.inputShape)
                let targetTD = Self.tensorData(device, target, shape: tower.inputShape)
                let result = try runner.run(inputTD, targetTD)

                XCTAssertTrue(result.loss.isFinite, "k=\(kernel) \(dtype) \(path) batch=\(batch): loss not finite")
                for (gi, g) in result.grads.enumerated() {
                    XCTAssertTrue(g.allSatisfy { $0.isFinite },
                                  "k=\(kernel) \(dtype) \(path) batch=\(batch): grad[\(gi)] has non-finite values")
                    XCTAssertTrue(g.contains { $0 != 0 },
                                  "k=\(kernel) \(dtype) \(path) batch=\(batch): grad[\(gi)] is all zeros")
                }
                let dtypeName = dtype == .float32 ? "fp32" : "bf16"
                resultsByArm["\(dtypeName)/\(path)"] = result
            }

            guard let reference = resultsByArm["fp32/graph.run"] else {
                XCTFail("missing fp32 reference arm"); return
            }
            report.append("--- kernel \(kernel)×\(kernel) batch=\(batch) ---")
            report.append(String(format: "fp32/graph.run loss=%.8e (reference)", reference.loss))
            for arm in ["bf16/graph.run", "bf16/executable(level0)", "bf16/executable(level1)"] {
                guard let r = resultsByArm[arm] else { continue }
                report.append(String(format: "%@ loss=%.8e", arm, r.loss))
                for gi in 0..<r.grads.count {
                    let stats = Self.compare(r.grads[gi], reference: reference.grads[gi])
                    report.append("  vs fp32 ref, gradW\(gi + 1): \(stats)")
                }
            }
            if let runArm = resultsByArm["bf16/graph.run"],
               let exeArm = resultsByArm["bf16/executable(level1)"] {
                report.append("  HEAD-TO-HEAD bf16 graph.run vs executable(level1) @ batch \(batch):")
                for gi in 0..<runArm.grads.count {
                    let stats = Self.compare(exeArm.grads[gi], reference: runArm.grads[gi])
                    report.append("    gradW\(gi + 1): \(stats)")
                }
            }
            report.append("")
        }

        try Self.appendReport(report)
    }

    // MARK: - Test 4: bf16 ULP-boundary adversarial regimes

    /// Stress regimes where two different conv implementations are FORCED to
    /// disagree if they differ anywhere — rounding direction, accumulation
    /// order, accumulator precision, or flush-to-zero behavior:
    ///
    /// - `tieJustBelowHalfUlp` / `tieExactHalfUlp` / `tieJustAboveHalfUlp`:
    ///   every input and weight sits a hair below / exactly at / a hair above
    ///   the bf16 round-to-nearest boundary, so the in-graph fp32→bf16 casts
    ///   (and any `.level1` cast→conv fusion) must round bit-identically or
    ///   the gradients diverge wholesale.
    /// - `mixedExponentCancellation`: magnitudes spread over thirteen binades
    ///   with random signs — accumulation-order differences become visible
    ///   through catastrophic cancellation.
    /// - `extremeExponentProducts`: first-conv weights scaled enormous and
    ///   inputs scaled tiny so every multiply spans the exponent range while
    ///   products land near unity.
    /// - `nearSubnormalGradients`: inputs and first-conv weights scaled so
    ///   products sit at the fp32/bf16 minimum-normal edge — accumulator
    ///   precision and flush-to-zero policy differences show here first.
    ///
    /// Pure measurement: per-regime gradients are compared head-to-head
    /// across `graph.run`, `executable(.level0)` and `executable(.level1)`,
    /// including non-finite counts (identical overflow/flush behavior is the
    /// "same kernel" signature; divergent behavior is the finding). No
    /// numeric assertions — adversarial regimes may legitimately flush to
    /// zero or overflow, and they must do so identically to pass judgement,
    /// which the report shows.
    func testUlpBoundaryAdversarialGradients() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let c = Self.channels
        let batch = Self.gradientTestBatch
        let inputCount = batch * c * Self.side * Self.side

        struct Regime {
            let name: String
            let input: (Int) -> [Float]      // count -> values
            let w1: (Int, Float) -> [Float]  // count, heScale -> values
            let w2: (Int, Float) -> [Float]
            let target: (Int) -> [Float]
        }
        let regimes: [Regime] = [
            Regime(
                name: "tieJustBelowHalfUlp",
                input: { Self.bf16UlpBoundaryValues(count: $0, scale: 1.0, halfUlpFraction: 0.498, seed: 0x1111_0001) },
                w1: { Self.bf16UlpBoundaryValues(count: $0, scale: $1, halfUlpFraction: 0.498, seed: 0x1111_0002) },
                w2: { Self.bf16UlpBoundaryValues(count: $0, scale: $1, halfUlpFraction: 0.498, seed: 0x1111_0003) },
                target: { Self.bf16RepresentableValues(count: $0, scale: 0.5, seed: 0x1111_0004) }
            ),
            Regime(
                name: "tieExactHalfUlp",
                input: { Self.bf16UlpBoundaryValues(count: $0, scale: 1.0, halfUlpFraction: 0.5, seed: 0x2222_0001) },
                w1: { Self.bf16UlpBoundaryValues(count: $0, scale: $1, halfUlpFraction: 0.5, seed: 0x2222_0002) },
                w2: { Self.bf16UlpBoundaryValues(count: $0, scale: $1, halfUlpFraction: 0.5, seed: 0x2222_0003) },
                target: { Self.bf16RepresentableValues(count: $0, scale: 0.5, seed: 0x2222_0004) }
            ),
            Regime(
                name: "tieJustAboveHalfUlp",
                input: { Self.bf16UlpBoundaryValues(count: $0, scale: 1.0, halfUlpFraction: 0.502, seed: 0x3333_0001) },
                w1: { Self.bf16UlpBoundaryValues(count: $0, scale: $1, halfUlpFraction: 0.502, seed: 0x3333_0002) },
                w2: { Self.bf16UlpBoundaryValues(count: $0, scale: $1, halfUlpFraction: 0.502, seed: 0x3333_0003) },
                target: { Self.bf16RepresentableValues(count: $0, scale: 0.5, seed: 0x3333_0004) }
            ),
            Regime(
                name: "mixedExponentCancellation",
                input: { Self.mixedExponentValues(count: $0, scale: 1.0, seed: 0x4444_0001) },
                w1: { Self.mixedExponentValues(count: $0, scale: $1, seed: 0x4444_0002) },
                w2: { Self.mixedExponentValues(count: $0, scale: $1, seed: 0x4444_0003) },
                target: { Self.bf16RepresentableValues(count: $0, scale: 0.5, seed: 0x4444_0004) }
            ),
            Regime(
                name: "extremeExponentProducts",
                input: { Self.bf16RepresentableValues(count: $0, scale: 1e-18, seed: 0x5555_0001) },
                w1: { Self.bf16RepresentableValues(count: $0, scale: $1 * 1e18, seed: 0x5555_0002) },
                w2: { Self.bf16RepresentableValues(count: $0, scale: $1, seed: 0x5555_0003) },
                target: { Self.bf16RepresentableValues(count: $0, scale: 0.5, seed: 0x5555_0004) }
            ),
            Regime(
                name: "nearSubnormalGradients",
                input: { Self.bf16RepresentableValues(count: $0, scale: 1e-19, seed: 0x6666_0001) },
                w1: { Self.bf16RepresentableValues(count: $0, scale: $1 * 1e-19, seed: 0x6666_0002) },
                w2: { Self.bf16RepresentableValues(count: $0, scale: $1, seed: 0x6666_0003) },
                target: { Self.bf16RepresentableValues(count: $0, scale: 0.5, seed: 0x6666_0004) }
            ),
        ]

        var report: [String] = [
            "=== bf16 ULP-boundary adversarial regimes (batch=\(batch)) ===",
            "head-to-head across paths; identical fp32 inputs to every path; bf16 compute",
            "",
        ]

        for regime in regimes {
            let input = regime.input(inputCount)
            let target = regime.target(inputCount)
            for kernel in [3, 7] {
                let gradCount = c * c * kernel * kernel
                let heScale = (2.0 / Float(c * kernel * kernel)).squareRoot()
                let w1 = regime.w1(gradCount, heScale)
                let w2 = regime.w2(gradCount, heScale)

                var resultsByPath: [String: (loss: Float, grads: [[Float]])] = [:]
                for path in [ExecutionPath.graphRun, .executable(.level0), .executable(.level1)] {
                    let tower = try Self.buildConvTower(
                        kernel: kernel, computeDataType: .bFloat16, batch: batch,
                        masterW1: w1, masterW2: w2, sgdLearningRate: nil
                    )
                    let runner = try Self.makeStepRunner(
                        tower: tower, path: path, device: device, queue: queue, gradCount: gradCount
                    )
                    let inputTD = Self.tensorData(device, input, shape: tower.inputShape)
                    let targetTD = Self.tensorData(device, target, shape: tower.inputShape)
                    resultsByPath["\(path)"] = try runner.run(inputTD, targetTD)
                }

                guard let runResult = resultsByPath["graph.run"] else {
                    XCTFail("\(regime.name) k=\(kernel): graph.run arm missing"); return
                }
                report.append("--- \(regime.name), kernel \(kernel)×\(kernel) ---")
                report.append(String(
                    format: "loss: run=%.8e level0=%.8e level1=%.8e",
                    runResult.loss,
                    resultsByPath["executable(level0)"]?.loss ?? .nan,
                    resultsByPath["executable(level1)"]?.loss ?? .nan
                ))
                for candidate in ["executable(level0)", "executable(level1)"] {
                    guard let r = resultsByPath[candidate] else { continue }
                    for gi in 0..<r.grads.count {
                        let stats = Self.compare(r.grads[gi], reference: runResult.grads[gi])
                        report.append("  \(candidate) vs run, gradW\(gi + 1): \(stats)")
                    }
                }
                report.append("")
            }
        }

        try Self.appendReport(report)
        XCTAssertTrue(FileManager.default.fileExists(atPath: Self.reportPath),
                      "measurement report missing at \(Self.reportPath)")
    }
}
