//
//  DropoutGraphSemanticsTests.swift
//  DrewsChessMachineTests
//
//  Verifies the MPSGraph op construction behind the per-block channel
//  dropout in `ChessNetwork` (training-mode graphs only): a fp32 rate
//  VARIABLE (set out-of-band via placeholder+assign), a Philox RNG state
//  VARIABLE (seeded once, advanced per step), a per-draw uniform tensor of
//  shape [N, C, 1, 1] whose shape tensor derives from the INPUT
//  placeholder (never from a tensor downstream of trainables — attaching
//  `shapeOf` to the trainable path breaks autodiff with "Couldn't get
//  gradient Tensor", the 2026-06-12 crash), a keep-mask `u >= rate`, and
//  inverted scaling `1/(1-rate)`.
//
//  Invariants under test:
//   1. rate 0 is an EXACT identity (mask all-ones, scale exactly 1.0).
//   2. nonzero rate masks whole channel slabs (never pinholes) and scales
//      survivors by exactly 1/(1-rate); consecutive draws differ once the
//      state-advance assign runs.
//   3. inverted scaling preserves the expectation: the average over many
//      draws approaches the unmasked input.
//   4. autodiff works through the masked multiply, and the gradient
//      w.r.t. the masked tensor equals the scaled mask (the mask branch
//      contributes no gradient path of its own).
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class DropoutGraphSemanticsTests: XCTestCase {

    // Small NCHW geometry: batch 4, 8 channels, 2×2 board. Spatial > 1 so
    // channel-slab masking is distinguishable from per-element masking.
    private let n = 4, c = 8, h = 2, w = 2
    private var elementCount: Int { n * c * h * w }

    /// Everything one test needs to drive the dropout subgraph.
    private struct Harness {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let graph: MPSGraph
        let input: MPSGraphTensor          // placeholder [n, c, h, w] fp32
        let output: MPSGraphTensor         // input × scaledMask
        let scaledMask: MPSGraphTensor     // mask × 1/(1−rate), [n, c, 1, 1]
        let rateVariable: MPSGraphTensor
        let rateLoadPlaceholder: MPSGraphTensor
        let rateAssign: MPSGraphOperation
        let seedOp: MPSGraphOperation
        let advanceOp: MPSGraphOperation
        let inputTensorData: MPSGraphTensorData
        let rateTensorData: MPSGraphTensorData
        let rateNDArray: MPSNDArray
    }

    private func makeHarness(inputValues: [Float]) throws -> Harness {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let queue = device.makeCommandQueue()!
        let g = MPSGraph()

        let input = g.placeholder(
            shape: [NSNumber(value: n), NSNumber(value: c), NSNumber(value: h), NSNumber(value: w)],
            dataType: .float32, name: "input"
        )

        // Rate variable + out-of-band setter (mirrors ChessNetwork).
        let rateVar = g.variable(
            with: withUnsafeBytes(of: Float(0)) { Data($0) },
            shape: [1], dataType: .float32, name: "dropout_rate"
        )
        let ratePh = g.placeholder(shape: [1], dataType: .float32, name: "dropout_rate_load")
        let rateAssign = g.assign(rateVar, tensor: ratePh, name: "dropout_rate_load_assign")

        // RNG state variable + seed (mirrors ChessNetwork; fixed seed so
        // test failures reproduce).
        let stateVar = g.variable(
            with: Data(count: 7 * MemoryLayout<Int32>.size),
            shape: [7], dataType: .int32, name: "dropout_rng_state"
        )
        let seeded = g.randomPhiloxStateTensor(withSeed: 0x5EED, name: "dropout_rng_seed")
        let seedOp = g.assign(stateVar, tensor: seeded, name: "dropout_rng_seed_assign")

        // Mask shape [N, C, 1, 1] from the INPUT placeholder's shape.
        let inputShape = g.shapeOf(input, name: "dropout_input_shape")
        let batchOnly = g.sliceTensor(inputShape, dimension: 0, start: 0, length: 1, name: "n_only")
        let channelDim = g.constant(Double(c), shape: [1], dataType: .int32)
        let spatialOnes = g.constant(1.0, shape: [2], dataType: .int32)
        let maskShape = g.concatTensors([batchOnly, channelDim, spatialOnes], dimension: 0, name: "mask_shape")

        guard let desc = MPSGraphRandomOpDescriptor(distribution: .uniform, dataType: .float32) else {
            throw XCTSkip("MPSGraphRandomOpDescriptor unavailable")
        }
        let drawn = g.randomTensor(
            withShapeTensor: maskShape, descriptor: desc,
            stateTensor: stateVar, name: "dropout_rng"
        )
        let advanceOp = g.assign(stateVar, tensor: drawn[1], name: "dropout_rng_advance")

        let keep = g.greaterThanOrEqualTo(drawn[0], rateVar, name: "keep")
        let maskF = g.cast(keep, to: .float32, name: "mask_f")
        let one = g.constant(1.0, shape: [1], dataType: .float32)
        let scale = g.division(one, g.subtraction(one, rateVar, name: "keep_frac"), name: "scale")
        let scaledMask = g.multiplication(maskF, scale, name: "scaled_mask")
        let output = g.multiplication(input, scaledMask, name: "output")

        // Feeds.
        let inDesc = MPSNDArrayDescriptor(
            dataType: .float32,
            shape: [NSNumber(value: n), NSNumber(value: c), NSNumber(value: h), NSNumber(value: w)]
        )
        let inNDA = MPSNDArray(device: device, descriptor: inDesc)
        var inVals = inputValues
        XCTAssertEqual(inVals.count, elementCount)
        inVals.withUnsafeMutableBytes { inNDA.writeBytes($0.baseAddress!, strideBytes: nil) }
        let rateDesc = MPSNDArrayDescriptor(dataType: .float32, shape: [1])
        let rateNDA = MPSNDArray(device: device, descriptor: rateDesc)

        return Harness(
            device: device, queue: queue, graph: g,
            input: input, output: output, scaledMask: scaledMask,
            rateVariable: rateVar, rateLoadPlaceholder: ratePh,
            rateAssign: rateAssign, seedOp: seedOp, advanceOp: advanceOp,
            inputTensorData: MPSGraphTensorData(inNDA),
            rateTensorData: MPSGraphTensorData(rateNDA),
            rateNDArray: rateNDA
        )
    }

    private func seed(_ hx: Harness) {
        _ = hx.graph.run(
            with: hx.queue,
            feeds: [hx.input: hx.inputTensorData],
            targetTensors: [hx.rateVariable],
            targetOperations: [hx.seedOp]
        )
    }

    private func setRate(_ hx: Harness, _ rate: Float) {
        var v = rate
        hx.rateNDArray.writeBytes(&v, strideBytes: nil)
        _ = hx.graph.run(
            with: hx.queue,
            feeds: [hx.input: hx.inputTensorData, hx.rateLoadPlaceholder: hx.rateTensorData],
            targetTensors: [hx.rateVariable],
            targetOperations: [hx.rateAssign]
        )
    }

    /// One masked forward; advances the RNG state (as a training step does).
    private func runOutput(_ hx: Harness) -> [Float] {
        let results = hx.graph.run(
            with: hx.queue,
            feeds: [hx.input: hx.inputTensorData],
            targetTensors: [hx.output],
            targetOperations: [hx.advanceOp]
        )
        return readFloats(results[hx.output]!, count: elementCount)
    }

    private func readFloats(_ td: MPSGraphTensorData, count: Int) -> [Float] {
        var out = [Float](repeating: .nan, count: count)
        out.withUnsafeMutableBytes { td.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return out
    }

    // MARK: 1. rate 0 — exact identity

    func testRateZeroIsExactIdentity() throws {
        // Distinct values everywhere so any perturbation is visible.
        let inputValues = (0..<elementCount).map { Float($0) * 0.37 - 11.0 }
        let hx = try makeHarness(inputValues: inputValues)
        seed(hx)
        // Two draws: the RNG advances between them, but at rate 0 every
        // mask is all-ones and the scale is exactly 1.0 — bit-identical
        // output both times.
        for pass in 0..<2 {
            let out = runOutput(hx)
            for i in 0..<elementCount {
                XCTAssertEqual(
                    out[i], inputValues[i],
                    "rate-0 output diverged from input at flat index \(i) (pass \(pass))"
                )
            }
        }
    }

    // MARK: 2. nonzero rate — channel slabs, exact scale, fresh masks

    func testHalfRateMasksWholeChannelsAndScalesSurvivors() throws {
        let inputValues = [Float](repeating: 1.0, count: elementCount)
        let hx = try makeHarness(inputValues: inputValues)
        seed(hx)
        setRate(hx, 0.5)

        let spatial = h * w
        let out1 = runOutput(hx)
        var droppedSlabs = 0
        var keptSlabs = 0
        for slab in 0..<(n * c) {
            let vals = Array(out1[(slab * spatial)..<((slab + 1) * spatial)])
            if vals.allSatisfy({ $0 == 0.0 }) {
                droppedSlabs += 1
            } else if vals.allSatisfy({ $0 == 2.0 }) {
                // survivors scaled by exactly 1/(1−0.5) = 2.0
                keptSlabs += 1
            } else {
                XCTFail("channel slab \(slab) is neither all-dropped nor all-kept: \(vals)")
            }
        }
        // P(all 32 slabs land the same way) = 2 × 2^-32 — effectively
        // impossible with a healthy uniform draw.
        XCTAssertGreaterThan(droppedSlabs, 0, "rate 0.5 dropped no channels — mask inert?")
        XCTAssertGreaterThan(keptSlabs, 0, "rate 0.5 dropped every channel")

        // The advance op ran, so the next draw must differ somewhere
        // (P(identical masks) = 2^-32).
        let out2 = runOutput(hx)
        XCTAssertNotEqual(out1, out2, "consecutive draws produced identical masks — RNG state not advancing")
    }

    // MARK: 3. inverted scaling preserves the expectation

    func testInvertedScalingPreservesExpectation() throws {
        let inputValues = [Float](repeating: 1.0, count: elementCount)
        let hx = try makeHarness(inputValues: inputValues)
        seed(hx)
        setRate(hx, 0.3)

        let draws = 300
        var sums = [Double](repeating: 0, count: elementCount)
        for _ in 0..<draws {
            let out = runOutput(hx)
            for i in 0..<elementCount { sums[i] += Double(out[i]) }
        }
        // Per-element mean should approach 1.0. Per-draw std at rate 0.3
        // is √(r/(1−r)) ≈ 0.655, so the 300-draw standard error is ≈ 0.038;
        // a ±0.2 tolerance is > 5σ — failures mean broken scaling, not luck.
        for i in 0..<elementCount {
            let mean = sums[i] / Double(draws)
            XCTAssertEqual(mean, 1.0, accuracy: 0.2, "expectation drifted at flat index \(i): \(mean)")
        }
    }

    // MARK: 4. autodiff through the masked multiply

    func testGradientThroughMaskEqualsScaledMask() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let queue = device.makeCommandQueue()!
        let g = MPSGraph()

        // The masked tensor is a VARIABLE here so we can ask for its
        // gradient — standing in for the conv activations the in-app
        // dropout masks.
        let xCount = n * c * h * w
        let xInit = [Float](repeating: 1.0, count: xCount)
        let xData = xInit.withUnsafeBufferPointer { Data(buffer: $0) }
        let x = g.variable(
            with: xData,
            shape: [NSNumber(value: n), NSNumber(value: c), NSNumber(value: h), NSNumber(value: w)],
            dataType: .float32, name: "x"
        )
        let rateVar = g.variable(
            with: withUnsafeBytes(of: Float(0.5)) { Data($0) },
            shape: [1], dataType: .float32, name: "rate"
        )
        let stateVar = g.variable(
            with: Data(count: 7 * MemoryLayout<Int32>.size),
            shape: [7], dataType: .int32, name: "state"
        )
        let seeded = g.randomPhiloxStateTensor(withSeed: 0xD1CE, name: "seed")
        let seedOp = g.assign(stateVar, tensor: seeded, name: "seed_assign")

        guard let desc = MPSGraphRandomOpDescriptor(distribution: .uniform, dataType: .float32) else {
            throw XCTSkip("MPSGraphRandomOpDescriptor unavailable")
        }
        // Constant mask-shape tensor [n, c, 1, 1] — nothing trainable
        // upstream of a constant, so this also documents the safe
        // construction (the crash mode was shapeOf on the trainable path).
        let shapeValues: [Int32] = [Int32(n), Int32(c), 1, 1]
        let maskShape = g.constant(
            shapeValues.withUnsafeBufferPointer { Data(buffer: $0) },
            shape: [4], dataType: .int32
        )
        let drawn = g.randomTensor(
            withShapeTensor: maskShape,
            descriptor: desc, stateTensor: stateVar, name: "rng"
        )
        let keep = g.greaterThanOrEqualTo(drawn[0], rateVar, name: "keep")
        let maskF = g.cast(keep, to: .float32, name: "mask_f")
        let one = g.constant(1.0, shape: [1], dataType: .float32)
        let scale = g.division(one, g.subtraction(one, rateVar, name: "kf"), name: "scale")
        let scaledMask = g.multiplication(maskF, scale, name: "scaled_mask")
        let out = g.multiplication(x, scaledMask, name: "out")
        let loss = g.reductionSum(with: g.reshape(out, shape: [-1], name: "flat"), axis: 0, name: "loss")

        // Seed the RNG state first.
        _ = g.run(with: queue, feeds: [:], targetTensors: [stateVar], targetOperations: [seedOp])

        // d(loss)/d(x) must exist (the 2026-06-12 crash mode was autodiff
        // failing to produce gradients when the mask branch was wired
        // wrong) and must equal the scaled mask broadcast over the board.
        let grads = g.gradients(of: loss, with: [x], name: "grads")
        guard let gradX = grads[x] else {
            XCTFail("autodiff produced no gradient for the masked tensor")
            return
        }
        let results = g.run(
            with: queue, feeds: [:],
            targetTensors: [gradX, scaledMask], targetOperations: nil
        )
        let gradVals = readFloats(results[gradX]!, count: xCount)
        let maskVals = readFloats(results[scaledMask]!, count: n * c)
        let spatial = h * w
        for slab in 0..<(n * c) {
            for s in 0..<spatial {
                XCTAssertEqual(
                    gradVals[slab * spatial + s], maskVals[slab],
                    "∂loss/∂x at slab \(slab) elem \(s) ≠ scaled mask value"
                )
            }
        }
    }
}
