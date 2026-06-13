//
//  DropoutMultiplierSemanticsTests.swift
//  DrewsChessMachineTests
//
//  Numeric semantics of the PER-GROUP dropout multiplier
//  (ARCHITECTURE_EXPANSION_PLAN.md Feature 2): each block composes the
//  global live rate variable with its group's baked multiplier constant,
//  `effective = min(rate × multiplier, 0.95)`, and BOTH the keep-compare
//  and the inverted 1/(1−rate) survivor scale read the composed rate.
//  The subgraph here mirrors `ChessNetwork.residualBlock`'s
//  `applyChannelDropout` construction line-for-line (multiplier branch
//  included) — if that construction changes, change this harness in
//  lockstep.
//
//  Invariants under test:
//   1. multiplier 0 is an EXACT identity at ANY rate — the documented
//      "exempt this group" knob (0 × rate = 0 → mask all-ones, scale 1).
//   2. a fractional multiplier scales the effective rate: at rate 0.7 ×
//      multiplier 0.5, the empirical drop fraction ≈ 0.35 and survivors
//      are scaled by exactly 1/(1 − 0.35) (same fp32 composition).
//   3. the 0.95 cap: rate 0.8 × multiplier 2 clamps to 0.95, keeping the
//      inverted scale finite (= 1/0.05, not 1/−0.6).
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class DropoutMultiplierSemanticsTests: XCTestCase {

    // Small NCHW geometry; mask is per (sample, channel) → n*c coins/draw.
    private let n = 4, c = 8, h = 2, w = 2
    private var elementCount: Int { n * c * h * w }

    private struct Harness {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let graph: MPSGraph
        let input: MPSGraphTensor
        let output: MPSGraphTensor
        let rateVariable: MPSGraphTensor
        let rateLoadPlaceholder: MPSGraphTensor
        let rateAssign: MPSGraphOperation
        let seedOp: MPSGraphOperation
        let advanceOp: MPSGraphOperation
        let inputTensorData: MPSGraphTensorData
        let rateTensorData: MPSGraphTensorData
        let rateNDArray: MPSNDArray
    }

    /// Mirrors `residualBlock.applyChannelDropout` with a group multiplier:
    /// rate variable → [× multiplier constant → min 0.95] → keep/scale.
    private func makeHarness(multiplier: Float, inputValues: [Float]) throws -> Harness {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let queue = device.makeCommandQueue()!
        let g = MPSGraph()

        let input = g.placeholder(
            shape: [NSNumber(value: n), NSNumber(value: c), NSNumber(value: h), NSNumber(value: w)],
            dataType: .float32, name: "input"
        )

        let rateVar = g.variable(
            with: withUnsafeBytes(of: Float(0)) { Data($0) },
            shape: [1], dataType: .float32, name: "dropout_rate"
        )
        let ratePh = g.placeholder(shape: [1], dataType: .float32, name: "dropout_rate_load")
        let rateAssign = g.assign(rateVar, tensor: ratePh, name: "dropout_rate_load_assign")

        // The Feature 2 composition under test (multiplier ≠ 1 branch).
        let rate: MPSGraphTensor
        if multiplier == 1 {
            rate = rateVar
        } else {
            let mult = g.constant(Double(multiplier), shape: [1], dataType: .float32)
            let scaled = g.multiplication(rateVar, mult, name: "rate_scaled")
            let cap = g.constant(0.95, shape: [1], dataType: .float32)
            rate = g.minimum(scaled, cap, name: "rate_capped")
        }

        let stateVar = g.variable(
            with: Data(count: 7 * MemoryLayout<Int32>.size),
            shape: [7], dataType: .int32, name: "dropout_rng_state"
        )
        let seeded = g.randomPhiloxStateTensor(withSeed: 0x5EED, name: "dropout_rng_seed")
        let seedOp = g.assign(stateVar, tensor: seeded, name: "dropout_rng_seed_assign")

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

        let keep = g.greaterThanOrEqualTo(drawn[0], rate, name: "keep")
        let maskF = g.cast(keep, to: .float32, name: "mask_f")
        let one = g.constant(1.0, shape: [1], dataType: .float32)
        let scale = g.division(one, g.subtraction(one, rate, name: "keep_frac"), name: "scale")
        let scaledMask = g.multiplication(maskF, scale, name: "scaled_mask")
        let output = g.multiplication(input, scaledMask, name: "output")

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
            input: input, output: output,
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

    private func runOutput(_ hx: Harness) -> [Float] {
        let results = hx.graph.run(
            with: hx.queue,
            feeds: [hx.input: hx.inputTensorData],
            targetTensors: [hx.output],
            targetOperations: [hx.advanceOp]
        )
        var out = [Float](repeating: .nan, count: elementCount)
        out.withUnsafeMutableBytes { results[hx.output]!.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return out
    }

    /// The same fp32 composition the graph computes, for exact expectations.
    private func effective(_ rate: Float, _ mult: Float) -> (rate: Float, scale: Float) {
        let e = min(rate * mult, 0.95)
        return (e, 1.0 / (1.0 - e))
    }

    // MARK: 1. multiplier 0 — exact identity at any rate (the exempt knob)

    func testMultiplierZeroIsExactIdentityAtAnyRate() throws {
        let inputs = (0..<elementCount).map { Float($0) - 60.5 }
        let hx = try makeHarness(multiplier: 0, inputValues: inputs)
        seed(hx)
        for rate: Float in [0.3, 0.7, 0.95] {
            setRate(hx, rate)
            for _ in 0..<3 {
                let out = runOutput(hx)
                XCTAssertEqual(out.map(\.bitPattern), inputs.map(\.bitPattern),
                               "multiplier 0 must be bit-exact identity at rate \(rate)")
            }
        }
    }

    // MARK: 2. fractional multiplier — effective rate and survivor scale

    func testHalfMultiplierHalvesEffectiveRate() throws {
        let inputs = [Float](repeating: 1, count: elementCount)
        let hx = try makeHarness(multiplier: 0.5, inputValues: inputs)
        seed(hx)
        setRate(hx, 0.7)
        let (eff, scale) = effective(0.7, 0.5)   // 0.35, 1/(1−0.35)

        var dropped = 0, total = 0
        for _ in 0..<200 {
            let out = runOutput(hx)
            for v in out {
                total += 1
                if v == 0 {
                    dropped += 1
                } else {
                    XCTAssertEqual(v, scale, accuracy: 1e-5,
                                   "survivors must carry the composed-rate inverted scale")
                }
            }
        }
        let fraction = Double(dropped) / Double(total)
        XCTAssertEqual(fraction, Double(eff), accuracy: 0.03,
                       "empirical drop fraction must track rate × multiplier")
    }

    // MARK: 3. cap at 0.95 — scale stays finite when rate × multiplier ≥ 1

    func testMultiplierAboveOneCapsAtPoint95() throws {
        let inputs = [Float](repeating: 1, count: elementCount)
        let hx = try makeHarness(multiplier: 2.0, inputValues: inputs)
        seed(hx)
        setRate(hx, 0.8)                          // 0.8 × 2 = 1.6 → capped 0.95
        let (eff, scale) = effective(0.8, 2.0)
        XCTAssertEqual(eff, 0.95)

        var dropped = 0, total = 0
        for _ in 0..<200 {
            let out = runOutput(hx)
            for v in out {
                total += 1
                if v == 0 {
                    dropped += 1
                } else {
                    XCTAssertEqual(v, scale, accuracy: 1e-4,
                                   "survivor scale must be 1/(1−0.95), not 1/(1−1.6)")
                    XCTAssertTrue(v.isFinite)
                }
            }
        }
        let fraction = Double(dropped) / Double(total)
        XCTAssertEqual(fraction, 0.95, accuracy: 0.03)
    }
}
