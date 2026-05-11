//
//  PolicyLossIllegalMassGradientTests.swift
//  DrewsChessMachineTests
//
//  Guards the invariant that the policy cross-entropy in
//  `ChessTrainer.buildTrainingOps` is computed over the RAW policy logits
//  (`network.policyOutput`), not the legal-masked logits.
//
//  Why this matters: commit `acc5340` once fed `maskedLogits` (= raw logits
//  with `-1e9` added at illegal cells) to the CE, reasoning that masking
//  would stop probability mass accumulating on illegal moves. It did the
//  opposite. With masked logits the softmax over illegal cells is ≈0 *by
//  the -1e9 bias*, so `∂CE/∂(illegal logit) ≈ softmax_masked - target ≈ 0`
//  — the CE became blind to illegal logits. The entropy bonus is also
//  masked, and the softmax-mass `illegalMassPenalty` has gradient ∝ p,
//  which → 0 once illegal mass ≈ 1, so there was *no* effective force
//  pushing illegal logits down: `pIllM` parked at ~0.997 for entire runs.
//
//  Over the RAW logits the smoothed CE target `(1-ε)·oneHot(played) +
//  ε·uniform(legal)` is still zero on illegal cells — the CE can never
//  *reward* illegal mass — but now `∂CE/∂(illegal logit) =
//  softmax_raw(illegal) ∈ [0, 1]`, a bounded gradient that always pushes
//  illegal logits down and vanishes only once illegal mass is genuinely
//  ≈ 0 (the correct fixed point).
//
//  These tests build a faithful fragment of the policy loss (smoothed-
//  target softMaxCrossEntropy + the softmax-mass illegalMassPenalty,
//  combined as `mean(advWeight·CE) + 1.0·illegalMassPenalty`) and check
//  the sign / magnitude of ∂(totalLoss)/∂(policyLogits) at deliberately
//  inflated illegal cells, for both the raw-logit CE (the fix) and the
//  masked-logit CE (the documented pre-fix counter-case). The property
//  under test is independent of policy-head size and batch size, so we use
//  a tiny 8-class / 2-position graph to keep the arithmetic legible.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class PolicyLossIllegalMassGradientTests: XCTestCase {

    private static let classes = 8
    private static let batch = 2
    private static let epsilon: Float = 0.1   // label-smoothing ε (matches the in-repo default)

    // Per position: legal cells {0, 3}; played move = cell 0; the (illegal)
    // cells 1 and 2 carry a large positive logit — the "leaked" mass we
    // want the loss to drain. Every other cell sits at logit 0.
    private static let legalCells: Set<Int> = [0, 3]
    private static let inflatedIllegalCells: [Int] = [1, 2]
    private static let inflatedLogit: Float = 5.0
    private static let playedMove: Int32 = 0

    /// Raw-logit CE (the fix): the CE must put a strong *positive* gradient
    /// on the inflated illegal logits (gradient descent drives them down)
    /// and a strong *negative* gradient on the played (legal) cell (driven
    /// up).
    func testRawLogitCEDrainsIllegalMass() throws {
        let g = try policyLossGradient(maskCEInput: false)

        for cell in Self.inflatedIllegalCells {
            XCTAssertGreaterThan(g[0][cell], 0.05,
                "raw-logit CE should put a strong positive gradient on inflated illegal cell \(cell) (got \(g[0][cell]))")
            XCTAssertGreaterThan(g[1][cell], 0.05,
                "raw-logit CE should put a strong positive gradient on inflated illegal cell \(cell), pos 1 (got \(g[1][cell]))")
        }
        XCTAssertLessThan(g[0][Int(Self.playedMove)], -0.1,
            "raw-logit CE should put a strong negative gradient on the played cell (got \(g[0][Int(Self.playedMove)]))")
    }

    /// Masked-logit CE (the bug, kept as a documented counter-case): the CE
    /// sees ≈0 softmax on the illegal cells, so `∂CE/∂(illegal logit) ≈ 0`.
    /// The only term left touching those logits is the softmax-mass
    /// `illegalMassPenalty`, whose gradient is `∝ softmax·(1 - illegalMass)`
    /// — and `(1 - illegalMass) ≈ 0` exactly when mass has piled onto the
    /// illegal cells. So the net gradient is ~two orders of magnitude
    /// smaller than the raw-CE case. (Legal-cell shaping still works under
    /// the masked CE — masking it only blinded the loss to illegal cells.)
    func testMaskedLogitCELeavesIllegalLogitsUntouched() throws {
        let masked = try policyLossGradient(maskCEInput: true)
        let raw = try policyLossGradient(maskCEInput: false)

        for cell in Self.inflatedIllegalCells {
            XCTAssertLessThan(abs(masked[0][cell]), 0.01,
                "masked-logit CE should leave ~no gradient on illegal cell \(cell) (got \(masked[0][cell]))")
            XCTAssertGreaterThan(raw[0][cell], abs(masked[0][cell]) * 10,
                "raw-CE gradient on illegal cell \(cell) should dwarf the masked-CE one (raw \(raw[0][cell]) vs masked \(masked[0][cell]))")
        }
        XCTAssertLessThan(masked[0][Int(Self.playedMove)], 0.0,
            "masked-logit CE should still push the played cell up (got \(masked[0][Int(Self.playedMove)]))")
    }

    // MARK: - Graph

    /// Mirrors `ChessTrainer.buildTrainingOps`'s policy-loss fragment and
    /// returns ∂(totalLoss)/∂(policyLogits) as `[batch][classes]`.
    /// `maskCEInput == false` feeds the raw logits to the CE (the fix);
    /// `true` feeds `policyLogits + illegalMask·(-1e9)` (the pre-fix
    /// behaviour). The `illegalMassPenalty` term always uses the raw
    /// softmax, exactly as in production.
    private func policyLossGradient(maskCEInput: Bool) throws -> [[Float]] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let cmdQueue = device.makeCommandQueue()!
        let graph = MPSGraph()
        let dtype = MPSDataType.float32
        let C = Self.classes
        let B = Self.batch

        // policyLogits — trainable variable so we can differentiate w.r.t. it.
        var logitValues = [Float](repeating: 0, count: B * C)
        for b in 0..<B {
            for cell in Self.inflatedIllegalCells { logitValues[b * C + cell] = Self.inflatedLogit }
        }
        let policyLogits = graph.variable(
            with: floatData(logitValues),
            shape: [NSNumber(value: B), NSNumber(value: C)],
            dataType: dtype,
            name: "policy_logits"
        )

        // legalMask — 1.0 at legal cells, 0.0 elsewhere.
        var maskValues = [Float](repeating: 0, count: B * C)
        for b in 0..<B {
            for cell in Self.legalCells { maskValues[b * C + cell] = 1.0 }
        }
        let legalMask = graph.constant(
            floatData(maskValues),
            shape: [NSNumber(value: B), NSNumber(value: C)],
            dataType: dtype
        )

        // movePlayed — int32 indices for the one-hot, one per batch element.
        let movePlayed = graph.constant(
            int32Data([Int32](repeating: Self.playedMove, count: B)),
            shape: [NSNumber(value: B)],
            dataType: .int32
        )

        // advWeight = max(0, advNorm); all positive (1.0) for the test.
        let advWeight = graph.constant(1.0, shape: [NSNumber(value: B), 1], dataType: dtype)
        let illegalMassWeight = graph.constant(1.0, shape: [1], dataType: dtype)
        let epsTensor = graph.constant(Double(Self.epsilon), shape: [1], dataType: dtype)

        // --- mirror buildTrainingOps ---
        let oneConst = graph.constant(1.0, dataType: dtype)
        let illegalMask = graph.subtraction(oneConst, legalMask, name: "illegal_mask")
        let largeNeg = graph.constant(-1e9, dataType: dtype)
        let additiveMask = graph.multiplication(illegalMask, largeNeg, name: "additive_mask")
        let maskedLogits = graph.addition(policyLogits, additiveMask, name: "masked_logits")

        let oneHot = graph.oneHot(
            withIndicesTensor: movePlayed,
            depth: C,
            axis: 1,
            dataType: dtype,
            onValue: 1.0,
            offValue: 0.0,
            name: "move_onehot"
        )
        let legalCountKeepDims = graph.reductionSum(with: legalMask, axis: 1, name: "legal_count")
        let legalCountSafe = graph.maximum(legalCountKeepDims, oneConst, name: "legal_count_safe")
        let uniformOverLegal = graph.division(legalMask, legalCountSafe, name: "uniform_over_legal")
        let oneMinusEps = graph.subtraction(oneConst, epsTensor, name: "one_minus_eps")
        let smoothedTarget = graph.addition(
            graph.multiplication(oneHot, oneMinusEps, name: "smoothed_onehot_part"),
            graph.multiplication(uniformOverLegal, epsTensor, name: "smoothed_uniform_part"),
            name: "smoothed_target"
        )

        let ceInput = maskCEInput ? maskedLogits : policyLogits
        let ceRaw = graph.softMaxCrossEntropy(
            ceInput,
            labels: smoothedTarget,
            axis: 1,
            reuctionType: .none,
            name: "policy_ce_raw"
        )
        let negLogProb = graph.reshape(ceRaw, shape: [-1, 1], name: "policy_ce_per_pos")
        let weightedCE = graph.multiplication(advWeight, negLogProb, name: "weighted_ce")
        let policyLoss = graph.mean(of: weightedCE, axes: [0, 1], name: "policy_loss")

        // Illegal-mass penalty — always over the RAW (unmasked) softmax.
        let unmaskedSoftmax = graph.softMax(with: policyLogits, axis: 1, name: "policy_softmax_unmasked")
        let illegalMassPerPos = graph.reductionSum(
            with: graph.multiplication(unmaskedSoftmax, illegalMask, name: "illegal_mass_per_cell"),
            axis: 1,
            name: "illegal_mass_per_pos"
        )
        let illegalMassPenalty = graph.mean(of: illegalMassPerPos, axes: [0, 1], name: "illegal_mass_penalty")
        let illegalTerm = graph.multiplication(illegalMassWeight, illegalMassPenalty, name: "illegal_mass_term")

        let totalLoss = graph.addition(policyLoss, illegalTerm, name: "total_loss")

        let grads = graph.gradients(of: totalLoss, with: [policyLogits], name: "grad")
        guard let gradLogits = grads[policyLogits] else {
            XCTFail("MPSGraph returned no gradient for policyLogits")
            return []
        }

        let results = graph.run(
            with: cmdQueue,
            feeds: [:],
            targetTensors: [gradLogits],
            targetOperations: nil
        )
        guard let gradData = results[gradLogits] else {
            XCTFail("graph.run returned no gradient data")
            return []
        }
        let flat = readFloats(gradData, count: B * C)
        var out: [[Float]] = []
        for b in 0..<B { out.append(Array(flat[(b * C)..<((b + 1) * C)])) }
        return out
    }

    // MARK: - Helpers

    private func floatData(_ values: [Float]) -> Data {
        var v = values
        return v.withUnsafeMutableBufferPointer {
            Data(bytes: $0.baseAddress!, count: $0.count * MemoryLayout<Float>.size)
        }
    }

    private func int32Data(_ values: [Int32]) -> Data {
        var v = values
        return v.withUnsafeMutableBufferPointer {
            Data(bytes: $0.baseAddress!, count: $0.count * MemoryLayout<Int32>.size)
        }
    }

    private func readFloats(_ tensorData: MPSGraphTensorData, count: Int) -> [Float] {
        let nda = tensorData.mpsndarray()
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { nda.readBytes($0.baseAddress!, strideBytes: nil) }
        return out
    }
}
