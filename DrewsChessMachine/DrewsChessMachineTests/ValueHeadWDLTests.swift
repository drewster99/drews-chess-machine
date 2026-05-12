//
//  ValueHeadWDLTests.swift
//  DrewsChessMachineTests
//
//  Pins the value-head W/D/L math introduced when the single tanh
//  scalar head was replaced by a 3-logit win/draw/loss softmax trained
//  with categorical cross-entropy (see wdl-value-head.md).
//
//  These are standalone MPSGraph tests in the style of
//  MPSGraphReshapeLayoutTests / PolicyHeadCorrectnessTests: they
//  rebuild the exact op chain that `ChessTrainer.buildTrainingOps`
//  (value-loss block) and `ChessNetwork.valueHead` (derived scalar)
//  use, run it on a real MTLDevice, and check it against a Swift
//  reference. `buildTrainingOps` / `valueHead` themselves are private
//  and need a full Metal/MPSGraph network to invoke, so the contract
//  worth pinning here is the math, not the call.
//
//  Invariants under test:
//    1. The training target slot index is `idx = clamp(int(1 − z), 0, 2)`
//       — z=+1 → 0 (win), z=0 → 1 (draw), z=−1 → 2 (loss); a
//       `-drawPenalty` rewrite of a draw (z ∈ [−1, 0)) still lands on
//       slot 1 unless the penalty is the full 1.0 (→ slot 2).
//    2. With ε=0 the value loss is hard-one-hot CE: `−log softmax(logits)[idx]`.
//    3. With ε>0 it's the label-smoothed CE:
//       `−Σ_c [(1−ε)·1[c==idx] + ε/3]·log softmax(logits)_c`.
//    4. The derived scalar value is `Σ_c softmax(logits)_c · [+1,0,−1]_c
//       = p_win − p_loss`, naturally in [−1, +1] (no tanh).
//    5. The value-head bias init `[0, ln 6, 0]` gives initial softmax
//       (0.125, 0.75, 0.125) and an initial scalar of exactly 0.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class ValueHeadWDLTests: XCTestCase {

    private let dtype = MPSDataType.float32

    // MARK: - Swift reference

    /// Numerically-stable softmax over a 3-vector.
    private func softmax3(_ logits: [Float]) -> [Float] {
        let m = logits.max()!
        let exps = logits.map { Float(exp(Double($0 - m))) }
        let s = exps.reduce(0, +)
        return exps.map { $0 / s }
    }

    /// Reference categorical cross-entropy `−Σ target·log softmax(logits)`
    /// for the W/D/L target built from outcome `z` with label-smoothing
    /// `eps`: `target_c = (1−eps)·1[c == clamp(int(1−z), 0, 2)] + eps/3`.
    private func referenceValueCE(logits: [Float], z: Float, eps: Float) -> Float {
        let p = softmax3(logits)
        let rawIdx = Int(Float(1) - z)         // C-style truncation toward zero
        let idx = max(0, min(2, rawIdx))
        var loss: Float = 0
        for c in 0..<3 {
            let target = (1 - eps) * (c == idx ? 1 : 0) + eps / 3
            loss -= target * Float(log(Double(max(p[c], Float.leastNormalMagnitude))))
        }
        return loss
    }

    // MARK: - Graph builders

    /// Build (and run) the value-CE op chain exactly as
    /// `ChessTrainer.buildTrainingOps` does: masked one-hot from
    /// `clamp(int(1 − z))`, blended with the ε·(1/3) uniform, fed to
    /// `softMaxCrossEntropy(..., reuctionType: .none)`, reshaped to
    /// [batch, 1], then mean over the batch.
    private func runValueCE(
        device: MTLDevice,
        logits: [[Float]],   // [batch][3]
        zs: [Float],         // [batch]
        eps: Float
    ) throws -> Float {
        let batch = logits.count
        precondition(zs.count == batch)
        let cmdQueue = device.makeCommandQueue()!
        let graph = MPSGraph()

        let flatLogits = logits.flatMap { $0 }
        let logitsT = graph.constant(
            makeFloatData(flatLogits),
            shape: [NSNumber(value: batch), 3],
            dataType: dtype
        )
        let zT = graph.constant(
            makeFloatData(zs),
            shape: [NSNumber(value: batch), 1],
            dataType: dtype
        )
        let one = graph.constant(1.0, dataType: dtype)
        let slotIdxFloat = graph.subtraction(one, zT, name: "slot_idx_float")
        let lo = graph.constant(0.0, dataType: dtype)
        let hi = graph.constant(Double(ChessNetwork.valueHeadClasses - 1), dataType: dtype)
        let slotIdxClamped = graph.minimum(
            graph.maximum(slotIdxFloat, lo, name: nil), hi, name: nil
        )
        let slotIdxInt = graph.cast(slotIdxClamped, to: .int32, name: "slot_idx_int")
        let slotIdxFlat = graph.reshape(slotIdxInt, shape: [-1], name: "slot_idx_flat")
        let oneHot = graph.oneHot(
            withIndicesTensor: slotIdxFlat,
            depth: 3, axis: 1, dataType: dtype,
            onValue: 1.0, offValue: 0.0, name: "value_onehot"
        )
        let epsT = graph.constant(Double(eps), shape: [1], dataType: dtype)
        let oneMinusEps = graph.subtraction(one, epsT, name: nil)
        let uniform = graph.constant(1.0 / 3.0, shape: [1, 3], dataType: dtype)
        let target = graph.addition(
            graph.multiplication(oneHot, oneMinusEps, name: nil),
            graph.multiplication(uniform, epsT, name: nil),
            name: "value_smoothed_target"
        )
        let cePerPos = graph.softMaxCrossEntropy(
            logitsT, labels: target, axis: 1, reuctionType: .none, name: "value_ce_raw"
        )
        let ceReshaped = graph.reshape(cePerPos, shape: [-1, 1], name: nil)
        let loss = graph.mean(of: ceReshaped, axes: [0, 1], name: "value_loss")

        let results = graph.run(
            with: cmdQueue, feeds: [:], targetTensors: [loss], targetOperations: nil
        )
        guard let lossData = results[loss] else {
            XCTFail("graph.run did not return the value loss"); return .nan
        }
        return readFloats(lossData, count: 1)[0]
    }

    /// Build (and run) the derived-scalar op chain exactly as
    /// `ChessNetwork.valueHead` does: `softMax(logits, axis: 1)` then
    /// `reductionSum(probs * [+1,0,−1] const, axis: 1)` → [batch, 1].
    private func runDerivedScalar(device: MTLDevice, logits: [[Float]]) throws -> [Float] {
        let batch = logits.count
        let cmdQueue = device.makeCommandQueue()!
        let graph = MPSGraph()
        let logitsT = graph.constant(
            makeFloatData(logits.flatMap { $0 }),
            shape: [NSNumber(value: batch), 3],
            dataType: dtype
        )
        let probs = graph.softMax(with: logitsT, axis: 1, name: "value_probs")
        let weights = graph.constant(
            makeFloatData([1.0, 0.0, -1.0]), shape: [1, 3], dataType: dtype
        )
        let weighted = graph.multiplication(probs, weights, name: nil)
        let scalar = graph.reductionSum(with: weighted, axis: 1, name: "value_scalar")
        let results = graph.run(
            with: cmdQueue, feeds: [:], targetTensors: [scalar], targetOperations: nil
        )
        guard let data = results[scalar] else {
            XCTFail("graph.run did not return the derived scalar"); return []
        }
        return readFloats(data, count: batch)
    }

    // MARK: - Tests

    func testValueHeadClassesIsThree() {
        XCTAssertEqual(ChessNetwork.valueHeadClasses, 3)
    }

    /// ε=0: the loss is the hard-one-hot CE `−log softmax(logits)[idx]`,
    /// with `idx = clamp(int(1 − z), 0, 2)` — exercises all of
    /// win/draw/loss plus a `-drawPenalty`-style drawn position.
    func testHardOneHotCEMatchesReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        // batch of 4: a clean win (z=+1 → slot 0), a clean draw
        // (z=0 → slot 1), a clean loss (z=−1 → slot 2), and a draw
        // with a partial draw-penalty applied (z=−0.3 → still slot 1).
        let logits: [[Float]] = [
            [ 2.0,  0.0, -1.0],
            [-0.5,  1.2,  0.3],
            [ 0.1, -0.2,  3.0],
            [ 0.7,  0.0, -0.4],
        ]
        let zs: [Float] = [ 1.0, 0.0, -1.0, -0.3 ]
        let got = try runValueCE(device: device, logits: logits, zs: zs, eps: 0.0)
        let want = zip(logits, zs).map { referenceValueCE(logits: $0.0, z: $0.1, eps: 0.0) }
            .reduce(0, +) / Float(logits.count)
        XCTAssertEqual(got, want, accuracy: 1e-4,
                       "ε=0 value CE should equal mean −log softmax(logits)[clamp(int(1−z))]")
    }

    /// A full draw-penalty (z = −1.0 for a drawn game) maps the value
    /// target to the *loss* slot — `int(1 − (−1)) = 2` — same as a real
    /// loss. Sub-1.0 penalties keep it on the draw slot (covered above).
    func testFullDrawPenaltyMapsToLossSlot() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let logits: [[Float]] = [[0.4, 0.1, -0.9]]
        // z = −1.0 (a draw rewritten with drawPenalty == 1.0).
        let got = try runValueCE(device: device, logits: logits, zs: [-1.0], eps: 0.0)
        let p = softmax3(logits[0])
        XCTAssertEqual(got, -Float(log(Double(p[2]))), accuracy: 1e-4,
                       "z=−1 (full draw penalty) should target the loss slot (idx 2)")
    }

    /// Pins the load-bearing assumption that MPSGraph's float→int32 cast
    /// **truncates toward zero** (not rounds): a fractional draw-penalty
    /// `drawPenalty ∈ (0, 1)` ⇒ `z ∈ (−1, 0)` ⇒ `1 − z ∈ (1, 2)`, and
    /// `int(1.7) = 1` (draw slot), `int(1.9) = 1` (draw slot). If
    /// MPSGraph ever rounds, `int(1.7)` would become 2 (loss slot) and a
    /// 0.7-penalized draw would be trained as a *loss* — a silent
    /// behavior change. If this test fails, add an explicit `graph.floor`
    /// before the int cast in `ChessTrainer.buildTrainingOps` (and in
    /// `runValueCE` here) and reconsider the intended slot for
    /// partial-penalty draws.
    func testFloatToIntCastTruncatesForDrawPenaltySlot() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let logits: [[Float]] = [
            [ 0.2, -0.1,  0.4],
            [-0.6,  0.3,  0.1],
        ]
        // z = −0.7 (1 − z = 1.7) and z = −0.9 (1 − z = 1.9): both must
        // truncate to slot 1 (draw).
        let zs: [Float] = [ -0.7, -0.9 ]
        let got = try runValueCE(device: device, logits: logits, zs: zs, eps: 0.0)
        let want = logits.map { -Float(log(Double(softmax3($0)[1]))) }.reduce(0, +) / Float(logits.count)
        XCTAssertEqual(got, want, accuracy: 1e-4,
                       "fractional draw penalty must keep the value target on the draw slot — "
                       + "MPSGraph float→int32 cast must truncate toward zero, not round")
    }

    /// ε>0: the loss is the label-smoothed CE
    /// `−Σ_c [(1−ε)·1[c==idx] + ε/3]·log softmax(logits)_c`.
    func testLabelSmoothedCEMatchesReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let logits: [[Float]] = [
            [ 1.5, -0.4,  0.2],
            [-1.0,  2.0,  0.0],
        ]
        let zs: [Float] = [ 1.0, 0.0 ]
        for eps in [Float(0.025), Float(0.05), Float(0.2)] {
            let got = try runValueCE(device: device, logits: logits, zs: zs, eps: eps)
            let want = zip(logits, zs).map { referenceValueCE(logits: $0.0, z: $0.1, eps: eps) }
                .reduce(0, +) / Float(logits.count)
            XCTAssertEqual(got, want, accuracy: 1e-4,
                           "label-smoothed value CE (ε=\(eps)) should match the reference")
        }
    }

    /// The derived scalar value is `p_win − p_loss` per position.
    func testDerivedScalarIsPWinMinusPLoss() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let logits: [[Float]] = [
            [ 3.0,  0.0, -3.0],   // strongly winning → near +1
            [-3.0,  0.0,  3.0],   // strongly losing → near −1
            [ 0.0,  5.0,  0.0],   // confident draw → near 0
            [ 0.4,  0.1, -0.4],   // mild edge
        ]
        let got = try runDerivedScalar(device: device, logits: logits)
        XCTAssertEqual(got.count, logits.count)
        for (i, l) in logits.enumerated() {
            let p = softmax3(l)
            XCTAssertEqual(got[i], p[0] - p[2], accuracy: 1e-5,
                           "derived scalar[\(i)] should be p_win − p_loss")
            XCTAssertGreaterThanOrEqual(got[i], -1.0001)
            XCTAssertLessThanOrEqual(got[i], 1.0001)
        }
    }

    /// The value-head bias init `[0, ln 6, 0]` ⇒ softmax (1/8, 6/8, 1/8)
    /// ⇒ derived scalar exactly 0 (matches the old `tanh(0) = 0`).
    func testBiasInitGivesDrawHeavyPriorAndZeroScalar() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let lnSix: Float = 1.791759469228055
        let scalar = try runDerivedScalar(device: device, logits: [[0.0, lnSix, 0.0]])
        XCTAssertEqual(scalar.count, 1)
        XCTAssertEqual(scalar[0], 0.0, accuracy: 1e-5,
                       "bias init [0, ln6, 0] should start the value scalar at 0")
        // And the implied softmax is the draw-heavy (0.125, 0.75, 0.125).
        let p = softmax3([0.0, lnSix, 0.0])
        XCTAssertEqual(p[0], 0.125, accuracy: 1e-5)
        XCTAssertEqual(p[1], 0.75, accuracy: 1e-5)
        XCTAssertEqual(p[2], 0.125, accuracy: 1e-5)
    }

    // MARK: - Helpers (copied per-file, matching the other MPSGraph tests)

    private func makeFloatData(_ values: [Float]) -> Data {
        var v = values
        return v.withUnsafeMutableBufferPointer { buf in
            Data(bytes: buf.baseAddress!, count: buf.count * MemoryLayout<Float>.size)
        }
    }

    private func readFloats(_ tensorData: MPSGraphTensorData, count: Int) -> [Float] {
        let nda = tensorData.mpsndarray()
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { buf in
            nda.readBytes(buf.baseAddress!, strideBytes: nil)
        }
        return out
    }
}
