//
//  MPSGraphGradientSemanticsTests.swift
//  DrewsChessMachineTests
//
//  Tests the semantics of MPSGraph's `gradients(of:with:name:)` to
//  determine whether it implements "stop_gradient via exclusion from
//  the `with` array" (the bird's claim) or standard autodiff
//  (paths through non-listed tensors still contribute gradient to
//  listed tensors via shared variables).
//
//  This matters for the value-head-as-baseline question: if exclusion
//  acts as stop_gradient, we have a one-line fix for the policy-
//  gradient-leak-into-tower problem. If it doesn't, we need a separate
//  forward pass (placeholder feed) or a target-network pattern.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class MPSGraphGradientSemanticsTests: XCTestCase {

    // MARK: - Test 1: Basic shared-variable path semantics
    //
    // Graph:
    //   W = variable(1.0)              [TRAINABLE, in `with`]
    //   W2 = variable(0.3)             [NOT in `with`]
    //   A = W * W2                     (path through W2)
    //   B = W                          (direct path)
    //   loss = A + B = W*W2 + W
    //
    // ∂loss/∂W = W2 + 1 = 1.3 (standard autodiff)
    //
    // If the bird's claim is right ("path leading to a tensor not in
    // the list has its gradient stopped"), then the path through W2
    // would be blocked and we'd get ∂loss/∂W = 1.0 (only the direct
    // path contributes).
    func testGradientThroughExcludedVariablePath() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let cmdQueue = device.makeCommandQueue()!
        let graph = MPSGraph()
        let dtype = MPSDataType.float32

        // Variables
        let wData = makeFloatData([1.0])
        let w = graph.variable(with: wData, shape: [1], dataType: dtype, name: "W")
        let w2Data = makeFloatData([0.3])
        let w2 = graph.variable(with: w2Data, shape: [1], dataType: dtype, name: "W2")

        // A = W * W2
        let pathThroughW2 = graph.multiplication(w, w2, name: "A")
        // loss = A + W = W*W2 + W
        let loss = graph.addition(pathThroughW2, w, name: "loss")

        // Compute gradient of loss w.r.t. ONLY W (W2 deliberately excluded
        // from the `with` array — this is what the bird's claim hinges on).
        let grads = graph.gradients(of: loss, with: [w], name: "grad")
        guard let gradW = grads[w] else {
            XCTFail("MPSGraph did not return gradient for W")
            return
        }

        // Need a placeholder feed — MPSGraph requires every placeholder
        // be fed even when it doesn't influence the targets. We have no
        // placeholders here, so the feeds dict is empty.
        let results = graph.run(
            with: cmdQueue,
            feeds: [:],
            targetTensors: [gradW],
            targetOperations: nil
        )

        guard let gradData = results[gradW] else {
            XCTFail("graph.run did not return gradW")
            return
        }
        let gradVal = readFloat(gradData)

        // Standard autodiff: gradVal == 1.3 (W2 contribution + direct contribution).
        // Bird's claim: gradVal == 1.0 (direct contribution only — path via W2 stopped).
        // Print the actual value for diagnostic visibility.
        print("[GRAD-TEST] ∂(W*W2 + W)/∂W with W2 excluded from `with` = \(gradVal)")
        SessionLogger.shared.log(
            "[GRAD-TEST] ∂(W*W2 + W)/∂W with W2 excluded from `with` = \(gradVal) (1.3 = standard, 1.0 = stop-grad-by-exclusion)"
        )

        // Verify which interpretation MPSGraph implements.
        XCTAssertEqual(gradVal, 1.3, accuracy: 1e-5,
                       "Expected standard autodiff (1.3); a value of 1.0 would mean the bird's claim is correct (paths through non-listed tensors are pruned).")
    }

    // MARK: - Test 2: Value-head-as-baseline scenario, distilled
    //
    // Mirrors the actual policy-gradient-with-baseline question:
    //
    //   T (tower) = variable(2.0)                 [TRAINABLE, in `with`]
    //   value_head = variable(0.5)                [TRAINABLE, in `with`]
    //   policy_head = variable(1.5)               [TRAINABLE, in `with`]
    //   tower_out = T (just rename for clarity)
    //   v = value_head * tower_out                (value head's output)
    //   pi_logit = policy_head * tower_out        (policy logit)
    //   value_loss = (z - v)^2                    where z = 0.0
    //   policy_loss = (z - v) * (-pi_logit)       advantage-weighted policy loss
    //   total_loss = value_loss + policy_loss
    //
    // Question: does ∂(policy_loss)/∂T include the contribution from
    // v (the path policy_loss → v → T)? Or does autodiff "stop" because
    // we wish v were a constant?
    //
    // Standard autodiff says yes, that contribution is included (which
    // is the leak that biases the tower).
    //
    // The whole question of whether "exclude value_head from `with`"
    // would block the leak. Test it.
    func testValueHeadAsBaselineLeak() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let cmdQueue = device.makeCommandQueue()!
        let graph = MPSGraph()
        let dtype = MPSDataType.float32

        // Trainable variables
        let t = graph.variable(with: makeFloatData([2.0]), shape: [1], dataType: dtype, name: "T")
        let valueHead = graph.variable(with: makeFloatData([0.5]), shape: [1], dataType: dtype, name: "valueHead")
        let policyHead = graph.variable(with: makeFloatData([1.5]), shape: [1], dataType: dtype, name: "policyHead")

        // Forward computation
        let v = graph.multiplication(valueHead, t, name: "v")              // 0.5 * 2 = 1.0
        let piLogit = graph.multiplication(policyHead, t, name: "piLogit") // 1.5 * 2 = 3.0
        let zConst = graph.constant(0.0, shape: [1], dataType: dtype)
        let advantage = graph.subtraction(zConst, v, name: "advantage")    // 0 - 1 = -1
        // value_loss = (z - v)^2 = advantage^2
        let valueLoss = graph.multiplication(advantage, advantage, name: "valueLoss") // 1
        // policy_loss = (z - v) * (-pi_logit) = advantage * (-piLogit)
        let negPiLogit = graph.negative(with: piLogit, name: "negPiLogit") // -3
        let policyLoss = graph.multiplication(advantage, negPiLogit, name: "policyLoss") // -1 * -3 = 3
        let totalLoss = graph.addition(valueLoss, policyLoss, name: "totalLoss") // 1 + 3 = 4

        // Manual gradients (T=2, vh=0.5, ph=1.5, v=1, pi_logit=3, z=0):
        //   ∂value_loss/∂T = 2*(z-v)*(-vh) = 2*(-1)*(-0.5) = 1
        //   ∂policy_loss/∂T = (-vh)*(-pi_logit) + (z-v)*(-ph) = 0.5*3 + (-1)*(-1.5) = 1.5 + 1.5 = 3
        //   ∂total_loss/∂T = 1 + 3 = 4
        //
        //   ∂value_loss/∂valueHead = 2*(z-v)*(-T) = 2*(-1)*(-2) = 4
        //   ∂policy_loss/∂valueHead = (-T)*(-pi_logit) = 2*3 = 6
        //   ∂total_loss/∂valueHead = 4 + 6 = 10
        //
        //   ∂value_loss/∂policyHead = 0
        //   ∂policy_loss/∂policyHead = (z-v)*(-T) = -1 * -2 = 2
        //   ∂total_loss/∂policyHead = 2
        //
        // Bird's claim: if we exclude valueHead from `with`, the path
        // through v in the policy_loss should be "stopped." Then:
        //   ∂total_loss/∂T (via "stopped" baseline) would be:
        //     ∂value_loss/∂T (= 1, value path still active)
        //     + ∂policy_loss/∂T_via_pi_only (= (z-v)*(-ph) = (-1)*(-1.5) = 1.5)
        //     = 1 + 1.5 = 2.5
        //   instead of the full 4.

        let withList = [t, policyHead]   // valueHead deliberately excluded
        let grads = graph.gradients(of: totalLoss, with: withList, name: "grad")
        guard let gradT = grads[t], let gradPolicy = grads[policyHead] else {
            XCTFail("MPSGraph did not return gradients for T or policyHead")
            return
        }

        let results = graph.run(
            with: cmdQueue,
            feeds: [:],
            targetTensors: [gradT, gradPolicy],
            targetOperations: nil
        )

        let gradTVal = readFloat(results[gradT]!)
        let gradPolicyVal = readFloat(results[gradPolicy]!)

        SessionLogger.shared.log(
            "[GRAD-TEST] valueHead-baseline scenario, valueHead excluded from `with`:"
        )
        SessionLogger.shared.log(
            "[GRAD-TEST]   ∂total_loss/∂T = \(gradTVal) (4.0 = standard, 2.5 = stop-grad-by-exclusion)"
        )
        SessionLogger.shared.log(
            "[GRAD-TEST]   ∂total_loss/∂policyHead = \(gradPolicyVal) (2.0 either way)"
        )

        // Standard autodiff prediction: 4.0. Bird's claim prediction: 2.5.
        // policyHead grad is 2.0 either way (it's not on a path that involves valueHead).
        XCTAssertEqual(gradTVal, 4.0, accuracy: 1e-4,
                       "T gradient: 4.0 = standard autodiff (full leak); 2.5 would mean bird's claim is correct")
        XCTAssertEqual(gradPolicyVal, 2.0, accuracy: 1e-4,
                       "policyHead gradient should be 2.0 regardless of interpretation")
    }

    // MARK: - Helpers

    private func makeFloatData(_ values: [Float]) -> Data {
        var v = values
        return v.withUnsafeMutableBufferPointer { buf in
            Data(bytes: buf.baseAddress!, count: buf.count * MemoryLayout<Float>.size)
        }
    }

    private func readFloat(_ tensorData: MPSGraphTensorData) -> Float {
        let nda = tensorData.mpsndarray()
        var out: Float = 0
        nda.readBytes(&out, strideBytes: nil)
        return out
    }
}
