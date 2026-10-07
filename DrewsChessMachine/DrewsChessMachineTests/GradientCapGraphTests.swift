//
//  GradientCapGraphTests.swift
//  DrewsChessMachineTests
//
//  The fed cap is what the graph clips with (plan X3). From zero velocity
//  the velocity after one step is exactly the clipped gradient
//  (`v = μ·0 + clipScale·g`, weight decay off), so three trainers from one
//  init seed on one batch show the clip directly: a cap that never binds
//  leaves `g`; a cap of ‖g‖/4 scales it to norm ‖g‖/4; and a cap that does
//  not bind (2‖g‖) gives a bit-identical step — `clipScale = cap / max(norm,
//  cap)` is `x / x`, exactly 1.0 in IEEE fp32 — which is what makes "identical
//  to the control until the first clip" a valid comparison for the
//  validation runs. A two-step test then shows the relative decision reaches
//  the placeholder.
//

import XCTest
@testable import DrewsChessMachine

final class GradientCapGraphTests: XCTestCase {

    private func l2(_ tensors: [[Float]]) -> Double {
        tensors.reduce(0) { sum, tensor in tensor.reduce(sum) { $0 + Double($1) * Double($1) } }.squareRoot()
    }

    func test_fedCapClipsTheStep_andANonBindingCapIsBitIdentical() async throws {
        let off = try RelativeGradientCapFixture.configuration(.off)
        let unclipped = try RelativeGradientCapFixture.makeTrainer(hardMax: 1.0e9, configuration: off)
        let timing1 = try await RelativeGradientCapFixture.step(
            unclipped, RelativeGradientCapFixture.makeReplayBuffer(arch: .current))
        let g = try await unclipped.exportVelocitySnapshot()
        let gNorm = l2(g)
        XCTAssertGreaterThan(gNorm, 0)
        XCTAssertEqual(Double(timing1.gradGlobalNorm), gNorm, accuracy: gNorm * 1.0e-3,
                       "the velocity after one step from zero is the raw gradient")
        XCTAssertEqual(timing1.gradientCap.fedCap, 1.0e9)

        // Cap ‖g‖/4: the velocity is (c/‖g‖)·g.
        let quarter = Float(Double(timing1.gradGlobalNorm) / 4)
        let clippedTrainer = try RelativeGradientCapFixture.makeTrainer(hardMax: quarter, configuration: off)
        let timing2 = try await RelativeGradientCapFixture.step(
            clippedTrainer, RelativeGradientCapFixture.makeReplayBuffer(arch: .current))
        XCTAssertEqual(timing2.gradientCap.fedCap, quarter)
        XCTAssertTrue(timing2.gradientCap.clipped(preClipNorm: timing2.gradGlobalNorm))
        let clippedVelocity = try await clippedTrainer.exportVelocitySnapshot()
        XCTAssertEqual(l2(clippedVelocity), Double(quarter), accuracy: Double(quarter) * 1.0e-3)
        let scale = Double(quarter) / Double(timing2.gradGlobalNorm)
        var residual = 0.0
        for (v, raw) in zip(clippedVelocity, g) {
            for (x, y) in zip(v, raw) {
                let d = Double(x) - scale * Double(y)
                residual += d * d
            }
        }
        XCTAssertLessThan(residual.squareRoot(), Double(quarter) * 1.0e-3, "velocity = (c/‖g‖)·g element-wise")

        // Cap 2‖g‖ does not bind: bit-identical to the never-binding cap.
        let loose = Float(Double(timing1.gradGlobalNorm) * 2)
        let looseTrainer = try RelativeGradientCapFixture.makeTrainer(hardMax: loose, configuration: off)
        let timing3 = try await RelativeGradientCapFixture.step(
            looseTrainer, RelativeGradientCapFixture.makeReplayBuffer(arch: .current))
        XCTAssertFalse(timing3.gradientCap.clipped(preClipNorm: timing3.gradGlobalNorm))
        let looseVelocity = try await looseTrainer.exportVelocitySnapshot()
        XCTAssertEqual(looseVelocity.map { $0.map(\.bitPattern) }, g.map { $0.map(\.bitPattern) },
                       "an unclipped step's velocity does not depend on the fed cap")
        let looseWeights = try await looseTrainer.exportTrainerWeights()
        let unclippedWeights = try await unclipped.exportTrainerWeights()
        XCTAssertEqual(looseWeights.map { $0.map(\.bitPattern) }, unclippedWeights.map { $0.map(\.bitPattern) },
                       "an unclipped step's weights do not depend on the fed cap")
    }

    func test_relativeDecisionReachesThePlaceholder() async throws {
        // k = 1, W = N = 1: step 2's cap is step 1's pre-clip norm.
        let config = try RelativeGradientCapFixture.configuration(.clip, k: 1, n: 1, w: 1)
        let trainer = try RelativeGradientCapFixture.makeTrainer(hardMax: 1.0e9, configuration: config)
        let buffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        let first = try await RelativeGradientCapFixture.step(trainer, buffer)
        XCTAssertEqual(first.gradientCap.binding, .hard, "no history before step 1")
        XCTAssertEqual(first.gradientCap.fedCap, 1.0e9)
        let second = try await RelativeGradientCapFixture.step(trainer, buffer)
        XCTAssertEqual(second.gradientCap.binding, .relative)
        XCTAssertEqual(second.gradientCap.fedCap, first.gradGlobalNorm)
        XCTAssertEqual(second.gradientCap.referenceMedian, Double(first.gradGlobalNorm))
        let history = try await trainer.exportGradNormHistory()
        XCTAssertEqual(history.lastTrainerStep, 2)
        XCTAssertEqual(history.preClipNorms, [first.gradGlobalNorm, second.gradGlobalNorm])
        XCTAssertEqual(history.fedCaps, [1.0e9, first.gradGlobalNorm])
        // The fed cap clipped step 2 exactly when its norm exceeded step 1's.
        XCTAssertEqual(second.gradientCap.clipped(preClipNorm: second.gradGlobalNorm),
                       second.gradGlobalNorm > first.gradGlobalNorm)
    }

    /// Review MAJOR 3: the relative cap reaches the graph. Three trainers
    /// from one seed on the same batches: B clips with the relative cap
    /// (k = 0.5, W = N = 1, so step 2's cap is half step 1's pre-clip norm);
    /// A has the cap off and is fed that same value as its hard max for step
    /// 2; C is never clipped. Step 1 is bit-identical in all three; after
    /// step 2, B equals A bit for bit, and B's step-2 increment
    /// (v₂ − μ·v₁) is C's scaled by cap/‖g₂‖. If `buildFeeds` fed the hard
    /// max (1e9) instead of the decision, B would equal C instead.
    func test_relativeCapIsWhatTheGraphClipsWith() async throws {
        let hugeMax: Float = 1.0e9
        let clipConfig = try RelativeGradientCapFixture.configuration(.clip, k: 0.5, n: 1, w: 1)
        let off = try RelativeGradientCapFixture.configuration(.off)
        let b = try RelativeGradientCapFixture.makeTrainer(hardMax: hugeMax, configuration: clipConfig)
        let a = try RelativeGradientCapFixture.makeTrainer(hardMax: hugeMax, configuration: off)
        let c = try RelativeGradientCapFixture.makeTrainer(hardMax: hugeMax, configuration: off)
        let bufferB = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        let bufferA = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        let bufferC = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)

        let b1 = try await RelativeGradientCapFixture.step(b, bufferB)
        _ = try await RelativeGradientCapFixture.step(a, bufferA)
        _ = try await RelativeGradientCapFixture.step(c, bufferC)
        let v1 = try await b.exportVelocitySnapshot()
        let v1a = try await a.exportVelocitySnapshot()
        XCTAssertEqual(v1.map { $0.map(\.bitPattern) }, v1a.map { $0.map(\.bitPattern) }, "step 1 is identical")

        let cap = Float(0.5 * Double(b1.gradGlobalNorm))
        a.gradClipMaxNorm = cap
        let b2 = try await RelativeGradientCapFixture.step(b, bufferB)
        let a2 = try await RelativeGradientCapFixture.step(a, bufferA)
        let c2 = try await RelativeGradientCapFixture.step(c, bufferC)
        XCTAssertEqual(b2.gradientCap.fedCap, cap)
        XCTAssertEqual(a2.gradientCap.fedCap, cap)
        XCTAssertGreaterThan(c2.gradGlobalNorm, cap, "the fixture's step 2 must exceed the cap for it to bind")

        let v2b = try await b.exportVelocitySnapshot()
        let v2a = try await a.exportVelocitySnapshot()
        let v2c = try await c.exportVelocitySnapshot()
        XCTAssertEqual(v2b.map { $0.map(\.bitPattern) }, v2a.map { $0.map(\.bitPattern) },
                       "the relative cap clips exactly as the same value fed as the hard max")
        let wb = try await b.exportTrainerWeights()
        let wa = try await a.exportTrainerWeights()
        XCTAssertEqual(wb.map { $0.map(\.bitPattern) }, wa.map { $0.map(\.bitPattern) })

        // B's increment is C's scaled by cap/‖g₂‖.
        let mu = Double(b.effectiveMomentum(completedSteps: 1))
        let scale = Double(cap) / Double(c2.gradGlobalNorm)
        var residual = 0.0
        var reference = 0.0
        for ((vb, vc), v) in zip(zip(v2b, v2c), v1) {
            for ((xb, xc), x) in zip(zip(vb, vc), v) {
                let incrementB = Double(xb) - mu * Double(x)
                let incrementC = Double(xc) - mu * Double(x)
                residual += (incrementB - scale * incrementC) * (incrementB - scale * incrementC)
                reference += (scale * incrementC) * (scale * incrementC)
            }
        }
        XCTAssertLessThan(residual.squareRoot(), reference.squareRoot() * 1.0e-3,
                          "the clipped increment is the unclipped one scaled by cap/‖g₂‖")
    }

    /// Log-only feeds the hard max, so over several steps it trains bit for
    /// bit like mode off (the validation runs' "identical until the first
    /// clip" rests on this).
    func test_logOnlyIsBitIdenticalToOffOverSeveralSteps() async throws {
        let logOnly = try RelativeGradientCapFixture.makeTrainer(
            hardMax: 15, configuration: try RelativeGradientCapFixture.configuration(.logOnly, k: 1, n: 1, w: 1))
        let off = try RelativeGradientCapFixture.makeTrainer(
            hardMax: 15, configuration: try RelativeGradientCapFixture.configuration(.off))
        let bufferL = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        let bufferO = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        for _ in 0..<4 {
            let l = try await RelativeGradientCapFixture.step(logOnly, bufferL)
            let o = try await RelativeGradientCapFixture.step(off, bufferO)
            XCTAssertEqual(l.gradientCap.fedCap, 15)
            XCTAssertEqual(l.gradGlobalNorm.bitPattern, o.gradGlobalNorm.bitPattern)
        }
        let wl = try await logOnly.exportTrainerWeights()
        let wo = try await off.exportTrainerWeights()
        XCTAssertEqual(wl.map { $0.map(\.bitPattern) }, wo.map { $0.map(\.bitPattern) })
    }

    func test_syntheticStep_feedsTheHardMaxAndRecordsNothing() async throws {
        let config = try RelativeGradientCapFixture.configuration(.clip, k: 1, n: 1, w: 1)
        let trainer = try RelativeGradientCapFixture.makeTrainer(hardMax: 15, configuration: config)
        let timing = try await trainer.trainStep(batchSize: RelativeGradientCapFixture.batchSize)
        XCTAssertEqual(timing.gradientCap, .hardMaxOnly(hardMax: 15))
        let history = try await trainer.exportGradNormHistory()
        XCTAssertTrue(history.isEmpty)
        XCTAssertEqual(trainer.completedTrainSteps, 0)
    }
}
