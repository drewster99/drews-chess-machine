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
