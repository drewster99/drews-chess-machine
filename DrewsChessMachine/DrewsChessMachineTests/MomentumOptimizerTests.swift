//
//  MomentumOptimizerTests.swift
//  DrewsChessMachineTests
//
//  Tests for the Polyak SGD momentum optimizer added to ChessTrainer.
//
//  Coverage:
//  - μ=0.0 default does not change the per-step update mathematically
//    (verified indirectly: trainer remains stable, loss bounded — same
//    invariant the existing pure-SGD tests rely on).
//  - Velocity buffers are allocated and zero-initialized at trainer
//    construction.
//  - Velocity buffers are updated by training steps when μ>0 (they
//    become non-zero after at least one step with random gradients).
//  - exportTrainerWeights() round-trips through loadTrainerWeights():
//    re-loaded weights and velocities match the saved snapshot
//    bit-exactly.
//  - Loading a v1-shape weight array (no velocity tensors) is
//    accepted; base weights are loaded; velocities are left at their
//    pre-load state (zero on a fresh trainer).
//  - resetVelocitiesToZero() clears all velocity buffers after they
//    have been populated.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class MomentumOptimizerTests: XCTestCase {

    // MARK: Helpers

    /// A handle into a trainer's serialized state, partitioned into
    /// base (trainables + bn) and velocity slices. Lets tests assert
    /// against each part separately without re-doing index arithmetic.
    private struct PartitionedWeights {
        let base: [[Float]]      // trainables + bnRunningStats
        let velocity: [[Float]]  // one entry per trainable
        var combined: [[Float]] { base + velocity }
    }

    private func partition(_ weights: [[Float]], trainer: ChessTrainer) -> PartitionedWeights {
        let baseCount = trainer.network.trainableVariables.count
            + trainer.network.bnRunningStatsVariables.count
        let velocityCount = trainer.network.trainableVariables.count
        // Either v1 (no velocity) or v2 (with velocity) layout.
        if weights.count == baseCount {
            return PartitionedWeights(base: weights, velocity: [])
        }
        precondition(weights.count == baseCount + velocityCount,
                     "Unexpected trainer weight count \(weights.count); expected \(baseCount) or \(baseCount + velocityCount)")
        return PartitionedWeights(
            base: Array(weights.prefix(baseCount)),
            velocity: Array(weights.suffix(velocityCount))
        )
    }

    /// L2 norm of every float in the array-of-arrays.
    private func l2Norm(_ aa: [[Float]]) -> Float {
        var sumSq: Double = 0
        for arr in aa {
            for v in arr {
                sumSq += Double(v) * Double(v)
            }
        }
        return Float(sqrt(sumSq))
    }

    /// Are all values across all sub-arrays exactly zero?
    private func allZero(_ aa: [[Float]]) -> Bool {
        for arr in aa {
            for v in arr where v != 0 { return false }
        }
        return true
    }

    // MARK: Tests

    /// Velocity buffers exist after trainer construction and are
    /// zero-initialized. Verifies the buildTrainingOps allocation
    /// path and the export's velocity-readback path simultaneously.
    func testInitialVelocityBuffersAreZero() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(lrWarmupSteps: 0)
        let weights = try await trainer.exportTrainerWeights()
        let p = partition(weights, trainer: trainer)
        XCTAssertEqual(
            p.velocity.count,
            trainer.network.trainableVariables.count,
            "exportTrainerWeights should append one velocity tensor per trainable"
        )
        XCTAssertGreaterThan(p.velocity.count, 0, "Trainer must have trainable variables")
        XCTAssertTrue(
            allZero(p.velocity),
            "Velocity buffers should be zero-initialized at trainer construction"
        )
    }

    /// Default μ=0.0 leaves velocities populated with the per-step
    /// update direction (since v_new = 0*v_old + combinedUpdate at μ=0).
    /// After one training step velocities should be NON-zero (random
    /// gradients flowed into combinedUpdate), verifying the velocity
    /// assign actually wires up.
    func testVelocityAssignFiresEachStep() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(momentumCoeff: 0.0, lrWarmupSteps: 0)
        _ = try await trainer.trainStep(batchSize: 32)
        let weights = try await trainer.exportTrainerWeights()
        let p = partition(weights, trainer: trainer)
        XCTAssertFalse(
            allZero(p.velocity),
            "After one trainStep at μ=0, velocity buffers should hold the per-step update direction (non-zero given random gradients)"
        )
    }

    /// At μ=0.9 the velocity norm should grow across steps (in
    /// expectation; we run enough steps that the geometric average
    /// dominates per-step random noise). Compare the velocity norm
    /// after 1 step vs after 8 steps — the longer run should be
    /// strictly larger.
    func testMomentumAccumulatesVelocityOverSteps() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(momentumCoeff: 0.9, lrWarmupSteps: 0)
        _ = try await trainer.trainStep(batchSize: 32)
        let after1 = try await trainer.exportTrainerWeights()
        let v1 = partition(after1, trainer: trainer).velocity
        let norm1 = l2Norm(v1)

        for _ in 0..<7 {
            _ = try await trainer.trainStep(batchSize: 32)
        }
        let after8 = try await trainer.exportTrainerWeights()
        let v8 = partition(after8, trainer: trainer).velocity
        let norm8 = l2Norm(v8)

        // Theoretical steady-state under random independent gradients:
        // ||v||² → ||g||² / (1 - μ²) = ||g||² / 0.19 ≈ 5.26 * ||g||²
        // So ||v_8|| should be meaningfully larger than ||v_1||.
        // Allow some slack for noise — assert at least 1.5x growth.
        XCTAssertGreaterThan(norm8, norm1 * 1.5,
            "At μ=0.9, velocity norm should grow substantially across 8 steps. norm1=\(norm1), norm8=\(norm8)")
    }

    /// Round-trip through exportTrainerWeights → loadTrainerWeights
    /// must restore the trainer to bit-exact the same state. This
    /// exercises the full v2 layout including velocities.
    func testTrainerWeightsRoundTripV2() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(momentumCoeff: 0.9, lrWarmupSteps: 0)
        // Run a few steps so weights AND velocities are non-trivial.
        for _ in 0..<3 {
            _ = try await trainer.trainStep(batchSize: 32)
        }
        let snapshot = try await trainer.exportTrainerWeights()

        // Mutate state by running another step, then restore.
        _ = try await trainer.trainStep(batchSize: 32)
        let mutated = try await trainer.exportTrainerWeights()

        // Sanity: state actually changed during the extra step.
        XCTAssertNotEqual(
            mutated[0],
            snapshot[0],
            "Weights should have changed after the post-snapshot training step (otherwise this test is meaningless)"
        )

        // Restore the snapshot.
        try await trainer.loadTrainerWeights(snapshot)
        let restored = try await trainer.exportTrainerWeights()

        // Bit-exact equality across every tensor.
        XCTAssertEqual(restored.count, snapshot.count, "Tensor count must round-trip")
        for i in 0..<snapshot.count {
            XCTAssertEqual(restored[i].count, snapshot[i].count,
                           "Tensor \(i) element count must round-trip")
            // Use exact equality. Float32 round-trip through writeFloats
            // / readFloats / mpsgraph assign / mpsgraph read should be
            // bit-exact since no math happens between snapshot and restore.
            for k in 0..<snapshot[i].count {
                if restored[i][k] != snapshot[i][k] {
                    XCTFail("Tensor \(i) element \(k) mismatch: snapshot=\(snapshot[i][k]) restored=\(restored[i][k])")
                    return
                }
            }
        }
    }

    /// Loading a v1-shape weight array (length = trainables + bn,
    /// no velocity tensors) must be accepted. Base weights load;
    /// velocities are NOT touched — they remain at whatever value
    /// they had pre-load (zero on a fresh trainer).
    /// TODO(persist-velocity, after 2026-06-04): can be removed
    /// once the v1 zero-pad migration window expires.
    func testV1ShapeLoadAcceptedWithVelocityZeroed() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(momentumCoeff: 0.9, lrWarmupSteps: 0)
        // Run a step so weights and velocities are both non-zero.
        _ = try await trainer.trainStep(batchSize: 32)
        let v2Snapshot = try await trainer.exportTrainerWeights()
        let p2 = partition(v2Snapshot, trainer: trainer)
        XCTAssertFalse(allZero(p2.velocity), "Velocity should be non-zero after one step at μ=0.9")

        // Build a v1-shape array: only the base portion.
        let v1Shape = p2.base
        XCTAssertEqual(
            v1Shape.count,
            trainer.network.trainableVariables.count + trainer.network.bnRunningStatsVariables.count,
            "v1 layout: trainables + bnRunningStats only"
        )

        // Reset the network to clear any existing velocity, then load
        // the v1 array. After load, velocities should be zero (initial
        // state from the resetNetwork rebuild) and base weights should
        // match.
        try await trainer.resetNetwork()
        try await trainer.loadTrainerWeights(v1Shape)
        let after = try await trainer.exportTrainerWeights()
        let pa = partition(after, trainer: trainer)

        XCTAssertEqual(pa.base.count, p2.base.count, "Base tensor count after v1 load should match input")
        for i in 0..<pa.base.count {
            XCTAssertEqual(pa.base[i], p2.base[i], "Base tensor \(i) should match v1-loaded value bit-exactly")
        }
        XCTAssertTrue(
            allZero(pa.velocity),
            "Velocity should be zero after v1-shape load (no velocity slots in source data; trainer.resetNetwork zeroed them)"
        )
    }

    /// resetVelocitiesToZero() clears all velocity buffers after
    /// they've been populated by training. Used post-promotion to
    /// discard velocity that points against the now-replaced weight
    /// surface.
    func testResetVelocitiesToZero() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(momentumCoeff: 0.9, lrWarmupSteps: 0)
        for _ in 0..<3 {
            _ = try await trainer.trainStep(batchSize: 32)
        }
        let beforeReset = try await trainer.exportTrainerWeights()
        let pBefore = partition(beforeReset, trainer: trainer)
        XCTAssertFalse(allZero(pBefore.velocity), "Velocity should be non-zero before reset")

        try await trainer.resetVelocitiesToZero()
        let afterReset = try await trainer.exportTrainerWeights()
        let pAfter = partition(afterReset, trainer: trainer)

        XCTAssertTrue(
            allZero(pAfter.velocity),
            "Velocity should be all zeros after resetVelocitiesToZero()"
        )

        // Base weights MUST be unchanged by velocity reset.
        XCTAssertEqual(pAfter.base.count, pBefore.base.count)
        for i in 0..<pBefore.base.count {
            XCTAssertEqual(pAfter.base[i], pBefore.base[i],
                           "Base tensor \(i) should be untouched by resetVelocitiesToZero()")
        }
    }

    /// Loading a malformed weights array (count not matching either
    /// v1 or v2 layout) must throw.
    func testLoadTrainerWeightsRejectsWrongCount() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(lrWarmupSteps: 0)
        let bogus: [[Float]] = [[1.0, 2.0, 3.0]] // single tiny tensor — wrong count
        do {
            try await trainer.loadTrainerWeights(bogus)
            XCTFail("Expected loadTrainerWeights to throw on wrong count")
        } catch let err as ChessTrainerError {
            switch err {
            case .trainerWeightCountMismatch:
                break  // expected
            default:
                XCTFail("Wrong error: \(err)")
            }
        }
    }

    /// ModelCheckpointFile decode must accept both v1 and v2 file
    /// versions. Verified by encoding a v2 file (the new default) and
    /// then loading it back — formatVersion on the decoded struct
    /// should be 2.
    func testModelCheckpointFileV2RoundTrip() throws {
        let now = Int64(Date().timeIntervalSince1970)
        let tinyWeights: [[Float]] = [[1.0, 2.0], [3.0]]
        let metadata = ModelCheckpointMetadata(
            creator: "momentum-test",
            trainingStep: 0,
            parentModelID: "",
            notes: "v2 round-trip fixture"
        )
        let original = ModelCheckpointFile(
            modelID: "test-modelid",
            createdAtUnix: now,
            metadata: metadata,
            weights: tinyWeights
        )
        XCTAssertEqual(original.formatVersion, ModelCheckpointFile.formatVersion,
                       "Default-constructed checkpoint should pick up the current write version")
        XCTAssertEqual(original.formatVersion, 2, "Current write version should be 2")

        let encoded = try original.encode()
        let decoded = try ModelCheckpointFile.decode(encoded)
        XCTAssertEqual(decoded.formatVersion, 2, "Decoded format version should be 2")
        XCTAssertEqual(decoded.weights.count, tinyWeights.count)
        XCTAssertEqual(decoded.weights[0], tinyWeights[0])
        XCTAssertEqual(decoded.weights[1], tinyWeights[1])
    }
}
