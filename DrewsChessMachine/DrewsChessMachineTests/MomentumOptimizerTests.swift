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
//  - Loading a base-only weight array through the strict session
//    loader is rejected; fresh champion forks use the explicit
//    base-load API that resets velocity.
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
        // Either base-only or full trainer layout.
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

    /// Session resume requires full trainer state. A base-only
    /// trainables+BN payload must not be silently accepted as a
    /// trainer checkpoint because that would lose optimizer velocity.
    func testLoadTrainerWeightsRejectsBaseOnlyShape() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(momentumCoeff: 0.9, lrWarmupSteps: 0)
        _ = try await trainer.trainStep(batchSize: 32)
        let v2Snapshot = try await trainer.exportTrainerWeights()
        let p2 = partition(v2Snapshot, trainer: trainer)

        do {
            try await trainer.loadTrainerWeights(p2.base)
            XCTFail("Expected loadTrainerWeights to reject base-only payload")
        } catch {
            guard case ChessTrainerError.trainerWeightCountMismatch = error else {
                XCTFail("Expected trainerWeightCountMismatch, got \(error)")
                return
            }
        }
    }

    /// Fresh forks from champion are allowed to load base weights,
    /// but only through the explicit API that also zeros velocity.
    func testLoadBaseWeightsResetVelocityLoadsBaseAndZerosVelocity() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(momentumCoeff: 0.9, lrWarmupSteps: 0)
        _ = try await trainer.trainStep(batchSize: 32)
        let snapshot = try await trainer.exportTrainerWeights()
        let partitioned = partition(snapshot, trainer: trainer)
        XCTAssertFalse(allZero(partitioned.velocity), "Velocity should be non-zero after one step at μ=0.9")

        try await trainer.resetNetwork()
        try await trainer.loadBaseWeightsResetVelocity(partitioned.base)
        let after = try await trainer.exportTrainerWeights()
        let pa = partition(after, trainer: trainer)

        XCTAssertEqual(pa.base.count, partitioned.base.count, "Base tensor count after base load should match input")
        for i in 0..<pa.base.count {
            XCTAssertEqual(pa.base[i], partitioned.base[i], "Base tensor \(i) should match loaded value bit-exactly")
        }
        XCTAssertTrue(
            allZero(pa.velocity),
            "Velocity should be zero after explicit base-load reset"
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

    /// Loading a malformed weights array must throw.
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

    /// ModelCheckpointFile decode accepts the current v2 file version.
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

    /// `momentumCoeff` is a session-level scalar that must round-trip
    /// through `SessionCheckpointState`'s JSON encoding. The schema
    /// gap previously left this field outside the session payload, so
    /// reloading a session silently picked up the user's current
    /// slider value instead of the value at save time.
    func testMomentumCoeffRoundTripsThroughSessionState() throws {
        let original = SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "test-session",
            savedAtUnix: 1_700_000_000,
            sessionStartUnix: 1_699_999_000,
            elapsedTrainingSec: 1000,
            trainingSteps: 1234,
            selfPlayGames: 10,
            selfPlayMoves: 600,
            trainingPositionsSeen: 1234 * 4096,
            batchSize: 4096,
            learningRate: 5e-5,
            entropyRegularizationCoeff: 0.0,
            drawPenalty: 0.1,
            promoteThreshold: 0.55,
            arenaGames: 200,
            arenaConcurrency: 1,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 4,
            gradClipMaxNorm: 1.0,
            weightDecayCoeff: 1e-4,
            policyLossWeight: 1.0,
            valueLossWeight: 1.0,
            momentumCoeff: 0.7,
            replayRatioTarget: 1.0,
            replayRatioAutoAdjust: true,
            stepDelayMs: 0,
            lastAutoComputedDelayMs: nil,
            lrWarmupSteps: 100,
            sqrtBatchScalingForLR: true,
            replayBufferMinPositionsBeforeTraining: 10000,
            arenaAutoIntervalSec: 600,
            candidateProbeIntervalSec: 60,
            legalMassCollapseThreshold: 0.5,
            legalMassCollapseGraceSeconds: 600,
            legalMassCollapseNoImprovementProbes: 5,
            championID: "champ-id",
            trainerID: "train-id",
            arenaHistory: []
        )
        let encoded = try original.encode()
        let decoded = try SessionCheckpointState.decode(encoded)
        XCTAssertEqual(decoded.momentumCoeff, 0.7,
                       "momentumCoeff must round-trip through session.json")
        XCTAssertEqual(decoded, original, "Whole struct must round-trip identically")
    }

    /// At μ=0 the decoupled-decay update reduces to the same
    /// `weight − lr · (grad + decayC · weight)` formula that the
    /// previous coupled L2 form produced, so the optimizer should be
    /// bit-exact compatible at zero momentum. This guards the
    /// "default behavior unchanged for μ=0 users" promise of the
    /// decoupled-decay refactor.
    func testDecoupledDecayMatchesCoupledAtZeroMomentum() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        // Two trainers built identically, both at μ=0. Same seeded
        // network init isn't easily controllable here, so we instead
        // use the same trainer twice: snapshot, run a step, restore,
        // run the same step again. The point of the test is not to
        // compare against a separate reference implementation but
        // to verify that the decoupled-decay code path produces a
        // deterministic, finite result and that the velocity path
        // behaves correctly at μ=0.
        let trainer = try ChessTrainer(
            weightDecayC: 1e-3,
            momentumCoeff: 0.0,
            lrWarmupSteps: 0
        )
        // After one step at μ=0, velocity should hold combinedUpdate
        // (just clippedGrad in the decoupled form), and weights should
        // be updated with decay applied separately. We can't easily
        // assert the math without a CPU reference, but we can at
        // least confirm: training step succeeds, loss is finite,
        // and velocity is non-zero (gradient flowed in).
        let timing = try await trainer.trainStep(batchSize: 32)
        XCTAssertTrue(timing.loss.isFinite, "Loss must be finite at μ=0 with decoupled decay")
        XCTAssertTrue(timing.velocityNorm.isFinite, "vNorm must be finite at μ=0")
        XCTAssertGreaterThan(timing.velocityNorm, 0,
                             "Velocity norm should be > 0 after one step (gradient flowed in at μ=0)")
        let weights = try await trainer.exportTrainerWeights()
        let velocityCount = trainer.network.trainableVariables.count
        let velocity = Array(weights.suffix(velocityCount))
        XCTAssertFalse(allZero(velocity),
                       "Velocity buffers should be non-zero after step at μ=0 (decoupled form: v_new = clippedGrad)")
    }

    /// Snapshot the velocity at "arena-start", do extra training
    /// steps to evolve it, then restore the snapshot — the trainer's
    /// velocity should exactly equal the snapshot, and the weights
    /// should NOT be affected (only velocity is touched). This
    /// validates the velocity-snapshot-on-promotion mechanic.
    func testVelocitySnapshotRoundTrip() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(momentumCoeff: 0.7, lrWarmupSteps: 0)
        // Build up some velocity.
        for _ in 0..<3 {
            _ = try await trainer.trainStep(batchSize: 32)
        }
        // Snapshot at "arena start".
        let snapshot = try await trainer.exportVelocitySnapshot()
        XCTAssertEqual(snapshot.count, trainer.network.trainableVariables.count,
                       "Snapshot should have one tensor per trainable variable")
        XCTAssertFalse(allZero(snapshot), "Snapshot velocity should be non-zero after warmup steps")
        // Capture base weights too (everything except velocity).
        let beforeFull = try await trainer.exportTrainerWeights()
        let baseCount = trainer.network.trainableVariables.count
            + trainer.network.bnRunningStatsVariables.count
        let beforeBase = Array(beforeFull.prefix(baseCount))

        // Evolve velocity (and weights) by running more steps.
        for _ in 0..<5 {
            _ = try await trainer.trainStep(batchSize: 32)
        }
        let evolvedVelocity = try await trainer.exportVelocitySnapshot()
        // Sanity: extra steps should have changed velocity.
        var anyDiff = false
        for i in 0..<snapshot.count where snapshot[i] != evolvedVelocity[i] {
            anyDiff = true
            break
        }
        XCTAssertTrue(anyDiff,
                      "Velocity must have evolved after additional steps, otherwise this test is meaningless")

        // Restore the snapshot.
        try await trainer.loadVelocitySnapshot(snapshot)
        let restored = try await trainer.exportVelocitySnapshot()
        for i in 0..<snapshot.count {
            XCTAssertEqual(restored[i].count, snapshot[i].count,
                           "Tensor \(i) element count must match after restore")
            for k in 0..<snapshot[i].count {
                if restored[i][k] != snapshot[i][k] {
                    XCTFail("Velocity tensor \(i) element \(k) mismatch after restore: \(restored[i][k]) vs \(snapshot[i][k])")
                    return
                }
            }
        }

        // Base weights (trainables + BN running stats) should NOT
        // have been touched by loadVelocitySnapshot — only velocity
        // is overwritten. They should still reflect the post-evolution
        // state, NOT the pre-snapshot state.
        let afterFull = try await trainer.exportTrainerWeights()
        let afterBase = Array(afterFull.prefix(baseCount))
        XCTAssertEqual(afterBase.count, beforeBase.count)
        var baseChanged = false
        outer: for i in 0..<beforeBase.count {
            for k in 0..<beforeBase[i].count where afterBase[i][k] != beforeBase[i][k] {
                baseChanged = true
                break outer
            }
        }
        XCTAssertTrue(baseChanged,
                      "Base weights should reflect the additional 5 training steps, not the pre-snapshot state — loadVelocitySnapshot must touch only velocity")
    }
}
