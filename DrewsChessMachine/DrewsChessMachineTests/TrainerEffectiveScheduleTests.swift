//
//  TrainerEffectiveScheduleTests.swift
//  DrewsChessMachineTests
//
//  The status bar shows the LR and momentum the optimizer is ACTUALLY being
//  fed each step. `ChessTrainer.effectiveLearningRate(...)` /
//  `effectiveMomentum(...)` are the UI-facing mirrors of the `buildFeeds`
//  resolution, so they must reflect the active LR/momentum cycle (not just
//  the static base values), fall back to the static values when cycling is
//  off, and still apply the warmup ramp. This pins that contract with pure-
//  math assertions over a single trainer build (no training steps run).
//

import XCTest
import Metal
@testable import DrewsChessMachine

final class TrainerEffectiveScheduleTests: XCTestCase {

    func test_effectiveLRandMomentum_cycleFallbackAndWarmup() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }

        // One trainer for the whole test (network build is the costly part).
        // 100-step warmup; no √batch scaling so effectiveLR == baseLR · warmupMul
        // and the cycle's contribution is isolated.
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 1),
            learningRate: 0.01,
            momentumCoeff: 0.5,
            sqrtBatchScalingForLR: false,
            lrWarmupSteps: 100,
            initialization: .seeded(initSeed: 1)
        )

        // --- Cycling off → static fallbacks (post-warmup so warmupMul == 1). ---
        trainer.lrMomentumCycle = .disabled
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: 200), 0.01, accuracy: 1e-6,
            "cycling off → effective LR is the static learningRate")
        XCTAssertEqual(
            trainer.effectiveMomentum(completedSteps: 200), 0.5, accuracy: 1e-6,
            "cycling off → effective momentum is the static momentumCoeff")

        // --- Both channels cycling (no invert), 1000-step period. ---
        // NB: `.constant` defaults momentumInvert = true; set both inverts
        // explicitly to false so the LR and momentum channels share the same
        // (un-inverted) cosine phase — min at the boundary, max at the midpoint.
        // Constant LR bounds (one keyframe), so only the phase moves.
        let cycle = LRMomentumCycle.constant(
            lrPeriodSteps: 1000, lrMin: 0.001, lrMax: 0.1, lrInvert: false,
            momentumEnabled: true, momentumPeriodSteps: 1000, momentumMin: 0.8, momentumMax: 0.95,
            momentumInvert: false)
        trainer.lrMomentumCycle = cycle

        // The cycle starts when the 100-step warmup ends, so global step 1100
        // is cycle step 1000 — a period boundary: LR at min, momentum at min.
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: 1100), Float(0.001), accuracy: 1e-6,
            "at the cycle boundary the effective LR is lrMin")
        XCTAssertEqual(
            trainer.effectiveMomentum(completedSteps: 1100), Float(0.8), accuracy: 1e-6,
            "at the cycle boundary the effective momentum is momentumMin")

        // Midpoint (global step 1600 → cycle step 1500 → phase 0.5): LR at max, momentum at max.
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: 1600), Float(0.1), accuracy: 1e-4,
            "at the cycle midpoint the effective LR is lrMax")
        XCTAssertEqual(
            trainer.effectiveMomentum(completedSteps: 1600), Float(0.95), accuracy: 1e-4,
            "at the cycle midpoint the effective momentum is momentumMax")

        // Must equal the cycle's own math at an arbitrary post-warmup step,
        // evaluated at the warmup-offset cycle step (the cycle evaluated with
        // no warmup there, so the offset is computed here, independently).
        let s = 1321
        let cycleStepForS = s - 100
        let atCycleStep = cycle.values(completedTrainSteps: cycleStepForS, lrWarmupSteps: 0)
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: s),
            Float(atCycleStep.learningRate ?? -1), accuracy: 1e-5,
            "effective LR must track LRMomentumCycle.values(...).learningRate")
        XCTAssertEqual(
            trainer.effectiveMomentum(completedSteps: s),
            Float(atCycleStep.momentum ?? -1), accuracy: 1e-5,
            "effective momentum must track LRMomentumCycle.values(...).momentum")

        // --- Warmup ramp composes on top of the cycled base LR. ---
        // Flat cycle (peak == trough) → base LR is exactly 0.05 at every step.
        let flat = LRMomentumCycle.constant(lrPeriodSteps: 1000, lrMin: 0.05, lrMax: 0.05)
        trainer.lrMomentumCycle = flat
        // Halfway through warmup (step 50 of 100): warmupMul 0.5 → 0.5 · 0.05.
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: 50), Float(0.025), accuracy: 1e-5,
            "mid-warmup effective LR is the ramped value of the cycled base LR (what the warm-up cell shows)")
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: 100), Float(0.05), accuracy: 1e-5,
            "after warmup the effective LR is the full cycled base LR")
    }
}
