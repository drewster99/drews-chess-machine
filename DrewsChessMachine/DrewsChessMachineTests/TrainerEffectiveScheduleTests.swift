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
            learningRate: 0.01,
            momentumCoeff: 0.5,
            sqrtBatchScalingForLR: false,
            lrWarmupSteps: 100
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
        // NB: `.disabled` defaults momentumInvert = true; set both inverts
        // explicitly to false so the LR and momentum channels share the same
        // (un-inverted) cosine phase — min at the boundary, max at the midpoint.
        var cycle = LRMomentumCycle.disabled
        cycle.lrEnabled = true
        cycle.lrMin = 0.001
        cycle.lrMax = 0.1
        cycle.lrPeriodSteps = 1000
        cycle.lrInvert = false
        cycle.momentumEnabled = true
        cycle.momentumMin = 0.8
        cycle.momentumMax = 0.95
        cycle.momentumPeriodSteps = 1000
        cycle.momentumInvert = false
        trainer.lrMomentumCycle = cycle

        // Period boundary (step 1000, post-warmup): LR at min, momentum at min.
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: 1000), Float(0.001), accuracy: 1e-6,
            "at the cycle boundary the effective LR is lrMin")
        XCTAssertEqual(
            trainer.effectiveMomentum(completedSteps: 1000), Float(0.8), accuracy: 1e-6,
            "at the cycle boundary the effective momentum is momentumMin")

        // Midpoint (step 1500 → phase 0.5): LR at max, momentum at max.
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: 1500), Float(0.1), accuracy: 1e-4,
            "at the cycle midpoint the effective LR is lrMax")
        XCTAssertEqual(
            trainer.effectiveMomentum(completedSteps: 1500), Float(0.95), accuracy: 1e-4,
            "at the cycle midpoint the effective momentum is momentumMax")

        // Must equal the cycle's own math at an arbitrary post-warmup step.
        let s = 1321
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: s),
            Float(cycle.learningRate(forStep: s) ?? -1), accuracy: 1e-5,
            "effective LR must track LRMomentumCycle.learningRate(forStep:)")
        XCTAssertEqual(
            trainer.effectiveMomentum(completedSteps: s),
            Float(cycle.momentum(forStep: s) ?? -1), accuracy: 1e-5,
            "effective momentum must track LRMomentumCycle.momentum(forStep:)")

        // --- Warmup ramp composes on top of the cycled base LR. ---
        var flat = LRMomentumCycle.disabled
        flat.lrEnabled = true
        flat.lrMin = 0.05
        flat.lrMax = 0.05    // flat cycle → base LR is exactly 0.05 at every step
        flat.lrPeriodSteps = 1000
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
