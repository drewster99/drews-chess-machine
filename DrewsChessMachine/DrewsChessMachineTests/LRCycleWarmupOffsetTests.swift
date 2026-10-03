//
//  LRCycleWarmupOffsetTests.swift
//  DrewsChessMachineTests
//
//  The LR/momentum cycle begins when LR warmup ends. During warmup the LR
//  ramps linearly to the cycle's starting value and momentum holds the
//  cycle's starting value; afterwards both channels run on a step axis
//  shifted back by the warmup length. These tests pin that contract at the
//  pure-math level (`LRMomentumCycle.cycleStep` and the offset accessors)
//  and at the trainer level (`effectiveLearningRate` / `effectiveMomentum`,
//  which mirror the SGD feed resolution).
//

import XCTest
import Metal
@testable import DrewsChessMachine

final class LRCycleWarmupOffsetTests: XCTestCase {

    private let warmupSteps = 100
    private let period = 1000

    /// Inverted LR + inverted momentum: LR starts at its max, which is the
    /// case where an un-offset cycle would already be descending while
    /// warmup ramps up.
    private func invertedCycle(lrCount: Int = 0, momentumCount: Int = 0) -> LRMomentumCycle {
        var cycle = LRMomentumCycle.disabled
        cycle.lrEnabled = true
        cycle.lrMin = 0.001
        cycle.lrMax = 0.1
        cycle.lrPeriodSteps = period
        cycle.lrCount = lrCount
        cycle.lrInvert = true
        cycle.momentumEnabled = true
        cycle.momentumMin = 0.85
        cycle.momentumMax = 0.95
        cycle.momentumPeriodSteps = period
        cycle.momentumCount = momentumCount
        cycle.momentumInvert = false
        return cycle
    }

    func test_cycleStep_isZeroThroughWarmupThenOffset() {
        XCTAssertEqual(LRMomentumCycle.cycleStep(completedTrainSteps: 0, lrWarmupSteps: warmupSteps), 0)
        XCTAssertEqual(LRMomentumCycle.cycleStep(completedTrainSteps: warmupSteps - 1, lrWarmupSteps: warmupSteps), 0)
        XCTAssertEqual(LRMomentumCycle.cycleStep(completedTrainSteps: warmupSteps, lrWarmupSteps: warmupSteps), 0)
        XCTAssertEqual(LRMomentumCycle.cycleStep(completedTrainSteps: warmupSteps + 37, lrWarmupSteps: warmupSteps), 37)
        // No warmup: the cycle step is the global step.
        XCTAssertEqual(LRMomentumCycle.cycleStep(completedTrainSteps: 37, lrWarmupSteps: 0), 37)
    }

    func test_afterWarmup_lrCycleIsOffsetByWarmup() throws {
        let cycle = invertedCycle()
        for globalStep in [warmupSteps, warmupSteps + 1, warmupSteps + 250, warmupSteps + 500, warmupSteps + 1733] {
            let offset = try XCTUnwrap(cycle.learningRate(completedTrainSteps: globalStep, lrWarmupSteps: warmupSteps))
            let direct = try XCTUnwrap(cycle.learningRate(forStep: globalStep - warmupSteps))
            XCTAssertEqual(offset, direct, accuracy: 1e-15, "global step \(globalStep)")
        }
        // Inverted cycle: max at the cycle start, min at the cycle midpoint.
        XCTAssertEqual(
            try XCTUnwrap(cycle.learningRate(completedTrainSteps: warmupSteps, lrWarmupSteps: warmupSteps)),
            0.1, accuracy: 1e-12)
        XCTAssertEqual(
            try XCTUnwrap(cycle.learningRate(completedTrainSteps: warmupSteps + period / 2, lrWarmupSteps: warmupSteps)),
            0.001, accuracy: 1e-12)
    }

    func test_momentumCycleIsOffsetAndHeldDuringWarmup() throws {
        let cycle = invertedCycle()
        let momentumAtCycleStart = try XCTUnwrap(cycle.momentum(forStep: 0))
        for globalStep in [0, 1, warmupSteps / 2, warmupSteps - 1, warmupSteps] {
            XCTAssertEqual(
                try XCTUnwrap(cycle.momentum(completedTrainSteps: globalStep, lrWarmupSteps: warmupSteps)),
                momentumAtCycleStart, accuracy: 1e-15, "warmup step \(globalStep) holds cycle-start momentum")
        }
        for globalStep in [warmupSteps + 1, warmupSteps + 250, warmupSteps + 500, warmupSteps + 1733] {
            XCTAssertEqual(
                try XCTUnwrap(cycle.momentum(completedTrainSteps: globalStep, lrWarmupSteps: warmupSteps)),
                try XCTUnwrap(cycle.momentum(forStep: globalStep - warmupSteps)),
                accuracy: 1e-15, "global step \(globalStep)")
        }
        // Un-inverted momentum peaks at the cycle midpoint, which is offset.
        XCTAssertEqual(
            try XCTUnwrap(cycle.momentum(completedTrainSteps: warmupSteps + period / 2, lrWarmupSteps: warmupSteps)),
            0.95, accuracy: 1e-12)
    }

    func test_countFreeze_isMeasuredFromCycleStart() throws {
        let cycle = invertedCycle(lrCount: 1, momentumCount: 1)
        // One cycle after the cycle start (not after global step 0) the LR
        // freezes at the boundary (inverted → max). Just before that it is
        // still inside the first cycle, near its end.
        let justBeforeFreeze = warmupSteps + period - 1
        let lrJustBefore = try XCTUnwrap(cycle.learningRate(completedTrainSteps: justBeforeFreeze, lrWarmupSteps: warmupSteps))
        XCTAssertEqual(lrJustBefore, try XCTUnwrap(cycle.learningRate(forStep: period - 1)), accuracy: 1e-15)
        // At global step `period` (inside the first cycle once offset), the
        // cycle must NOT be frozen yet — it is at cycle step period - warmup.
        XCTAssertEqual(
            try XCTUnwrap(cycle.learningRate(completedTrainSteps: period, lrWarmupSteps: warmupSteps)),
            try XCTUnwrap(cycle.learningRate(forStep: period - warmupSteps)), accuracy: 1e-15)
        XCTAssertNotEqual(
            try XCTUnwrap(cycle.learningRate(completedTrainSteps: period + warmupSteps / 2, lrWarmupSteps: warmupSteps)),
            0.1, accuracy: 1e-6)
        for globalStep in [warmupSteps + period, warmupSteps + period + period / 2, warmupSteps + 5 * period] {
            XCTAssertEqual(
                try XCTUnwrap(cycle.learningRate(completedTrainSteps: globalStep, lrWarmupSteps: warmupSteps)),
                0.1, accuracy: 1e-12, "frozen LR at global step \(globalStep)")
            XCTAssertEqual(
                try XCTUnwrap(cycle.momentum(completedTrainSteps: globalStep, lrWarmupSteps: warmupSteps)),
                0.85, accuracy: 1e-12, "frozen momentum at global step \(globalStep)")
        }
    }

    func test_trainer_warmupRampEndsAtCycleStart_noDiscontinuity_andCyclingOffUnchanged() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }

        // No √batch scaling so the effective LR is cycle LR × warmup only.
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 1),
            learningRate: 0.01,
            momentumCoeff: 0.5,
            sqrtBatchScalingForLR: false,
            lrWarmupSteps: warmupSteps
        )
        let cycle = invertedCycle()
        trainer.lrMomentumCycle = cycle
        let lrAtCycleStart = try XCTUnwrap(cycle.learningRate(forStep: 0))
        let momentumAtCycleStart = try XCTUnwrap(cycle.momentum(forStep: 0))

        // The warmup ramp is a linear ramp to the cycle's starting LR.
        for globalStep in [0, 1, warmupSteps / 4, warmupSteps / 2, warmupSteps - 1] {
            let expected = Float(lrAtCycleStart) * Float(globalStep) / Float(warmupSteps)
            XCTAssertEqual(
                trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: globalStep),
                expected, accuracy: 1e-6, "warmup step \(globalStep)")
            XCTAssertEqual(
                trainer.effectiveMomentum(completedSteps: globalStep),
                Float(momentumAtCycleStart), accuracy: 1e-6, "warmup momentum at step \(globalStep)")
        }
        // The ramp lands exactly on the cycle's starting value.
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: warmupSteps),
            Float(lrAtCycleStart), accuracy: 1e-6)

        // No discontinuity across the boundary: each single-step change near
        // the boundary is no larger than one warmup increment, and the step
        // right after warmup is the cycle's first step.
        let warmupIncrement = Float(lrAtCycleStart) / Float(warmupSteps)
        var previous = trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: warmupSteps - 3)
        for globalStep in (warmupSteps - 2)...(warmupSteps + 3) {
            let current = trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: globalStep)
            XCTAssertLessThanOrEqual(abs(current - previous), warmupIncrement * 1.0001, "jump at step \(globalStep)")
            previous = current
        }
        XCTAssertEqual(
            trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: warmupSteps + 1),
            Float(try XCTUnwrap(cycle.learningRate(forStep: 1))), accuracy: 1e-6)

        // Cycling off: static LR × warmup, static momentum, as before.
        trainer.lrMomentumCycle = .disabled
        XCTAssertEqual(trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: warmupSteps / 2), 0.005, accuracy: 1e-7)
        XCTAssertEqual(trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: warmupSteps), 0.01, accuracy: 1e-7)
        XCTAssertEqual(trainer.effectiveLearningRate(forBatchSize: 256, completedSteps: 5 * warmupSteps), 0.01, accuracy: 1e-7)
        XCTAssertEqual(trainer.effectiveMomentum(completedSteps: warmupSteps / 2), 0.5, accuracy: 1e-7)
        XCTAssertEqual(trainer.effectiveMomentum(completedSteps: 5 * warmupSteps), 0.5, accuracy: 1e-7)
    }
}
