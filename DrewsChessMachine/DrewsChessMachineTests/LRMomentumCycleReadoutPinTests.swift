//
//  LRMomentumCycleReadoutPinTests.swift
//  DrewsChessMachineTests
//
//  `LRMomentumCycleReadout` is the one function that turns a schedule and a
//  trainer clock into the learning rate and momentum the SGD step is fed
//  (hyperparameter recording plan O-18): the lineage record's
//  `schedule_at_save` comes from it, and so do the trainer's own readouts
//  and feeds. This pins it to the trainer, bit for bit, across warmup,
//  √batch scaling, both cycle channels, a decaying envelope and momentum
//  following, so the recorded values are the fed values and a later change
//  to either side cannot drift silently.
//
//  The schedules are the ones this test pinned before LR keyframes, written
//  in the previous form (`LegacyLRMomentumCycleReference`) and converted
//  exactly (LR_KEYFRAMES_PLAN.md, owner decision Q-4), so the values fed are
//  unchanged; each step also checks the cycle against the previous code.
//

import Metal
import XCTest
@testable import DrewsChessMachine

final class LRMomentumCycleReadoutPinTests: XCTestCase {

    /// The pinned schedules in their previous form, and each converted for a
    /// trainer with warmup `lrWarmupSteps` (the previous decay counted from
    /// the end of warmup, so the keyframes depend on it).
    private static func cycles(lrWarmupSteps: Int) throws
        -> [(name: String, legacy: LegacyLRMomentumCycleReference, cycle: LRMomentumCycle)] {
        let independent = LegacyLRMomentumCycleReference(
            lrEnabled: true, lrPeriodSteps: 200, lrCount: 0, lrMin: 0.001, lrMax: 0.02, lrInvert: false,
            momentumEnabled: true, momentumPeriodSteps: 300, momentumCount: 2, momentumMin: 0.8, momentumMax: 0.95,
            momentumInvert: true)
        var lrOnly = LegacyLRMomentumCycleReference.disabledReference
        lrOnly.lrEnabled = true
        var decayingAndFollowing = independent
        decayingAndFollowing.envelope = LegacyLRMomentumCycleEnvelopeReference(
            lrPeakEnd: 0.004, lrTroughEnd: 0.0002, decayHorizonSteps: 1_000, momentumFollowsLRCycle: true,
            momentumFollowStartLow: 0.82, momentumFollowStartHigh: 0.93,
            momentumFollowEndLow: 0.88, momentumFollowEndHigh: 0.97)
        return [
            ("disabled", .disabledReference, .disabled),
            ("independent", independent, try .converted(from: independent, lrWarmupSteps: lrWarmupSteps)),
            ("lrOnly", lrOnly, try .converted(from: lrOnly, lrWarmupSteps: lrWarmupSteps)),
            ("decayingAndFollowing", decayingAndFollowing,
             try .converted(from: decayingAndFollowing, lrWarmupSteps: lrWarmupSteps)),
        ]
    }

    func testTheReadoutIsWhatTheTrainerReportsFeeding() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        let staticLearningRate: Float = 0.0007
        let staticMomentum: Float = 0.6
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 1), learningRate: staticLearningRate, momentumCoeff: staticMomentum,
            sqrtBatchScalingForLR: false, lrWarmupSteps: 0, initialization: .seeded(initSeed: 1))
        var compared = 0
        for warmup in [0, 50] {
            for (name, legacy, cycle) in try Self.cycles(lrWarmupSteps: warmup) {
                trainer.lrMomentumCycle = cycle
                trainer.lrWarmupSteps = warmup
                for sqrtScaling in [false, true] {
                    trainer.sqrtBatchScalingForLR = sqrtScaling
                    for batchSize in [256, 4096] {
                        for step in [0, 1, 25, 49, 50, 51, 199, 250, 777, 1_049, 1_050, 5_000] {
                            let fed = LRMomentumCycleReadout.values(
                                completedTrainSteps: step, lrWarmupSteps: warmup, cycle: cycle,
                                staticLearningRate: staticLearningRate, staticMomentum: staticMomentum,
                                batchSize: batchSize, sqrtBatchScaling: sqrtScaling,
                                sqrtScaleBaseBatchSize: ChessTrainer.sqrtScaleBaseBatchSize)
                            let context = "\(name) warmup=\(warmup) sqrt=\(sqrtScaling) batch=\(batchSize) step=\(step)"
                            XCTAssertEqual(fed.learningRate.bitPattern,
                                           trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: step).bitPattern,
                                           "learning rate, \(context)")
                            XCTAssertEqual(fed.momentum.bitPattern,
                                           trainer.effectiveMomentum(completedSteps: step).bitPattern,
                                           "momentum, \(context)")
                            XCTAssertEqual(fed.cycleStep,
                                           LRMomentumCycle.cycleStep(completedTrainSteps: step, lrWarmupSteps: warmup),
                                           "cycle step, \(context)")
                            // The schedule itself is the one pinned before LR keyframes.
                            let now = cycle.values(completedTrainSteps: step, lrWarmupSteps: warmup)
                            let before = legacy.values(completedTrainSteps: step, lrWarmupSteps: warmup)
                            XCTAssertEqual(now.learningRate.map(\.bitPattern), before.learningRate.map(\.bitPattern),
                                           "cycle learning rate vs the previous code, \(context)")
                            XCTAssertEqual(now.momentum.map(\.bitPattern), before.momentum.map(\.bitPattern),
                                           "cycle momentum vs the previous code, \(context)")
                            compared += 1
                        }
                    }
                }
            }
        }
        XCTAssertEqual(compared, 4 * 2 * 2 * 2 * 12)
    }

    /// `schedule_at_save` is the readout of the record's own inputs: the
    /// in-force snapshot's schedule, static values and batch size.
    func testScheduleAtSaveReadsTheSnapshotItDescribes() throws {
        let schedule = TrainerScheduleState(completedTrainSteps: 0, lrWarmupSteps: 40,
                                            lrMomentumCycle: try Self.cycles(lrWarmupSteps: 40)[3].cycle)
        let inForce = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            LearningRate.id: LearningRate.encode(0.0007), MomentumCoeff.id: MomentumCoeff.encode(0.6),
            TrainingBatchSize.id: TrainingBatchSize.encode(1024), SqrtBatchScalingLR.id: SqrtBatchScalingLR.encode(true),
        ]).adoptingSchedule(schedule)
        for step in [0, 20, 40, 41, 900, 1_100] {
            let atSave = LRMomentumCycleReadout.scheduleAtSave(inForce: inForce, completedTrainSteps: step)
            let fed = LRMomentumCycleReadout.values(
                completedTrainSteps: step, lrWarmupSteps: 40, cycle: inForce.lrMomentumCycle,
                staticLearningRate: Float(0.0007), staticMomentum: Float(0.6), batchSize: 1024,
                sqrtBatchScaling: true, sqrtScaleBaseBatchSize: ChessTrainer.sqrtScaleBaseBatchSize)
            XCTAssertEqual(atSave.cycleStep, fed.cycleStep, "step \(step)")
            XCTAssertEqual(atSave.learningRateFed, Double(fed.learningRate), "step \(step)")
            XCTAssertEqual(atSave.momentumFed, Double(fed.momentum), "step \(step)")
        }
    }
}
