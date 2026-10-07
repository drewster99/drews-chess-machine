//
//  LRMomentumCycleReadoutPinTests.swift
//  DrewsChessMachineTests
//
//  `LRMomentumCycleReadout` is the one function that turns a schedule and a
//  trainer clock into the learning rate and momentum the SGD step is fed
//  (hyperparameter recording plan O-18): the lineage record's
//  `schedule_at_save` comes from it, and so do the trainer's own readouts
//  and feeds. This pins it to the trainer, bit for bit, across warmup,
//  √batch scaling, both cycle channels, the decay envelope and momentum
//  following, so the recorded values are the fed values and a later change
//  to either side cannot drift silently.
//

import Metal
import XCTest
@testable import DrewsChessMachine

final class LRMomentumCycleReadoutPinTests: XCTestCase {

    private static func cycles() -> [(String, LRMomentumCycle)] {
        let independent = LRMomentumCycle(
            lrEnabled: true, lrPeriodSteps: 200, lrCount: 0, lrMin: 0.001, lrMax: 0.02, lrInvert: false,
            momentumEnabled: true, momentumPeriodSteps: 300, momentumCount: 2, momentumMin: 0.8, momentumMax: 0.95,
            momentumInvert: true)
        var lrOnly = LRMomentumCycle.disabled
        lrOnly.lrEnabled = true
        var decayingAndFollowing = independent
        decayingAndFollowing.envelope = LRMomentumCycleEnvelope(
            lrPeakEnd: 0.004, lrTroughEnd: 0.0002, decayHorizonSteps: 1_000, momentumFollowsLRCycle: true,
            momentumFollowStartLow: 0.82, momentumFollowStartHigh: 0.93,
            momentumFollowEndLow: 0.88, momentumFollowEndHigh: 0.97)
        return [("disabled", .disabled), ("independent", independent), ("lrOnly", lrOnly),
                ("decayingAndFollowing", decayingAndFollowing)]
    }

    func testTheReadoutIsWhatTheTrainerReportsFeeding() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        let staticLearningRate: Float = 0.0007
        let staticMomentum: Float = 0.6
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 1), learningRate: staticLearningRate, momentumCoeff: staticMomentum,
            sqrtBatchScalingForLR: false, lrWarmupSteps: 0, initialization: .seeded(initSeed: 1))
        var compared = 0
        for (name, cycle) in Self.cycles() {
            trainer.lrMomentumCycle = cycle
            for warmup in [0, 50] {
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
                                            lrMomentumCycle: Self.cycles()[3].1)
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
