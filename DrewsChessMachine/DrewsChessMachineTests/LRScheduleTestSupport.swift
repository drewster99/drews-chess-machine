//
//  LRScheduleTestSupport.swift
//  DrewsChessMachineTests
//
//  Builds a keyframed `LRMomentumCycle` from the single-decay settings tests
//  were written against (`LegacyLRMomentumCycleReference`), through the app's
//  own exact converter — so a test's expected numbers stay the ones it always
//  pinned (LR_KEYFRAMES_PLAN.md, owner decision Q-4).
//

import Foundation
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

extension LegacyLRSchedule {
    init(reference old: LegacyLRMomentumCycleReference, lrWarmupSteps: Int) {
        let e = old.envelope
        self.init(
            lrEnabled: old.lrEnabled, lrPeriodSteps: old.lrPeriodSteps, lrCount: old.lrCount,
            lrMin: old.lrMin, lrMax: old.lrMax, lrInvert: old.lrInvert,
            momentumEnabled: old.momentumEnabled, momentumPeriodSteps: old.momentumPeriodSteps,
            momentumCount: old.momentumCount, momentumMin: old.momentumMin, momentumMax: old.momentumMax,
            momentumInvert: old.momentumInvert, lrPeakEnd: e.lrPeakEnd, lrTroughEnd: e.lrTroughEnd,
            decayHorizonSteps: e.decayHorizonSteps, momentumFollowsLRCycle: e.momentumFollowsLRCycle,
            momentumFollowStartLow: e.momentumFollowStartLow, momentumFollowStartHigh: e.momentumFollowStartHigh,
            momentumFollowEndLow: e.momentumFollowEndLow, momentumFollowEndHigh: e.momentumFollowEndHigh,
            lrWarmupSteps: lrWarmupSteps)
    }
}

extension LRMomentumCycle {
    /// `old` as a keyframed cycle for a trainer whose warmup is
    /// `lrWarmupSteps` (the old decay counted from the end of warmup).
    /// Throws only for a schedule the keyframes cannot express (follow bounds
    /// inverted on part of the run).
    static func converted(from old: LegacyLRMomentumCycleReference, lrWarmupSteps: Int = 0) throws -> LRMomentumCycle {
        try LRMomentumCycle.converting(LegacyLRSchedule(reference: old, lrWarmupSteps: lrWarmupSteps)).cycle
    }

    /// A cycle with one constant keyframe: peak `lrMax`, trough `lrMin`,
    /// follow momentum `momentumFollowLow` / `momentumFollowHigh` — the
    /// shape every no-decay test used.
    static func constant(lrEnabled: Bool = true, lrPeriodSteps: Int, lrCount: Int = 0, lrMin: Double, lrMax: Double,
                         lrInvert: Bool = false, momentumEnabled: Bool = false, momentumPeriodSteps: Int = 2000,
                         momentumCount: Int = 0, momentumMin: Double = 0.85, momentumMax: Double = 0.95,
                         momentumInvert: Bool = true, momentumFollowsLRCycle: Bool = false,
                         momentumFollowLow: Double = 0.85, momentumFollowHigh: Double = 0.95) -> LRMomentumCycle {
        LRMomentumCycle(
            lrEnabled: lrEnabled, lrPeriodSteps: lrPeriodSteps, lrCount: lrCount, lrInvert: lrInvert,
            lrKeyframes: LRKeyframeSchedule(keyframes: [
                LRKeyframe(step: 0, peak: lrMax, trough: lrMin, momentumAtPeak: momentumFollowLow, momentumAtTrough: momentumFollowHigh),
            ]),
            momentumEnabled: momentumEnabled, momentumPeriodSteps: momentumPeriodSteps, momentumCount: momentumCount,
            momentumMin: momentumMin, momentumMax: momentumMax, momentumInvert: momentumInvert,
            momentumFollowsLRCycle: momentumFollowsLRCycle)
    }
}
