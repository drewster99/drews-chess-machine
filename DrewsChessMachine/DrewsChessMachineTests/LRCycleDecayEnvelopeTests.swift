//
//  LRCycleDecayEnvelopeTests.swift
//  DrewsChessMachineTests
//
//  Correctness tests for the LR cycle's decaying envelope and the
//  momentum-follows-LR mode (see `LRMomentumCycle.values(completedTrainSteps:
//  lrWarmupSteps:)`). The schedule is a pure function of the step, so these
//  pin its behavior exactly: the peak/trough decay endpoints and midpoint,
//  the hold after the horizon, that a zero horizon reproduces the
//  pre-envelope schedule, geometric in-cycle interpolation, momentum's
//  inverse phase and endpoint drift, the follow switch, and the warmup offset.
//
//  LR keyframes replaced the single decay envelope
//  (documentation/plans-active/LR_KEYFRAMES_PLAN.md). The schedules here are
//  still written in the previous form (`LegacyLRMomentumCycleReference`) and
//  converted exactly (`LRMomentumCycle.converted(from:lrWarmupSteps:)`), so
//  every expected number is the one these tests always pinned. With no
//  warmup the cycle step is the trainer step, so `scheduleValues(_:atCycleStep:)` evaluates at
//  the old cycle step.
//

import XCTest
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

final class LRCycleDecayEnvelopeTests: XCTestCase {

    private let relativeTolerance = 1e-9

    private let peakStart = 1.0e-1
    private let peakEnd = 1.0e-4
    private let troughStart = 1.0e-3
    private let troughEnd = 1.0e-6
    private let horizon = 1_000_000
    private let period = 20_000

    /// The owner-approved default schedule: inverted LR cycle (starts at the
    /// peak), decaying envelope, momentum following the LR cycle.
    private func makeLegacyDecayingSchedule(momentumFollows: Bool = true) -> LegacyLRMomentumCycleReference {
        var cycle = LegacyLRMomentumCycleReference.disabledReference
        cycle.lrEnabled = true
        cycle.lrMin = troughStart
        cycle.lrMax = peakStart
        cycle.lrPeriodSteps = period
        cycle.lrCount = 0
        cycle.lrInvert = true
        cycle.momentumEnabled = true
        cycle.momentumMin = 0.80
        cycle.momentumMax = 0.98
        cycle.momentumPeriodSteps = 3_000
        cycle.momentumCount = 0
        cycle.momentumInvert = false
        cycle.envelope = LegacyLRMomentumCycleEnvelopeReference(
            lrPeakEnd: peakEnd,
            lrTroughEnd: troughEnd,
            decayHorizonSteps: horizon,
            momentumFollowsLRCycle: momentumFollows,
            momentumFollowStartLow: 0.85,
            momentumFollowStartHigh: 0.95,
            momentumFollowEndLow: 0.90,
            momentumFollowEndHigh: 0.95
        )
        return cycle
    }

    /// `makeLegacyDecayingSchedule`, converted to keyframes for a trainer
    /// with warmup `lrWarmupSteps`.
    private func makeDecayingSchedule(momentumFollows: Bool = true, lrWarmupSteps: Int = 0) throws -> LRMomentumCycle {
        try LRMomentumCycle.converted(from: makeLegacyDecayingSchedule(momentumFollows: momentumFollows),
                                      lrWarmupSteps: lrWarmupSteps)
    }

    /// The schedule at cycle step `cycleStep` of a cycle converted with no
    /// warmup (cycle step == trainer step).
    private func scheduleValues(_ schedule: LRMomentumCycle, atCycleStep cycleStep: Int) -> LRMomentumCycle.Values {
        schedule.values(completedTrainSteps: cycleStep, lrWarmupSteps: 0)
    }

    private func assertRelativelyEqual(
        _ actual: Double?,
        _ expected: Double,
        _ message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard let actual else {
            XCTFail("\(message): got nil, expected \(expected)", file: file, line: line)
            return
        }
        XCTAssertEqual(actual, expected, accuracy: abs(expected) * relativeTolerance, message, file: file, line: line)
    }

    // MARK: - Envelope endpoints and midpoint

    func testPeakAndTroughAtStartOfHorizon() throws {
        let values = scheduleValues(try makeDecayingSchedule(), atCycleStep: 0)
        assertRelativelyEqual(values.lrPeak, peakStart, "peak at t=0")
        assertRelativelyEqual(values.lrTrough, troughStart, "trough at t=0")
        // Inverted: the cycle opens at its peak.
        assertRelativelyEqual(values.learningRate, peakStart, "LR at t=0 is the peak")
    }

    func testPeakAndTroughAtHalfHorizonAreGeometricMidpoints() throws {
        let values = scheduleValues(try makeDecayingSchedule(), atCycleStep: horizon / 2)
        assertRelativelyEqual(values.lrPeak, (peakStart * peakEnd).squareRoot(), "peak at t=horizon/2")
        assertRelativelyEqual(values.lrTrough, (troughStart * troughEnd).squareRoot(), "trough at t=horizon/2")
        // horizon/2 is a whole number of periods, so the cycle is at its peak.
        XCTAssertEqual((horizon / 2) % period, 0)
        assertRelativelyEqual(values.learningRate, (peakStart * peakEnd).squareRoot(), "LR at t=horizon/2")
    }

    func testPeakAndTroughAtEndOfHorizon() throws {
        let values = scheduleValues(try makeDecayingSchedule(), atCycleStep: horizon)
        assertRelativelyEqual(values.lrPeak, peakEnd, "peak at t=horizon")
        assertRelativelyEqual(values.lrTrough, troughEnd, "trough at t=horizon")
        XCTAssertEqual(horizon % period, 0)
        assertRelativelyEqual(values.learningRate, peakEnd, "LR at t=horizon")
    }

    func testEnvelopeHoldsAfterHorizonAndKeepsCycling() throws {
        let schedule = try makeDecayingSchedule()
        for extraPeriods in [1, 7, 50] {
            let peakStep = horizon + extraPeriods * period
            let atPeak = scheduleValues(schedule, atCycleStep: peakStep)
            assertRelativelyEqual(atPeak.lrPeak, peakEnd, "peak held at \(peakStep)")
            assertRelativelyEqual(atPeak.lrTrough, troughEnd, "trough held at \(peakStep)")
            assertRelativelyEqual(atPeak.learningRate, peakEnd, "LR at a post-horizon cycle start")
            // Half a period later the inverted cycle is at its trough.
            let troughStep = peakStep + period / 2
            assertRelativelyEqual(
                scheduleValues(schedule, atCycleStep: troughStep).learningRate,
                troughEnd,
                "LR at a post-horizon cycle midpoint"
            )
        }
    }

    /// LR keyframes replaced the decay envelope and its `decayFraction`
    /// helper; this pins the same clamping on the keyframe bounds instead:
    /// fraction 0 before the first keyframe, 0.5 halfway, 1 at and after the
    /// last keyframe, and a single keyframe (zero horizon) never decays.
    func testDecayFractionClampsAndZeroHorizonIsNoDecay() throws {
        let start = LRKeyframe(step: 0, peak: peakStart, trough: troughStart, momentumAtPeak: 0.85, momentumAtTrough: 0.95)
        let end = LRKeyframe(step: 100, peak: peakEnd, trough: troughEnd, momentumAtPeak: 0.90, momentumAtTrough: 0.95)
        let decaying = try LRKeyframeSchedule(keyframes: [start, end]).validated()
        func assertBounds(_ bounds: LRKeyframeSchedule.Bounds, atFraction f: Double, _ message: String,
                          line: UInt = #line) {
            XCTAssertEqual(bounds.peak, peakStart * pow(peakEnd / peakStart, f), "peak, \(message)", line: line)
            XCTAssertEqual(bounds.trough, troughStart * pow(troughEnd / troughStart, f), "trough, \(message)", line: line)
            XCTAssertEqual(bounds.momentumAtPeak, 0.85 + (0.90 - 0.85) * f, "momentum at peak, \(message)", line: line)
            XCTAssertEqual(bounds.momentumAtTrough, 0.95, "momentum at trough, \(message)", line: line)
        }
        assertBounds(decaying.bounds(atTrainerStep: -5), atFraction: 0.0, "step -5")
        assertBounds(decaying.bounds(atTrainerStep: 50), atFraction: 0.5, "step 50")
        assertBounds(decaying.bounds(atTrainerStep: 100), atFraction: 1.0, "step 100")
        assertBounds(decaying.bounds(atTrainerStep: 10_000), atFraction: 1.0, "step 10000")
        let noDecay = try LRKeyframeSchedule(keyframes: [start]).validated()
        assertBounds(noDecay.bounds(atTrainerStep: 10_000), atFraction: 0.0, "single keyframe, step 10000")
    }

    // MARK: - Horizon 0 is the pre-envelope behavior

    func testZeroHorizonReproducesPreEnvelopeSchedule() throws {
        var legacy = makeLegacyDecayingSchedule(momentumFollows: false)
        legacy.envelope.decayHorizonSteps = 0
        let schedule = try LRMomentumCycle.converted(from: legacy)
        let steps = [0, 1, 2_500, 5_000, 9_999, 10_000, 17_321, 20_000, 123_457, 5_000_003]
        for step in steps {
            // The pre-envelope LR formula, written out independently.
            let lrFraction = LRMomentumCycle.cycleFraction(step: step, period: period, count: 0, invert: true)
            let expectedLR = troughStart * pow(peakStart / troughStart, lrFraction)
            let momentumFraction = LRMomentumCycle.cycleFraction(step: step, period: 3_000, count: 0, invert: false)
            let expectedMomentum = 0.80 + (0.98 - 0.80) * momentumFraction
            let values = scheduleValues(schedule, atCycleStep: step)
            XCTAssertEqual(values.learningRate, expectedLR, "LR at step \(step) must match the pre-envelope formula exactly")
            XCTAssertEqual(values.momentum, expectedMomentum, "momentum at step \(step) must match the pre-envelope formula exactly")
            XCTAssertEqual(values.lrPeak, peakStart)
            XCTAssertEqual(values.lrTrough, troughStart)
        }
    }

    /// LR keyframes replaced the envelope type: the all-off cycle now has one
    /// keyframe (no decay) and momentum following off, and the previous
    /// all-off cycle converts to exactly it.
    func testDefaultEnvelopeIsNoDecayAndNoFollow() throws {
        let disabled = LRMomentumCycle.disabled
        XCTAssertEqual(disabled.lrKeyframes.keyframes.count, 1)
        XCTAssertFalse(disabled.momentumFollowsLRCycle)
        XCTAssertEqual(LegacyLRMomentumCycleEnvelopeReference.noDecay.decayHorizonSteps, 0)
        XCTAssertEqual(try LRMomentumCycle.converted(from: .disabledReference), disabled)
    }

    // MARK: - Geometric interpolation within a cycle

    func testInCycleInterpolationIsGeometricBetweenCurrentPeakAndTrough() throws {
        let schedule = try makeDecayingSchedule()
        // A quarter period into a cycle the cosine fraction is exactly one
        // half, so the LR is the geometric mean of the current bounds.
        for cycleStart in [0, 40_000, 480_000, horizon, horizon + 200_000] {
            let step = cycleStart + period / 4
            let values = scheduleValues(schedule, atCycleStep: step)
            guard let peak = values.lrPeak, let trough = values.lrTrough else {
                XCTFail("envelope bounds missing at \(step)")
                continue
            }
            assertRelativelyEqual(values.learningRate, (peak * trough).squareRoot(), "quarter-period LR at \(step)")
        }
        // At an arbitrary phase the LR is trough·(peak/trough)^fraction.
        let step = 123_457
        let values = scheduleValues(schedule, atCycleStep: step)
        let decay = Double(step) / Double(horizon)
        let expectedPeak = peakStart * pow(peakEnd / peakStart, decay)
        let expectedTrough = troughStart * pow(troughEnd / troughStart, decay)
        let fraction = LRMomentumCycle.cycleFraction(step: step, period: period, count: 0, invert: true)
        assertRelativelyEqual(values.lrPeak, expectedPeak, "peak at \(step)")
        assertRelativelyEqual(values.lrTrough, expectedTrough, "trough at \(step)")
        assertRelativelyEqual(values.learningRate, expectedTrough * pow(expectedPeak / expectedTrough, fraction), "LR at \(step)")
    }

    func testUnusableEndpointsDisableTheLRChannel() throws {
        // Keyframes cannot hold a peak below its trough; the previous settings
        // convert to a disabled LR channel, which feeds what they fed.
        var legacy = makeLegacyDecayingSchedule()
        legacy.envelope.lrPeakEnd = 1.0e-7
        legacy.envelope.lrTroughEnd = 1.0e-5
        let schedule = try LRMomentumCycle.converted(from: legacy)
        XCTAssertFalse(schedule.lrEnabled)
        let values = scheduleValues(schedule, atCycleStep: 0)
        XCTAssertNil(values.learningRate, "an end peak below the end trough must fall back to the static LR")
        XCTAssertNil(values.lrPeak)
        XCTAssertNil(values.lrTrough)
        XCTAssertNil(values.momentum, "following momentum has no phase without a usable LR cycle")
    }

    // MARK: - Momentum follows the LR cycle

    func testFollowingMomentumIsLowAtLRPeakAndHighAtLRTrough() throws {
        let schedule = try makeDecayingSchedule()
        // Start of horizon.
        assertRelativelyEqual(scheduleValues(schedule, atCycleStep: 0).momentum, 0.85, "momentum at the first LR peak")
        assertRelativelyEqual(
            scheduleValues(schedule, atCycleStep: period / 2).momentum,
            0.95,
            "momentum at the first LR trough"
        )
        // Quarter period: the LR fraction is one half, so momentum is mid-band.
        let quarter = scheduleValues(schedule, atCycleStep: period / 4)
        let decay = Double(period / 4) / Double(horizon)
        let low = 0.85 + (0.90 - 0.85) * decay
        let high = 0.95
        assertRelativelyEqual(quarter.momentum, (low + high) / 2, "momentum at a quarter period")
    }

    func testFollowingMomentumBoundsDriftLinearlyOverHorizonThenHold() throws {
        let schedule = try makeDecayingSchedule()
        assertRelativelyEqual(scheduleValues(schedule, atCycleStep: horizon / 2).momentum, 0.875, "low bound at t=horizon/2")
        assertRelativelyEqual(scheduleValues(schedule, atCycleStep: horizon).momentum, 0.90, "low bound at t=horizon")
        assertRelativelyEqual(
            scheduleValues(schedule, atCycleStep: horizon + period / 2).momentum,
            0.95,
            "high bound after the horizon"
        )
        assertRelativelyEqual(
            scheduleValues(schedule, atCycleStep: horizon + 30 * period).momentum,
            0.90,
            "low bound held after the horizon"
        )
    }

    func testFollowingMomentumIgnoresTheIndependentMomentumCycle() throws {
        var a = try makeDecayingSchedule()
        var b = try makeDecayingSchedule()
        a.momentumPeriodSteps = 7
        a.momentumMin = 0.1
        a.momentumInvert = true
        b.momentumPeriodSteps = 99_999
        b.momentumMax = 0.99
        for step in [0, 3, 5_000, 250_001, 2_000_000] {
            XCTAssertEqual(scheduleValues(a, atCycleStep: step).momentum, scheduleValues(b, atCycleStep: step).momentum)
        }
    }

    func testFollowingMomentumNeedsBothMomentumCyclingAndAnLRCycle() throws {
        var noMomentumCycling = try makeDecayingSchedule()
        noMomentumCycling.momentumEnabled = false
        XCTAssertNil(scheduleValues(noMomentumCycling, atCycleStep: 0).momentum)

        var noLRCycle = try makeDecayingSchedule()
        noLRCycle.lrEnabled = false
        XCTAssertNil(scheduleValues(noLRCycle, atCycleStep: 0).momentum)
        XCTAssertNil(scheduleValues(noLRCycle, atCycleStep: 0).learningRate)
    }

    /// LR keyframes made inverted follow bounds a validation error, so they
    /// no longer reach the evaluator from any stored schedule. Still pinned:
    /// the evaluator reports no momentum for them (its guard, for an
    /// unvalidated schedule); the validator refuses them; and the previous
    /// settings, inverted only at the start of the horizon, are refused by
    /// the converter (no keyframe schedule reproduces them).
    func testInvertedFollowBoundsReportNoMomentum() throws {
        var schedule = try makeDecayingSchedule()
        schedule.lrKeyframes.keyframes[0].momentumAtPeak = 0.97
        XCTAssertNil(scheduleValues(schedule, atCycleStep: 0).momentum)
        XCTAssertThrowsError(try schedule.lrKeyframes.validated()) { error in
            XCTAssertEqual(error as? LRKeyframeScheduleError,
                           .momentumAtPeakAboveTrough(index: 0, atPeak: 0.97, atTrough: 0.95))
        }

        var legacy = makeLegacyDecayingSchedule()
        legacy.envelope.momentumFollowStartLow = 0.97
        XCTAssertThrowsError(try LRMomentumCycle.converted(from: legacy)) { error in
            XCTAssertTrue(error is LegacyLRScheduleConversionError, "\(error)")
        }
    }

    // MARK: - Follow switch off

    func testFollowOffLeavesTheIndependentMomentumCycleUnchanged() throws {
        let schedule = try makeDecayingSchedule(momentumFollows: false)
        for step in [0, 1, 750, 1_500, 2_999, 3_000, 500_000, horizon, horizon + 4_321] {
            let fraction = LRMomentumCycle.cycleFraction(step: step, period: 3_000, count: 0, invert: false)
            XCTAssertEqual(
                scheduleValues(schedule, atCycleStep: step).momentum,
                0.80 + (0.98 - 0.80) * fraction,
                "independent momentum at step \(step)"
            )
        }
    }

    // MARK: - Warmup offset

    func testWarmupOffsetStillApplies() throws {
        // The keyframes are on the trainer step, so the schedule is converted
        // for this warmup (the previous decay counted from the end of warmup)
        // and compared with the same schedule converted for no warmup, where
        // the cycle step is the trainer step.
        let warmup = 5_000
        let schedule = try makeDecayingSchedule(lrWarmupSteps: warmup)
        let noWarmup = try makeDecayingSchedule()
        // Inside warmup every step maps to cycle step 0.
        for global in [0, 1, warmup / 2, warmup] {
            XCTAssertEqual(
                schedule.values(completedTrainSteps: global, lrWarmupSteps: warmup),
                scheduleValues(noWarmup, atCycleStep: 0),
                "global step \(global) inside warmup"
            )
        }
        // After warmup the schedule is shifted back by the warmup length,
        // including the decay, which is measured in cycle steps.
        for cycleStep in [1, period / 4, horizon / 2, horizon, horizon + period / 2] {
            XCTAssertEqual(
                schedule.values(completedTrainSteps: warmup + cycleStep, lrWarmupSteps: warmup),
                scheduleValues(noWarmup, atCycleStep: cycleStep),
                "cycle step \(cycleStep) after warmup"
            )
        }
        assertRelativelyEqual(
            schedule.values(completedTrainSteps: warmup + horizon, lrWarmupSteps: warmup).learningRate,
            peakEnd,
            "the horizon ends a full horizon after warmup, not after step zero"
        )
        assertRelativelyEqual(
            schedule.values(completedTrainSteps: warmup + horizon, lrWarmupSteps: warmup).momentum,
            0.90,
            "momentum reaches its end low bound a full horizon after warmup"
        )
    }

    // MARK: - Persistence shape

    /// LR keyframes replaced the separately persisted envelope: the cycle's
    /// one encoded form now carries the keyframes and the follow flag, and
    /// round-trips exactly.
    func testCycleEncodingOmitsTheEnvelopeSoOlderSessionsStillDecode() throws {
        let schedule = try makeDecayingSchedule()
        let encoded = try JSONEncoder().encode(schedule)
        let object = try JSONSerialization.jsonObject(with: encoded)
        guard let dictionary = object as? [String: Any] else {
            XCTFail("encoded cycle is not a JSON object")
            return
        }
        XCTAssertNil(dictionary["envelope"], "no separate envelope any more")
        XCTAssertNotNil(dictionary["lr_keyframes"], "the decay is carried by the keyframes")
        XCTAssertEqual(dictionary["momentum_follows_lr_cycle"] as? Bool, true)
        XCTAssertEqual(dictionary["schedule_format"] as? Int, LRMomentumCycle.currentScheduleFormat)
        let decoded = try JSONDecoder().decode(LRMomentumCycle.self, from: encoded)
        XCTAssertEqual(decoded, schedule)
    }

    /// LR keyframes replaced the session's separate envelope field: a session
    /// written before the envelope existed carries only the cycle, and its
    /// conversion is no decay (one keyframe at the cycle's bounds) and no
    /// momentum following; with the envelope present, its decay is kept.
    func testMissingSavedEnvelopeResolvesToNoDecayAndNoFollow() throws {
        let legacy = makeLegacyDecayingSchedule()
        let savedCycle = LegacyLRMomentumCycleCoding(
            lrEnabled: legacy.lrEnabled, lrPeriodSteps: legacy.lrPeriodSteps, lrCount: legacy.lrCount,
            lrMin: legacy.lrMin, lrMax: legacy.lrMax, lrInvert: legacy.lrInvert,
            momentumEnabled: legacy.momentumEnabled, momentumPeriodSteps: legacy.momentumPeriodSteps,
            momentumCount: legacy.momentumCount, momentumMin: legacy.momentumMin, momentumMax: legacy.momentumMax,
            momentumInvert: legacy.momentumInvert)

        let withoutEnvelope = try LRMomentumCycle.converting(
            LegacyLRSchedule(cycle: savedCycle, envelope: nil, lrWarmupSteps: 0)).cycle
        XCTAssertEqual(withoutEnvelope.lrKeyframes.keyframes.count, 1)
        XCTAssertFalse(withoutEnvelope.momentumFollowsLRCycle)
        let bounds = withoutEnvelope.lrKeyframes.bounds(atTrainerStep: 10_000_000)
        XCTAssertEqual(bounds.peak, peakStart)
        XCTAssertEqual(bounds.trough, troughStart)

        let e = legacy.envelope
        let savedEnvelope = LegacyLRMomentumCycleEnvelopeCoding(
            lrPeakEnd: e.lrPeakEnd, lrTroughEnd: e.lrTroughEnd, decayHorizonSteps: e.decayHorizonSteps,
            momentumFollowsLRCycle: e.momentumFollowsLRCycle,
            momentumFollowStartLow: e.momentumFollowStartLow, momentumFollowStartHigh: e.momentumFollowStartHigh,
            momentumFollowEndLow: e.momentumFollowEndLow, momentumFollowEndHigh: e.momentumFollowEndHigh)
        let withEnvelope = try LRMomentumCycle.converting(
            LegacyLRSchedule(cycle: savedCycle, envelope: savedEnvelope, lrWarmupSteps: 0)).cycle
        XCTAssertEqual(withEnvelope, try makeDecayingSchedule())
    }
}
