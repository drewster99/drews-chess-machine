//
//  LRCycleDecayEnvelopeTests.swift
//  DrewsChessMachineTests
//
//  Correctness tests for the LR cycle's decaying envelope and the
//  momentum-follows-LR mode (see `LRMomentumCycle.values(forCycleStep:)`).
//  The schedule is a pure function of the cycle step, so these pin its
//  behavior exactly: the peak/trough decay endpoints and midpoint, the hold
//  after the horizon, that a zero horizon reproduces the pre-envelope
//  schedule, geometric in-cycle interpolation, momentum's inverse phase and
//  endpoint drift, the follow switch, and the warmup offset.
//

import XCTest
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
    private func makeDecayingSchedule(momentumFollows: Bool = true) -> LRMomentumCycle {
        var cycle = LRMomentumCycle.disabled
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
        cycle.envelope = LRMomentumCycleEnvelope(
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

    func testPeakAndTroughAtStartOfHorizon() {
        let values = makeDecayingSchedule().values(forCycleStep: 0)
        assertRelativelyEqual(values.lrPeak, peakStart, "peak at t=0")
        assertRelativelyEqual(values.lrTrough, troughStart, "trough at t=0")
        // Inverted: the cycle opens at its peak.
        assertRelativelyEqual(values.learningRate, peakStart, "LR at t=0 is the peak")
    }

    func testPeakAndTroughAtHalfHorizonAreGeometricMidpoints() {
        let values = makeDecayingSchedule().values(forCycleStep: horizon / 2)
        assertRelativelyEqual(values.lrPeak, (peakStart * peakEnd).squareRoot(), "peak at t=horizon/2")
        assertRelativelyEqual(values.lrTrough, (troughStart * troughEnd).squareRoot(), "trough at t=horizon/2")
        // horizon/2 is a whole number of periods, so the cycle is at its peak.
        XCTAssertEqual((horizon / 2) % period, 0)
        assertRelativelyEqual(values.learningRate, (peakStart * peakEnd).squareRoot(), "LR at t=horizon/2")
    }

    func testPeakAndTroughAtEndOfHorizon() {
        let values = makeDecayingSchedule().values(forCycleStep: horizon)
        assertRelativelyEqual(values.lrPeak, peakEnd, "peak at t=horizon")
        assertRelativelyEqual(values.lrTrough, troughEnd, "trough at t=horizon")
        XCTAssertEqual(horizon % period, 0)
        assertRelativelyEqual(values.learningRate, peakEnd, "LR at t=horizon")
    }

    func testEnvelopeHoldsAfterHorizonAndKeepsCycling() {
        let schedule = makeDecayingSchedule()
        for extraPeriods in [1, 7, 50] {
            let peakStep = horizon + extraPeriods * period
            let atPeak = schedule.values(forCycleStep: peakStep)
            assertRelativelyEqual(atPeak.lrPeak, peakEnd, "peak held at \(peakStep)")
            assertRelativelyEqual(atPeak.lrTrough, troughEnd, "trough held at \(peakStep)")
            assertRelativelyEqual(atPeak.learningRate, peakEnd, "LR at a post-horizon cycle start")
            // Half a period later the inverted cycle is at its trough.
            let troughStep = peakStep + period / 2
            assertRelativelyEqual(
                schedule.values(forCycleStep: troughStep).learningRate,
                troughEnd,
                "LR at a post-horizon cycle midpoint"
            )
        }
    }

    func testDecayFractionClampsAndZeroHorizonIsNoDecay() {
        XCTAssertEqual(LRMomentumCycle.decayFraction(step: -5, horizonSteps: 100), 0.0)
        XCTAssertEqual(LRMomentumCycle.decayFraction(step: 50, horizonSteps: 100), 0.5)
        XCTAssertEqual(LRMomentumCycle.decayFraction(step: 100, horizonSteps: 100), 1.0)
        XCTAssertEqual(LRMomentumCycle.decayFraction(step: 10_000, horizonSteps: 100), 1.0)
        XCTAssertEqual(LRMomentumCycle.decayFraction(step: 10_000, horizonSteps: 0), 0.0)
    }

    // MARK: - Horizon 0 is the pre-envelope behavior

    func testZeroHorizonReproducesPreEnvelopeSchedule() {
        var schedule = makeDecayingSchedule(momentumFollows: false)
        schedule.envelope.decayHorizonSteps = 0
        let steps = [0, 1, 2_500, 5_000, 9_999, 10_000, 17_321, 20_000, 123_457, 5_000_003]
        for step in steps {
            // The pre-envelope LR formula, written out independently.
            let lrFraction = LRMomentumCycle.cycleFraction(step: step, period: period, count: 0, invert: true)
            let expectedLR = troughStart * pow(peakStart / troughStart, lrFraction)
            let momentumFraction = LRMomentumCycle.cycleFraction(step: step, period: 3_000, count: 0, invert: false)
            let expectedMomentum = 0.80 + (0.98 - 0.80) * momentumFraction
            let values = schedule.values(forCycleStep: step)
            XCTAssertEqual(values.learningRate, expectedLR, "LR at step \(step) must match the pre-envelope formula exactly")
            XCTAssertEqual(values.momentum, expectedMomentum, "momentum at step \(step) must match the pre-envelope formula exactly")
            XCTAssertEqual(values.lrPeak, peakStart)
            XCTAssertEqual(values.lrTrough, troughStart)
        }
    }

    func testDefaultEnvelopeIsNoDecayAndNoFollow() {
        let noDecay = LRMomentumCycleEnvelope.noDecay
        XCTAssertEqual(noDecay.decayHorizonSteps, 0)
        XCTAssertFalse(noDecay.momentumFollowsLRCycle)
        XCTAssertEqual(LRMomentumCycle.disabled.envelope, noDecay)
    }

    // MARK: - Geometric interpolation within a cycle

    func testInCycleInterpolationIsGeometricBetweenCurrentPeakAndTrough() {
        let schedule = makeDecayingSchedule()
        // A quarter period into a cycle the cosine fraction is exactly one
        // half, so the LR is the geometric mean of the current bounds.
        for cycleStart in [0, 40_000, 480_000, horizon, horizon + 200_000] {
            let step = cycleStart + period / 4
            let values = schedule.values(forCycleStep: step)
            guard let peak = values.lrPeak, let trough = values.lrTrough else {
                XCTFail("envelope bounds missing at \(step)")
                continue
            }
            assertRelativelyEqual(values.learningRate, (peak * trough).squareRoot(), "quarter-period LR at \(step)")
        }
        // At an arbitrary phase the LR is trough·(peak/trough)^fraction.
        let step = 123_457
        let values = schedule.values(forCycleStep: step)
        let decay = Double(step) / Double(horizon)
        let expectedPeak = peakStart * pow(peakEnd / peakStart, decay)
        let expectedTrough = troughStart * pow(troughEnd / troughStart, decay)
        let fraction = LRMomentumCycle.cycleFraction(step: step, period: period, count: 0, invert: true)
        assertRelativelyEqual(values.lrPeak, expectedPeak, "peak at \(step)")
        assertRelativelyEqual(values.lrTrough, expectedTrough, "trough at \(step)")
        assertRelativelyEqual(values.learningRate, expectedTrough * pow(expectedPeak / expectedTrough, fraction), "LR at \(step)")
    }

    func testUnusableEndpointsDisableTheLRChannel() {
        var schedule = makeDecayingSchedule()
        schedule.envelope.lrPeakEnd = 1.0e-7
        schedule.envelope.lrTroughEnd = 1.0e-5
        let values = schedule.values(forCycleStep: 0)
        XCTAssertNil(values.learningRate, "an end peak below the end trough must fall back to the static LR")
        XCTAssertNil(values.lrPeak)
        XCTAssertNil(values.lrTrough)
        XCTAssertNil(values.momentum, "following momentum has no phase without a usable LR cycle")
    }

    // MARK: - Momentum follows the LR cycle

    func testFollowingMomentumIsLowAtLRPeakAndHighAtLRTrough() {
        let schedule = makeDecayingSchedule()
        // Start of horizon.
        assertRelativelyEqual(schedule.values(forCycleStep: 0).momentum, 0.85, "momentum at the first LR peak")
        assertRelativelyEqual(
            schedule.values(forCycleStep: period / 2).momentum,
            0.95,
            "momentum at the first LR trough"
        )
        // Quarter period: the LR fraction is one half, so momentum is mid-band.
        let quarter = schedule.values(forCycleStep: period / 4)
        let decay = Double(period / 4) / Double(horizon)
        let low = 0.85 + (0.90 - 0.85) * decay
        let high = 0.95
        assertRelativelyEqual(quarter.momentum, (low + high) / 2, "momentum at a quarter period")
    }

    func testFollowingMomentumBoundsDriftLinearlyOverHorizonThenHold() {
        let schedule = makeDecayingSchedule()
        assertRelativelyEqual(schedule.values(forCycleStep: horizon / 2).momentum, 0.875, "low bound at t=horizon/2")
        assertRelativelyEqual(schedule.values(forCycleStep: horizon).momentum, 0.90, "low bound at t=horizon")
        assertRelativelyEqual(
            schedule.values(forCycleStep: horizon + period / 2).momentum,
            0.95,
            "high bound after the horizon"
        )
        assertRelativelyEqual(
            schedule.values(forCycleStep: horizon + 30 * period).momentum,
            0.90,
            "low bound held after the horizon"
        )
    }

    func testFollowingMomentumIgnoresTheIndependentMomentumCycle() {
        var a = makeDecayingSchedule()
        var b = makeDecayingSchedule()
        a.momentumPeriodSteps = 7
        a.momentumMin = 0.1
        a.momentumInvert = true
        b.momentumPeriodSteps = 99_999
        b.momentumMax = 0.99
        for step in [0, 3, 5_000, 250_001, 2_000_000] {
            XCTAssertEqual(a.values(forCycleStep: step).momentum, b.values(forCycleStep: step).momentum)
        }
    }

    func testFollowingMomentumNeedsBothMomentumCyclingAndAnLRCycle() {
        var noMomentumCycling = makeDecayingSchedule()
        noMomentumCycling.momentumEnabled = false
        XCTAssertNil(noMomentumCycling.values(forCycleStep: 0).momentum)

        var noLRCycle = makeDecayingSchedule()
        noLRCycle.lrEnabled = false
        XCTAssertNil(noLRCycle.values(forCycleStep: 0).momentum)
        XCTAssertNil(noLRCycle.values(forCycleStep: 0).learningRate)
    }

    func testInvertedFollowBoundsReportNoMomentum() {
        var schedule = makeDecayingSchedule()
        schedule.envelope.momentumFollowStartLow = 0.97
        XCTAssertNil(schedule.values(forCycleStep: 0).momentum)
    }

    // MARK: - Follow switch off

    func testFollowOffLeavesTheIndependentMomentumCycleUnchanged() {
        let schedule = makeDecayingSchedule(momentumFollows: false)
        for step in [0, 1, 750, 1_500, 2_999, 3_000, 500_000, horizon, horizon + 4_321] {
            let fraction = LRMomentumCycle.cycleFraction(step: step, period: 3_000, count: 0, invert: false)
            XCTAssertEqual(
                schedule.values(forCycleStep: step).momentum,
                0.80 + (0.98 - 0.80) * fraction,
                "independent momentum at step \(step)"
            )
        }
    }

    // MARK: - Warmup offset

    func testWarmupOffsetStillApplies() {
        let schedule = makeDecayingSchedule()
        let warmup = 5_000
        // Inside warmup every step maps to cycle step 0.
        for global in [0, 1, warmup / 2, warmup] {
            XCTAssertEqual(
                schedule.values(completedTrainSteps: global, lrWarmupSteps: warmup),
                schedule.values(forCycleStep: 0),
                "global step \(global) inside warmup"
            )
        }
        // After warmup the schedule is shifted back by the warmup length,
        // including the decay, which is measured in cycle steps.
        for cycleStep in [1, period / 4, horizon / 2, horizon, horizon + period / 2] {
            XCTAssertEqual(
                schedule.values(completedTrainSteps: warmup + cycleStep, lrWarmupSteps: warmup),
                schedule.values(forCycleStep: cycleStep),
                "cycle step \(cycleStep) after warmup"
            )
        }
        assertRelativelyEqual(
            schedule.learningRate(completedTrainSteps: warmup + horizon, lrWarmupSteps: warmup),
            peakEnd,
            "the horizon ends a full horizon after warmup, not after step zero"
        )
        assertRelativelyEqual(
            schedule.momentum(completedTrainSteps: warmup + horizon, lrWarmupSteps: warmup),
            0.90,
            "momentum reaches its end low bound a full horizon after warmup"
        )
    }

    // MARK: - Persistence shape

    func testCycleEncodingOmitsTheEnvelopeSoOlderSessionsStillDecode() throws {
        let schedule = makeDecayingSchedule()
        let encoded = try JSONEncoder().encode(schedule)
        let object = try JSONSerialization.jsonObject(with: encoded)
        guard let dictionary = object as? [String: Any] else {
            XCTFail("encoded cycle is not a JSON object")
            return
        }
        XCTAssertNil(dictionary["envelope"], "the envelope is persisted separately on the session")
        let decoded = try JSONDecoder().decode(LRMomentumCycle.self, from: encoded)
        XCTAssertEqual(decoded.envelope, .noDecay)
        var expected = schedule
        expected.envelope = .noDecay
        XCTAssertEqual(decoded, expected)
    }

    func testMissingSavedEnvelopeResolvesToNoDecayAndNoFollow() {
        let current = makeDecayingSchedule().envelope
        let resolved = SessionCheckpointState.resolvedLRMomentumCycleEnvelope(saved: nil, current: current)
        XCTAssertEqual(resolved.decayHorizonSteps, 0)
        XCTAssertFalse(resolved.momentumFollowsLRCycle)
        XCTAssertEqual(resolved.lrPeakEnd, current.lrPeakEnd)
        XCTAssertEqual(resolved.lrTroughEnd, current.lrTroughEnd)
        XCTAssertEqual(
            SessionCheckpointState.resolvedLRMomentumCycleEnvelope(saved: current, current: .noDecay),
            current
        )
    }
}
