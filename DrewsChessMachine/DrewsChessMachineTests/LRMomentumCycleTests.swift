//
//  LRMomentumCycleTests.swift
//  DrewsChessMachineTests
//
//  Correctness tests for the cyclical LR / inverse-coupled momentum
//  schedule (see TRAINING_DYNAMICS_PLAN.md §3 and LRMomentumCycle.swift).
//  The math is a pure function of the global step, so these invariants
//  fully pin the runtime behavior: cosine endpoints, geometric-vs-linear
//  interpolation, the inverse coupling via `invert`, and the cycleCount
//  completion freeze. A regression here would silently mis-schedule the
//  optimizer with no build failure.
//

import XCTest
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

final class LRMomentumCycleTests: XCTestCase {

    private let tol = 1e-9

    /// An LR cycle with constant bounds (one keyframe): trough `min`, peak `max`.
    private func makeLR(min: Double, max: Double, period: Int, count: Int = 0, invert: Bool = false) -> LRMomentumCycle {
        LRMomentumCycle.constant(lrPeriodSteps: period, lrCount: count, lrMin: min, lrMax: max, lrInvert: invert)
    }

    /// The cycled LR at cycle step `step` (no warmup, so the cycle step is
    /// the trainer step).
    private func learningRate(_ c: LRMomentumCycle, atStep step: Int) -> Double? {
        c.values(completedTrainSteps: step, lrWarmupSteps: 0).learningRate
    }

    /// The cycled momentum at cycle step `step` (no warmup).
    private func momentum(_ c: LRMomentumCycle, atStep step: Int) -> Double? {
        c.values(completedTrainSteps: step, lrWarmupSteps: 0).momentum
    }

    private func makeMomentum(min: Double, max: Double, period: Int, count: Int = 0, invert: Bool = false) -> LRMomentumCycle {
        var c = LRMomentumCycle.disabled
        c.momentumEnabled = true
        c.momentumMin = min
        c.momentumMax = max
        c.momentumPeriodSteps = period
        c.momentumCount = count
        c.momentumInvert = invert
        return c
    }

    // MARK: - Disabled / fallback

    func testDisabledReturnsNil() {
        let c = LRMomentumCycle.disabled
        XCTAssertNil(learningRate(c, atStep: 0))
        XCTAssertNil(momentum(c, atStep: 123))
        XCTAssertFalse(c.isActive)
    }

    func testMomentumWithReversedEndpointsFallsBackInsteadOfRunningBackward() throws {
        // A reversed pair (max < min) is a misconfiguration: the explicit
        // `momentumInvert` flag is the sanctioned way to run the schedule
        // backward. The evaluator must fall back to nil (→ the static
        // coefficient) rather than silently run reversed, mirroring the LR
        // keyframes' peak >= trough rule.
        let reversed = makeMomentum(min: 0.95, max: 0.85, period: 1000)
        XCTAssertNil(momentum(reversed, atStep: 0))
        XCTAssertNil(momentum(reversed, atStep: 500))

        // Equal endpoints are valid (a degenerate constant schedule), not a
        // misconfiguration — must stay active and return the shared value.
        let flat = makeMomentum(min: 0.9, max: 0.9, period: 1000)
        XCTAssertEqual(try XCTUnwrap(momentum(flat, atStep: 250)), 0.9, accuracy: tol)

        // A normal ascending pair stays active.
        let ok = makeMomentum(min: 0.85, max: 0.95, period: 1000)
        XCTAssertNotNil(momentum(ok, atStep: 250))
    }

    // MARK: - Cosine fraction shape

    func testFractionIsZeroAtBoundaryOneAtMidpoint() {
        let period = 1000
        // phase 0 → frac 0
        XCTAssertEqual(LRMomentumCycle.cycleFraction(step: 0, period: period, count: 0, invert: false), 0.0, accuracy: tol)
        // phase 0.5 (midpoint) → frac 1
        XCTAssertEqual(LRMomentumCycle.cycleFraction(step: period / 2, period: period, count: 0, invert: false), 1.0, accuracy: tol)
        // wraps: one full period later is the boundary again
        XCTAssertEqual(LRMomentumCycle.cycleFraction(step: period, period: period, count: 0, invert: false), 0.0, accuracy: tol)
    }

    func testFractionIsBoundedAndSymmetric() {
        let period = 360
        for step in 0..<(period * 3) {
            let f = LRMomentumCycle.cycleFraction(step: step, period: period, count: 0, invert: false)
            XCTAssertGreaterThanOrEqual(f, -tol)
            XCTAssertLessThanOrEqual(f, 1.0 + tol)
        }
        // Symmetric about the midpoint: frac(¼ period) == frac(¾ period).
        let q = LRMomentumCycle.cycleFraction(step: period / 4, period: period, count: 0, invert: false)
        let tq = LRMomentumCycle.cycleFraction(step: 3 * period / 4, period: period, count: 0, invert: false)
        XCTAssertEqual(q, tq, accuracy: 1e-9)
    }

    func testInvertIsOneMinusFraction() {
        let period = 800
        for step in stride(from: 0, to: period, by: 37) {
            let normal = LRMomentumCycle.cycleFraction(step: step, period: period, count: 0, invert: false)
            let inverted = LRMomentumCycle.cycleFraction(step: step, period: period, count: 0, invert: true)
            XCTAssertEqual(inverted, 1.0 - normal, accuracy: tol)
        }
    }

    // MARK: - LR geometric interpolation

    func testLRHitsEndpoints() throws {
        let c = makeLR(min: 1e-3, max: 1e-1, period: 1000)
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 0)), 1e-3, accuracy: 1e-12)        // boundary → min
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 500)), 1e-1, accuracy: 1e-12)      // midpoint → max
    }

    func testLRIsGeometricAtMidFraction() throws {
        // At frac = 0.5 the geometric mean of the endpoints is expected,
        // NOT the arithmetic mean — this is what distinguishes geometric
        // (log-space) interpolation from linear.
        let minV = 1e-4, maxV = 1e-2
        let c = makeLR(min: minV, max: maxV, period: 1000)
        // Find a step whose frac is 0.5: frac = 0.5(1-cos(2π·phase)) = 0.5
        //   → cos(2π·phase) = 0 → phase = 0.25 → step = period/4.
        let lr = try XCTUnwrap(learningRate(c, atStep: 250))
        let geoMean = (minV * maxV).squareRoot()      // 1e-3
        let arithMean = (minV + maxV) / 2.0           // 5.05e-3
        XCTAssertEqual(lr, geoMean, accuracy: 1e-9)
        XCTAssertNotEqual(lr, arithMean, accuracy: 1e-4)
    }

    func testLRInvertStartsAtMax() throws {
        let c = makeLR(min: 1e-3, max: 1e-1, period: 1000, invert: true)
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 0)), 1e-1, accuracy: 1e-12)        // boundary → max when inverted
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 500)), 1e-3, accuracy: 1e-12)      // midpoint → min
    }

    func testLRRequiresPositiveMin() throws {
        // LR keyframes moved the "trough must be > 0" check from the evaluator
        // to the keyframe validator, so a zero trough can no longer reach the
        // evaluator; the previous settings with a zero minimum convert to a
        // disabled LR channel, which still yields nil (fall back to base LR).
        // Geometric interpolation is undefined at zero.
        var old = LegacyLRMomentumCycleReference.disabledReference
        old.lrEnabled = true
        old.lrMin = 0.0
        old.lrMax = 1e-1
        old.lrPeriodSteps = 1000
        let c = try LRMomentumCycle.converted(from: old)
        XCTAssertNil(learningRate(c, atStep: 250))

        let zeroTrough = LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 1e-1, trough: 0.0, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
        ])
        XCTAssertThrowsError(try zeroTrough.validated()) { error in
            XCTAssertEqual(error as? LRKeyframeScheduleError, .troughNotPositive(index: 0, trough: 0.0))
        }
    }

    func testLREqualEndpointsIsConstant() throws {
        let c = makeLR(min: 5e-3, max: 5e-3, period: 1000)
        for step in stride(from: 0, to: 1000, by: 101) {
            XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: step)), 5e-3, accuracy: 1e-12)
        }
    }

    // MARK: - Momentum linear interpolation

    func testMomentumHitsEndpoints() throws {
        let c = makeMomentum(min: 0.85, max: 0.95, period: 1000)
        XCTAssertEqual(try XCTUnwrap(momentum(c, atStep: 0)), 0.85, accuracy: tol)              // boundary → min
        XCTAssertEqual(try XCTUnwrap(momentum(c, atStep: 500)), 0.95, accuracy: tol)            // midpoint → max
    }

    func testMomentumIsLinearAtMidFraction() throws {
        // At frac 0.5 momentum is the ARITHMETIC mean (linear interpolation).
        let c = makeMomentum(min: 0.85, max: 0.95, period: 1000)
        XCTAssertEqual(try XCTUnwrap(momentum(c, atStep: 250)), 0.90, accuracy: 1e-9)
    }

    // MARK: - Inverse coupling

    func testEqualPeriodInverseCoupling() throws {
        // LR not inverted, momentum inverted, same period → momentum troughs
        // exactly when LR peaks (Smith's inverse coupling).
        let period = 1000
        let c = LRMomentumCycle.constant(
            lrPeriodSteps: period, lrMin: 1e-3, lrMax: 1e-1, lrInvert: false,
            momentumEnabled: true, momentumPeriodSteps: period, momentumMin: 0.85, momentumMax: 0.95, momentumInvert: true)

        // Midpoint: LR at max, momentum at min.
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: period / 2)), 1e-1, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(momentum(c, atStep: period / 2)), 0.85, accuracy: tol)
        // Boundary: LR at min, momentum at max.
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 0)), 1e-3, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(momentum(c, atStep: 0)), 0.95, accuracy: tol)
    }

    // MARK: - cycleCount completion freeze

    func testLRCountFreezesAtBoundary() throws {
        // After `count` cycles, LR (not inverted) freezes at min.
        let c = makeLR(min: 1e-3, max: 1e-1, period: 1000, count: 2)
        // Within the 2 cycles the midpoint still peaks.
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 500)), 1e-1, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 1500)), 1e-1, accuracy: 1e-12)
        // step / period >= 2 → frozen at boundary (min), even at what would
        // otherwise be the peak.
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 2000)), 1e-3, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 2500)), 1e-3, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 9999)), 1e-3, accuracy: 1e-12)
    }

    func testMomentumCountFreezeRespectsInvert() throws {
        // Inverted momentum frozen after completion lands on MAX (not min),
        // i.e. the high-momentum converged regime — the design's whole point.
        let c = makeMomentum(min: 0.85, max: 0.95, period: 1000, count: 1, invert: true)
        XCTAssertEqual(try XCTUnwrap(momentum(c, atStep: 5000)), 0.95, accuracy: tol)
    }

    func testCountZeroIsUnbounded() throws {
        let c = makeLR(min: 1e-3, max: 1e-1, period: 1000, count: 0)
        // Far in the future the cycle still oscillates.
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 1_000_000 + 500)), 1e-1, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: 1_000_000)), 1e-3, accuracy: 1e-12)
    }

    // MARK: - Determinism / resume

    func testPhaseIsPureFunctionOfStep() throws {
        // The same step always yields the same value (no hidden state) — this
        // is what makes stop/resume seamless.
        let c = makeLR(min: 1e-3, max: 1e-1, period: 777)
        for step in stride(from: 0, to: 5000, by: 13) {
            XCTAssertEqual(try XCTUnwrap(learningRate(c, atStep: step)), try XCTUnwrap(learningRate(c, atStep: step)), accuracy: 1e-15)
        }
    }
}
