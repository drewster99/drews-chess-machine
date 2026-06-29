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
@testable import DrewsChessMachine

final class LRMomentumCycleTests: XCTestCase {

    private let tol = 1e-9

    private func makeLR(min: Double, max: Double, period: Int, count: Int = 0, invert: Bool = false) -> LRMomentumCycle {
        var c = LRMomentumCycle.disabled
        c.lrEnabled = true
        c.lrMin = min
        c.lrMax = max
        c.lrPeriodSteps = period
        c.lrCount = count
        c.lrInvert = invert
        return c
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
        XCTAssertNil(c.learningRate(forStep: 0))
        XCTAssertNil(c.momentum(forStep: 123))
        XCTAssertFalse(c.isActive)
    }

    func testMomentumWithReversedEndpointsFallsBackInsteadOfRunningBackward() throws {
        // A reversed pair (max < min) is a misconfiguration: the explicit
        // `momentumInvert` flag is the sanctioned way to run the schedule
        // backward. momentum(forStep:) must fall back to nil (→ the static
        // coefficient) rather than silently run reversed, mirroring the LR
        // channel's lrMax >= lrMin guard.
        let reversed = makeMomentum(min: 0.95, max: 0.85, period: 1000)
        XCTAssertNil(reversed.momentum(forStep: 0))
        XCTAssertNil(reversed.momentum(forStep: 500))

        // Equal endpoints are valid (a degenerate constant schedule), not a
        // misconfiguration — must stay active and return the shared value.
        let flat = makeMomentum(min: 0.9, max: 0.9, period: 1000)
        XCTAssertEqual(try XCTUnwrap(flat.momentum(forStep: 250)), 0.9, accuracy: tol)

        // A normal ascending pair stays active.
        let ok = makeMomentum(min: 0.85, max: 0.95, period: 1000)
        XCTAssertNotNil(ok.momentum(forStep: 250))
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

    func testLRHitsEndpoints() {
        let c = makeLR(min: 1e-3, max: 1e-1, period: 1000)
        XCTAssertEqual(c.learningRate(forStep: 0)!, 1e-3, accuracy: 1e-12)        // boundary → min
        XCTAssertEqual(c.learningRate(forStep: 500)!, 1e-1, accuracy: 1e-12)      // midpoint → max
    }

    func testLRIsGeometricAtMidFraction() {
        // At frac = 0.5 the geometric mean of the endpoints is expected,
        // NOT the arithmetic mean — this is what distinguishes geometric
        // (log-space) interpolation from linear.
        let minV = 1e-4, maxV = 1e-2
        let c = makeLR(min: minV, max: maxV, period: 1000)
        // Find a step whose frac is 0.5: frac = 0.5(1-cos(2π·phase)) = 0.5
        //   → cos(2π·phase) = 0 → phase = 0.25 → step = period/4.
        let lr = c.learningRate(forStep: 250)!
        let geoMean = (minV * maxV).squareRoot()      // 1e-3
        let arithMean = (minV + maxV) / 2.0           // 5.05e-3
        XCTAssertEqual(lr, geoMean, accuracy: 1e-9)
        XCTAssertNotEqual(lr, arithMean, accuracy: 1e-4)
    }

    func testLRInvertStartsAtMax() {
        let c = makeLR(min: 1e-3, max: 1e-1, period: 1000, invert: true)
        XCTAssertEqual(c.learningRate(forStep: 0)!, 1e-1, accuracy: 1e-12)        // boundary → max when inverted
        XCTAssertEqual(c.learningRate(forStep: 500)!, 1e-3, accuracy: 1e-12)      // midpoint → min
    }

    func testLRRequiresPositiveMin() {
        // Geometric interpolation is undefined at zero → nil (fall back to base LR).
        let c = makeLR(min: 0.0, max: 1e-1, period: 1000)
        XCTAssertNil(c.learningRate(forStep: 250))
    }

    func testLREqualEndpointsIsConstant() {
        let c = makeLR(min: 5e-3, max: 5e-3, period: 1000)
        for step in stride(from: 0, to: 1000, by: 101) {
            XCTAssertEqual(c.learningRate(forStep: step)!, 5e-3, accuracy: 1e-12)
        }
    }

    // MARK: - Momentum linear interpolation

    func testMomentumHitsEndpoints() {
        let c = makeMomentum(min: 0.85, max: 0.95, period: 1000)
        XCTAssertEqual(c.momentum(forStep: 0)!, 0.85, accuracy: tol)              // boundary → min
        XCTAssertEqual(c.momentum(forStep: 500)!, 0.95, accuracy: tol)            // midpoint → max
    }

    func testMomentumIsLinearAtMidFraction() {
        // At frac 0.5 momentum is the ARITHMETIC mean (linear interpolation).
        let c = makeMomentum(min: 0.85, max: 0.95, period: 1000)
        XCTAssertEqual(c.momentum(forStep: 250)!, 0.90, accuracy: 1e-9)
    }

    // MARK: - Inverse coupling

    func testEqualPeriodInverseCoupling() {
        // LR not inverted, momentum inverted, same period → momentum troughs
        // exactly when LR peaks (Smith's inverse coupling).
        let period = 1000
        var c = LRMomentumCycle.disabled
        c.lrEnabled = true; c.lrMin = 1e-3; c.lrMax = 1e-1; c.lrPeriodSteps = period; c.lrInvert = false
        c.momentumEnabled = true; c.momentumMin = 0.85; c.momentumMax = 0.95; c.momentumPeriodSteps = period; c.momentumInvert = true

        // Midpoint: LR at max, momentum at min.
        XCTAssertEqual(c.learningRate(forStep: period / 2)!, 1e-1, accuracy: 1e-12)
        XCTAssertEqual(c.momentum(forStep: period / 2)!, 0.85, accuracy: tol)
        // Boundary: LR at min, momentum at max.
        XCTAssertEqual(c.learningRate(forStep: 0)!, 1e-3, accuracy: 1e-12)
        XCTAssertEqual(c.momentum(forStep: 0)!, 0.95, accuracy: tol)
    }

    // MARK: - cycleCount completion freeze

    func testLRCountFreezesAtBoundary() {
        // After `count` cycles, LR (not inverted) freezes at min.
        let c = makeLR(min: 1e-3, max: 1e-1, period: 1000, count: 2)
        // Within the 2 cycles the midpoint still peaks.
        XCTAssertEqual(c.learningRate(forStep: 500)!, 1e-1, accuracy: 1e-12)
        XCTAssertEqual(c.learningRate(forStep: 1500)!, 1e-1, accuracy: 1e-12)
        // step / period >= 2 → frozen at boundary (min), even at what would
        // otherwise be the peak.
        XCTAssertEqual(c.learningRate(forStep: 2000)!, 1e-3, accuracy: 1e-12)
        XCTAssertEqual(c.learningRate(forStep: 2500)!, 1e-3, accuracy: 1e-12)
        XCTAssertEqual(c.learningRate(forStep: 9999)!, 1e-3, accuracy: 1e-12)
    }

    func testMomentumCountFreezeRespectsInvert() {
        // Inverted momentum frozen after completion lands on MAX (not min),
        // i.e. the high-momentum converged regime — the design's whole point.
        let c = makeMomentum(min: 0.85, max: 0.95, period: 1000, count: 1, invert: true)
        XCTAssertEqual(c.momentum(forStep: 5000)!, 0.95, accuracy: tol)
    }

    func testCountZeroIsUnbounded() {
        let c = makeLR(min: 1e-3, max: 1e-1, period: 1000, count: 0)
        // Far in the future the cycle still oscillates.
        XCTAssertEqual(c.learningRate(forStep: 1_000_000 + 500)!, 1e-1, accuracy: 1e-12)
        XCTAssertEqual(c.learningRate(forStep: 1_000_000)!, 1e-3, accuracy: 1e-12)
    }

    // MARK: - Determinism / resume

    func testPhaseIsPureFunctionOfStep() {
        // The same step always yields the same value (no hidden state) — this
        // is what makes stop/resume seamless.
        let c = makeLR(min: 1e-3, max: 1e-1, period: 777)
        for step in stride(from: 0, to: 5000, by: 13) {
            XCTAssertEqual(c.learningRate(forStep: step)!, c.learningRate(forStep: step)!, accuracy: 1e-15)
        }
    }
}
