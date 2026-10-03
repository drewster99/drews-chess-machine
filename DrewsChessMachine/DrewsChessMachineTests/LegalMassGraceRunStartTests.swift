//
//  LegalMassGraceRunStartTests.swift
//  DrewsChessMachineTests
//
//  The legal-mass-collapse grace period counts training time only: it
//  starts at the first probe that sees an SGD step of this run. A resumed
//  run's step count starts at the session's count, not 0, so a gate of
//  "steps > 0" passed on the first probe — while the replay buffer was
//  still refilling — and the refill was charged against the grace. The
//  gate compares against the count the run started from.
//

import XCTest
@testable import DrewsChessMachine

final class LegalMassGraceRunStartTests: XCTestCase {

    private let t0 = Date(timeIntervalSince1970: 1_790_000_000)

    func testResumedStepCountDoesNotStartGrace() {
        let detector = LegalMassCollapseDetectorBox()
        detector.restore(LegalMassCollapseDetectorState(legalMassWindow: [0.004], graceElapsedSec: 90))
        XCTAssertNil(detector.graceElapsed(trainingSteps: 5000, stepsAtRunStart: 5000, at: t0))
        XCTAssertEqual(detector.graceElapsed(trainingSteps: 5001, stepsAtRunStart: 5000, at: t0.addingTimeInterval(600)), 90)
        XCTAssertEqual(detector.graceElapsed(trainingSteps: 5100, stepsAtRunStart: 5000, at: t0.addingTimeInterval(630)), 120)
    }

    func testAFreshRunStartsGraceAtItsFirstStep() {
        let detector = LegalMassCollapseDetectorBox()
        XCTAssertNil(detector.graceElapsed(trainingSteps: 0, stepsAtRunStart: 0, at: t0))
        XCTAssertEqual(detector.graceElapsed(trainingSteps: 1, stepsAtRunStart: 0, at: t0.addingTimeInterval(10)), 0)
        XCTAssertEqual(detector.graceElapsed(trainingSteps: 40, stepsAtRunStart: 0, at: t0.addingTimeInterval(55)), 45)
    }
}
