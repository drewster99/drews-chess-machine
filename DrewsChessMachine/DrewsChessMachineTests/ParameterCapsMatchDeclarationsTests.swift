//
//  ParameterCapsMatchDeclarationsTests.swift
//  DrewsChessMachineTests
//
//  The app's hard caps on arena concurrency and the two replay-ratio delays
//  are the declared ranges' upper bounds — one source of truth. They used to
//  be separate constants below the declarations, so a value the declaration
//  accepted (from `parameters.json`, a stored setting or a resumed session)
//  was clamped in some places, refused in others, and accepted in the rest.
//

import XCTest
@testable import DrewsChessMachine

final class ParameterCapsMatchDeclarationsTests: XCTestCase {

    func testArenaConcurrencyCapIsTheDeclaredRange() {
        XCTAssertEqual(UpperContentView.absoluteMaxArenaConcurrency, ArenaConcurrency.declaredClosedRange.upperBound)
    }

    func testDelayCapsAreTheDeclaredRanges() {
        XCTAssertEqual(UpperContentView.selfPlayDelayMaxMs, SelfPlayDelayMs.declaredClosedRange.upperBound)
        XCTAssertEqual(UpperContentView.stepDelayMaxMs, TrainingStepDelayMs.declaredClosedRange.upperBound)
    }

    func testSelfPlayWorkerCapIsTheDeclaredRange() {
        XCTAssertEqual(UpperContentView.absoluteMaxSelfPlayWorkers, SelfPlayConcurrency.declaredClosedRange.upperBound)
    }
}
