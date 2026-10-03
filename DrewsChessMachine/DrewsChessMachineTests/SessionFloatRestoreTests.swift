//
//  SessionFloatRestoreTests.swift
//  DrewsChessMachineTests
//
//  Pins how a session resume widens the trainer hyperparameters a
//  `.dcmsession` stores as `Float` back into the `Double` training
//  parameters. Widening with `Double(_:)` kept the float's binary value, so a
//  resumed 0.1 became 0.10000000149011612 and was persisted to the user's
//  settings and `parameters.json`; every Float restore site now goes through
//  `TrainingParameters.restoreFromSession(_:savedFloat:into:)`.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class SessionFloatRestoreTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    func testSavedFloatWidensToTheDecimalThatWasSet() {
        let typed: [Double] = [0.1, 0.013, 0.0033, 0.02, 3e-4, 0.9, 0.48, 1e-5]
        for value in typed {
            let saved = Float(value)
            XCTAssertNotEqual(Double(saved), value, "precondition: \(value) is not exact in binary32")
            XCTAssertEqual(TrainingParameters.doubleFromSavedFloat(saved), value)
        }
    }

    func testSavedFloatKeepsExactAndNonFiniteValues() {
        XCTAssertEqual(TrainingParameters.doubleFromSavedFloat(0.5), 0.5)
        XCTAssertEqual(TrainingParameters.doubleFromSavedFloat(0), 0)
        XCTAssertEqual(TrainingParameters.doubleFromSavedFloat(.infinity), .infinity)
        XCTAssertTrue(TrainingParameters.doubleFromSavedFloat(.nan).isNaN)
    }

    func testRestoringASavedFloatParameterStoresTheTypedValue() {
        let p = TrainingParameters.shared
        p.restoreFromSession(PolicyLabelSmoothingEpsilon.self, savedFloat: Float(0.1), into: \.policyLabelSmoothingEpsilon)
        XCTAssertEqual(p.policyLabelSmoothingEpsilon, 0.1)
        p.restoreFromSession(ArenaTargetTau.self, savedFloat: Float(0.2), into: \.arenaTargetTau)
        XCTAssertEqual(p.arenaTargetTau, 0.2)
        p.restoreFromSession(WeightDecay.self, savedFloat: Float(3e-4), into: \.weightDecay)
        XCTAssertEqual(p.weightDecay, 3e-4)
    }
}
