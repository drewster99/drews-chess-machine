//
//  TrainingParametersRunHoldTests.swift
//  DrewsChessMachineTests
//
//  A resume holds some values for its run only: a key the session predates
//  gets its declared pre-feature value, and a saved value outside today's
//  declared range is restored as it is. Neither is written to the user's
//  saved settings — but the singleton holds them in memory, so a later
//  fresh run in the same launch used to train under them (dropout 0, label
//  smoothing off, …) while the user's setting said otherwise. These tests
//  pin that a hold records the value it replaced (the first hold of a key
//  wins), that an ordinary assignment during the run — a user edit, a Load
//  Parameters apply — replaces the hold's claim, and that `releaseRunHolds`
//  puts back what each remaining hold replaced.
//
//  Persistence is suppressed throughout, so the user's saved settings are
//  never written.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class TrainingParametersRunHoldTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        TrainingParameters.suppressPersistence = true
        try TrainingParameters.shared.releaseRunHolds()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
    }

    override func tearDown() async throws {
        TrainingParameters.suppressPersistence = true
        try TrainingParameters.shared.releaseRunHolds()
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    func testReleaseRestoresTheValueAHoldReplaced() throws {
        let p = TrainingParameters.shared
        p.dropoutRate = 0.7
        p.holdForThisRun(DropoutRate.self, 0, into: \.dropoutRate)
        XCTAssertEqual(p.dropoutRate, 0)
        try p.releaseRunHolds()
        XCTAssertEqual(p.dropoutRate, 0.7)
    }

    func testAnEditDuringTheRunIsKeptOnRelease() throws {
        let p = TrainingParameters.shared
        p.dropoutRate = 0.7
        p.holdForThisRun(DropoutRate.self, 0, into: \.dropoutRate)
        p.dropoutRate = 0.5
        try p.releaseRunHolds()
        XCTAssertEqual(p.dropoutRate, 0.5)
    }

    /// A run override applied without persisting (`--parameters`, Load
    /// Parameters) before the hold is what comes back — not the stored
    /// setting.
    func testARunOverrideBeforeTheHoldComesBack() throws {
        let p = TrainingParameters.shared
        try p.apply([DropoutRate.id: .double(0.3)])
        p.holdForThisRun(DropoutRate.self, 0, into: \.dropoutRate)
        try p.releaseRunHolds()
        XCTAssertEqual(p.dropoutRate, 0.3)
    }

    func testTheFirstHoldWins() throws {
        let p = TrainingParameters.shared
        p.dropoutRate = 0.7
        p.holdForThisRun(DropoutRate.self, 0, into: \.dropoutRate)
        p.holdForThisRun(DropoutRate.self, 0.1, into: \.dropoutRate)
        try p.releaseRunHolds()
        XCTAssertEqual(p.dropoutRate, 0.7)
    }

    func testAnOutOfRangeSessionValueIsHeldAndReleased() throws {
        let p = TrainingParameters.shared
        p.dropoutRate = 0.2
        p.restoreFromSession(DropoutRate.self, 0.99, into: \.dropoutRate)
        XCTAssertEqual(p.dropoutRate, 0.99)
        try p.releaseRunHolds()
        XCTAssertEqual(p.dropoutRate, 0.2)
    }

    func testAKeyedClosureHoldIsReleased() throws {
        let p = TrainingParameters.shared
        p.arenaPromotionCriterion = .sprt
        p.holdForThisRun(ArenaPromotionCriterionParameter.self) { p.arenaPromotionCriterion = .scoreThreshold }
        XCTAssertEqual(p.arenaPromotionCriterion, .scoreThreshold)
        try p.releaseRunHolds()
        XCTAssertEqual(p.arenaPromotionCriterion, .sprt)
    }
}
