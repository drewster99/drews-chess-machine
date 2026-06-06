//
//  TrainingSettingsCycleValidationTests.swift
//  DrewsChessMachineTests
//
//  Guards the cross-field min<=max validation for the LR / momentum cycle
//  editors in `TrainingSettingsPopoverModel.save()`. Each endpoint is range-
//  checked individually, but an inverted pair (max < min) is a distinct
//  misconfiguration: the LR cycle silently falls back to the static LR
//  (`learningRate(forStep:)` guards `lrMax >= lrMin`) and the momentum cycle
//  silently runs reversed (`momentum(forStep:)` has no such guard), in both
//  cases with no signal to the user. The cross-field check flags the max
//  field so the misconfig surfaces as the red invalid-input overlay.
//
//  Pure model/validation logic (no Metal). `save()` writes through to
//  `TrainingParameters.shared`, so each test snapshots and restores the four
//  cycle endpoints it can touch.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class TrainingSettingsCycleValidationTests: XCTestCase {

    private func makeModel() -> TrainingSettingsPopoverModel {
        // trainerProvider defaults to { nil }, so save() commits to
        // TrainingParameters.shared without needing a live trainer.
        TrainingSettingsPopoverModel(
            selfPlayDelayMaxMs: 1000,
            stepDelayMaxMs: 1000,
            maxSelfPlayWorkers: 8
        )
    }

    func test_lrCycleMaxBelowMin_flagsMaxError() {
        let p = TrainingParameters.shared
        let savedMin = p.lrCycleMin
        let savedMax = p.lrCycleMax
        defer { p.lrCycleMin = savedMin; p.lrCycleMax = savedMax }

        let model = makeModel()
        model.lrCycleMinText = "0.1"
        model.lrCycleMaxText = "0.01"   // max < min
        model.save()

        XCTAssertTrue(
            model.lrCycleMaxError,
            "LR cycle max < min must flag the max field (would otherwise fall back to static LR silently)")
    }

    func test_momentumCycleMaxBelowMin_flagsMaxError() {
        let p = TrainingParameters.shared
        let savedMin = p.momentumCycleMin
        let savedMax = p.momentumCycleMax
        defer { p.momentumCycleMin = savedMin; p.momentumCycleMax = savedMax }

        let model = makeModel()
        model.momentumCycleMinText = "0.95"
        model.momentumCycleMaxText = "0.85"   // max < min
        model.save()

        XCTAssertTrue(
            model.momentumCycleMaxError,
            "Momentum cycle max < min must flag the max field (would otherwise run the schedule reversed silently)")
    }

    func test_validCycleEndpoints_noCrossFieldError() {
        let p = TrainingParameters.shared
        let savedLRMin = p.lrCycleMin, savedLRMax = p.lrCycleMax
        let savedMMin = p.momentumCycleMin, savedMMax = p.momentumCycleMax
        defer {
            p.lrCycleMin = savedLRMin; p.lrCycleMax = savedLRMax
            p.momentumCycleMin = savedMMin; p.momentumCycleMax = savedMMax
        }

        let model = makeModel()
        model.lrCycleMinText = "0.001"
        model.lrCycleMaxText = "0.03"          // max > min
        model.momentumCycleMinText = "0.85"
        model.momentumCycleMaxText = "0.95"    // max > min
        model.save()

        XCTAssertFalse(model.lrCycleMaxError, "valid LR endpoints must not flag the cross-field error")
        XCTAssertFalse(model.momentumCycleMaxError, "valid momentum endpoints must not flag the cross-field error")
    }

    func test_equalCycleEndpoints_allowed() {
        let p = TrainingParameters.shared
        let savedLRMin = p.lrCycleMin, savedLRMax = p.lrCycleMax
        defer { p.lrCycleMin = savedLRMin; p.lrCycleMax = savedLRMax }

        let model = makeModel()
        model.lrCycleMinText = "0.01"
        model.lrCycleMaxText = "0.01"          // max == min is legal (flat cycle)
        model.save()

        XCTAssertFalse(model.lrCycleMaxError, "max == min is a valid (degenerate flat) cycle")
    }
}
