//
//  TrainingPopoverLiveFieldRangeTests.swift
//  DrewsChessMachineTests
//
//  The Replay tab's live fields (target ratio, the two delays) and their
//  steppers take their ranges from the parameter declarations, and Save
//  commits what was typed. The steppers and text handlers used to restate
//  narrower ranges as literals, so a ratio typed inside the declaration but
//  outside the literal range was never applied while Save closed as if it
//  had been, and a train-step delay above the engine's cap was silently
//  clamped while the field still showed the typed value.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class TrainingPopoverLiveFieldRangeTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
        var defaults: [String: ParameterValue] = [:]
        for key in TrainingParameters.allKeys {
            defaults[key.id] = key.definition.defaultValue
        }
        try TrainingParameters.shared.apply(defaults)
    }

    override func tearDown() async throws {
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    private func makeModel() -> TrainingSettingsPopoverModel {
        TrainingSettingsPopoverModel(
            selfPlayDelayMaxMs: UpperContentView.selfPlayDelayMaxMs,
            stepDelayMaxMs: UpperContentView.stepDelayMaxMs,
            maxSelfPlayWorkers: UpperContentView.absoluteMaxSelfPlayWorkers
        )
    }

    func testTypedReplayRatioInsideTheDeclarationButOutsideTheOldStepperRangeIsApplied() {
        let p = TrainingParameters.shared
        let model = makeModel()
        model.isPresented = true
        model.replayRatioTargetText = "0.05"
        model.save()
        XCTAssertFalse(model.isPresented)
        XCTAssertEqual(p.replayRatioTarget, 0.05)
    }

    func testTrainStepDelayAboveTheEngineCapIsRejectedNotClamped() {
        let p = TrainingParameters.shared
        p.replayRatioAutoAdjust = false
        let model = makeModel()
        model.isPresented = true
        model.replayTrainingStepDelayText = "5000"
        model.save()
        XCTAssertTrue(model.replayTrainingStepDelayError)
        XCTAssertTrue(model.isPresented)
    }

    func testStepperRangesMatchTheDeclarations() {
        let model = makeModel()
        XCTAssertEqual(model.selfPlayConcurrencyRange, SelfPlayConcurrency.declaredClosedRange)
        XCTAssertEqual(model.selfPlayDelayRange, SelfPlayDelayMs.declaredClosedRange)
        XCTAssertEqual(model.trainingStepDelayRange, TrainingStepDelayMs.declaredClosedRange)
        let capped = TrainingSettingsPopoverModel(selfPlayDelayMaxMs: 100, stepDelayMaxMs: 200, maxSelfPlayWorkers: 8)
        XCTAssertEqual(capped.selfPlayConcurrencyRange, 1...8)
        XCTAssertEqual(capped.selfPlayDelayRange, 0...100)
        XCTAssertEqual(capped.trainingStepDelayRange, 0...200)
    }

    func testTypedDelaysAreAppliedOnSave() {
        let p = TrainingParameters.shared
        p.replayRatioAutoAdjust = false
        let model = makeModel()
        model.isPresented = true
        model.replaySelfPlayDelayText = "250"
        model.replayTrainingStepDelayText = "75"
        model.save()
        XCTAssertFalse(model.isPresented)
        XCTAssertEqual(p.selfPlayDelayMs, 250)
        XCTAssertEqual(p.trainingStepDelayMs, 75)
    }
}
