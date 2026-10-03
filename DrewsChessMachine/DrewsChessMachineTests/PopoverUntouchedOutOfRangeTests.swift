//
//  PopoverUntouchedOutOfRangeTests.swift
//  DrewsChessMachineTests
//
//  A field the user did not touch keeps its live value on Save, whatever
//  that value is. A session resume deliberately holds a saved value that
//  lies outside today's declared range (`restoreFromSession`), and Save used
//  to re-parse every whole-number field and every live-propagated field
//  against the declaration — so the popover refused to close over a value
//  nobody edited, often on a tab that was not even showing. The autosave
//  interval and the arena interval were also rewritten on every Save from
//  their rounded text (90 s became 120 s; 900.5 s became 900 s).
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class PopoverUntouchedOutOfRangeTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
        // Every other field starts from its declared default, so only the
        // value under test can keep Save from closing the popover.
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

    private func makeTrainingModel() -> TrainingSettingsPopoverModel {
        TrainingSettingsPopoverModel(
            selfPlayDelayMaxMs: UpperContentView.selfPlayDelayMaxMs,
            stepDelayMaxMs: UpperContentView.stepDelayMaxMs,
            maxSelfPlayWorkers: UpperContentView.absoluteMaxSelfPlayWorkers
        )
    }

    private func makeArenaModel() -> ArenaSettingsPopoverModel {
        ArenaSettingsPopoverModel(
            maxConcurrency: UpperContentView.absoluteMaxArenaConcurrency,
            formatDurationSpec: UpperContentView.formatDurationSpec,
            parseDurationSpec: UpperContentView.parseDurationSpec
        )
    }

    func testTrainingSaveAcceptsUntouchedSessionValuesOutsideTheDeclaredRange() {
        let p = TrainingParameters.shared
        p.restoreFromSession(KLProbeInterval.self, 20_000, into: \.klProbeInterval)
        p.restoreFromSession(LRWarmupSteps.self, 200_000, into: \.lrWarmupSteps)
        p.restoreFromSession(MaxPliesFromAnyOneGame.self, 500, into: \.maxPliesFromAnyOneGame)
        p.restoreFromSession(DrawWatchPDrawThreshold.self, 0.3, into: \.drawWatchPDrawThreshold)
        let model = makeTrainingModel()
        model.isPresented = true
        model.save()
        XCTAssertFalse(model.klProbeIntervalError)
        XCTAssertFalse(model.warmupError)
        XCTAssertFalse(model.maxPliesFromAnyOneGameError)
        XCTAssertFalse(model.drawWatchPDrawThresholdError)
        XCTAssertFalse(model.isPresented, "Save must close over values nobody edited")
        XCTAssertEqual(p.klProbeInterval, 20_000)
        XCTAssertEqual(p.lrWarmupSteps, 200_000)
        XCTAssertEqual(p.maxPliesFromAnyOneGame, 500)
        XCTAssertEqual(p.drawWatchPDrawThreshold, 0.3)
    }

    /// The self-play delay field is hidden while automatic control is on, so
    /// an untouched value it cannot show must not hold Save either.
    func testTrainingSaveAcceptsAnUntouchedSelfPlayDelayAboveTheCap() {
        let p = TrainingParameters.shared
        p.replayRatioAutoAdjust = true
        p.restoreFromSession(SelfPlayDelayMs.self, 5000, into: \.selfPlayDelayMs)
        let model = makeTrainingModel()
        model.isPresented = true
        model.save()
        XCTAssertFalse(model.replaySelfPlayDelayError)
        XCTAssertFalse(model.isPresented)
        XCTAssertEqual(p.selfPlayDelayMs, 5000)
    }

    func testTrainingSaveKeepsAnUntouchedNonWholeMinuteAutosaveInterval() {
        let p = TrainingParameters.shared
        p.periodicAutosaveIntervalSec = 90
        let model = makeTrainingModel()
        model.save()
        XCTAssertEqual(p.periodicAutosaveIntervalSec, 90)
    }

    func testArenaSaveAcceptsAnUntouchedIntervalOutsideTheDeclaredRange() {
        let p = TrainingParameters.shared
        p.restoreFromSession(ArenaAutoIntervalSec.self, 30, into: \.arenaAutoIntervalSec)
        let model = makeArenaModel()
        model.isPresented = true
        model.save()
        XCTAssertFalse(model.intervalError)
        XCTAssertFalse(model.isPresented)
        XCTAssertEqual(p.arenaAutoIntervalSec, 30)
    }

    func testArenaSaveKeepsAnUntouchedFractionalInterval() {
        let p = TrainingParameters.shared
        p.arenaAutoIntervalSec = 900.5
        let model = makeArenaModel()
        model.save()
        XCTAssertEqual(p.arenaAutoIntervalSec, 900.5)
    }

    func testArenaSaveAcceptsAnUntouchedConcurrencyAboveTheCap() {
        let p = TrainingParameters.shared
        p.restoreFromSession(ArenaConcurrency.self, 2000, into: \.arenaConcurrency)
        let model = makeArenaModel()
        model.isPresented = true
        model.save()
        XCTAssertFalse(model.concurrencyError)
        XCTAssertFalse(model.isPresented)
        XCTAssertEqual(p.arenaConcurrency, 2000)
    }
}
