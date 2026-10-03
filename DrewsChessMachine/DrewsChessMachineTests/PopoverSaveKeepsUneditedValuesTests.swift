//
//  PopoverSaveKeepsUneditedValuesTests.swift
//  DrewsChessMachineTests
//
//  The settings popovers seed each Double field with a rounded rendering of
//  the live value ("%.4f", "%.2e", …). Save used to parse every field and
//  write it back whenever it differed by more than one ulp, so opening the
//  popover and clicking Save to change something unrelated silently rounded
//  a precise value (a `parameters.json` δ of 1/300 became 0.0033), and the
//  `%.4f -> %.4f` log line printed the same text on both sides. Save now
//  writes only fields whose text was edited.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class PopoverSaveKeepsUneditedValuesTests: XCTestCase {

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

    private func makeTrainingModel() -> TrainingSettingsPopoverModel {
        TrainingSettingsPopoverModel(selfPlayDelayMaxMs: 1000, stepDelayMaxMs: 1000, maxSelfPlayWorkers: 8)
    }

    private func makeArenaModel() -> ArenaSettingsPopoverModel {
        ArenaSettingsPopoverModel(
            maxConcurrency: 4096,
            formatDurationSpec: { "\(Int($0))s" },
            parseDurationSpec: { Double($0.replacingOccurrences(of: "s", with: "")) }
        )
    }

    func testSaveLeavesAnUneditedPerMoveDeltaAndCapUnchanged() {
        let p = TrainingParameters.shared
        p.policyLabelSmoothingMode = .perMove
        p.policyLabelSmoothingPerMove = 1.0 / 300
        p.policyLabelSmoothingPerMoveCap = 1.0 / 3
        let model = makeTrainingModel()
        model.save()
        XCTAssertEqual(p.policyLabelSmoothingPerMove, 1.0 / 300)
        XCTAssertEqual(p.policyLabelSmoothingPerMoveCap, 1.0 / 3)
    }

    func testSaveLeavesUneditedOptimizerValuesUnchanged() {
        let p = TrainingParameters.shared
        p.learningRate = 0.0012345678
        p.momentumCoeff = 0.912345
        p.weightDecay = 0.000123456
        p.policyLabelSmoothingEpsilon = 0.1234567
        let model = makeTrainingModel()
        model.save()
        XCTAssertEqual(p.learningRate, 0.0012345678)
        XCTAssertEqual(p.momentumCoeff, 0.912345)
        XCTAssertEqual(p.weightDecay, 0.000123456)
        XCTAssertEqual(p.policyLabelSmoothingEpsilon, 0.1234567)
    }

    func testSaveWritesAnEditedPerMoveDelta() {
        let p = TrainingParameters.shared
        p.policyLabelSmoothingMode = .perMove
        p.policyLabelSmoothingPerMove = 1.0 / 300
        let model = makeTrainingModel()
        model.policyLabelSmoothingPerMoveText = "0.004"
        model.save()
        XCTAssertEqual(p.policyLabelSmoothingPerMove, 0.004)
    }

    func testArenaSaveLeavesUneditedTauAndSPRTValuesUnchanged() {
        let p = TrainingParameters.shared
        p.arenaStartTau = 0.234567
        p.arenaTargetTau = 0.0234567
        p.arenaTauDecayPerPly = 0.0123456
        p.arenaSPRTAlpha = 0.0512345
        let model = makeArenaModel()
        model.save()
        XCTAssertEqual(p.arenaStartTau, 0.234567)
        XCTAssertEqual(p.arenaTargetTau, 0.0234567)
        XCTAssertEqual(p.arenaTauDecayPerPly, 0.0123456)
        XCTAssertEqual(p.arenaSPRTAlpha, 0.0512345)
    }
}
