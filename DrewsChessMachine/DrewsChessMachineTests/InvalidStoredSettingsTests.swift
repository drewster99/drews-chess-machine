//
//  InvalidStoredSettingsTests.swift
//  DrewsChessMachineTests
//
//  Settings that cannot be used as found are shown to the user and replaced
//  only when they choose — never silently. Two sources:
//
//  - Stored preferences (`UserDefaults`, one entry per training parameter).
//    The loader used to swallow a wrong-typed or out-of-range entry with
//    `try?` and quietly run on the default.
//  - A resumed session's `session.json`. An unknown policy-smoothing mode, a
//    partial smoothing set, an unknown arena criterion or an incomplete SPRT
//    block used to resume on the live setting with only a log line.
//

import XCTest
@testable import DrewsChessMachine

final class InvalidStoredSettingsTests: XCTestCase {

    private var suiteName = ""
    private var defaults = UserDefaults()

    override func setUpWithError() throws {
        try super.setUpWithError()
        suiteName = "InvalidStoredSettingsTests-\(UUID().uuidString)"
        guard let suite = UserDefaults(suiteName: suiteName) else {
            throw XCTSkip("could not create a private UserDefaults suite")
        }
        defaults = suite
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suiteName)
        try super.tearDownWithError()
    }

    // MARK: - Stored preferences

    func testAbsentStoredValueIsAbsent() {
        guard case .absent = TrainingParameters.inspectStored(LearningRate.self, in: defaults) else {
            return XCTFail("nothing stored must read as absent")
        }
    }

    func testValidStoredValueIsUsed() {
        defaults.set(0.002, forKey: LearningRate.id)
        guard case .valid(let value) = TrainingParameters.inspectStored(LearningRate.self, in: defaults) else {
            return XCTFail("an in-range number must be valid")
        }
        XCTAssertEqual(value, 0.002)
    }

    func testOutOfRangeStoredValueIsReportedNotSwallowed() {
        defaults.set(0.99, forKey: PolicyLabelSmoothingPerMove.id)
        guard case .invalid(let finding) = TrainingParameters.inspectStored(PolicyLabelSmoothingPerMove.self, in: defaults) else {
            return XCTFail("an out-of-range value must be reported")
        }
        XCTAssertEqual(finding.id, PolicyLabelSmoothingPerMove.id)
        XCTAssertEqual(finding.found, "0.99")
        XCTAssertTrue(finding.problem.contains("out of range"), finding.problem)
        XCTAssertEqual(finding.replacement, PolicyLabelSmoothingPerMove.definition.defaultValue.displayText)
    }

    func testWrongTypeStoredValueIsReportedNotSwallowed() {
        defaults.set("fast", forKey: LearningRate.id)
        guard case .invalid(let finding) = TrainingParameters.inspectStored(LearningRate.self, in: defaults) else {
            return XCTFail("a string where a number belongs must be reported")
        }
        XCTAssertEqual(finding.found, "fast")
        XCTAssertTrue(finding.problem.contains("not a number"), finding.problem)
    }

    func testOutOfRangeIntAndWrongTypeBoolAreReported() {
        defaults.set(-5, forKey: LRWarmupSteps.id)
        guard case .invalid = TrainingParameters.inspectStored(LRWarmupSteps.self, in: defaults) else {
            return XCTFail("a negative warmup must be reported")
        }
        defaults.set("yes", forKey: SqrtBatchScalingLR.id)
        guard case .invalid = TrainingParameters.inspectStored(SqrtBatchScalingLR.self, in: defaults) else {
            return XCTFail("a string where a Bool belongs must be reported")
        }
    }

    // MARK: - Saved session settings

    @MainActor
    private func current() -> TrainingParametersSnapshot {
        TrainingParameters.shared.snapshot()
    }

    private func sessionState() -> SessionCheckpointState {
        SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "test-session",
            savedAtUnix: 1_700_000_000,
            sessionStartUnix: 1_699_999_000,
            elapsedTrainingSec: 1000,
            trainingSteps: 1234,
            selfPlayGames: 10,
            selfPlayMoves: 600,
            trainingPositionsSeen: 1234 * 4096,
            batchSize: 4096,
            learningRate: 5e-5,
            promoteThreshold: 0.55,
            arenaGames: 200,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 4,
            championID: "champ-id",
            trainerID: "train-id",
            arenaHistory: []
        )
    }

    @MainActor
    func testASessionTheAppWroteHasNoFindings() {
        var state = sessionState()
        state.policyLabelSmoothingMode = "per_move"
        state.policyLabelSmoothingPerMove = 0.004
        state.policyLabelSmoothingPerMoveCap = 0.4
        state.arenaPromotionCriterion = ArenaPromotionCriterion.sprt.logToken
        state.arenaSPRTElo0 = 0
        state.arenaSPRTElo1 = 10
        state.arenaSPRTAlpha = 0.05
        state.arenaSPRTBeta = 0.05
        state.arenaSPRTMinGames = 32
        state.arenaSPRTMaxGames = 400
        state.periodicAutosaveIntervalSec = 3600
        XCTAssertEqual(state.invalidSavedSettings(current: current()), [])
        XCTAssertEqual(sessionState().invalidSavedSettings(current: current()), [], "a pre-feature session has none of these fields")
    }

    @MainActor
    func testUnknownSmoothingModeIsAFinding() {
        var state = sessionState()
        state.policyLabelSmoothingMode = "per_mvoe"
        state.policyLabelSmoothingPerMove = 0.004
        state.policyLabelSmoothingPerMoveCap = 0.4
        let findings = state.invalidSavedSettings(current: current())
        XCTAssertEqual(findings.map(\.id), [SessionCheckpointState.SavedSettingID.policyLabelSmoothing])
        XCTAssertTrue(findings[0].problem.contains("per_mvoe"))
    }

    @MainActor
    func testPartialSmoothingSetIsAFinding() {
        var state = sessionState()
        state.policyLabelSmoothingMode = "per_move"
        state.policyLabelSmoothingPerMove = 0.004
        XCTAssertEqual(state.invalidSavedSettings(current: current()).map(\.id),
                       [SessionCheckpointState.SavedSettingID.policyLabelSmoothing])
        var deltaOnly = sessionState()
        deltaOnly.policyLabelSmoothingPerMove = 0.004
        XCTAssertEqual(deltaOnly.invalidSavedSettings(current: current()).map(\.id),
                       [SessionCheckpointState.SavedSettingID.policyLabelSmoothing])
    }

    @MainActor
    func testUnknownOrIncompleteArenaCriterionIsAFinding() {
        var unknown = sessionState()
        unknown.arenaPromotionCriterion = "coin_flip"
        XCTAssertEqual(unknown.invalidSavedSettings(current: current()).map(\.id),
                       [SessionCheckpointState.SavedSettingID.arenaPromotionCriterion])
        var incomplete = sessionState()
        incomplete.arenaPromotionCriterion = ArenaPromotionCriterion.sprt.logToken
        incomplete.arenaSPRTElo0 = 0
        XCTAssertEqual(incomplete.invalidSavedSettings(current: current()).map(\.id),
                       [SessionCheckpointState.SavedSettingID.arenaPromotionCriterion])
    }

    @MainActor
    func testNonPositiveAutosaveIntervalIsAFinding() {
        var state = sessionState()
        state.periodicAutosaveIntervalSec = 0
        XCTAssertEqual(state.invalidSavedSettings(current: current()).map(\.id),
                       [SessionCheckpointState.SavedSettingID.periodicAutosaveInterval])
    }
}
