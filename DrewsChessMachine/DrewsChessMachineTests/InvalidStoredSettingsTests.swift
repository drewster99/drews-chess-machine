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

    /// A whole-number setting stored as a real number is reported, not
    /// truncated: 7.9 read as 7 would silently run a different setting.
    func testFractionalNumberForWholeNumberParameterIsReported() {
        defaults.set(7.9, forKey: LRWarmupSteps.id)
        guard case .invalid(let finding) = TrainingParameters.inspectStored(LRWarmupSteps.self, in: defaults) else {
            return XCTFail("a real number where a whole number belongs must be reported")
        }
        XCTAssertEqual(finding.found, "7.9")
        XCTAssertTrue(finding.problem.contains("not a whole number"), finding.problem)
    }

    func testTrueForWholeNumberParameterIsReported() {
        defaults.set(true, forKey: LRWarmupSteps.id)
        guard case .invalid(let finding) = TrainingParameters.inspectStored(LRWarmupSteps.self, in: defaults) else {
            return XCTFail("true/false where a whole number belongs must be reported, not read as 1")
        }
        XCTAssertTrue(finding.problem.contains("not a whole number"), finding.problem)
    }

    func testNumberForTrueFalseParameterIsReported() {
        defaults.set(1, forKey: SqrtBatchScalingLR.id)
        guard case .invalid(let finding) = TrainingParameters.inspectStored(SqrtBatchScalingLR.self, in: defaults) else {
            return XCTFail("a number where true/false belongs must be reported, not read as true")
        }
        XCTAssertTrue(finding.problem.contains("not a true/false value"), finding.problem)
    }

    // MARK: - Reset repairs the stored entry only
    //
    // The settings list's Reset used to assign the declared default to the
    // live parameter (and persist it), replacing a value this run had set on
    // purpose — a `--parameters` file's or a resumed session's — and
    // overwriting a stored value the user had meanwhile made valid. A reset
    // now rewrites the stored entry, and only while it is still unusable.

    func testRepairReplacesAStillInvalidStoredValueWithTheDeclaredDefault() {
        defaults.set("not a number", forKey: LearningRate.id)
        let repair = TrainingParameters.repairStoredValue(LearningRate.self, in: defaults)
        XCTAssertEqual(repair, .replacedWithDeclaredDefault(LearningRate.definition.defaultValue))
        guard case .valid(let value) = TrainingParameters.inspectStored(LearningRate.self, in: defaults) else {
            return XCTFail("the repaired entry must read back as usable")
        }
        XCTAssertEqual(value, LearningRate.declaredDefault)
    }

    func testRepairLeavesAValidReplacementAlone() {
        defaults.set(0.002, forKey: LearningRate.id)
        XCTAssertEqual(TrainingParameters.repairStoredValue(LearningRate.self, in: defaults), .alreadyUsable)
        XCTAssertEqual(defaults.double(forKey: LearningRate.id), 0.002)
    }

    func testRepairStoresUInt64AsDecimalString() {
        defaults.set(42, forKey: RandomSeed.id)
        XCTAssertEqual(TrainingParameters.repairStoredValue(RandomSeed.self, in: defaults),
                       .replacedWithDeclaredDefault(RandomSeed.definition.defaultValue))
        XCTAssertEqual(defaults.string(forKey: RandomSeed.id), String(RandomSeed.declaredDefault))
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

    /// A session saves the criterion and its six SPRT values together or not
    /// at all. Hypotheses without a criterion are not a pre-feature session;
    /// resuming them would install an SPRT test nobody validated (elo1 below
    /// elo0 here), which the next SPRT arena would refuse.
    @MainActor
    func testSPRTHypothesesWithoutACriterionAreAFinding() {
        var state = sessionState()
        state.arenaSPRTElo0 = 10
        state.arenaSPRTElo1 = 0
        state.arenaSPRTAlpha = 0.05
        state.arenaSPRTBeta = 0.05
        state.arenaSPRTMinGames = 32
        state.arenaSPRTMaxGames = 400
        XCTAssertEqual(state.invalidSavedSettings(current: current()).map(\.id),
                       [SessionCheckpointState.SavedSettingID.arenaPromotionCriterion])
    }

    @MainActor
    func testAPartialSPRTSetWithoutACriterionIsAFinding() {
        var state = sessionState()
        state.arenaSPRTElo1 = 10
        XCTAssertEqual(state.invalidSavedSettings(current: current()).map(\.id),
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
