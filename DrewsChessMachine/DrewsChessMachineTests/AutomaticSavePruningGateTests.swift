//
//  AutomaticSavePruningGateTests.swift
//  DrewsChessMachineTests
//
//  Pins the gate in front of automatic-save retention (owner decision
//  2026-10-01): pruning runs only when the build's kill switch
//  (`CheckpointPaths.automaticSavePruningForcedOff`) is lifted, the
//  `automatic_save_pruning_enabled` setting is on, and the
//  `max_periodic_autosaves_kept` cap is above zero.
//
//  - The shipped kill switch is pinned to `true`, so re-enabling pruning
//    is a deliberate, visible change to this file, not a one-line flip
//    nobody notices.
//  - The decision function takes the switch as an argument, so the paths
//    a build with the switch lifted would take are tested here while the
//    real constant stays thrown.
//  - The new parameter is pinned to its id, default, category and live
//    tunability, its value round-trips, it reaches the emitted defaults,
//    and its `session.json` field decodes when absent (older sessions).
//
//  The retention sweep itself is covered by `CheckpointHousekeepingTests`.
//

import XCTest
@testable import DrewsChessMachine

final class AutomaticSavePruningGateTests: XCTestCase {

    /// Caps spanning the declared range: zero (unlimited), the smallest
    /// real cap, the declared default, and the declared maximum.
    private var capsToCheck: [Int] {
        let range = MaxPeriodicAutosavesKept.declaredClosedRange
        return [range.lowerBound, 1, MaxPeriodicAutosavesKept.declaredDefault, range.upperBound]
    }

    // MARK: - Kill switch

    func testShippedBuildForcesAutomaticSavePruningOff() {
        XCTAssertTrue(
            CheckpointPaths.automaticSavePruningForcedOff,
            "Automatic-save pruning is forced off by owner decision 2026-10-01; lifting it must be a deliberate change"
        )
    }

    func testForcedOffNeverPrunesWhateverTheSettingOrCap() {
        for settingEnabled in [false, true] {
            for cap in capsToCheck {
                XCTAssertEqual(
                    CheckpointPaths.automaticSavePruningDecision(forcedOff: true, settingEnabled: settingEnabled, cap: cap),
                    .forcedOff,
                    "setting=\(settingEnabled) cap=\(cap)"
                )
            }
        }
    }

    func testShippedDecisionForTheDeclaredDefaultsIsForcedOff() {
        let settingDefault = AutomaticSavePruningEnabled.declaredDefault
        XCTAssertEqual(
            CheckpointPaths.automaticSavePruningDecision(
                forcedOff: CheckpointPaths.automaticSavePruningForcedOff,
                settingEnabled: settingDefault,
                cap: MaxPeriodicAutosavesKept.declaredDefault
            ),
            .forcedOff
        )
    }

    // MARK: - Decision with the switch lifted

    func testSettingOffNeverPrunes() {
        for cap in capsToCheck {
            XCTAssertEqual(
                CheckpointPaths.automaticSavePruningDecision(forcedOff: false, settingEnabled: false, cap: cap),
                .disabledBySetting,
                "cap=\(cap)"
            )
        }
    }

    func testSettingOnWithZeroCapDoesNotPrune() {
        XCTAssertEqual(
            CheckpointPaths.automaticSavePruningDecision(forcedOff: false, settingEnabled: true, cap: 0),
            .unlimitedCap
        )
    }

    func testSettingOnWithPositiveCapPrunesToThatCap() {
        for cap in capsToCheck where cap > 0 {
            XCTAssertEqual(
                CheckpointPaths.automaticSavePruningDecision(forcedOff: false, settingEnabled: true, cap: cap),
                .prune(keeping: cap),
                "cap=\(cap)"
            )
        }
    }

    /// The setting's default alone keeps pruning off, independently of the
    /// kill switch — lifting the switch does not by itself start deleting.
    func testDeclaredDefaultsDoNotPruneEvenWithTheSwitchLifted() {
        let settingDefault = AutomaticSavePruningEnabled.declaredDefault
        XCTAssertEqual(
            CheckpointPaths.automaticSavePruningDecision(
                forcedOff: false,
                settingEnabled: settingDefault,
                cap: MaxPeriodicAutosavesKept.declaredDefault
            ),
            .disabledBySetting
        )
    }

    // MARK: - Log text

    func testLogDescriptionNamesTheReasonAndBothSettings() {
        let forcedOff = CheckpointPaths.AutomaticSavePruningDecision.forcedOff
            .logDescription(settingEnabled: true, cap: 7)
        XCTAssertTrue(forcedOff.contains("forced off in this build"), forcedOff)
        XCTAssertTrue(forcedOff.contains("\(AutomaticSavePruningEnabled.id)=true"), forcedOff)
        XCTAssertTrue(forcedOff.contains("\(MaxPeriodicAutosavesKept.id)=7"), forcedOff)

        let disabled = CheckpointPaths.AutomaticSavePruningDecision.disabledBySetting
            .logDescription(settingEnabled: false, cap: 3)
        XCTAssertTrue(disabled.contains("disabled by setting"), disabled)
        XCTAssertTrue(disabled.contains("\(AutomaticSavePruningEnabled.id)=false"), disabled)

        let unlimited = CheckpointPaths.AutomaticSavePruningDecision.unlimitedCap
            .logDescription(settingEnabled: true, cap: 0)
        XCTAssertTrue(unlimited.contains("unlimited"), unlimited)
        XCTAssertTrue(unlimited.contains("\(MaxPeriodicAutosavesKept.id)=0"), unlimited)

        let prune = CheckpointPaths.AutomaticSavePruningDecision.prune(keeping: 4)
            .logDescription(settingEnabled: true, cap: 4)
        XCTAssertTrue(prune.hasPrefix("on"), prune)
    }

    // MARK: - Parameter declaration

    func testParameterDeclaration() {
        let definition = AutomaticSavePruningEnabled.definition
        XCTAssertEqual(AutomaticSavePruningEnabled.id, "automatic_save_pruning_enabled")
        XCTAssertEqual(definition.id, AutomaticSavePruningEnabled.id)
        XCTAssertEqual(definition.defaultValue, .bool(false), "pruning is off by default")
        XCTAssertTrue(definition.liveTunable, "read live after each automatic save")
        XCTAssertEqual(
            definition.category,
            MaxPeriodicAutosavesKept.definition.category,
            "lives beside the cap it gates"
        )
        XCTAssertNoThrow(try definition.validate(definition.defaultValue))
        XCTAssertThrowsError(try definition.validate(.int(1)), "a Bool parameter must reject a non-Bool value")
    }

    func testParameterIsRegisteredExactlyOnce() {
        let matches = TrainingParameters.allKeys.filter { $0.id == AutomaticSavePruningEnabled.id }
        XCTAssertEqual(matches.count, 1)
    }

    func testParameterValueRoundTrips() throws {
        for value in [false, true] {
            XCTAssertEqual(try AutomaticSavePruningEnabled.decode(AutomaticSavePruningEnabled.encode(value)), value)
        }
    }

    func testParameterAppearsInEmittedDefaults() throws {
        let json = try TrainingParameters.defaultsJSON()
        let object = try JSONSerialization.jsonObject(with: json)
        let dictionary = try XCTUnwrap(object as? [String: Any])
        let value = try XCTUnwrap(dictionary[AutomaticSavePruningEnabled.id] as? Bool, "defaults JSON must carry the key as a Bool")
        XCTAssertFalse(value)

        XCTAssertTrue(TrainingParameters.defaultsMarkdown().contains("### \(AutomaticSavePruningEnabled.id)"))
    }

    // MARK: - Session file field

    /// A valid `SessionCheckpointState` holding only the non-Optional
    /// fields, so `automaticSavePruningEnabled` is absent — the shape of a
    /// session saved before the field existed.
    private func makeStateWithoutOptionalFields() throws -> SessionCheckpointState {
        let sessionID = "20261001-1-AbCd"
        let jsonText = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "sessionID": "\(sessionID)",
          "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400,
          "elapsedTrainingSec": 3600,
          "trainingSteps": 12345,
          "selfPlayGames": 678,
          "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280,
          "batchSize": 1024,
          "learningRate": 5.0e-5,
          "promoteThreshold": 0.55,
          "arenaGames": 200,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.05, "floorTau": 0.4},
          "arenaTau": {"startTau": 1.0, "decayPerPly": 0.05, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "championID": "\(sessionID)",
          "trainerID": "\(sessionID)-1",
          "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(jsonText.utf8))
    }

    func testSessionFieldIsAbsentInOlderSessions() throws {
        XCTAssertNil(try makeStateWithoutOptionalFields().automaticSavePruningEnabled)
    }

    func testSessionFieldRoundTrips() throws {
        for value in [false, true] {
            var state = try makeStateWithoutOptionalFields()
            state.automaticSavePruningEnabled = value
            let decoded = try SessionCheckpointState.decode(state.encode())
            XCTAssertEqual(decoded.automaticSavePruningEnabled, value)
        }
    }
}
