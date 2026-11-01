//
//  LichessBotSettingsStoreTests.swift
//  DrewsChessMachineTests
//
//  Saved Lichess bot settings must survive the settings type gaining fields.
//  Adding `alerts` (c5542b8) made every earlier save fail to decode and forced
//  a Reset to defaults. These pin the replacement behaviour: a field the save
//  predates is filled from today's default and reported; an optional the save
//  left absent stays nil (it may be a deliberate nil); a saved field the type
//  no longer has is ignored and reported; a wrong-typed value is still an
//  unreadable error, never a silent default.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotSettingsStoreTests: XCTestCase {

    /// `settings` encoded as a JSON object, edited by `edit`, stored as the
    /// saved blob.
    private func store(
        _ settings: LichessBotSettings, in defaults: UserDefaults, edit: (inout [String: Any]) throws -> Void
    ) throws {
        let data = try JSONEncoder().encode(settings)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        try edit(&object)
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: LichessBotSettingsStore.defaultsKey)
    }

    func testCompleteSaveLoadsWithNothingFilled() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.challenge.maxConcurrentGames = 3
        try LichessBotSettingsStore.save(settings, to: defaults)
        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings, settings)
        XCTAssertEqual(result.filledFromDefaults, [])
        XCTAssertEqual(result.ignoredSavedKeys, [])
    }

    /// A save from before `alerts` existed (the c5542b8 break) keeps every
    /// saved value and gets today's `alerts` defaults.
    func testSaveMissingAWholeSectionKeepsEverythingElse() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.challenge.maxConcurrentGames = 7
        settings.play.temperatureStart = 0.3
        settings.matchmaking.enabled = false
        try store(settings, in: defaults) { $0.removeValue(forKey: "alerts") }

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.filledFromDefaults, ["alerts"])
        XCTAssertEqual(result.settings.challenge.maxConcurrentGames, 7)
        XCTAssertEqual(result.settings.play.temperatureStart, 0.3)
        XCTAssertFalse(result.settings.matchmaking.enabled)
        XCTAssertEqual(result.settings.alerts, LichessBotAlertSettings())
    }

    func testSaveMissingANestedFieldFillsOnlyThatField() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.challenge.maxConcurrentGames = 4
        settings.challenge.maximumOpponentRating = 1800
        try store(settings, in: defaults) { object in
            var challenge = try XCTUnwrap(object["challenge"] as? [String: Any])
            challenge.removeValue(forKey: "maximumOpponentRating")
            object["challenge"] = challenge
        }

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.filledFromDefaults, ["challenge.maximumOpponentRating"])
        XCTAssertEqual(result.settings.challenge.maxConcurrentGames, 4)
        XCTAssertEqual(result.settings.challenge.maximumOpponentRating, LichessBotChallengeSettings().maximumOpponentRating)
    }

    /// `model.filePath` is optional with a non-nil default. A save that left
    /// it out (the operator cleared it) must load as nil, not as the default.
    func testAbsentOptionalStaysNil() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.model.source = .champion
        settings.model.filePath = nil
        try LichessBotSettingsStore.save(settings, to: defaults)

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertNil(result.settings.model.filePath)
        XCTAssertEqual(result.filledFromDefaults, [])
        XCTAssertEqual(result.settings, settings)
    }

    /// An optional whose default is nil, saved with a value, keeps the value
    /// and is not reported as an unknown key.
    func testSavedOptionalWithNilDefaultIsKept() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.alerts.botChallengeSoundName = "Ping"
        try LichessBotSettingsStore.save(settings, to: defaults)

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings.alerts.botChallengeSoundName, "Ping")
        XCTAssertEqual(result.ignoredSavedKeys, [])
    }

    /// A section saved empty (every field a nil optional, e.g. `alerts` with
    /// both tones off) still merges, so the section can gain non-optional
    /// fields later. Simulated by emptying `matchmaking` in the save: every
    /// one of its fields is filled from the defaults.
    func testEmptySavedSectionStillMerges() throws {
        let defaults = try makeTemporaryDefaults()
        try store(LichessBotSettings(), in: defaults) { $0["matchmaking"] = [String: Any]() }
        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings, LichessBotSettings())
        XCTAssertTrue(result.filledFromDefaults.contains("matchmaking.enabled"))
        XCTAssertTrue(result.filledFromDefaults.allSatisfy { $0.hasPrefix("matchmaking.") })
    }

    func testSavedFieldTheTypeNoLongerHasIsIgnoredAndReported() throws {
        let defaults = try makeTemporaryDefaults()
        try store(LichessBotSettings(), in: defaults) { object in
            var challenge = try XCTUnwrap(object["challenge"] as? [String: Any])
            challenge["retiredSetting"] = 95
            object["challenge"] = challenge
            object["retiredSection"] = ["x": 1]
        }

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings, LichessBotSettings())
        XCTAssertEqual(result.ignoredSavedKeys, ["challenge.retiredSetting", "retiredSection"])
    }

    /// Only missing fields are filled; a present value of the wrong type is
    /// still an unreadable error.
    func testWrongTypedValueIsStillUnreadable() throws {
        let defaults = try makeTemporaryDefaults()
        try store(LichessBotSettings(), in: defaults) { object in
            var challenge = try XCTUnwrap(object["challenge"] as? [String: Any])
            challenge["maxConcurrentGames"] = "twelve"
            object["challenge"] = challenge
        }
        XCTAssertThrowsError(try LichessBotSettingsStore.loadReporting(from: defaults)) { error in
            guard case .unreadable? = error as? LichessBotSettingsStoreError else {
                return XCTFail("expected unreadable, got \(error)")
            }
        }
    }

    // MARK: Per-opponent daily limit split (2026-10-12)

    /// A save with the single per-opponent daily limit carries its value into
    /// both the human and the bot limit; nothing is filled or ignored.
    func testTheSinglePerOpponentLimitCarriesIntoBothNewLimits() throws {
        let defaults = try makeTemporaryDefaults()
        try store(LichessBotSettings(), in: defaults) { object in
            var challenge = try XCTUnwrap(object["challenge"] as? [String: Any])
            challenge["maxGamesPerHumanOpponentPerDay"] = nil
            challenge["maxGamesPerBotOpponentPerDay"] = nil
            challenge["maxGamesPerOpponentPerDay"] = 17
            object["challenge"] = challenge
        }
        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings.challenge.maxGamesPerHumanOpponentPerDay, 17)
        XCTAssertEqual(result.settings.challenge.maxGamesPerBotOpponentPerDay, 17)
        XCTAssertEqual(result.filledFromDefaults, [])
        XCTAssertEqual(result.ignoredSavedKeys, [])
        XCTAssertEqual(result.carriedForward.count, 1)
    }

    /// A save that already holds a new limit keeps it; the old key is dropped.
    func testASaveWithTheNewLimitsKeepsThem() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.challenge.maxGamesPerHumanOpponentPerDay = 3
        settings.challenge.maxGamesPerBotOpponentPerDay = 9
        try store(settings, in: defaults) { object in
            var challenge = try XCTUnwrap(object["challenge"] as? [String: Any])
            challenge["maxGamesPerOpponentPerDay"] = 17
            object["challenge"] = challenge
        }
        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings.challenge.maxGamesPerHumanOpponentPerDay, 3)
        XCTAssertEqual(result.settings.challenge.maxGamesPerBotOpponentPerDay, 9)
        XCTAssertEqual(result.carriedForward, [])
        XCTAssertEqual(result.ignoredSavedKeys, [])
    }

    // MARK: Minimum think time split (2026-11-01)

    /// Protects: a save made before the split keeps its think time for every
    /// opponent. Fails if the carry-forward is dropped (both fall to 0) or
    /// fills only one side.
    func testTheSingleMinimumThinkTimeCarriesIntoBothNewTimes() throws {
        let defaults = try makeTemporaryDefaults()
        try store(LichessBotSettings(), in: defaults) { object in
            var play = try XCTUnwrap(object["play"] as? [String: Any])
            play["minimumThinkMillisecondsVersusHumans"] = nil
            play["minimumThinkMillisecondsVersusBots"] = nil
            play["minimumThinkMilliseconds"] = 7000
            object["play"] = play
        }
        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings.play.minimumThinkMillisecondsVersusHumans, 7000)
        XCTAssertEqual(result.settings.play.minimumThinkMillisecondsVersusBots, 7000)
        XCTAssertEqual(result.filledFromDefaults, [])
        XCTAssertEqual(result.ignoredSavedKeys, [])
        XCTAssertEqual(result.carriedForward.count, 1)
    }

    /// Protects: a save that already holds the split times keeps them; a
    /// stray old key never overwrites them.
    func testASaveWithTheSplitThinkTimesKeepsThem() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.play.minimumThinkMillisecondsVersusHumans = 7000
        settings.play.minimumThinkMillisecondsVersusBots = 0
        try store(settings, in: defaults) { object in
            var play = try XCTUnwrap(object["play"] as? [String: Any])
            play["minimumThinkMilliseconds"] = 3000
            object["play"] = play
        }
        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings.play.minimumThinkMillisecondsVersusHumans, 7000)
        XCTAssertEqual(result.settings.play.minimumThinkMillisecondsVersusBots, 0)
        XCTAssertEqual(result.carriedForward, [])
        XCTAssertEqual(result.ignoredSavedKeys, [])
    }

    /// Protects: humans get the human time; bots and Lichess's AI the bots'.
    /// Fails if the kinds are swapped or Lichess's AI is treated as human.
    func testMinimumThinkTimeFollowsTheOpponentKind() {
        var play = LichessBotPlaySettings()
        play.minimumThinkMillisecondsVersusHumans = 7000
        play.minimumThinkMillisecondsVersusBots = 250
        XCTAssertEqual(play.minimumThinkMilliseconds(against: .human), 7000)
        XCTAssertEqual(play.minimumThinkMilliseconds(against: .bot), 250)
        XCTAssertEqual(play.minimumThinkMilliseconds(against: .lichessAI), 250)
    }

    /// Protects: a negative time on either side is refused.
    func testANegativeMinimumThinkTimeIsInvalid() {
        for side in ["humans", "bots"] {
            var settings = LichessBotSettings()
            if side == "humans" {
                settings.play.minimumThinkMillisecondsVersusHumans = -1
            } else {
                settings.play.minimumThinkMillisecondsVersusBots = -1
            }
            XCTAssertFalse(settings.validationProblems().isEmpty, side)
        }
    }
}
