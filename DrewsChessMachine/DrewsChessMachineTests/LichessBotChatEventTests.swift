import XCTest
@testable import DrewsChessMachine

/// The event-message feature (chat-events plan §2): trigger detection, the
/// per-game spam guard, forward-compatible settings decode, the room
/// carry-forward from the old single `room` picker, and event-template
/// validation.
final class LichessBotChatEventTests: XCTestCase {

    // MARK: - Context helper

    private func context(
        ply: Int = 10,
        moveNumber: Int = 5,
        ourQueensBefore: Int = 1,
        ourQueensNow: Int = 1,
        theirQueensBefore: Int = 1,
        theirQueensNow: Int = 1,
        inCheckNow: Bool = false,
        promotionJustPlayed: Bool = false,
        ourWinPercent: Int? = nil,
        previousOurWinPercent: Int? = nil,
        newReadingThisStep: Bool = false,
        opponentName: String? = "Opp"
    ) -> LichessBotChatEventContext {
        LichessBotChatEventContext(
            ply: ply, moveNumber: moveNumber,
            ourQueensBefore: ourQueensBefore, ourQueensNow: ourQueensNow,
            theirQueensBefore: theirQueensBefore, theirQueensNow: theirQueensNow,
            inCheckNow: inCheckNow, promotionJustPlayed: promotionJustPlayed,
            ourWinPercent: ourWinPercent, previousOurWinPercent: previousOurWinPercent,
            newReadingThisStep: newReadingThisStep, opponentName: opponentName)
    }

    private let rules = LichessBotChatEventSettings.starterRules

    private func matches(_ context: LichessBotChatEventContext) -> [LichessBotChatEventTrigger] {
        LichessBotChatEventEvaluator.matchingTriggers(context, rules: rules)
    }

    // MARK: - Trigger detection

    func testQueenCaptureTriggers() {
        XCTAssertTrue(matches(context(ourQueensBefore: 1, ourQueensNow: 0)).contains(.weLostQueen))
        XCTAssertFalse(matches(context(ourQueensBefore: 1, ourQueensNow: 1)).contains(.weLostQueen))
        XCTAssertTrue(matches(context(theirQueensBefore: 1, theirQueensNow: 0)).contains(.weWonQueen))
        XCTAssertFalse(matches(context(theirQueensBefore: 1, theirQueensNow: 1)).contains(.weWonQueen))
    }

    func testEarlyCheckRespectsPlyThreshold() {
        // Default threshold is 12 plies.
        XCTAssertTrue(matches(context(ply: 8, inCheckNow: true)).contains(.earlyCheck))
        XCTAssertFalse(matches(context(ply: 40, inCheckNow: true)).contains(.earlyCheck))
        XCTAssertFalse(matches(context(ply: 8, inCheckNow: false)).contains(.earlyCheck))
    }

    func testWinningAndLosingFireOnlyOnTheCrossingMove() {
        // Default thresholds 90 / 10.
        XCTAssertTrue(matches(context(ourWinPercent: 92, previousOurWinPercent: 80, newReadingThisStep: true)).contains(.winningBig))
        // Already above the line last move: no repeat.
        XCTAssertFalse(matches(context(ourWinPercent: 94, previousOurWinPercent: 91, newReadingThisStep: true)).contains(.winningBig))
        // No new reading (an opponent move): nothing.
        XCTAssertFalse(matches(context(ourWinPercent: 92, previousOurWinPercent: 80, newReadingThisStep: false)).contains(.winningBig))
        XCTAssertTrue(matches(context(ourWinPercent: 8, previousOurWinPercent: 30, newReadingThisStep: true)).contains(.losingBig))
    }

    func testBigSwingNeedsTwoReadingsAndTheThreshold() {
        // Default swing threshold 30.
        XCTAssertTrue(matches(context(ourWinPercent: 60, previousOurWinPercent: 20, newReadingThisStep: true)).contains(.bigSwing))
        XCTAssertFalse(matches(context(ourWinPercent: 60, previousOurWinPercent: 45, newReadingThisStep: true)).contains(.bigSwing))
        XCTAssertFalse(matches(context(ourWinPercent: 60, previousOurWinPercent: nil, newReadingThisStep: true)).contains(.bigSwing))
    }

    func testPromotionAndPassedMove() {
        XCTAssertTrue(matches(context(promotionJustPlayed: true)).contains(.promotion))
        // Default passed-move threshold 80: fires only on the crossing move.
        XCTAssertTrue(matches(context(moveNumber: 80)).contains(.passedMove))
        XCTAssertFalse(matches(context(moveNumber: 81)).contains(.passedMove))
        XCTAssertFalse(matches(context(moveNumber: 79)).contains(.passedMove))
    }

    func testMatchesAreSortedByPriority() {
        // Losing-big outranks an early check when both fire.
        let both = matches(context(ply: 6, inCheckNow: true, ourWinPercent: 5, previousOurWinPercent: 40, newReadingThisStep: true))
        XCTAssertEqual(both.first, .losingBig)
        XCTAssertTrue(both.contains(.earlyCheck))
        XCTAssertLessThan(both.firstIndex(of: .losingBig)!, both.firstIndex(of: .earlyCheck)!)
    }

    // MARK: - Spam guard

    func testGuardEnforcesMaxPerGame() {
        var settings = LichessBotChatEventSettings()
        settings.maxMessagesPerGame = 1
        settings.minPliesBetweenMessages = 0
        settings.minSecondsBetweenMessages = 0
        var guardState = LichessBotChatEventGuard()
        XCTAssertTrue(guardState.admits(trigger: .promotion, ply: 10, now: .seconds(10), settings: settings))
        guardState.recordSend(trigger: .promotion, ply: 10, now: .seconds(10))
        XCTAssertFalse(guardState.admits(trigger: .earlyCheck, ply: 20, now: .seconds(40), settings: settings))
    }

    func testGuardFiresEachTriggerOnlyOnce() {
        var settings = LichessBotChatEventSettings()
        settings.maxMessagesPerGame = 10
        settings.minPliesBetweenMessages = 0
        settings.minSecondsBetweenMessages = 0
        var guardState = LichessBotChatEventGuard()
        guardState.recordSend(trigger: .promotion, ply: 10, now: .seconds(10))
        XCTAssertFalse(guardState.admits(trigger: .promotion, ply: 30, now: .seconds(60), settings: settings))
        XCTAssertTrue(guardState.admits(trigger: .earlyCheck, ply: 30, now: .seconds(60), settings: settings))
    }

    func testGuardEnforcesPlyAndTimeSpacing() {
        var settings = LichessBotChatEventSettings()
        settings.maxMessagesPerGame = 10
        settings.minPliesBetweenMessages = 8
        settings.minSecondsBetweenMessages = 20
        var guardState = LichessBotChatEventGuard()
        guardState.recordSend(trigger: .promotion, ply: 10, now: .seconds(100))
        // Too close in plies.
        XCTAssertFalse(guardState.admits(trigger: .earlyCheck, ply: 15, now: .seconds(140), settings: settings))
        // Far enough in plies but too soon in seconds.
        XCTAssertFalse(guardState.admits(trigger: .earlyCheck, ply: 30, now: .seconds(110), settings: settings))
        // Far enough in both.
        XCTAssertTrue(guardState.admits(trigger: .earlyCheck, ply: 30, now: .seconds(140), settings: settings))
    }

    // MARK: - Model defaults and normalization

    func testStarterRulesCoverEveryTriggerDisabledInSpectator() {
        let starter = LichessBotChatEventSettings.starterRules
        XCTAssertEqual(starter.map(\.trigger), LichessBotChatEventTrigger.allCases)
        XCTAssertTrue(starter.allSatisfy { !$0.enabled })
        XCTAssertTrue(starter.allSatisfy { $0.room == .spectator })
        XCTAssertTrue(starter.allSatisfy { !$0.templates.isEmpty })
    }

    func testEventSettingsDefaultsFollowOwnerDecisions() {
        let settings = LichessBotChatEventSettings()
        XCTAssertFalse(settings.enabled)
        XCTAssertEqual(settings.minClockSecondsEitherSide, 240)
    }

    func testNormalizeKeepsEditsAddsMissingDropsUnknown() {
        var edited = LichessBotChatEventRule(trigger: .promotion)
        edited.enabled = true
        edited.chancePercent = 50
        let normalized = LichessBotChatEventSettings.normalized([edited])
        XCTAssertEqual(normalized.map(\.trigger), LichessBotChatEventTrigger.allCases)
        let promotion = normalized.first { $0.trigger == .promotion }
        XCTAssertEqual(promotion?.enabled, true)
        XCTAssertEqual(promotion?.chancePercent, 50)
    }

    // MARK: - Forward-compatible decode

    func testRuleDecodesWithOnlyATrigger() throws {
        let data = Data(#"{"trigger":"promotion"}"#.utf8)
        let rule = try JSONDecoder().decode(LichessBotChatEventRule.self, from: data)
        XCTAssertFalse(rule.enabled)
        XCTAssertEqual(rule.room, .spectator)
        XCTAssertEqual(rule.chancePercent, 100)
        XCTAssertEqual(rule.templates, LichessBotChatEventTrigger.promotion.defaultTemplates)
    }

    func testEventSettingsDecodeWithoutRulesGetsTheStarterSet() throws {
        let data = Data(#"{"enabled":true}"#.utf8)
        let settings = try JSONDecoder().decode(LichessBotChatEventSettings.self, from: data)
        XCTAssertTrue(settings.enabled)
        XCTAssertEqual(settings.minClockSecondsEitherSide, 240)
        XCTAssertEqual(settings.rules.map(\.trigger), LichessBotChatEventTrigger.allCases)
    }

    // MARK: - Templates

    func testEventPlaceholdersAreAcceptedAndOthersRejected() {
        XCTAssertNil(LichessBotChat.templateProblem("Up to {winPct}% on move {move}.", allowed: LichessBotChat.eventPlaceholders))
        XCTAssertNotNil(LichessBotChat.templateProblem("Model {modelID}", allowed: LichessBotChat.eventPlaceholders))
    }

    func testEveryDefaultTemplateIsValid() {
        for trigger in LichessBotChatEventTrigger.allCases {
            for template in trigger.defaultTemplates {
                XCTAssertNil(LichessBotChat.templateProblem(template, allowed: LichessBotChat.eventPlaceholders), "\(trigger): \(template)")
            }
        }
    }

    // MARK: - Room carry-forward (chat-events plan §1)

    /// Store `settings` with its `chat` section replaced by `legacyChat` (an
    /// old save's shape) and load it back through the real overlay +
    /// carry-forward path.
    private func loadWithLegacyChat(_ legacyChat: [String: Any]) throws -> LichessBotSettingsStore.LoadResult {
        let defaults = try makeTemporaryDefaults()
        let data = try JSONEncoder().encode(LichessBotSettings())
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["chat"] = legacyChat
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: LichessBotSettingsStore.defaultsKey)
        return try LichessBotSettingsStore.loadReporting(from: defaults)
    }

    func testOldSpectatorRoomMovesGreetingAndGoodbyeToSpectator() throws {
        let result = try loadWithLegacyChat([
            "greetingEnabled": true, "greetingTemplate": "Hi {modelID}",
            "goodbyeEnabled": true, "goodbyeTemplate": "Bye {opponent}",
            "room": "spectator",
        ])
        let chat = result.settings.chat
        XCTAssertFalse(chat.greetingEnabled)
        XCTAssertTrue(chat.greetingSpectatorEnabled)
        XCTAssertEqual(chat.greetingSpectatorTemplate, "Hi {modelID}")
        XCTAssertFalse(chat.goodbyeEnabled)
        XCTAssertTrue(chat.goodbyeSpectatorEnabled)
        XCTAssertEqual(chat.goodbyeSpectatorTemplate, "Bye {opponent}")
        XCTAssertTrue(result.carriedForward.contains { $0.contains("spectator") })
    }

    func testOldPlayerRoomKeepsGreetingOnThePlayerSide() throws {
        let result = try loadWithLegacyChat([
            "greetingEnabled": true, "greetingTemplate": "Hi {modelID}",
            "goodbyeEnabled": false, "goodbyeTemplate": "Bye",
            "room": "player",
        ])
        let chat = result.settings.chat
        XCTAssertTrue(chat.greetingEnabled)
        XCTAssertEqual(chat.greetingTemplate, "Hi {modelID}")
        XCTAssertFalse(chat.greetingSpectatorEnabled)
        // The default event settings fill in and stay off.
        XCTAssertFalse(chat.events.enabled)
    }
}
