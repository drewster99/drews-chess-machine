import XCTest
@testable import DrewsChessMachine

/// The follow-lineage settings (follow-lineage plan §3.1): what today's
/// saves load as, the new fields' round trip and validation, the poll
/// cadence as the one source of the check interval's floor, and the settings
/// store reading a saved struct-typed optional.
final class LichessBotFollowLineageSettingsTests: XCTestCase {

    /// The operator's saved settings blob as the build before these fields
    /// wrote it (read from this Mac's preferences on 2026-10-06, with
    /// `matchmaking.fallBackToCasual`, which today's build writes): every
    /// section, the file source's fields, no follow-lineage keys.
    private static let todaysSave = #"""
    {"alerts":{"botChallengeSoundName":"Bottle","humanChallengeSoundName":"Blow"},"challenge":{"acceptBots":true,"acceptCasual":true,"acceptHumans":true,"acceptProvisionalOpponents":true,"acceptRated":false,"acceptRematches":true,"allowedSpeeds":["bullet","rapid","classical","ultraBullet","blitz"],"blockedUserIDs":[],"challengeResponseBudgetPerMinute":10,"gamesReservedForHumans":2,"maxConcurrentGames":12,"maxGamesPerDay":2000,"maxGamesPerOpponentPerDay":5,"maxSimultaneousGamesPerOpponent":1,"maximumIncrementSeconds":30,"maximumInitialSeconds":1800,"maximumOpponentRating":2500,"minimumIncrementSeconds":0,"minimumInitialSeconds":15,"minimumOpponentRating":0,"outgoingChallengeTimeoutSeconds":999},"chat":{"goodbyeEnabled":false,"goodbyeTemplate":"Thanks for the game, {opponent}!","greetingEnabled":true,"greetingTemplate":"Hi, I'm DrewsChessMachine ({modelID}), a from-scratch neural net, no search. Type !help for commands.","room":"player"},"connection":{"accountRefreshMinimumIntervalSeconds":300,"autoConnectOnLaunch":false,"breakerWindowMinutes":10,"eventStreamStallTimeoutSeconds":25,"expectedAccountID":"drewschessmachine","exportMinimumSpacingSeconds":10,"gameStreamResyncSeconds":60,"lostOnTimeBreakerCount":3,"lostOnTimeBreakerWindowGames":10,"lowClockThresholdMilliseconds":10000,"movePostFailureBreakerCount":5,"postRateLimitDrainMinutes":15,"preventSleepWhileOnline":true,"rateLimitBreakerWindowMinutes":60,"reconnectCapSeconds":60,"reconnectInitialSeconds":2,"reconnectStormBreakerCount":10},"display":{"finishedGameRetentionMinutes":99},"matchmaking":{"declineCooldownHours":4,"enabled":true,"fallBackToCasual":true,"fillMode":"everyFreeSlot","maxChallengesPerHour":20,"maximumRatingOffset":300,"maximumRatingWithoutOwnRating":2200,"minimumRatingOffset":-300,"minimumRatingWithoutOwnRating":0,"preferFavorites":false,"rated":true,"timeControls":["1+0","30+0","1+1","2+1","10+0","15+10","30+20","¼+0","10+5"]},"model":{"filePath":"/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20260713-v5cont-resume-replay-step270000.safetensors","liveTrainerRefreshIntervalSeconds":120,"midGameRefresh":false,"source":"file"},"play":{"acceptDrawEnabled":false,"acceptDrawExpectedScore":0.45,"claimWhenOpponentGone":true,"maxTakebacksAcceptedPerGame":0,"minimumThinkMilliseconds":0,"offerDrawConsecutiveMoves":8,"offerDrawEnabled":false,"offerDrawMinimumPly":60,"offerDrawProbability":0.85,"resignConsecutiveMoves":6,"resignEnabled":false,"resignLossProbability":0.95,"resignMinimumPly":40,"temperatureDecayPerPly":0.02,"temperatureFloor":0.01,"temperatureStart":0.11}}
    """#

    private let lineage = LichessBotFollowedLineage(lineageRunID: "783BF744-5FCB-4869-BBE8-7126EDD0C72B", anchorSegmentID: "0E4B3E7C-6A0B-4F7E-9A57-2E9D1F3C5B21")

    func testTodaysSaveWithoutTheNewFieldsLoads() throws {
        let defaults = try makeTemporaryDefaults()
        defaults.set(Data(Self.todaysSave.utf8), forKey: LichessBotSettingsStore.defaultsKey)
        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertNil(result.settings.model.followedLineage)
        XCTAssertEqual(result.settings.model.lineageCheckIntervalSeconds, LichessBotModelSettings().lineageCheckIntervalSeconds)
        XCTAssertEqual(result.filledFromDefaults, ["model.lineageCheckIntervalSeconds"])
        XCTAssertEqual(result.ignoredSavedKeys, [])
        XCTAssertEqual(result.settings.model.source, .file)
        XCTAssertEqual(result.settings.challenge.maxConcurrentGames, 12, "everything saved is kept")
    }

    func testFollowLineageRoundTrips() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings.testBaseline()
        settings.model.followedLineage = lineage
        settings.model.lineageCheckIntervalSeconds = 90
        try LichessBotSettingsStore.save(settings, to: defaults)
        let loaded = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(loaded.settings, settings)
        XCTAssertEqual(loaded.filledFromDefaults, [])
        XCTAssertEqual(loaded.ignoredSavedKeys, [])
    }

    func testCheckIntervalBelowTheMinimumIsInvalid() {
        let baseline = LichessBotSettings.testBaseline()
        var draft = baseline
        draft.model.lineageCheckIntervalSeconds = LichessBotLimits.minimumLineageCheckSeconds - 1
        XCTAssertFalse(draft.validationProblems().isEmpty)
        XCTAssertEqual(LichessBotSettingsTab.tabsWithProblems(in: draft, comparedWith: baseline), [.play])
        draft.model.lineageCheckIntervalSeconds = LichessBotLimits.minimumLineageCheckSeconds
        XCTAssertEqual(draft.validationProblems(), [])
    }

    func testFollowLineageWithoutALineageIsInvalid() {
        let baseline = LichessBotSettings.testBaseline()
        var draft = baseline
        draft.model.source = .followLineage
        draft.model.followedLineage = nil
        XCTAssertTrue(draft.validationProblems().contains("Choose a lineage to follow"), "\(draft.validationProblems())")
        XCTAssertEqual(LichessBotSettingsTab.tabsWithProblems(in: draft, comparedWith: baseline), [.play])
        draft.model.followedLineage = LichessBotFollowedLineage(lineageRunID: "", anchorSegmentID: lineage.anchorSegmentID)
        XCTAssertFalse(draft.validationProblems().isEmpty, "a lineage with no run ID is no lineage")
        draft.model.followedLineage = lineage
        XCTAssertEqual(draft.validationProblems(), [])
    }

    func testMinimumCheckIntervalIsThePollInterval() {
        XCTAssertEqual(LichessBotLimits.minimumLineageCheckSeconds, LichessBotLimits.modelRefreshPollSeconds)
    }

    /// Regression (follow-lineage plan §3.1): a saved struct-typed optional
    /// whose default is nil made the whole saved settings "unreadable".
    func testSavedFollowedLineageLoadsThroughTheStore() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings.testBaseline()
        settings.model.followedLineage = lineage
        try LichessBotSettingsStore.save(settings, to: defaults)
        let loaded = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(loaded.settings, settings)
        XCTAssertEqual(loaded.ignoredSavedKeys, [])
    }

    /// The fix is no catch-all: a missing key at another path still fails.
    func testAnUnrelatedMissingKeyIsStillUnreadable() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings.testBaseline()
        settings.model.followedLineage = lineage
        let data = try JSONEncoder().encode(settings)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var model = try XCTUnwrap(object["model"] as? [String: Any])
        // A followed lineage missing one of its required fields.
        model["followedLineage"] = ["lineageRunID": lineage.lineageRunID]
        object["model"] = model
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: LichessBotSettingsStore.defaultsKey)
        XCTAssertThrowsError(try LichessBotSettingsStore.loadReporting(from: defaults)) { error in
            guard case .unreadable? = error as? LichessBotSettingsStoreError else {
                return XCTFail("expected unreadable, got \(error)")
            }
        }
    }
}
