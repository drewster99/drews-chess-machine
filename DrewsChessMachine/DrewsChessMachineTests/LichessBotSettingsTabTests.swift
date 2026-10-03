import XCTest
@testable import DrewsChessMachine

/// The Settings tabs' ownership of settings fields — which tab a validation
/// problem is shown on — the tab Settings opens on, and the Account tab's
/// status line.
///
/// Ownership matters because the screen marks tabs from these rules alone:
/// a field owned by the wrong tab sends the operator to a tab where the
/// problem can't be fixed, and a field owned by no tab leaves a problem
/// with nowhere to look.
final class LichessBotSettingsTabTests: XCTestCase {

    private func makeBaseline() -> LichessBotSettings {
        let settings = LichessBotSettings.testBaseline()
        XCTAssertEqual(settings.validationProblems(), [], "the baseline must be valid, as the settings in force always are")
        return settings
    }

    func testValidDraftMarksNoTab() {
        let baseline = makeBaseline()
        var draft = baseline
        draft.challenge.maxGamesPerDay = 7
        draft.play.minimumThinkMilliseconds = 5
        XCTAssertEqual(draft.validationProblems(), [])
        XCTAssertEqual(LichessBotSettingsTab.tabsWithProblems(in: draft, comparedWith: baseline), [])
    }

    func testEachProblemIsShownOnTheTabThatEditsTheField() {
        let baseline = makeBaseline()
        let cases: [(String, LichessBotSettingsTab, (inout LichessBotSettings) -> Void)] = [
            ("challenge speeds", .games, { $0.challenge.allowedSpeeds = [] }),
            ("matchmaking time controls", .games, { $0.matchmaking.timeControls = [] }),
            ("temperature decay", .play, { $0.play.temperatureDecayPerPly = -1 }),
            ("live-trainer refresh", .play, { $0.model.liveTrainerRefreshIntervalSeconds = 1 }),
            ("greeting", .chat, { $0.chat.greetingEnabled = true; $0.chat.greetingTemplate = "" }),
            ("reconnect delay", .connection, { $0.connection.reconnectInitialSeconds = 0 }),
            ("finished-game retention", .connection, { $0.display.finishedGameRetentionMinutes = -1 }),
            ("account id", .account, { $0.connection.expectedAccountID = "" }),
        ]
        for (name, tab, breakField) in cases {
            var draft = baseline
            breakField(&draft)
            XCTAssertFalse(draft.validationProblems().isEmpty, "\(name): the edit must be invalid for this case to mean anything")
            XCTAssertEqual(LichessBotSettingsTab.tabsWithProblems(in: draft, comparedWith: baseline), [tab], name)
        }
    }

    func testProblemsOnSeveralTabsAreListedInTabOrder() {
        let baseline = makeBaseline()
        var draft = baseline
        draft.connection.expectedAccountID = ""
        draft.chat.greetingEnabled = true
        draft.chat.greetingTemplate = "  "
        draft.challenge.acceptBots = false
        draft.challenge.acceptHumans = false
        XCTAssertEqual(LichessBotSettingsTab.tabsWithProblems(in: draft, comparedWith: baseline), [.games, .chat, .account])
    }

    /// Every field belongs to exactly one tab: laying every tab's fields from
    /// a draft over a baseline gives back the draft, and the Connection tab
    /// leaves the account id alone.
    func testEveryFieldBelongsToOneTab() {
        let baseline = makeBaseline()
        var draft = baseline
        draft.challenge.maxGamesPerDay = 7
        draft.matchmaking.maxChallengesPerHour = 3
        draft.play.minimumThinkMilliseconds = 5
        draft.model.liveTrainerRefreshIntervalSeconds = 300
        draft.chat.goodbyeEnabled = !baseline.chat.goodbyeEnabled
        draft.alerts.botChallengeSoundName = "Ping"
        draft.connection.reconnectCapSeconds = 90
        draft.connection.expectedAccountID = "someoneelse"
        draft.display.finishedGameRetentionMinutes = 3
        var rebuilt = baseline
        for tab in LichessBotSettingsTab.allCases {
            rebuilt = tab.overlaying(fieldsFrom: draft, onto: rebuilt)
        }
        XCTAssertEqual(rebuilt, draft)
        let connectionOnly = LichessBotSettingsTab.connection.overlaying(fieldsFrom: draft, onto: baseline)
        XCTAssertEqual(connectionOnly.connection.expectedAccountID, baseline.connection.expectedAccountID)
        XCTAssertEqual(connectionOnly.connection.reconnectCapSeconds, 90)
        let accountOnly = LichessBotSettingsTab.account.overlaying(fieldsFrom: draft, onto: baseline)
        XCTAssertEqual(accountOnly.connection.expectedAccountID, "someoneelse")
        XCTAssertEqual(accountOnly.connection.reconnectCapSeconds, baseline.connection.reconnectCapSeconds)
    }

    /// The settings groups `LichessBotSettingsTab.overlaying` assigns. A new
    /// group fails this until it is given a tab there.
    func testSettingsGroupsAreTheOnesTheTabsAssign() {
        let groups = Set(Mirror(reflecting: LichessBotSettings()).children.compactMap(\.label))
        XCTAssertEqual(groups, ["challenge", "play", "chat", "model", "connection", "display", "matchmaking", "alerts"])
    }

    func testOpensOnAccountOnlyWhenNoTokenIsSaved() {
        let info = LichessBotTokenInfo(userId: "drewschessmachine", scopes: "bot:play", expires: nil)
        XCTAssertEqual(LichessBotSettingsTab.opening(tokenState: LichessBotController.TokenState.none, remembered: .chat), .account)
        XCTAssertEqual(LichessBotSettingsTab.opening(tokenState: .unknown, remembered: .chat), .chat)
        XCTAssertEqual(LichessBotSettingsTab.opening(tokenState: .checking, remembered: .chat), .chat)
        XCTAssertEqual(LichessBotSettingsTab.opening(tokenState: .saved(info), remembered: .play), .play)
        XCTAssertEqual(LichessBotSettingsTab.opening(tokenState: .error("This token has expired"), remembered: .games), .games)
    }

    // MARK: - Account status line

    private func account(title: String?, gamesPlayed: Int?) -> LichessBotAccount {
        LichessBotAccount(
            id: "drewschessmachine",
            username: "DrewsChessMachine",
            title: title,
            count: gamesPlayed.map { LichessBotAccountCount(all: $0, rated: nil, win: nil, loss: nil, draw: nil, playing: nil) },
            perfs: nil,
            disabled: nil,
            tosViolation: nil,
            createdAt: nil,
            seenAt: nil
        )
    }

    private let neverExpiring = LichessBotTokenInfo(userId: "drewschessmachine", scopes: "bot:play,challenge:write", expires: nil)

    func testStatusWithoutAVerifiedTokenNamesOnlyTheConfiguredAccount() {
        let stale = account(title: "BOT", gamesPlayed: 12)
        let none = LichessBotAccountStatus(tokenState: LichessBotController.TokenState.none, account: nil, canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(none.identity, "drewschessmachine")
        XCTAssertEqual(none.token, "no token")
        XCTAssertEqual(none.tokenTone, .needsSetup)

        let checking = LichessBotAccountStatus(tokenState: .checking, account: stale, canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(checking.identity, "drewschessmachine", "an account fetched for an earlier token is not shown while a check is pending")
        XCTAssertEqual(checking.token, "checking…")
        XCTAssertEqual(checking.tokenTone, .neutral)

        let unknown = LichessBotAccountStatus(tokenState: .unknown, account: nil, canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(unknown.token, "token not checked yet")
        XCTAssertEqual(unknown.tokenTone, .neutral)

        let failed = LichessBotAccountStatus(tokenState: .error("This token has expired"), account: stale, canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(failed.identity, "drewschessmachine")
        XCTAssertEqual(failed.token, "token problem: This token has expired")
        XCTAssertEqual(failed.tokenTone, .problem)
    }

    func testStatusWithAVerifiedTokenStatesTheBotStanding() {
        let bot = LichessBotAccountStatus(tokenState: .saved(neverExpiring), account: account(title: "BOT", gamesPlayed: 500), canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(bot.identity, "DrewsChessMachine · BOT account")
        XCTAssertEqual(bot.token, "token OK · expires never")
        XCTAssertEqual(bot.tokenTone, .good)

        let played = LichessBotAccountStatus(tokenState: .saved(neverExpiring), account: account(title: nil, gamesPlayed: 92), canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(played.identity, "DrewsChessMachine · not a BOT account · upgrade no longer possible: 92 games played")

        let playedOnce = LichessBotAccountStatus(tokenState: .saved(neverExpiring), account: account(title: nil, gamesPlayed: 1), canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(playedOnce.identity, "DrewsChessMachine · not a BOT account · upgrade no longer possible: 1 game played")

        let fresh = LichessBotAccountStatus(tokenState: .saved(neverExpiring), account: account(title: nil, gamesPlayed: 0), canUpgradeToBot: true, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(fresh.identity, "DrewsChessMachine · not a BOT account yet · upgrade below")

        let unreported = LichessBotAccountStatus(tokenState: .saved(neverExpiring), account: account(title: nil, gamesPlayed: nil), canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(unreported.identity, "DrewsChessMachine · not a BOT account · games played not reported, so no upgrade")

        let missing = LichessBotAccountStatus(tokenState: .saved(neverExpiring), account: nil, canUpgradeToBot: false, configuredAccountID: "drewschessmachine")
        XCTAssertEqual(missing.identity, "drewschessmachine · account details not loaded")
    }
}
