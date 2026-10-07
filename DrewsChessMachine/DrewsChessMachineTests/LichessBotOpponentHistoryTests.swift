import XCTest
@testable import DrewsChessMachine

/// Matchmaking's memory of other bots, derived from the challenge log's rows
/// and the games: declines that rule out a kind of challenge for a while,
/// and each bot's latest contact for the "not contacted recently"
/// preference.
final class LichessBotOpponentHistoryTests: XCTestCase {

    private let now = Date(timeIntervalSince1970: 1_790_000_000)
    private let day: TimeInterval = 86_400

    private var settings: LichessBotMatchmakingSettings {
        var settings = LichessBotMatchmakingSettings()
        settings.noBotDeclineBlockDays = 30
        settings.specificDeclineBlockDays = 14
        settings.ratedCasualDeclineBlockDays = 30
        return settings
    }

    private func player(_ id: String) -> LichessBotChallengeLogRow.Opponent {
        .player(LichessBotChallengeLogRow.Player(id: id, name: id, title: "BOT", rating: 1500))
    }

    private func row(_ id: String, at: Date, direction: LichessBotChallengeLogDirection, state: LichessBotChallengeLogState,
                     limit: Int = 60, increment: Int = 0, rated: Bool = false) -> LichessBotChallengeLogRow {
        LichessBotChallengeLogRow(
            key: .challenge(id: UUID().uuidString), at: at, direction: direction, opponent: player(id),
            terms: LichessBotChallengeLogRow.Terms(rated: rated, limitSeconds: limit, incrementSeconds: increment, daysPerTurn: nil, color: LichessBotOpenValue(raw: "random")),
            initiative: direction == .outgoing ? .senderNotRecorded : .noDecisionRecorded,
            state: .challenge(state, isPending: false), notes: [], anomalies: [], credits: .notRecorded, source: .live)
    }

    private func declined(_ id: String, _ reason: LichessBotDeclineReason, daysAgo: Double = 1, limit: Int = 60, increment: Int = 0, rated: Bool = false) -> LichessBotChallengeLogRow {
        row(id, at: now.addingTimeInterval(-daysAgo * day), direction: .outgoing, state: .declined(.known(reason)), limit: limit, increment: increment, rated: rated)
    }

    private func history(_ rows: [LichessBotChallengeLogRow], games: [LichessBotOpponentHistory.GameStart] = []) -> LichessBotOpponentHistory {
        LichessBotOpponentHistory(challengeRows: rows, gameStarts: games, now: now, settings: settings)
    }

    private func terms(_ limit: Int, _ increment: Int, rated: Bool = false) -> LichessBotChallengeTerms {
        LichessBotChallengeTerms(limitSeconds: limit, incrementSeconds: increment, rated: rated)
    }

    // MARK: - Declines

    func testNoBotBlocksEveryChallengeForItsWindow() throws {
        let recent = history([declined("nobot", .noBot, daysAgo: 29)])
        let block = try XCTUnwrap(recent.blockingDecline(of: "NoBot", terms: terms(1800, 20, rated: true)))
        XCTAssertEqual(block.scope, .everyChallenge)
        XCTAssertEqual(block.until, now.addingTimeInterval(-29 * day + 30 * day))
        XCTAssertNil(history([declined("nobot", .noBot, daysAgo: 31)]).blockingDecline(of: "nobot", terms: terms(60, 0)))
    }

    func testTooFastBlocksThatClockAndFaster() {
        // Declined at 3+0 (180 s estimated).
        let history = history([declined("b", .tooFast, limit: 180, increment: 0)])
        XCTAssertNotNil(history.blockingDecline(of: "b", terms: terms(180, 0)))
        XCTAssertNotNil(history.blockingDecline(of: "b", terms: terms(60, 1)), "1+1 is 100 s estimated")
        XCTAssertNil(history.blockingDecline(of: "b", terms: terms(180, 2)), "3+2 is 260 s estimated")
        XCTAssertNil(history.blockingDecline(of: "other", terms: terms(60, 0)))
    }

    func testTooSlowBlocksThatClockAndSlower() {
        // Declined at 10+0 (600 s estimated).
        let history = history([declined("b", .tooSlow, limit: 600, increment: 0)])
        XCTAssertNotNil(history.blockingDecline(of: "b", terms: terms(900, 10)))
        XCTAssertNotNil(history.blockingDecline(of: "b", terms: terms(600, 0)))
        XCTAssertNil(history.blockingDecline(of: "b", terms: terms(300, 3)), "5+3 is 420 s estimated")
    }

    func testTimeControlBlocksOnlyThatClockAndExpires() {
        let current = history([declined("b", .timeControl, daysAgo: 13, limit: 300, increment: 3)])
        XCTAssertNotNil(current.blockingDecline(of: "b", terms: terms(300, 3, rated: true)))
        XCTAssertNil(current.blockingDecline(of: "b", terms: terms(300, 0)))
        XCTAssertNil(history([declined("b", .timeControl, daysAgo: 15, limit: 300, increment: 3)]).blockingDecline(of: "b", terms: terms(300, 3)))
    }

    func testRatedAndCasualDeclinesBlockTheOtherKind() {
        // "casual" declines a rated challenge: no rated ones.
        let wantsCasual = history([declined("b", .casual, rated: true)])
        XCTAssertNotNil(wantsCasual.blockingDecline(of: "b", terms: terms(60, 0, rated: true)))
        XCTAssertNil(wantsCasual.blockingDecline(of: "b", terms: terms(60, 0, rated: false)))
        // "rated" declines a casual challenge: no casual ones.
        let wantsRated = history([declined("b", .rated, rated: false)])
        XCTAssertNotNil(wantsRated.blockingDecline(of: "b", terms: terms(60, 0, rated: false)))
        XCTAssertNil(wantsRated.blockingDecline(of: "b", terms: terms(60, 0, rated: true)))
    }

    /// A bot that asked for casual is left out of rated challenges for the
    /// rated/casual window (30 days), longer than a clock block (14).
    func testACasualOnlyBotIsLeftOutOfRatedChallengesForItsOwnWindow() {
        let within = history([declined("b", .casual, daysAgo: 29, rated: true)])
        XCTAssertNotNil(within.blockingDecline(of: "b", terms: terms(60, 0, rated: true)))
        XCTAssertNil(within.blockingDecline(of: "b", terms: terms(60, 0, rated: false)))
        XCTAssertNil(history([declined("b", .casual, daysAgo: 31, rated: true)]).blockingDecline(of: "b", terms: terms(60, 0, rated: true)))
        // The clock window is separate and shorter.
        XCTAssertNil(history([declined("b", .timeControl, daysAgo: 29, limit: 60, increment: 0)]).blockingDecline(of: "b", terms: terms(60, 0)))
    }

    func testReasonsThatNameNothingBlockNothing() {
        for reason in [LichessBotDeclineReason.generic, .later, .standard, .variant, .onlyBot] {
            XCTAssertNil(history([declined("b", reason)]).blockingDecline(of: "b", terms: terms(60, 0)), reason.rawValue)
        }
        // An incoming challenge DCM declined is not a bot's refusal.
        let incoming = row("b", at: now.addingTimeInterval(-day), direction: .incoming, state: .declined(.known(.noBot)))
        XCTAssertNil(history([incoming]).blockingDecline(of: "b", terms: terms(60, 0)))
    }

    /// A row whose terms were not recorded can't say what a clock or
    /// rated/casual decline ruled out, so only `noBot` (which needs no
    /// terms) blocks.
    func testUnrecordedTermsBlockOnlyForNoBot() {
        func untermed(_ id: String, _ reason: LichessBotDeclineReason) -> LichessBotChallengeLogRow {
            LichessBotChallengeLogRow(
                key: .challenge(id: UUID().uuidString), at: now.addingTimeInterval(-day), direction: .outgoing, opponent: player(id),
                terms: nil, initiative: .senderNotRecorded, state: .challenge(.declined(.known(reason)), isPending: false),
                notes: [], anomalies: [], credits: .notRecorded, source: .live)
        }
        for reason in [LichessBotDeclineReason.rated, .casual, .tooFast, .tooSlow, .timeControl] {
            let history = history([untermed("b", reason)])
            XCTAssertNil(history.blockingDecline(of: "b", terms: terms(60, 0, rated: false)), reason.rawValue)
            XCTAssertNil(history.blockingDecline(of: "b", terms: terms(60, 0, rated: true)), reason.rawValue)
        }
        XCTAssertNotNil(history([untermed("b", .noBot)]).blockingDecline(of: "b", terms: terms(60, 0)))
    }

    func testAZeroDayWindowRecordsNoBlock() {
        var settings = settings
        settings.noBotDeclineBlockDays = 0
        settings.specificDeclineBlockDays = 0
        settings.ratedCasualDeclineBlockDays = 0
        let history = LichessBotOpponentHistory(challengeRows: [declined("b", .noBot, daysAgo: 0)], gameStarts: [], now: now, settings: settings)
        XCTAssertNil(history.blockingDecline(of: "b", terms: terms(60, 0)))
    }

    // MARK: - Contacts

    func testLatestContactIsTheNewestChallengeEitherWayOrGame() throws {
        let rows = [
            row("b", at: now.addingTimeInterval(-3 * day), direction: .outgoing, state: .accepted(gameStarted: true)),
            row("b", at: now.addingTimeInterval(-2 * day), direction: .incoming, state: .open),
        ]
        let games = [LichessBotOpponentHistory.GameStart(opponentID: "B", at: now.addingTimeInterval(-1 * day))]
        let contact = try XCTUnwrap(history(rows, games: games).contact("b"))
        XCTAssertEqual(contact.lastChallengeSent, now.addingTimeInterval(-3 * day))
        XCTAssertEqual(contact.lastChallengeReceived, now.addingTimeInterval(-2 * day))
        XCTAssertEqual(contact.lastGame, now.addingTimeInterval(-1 * day))
        XCTAssertEqual(contact.latest, now.addingTimeInterval(-1 * day))
        XCTAssertNil(history(rows).contact("never"))
    }

    // MARK: - Matchmaking

    private func bot(_ id: String) -> LichessBotUserSummary {
        LichessBotUserSummary(id: id, username: id, title: "BOT", perfs: ["blitz": LichessBotPerfRating(games: 50, rating: 1500, rd: 60, prog: 0, prov: nil)],
                              online: true, disabled: nil, tosViolation: nil, createdAt: nil, seenAt: nil, profile: nil)
    }

    private func context(_ history: LichessBotOpponentHistory) -> LichessBotMatchmaking.CandidateContext {
        LichessBotMatchmaking.CandidateContext(ourAccountID: "dcm", blockedUserIDs: [], engagedUserIDs: [], notes: LichessBotPlayerNotes(),
                                               gamesTodayByOpponent: [:], maxGamesPerOpponentPerDay: 5, opponentHistory: history, now: now)
    }

    func testDeclinesExcludeCandidates() {
        let bounds = LichessBotMatchmaking.RatingBounds(minimum: 1000, maximum: 2000, basis: .absolute)
        let history = history([declined("nobot", .noBot), declined("fastnot", .tooFast, limit: 300, increment: 0)])
        XCTAssertEqual(LichessBotMatchmaking.exclusion(of: bot("nobot"), terms: terms(300, 3), bounds: bounds, context: context(history)), .refusesBots)
        XCTAssertEqual(LichessBotMatchmaking.exclusion(of: bot("fastnot"), terms: terms(300, 0), bounds: bounds, context: context(history)), .declinedThisKind)
        XCTAssertNil(LichessBotMatchmaking.exclusion(of: bot("fastnot"), terms: terms(300, 3), bounds: bounds, context: context(history)))
    }

    func testThePickPrefersBotsNotContactedRecently() {
        var settings = LichessBotMatchmakingSettings()
        settings.timeControls = [.blitz5plus3]
        settings.rated = false
        settings.minimumRatingWithoutOwnRating = 1000
        settings.maximumRatingWithoutOwnRating = 2000
        settings.preferNotRecentlyContacted = true
        settings.recentContactHours = 24
        let recent = history([], games: [LichessBotOpponentHistory.GameStart(opponentID: "recent", at: now.addingTimeInterval(-3600))])
        var generator = LichessBotSeededGenerator(seed: 7)
        for _ in 0..<20 {
            guard case .picked(let pick) = LichessBotMatchmaking.pick(from: [bot("recent"), bot("fresh")], settings: settings, ourPerfs: nil, context: context(recent), using: &generator) else {
                return XCTFail("expected a pick")
            }
            XCTAssertEqual(pick.bot.id, "fresh")
            XCTAssertEqual(pick.recency, .notContactedRecently(count: 1))
        }
        // Everyone contacted inside the window: the one contacted longest ago.
        let both = history([], games: [
            LichessBotOpponentHistory.GameStart(opponentID: "recent", at: now.addingTimeInterval(-600)),
            LichessBotOpponentHistory.GameStart(opponentID: "older", at: now.addingTimeInterval(-7200)),
        ])
        guard case .picked(let pick) = LichessBotMatchmaking.pick(from: [bot("recent"), bot("older")], settings: settings, ourPerfs: nil, context: context(both), using: &generator) else {
            return XCTFail("expected a pick")
        }
        XCTAssertEqual(pick.bot.id, "older")
        XCTAssertEqual(pick.recency, .contactedLongestAgo(lastContact: now.addingTimeInterval(-7200)))
        // Off: no narrowing, and no recency recorded.
        settings.preferNotRecentlyContacted = false
        guard case .picked(let unpreferred) = LichessBotMatchmaking.pick(from: [bot("recent")], settings: settings, ourPerfs: nil, context: context(recent), using: &generator) else {
            return XCTFail("expected a pick")
        }
        XCTAssertNil(unpreferred.recency)
    }
}
