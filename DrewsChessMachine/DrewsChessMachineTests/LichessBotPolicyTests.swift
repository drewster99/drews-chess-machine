import XCTest
@testable import DrewsChessMachine

/// The pure Lichess bot policies: challenge acceptance (plan §7), in-game
/// value-head decisions (§12.4), chat templates (§12.5, E52) and settings
/// validation (§12.1).
final class LichessBotPolicyTests: XCTestCase {

    // MARK: - Challenge policy fixtures

    private func challenge(
        rated: Bool = false,
        speed: LichessBotSpeed = .blitz,
        limit: Int? = 300,
        increment: Int? = 3,
        timeControlType: LichessBotTimeControlType = .clock,
        variant: LichessBotVariantKey = .standard,
        challengerID: String = "alice",
        title: String? = nil,
        rating: Int? = 1500,
        provisional: Bool? = nil,
        direction: LichessBotChallengeDirection? = .incoming,
        initialFen: String? = nil,
        rematchOf: String? = nil
    ) -> LichessBotChallenge {
        LichessBotChallenge(
            id: "c1",
            url: nil,
            status: LichessBotOpenValue(.created),
            challenger: LichessBotChallengeUser(id: challengerID, name: challengerID, rating: rating, title: title, provisional: provisional, online: true, lag: nil),
            destUser: nil,
            variant: LichessBotVariant(key: LichessBotOpenValue(variant), name: nil, short: nil),
            rated: rated,
            speed: LichessBotOpenValue(speed),
            timeControl: LichessBotTimeControl(
                type: LichessBotOpenValue(timeControlType),
                limit: limit.map(LichessBotSeconds.init),
                increment: increment.map(LichessBotSeconds.init),
                daysPerTurn: nil,
                show: nil
            ),
            color: LichessBotOpenValue(.random),
            finalColor: nil,
            perf: nil,
            direction: direction.map(LichessBotOpenValue.init),
            initialFen: initialFen,
            rematchOf: rematchOf
        )
    }

    private func context(
        accepting: Bool = true,
        modelReady: Bool = true,
        activeGames: Int = 0,
        activeByOpponent: [String: Int] = [:],
        gamesToday: Int = 0,
        todayByOpponent: [String: Int] = [:],
        responsesLastMinute: Int = 0
    ) -> LichessBotChallengeContext {
        LichessBotChallengeContext(
            acceptingNewGames: accepting,
            modelReady: modelReady,
            activeGames: activeGames,
            activeGamesByOpponent: activeByOpponent,
            gamesToday: gamesToday,
            gamesTodayByOpponent: todayByOpponent,
            challengeResponsesInLastMinute: responsesLastMinute
        )
    }

    private let defaults = LichessBotChallengeSettings()

    private func decide(_ c: LichessBotChallenge, compat: LichessBotCompat? = nil, settings: LichessBotChallengeSettings? = nil, context ctx: LichessBotChallengeContext? = nil) -> LichessBotChallengeDecision {
        LichessBotChallengePolicy.decide(c, compat: compat, settings: settings ?? defaults, context: ctx ?? context())
    }

    private func reason(_ decision: LichessBotChallengeDecision) -> LichessBotDeclineReason? {
        if case .decline(let reason, _) = decision { return reason }
        return nil
    }

    // MARK: - Challenge policy

    func testDefaultsAcceptACasualBlitzChallenge() {
        XCTAssertEqual(decide(challenge()), .accept)
    }

    func testOwnOutgoingChallengeIsIgnored() {
        guard case .ignore = decide(challenge(direction: .outgoing)) else {
            return XCTFail("an outgoing challenge must never be answered")
        }
    }

    func testOverBudgetIsIgnoredNotDeclined() {
        guard case .ignore = decide(challenge(), context: context(responsesLastMinute: defaults.challengeResponseBudgetPerMinute)) else {
            return XCTFail("over budget must spend no request")
        }
    }

    func testDrainingAndModelNotReadyDeclineLater() {
        XCTAssertEqual(reason(decide(challenge(), context: context(accepting: false))), .later)
        XCTAssertEqual(reason(decide(challenge(), context: context(modelReady: false))), .later)
    }

    func testVariantsAndFromPositionDeclineStandard() {
        XCTAssertEqual(reason(decide(challenge(variant: .chess960))), .standard)
        XCTAssertEqual(reason(decide(challenge(initialFen: "8/8/8/8/8/8/8/K6k w - - 0 1"))), .standard)
        XCTAssertEqual(decide(challenge(initialFen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")), .accept, "a FEN of the start position is the start position")
    }

    func testNonClockAndBotIncompatibleDeclineTimeControl() {
        XCTAssertEqual(reason(decide(challenge(speed: .correspondence, limit: nil, increment: nil, timeControlType: .correspondence))), .timeControl)
        XCTAssertEqual(reason(decide(challenge(speed: .correspondence, limit: nil, increment: nil, timeControlType: .unlimited))), .timeControl)
        XCTAssertEqual(reason(decide(challenge(), compat: LichessBotCompat(bot: false, board: true))), .timeControl)
    }

    func testSpeedsOutsideTheAllowlist() {
        XCTAssertEqual(reason(decide(challenge(speed: .bullet, limit: 60, increment: 0))), .tooFast)
        XCTAssertEqual(reason(decide(challenge(speed: .classical, limit: 1800, increment: 20))), .tooSlow)
    }

    func testClockAndIncrementBounds() {
        XCTAssertEqual(reason(decide(challenge(limit: 120))), .tooFast)
        XCTAssertEqual(reason(decide(challenge(speed: .rapid, limit: 3600))), .tooSlow)
        XCTAssertEqual(reason(decide(challenge(increment: 45))), .timeControl)
    }

    func testRatedAndCasual() {
        XCTAssertEqual(reason(decide(challenge(rated: true))), .casual, "rated is declined by default with the 'casual only' reason")
        var ratedOnly = defaults
        ratedOnly.acceptRated = true
        ratedOnly.acceptCasual = false
        XCTAssertEqual(reason(decide(challenge(rated: false), settings: ratedOnly)), .rated)
        XCTAssertEqual(decide(challenge(rated: true), settings: ratedOnly), .accept)
    }

    func testBotsAndHumans() {
        var noBots = defaults
        noBots.acceptBots = false
        XCTAssertEqual(reason(decide(challenge(title: "BOT"), settings: noBots)), .noBot)
        var onlyBots = defaults
        onlyBots.acceptHumans = false
        XCTAssertEqual(reason(decide(challenge(), settings: onlyBots)), .onlyBot)
    }

    func testOpponentFilters() {
        var settings = defaults
        settings.blockedUserIDs = ["alice"]
        XCTAssertEqual(reason(decide(challenge(), settings: settings)), .generic)
        settings = defaults
        settings.minimumOpponentRating = 1600
        XCTAssertEqual(reason(decide(challenge(rating: 1500), settings: settings)), .generic)
        settings = defaults
        settings.acceptProvisionalOpponents = false
        XCTAssertEqual(reason(decide(challenge(provisional: true), settings: settings)), .generic)
        settings = defaults
        settings.acceptRematches = false
        XCTAssertEqual(reason(decide(challenge(rematchOf: "g0"), settings: settings)), .generic)
    }

    func testCapacityLimits() {
        XCTAssertEqual(reason(decide(challenge(), context: context(activeGames: defaults.maxConcurrentGames))), .later)
        XCTAssertEqual(reason(decide(challenge(), context: context(activeGames: 1, activeByOpponent: ["alice": 1]))), .later)
        XCTAssertEqual(reason(decide(challenge(), context: context(gamesToday: defaults.maxGamesPerDay))), .later)
        XCTAssertEqual(reason(decide(challenge(), context: context(todayByOpponent: ["alice": defaults.maxGamesPerOpponentPerDay]))), .later)
    }

    func testReservedHumanSlotsTurnAwayBots() {
        var settings = defaults
        settings.maxConcurrentGames = 3
        settings.gamesReservedForHumans = 1
        XCTAssertEqual(reason(decide(challenge(title: "BOT"), settings: settings, context: context(activeGames: 2))), .later)
        XCTAssertEqual(decide(challenge(), settings: settings, context: context(activeGames: 2)), .accept, "a human may take the reserved slot")
    }

    func testBotPairDailyStop() {
        var settings = defaults
        settings.maxGamesPerOpponentPerDay = 200
        XCTAssertEqual(reason(decide(challenge(title: "BOT"), settings: settings, context: context(todayByOpponent: ["alice": settings.botPairDailyStop]))), .later)
    }

    // MARK: - Play policy

    private func readings(_ values: [(ply: Int, win: Float, draw: Float, loss: Float)]) -> [LichessBotValueReading] {
        values.map { LichessBotValueReading(ply: $0.ply, win: $0.win, draw: $0.draw, loss: $0.loss) }
    }

    func testResignIsOffByDefault() {
        let hopeless = readings((0..<10).map { (ply: 60 + 2 * $0, win: 0.0, draw: 0.0, loss: 1.0) })
        XCTAssertFalse(LichessBotPlayPolicy.shouldResign(readings: hopeless, settings: LichessBotPlaySettings()))
    }

    func testResignNeedsTheFullStreakPastTheMinimumPly() {
        var settings = LichessBotPlaySettings()
        settings.resignEnabled = true
        settings.resignConsecutiveMoves = 3
        settings.resignMinimumPly = 40
        settings.resignLossProbability = 0.9

        let streak = readings([(40, 0, 0, 0.95), (42, 0, 0, 0.95), (44, 0, 0, 0.95)])
        XCTAssertTrue(LichessBotPlayPolicy.shouldResign(readings: streak, settings: settings))
        XCTAssertFalse(LichessBotPlayPolicy.shouldResign(readings: Array(streak.prefix(2)), settings: settings), "streak too short")
        let broken = readings([(40, 0, 0, 0.95), (42, 0.2, 0, 0.8), (44, 0, 0, 0.95)])
        XCTAssertFalse(LichessBotPlayPolicy.shouldResign(readings: broken, settings: settings), "streak broken")
        let early = readings([(20, 0, 0, 0.99), (22, 0, 0, 0.99), (24, 0, 0, 0.99)])
        XCTAssertFalse(LichessBotPlayPolicy.shouldResign(readings: early, settings: settings), "before the minimum ply")
    }

    func testOfferDraw() {
        var settings = LichessBotPlaySettings()
        XCTAssertFalse(LichessBotPlayPolicy.shouldOfferDraw(readings: readings([(80, 0, 1, 0)]), settings: settings), "off by default")
        settings.offerDrawEnabled = true
        settings.offerDrawConsecutiveMoves = 2
        settings.offerDrawMinimumPly = 60
        settings.offerDrawProbability = 0.8
        XCTAssertTrue(LichessBotPlayPolicy.shouldOfferDraw(readings: readings([(78, 0.1, 0.85, 0.05), (80, 0.1, 0.85, 0.05)]), settings: settings))
        XCTAssertFalse(LichessBotPlayPolicy.shouldOfferDraw(readings: readings([(78, 0.1, 0.85, 0.05), (80, 0.3, 0.6, 0.1)]), settings: settings))
    }

    func testAcceptDrawUsesExpectedScore() {
        var settings = LichessBotPlaySettings()
        let losing = LichessBotValueReading(ply: 50, win: 0.1, draw: 0.3, loss: 0.6)
        XCTAssertFalse(LichessBotPlayPolicy.shouldAcceptDraw(current: losing, settings: settings), "declines all by default")
        settings.acceptDrawEnabled = true
        settings.acceptDrawExpectedScore = 0.45
        XCTAssertTrue(LichessBotPlayPolicy.shouldAcceptDraw(current: losing, settings: settings))
        let winning = LichessBotValueReading(ply: 50, win: 0.6, draw: 0.3, loss: 0.1)
        XCTAssertFalse(LichessBotPlayPolicy.shouldAcceptDraw(current: winning, settings: settings))
    }

    func testTakebacks() {
        var settings = LichessBotPlaySettings()
        XCTAssertFalse(LichessBotPlayPolicy.shouldAcceptTakeback(acceptedSoFar: 0, settings: settings))
        settings.maxTakebacksAcceptedPerGame = 1
        XCTAssertTrue(LichessBotPlayPolicy.shouldAcceptTakeback(acceptedSoFar: 0, settings: settings))
        XCTAssertFalse(LichessBotPlayPolicy.shouldAcceptTakeback(acceptedSoFar: 1, settings: settings))
    }

    // MARK: - Chat

    func testChatExpansionAndValidation() {
        XCTAssertEqual(LichessBotChat.expand("Hi {opponent}, model {modelID}", values: ["opponent": "alice", "modelID": "m1"]), "Hi alice, model m1")
        XCTAssertNil(LichessBotChat.templateProblem(LichessBotChatSettings().greetingTemplate))
        XCTAssertNotNil(LichessBotChat.templateProblem("Hello {nobody}"))
        XCTAssertNotNil(LichessBotChat.templateProblem("   "))
        XCTAssertNotNil(LichessBotChat.templateProblem(String(repeating: "x", count: LichessBotChat.maximumLength - 5) + "{opponent}"), "must fit with long values")
        XCTAssertNil(LichessBotChat.message(from: String(repeating: "y", count: LichessBotChat.maximumLength + 1), values: [:]))
    }

    // MARK: - Settings

    func testDefaultSettingsAreValidAndRoundTrip() throws {
        let settings = LichessBotSettings()
        XCTAssertEqual(settings.validationProblems(), [])
        let decoded = try JSONDecoder().decode(LichessBotSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(decoded, settings)
    }

    func testDefaultPostureMatchesThePlan() {
        let settings = LichessBotSettings()
        XCTAssertFalse(settings.challenge.acceptRated)
        XCTAssertEqual(settings.challenge.allowedSpeeds, [.blitz, .rapid])
        XCTAssertFalse(settings.play.resignEnabled)
        XCTAssertFalse(settings.play.offerDrawEnabled)
        XCTAssertFalse(settings.play.acceptDrawEnabled)
        XCTAssertTrue(settings.play.claimWhenOpponentGone)
        XCTAssertFalse(settings.connection.autoConnectOnLaunch)
        XCTAssertTrue(settings.connection.preventSleepWhileOnline)
        XCTAssertEqual(settings.connection.expectedAccountID, "drewschessmachine")
    }

    func testInvalidSettingsAreReported() {
        var settings = LichessBotSettings()
        settings.challenge.allowedSpeeds = []
        settings.play.temperatureFloor = 0
        settings.connection.eventStreamStallTimeoutSeconds = LichessBotLimits.eventStreamKeepAliveSeconds
        settings.challenge.botPairDailyStop = LichessBotLimits.botPairDailyCap
        XCTAssertEqual(settings.validationProblems().count, 4)
    }
}
