import XCTest
@testable import DrewsChessMachine

/// DCM at its own daily bot-game limit (plan
/// `LICHESS_BOT_OWN_BOT_LIMIT_PLAN.md`): telling Lichess's 429 apart from a
/// request rate, keeping the gate open, and pausing only bot games until
/// Lichess's time.
final class LichessBotOwnBotGameLimitTests: XCTestCase {

    /// 2026-10-14 06:40:22 UTC, the first refusal of the incident.
    private let respondedAt = Date(timeIntervalSince1970: 1_791_960_022)
    private let hour: TimeInterval = 3600

    private var utc: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        return calendar
    }

    private static let ownMessage = "You played 100 games against other bots today, please wait before challenging another bot."
    private static let ownLimitBody = Data(#"{"error":"You played 100 games against other bots today, please wait before challenging another bot.","ratelimit":{"key":"bot.vsBot.day","seconds":6311}}"#.utf8)
    private static let opponentLimitBody = Data(#"{"error":"Jibbby played 100 games against other bots today, please wait until 2026-10-14T10:11:36.542Z to challenge them.","ratelimit":{"key":"bot.vsBot.day","seconds":12434}}"#.utf8)
    private static let requestRateBody = Data(#"{"error":"Too many requests. Try again later."}"#.utf8)

    private func parse(_ status: Int, _ body: Data) -> LichessBotOwnBotGameLimitRefusal? {
        LichessBotOwnBotGameLimitRefusal.parse(status: status, body: body, respondedAt: respondedAt)
    }

    // MARK: - Classification

    func testLilaOwnLimitAnswerIsRecognizedWithItsSeconds() {
        XCTAssertEqual(parse(429, Self.ownLimitBody), LichessBotOwnBotGameLimitRefusal(message: Self.ownMessage, secondsUntilClear: 6311, respondedAt: respondedAt))
    }

    func testTheKeyAloneOrTheMessageAloneIsEnough() {
        let keyOnly = Data(#"{"error":"reworded","ratelimit":{"key":"bot.vsBot.day","seconds":60}}"#.utf8)
        XCTAssertEqual(parse(429, keyOnly)?.secondsUntilClear, 60)
        let messageOnly = Data(#"{"error":"You played 120 games against other bots today, please wait before challenging another bot."}"#.utf8)
        let refusal = parse(429, messageOnly)
        XCTAssertNotNil(refusal)
        XCTAssertNil(refusal?.secondsUntilClear, "seconds are taken only with lila's key")
    }

    func testAnotherKeysSecondsAreNotTaken() {
        let otherKey = Data(#"{"error":"You played 100 games against other bots today, please wait before challenging another bot.","ratelimit":{"key":"challenge.user","seconds":60}}"#.utf8)
        XCTAssertNil(parse(429, otherKey)?.secondsUntilClear)
    }

    func testPlainRequestRate429IsNotTheOwnLimit() {
        XCTAssertNil(parse(429, Self.requestRateBody))
        XCTAssertNil(parse(429, Data()))
        XCTAssertNil(parse(429, Data("<html>Too Many Requests</html>".utf8)), "a proxy's page is no answer from lila")
    }

    func testTheOpponentAtItsLimitIsNotTheOwnLimit() {
        XCTAssertNil(parse(400, Self.opponentLimitBody), "the opponent's limit is a 400")
        XCTAssertNil(parse(400, Self.ownLimitBody), "only a 429 is DCM's own limit")
        XCTAssertNotNil(LichessBotBotLimitRefusal.parse("Jibbby played 100 games against other bots today, please wait until 2026-10-14T10:11:36.542Z to challenge them."), "the opponent's refusal keeps its own handling")
    }

    func testOwnLimitMessageIsAnchoredAtYou() {
        XCTAssertTrue(LichessBotOwnBotGameLimitRefusal.isOwnLimitMessage(Self.ownMessage))
        XCTAssertFalse(LichessBotOwnBotGameLimitRefusal.isOwnLimitMessage("Jibbby played 100 games against other bots today, please wait until 2026-10-14T10:11:36.542Z to challenge them."))
    }

    // MARK: - The end time

    func testEndIsLichessSecondsPlusTheRoundingMargin() throws {
        let refusal = try XCTUnwrap(parse(429, Self.ownLimitBody))
        let limit = LichessBotOwnBotGameLimit(refusal)
        XCTAssertEqual(limit.endSource, .lichess)
        XCTAssertEqual(limit.until, respondedAt.addingTimeInterval(6311 + LichessBotOwnBotGameLimit.roundingMargin))
    }

    func testWithoutSecondsDCMAsksAgainAfterTheRecheckInterval() {
        let limit = LichessBotOwnBotGameLimit(LichessBotOwnBotGameLimitRefusal(message: Self.ownMessage, secondsUntilClear: nil, respondedAt: respondedAt))
        XCTAssertEqual(limit.endSource, .recheck)
        XCTAssertEqual(limit.until, respondedAt.addingTimeInterval(LichessBotOwnBotGameLimit.recheckInterval))
    }

    // MARK: - Gate

    private func gateAndLog() -> (LichessBotRequestGate, SyncBox<[LichessBotGateEvent]>) {
        let events = SyncBox<[LichessBotGateEvent]>([])
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600), wallClock: { [respondedAt] in respondedAt }) { event in
            events.modify { $0.append(event) }
        }
        return (gate, events)
    }

    func testOwnLimit429NeitherCoolsDownNorCountsTowardTheBreaker() async throws {
        let (gate, events) = gateAndLog()
        let limited = try lichessBotTestResponse(status: 429)
        let refusal = LichessBotOwnBotGameLimitRefusal(message: Self.ownMessage, secondsUntilClear: 6311, respondedAt: respondedAt)
        for attempt in 0..<3 {
            let result = try await gate.perform(priority: .challengeResponse, label: "challenge \(attempt)", causeOf429: { _, _ in .ownBotGameLimit(refusal) }) {
                (value: attempt, response: limited)
            }
            XCTAssertEqual(result.response.statusCode, 429, "returned like any other refusal")
        }
        let phase = await gate.snapshot().phase
        XCTAssertEqual(phase, .open)
        let reported = events.value.filter { if case .ownBotGameLimitRefused = $0 { return true } else { return false } }
        XCTAssertEqual(reported.count, 3)
        XCTAssertFalse(events.value.contains { if case .rateLimited = $0 { return true } else { return false } })
        XCTAssertFalse(events.value.contains { if case .breakerTripped = $0 { return true } else { return false } })
    }

    func testTheClassifierSeesTheGatesWallClock() async throws {
        let (gate, _) = gateAndLog()
        let seen = SyncBox<Date?>(nil)
        _ = try await gate.perform(priority: .challengeResponse, label: "challenge", causeOf429: { _, at in
            seen.value = at
            return .ownBotGameLimit(LichessBotOwnBotGameLimitRefusal(message: nil, secondsUntilClear: nil, respondedAt: at))
        }) {
            (value: 0, response: try lichessBotTestResponse(status: 429))
        }
        XCTAssertEqual(seen.value, respondedAt)
    }

    func testARequestRate429StillCoolsDown() async throws {
        let (gate, events) = gateAndLog()
        do {
            _ = try await gate.perform(priority: .challengeResponse, label: "challenge", causeOf429: { _, _ in .requestRate }) {
                (value: 0, response: try lichessBotTestResponse(status: 429))
            }
            XCTFail("a request-rate 429 must throw")
        } catch LichessBotGateError.rateLimited {
            // expected
        }
        XCTAssertTrue(events.value.contains { if case .rateLimited = $0 { return true } else { return false } })
    }

    // MARK: - API client

    private func makeClient(status: Int, body: Data) throws -> (LichessBotAPIClient, LichessBotRequestGate, SyncBox<[LichessBotGateEvent]>) {
        let (gate, events) = gateAndLog()
        let transport = LichessBotScriptedTransport { _ in (status, body, [:]) }
        let client = LichessBotAPIClient(baseURL: try LichessBotAPIClient.lichessBaseURL(), token: "lip_TESTTOKEN123", transport: transport, gate: gate)
        return (client, gate, events)
    }

    private let request = LichessBotOutgoingChallenge(rated: true, clockLimitSeconds: 1800, clockIncrementSeconds: 20, color: .random)

    func testClientThrowsTheOwnLimitWithTheSameRefusalTheGateReported() async throws {
        let (client, gate, events) = try makeClient(status: 429, body: Self.ownLimitBody)
        do {
            _ = try await client.challenge(username: "DarkOnWeakBot", request: request)
            XCTFail("the refusal must throw")
        } catch LichessBotAPIError.ownBotGameLimit(let refusal) {
            XCTAssertEqual(refusal.secondsUntilClear, 6311)
            XCTAssertEqual(refusal.respondedAt, respondedAt)
            let reported = events.value.compactMap { event -> LichessBotOwnBotGameLimitRefusal? in
                if case .ownBotGameLimitRefused(let reported, _, _) = event { return reported }
                return nil
            }
            XCTAssertEqual(reported, [refusal])
        }
        let phase = await gate.snapshot().phase
        XCTAssertEqual(phase, .open)
    }

    func testAcceptingWhileAtTheLimitIsTheSameRefusal() async throws {
        let (client, _, _) = try makeClient(status: 429, body: Self.ownLimitBody)
        do {
            try await client.acceptChallenge(id: "c1")
            XCTFail("the refusal must throw")
        } catch LichessBotAPIError.ownBotGameLimit {
            // expected: lila's `acceptLimitError` answers the same `Limited`
        }
    }

    func testClientStillTreatsAPlain429AsARequestRate() async throws {
        let (client, _, events) = try makeClient(status: 429, body: Self.requestRateBody)
        do {
            _ = try await client.challenge(username: "DarkOnWeakBot", request: request)
            XCTFail("a 429 must throw")
        } catch LichessBotGateError.rateLimited {
            // expected
        }
        // Read from the gate's events, not its phase: the virtual clock's
        // sleep advances at once, so the cooldown may already have ended.
        let cooledDown = events.value.contains { event in
            if case .rateLimited = event { return true }
            return false
        }
        let reportedOwnLimit = events.value.contains { event in
            if case .ownBotGameLimitRefused = event { return true }
            return false
        }
        XCTAssertTrue(cooledDown, "a request-rate 429 cools the gate down")
        XCTAssertFalse(reportedOwnLimit, "a plain 429 is not DCM's own bot-game limit")
    }

    func testChallengeRefusalOfTheOwnLimitIsItsOwnKindAndCharged() throws {
        let refusal = LichessBotOwnBotGameLimitRefusal(message: Self.ownMessage, secondsUntilClear: 6311, respondedAt: respondedAt)
        let classified = try XCTUnwrap(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.ownBotGameLimit(refusal)))
        XCTAssertEqual(classified, LichessBotChallengeRefusal(kind: .ownBotDailyGameLimit, httpStatus: 429, text: Self.ownMessage))
        XCTAssertTrue(classified.spendsCredits, "lila checks the limit inside the charged challenge block")
        XCTAssertEqual(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.http(status: 400, message: "Jibbby played 100 games against other bots today, please wait until 2026-10-14T10:11:36.542Z to challenge them."))?.kind, .botDailyGameLimit)
        XCTAssertEqual(LichessBotChallengeRefusal.classify(postError: LichessBotGateError.rateLimited(cooldown: .seconds(60)))?.kind, .rateLimited)
    }

    func testQueueSkipsABotRefusedForTheOwnLimit() {
        let refusal = LichessBotOwnBotGameLimitRefusal(message: Self.ownMessage, secondsUntilClear: 60, respondedAt: Date())
        guard case .skipped = LichessBotController.queueOutcome(for: LichessBotAPIError.ownBotGameLimit(refusal)) else {
            return XCTFail("a human further down the queue can still be challenged")
        }
        let block = LichessBotBotGameBudget.Block.lichessReportedLimit(until: Date(), reason: "paused")
        XCTAssertEqual(LichessBotController.queueOutcome(for: LichessBotControllerError.botGameBudget(block)), .skipped(reason: "paused"))
    }

    // MARK: - The window takes Lichess's word

    /// The incident: DCM's replay counted 67 when Lichess said 100.
    private func incidentWindow(reportedUntil: Date?) -> LichessBotBotGameWindow {
        let starts = (0..<67).map { respondedAt.addingTimeInterval(-15 * hour + Double($0) * 60) }
        return LichessBotBotGameWindow(botGameStarts: starts, lichessReportedLimitUntil: reportedUntil)
    }

    func testLichessReportReplacesALowerReplayedCountUntilItsEnd() {
        let until = respondedAt.addingTimeInterval(6312)
        let replayed = incidentWindow(reportedUntil: nil)
        XCTAssertEqual(replayed.count(asOf: respondedAt)?.games, 67)
        let reported = incidentWindow(reportedUntil: until)
        XCTAssertEqual(reported.count(asOf: respondedAt), LichessBotBotGameWindow.Count(games: 100, clearsAt: until))
        XCTAssertEqual(reported.lichessReportedLimit(asOf: respondedAt), until)
        XCTAssertEqual(reported.lichessReportedLimit(asOf: until - 1), until)
        XCTAssertNil(reported.lichessReportedLimit(asOf: until), "the limit is over at its end")
        XCTAssertNil(reported.count(asOf: until + 1), "past its clear time the reported count limits nothing")
    }

    func testTheFirstBotGameAfterTheReportedEndStartsANewCount() {
        let until = respondedAt.addingTimeInterval(6312)
        var starts = (0..<67).map { respondedAt.addingTimeInterval(-15 * hour + Double($0) * 60) }
        starts.append(until + 60)
        let window = LichessBotBotGameWindow(botGameStarts: starts, lichessReportedLimitUntil: until)
        XCTAssertEqual(window.entry, LichessBotBotGameWindow.Entry(count: 1, clearAt: until + 60 + LichessBotBotGameWindow.length, lastWrittenAt: until + 60))
    }

    func testAReportWithNoRecordedGamesStillLimits() {
        let until = respondedAt.addingTimeInterval(600)
        let window = LichessBotBotGameWindow(botGameStarts: [], lichessReportedLimitUntil: until)
        XCTAssertEqual(window.count(asOf: respondedAt), LichessBotBotGameWindow.Count(games: 100, clearsAt: until))
    }

    // MARK: - Budget and matchmaking

    func testReportedLimitBlocksEverySenderUntilItsEnd() {
        let until = respondedAt.addingTimeInterval(2 * hour)
        let window = LichessBotBotGameWindow(botGameStarts: [], lichessReportedLimitUntil: until)
        for sender: LichessBotBotGameBudget.Sender in [.matchmaking, .challengeQueue, .operatorChallenge] {
            let block = LichessBotBotGameBudget.block(window: window, prospectiveBotGames: 0, sender: sender, settings: LichessBotChallengeSettings(), now: respondedAt, calendar: utc)
            guard case .lichessReportedLimit(let blockedUntil, let reason)? = block else {
                return XCTFail("\(sender) must stop at Lichess's limit")
            }
            XCTAssertEqual(blockedUntil, until)
            XCTAssertEqual(reason, LichessBotOwnBotGameLimit.pausedText(until: until, now: respondedAt, calendar: utc))
            XCTAssertTrue(reason.hasPrefix("Bot games paused: Lichess daily limit until "))
            // `until` already lies a rounding margin past lila's clear time,
            // which lila itself still counts; a moment later all is clear.
            XCTAssertNil(LichessBotBotGameBudget.block(window: window, prospectiveBotGames: 0, sender: sender, settings: LichessBotChallengeSettings(), now: until + 1, calendar: utc), "\(sender) resumes at the end")
        }
        XCTAssertNotNil(LichessBotBotGameBudget.lichessReportedBlock(window: window, now: respondedAt, calendar: utc))
        XCTAssertNil(LichessBotBotGameBudget.lichessReportedBlock(window: window, now: until, calendar: utc))
        XCTAssertEqual(LichessBotBotGameBudget.statusText(window: window, now: respondedAt, calendar: utc), "DCM: 100/100 bot games (Lichess) · clears \(LichessBotBotGameBudget.clearTimeText(until, now: respondedAt, calendar: utc))")
    }

    private func passConditions(window: LichessBotBotGameWindow, now: Date) -> LichessBotMatchmaking.PassConditions {
        LichessBotMatchmaking.PassConditions(isOnline: true, rateLimitHoldActive: false, gateOpen: true, hasChallengeScope: true, playOneGameActive: false, queueHasEntriesToSend: false, botGameWindow: window, prospectiveBotGames: 0, now: now, challengeSettings: LichessBotChallengeSettings())
    }

    func testMatchmakingSkipsBotsWhileLimitedAndResumesAfter() {
        let until = respondedAt.addingTimeInterval(2 * hour)
        let window = incidentWindow(reportedUntil: until)
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(passConditions(window: window, now: respondedAt)))
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(passConditions(window: window, now: until - 1)))
        XCTAssertNil(LichessBotMatchmaking.passBlockedReason(passConditions(window: window, now: until + 1)), "the reported limit is over, and with it the count it set")
        XCTAssertNil(LichessBotMatchmaking.passBlockedReason(passConditions(window: incidentWindow(reportedUntil: nil), now: respondedAt)), "without Lichess's report the estimate allowed it: the incident")
    }

    func testNextPassAfterTheOwnLimitIsTheUnproductiveRetry() {
        let outcome = LichessBotController.MatchmakingPassOutcome.blocked(reason: "Bot games paused")
        XCTAssertEqual(LichessBotController.nextAutomaticPass(after: outcome, now: respondedAt), respondedAt.addingTimeInterval(LichessBotMatchmaking.retryAfterUnproductivePass))
    }

    // MARK: - Incoming challenges

    private func challenge(id: String, title: String?) -> LichessBotChallenge {
        LichessBotChallenge(
            id: id,
            url: nil,
            status: LichessBotOpenValue(.created),
            challenger: LichessBotChallengeUser(id: "\(id)-player", name: "\(id)-player", rating: 1500, title: title, provisional: nil, online: true, lag: nil),
            destUser: nil,
            variant: LichessBotVariant(key: LichessBotOpenValue(.standard), name: nil, short: nil),
            rated: false,
            speed: LichessBotOpenValue(.blitz),
            timeControl: LichessBotTimeControl(type: LichessBotOpenValue(.clock), limit: LichessBotSeconds(300), increment: LichessBotSeconds(3), daysPerTurn: nil, show: nil),
            color: LichessBotOpenValue(.random),
            finalColor: nil,
            perf: nil,
            direction: LichessBotOpenValue(.incoming),
            initialFen: nil,
            rematchOf: nil
        )
    }

    private func context(ownLimitUntil: Date?) -> LichessBotChallengeContext {
        LichessBotChallengeContext(acceptingNewGames: true, activeGames: 0, activeGamesByOpponent: [:], gamesToday: 0, gamesTodayByOpponent: [:], challengeResponsesInLastMinute: 0, ownBotGameLimitUntil: ownLimitUntil)
    }

    func testBotChallengeDeclinedLaterWhileLimitedHumanUnaffected() {
        let settings = LichessBotChallengeSettings.testBaseline()
        let until = respondedAt.addingTimeInterval(hour)
        guard case .decline(.later, _) = LichessBotChallengePolicy.decide(challenge(id: "bot", title: "BOT"), compat: nil, settings: settings, context: context(ownLimitUntil: until)) else {
            return XCTFail("a bot's challenge is declined while Lichess holds DCM at its limit")
        }
        XCTAssertEqual(LichessBotChallengePolicy.decide(challenge(id: "human", title: nil), compat: nil, settings: settings, context: context(ownLimitUntil: until)), .accept)
        XCTAssertEqual(LichessBotChallengePolicy.decide(challenge(id: "bot", title: "BOT"), compat: nil, settings: settings, context: context(ownLimitUntil: nil)), .accept, "no limit reported: bots are accepted")
    }

    /// Through the session manager: its provider is read for each incoming
    /// challenge, so the bot is declined and the human accepted.
    func testManagerDeclinesBotsAndAcceptsHumansWhileLimited() async throws {
        let botLine = #"{"type":"challenge","challenge":{"id":"cbot","status":"created","challenger":{"id":"somebot","name":"SomeBot","rating":1500,"title":"BOT"},"variant":{"key":"standard","name":"x","short":"x"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3,"show":"5+3"},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#
        let humanLine = #"{"type":"challenge","challenge":{"id":"chuman","status":"created","challenger":{"id":"alice","name":"alice","rating":1500},"variant":{"key":"standard","name":"x","short":"x"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3,"show":"5+3"},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        let frozen = settings
        let time = LichessBotManualTime()
        let account = LichessBotFakeAccountAPI(script: [.openThenClose(lines: [botLine, humanLine])])
        let gate = LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in }
        let slots = try await LichessBotModelSlots.prepare(for: settings.model, provider: try await LichessBotFakeModelProvider.randomChampion(), time: time, folderScanner: LichessBotNoModelsFolderScanner(), log: { _ in })
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let until = Date().addingTimeInterval(hour)
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: try LichessBotFakeGameServer(),
            gate: gate,
            slots: slots,
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { frozen },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } },
            journalReader: LichessBotTestJournalReaders.none,
            ownBotGameLimitProvider: { until }
        )
        let run = Task { await manager.run() }
        for _ in 0..<4000 {
            let stopped = events.value.contains { if case .stopped = $0 { return true } else { return false } }
            if stopped { break }
            time.advance(by: .seconds(2))
            try await Task.sleep(for: .milliseconds(5))
        }
        await run.value
        let declined = await account.declined
        let accepted = await account.accepted
        XCTAssertEqual(declined.map(\.id), ["cbot"])
        XCTAssertEqual(declined.map(\.reason), [.later])
        XCTAssertEqual(accepted, ["chuman"])
    }
}
