import XCTest
@testable import DrewsChessMachine

/// Matchmaking's "Fall back to casual when asked"
/// (`LichessBotMatchmakingSettings.fallBackToCasual`), over a whole runtime
/// against `LichessBotFakeLichess`. A bot that declines one of matchmaking's
/// rated challenges with Lichess's `casual` reason gets the same challenge
/// once more, unrated, through matchmaking's own send path — and no decline
/// cool-down unless that resend is declined too or can't be sent. With the
/// setting off, for any other reason, and for the operator's own
/// challenges, a decline is answered exactly as before: the cool-down, plus
/// the manual Resend as Casual offer for a `casual` decline of a rated
/// challenge. Also pins that settings saved before the field existed load
/// it as off.
@MainActor
final class LichessBotCasualFallbackTests: XCTestCase {

    private static let fitBotLine = #"{"id":"fitbot","username":"FitBot","title":"BOT","perfs":{"blitz":{"games":50,"rating":1600,"rd":60,"prog":0}}}"#
    private static let casualReason = "Please send me a casual challenge instead."

    /// What matchmaking sends with the settings below: 5+3, color random.
    private static func matchmakingRequest(rated: Bool) -> LichessBotOutgoingChallenge {
        LichessBotOutgoingChallenge(rated: rated, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)
    }

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// An online controller with its own defaults suite, data folder and
    /// fake Lichess, all removed after the test. DCM is established at
    /// 1500 blitz and FitBot (1600 blitz) is the only online bot, so it is
    /// the one matchmaking candidate. One slot is open to matchmaking (the
    /// other is reserved for humans), automatic matchmaking is off so only
    /// the test's Fill Open Slots sends, and matchmaking sends rated.
    private func makeOnlineController(
        lichess: LichessBotFakeLichess,
        configure: (inout LichessBotSettings) -> Void
    ) async throws -> LichessBotController {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotCasualFallbackTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        // No automatic withdrawal: the tests decide when challenges end.
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        settings.challenge.maxConcurrentGames = 2
        settings.challenge.gamesReservedForHumans = 1
        settings.matchmaking.enabled = false
        settings.matchmaking.timeControls = [.blitz5plus3]
        settings.matchmaking.rated = true
        settings.matchmaking.declineCooldownHours = 6
        configure(&settings)
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = LichessBotController(
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { lichess },
                readToken: { _ in token }
            )
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            // Writes the stopped runtime already queued land; anything later
            // is refused, so nothing races the removal of its folder.
            await controller.shutdown(reason: "test teardown")
            if FileManager.default.fileExists(atPath: root.path) {
                do {
                    try FileManager.default.removeItem(at: root)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        await controller.loadPlayerNotes()
        XCTAssertNotNil(controller.playerNotes, "the decline cool-down is recorded in the player notes")
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        XCTAssertTrue(controller.hasChallengeScope)
        try await waitUntil("the event stream is open") { lichess.eventStreamIsOpen }
        return controller
    }

    private func makeLichess() -> LichessBotFakeLichess {
        let lichess = LichessBotFakeLichess(accountPerfsJSON: #"{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}"#)
        lichess.onlineBotsNDJSON.value = Self.fitBotLine + "\n"
        return lichess
    }

    /// Fill Open Slots sends matchmaking's rated challenge to FitBot.
    private func sendMatchmakingChallengeToFitBot(_ controller: LichessBotController, _ lichess: LichessBotFakeLichess) async {
        await controller.fillOpenSlots()
        XCTAssertEqual(lichess.challengedNames.value, ["FitBot"])
        XCTAssertEqual(controller.pendingChallenges.map(\.username), ["FitBot"])
        XCTAssertEqual(controller.pendingChallenges.first?.request, Self.matchmakingRequest(rated: true))
    }

    private func decline(_ username: String, reasonKey: String, reason: String, on lichess: LichessBotFakeLichess) {
        lichess.sendEvent(#"{"type":"challengeDeclined","challenge":{"id":"\#(LichessBotFakeLichess.challengeID(for: username))","declineReason":"\#(reason)","declineReasonKey":"\#(reasonKey)"}}"#)
    }

    private func cooldownEnds(_ controller: LichessBotController, _ userID: String) -> Date? {
        controller.playerNotes?.declineCooldownEnds(userID, now: Date())
    }

    // MARK: - Behavior

    /// Off (today's behavior): the casual decline gets the cool-down and the
    /// manual offer; nothing is resent.
    func testOffACasualDeclineGetsTheCooldownAndTheManualOfferOnly() async throws {
        let lichess = makeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.matchmaking.fallBackToCasual = false
        }
        await sendMatchmakingChallengeToFitBot(controller, lichess)

        decline("FitBot", reasonKey: "casual", reason: Self.casualReason, on: lichess)
        try await waitUntil("the manual offer is made") { controller.casualResendOffer != nil }

        XCTAssertEqual(controller.casualResendOffer, LichessBotController.CasualResendOffer(username: "FitBot", request: Self.matchmakingRequest(rated: false)))
        XCTAssertNotNil(cooldownEnds(controller, "fitbot"), "the decline starts the cool-down")
        XCTAssertEqual(lichess.challengedNames.value, ["FitBot"], "nothing is resent")
        XCTAssertTrue(controller.pendingChallenges.isEmpty)
        XCTAssertEqual(controller.lastChallengeOutcome, "FitBot: declined: \(Self.casualReason)")
        XCTAssertEqual(controller.matchmakingRateLimiter.challengesInLastHour(now: Date()), 1)
    }

    /// On: exactly one casual challenge goes out, same clock and color, as a
    /// matchmaking send (counted by the rate limiter), and no cool-down or
    /// manual offer is recorded.
    func testOnACasualDeclineOfARatedMatchmakingChallengeIsResentOnceAsCasual() async throws {
        let lichess = makeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.matchmaking.fallBackToCasual = true
        }
        await sendMatchmakingChallengeToFitBot(controller, lichess)

        decline("FitBot", reasonKey: "casual", reason: Self.casualReason, on: lichess)
        try await waitUntil("the resend is reported") {
            controller.lastChallengeOutcome == "FitBot: declined rated (casual); resent as casual"
        }

        XCTAssertEqual(lichess.challengedNames.value, ["FitBot", "FitBot"], "exactly one resend")
        XCTAssertEqual(controller.pendingChallenges.map(\.username), ["FitBot"])
        XCTAssertEqual(controller.pendingChallenges.first?.request, Self.matchmakingRequest(rated: false))
        XCTAssertEqual(controller.pendingChallenges.first?.origin, LichessBotController.ChallengeOrigin.matchmakingCasualResend)
        XCTAssertNil(cooldownEnds(controller, "fitbot"), "no cool-down for the decline that asked for casual")
        XCTAssertNil(controller.casualResendOffer, "no manual offer: the resend was sent")
        XCTAssertEqual(controller.matchmakingRateLimiter.challengesInLastHour(now: Date()), 2, "the resend is a matchmaking send")
    }

    /// The resend declined — even again with the casual reason — gets the
    /// ordinary cool-down and is never resent.
    func testADeclinedResendGetsTheCooldownAndIsNotResent() async throws {
        let lichess = makeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.matchmaking.fallBackToCasual = true
        }
        await sendMatchmakingChallengeToFitBot(controller, lichess)
        decline("FitBot", reasonKey: "casual", reason: Self.casualReason, on: lichess)
        try await waitUntil("the resend is pending") {
            controller.lastChallengeOutcome == "FitBot: declined rated (casual); resent as casual"
        }

        decline("FitBot", reasonKey: "casual", reason: Self.casualReason, on: lichess)
        try await waitUntil("the resend's decline starts the cool-down") { self.cooldownEnds(controller, "fitbot") != nil }

        XCTAssertEqual(lichess.challengedNames.value, ["FitBot", "FitBot"], "no further sends")
        XCTAssertTrue(controller.pendingChallenges.isEmpty)
        XCTAssertNil(controller.casualResendOffer, "the resend was already casual")
        XCTAssertEqual(controller.lastChallengeOutcome, "FitBot: declined: \(Self.casualReason)")
    }

    /// On, any reason other than `casual` is answered as before: the
    /// cool-down, no offer, no resend.
    func testOnOtherDeclineReasonsAreUnchanged() async throws {
        let lichess = makeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.matchmaking.fallBackToCasual = true
        }
        await sendMatchmakingChallengeToFitBot(controller, lichess)

        decline("FitBot", reasonKey: "later", reason: "I'm not accepting challenges at the moment.", on: lichess)
        try await waitUntil("the decline starts the cool-down") { self.cooldownEnds(controller, "fitbot") != nil }

        XCTAssertEqual(lichess.challengedNames.value, ["FitBot"])
        XCTAssertTrue(controller.pendingChallenges.isEmpty)
        XCTAssertNil(controller.casualResendOffer)
        XCTAssertEqual(controller.lastChallengeOutcome, "FitBot: declined: I'm not accepting challenges at the moment.")
    }

    /// On, the operator's own rated challenge declined as casual gets only
    /// the manual offer (and the cool-down), never an automatic resend.
    func testOnTheOperatorsOwnChallengeGetsOnlyTheManualOffer() async throws {
        let lichess = makeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.matchmaking.fallBackToCasual = true
        }
        let rated = LichessBotOutgoingChallenge(rated: true, clockLimitSeconds: 180, clockIncrementSeconds: 2, color: .white)
        try await controller.sendChallenge(to: "bob", request: rated)
        XCTAssertEqual(controller.pendingChallenges.first?.origin, LichessBotController.ChallengeOrigin.manual)

        decline("bob", reasonKey: "casual", reason: Self.casualReason, on: lichess)
        try await waitUntil("the manual offer is made") { controller.casualResendOffer != nil }

        XCTAssertEqual(controller.casualResendOffer, LichessBotController.CasualResendOffer(username: "bob", request: LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 180, clockIncrementSeconds: 2, color: .white)))
        XCTAssertNotNil(cooldownEnds(controller, "bob"))
        XCTAssertEqual(lichess.challengedNames.value, ["bob"], "no automatic resend")
        XCTAssertTrue(controller.pendingChallenges.isEmpty)
    }

    /// On, a matchmaking guard that blocks the resend — here the hourly cap,
    /// reached by the challenge being declined — means no send, and the
    /// decline is answered as with the setting off, with the reason in the
    /// outcome line.
    func testAGuardBlockingTheResendFallsBackToTheCooldownAndTheManualOffer() async throws {
        let lichess = makeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.matchmaking.fallBackToCasual = true
            settings.matchmaking.maxChallengesPerHour = 1
        }
        await sendMatchmakingChallengeToFitBot(controller, lichess)

        decline("FitBot", reasonKey: "casual", reason: Self.casualReason, on: lichess)
        try await waitUntil("the fallback's manual offer is made") { controller.casualResendOffer != nil }

        XCTAssertEqual(lichess.challengedNames.value, ["FitBot"], "the hourly cap stops the resend")
        XCTAssertTrue(controller.pendingChallenges.isEmpty)
        XCTAssertEqual(controller.casualResendOffer, LichessBotController.CasualResendOffer(username: "FitBot", request: Self.matchmakingRequest(rated: false)))
        XCTAssertNotNil(cooldownEnds(controller, "fitbot"), "the cool-down applies after all")
        let outcome = try XCTUnwrap(controller.lastChallengeOutcome)
        XCTAssertTrue(outcome.hasPrefix("FitBot: resending as casual failed: "), outcome)
        XCTAssertTrue(outcome.contains("the cap"), outcome)
        XCTAssertEqual(controller.matchmakingRateLimiter.challengesInLastHour(now: Date()), 1)
    }

    // MARK: - Settings

    /// Settings saved before `fallBackToCasual` existed still load, keep
    /// every saved value, and get the field off — the behavior they were
    /// saved under — reported as filled from the defaults.
    func testSettingsSavedBeforeTheFieldLoadWithItOff() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.matchmaking.rated = true
        settings.matchmaking.maxChallengesPerHour = 7
        settings.matchmaking.declineCooldownHours = 3
        let data = try JSONEncoder().encode(settings)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var matchmaking = try XCTUnwrap(object["matchmaking"] as? [String: Any])
        XCTAssertNotNil(matchmaking["fallBackToCasual"], "the key this test removes")
        matchmaking.removeValue(forKey: "fallBackToCasual")
        object["matchmaking"] = matchmaking
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: LichessBotSettingsStore.defaultsKey)

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertFalse(result.settings.matchmaking.fallBackToCasual)
        XCTAssertEqual(result.filledFromDefaults, ["matchmaking.fallBackToCasual"])
        XCTAssertEqual(result.ignoredSavedKeys, [])
        XCTAssertTrue(result.settings.matchmaking.rated)
        XCTAssertEqual(result.settings.matchmaking.maxChallengesPerHour, 7)
        XCTAssertEqual(result.settings.matchmaking.declineCooldownHours, 3)
    }

    func testFallBackToCasualRoundTrips() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.matchmaking.fallBackToCasual = true
        try LichessBotSettingsStore.save(settings, to: defaults)

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertTrue(result.settings.matchmaking.fallBackToCasual)
        XCTAssertEqual(result.settings, settings)
        XCTAssertEqual(result.filledFromDefaults, [])
    }

    func testFallBackToCasualIsOffByDefault() {
        XCTAssertFalse(LichessBotMatchmakingSettings().fallBackToCasual)
    }
}
