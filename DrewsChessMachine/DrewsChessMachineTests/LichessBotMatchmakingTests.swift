import XCTest
@testable import DrewsChessMachine

/// A deterministic generator for the matchmaking picks (SplitMix64).
struct LichessBotSeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// `LichessBotMatchmaking` and `LichessBotMatchmakingRateLimiter` (plan
/// §7.3 B): one test per candidate rule, the rating window, reserved human
/// slots and fill modes, the per-hour cap and spacing, when a pass may run,
/// and the settings and notes that feed them.
final class LichessBotMatchmakingTests: XCTestCase {

    private let now = Date(timeIntervalSince1970: 1_790_000_000)

    private func bot(_ id: String, blitz: Int?, provisional: Bool = false) -> LichessBotUserSummary {
        let perfs: [String: LichessBotPerfRating]? = blitz.map {
            ["blitz": LichessBotPerfRating(games: 50, rating: $0, rd: 60, prog: 0, prov: provisional ? true : nil)]
        }
        return LichessBotUserSummary(id: id, username: id.capitalized, title: "BOT", perfs: perfs, online: true, disabled: nil, tosViolation: nil, createdAt: nil, seenAt: nil, profile: nil)
    }

    private func context(
        engaged: Set<String> = [],
        blocked: Set<String> = [],
        notes: LichessBotPlayerNotes? = LichessBotPlayerNotes(),
        gamesToday: [String: Int] = [:]
    ) -> LichessBotMatchmaking.CandidateContext {
        LichessBotMatchmaking.CandidateContext(
            ourAccountID: "drewschessmachine",
            blockedUserIDs: blocked,
            engagedUserIDs: engaged,
            notes: notes,
            gamesTodayByOpponent: gamesToday,
            maxGamesPerOpponentPerDay: 3,
            now: now
        )
    }

    private let window = LichessBotMatchmaking.RatingBounds(minimum: 1200, maximum: 1800, basis: .relative(ourRating: 1500))

    private func exclusion(_ bot: LichessBotUserSummary, _ context: LichessBotMatchmaking.CandidateContext) -> LichessBotMatchmaking.Exclusion? {
        LichessBotMatchmaking.exclusion(of: bot, speed: .blitz, bounds: window, context: context)
    }

    // MARK: - Candidate rules, one per rule

    func testAFittingBotIsACandidate() {
        XCTAssertNil(exclusion(bot("fits", blitz: 1500), context()))
        XCTAssertNil(exclusion(bot("edge", blitz: 1200), context()), "the window's bounds are inclusive")
        XCTAssertNil(exclusion(bot("edge", blitz: 1800), context()))
    }

    func testDCMItselfIsExcluded() {
        XCTAssertEqual(exclusion(bot("DrewsChessMachine", blitz: 1500), context()), .ourselves)
    }

    func testABotWithoutARatingAtTheSpeedIsExcluded() {
        XCTAssertEqual(exclusion(bot("unrated", blitz: nil), context()), .noRating)
    }

    func testAProvisionalBotIsExcluded() {
        XCTAssertEqual(exclusion(bot("newbot", blitz: 1500, provisional: true), context()), .provisional)
    }

    func testABotOutsideTheRatingWindowIsExcluded() {
        XCTAssertEqual(exclusion(bot("weak", blitz: 1199), context()), .outsideRatingWindow)
        XCTAssertEqual(exclusion(bot("strong", blitz: 1801), context()), .outsideRatingWindow)
    }

    func testABlockedBotIsExcluded() {
        XCTAssertEqual(exclusion(bot("rude", blitz: 1500), context(blocked: ["rude"])), .blocked)
    }

    func testABotWithAGamePendingOrQueuedChallengeIsExcluded() {
        XCTAssertEqual(exclusion(bot("Busy", blitz: 1500), context(engaged: ["busy"])), .alreadyEngaged)
    }

    func testABotAtItsBotLimitIsExcluded() {
        var notes = LichessBotPlayerNotes()
        notes.botLimitUntil["tired"] = now.addingTimeInterval(600)
        XCTAssertEqual(exclusion(bot("tired", blitz: 1500), context(notes: notes)), .atBotLimit)
        notes.botLimitUntil["tired"] = now.addingTimeInterval(-1)
        XCTAssertNil(exclusion(bot("tired", blitz: 1500), context(notes: notes)), "a limit time that has passed no longer excludes")
    }

    func testABotThatDeclinedWithinTheCooldownIsExcluded() {
        var notes = LichessBotPlayerNotes()
        notes.recordDeclineCooldown("Picky", until: now.addingTimeInterval(3600))
        XCTAssertEqual(exclusion(bot("picky", blitz: 1500), context(notes: notes)), .declineCooldown)
        notes.recordDeclineCooldown("picky", until: now.addingTimeInterval(-1))
        XCTAssertNil(exclusion(bot("picky", blitz: 1500), context(notes: notes)), "an expired cool-down no longer excludes")
    }

    func testABotAtThePerOpponentDailyLimitIsExcluded() {
        XCTAssertEqual(exclusion(bot("regular", blitz: 1500), context(gamesToday: ["regular": 3])), .dailyOpponentLimit)
        XCTAssertNil(exclusion(bot("regular", blitz: 1500), context(gamesToday: ["regular": 2])))
    }

    // MARK: - Rating window

    func testTheWindowFollowsDCMsEstablishedRating() {
        let settings = LichessBotMatchmakingSettings.testBaseline()
        let perfs = ["blitz": LichessBotPerfRating(games: 80, rating: 1650, rd: 50, prog: 0, prov: nil)]
        let bounds = LichessBotMatchmaking.ratingBounds(settings: settings, ourPerfs: perfs, speed: .blitz)
        XCTAssertEqual(bounds, LichessBotMatchmaking.RatingBounds(minimum: 1350, maximum: 1950, basis: .relative(ourRating: 1650)))
    }

    func testTheAbsoluteBoundsApplyWithoutAnEstablishedRating() {
        let settings = LichessBotMatchmakingSettings.testBaseline()
        let absolute = LichessBotMatchmaking.RatingBounds(minimum: 1000, maximum: 2200, basis: .absolute)
        XCTAssertEqual(LichessBotMatchmaking.ratingBounds(settings: settings, ourPerfs: nil, speed: .blitz), absolute)
        XCTAssertEqual(LichessBotMatchmaking.ratingBounds(settings: settings, ourPerfs: [:], speed: .rapid), absolute, "no rating at the speed")
        let provisional = ["blitz": LichessBotPerfRating(games: 3, rating: 1500, rd: 300, prog: 0, prov: true)]
        XCTAssertEqual(LichessBotMatchmaking.ratingBounds(settings: settings, ourPerfs: provisional, speed: .blitz), absolute, "a provisional rating is not established")
    }

    // MARK: - Picking

    private func pickSettings(preferFavorites: Bool = false) -> LichessBotMatchmakingSettings {
        var settings = LichessBotMatchmakingSettings.testBaseline()
        settings.timeControls = [.blitz5plus3]
        settings.preferFavorites = preferFavorites
        return settings
    }

    private let ourPerfs = ["blitz": LichessBotPerfRating(games: 80, rating: 1500, rd: 50, prog: 0, prov: nil)]

    func testThePickIsUniformAmongCandidates() {
        let bots = [bot("a", blitz: 1400), bot("b", blitz: 1500), bot("c", blitz: 1600), bot("far", blitz: 2500)]
        var generator = LichessBotSeededGenerator(seed: 7)
        var counts: [String: Int] = [:]
        for _ in 0..<600 {
            guard case .picked(let pick) = LichessBotMatchmaking.pick(from: bots, settings: pickSettings(), ourPerfs: ourPerfs, context: context(), using: &generator) else {
                return XCTFail("three bots fit")
            }
            XCTAssertEqual(pick.clock, .blitz5plus3)
            XCTAssertEqual(pick.candidateCount, 3)
            XCTAssertEqual(pick.exclusions, [.outsideRatingWindow: 1])
            counts[pick.bot.id, default: 0] += 1
        }
        XCTAssertEqual(Set(counts.keys), ["a", "b", "c"])
        for (id, count) in counts {
            XCTAssertGreaterThan(count, 150, "\(id) picked \(count) of 600 times")
        }
    }

    func testPreferFavoritesPicksAmongFittingFavoritesFirst() {
        var notes = LichessBotPlayerNotes()
        notes.toggleFavorite("b")
        notes.toggleFavorite("far")
        let bots = [bot("a", blitz: 1400), bot("b", blitz: 1500), bot("far", blitz: 2500)]
        var generator = LichessBotSeededGenerator(seed: 11)
        for _ in 0..<50 {
            guard case .picked(let pick) = LichessBotMatchmaking.pick(from: bots, settings: pickSettings(preferFavorites: true), ourPerfs: ourPerfs, context: context(notes: notes), using: &generator) else {
                return XCTFail("a favorite fits")
            }
            XCTAssertEqual(pick.bot.id, "b", "the only favorite that fits")
            XCTAssertTrue(pick.fromFavorites)
        }
    }

    func testPreferFavoritesFallsBackToEveryCandidateWhenNoFavoriteFits() {
        var notes = LichessBotPlayerNotes()
        notes.toggleFavorite("far")
        var generator = LichessBotSeededGenerator(seed: 3)
        guard case .picked(let pick) = LichessBotMatchmaking.pick(from: [bot("a", blitz: 1400), bot("far", blitz: 2500)], settings: pickSettings(preferFavorites: true), ourPerfs: ourPerfs, context: context(notes: notes), using: &generator) else {
            return XCTFail("a non-favorite fits")
        }
        XCTAssertEqual(pick.bot.id, "a")
        XCTAssertFalse(pick.fromFavorites)
    }

    func testTheClockIsChosenFromTheConfiguredOnes() {
        var settings = pickSettings()
        settings.timeControls = [.blitz3plus2, .rapid10plus5]
        let perfs = ["blitz": LichessBotPerfRating(games: 80, rating: 1500, rd: 50, prog: 0, prov: nil)]
        let bots = [LichessBotUserSummary(id: "both", username: "Both", title: "BOT", perfs: [
            "blitz": LichessBotPerfRating(games: 50, rating: 1500, rd: 60, prog: 0, prov: nil),
            "rapid": LichessBotPerfRating(games: 50, rating: 1500, rd: 60, prog: 0, prov: nil),
        ], online: true, disabled: nil, tosViolation: nil, createdAt: nil, seenAt: nil, profile: nil)]
        var generator = LichessBotSeededGenerator(seed: 5)
        var clocks: Set<LichessBotClockChoice> = []
        for _ in 0..<100 {
            guard case .picked(let pick) = LichessBotMatchmaking.pick(from: bots, settings: settings, ourPerfs: perfs, context: context(), using: &generator) else {
                return XCTFail("the bot fits at both speeds")
            }
            clocks.insert(pick.clock)
            if pick.clock == .rapid10plus5 {
                XCTAssertEqual(pick.bounds.basis, .absolute, "DCM has no rapid rating")
            }
        }
        XCTAssertEqual(clocks, [.blitz3plus2, .rapid10plus5])
    }

    func testNoCandidateReportsWhyEachBotWasLeftOut() {
        var generator = LichessBotSeededGenerator(seed: 1)
        let bots = [bot("drewschessmachine", blitz: 1500), bot("far", blitz: 2500), bot("far", blitz: 2500), bot("new", blitz: 1500, provisional: true)]
        let result = LichessBotMatchmaking.pick(from: bots, settings: pickSettings(), ourPerfs: ourPerfs, context: context(), using: &generator)
        guard case .noCandidate(let clock, _, let listed, let exclusions) = result else {
            return XCTFail("nobody fits: \(result)")
        }
        XCTAssertEqual(clock, .blitz5plus3)
        XCTAssertEqual(listed, 3, "a bot listed twice counts once")
        XCTAssertEqual(exclusions, [.ourselves: 1, .outsideRatingWindow: 1, .provisional: 1])
        XCTAssertEqual(LichessBotMatchmaking.describe(exclusions), "DCM itself 1, provisional at the speed 1, outside the rating window 1")
    }

    // MARK: - Slots

    func testReservedHumanSlotsAreNeverUsed() {
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 3, gamesReservedForHumans: 1, fillMode: .everyFreeSlot, committed: 0), 2)
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 3, gamesReservedForHumans: 1, fillMode: .everyFreeSlot, committed: 1), 1)
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 3, gamesReservedForHumans: 1, fillMode: .everyFreeSlot, committed: 2), 0, "the last slot stays free for a human")
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 3, gamesReservedForHumans: 1, fillMode: .everyFreeSlot, committed: 3), 0)
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 2, gamesReservedForHumans: 2, fillMode: .everyFreeSlot, committed: 0), 0)
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 2, gamesReservedForHumans: 2, fillMode: .onlyWhenIdle, committed: 0), 0)
    }

    func testIdleFillModeSendsOnlyWhenNothingIsInPlay() {
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 4, gamesReservedForHumans: 0, fillMode: .onlyWhenIdle, committed: 0), 1)
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 4, gamesReservedForHumans: 0, fillMode: .onlyWhenIdle, committed: 1), 0)
        XCTAssertEqual(LichessBotMatchmaking.openSlots(maxConcurrentGames: 4, gamesReservedForHumans: 0, fillMode: .everyFreeSlot, committed: 1), 3)
    }

    // MARK: - Rate

    func testSpacingBetweenSends() {
        var limiter = LichessBotMatchmakingRateLimiter()
        XCTAssertEqual(limiter.decision(now: now, perHourCap: 20, minimumSpacing: 20), .allowed)
        limiter.recordAttempt(at: now, reachedLichess: false)
        XCTAssertEqual(limiter.decision(now: now.addingTimeInterval(10), perHourCap: 20, minimumSpacing: 20), .spacing(until: now.addingTimeInterval(20)), "every attempt is spaced, even one that never reached Lichess")
        XCTAssertEqual(limiter.decision(now: now.addingTimeInterval(20), perHourCap: 20, minimumSpacing: 20), .allowed)
        XCTAssertEqual(limiter.challengesInLastHour(now: now.addingTimeInterval(20)), 0, "an attempt that never reached Lichess isn't counted")
    }

    func testThePerHourCap() {
        var limiter = LichessBotMatchmakingRateLimiter()
        for index in 0..<3 {
            limiter.recordAttempt(at: now.addingTimeInterval(TimeInterval(index * 20)), reachedLichess: true)
        }
        let later = now.addingTimeInterval(600)
        XCTAssertEqual(limiter.challengesInLastHour(now: later), 3)
        XCTAssertEqual(limiter.decision(now: later, perHourCap: 3, minimumSpacing: 20), .hourlyCap(until: now.addingTimeInterval(3600), count: 3))
        XCTAssertEqual(limiter.decision(now: later, perHourCap: 4, minimumSpacing: 20), .allowed)
        XCTAssertEqual(limiter.decision(now: now.addingTimeInterval(3600), perHourCap: 3, minimumSpacing: 20), .allowed, "the oldest challenge has left the hour")
    }

    // MARK: - When a pass runs

    private func conditions() -> LichessBotMatchmaking.PassConditions {
        LichessBotMatchmaking.PassConditions(isOnline: true, rateLimitHoldActive: false, gateOpen: true, hasChallengeScope: true, playOneGameActive: false, queueHasEntriesToSend: false, botGamesInLastDay: 10)
    }

    func testAPassRunsOnlyWhileOnlineWithNoHoldAndAnEmptyQueue() {
        XCTAssertNil(LichessBotMatchmaking.passBlockedReason(conditions()))
        var offline = conditions()
        offline.isOnline = false
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(offline))
        var held = conditions()
        held.rateLimitHoldActive = true
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(held))
        var gateShut = conditions()
        gateShut.gateOpen = false
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(gateShut))
        var noScope = conditions()
        noScope.hasChallengeScope = false
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(noScope))
        var oneGame = conditions()
        oneGame.playOneGameActive = true
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(oneGame))
        var queued = conditions()
        queued.queueHasEntriesToSend = true
        XCTAssertEqual(LichessBotMatchmaking.passBlockedReason(queued), "the challenge queue goes first")
    }

    func testAPassStopsAtDCMsBotGameBudget() {
        var unloaded = conditions()
        unloaded.botGamesInLastDay = nil
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(unloaded))
        var spent = conditions()
        spent.botGamesInLastDay = LichessBotLimits.botGamesPerDay
        XCTAssertNotNil(LichessBotMatchmaking.passBlockedReason(spent))
        var almost = conditions()
        almost.botGamesInLastDay = LichessBotLimits.botGamesPerDay - 1
        XCTAssertNil(LichessBotMatchmaking.passBlockedReason(almost))
    }

    func testTheNextAutomaticPassWaitsAfterAnUnproductiveOne() {
        XCTAssertEqual(LichessBotController.nextAutomaticPass(after: .sent(username: "b"), now: now), now)
        XCTAssertEqual(LichessBotController.nextAutomaticPass(after: .rateLimited(.spacing(until: now.addingTimeInterval(9))), now: now), now.addingTimeInterval(9))
        XCTAssertEqual(LichessBotController.nextAutomaticPass(after: .rateLimited(.hourlyCap(until: now.addingTimeInterval(900), count: 20)), now: now), now.addingTimeInterval(900))
        XCTAssertEqual(LichessBotController.nextAutomaticPass(after: .noCandidate("nobody"), now: now), now.addingTimeInterval(LichessBotMatchmaking.retryAfterUnproductivePass))
    }

    // MARK: - Settings and notes

    func testMatchmakingIsOffByDefaultWithThePlansDefaults() {
        let settings = LichessBotSettings()
        XCTAssertTrue(settings.validationProblems().isEmpty)
        let m = settings.matchmaking
        XCTAssertTrue(m.enabled)
        XCTAssertEqual(m.fillMode, .everyFreeSlot)
        XCTAssertEqual(m.timeControls, [.bullet1plus0, .bullet2plus1, .ultraBulletQuarterPlus0, .blitz3plus0, .blitz3plus2, .bullet1plus1, .blitz5plus3, .rapid10plus0, .blitz5plus0])
        XCTAssertTrue(m.rated)
        XCTAssertEqual(m.minimumRatingOffset, -300)
        XCTAssertEqual(m.maximumRatingOffset, 300)
        XCTAssertEqual(m.minimumRatingWithoutOwnRating, 0)
        XCTAssertEqual(m.maximumRatingWithoutOwnRating, 2200)
        XCTAssertEqual(m.maxChallengesPerHour, 10)
        XCTAssertEqual(m.declineCooldownHours, 2)
    }

    func testInvalidMatchmakingSettingsAreRejected() {
        var noClock = LichessBotSettings.testBaseline()
        noClock.matchmaking.timeControls = []
        XCTAssertFalse(noClock.validationProblems().isEmpty)
        var reversed = LichessBotSettings.testBaseline()
        reversed.matchmaking.minimumRatingOffset = 100
        reversed.matchmaking.maximumRatingOffset = -100
        XCTAssertFalse(reversed.validationProblems().isEmpty)
        var absolute = LichessBotSettings.testBaseline()
        absolute.matchmaking.minimumRatingWithoutOwnRating = 2300
        XCTAssertFalse(absolute.validationProblems().isEmpty)
        var noCap = LichessBotSettings.testBaseline()
        noCap.matchmaking.maxChallengesPerHour = 0
        XCTAssertFalse(noCap.validationProblems().isEmpty)
        var negativeCooldown = LichessBotSettings.testBaseline()
        negativeCooldown.matchmaking.declineCooldownHours = -1
        XCTAssertFalse(negativeCooldown.validationProblems().isEmpty)
    }

    func testMatchmakingSettingsRoundTrip() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings()
        settings.matchmaking.enabled = true
        settings.matchmaking.fillMode = .onlyWhenIdle
        settings.matchmaking.timeControls = [.rapid10plus5]
        settings.matchmaking.preferFavorites = false
        try LichessBotSettingsStore.save(settings, to: defaults)
        XCTAssertEqual(try LichessBotSettingsStore.load(from: defaults), settings)
    }

    func testDeclineCooldownsPersistAndArePruned() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("player-notes-\(UUID().uuidString).json")
        addTeardownBlock {
            if FileManager.default.fileExists(atPath: url.path) {
                do {
                    try FileManager.default.removeItem(at: url)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        var notes = LichessBotPlayerNotes()
        notes.recordDeclineCooldown("Picky", until: now.addingTimeInterval(3600))
        notes.recordDeclineCooldown("gone", until: now.addingTimeInterval(-60))
        try notes.save(to: url)
        var loaded = try LichessBotPlayerNotes.load(from: url)
        XCTAssertEqual(loaded, notes)
        loaded.pruneExpiredLimits(now: now)
        XCTAssertEqual(loaded.declineCooldownUntil, ["picky": now.addingTimeInterval(3600)])
        XCTAssertEqual(loaded.declineCooldownEnds("PICKY", now: now), now.addingTimeInterval(3600))
    }

    /// A notes file written before decline cool-downs existed still loads,
    /// favorites intact.
    func testNotesWithoutCooldownsStillLoad() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("player-notes-\(UUID().uuidString).json")
        addTeardownBlock {
            if FileManager.default.fileExists(atPath: url.path) {
                do {
                    try FileManager.default.removeItem(at: url)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        try Data(#"{"botLimitUntil":{},"favoriteIDs":["alpha"]}"#.utf8).write(to: url)
        let loaded = try LichessBotPlayerNotes.load(from: url)
        XCTAssertEqual(loaded.favoriteIDs, ["alpha"])
        XCTAssertNil(loaded.declineCooldownUntil)
        XCTAssertNil(loaded.declineCooldownEnds("alpha", now: now))
    }
}
