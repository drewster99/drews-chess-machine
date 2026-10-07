import XCTest
@testable import DrewsChessMachine

/// Lichess' bot-vs-bot daily limit as lila keeps it (RateLimit + Caffeine
/// expire-after-write), and the reserves matchmaking and the queue leave.
final class LichessBotBotGameWindowTests: XCTestCase {

    /// 2026-10-05 08:00:00 UTC.
    private let t0 = Date(timeIntervalSince1970: 1_791_187_200)
    private let hour: TimeInterval = 3600
    private let day = LichessBotBotGameWindow.length

    private var utc: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        return calendar
    }

    private typealias Entry = LichessBotBotGameWindow.Entry
    private typealias Count = LichessBotBotGameWindow.Count

    // MARK: - The model

    func testNoGamesMeansNoEntry() {
        let window = LichessBotBotGameWindow(botGameStarts: [])
        XCTAssertNil(window.entry)
        XCTAssertNil(window.count(asOf: t0))
    }

    func testGamesCountUntilADayAfterTheFirst() {
        let window = LichessBotBotGameWindow(botGameStarts: [t0, t0 + hour, t0 + 2 * hour])
        XCTAssertEqual(window.entry, Entry(count: 3, clearAt: t0 + day, lastWrittenAt: t0 + 2 * hour))
        XCTAssertEqual(window.count(asOf: t0 + 3 * hour), Count(games: 3, clearsAt: t0 + day))
    }

    /// lila resets only when `now > clearAt`: the clear time itself still counts.
    func testTheClearTimeItselfStillCounts() {
        let window = LichessBotBotGameWindow(botGameStarts: [t0, t0 + hour])
        XCTAssertEqual(window.count(asOf: t0 + day), Count(games: 2, clearsAt: t0 + day))
        XCTAssertNil(window.count(asOf: t0 + day + 1))
    }

    /// Caffeine drops an entry at exactly a day after its last write (`>=`).
    func testAnEntryExpiresADayAfterItsLastWrite() {
        let expired = LichessBotBotGameWindow(botGameStarts: [t0, t0 + day])
        XCTAssertEqual(expired.entry, Entry(count: 1, clearAt: t0 + 2 * day, lastWrittenAt: t0 + day))
        let kept = LichessBotBotGameWindow(botGameStarts: [t0, t0 + hour, t0 + day])
        XCTAssertEqual(kept.entry, Entry(count: 3, clearAt: t0 + day, lastWrittenAt: t0 + day), "a later write kept it, so a game at the clear time joins it")
    }

    /// Below the limit, lila counts on past the clear time while games come
    /// less than a day apart, but that count can no longer limit the account.
    func testBelowTheLimitAnEntryOutlivesItsClearTimeButCannotLimit() {
        let window = LichessBotBotGameWindow(botGameStarts: [t0, t0 + 23 * hour, t0 + 25 * hour])
        XCTAssertEqual(window.entry, Entry(count: 3, clearAt: t0 + day, lastWrittenAt: t0 + 25 * hour))
        XCTAssertNil(window.count(asOf: t0 + 25 * hour))
    }

    /// Reaching the limit after the clear time limits nothing; the next game
    /// replaces the entry.
    func testReachingTheLimitAfterTheClearTimeStartsANewEntryWithTheNextGame() {
        let late = (0..<98).map { t0 + 25 * hour + Double($0) }
        let full = LichessBotBotGameWindow(botGameStarts: [t0, t0 + 23 * hour] + late)
        XCTAssertEqual(full.entry?.count, 100)
        XCTAssertNil(full.count(asOf: t0 + 26 * hour))
        let next = LichessBotBotGameWindow(botGameStarts: [t0, t0 + 23 * hour] + late + [t0 + 26 * hour])
        XCTAssertEqual(next.entry, Entry(count: 1, clearAt: t0 + 50 * hour, lastWrittenAt: t0 + 26 * hour))
        XCTAssertEqual(next.count(asOf: t0 + 26 * hour), Count(games: 1, clearsAt: t0 + 50 * hour))
    }

    /// At the limit, a game before or at the clear time writes nothing; one
    /// after it opens a new entry.
    func testAtTheLimitAGameUpToTheClearTimeIsNotCounted() {
        let hundred = (0..<100).map { t0 + Double($0) }
        let limited = LichessBotBotGameWindow(botGameStarts: hundred + [t0 + 12 * hour, t0 + day])
        XCTAssertEqual(limited.entry, Entry(count: 100, clearAt: t0 + day, lastWrittenAt: t0 + 99))
        XCTAssertEqual(limited.count(asOf: t0 + day), Count(games: 100, clearsAt: t0 + day))
        let after = LichessBotBotGameWindow(botGameStarts: hundred + [t0 + day + 1])
        XCTAssertEqual(after.entry, Entry(count: 1, clearAt: t0 + 2 * day + 1, lastWrittenAt: t0 + day + 1))
    }

    func testStartOrderDoesNotMatter() {
        let window = LichessBotBotGameWindow(botGameStarts: [t0 + 2 * hour, t0, t0 + hour])
        XCTAssertEqual(window.entry, Entry(count: 3, clearAt: t0 + day, lastWrittenAt: t0 + 2 * hour))
    }

    /// The case that prompted this model: 102 games from 2026-10-06 16:33:58
    /// UTC ten minutes apart; the last two are not counted, and all 100 come
    /// back at 16:33:58 the next day.
    func testAFullWindowClearsADayAfterItsFirstGame() {
        let first = Date(timeIntervalSince1970: 1_791_304_438)
        let window = LichessBotBotGameWindow(botGameStarts: (0..<102).map { first + Double($0) * 600 })
        XCTAssertEqual(window.entry, Entry(count: 100, clearAt: first + day, lastWrittenAt: first + 99 * 600))
        XCTAssertEqual(window.count(asOf: first + 23 * hour), Count(games: 100, clearsAt: first + day))
        XCTAssertNil(window.count(asOf: first + day + 1))
    }

    // MARK: - The reserves

    private func window(games: Int) -> LichessBotBotGameWindow {
        LichessBotBotGameWindow(botGameStarts: (0..<games).map { t0 + Double($0) })
    }

    private func settings(incoming: Int, queue: Int) -> LichessBotChallengeSettings {
        var settings = LichessBotChallengeSettings()
        settings.botGamesReservedForIncoming = incoming
        settings.botGamesReservedForChallengeQueue = queue
        return settings
    }

    private func block(_ window: LichessBotBotGameWindow?, prospective: Int = 0, _ sender: LichessBotBotGameBudget.Sender, _ settings: LichessBotChallengeSettings, asOf: Date? = nil) -> LichessBotBotGameBudget.Block? {
        LichessBotBotGameBudget.block(window: window, prospectiveBotGames: prospective, sender: sender, settings: settings, now: asOf ?? t0 + hour, calendar: utc)
    }

    func testAllowances() {
        let reserving = settings(incoming: 10, queue: 5)
        XCTAssertEqual(LichessBotBotGameBudget.allowance(for: .matchmaking, settings: reserving), 85)
        XCTAssertEqual(LichessBotBotGameBudget.allowance(for: .challengeQueue, settings: reserving), 90)
        XCTAssertEqual(LichessBotBotGameBudget.allowance(for: .operatorChallenge, settings: reserving), 100)
        XCTAssertEqual(LichessBotBotGameBudget.allowance(for: .matchmaking, settings: LichessBotChallengeSettings()), LichessBotLimits.botGamesPerDay)
    }

    func testNothingIsDecidedBeforeTheRecordsLoad() {
        XCTAssertEqual(block(nil, .matchmaking, LichessBotChallengeSettings()), .recordsNotLoaded)
        XCTAssertEqual(LichessBotBotGameBudget.Block.recordsNotLoaded.reason, "DCM's game records have not loaded")
    }

    func testEachSenderStopsAtItsOwnAllowance() throws {
        let reserving = settings(incoming: 10, queue: 5)
        XCTAssertNil(block(window(games: 84), .matchmaking, reserving))
        guard case .allowanceUsed(let matchmaking)? = block(window(games: 85), .matchmaking, reserving) else { return XCTFail("85 uses matchmaking's allowance") }
        XCTAssertTrue(matchmaking.hasPrefix("DCM has used 85 of Lichess' 100 daily bot games (10 held for incoming challenges, 5 for the challenge queue); resumes about "), matchmaking)
        XCTAssertTrue(matchmaking.hasSuffix(" tomorrow"), matchmaking)
        XCTAssertNil(block(window(games: 89), .challengeQueue, reserving))
        guard case .allowanceUsed(let queue)? = block(window(games: 90), .challengeQueue, reserving) else { return XCTFail("90 uses the queue's allowance") }
        XCTAssertTrue(queue.hasPrefix("DCM has used 90 of Lichess' 100 daily bot games (10 held for incoming challenges); resumes about "), queue)
        XCTAssertNil(block(window(games: 99), .operatorChallenge, reserving))
        guard case .allowanceUsed(let manual)? = block(window(games: 100), .operatorChallenge, reserving) else { return XCTFail("100 uses the whole limit") }
        XCTAssertTrue(manual.hasPrefix("DCM has used 100 of Lichess' 100 daily bot games; resumes about "), manual)
    }

    func testGamesThatMayStartCountTowardTheReserves() {
        let reserving = settings(incoming: 10, queue: 5)
        XCTAssertNil(block(window(games: 83), prospective: 1, .matchmaking, reserving))
        guard case .awaitingGamesThatMayStart(let reason)? = block(window(games: 83), prospective: 2, .matchmaking, reserving) else { return XCTFail("83 + 2 reaches 85") }
        XCTAssertTrue(reason.hasPrefix("DCM has used 83 of Lichess' 100 daily bot games and 2 more may start (10 held for incoming challenges, 5 for the challenge queue); resumes about "), reason)
        XCTAssertTrue(reason.hasSuffix(" tomorrow, or sooner if enough of those don't start"), reason)
    }

    func testGamesThatMayStartAloneWaitForThemToResolve() {
        let small = settings(incoming: 60, queue: 35)
        XCTAssertEqual(block(window(games: 0), prospective: 5, .matchmaking, small),
                       .awaitingGamesThatMayStart(reason: "DCM has used 0 of Lichess' 100 daily bot games and 5 more may start (60 held for incoming challenges, 35 for the challenge queue); resumes as those resolve"))
        guard case .allowanceUsed(let both)? = block(window(games: 5), prospective: 5, .matchmaking, small) else { return XCTFail("counted games alone use it") }
        XCTAssertTrue(both.hasPrefix("DCM has used 5 of Lichess' 100 daily bot games and 5 more may start (60 held for incoming challenges, 35 for the challenge queue); resumes once those have resolved, not before about "), both)
    }

    /// Reserves taking the whole limit block matchmaking with nothing counted
    /// and no resume time.
    func testAZeroAllowanceBlocksWithNoGames() {
        XCTAssertEqual(block(LichessBotBotGameWindow(botGameStarts: []), .matchmaking, settings(incoming: 60, queue: 40)),
                       .allowanceUsed(reason: "DCM has used 0 of Lichess' 100 daily bot games (60 held for incoming challenges, 40 for the challenge queue)"))
    }

    /// A count past its clear time can't limit DCM, so it blocks nobody.
    func testACountPastItsClearTimeBlocksNothing() {
        let late = LichessBotBotGameWindow(botGameStarts: [t0, t0 + 23 * hour] + (0..<88).map { t0 + 25 * hour + Double($0) })
        XCTAssertEqual(late.entry?.count, 90)
        XCTAssertNil(block(late, .matchmaking, settings(incoming: 10, queue: 5), asOf: t0 + 26 * hour))
    }

    func testTheClearTimeSaysTomorrowWhenItIsOnTheNextDay() {
        // 2026-10-06 16:33:58 UTC; a day later is 2026-10-07 16:33:58 UTC.
        let first = Date(timeIntervalSince1970: 1_791_304_438)
        XCTAssertFalse(LichessBotBotGameBudget.clearTimeText(first + day, now: first + 20 * hour, calendar: utc).hasSuffix("tomorrow"))
        XCTAssertTrue(LichessBotBotGameBudget.clearTimeText(first + day, now: first + hour, calendar: utc).hasSuffix(" tomorrow"))
    }

    func testStatusText() {
        XCTAssertEqual(LichessBotBotGameBudget.statusText(window: nil, now: t0, calendar: utc), "DCM bot games: loading")
        XCTAssertEqual(LichessBotBotGameBudget.statusText(window: LichessBotBotGameWindow(botGameStarts: []), now: t0, calendar: utc), "DCM: 0/100 bot games")
        let text = LichessBotBotGameBudget.statusText(window: window(games: 3), now: t0 + hour, calendar: utc)
        XCTAssertTrue(text.hasPrefix("DCM: 3/100 bot games · clears "), text)
        XCTAssertTrue(text.hasSuffix(" tomorrow"), text)
    }

    // MARK: - Settings (unchanged)

    func testReserveValidation() {
        var valid = LichessBotSettings()
        valid.challenge.botGamesReservedForIncoming = 60
        valid.challenge.botGamesReservedForChallengeQueue = 40
        XCTAssertFalse(valid.validationProblems().contains { $0.hasPrefix("Reserved bot games") })
        var over = valid
        over.challenge.botGamesReservedForChallengeQueue = 41
        XCTAssertTrue(over.validationProblems().contains("Reserved bot games cannot exceed Lichess' 100 per day"))
        var negative = LichessBotSettings()
        negative.challenge.botGamesReservedForIncoming = -1
        XCTAssertTrue(negative.validationProblems().contains("Reserved bot games cannot be negative"))
    }

    func testReservesDefaultToNone() {
        let defaults = LichessBotChallengeSettings()
        XCTAssertEqual(defaults.botGamesReservedForIncoming, 0)
        XCTAssertEqual(defaults.botGamesReservedForChallengeQueue, 0)
    }
}
