import XCTest
@testable import DrewsChessMachine

/// Lichess' bot-vs-bot daily limit as lila enforces it — a fixed 24-hour
/// window opened by the first game, cleared all at once — and the reserves
/// matchmaking and the challenge queue leave short of it.
final class LichessBotBotGameWindowTests: XCTestCase {

    private let opened = Date(timeIntervalSince1970: 1_790_000_000)
    private let day = LichessBotBotGameWindow.length

    private var utc: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        return calendar
    }

    // MARK: - The window

    func testNoGamesMeansNoWindow() {
        let window = LichessBotBotGameWindow(botGameStarts: [], now: opened)
        XCTAssertNil(window.openedAt)
        XCTAssertNil(window.closesAt)
        XCTAssertEqual(window.gamesCounted, 0)
    }

    func testTheFirstGameOpensAWindowEveryLaterGameInItCounts() {
        let starts = [opened, opened.addingTimeInterval(3600), opened.addingTimeInterval(7200)]
        let window = LichessBotBotGameWindow(botGameStarts: starts, now: opened.addingTimeInterval(8000))
        XCTAssertEqual(window.openedAt, opened)
        XCTAssertEqual(window.closesAt, opened.addingTimeInterval(day))
        XCTAssertEqual(window.gamesCounted, 3)
    }

    /// Not a rolling day: an old game leaving the last 24 hours frees
    /// nothing; the whole count clears only when the window ends.
    func testTheCountClearsAllAtOnceWhenTheWindowEnds() {
        let starts = [opened, opened.addingTimeInterval(20 * 3600), opened.addingTimeInterval(23 * 3600)]
        let justBefore = LichessBotBotGameWindow(botGameStarts: starts, now: opened.addingTimeInterval(day))
        XCTAssertEqual(justBefore.gamesCounted, 3, "a rolling count would still be 3 here too; the point is the next line")
        let after = LichessBotBotGameWindow(botGameStarts: starts, now: opened.addingTimeInterval(day + 1))
        XCTAssertNil(after.openedAt)
        XCTAssertEqual(after.gamesCounted, 0, "a rolling day would still hold the two later games")
    }

    /// lila resets only for a game after the window's end
    /// (`nowMillis > clearAt`): a game exactly at the end still counts in
    /// the old window, one a second later opens a new one.
    func testAGameAtTheEndCountsInTheOldWindowAndOneAfterOpensANewOne() {
        let atEnd = LichessBotBotGameWindow(botGameStarts: [opened, opened.addingTimeInterval(day)], now: opened.addingTimeInterval(day))
        XCTAssertEqual(atEnd.openedAt, opened)
        XCTAssertEqual(atEnd.gamesCounted, 2)
        let second = opened.addingTimeInterval(day + 1)
        let reopened = LichessBotBotGameWindow(botGameStarts: [opened, second], now: second)
        XCTAssertEqual(reopened.openedAt, second)
        XCTAssertEqual(reopened.gamesCounted, 1)
        XCTAssertEqual(reopened.closesAt, second.addingTimeInterval(day))
    }

    func testStartOrderDoesNotMatter() {
        let starts = [opened.addingTimeInterval(7200), opened, opened.addingTimeInterval(3600)]
        let window = LichessBotBotGameWindow(botGameStarts: starts, now: opened.addingTimeInterval(8000))
        XCTAssertEqual(window.openedAt, opened)
        XCTAssertEqual(window.gamesCounted, 3)
    }

    /// The case that prompted this model: 102 games from 2026-10-06 16:33:58
    /// UTC; Lichess gives all 100 back at 16:33:58 the next day.
    func testAFullWindowClearsADayAfterItsFirstGame() {
        let first = Date(timeIntervalSince1970: 1_791_304_438)
        let starts = (0..<102).map { first.addingTimeInterval(Double($0) * 600) }
        let full = LichessBotBotGameWindow(botGameStarts: starts, now: first.addingTimeInterval(23 * 3600))
        XCTAssertEqual(full.gamesCounted, 102)
        XCTAssertEqual(full.closesAt, first.addingTimeInterval(day))
    }

    // MARK: - The reserves

    private func window(games: Int, asOf: Date) -> LichessBotBotGameWindow {
        LichessBotBotGameWindow(botGameStarts: (0..<games).map { opened.addingTimeInterval(Double($0)) }, now: asOf)
    }

    private func settings(incoming: Int, queue: Int) -> LichessBotChallengeSettings {
        var settings = LichessBotChallengeSettings()
        settings.botGamesReservedForIncoming = incoming
        settings.botGamesReservedForChallengeQueue = queue
        return settings
    }

    func testAllowances() {
        let reserving = settings(incoming: 10, queue: 5)
        XCTAssertEqual(LichessBotBotGameBudget.allowance(for: .matchmaking, settings: reserving), 85)
        XCTAssertEqual(LichessBotBotGameBudget.allowance(for: .challengeQueue, settings: reserving), 90)
        XCTAssertEqual(LichessBotBotGameBudget.allowance(for: .operatorChallenge, settings: reserving), 100)
        let none = LichessBotChallengeSettings()
        XCTAssertEqual(LichessBotBotGameBudget.allowance(for: .matchmaking, settings: none), LichessBotLimits.botGamesPerDay)
    }

    func testEachSenderStopsAtItsOwnAllowance() throws {
        let reserving = settings(incoming: 10, queue: 5)
        let asOf = opened.addingTimeInterval(3600)
        XCTAssertNil(LichessBotBotGameBudget.blockedReason(window: window(games: 84, asOf: asOf), sender: .matchmaking, settings: reserving, calendar: utc))
        let matchmaking = try XCTUnwrap(LichessBotBotGameBudget.blockedReason(window: window(games: 85, asOf: asOf), sender: .matchmaking, settings: reserving, calendar: utc))
        XCTAssertTrue(matchmaking.hasPrefix("DCM has played 85 of Lichess' 100 daily bot games (10 held for incoming challenges, 5 for the challenge queue); resumes about "), matchmaking)
        XCTAssertNil(LichessBotBotGameBudget.blockedReason(window: window(games: 89, asOf: asOf), sender: .challengeQueue, settings: reserving, calendar: utc))
        let queue = try XCTUnwrap(LichessBotBotGameBudget.blockedReason(window: window(games: 90, asOf: asOf), sender: .challengeQueue, settings: reserving, calendar: utc))
        XCTAssertTrue(queue.hasPrefix("DCM has played 90 of Lichess' 100 daily bot games (10 held for incoming challenges); resumes about "), queue)
        XCTAssertNil(LichessBotBotGameBudget.blockedReason(window: window(games: 99, asOf: asOf), sender: .operatorChallenge, settings: reserving, calendar: utc))
        let manual = try XCTUnwrap(LichessBotBotGameBudget.blockedReason(window: window(games: 100, asOf: asOf), sender: .operatorChallenge, settings: reserving, calendar: utc))
        XCTAssertTrue(manual.hasPrefix("DCM has played 100 of Lichess' 100 daily bot games; resumes about "), manual)
    }

    /// Reserves that take the whole limit block matchmaking even with no
    /// window open, and then there is no resume time to give.
    func testAZeroAllowanceBlocksWithoutAWindow() throws {
        let everything = settings(incoming: 60, queue: 40)
        let closed = LichessBotBotGameWindow(botGameStarts: [], now: opened)
        let reason = try XCTUnwrap(LichessBotBotGameBudget.blockedReason(window: closed, sender: .matchmaking, settings: everything, calendar: utc))
        XCTAssertEqual(reason, "DCM has played 0 of Lichess' 100 daily bot games (60 held for incoming challenges, 40 for the challenge queue)")
    }

    func testTheResumeTimeSaysTomorrowWhenTheWindowEndsAfterMidnight() throws {
        // Opened 2026-10-06 16:33:58 UTC: it ends 2026-10-07 16:33:58 UTC.
        let first = Date(timeIntervalSince1970: 1_791_304_438)
        let sameDay = LichessBotBotGameWindow(botGameStarts: [first], now: first.addingTimeInterval(20 * 3600))
        XCTAssertFalse(LichessBotBotGameBudget.resumeText(sameDay, calendar: utc).hasSuffix("tomorrow"))
        let dayBefore = LichessBotBotGameWindow(botGameStarts: [first], now: first.addingTimeInterval(3600))
        XCTAssertTrue(LichessBotBotGameBudget.resumeText(dayBefore, calendar: utc).hasSuffix("tomorrow"))
        XCTAssertEqual(LichessBotBotGameBudget.resumeText(LichessBotBotGameWindow(botGameStarts: [], now: first), calendar: utc), "")
    }

    // MARK: - Settings

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
