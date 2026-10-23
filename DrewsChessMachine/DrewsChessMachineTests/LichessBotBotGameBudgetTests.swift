import XCTest
@testable import DrewsChessMachine

/// DCM's side of Lichess' daily bot-vs-bot limit after the owner's decision
/// of 2026-10-23: the limit is advisory, only Lichess's refusal pauses bot
/// games, and DCM's count is its own count of today's bot games.
final class LichessBotBotGameBudgetTests: XCTestCase {

    /// 2026-10-05 08:00:00 UTC.
    private let t0 = Date(timeIntervalSince1970: 1_791_187_200)
    private let hour: TimeInterval = 3600
    private let day: TimeInterval = 24 * 3600

    private var utc: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        return calendar
    }

    private func start(_ at: Date, speed: String? = "blitz", bot: Bool?) -> LichessBotGameStart {
        LichessBotGameStart(startedAt: at, speed: speed, opponentID: "opponent", opponentIsBot: bot)
    }

    private func counts(_ starts: [LichessBotGameStart], now: Date) throws -> LichessBotAccountGameCounts {
        try LichessBotAccountGameCounts(filedRows: [], gameStarts: starts, now: now, calendar: utc)
    }

    // MARK: - DCM's count of today's bot games

    /// Protects: the count is DCM's own count of bot games started today,
    /// with no cap at Lichess's 100 and no window of Lichess's.
    /// Fails if: it stops at the limit, counts humans or yesterday's games,
    /// or drops a bot game whose speed isn't known yet.
    func testBotGamesTodayCountsEveryBotGameStartedTodayAndNothingElse() throws {
        let midnight = utc.startOfDay(for: t0)
        let now = midnight + 23 * hour
        var starts = (0..<150).map { start(midnight + Double($0) * 60, bot: true) }
        starts.append(start(midnight + hour, speed: nil, bot: true))
        starts.append(start(midnight + 2 * hour, bot: false))
        starts.append(start(midnight + 3 * hour, bot: nil))
        starts.append(start(midnight - 1, bot: true))
        XCTAssertEqual(try counts(starts, now: now).botGamesToday, 151)
    }

    /// Protects: "today" is the calendar day, the Record card's Today, not
    /// the last 24 hours. Fails if: a game from a minute before midnight
    /// still counts just after it.
    func testBotGamesTodayStartsOverAtMidnight() throws {
        let midnight = utc.startOfDay(for: t0) + day
        let lateYesterday = start(midnight - 60, bot: true)
        XCTAssertEqual(try counts([lateYesterday], now: midnight - 30).botGamesToday, 1, "still today before midnight")
        XCTAssertEqual(try counts([lateYesterday, start(midnight + 60, bot: true)], now: midnight + 120).botGamesToday, 1)
    }

    // MARK: - Lichess's refusal is the only stop

    /// Protects: no pause without Lichess's report, however many bot games
    /// DCM has played. Fails if: a block returns with no report in force.
    func testNoBlockWithoutALichessReport() {
        XCTAssertNil(LichessBotBotGameBudget.block(ownLimitUntil: nil, now: t0, calendar: utc))
        XCTAssertNil(LichessBotBotGameBudget.block(ownLimitUntil: t0 - 1, now: t0, calendar: utc), "a report that has ended")
    }

    // MARK: - Text

    func testTheClearTimeSaysTomorrowWhenItIsOnTheNextDay() {
        // 2026-10-06 16:33:58 UTC; a day later is 2026-10-07 16:33:58 UTC.
        let first = Date(timeIntervalSince1970: 1_791_304_438)
        XCTAssertFalse(LichessBotBotGameBudget.clearTimeText(first + day, now: first + 20 * hour, calendar: utc).hasSuffix("tomorrow"))
        XCTAssertTrue(LichessBotBotGameBudget.clearTimeText(first + day, now: first + hour, calendar: utc).hasSuffix(" tomorrow"))
    }

    func testStatusTextShowsDCMsCountAndLichesssPause() throws {
        XCTAssertEqual(LichessBotBotGameBudget.statusText(counts: .loading, ownLimitUntil: nil, now: t0, calendar: utc), "DCM bot games today: loading")
        XCTAssertEqual(LichessBotBotGameBudget.statusText(counts: .failed("no day"), ownLimitUntil: nil, now: t0, calendar: utc), "DCM bot games today: unavailable")
        let counted = LichessBotAccountGameCounts.Reading.counted(try counts((0..<118).map { start(t0 - Double($0), bot: true) }, now: t0))
        XCTAssertEqual(LichessBotBotGameBudget.statusText(counts: counted, ownLimitUntil: nil, now: t0, calendar: utc), "DCM: 118 bot games today")
        let until = t0 + 2 * hour
        XCTAssertEqual(LichessBotBotGameBudget.statusText(counts: counted, ownLimitUntil: until, now: t0, calendar: utc),
                       "DCM: 118 bot games today · paused by Lichess until \(LichessBotBotGameBudget.clearTimeText(until, now: t0, calendar: utc))")
        XCTAssertEqual(LichessBotBotGameBudget.statusText(counts: counted, ownLimitUntil: until, now: until, calendar: utc), "DCM: 118 bot games today", "the pause has ended")
    }

    // MARK: - Settings saved before the reserves were removed

    /// Protects: a save holding the removed reserve settings still loads,
    /// keeps everything else, and reports the two keys as ignored.
    /// Fails if: the old keys break the load or are silently dropped.
    func testASaveWithTheRemovedReservesLoads() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings.testBaseline()
        settings.challenge.maxConcurrentGames = 7
        try LichessBotSettingsStore.save(settings, to: defaults)
        let saved = try XCTUnwrap(defaults.data(forKey: LichessBotSettingsStore.defaultsKey))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: saved) as? [String: Any])
        var challenge = try XCTUnwrap(object["challenge"] as? [String: Any])
        challenge["botGamesReservedForIncoming"] = 10
        challenge["botGamesReservedForChallengeQueue"] = 5
        object["challenge"] = challenge
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: LichessBotSettingsStore.defaultsKey)
        let loaded = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(loaded.settings, settings)
        XCTAssertEqual(Set(loaded.ignoredSavedKeys), ["challenge.botGamesReservedForIncoming", "challenge.botGamesReservedForChallengeQueue"])
    }
}
