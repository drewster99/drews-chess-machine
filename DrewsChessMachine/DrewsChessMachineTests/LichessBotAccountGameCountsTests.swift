import XCTest
@testable import DrewsChessMachine

/// The account grid's Unrated / Today / Last 24 h columns: Today is the
/// Record card's Today (`LichessBotStatsPeriods`), and live games not filed
/// yet count in Today and Last 24 h.
final class LichessBotAccountGameCountsTests: XCTestCase {

    private func calendar(_ identifier: String) throws -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(identifier: identifier))
        return calendar
    }

    private func date(_ text: String) throws -> Date {
        try XCTUnwrap(ISO8601DateFormatter().date(from: text), text)
    }

    func testLiveGamesCountInTodayAndTheLastDay() throws {
        let now = try date("2026-10-07T12:00:00Z")
        let rows = [
            try LichessBotStatsFixtures.row(id: "unratedToday", at: now.addingTimeInterval(-2 * 3600), score: 1, rated: false),
            try LichessBotStatsFixtures.row(id: "ratedYesterday", at: now.addingTimeInterval(-16 * 3600), score: 0),
            try LichessBotStatsFixtures.row(id: "unratedOld", at: now.addingTimeInterval(-30 * 3600), score: 0.5, speed: "bullet", rated: false),
        ]
        let live = LichessBotGameStart(startedAt: now.addingTimeInterval(-60), speed: "blitz", opponentID: "somebot", opponentIsBot: true)
        let speedNotKnownYet = LichessBotGameStart(startedAt: now.addingTimeInterval(-30), speed: nil, opponentID: nil, opponentIsBot: nil)
        let starts = rows.map { LichessBotGameStart(filed: $0) } + [live, speedNotKnownYet]
        let counts = try LichessBotAccountGameCounts(filedRows: rows, gameStarts: starts, now: now, calendar: calendar("UTC"))
        XCTAssertEqual(counts.unrated, ["blitz": 1, "bullet": 1])
        XCTAssertEqual(counts.today, ["blitz": 2], "this morning's filed game and the live one")
        XCTAssertEqual(counts.lastDay, ["blitz": 3], "yesterday evening's game too; the 30-hour-old one is out")
    }

    func testTodayIsTheRecordCardsToday() throws {
        let paris = try calendar("Europe/Paris")
        // 00:30 on 2026-10-07 in Paris (22:30 UTC the day before).
        let now = try date("2026-10-06T22:30:00Z")
        let beforeMidnight = LichessBotGameStart(startedAt: try date("2026-10-06T21:45:00Z"), speed: "rapid", opponentID: "a", opponentIsBot: false)
        let afterMidnight = LichessBotGameStart(startedAt: try date("2026-10-06T22:10:00Z"), speed: "rapid", opponentID: "b", opponentIsBot: false)
        let counts = try LichessBotAccountGameCounts(filedRows: [], gameStarts: [beforeMidnight, afterMidnight], now: now, calendar: paris)
        let periods = try LichessBotStatsPeriods.starts(now: now, calendar: paris)
        XCTAssertTrue(periods.contains(afterMidnight.startedAt, in: .today))
        XCTAssertFalse(periods.contains(beforeMidnight.startedAt, in: .today))
        XCTAssertEqual(counts.today, ["rapid": 1])
        XCTAssertEqual(counts.lastDay, ["rapid": 2])
    }

    func testReadingCells() throws {
        let now = try date("2026-10-07T12:00:00Z")
        let counts = try LichessBotAccountGameCounts(filedRows: [], gameStarts: [LichessBotGameStart(startedAt: now, speed: "blitz", opponentID: nil, opponentIsBot: false)], now: now, calendar: calendar("UTC"))
        let counted = LichessBotAccountGameCounts.Reading.counted(counts)
        XCTAssertEqual(counted.text(\.today, speed: "blitz"), "1")
        XCTAssertEqual(counted.text(\.today, speed: "rapid"), "0")
        XCTAssertFalse(counted.isFailed)
        XCTAssertEqual(LichessBotAccountGameCounts.Reading.loading.text(\.today, speed: "blitz"), "…")
        let failed = LichessBotAccountGameCounts.Reading.failed("no day")
        XCTAssertEqual(failed.text(\.unrated, speed: "blitz"), "!")
        XCTAssertEqual(failed.failureText, "Game counts unavailable: no day")
    }

    @MainActor
    func testAFiledGameStillInTheLiveListIsCountedOnce() throws {
        let now = try date("2026-10-07T12:00:00Z")
        let filed = try LichessBotStatsFixtures.row(id: "g1", at: now.addingTimeInterval(-600), score: 1)
        let stillListed = LichessBotLiveGame(id: "g1", startedAt: now.addingTimeInterval(-600), ourAccountID: "dcm")
        let starting = LichessBotLiveGame(id: "g2", startedAt: now.addingTimeInterval(-5), ourAccountID: "dcm")
        let starts = LichessBotGameStart.all(filedRows: [filed], liveGames: [stillListed, starting])
        XCTAssertEqual(starts, [
            LichessBotGameStart(filed: filed),
            LichessBotGameStart(startedAt: now.addingTimeInterval(-5), speed: nil, opponentID: nil, opponentIsBot: nil),
        ])
    }
}
