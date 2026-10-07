import XCTest
@testable import DrewsChessMachine

/// Period boundaries and the moment they next move
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.2), in several time zones, on a
/// DST day, and with both week starts.
final class LichessBotStatsPeriodsTests: XCTestCase {

    private func calendar(_ identifier: String, firstWeekday: Int = 2) throws -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(identifier: identifier))
        calendar.firstWeekday = firstWeekday
        return calendar
    }

    private func date(_ text: String) throws -> Date {
        try XCTUnwrap(ISO8601DateFormatter().date(from: text), text)
    }

    func testBoundariesInParis() throws {
        let calendar = try calendar("Europe/Paris")
        // Wednesday 2026-10-07 00:30 in Paris (22:30 UTC the day before).
        let now = try date("2026-10-06T22:30:00Z")
        let starts = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        XCTAssertEqual(starts.lastHour, try date("2026-10-06T21:30:00Z"))
        XCTAssertEqual(starts.today, try date("2026-10-06T22:00:00Z"))
        XCTAssertEqual(starts.thisWeek, try date("2026-10-04T22:00:00Z"), "Monday 2026-10-05 00:00 CEST")
        XCTAssertEqual(starts.thisMonth, try date("2026-09-30T22:00:00Z"))
        XCTAssertEqual(starts.thisYear, try date("2025-12-31T23:00:00Z"), "1 January in CET (UTC+1)")
        // The last hour reaches back over midnight: a game at 23:45 Paris
        // the day before is in the last hour but not today.
        let lateYesterday = try date("2026-10-06T21:45:00Z")
        XCTAssertTrue(starts.contains(lateYesterday, in: .lastHour))
        XCTAssertFalse(starts.contains(lateYesterday, in: .today))
    }

    func testChicagoDSTDayStartsAtLocalMidnight() throws {
        let calendar = try calendar("America/Chicago")
        // 2026-11-01 is the Sunday Chicago leaves DST (02:00 CDT → 01:00 CST).
        let now = try date("2026-11-01T18:00:00Z")
        let starts = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        XCTAssertEqual(starts.today, try date("2026-11-01T05:00:00Z"), "midnight CDT")
        // The day is 25 hours long: the next day starts at midnight CST.
        let next = try LichessBotStatsPeriods.nextChange(after: now, rows: [], calendar: calendar)
        XCTAssertEqual(next, try date("2026-11-02T06:00:00Z"))
    }

    func testAucklandIsAheadOfUTC() throws {
        let calendar = try calendar("Pacific/Auckland")
        // 2026-10-07 09:00 NZDT (UTC+13) = 2026-10-06 20:00 UTC.
        let now = try date("2026-10-06T20:00:00Z")
        let starts = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        XCTAssertEqual(starts.today, try date("2026-10-06T11:00:00Z"))
        XCTAssertEqual(starts.thisMonth, try date("2026-09-30T11:00:00Z"))
        XCTAssertEqual(starts.thisYear, try date("2025-12-31T11:00:00Z"))
    }

    func testWeekStartFollowsFirstWeekday() throws {
        // Wednesday 2026-10-07 12:00 UTC.
        let now = try date("2026-10-07T12:00:00Z")
        let sundayWeeks = try LichessBotStatsPeriods.starts(now: now, calendar: calendar("UTC", firstWeekday: 1))
        let mondayWeeks = try LichessBotStatsPeriods.starts(now: now, calendar: calendar("UTC", firstWeekday: 2))
        XCTAssertEqual(sundayWeeks.thisWeek, try date("2026-10-04T00:00:00Z"))
        XCTAssertEqual(mondayWeeks.thisWeek, try date("2026-10-05T00:00:00Z"))
    }

    func testNextChangePicksTheEarliestCandidate() throws {
        let calendar = try calendar("UTC")
        // Wednesday 2026-10-07 12:00 UTC.
        let now = try date("2026-10-07T12:00:00Z")
        // No games: the next midnight.
        XCTAssertEqual(try LichessBotStatsPeriods.nextChange(after: now, rows: [], calendar: calendar), try date("2026-10-08T00:00:00Z"))
        // A game 20 minutes ago leaves the last hour in 40 minutes; one 50
        // minutes ago leaves first; one two hours ago changes nothing later.
        let rows = [
            try LichessBotStatsFixtures.row(id: "a", at: now.addingTimeInterval(-20 * 60), score: 1),
            try LichessBotStatsFixtures.row(id: "b", at: now.addingTimeInterval(-50 * 60), score: 0),
            try LichessBotStatsFixtures.row(id: "c", at: now.addingTimeInterval(-120 * 60), score: 0.5),
        ]
        XCTAssertEqual(try LichessBotStatsPeriods.nextChange(after: now, rows: rows, calendar: calendar), now.addingTimeInterval(10 * 60))
        // Just before midnight on the last day of the year, the day, week
        // (Thursday 2026-12-31 → the week does not end), month and year
        // boundaries coincide.
        let newYearsEve = try date("2026-12-31T23:59:00Z")
        XCTAssertEqual(try LichessBotStatsPeriods.nextChange(after: newYearsEve, rows: [], calendar: calendar), try date("2027-01-01T00:00:00Z"))
        // On a Sunday evening with Monday weeks the week ends at the same
        // midnight as the day.
        let sunday = try date("2026-10-11T23:00:00Z")
        XCTAssertEqual(try LichessBotStatsPeriods.nextChange(after: sunday, rows: [], calendar: calendar), try date("2026-10-12T00:00:00Z"))
    }

    func testFutureGameCountsInEveryPeriodAndLeavesTheHourAnHourAfterItsStart() throws {
        let calendar = try calendar("UTC")
        let now = try date("2026-10-07T12:00:00Z")
        let future = now.addingTimeInterval(90)
        let starts = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        for period in LichessBotStatsPeriod.allCases {
            XCTAssertTrue(starts.contains(future, in: period), period.rawValue)
        }
        let row = try LichessBotStatsFixtures.row(id: "f", at: future, score: 1)
        XCTAssertEqual(try LichessBotStatsPeriods.nextChange(after: now, rows: [row], calendar: calendar), future.addingTimeInterval(3600))
    }

    func testAGameExactlyAnHourOldIsStillInTheLastHour() throws {
        let calendar = try calendar("UTC")
        let now = try date("2026-10-07T12:00:00Z")
        let edge = now.addingTimeInterval(-3600)
        let starts = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        XCTAssertTrue(starts.contains(edge, in: .lastHour))
        XCTAssertFalse(starts.contains(edge.addingTimeInterval(-0.001), in: .lastHour))
        // It leaves just after now, so the next check drops it.
        let row = try LichessBotStatsFixtures.row(id: "e", at: edge, score: 1)
        XCTAssertEqual(try LichessBotStatsPeriods.nextChange(after: now, rows: [row], calendar: calendar), now)
    }

    func testPeriodIdentifiersAreStable() {
        // Persisted in the defaults: renaming one silently resets the
        // operator's choice.
        XCTAssertEqual(LichessBotStatsPeriod.allCases.map(\.rawValue), ["lastHour", "today", "thisWeek", "thisMonth", "thisYear", "allTime"])
        XCTAssertEqual(LichessBotStatsPeriod.allCases.map(\.label), ["Last hour", "Today", "This week", "This month", "This year", "All time"])
        XCTAssertEqual(LichessBotStatsFilter.allCases.map(\.rawValue), ["all", "rated", "casual"])
    }

    func testRecordSummaryFillsEveryPeriod() throws {
        let calendar = try calendar("UTC")
        let now = try date("2026-10-07T12:00:00Z")
        let rows = [
            try LichessBotStatsFixtures.row(id: "hour", at: now.addingTimeInterval(-600), score: 1),
            try LichessBotStatsFixtures.row(id: "day", at: now.addingTimeInterval(-5 * 3600), score: 0),
            try LichessBotStatsFixtures.row(id: "week", at: try date("2026-10-05T08:00:00Z"), score: 0.5),
            try LichessBotStatsFixtures.row(id: "month", at: try date("2026-10-01T08:00:00Z"), score: 1),
            try LichessBotStatsFixtures.row(id: "year", at: try date("2026-02-01T08:00:00Z"), score: 1),
            try LichessBotStatsFixtures.row(id: "old", at: try date("2025-02-01T08:00:00Z"), score: nil),
        ]
        let records = try LichessBotRecordSummary.compute(rows: rows, now: now, calendar: calendar)
        XCTAssertEqual(records.lastHour.all.games, 1)
        XCTAssertEqual(records.today.all.games, 2)
        XCTAssertEqual(records.thisWeek.all.games, 3)
        XCTAssertEqual(records.thisMonth.all.games, 4)
        XCTAssertEqual(records.thisYear.all.games, 5)
        XCTAssertEqual(records.allTime.all.games, 6)
        XCTAssertEqual(records.allTime.all.unscored, 1)
        for (period, games) in zip(LichessBotStatsPeriod.allCases, [1, 2, 3, 4, 5, 6]) {
            XCTAssertEqual(records[period].all.games, games, period.rawValue)
        }
    }
}
