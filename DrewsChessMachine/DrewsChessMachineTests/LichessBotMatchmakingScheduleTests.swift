import XCTest
@testable import DrewsChessMachine

/// Matchmaking's time-of-day cap (`LichessBotMatchmakingSchedule`): the peak
/// window in local hours, its wraparound, DST, and the rate limiter under it.
final class LichessBotMatchmakingScheduleTests: XCTestCase {

    private var chicago: Calendar {
        get throws {
            var calendar = Calendar(identifier: .gregorian)
            calendar.timeZone = try XCTUnwrap(TimeZone(identifier: "America/Chicago"))
            return calendar
        }
    }

    /// 2026-10-14 08:00Z = 03:00 CDT.
    private let threeAMLocal = Date(timeIntervalSince1970: 1_791_964_800)
    /// 2026-10-14 15:00Z = 10:00 CDT.
    private let tenAMLocal = Date(timeIntervalSince1970: 1_791_990_000)
    /// 2026-10-14 17:00Z = 12:00 CDT.
    private let noonLocal = Date(timeIntervalSince1970: 1_791_997_200)
    /// 2026-10-15 07:00Z = 02:00 CDT.
    private let nextTwoAMLocal = Date(timeIntervalSince1970: 1_792_047_600)

    private func settings(cap: Int = 10, start: Int = 2, end: Int = 12, offPeak: Int = 40) -> LichessBotMatchmakingSettings {
        var settings = LichessBotMatchmakingSettings()
        settings.maxChallengesPerHour = cap
        settings.peakStartHour = start
        settings.peakEndHour = end
        settings.offPeakRatePercent = offPeak
        return settings
    }

    func testDefaultsArePeakTwoToNoonAtFortyPercentOffPeak() {
        let defaults = LichessBotMatchmakingSettings()
        XCTAssertEqual(defaults.peakStartHour, 2)
        XCTAssertEqual(defaults.peakEndHour, 12)
        XCTAssertEqual(defaults.offPeakRatePercent, 40)
    }

    func testInsideAndOutsideTheWindow() throws {
        let calendar = try chicago
        let atThree = LichessBotMatchmakingSchedule.rate(settings: settings(), now: threeAMLocal, calendar: calendar)
        XCTAssertEqual(atThree, LichessBotMatchmakingSchedule.Rate(hourlyCap: 10, isPeak: true, percent: 100, changesAt: noonLocal))
        XCTAssertEqual(LichessBotMatchmakingSchedule.rate(settings: settings(), now: tenAMLocal, calendar: calendar).hourlyCap, 10)
        let atNoon = LichessBotMatchmakingSchedule.rate(settings: settings(), now: noonLocal, calendar: calendar)
        XCTAssertEqual(atNoon, LichessBotMatchmakingSchedule.Rate(hourlyCap: 4, isPeak: false, percent: 40, changesAt: nextTwoAMLocal), "the end hour is outside the window")
        XCTAssertEqual(atNoon.text, "off-peak 40%: 4/h")
        XCTAssertEqual(atThree.text, "peak 10/h")
    }

    func testWindowMembershipIncludingWraparound() {
        XCTAssertTrue(LichessBotMatchmakingSchedule.isPeak(hour: 2, start: 2, end: 12))
        XCTAssertTrue(LichessBotMatchmakingSchedule.isPeak(hour: 11, start: 2, end: 12))
        XCTAssertFalse(LichessBotMatchmakingSchedule.isPeak(hour: 12, start: 2, end: 12))
        XCTAssertFalse(LichessBotMatchmakingSchedule.isPeak(hour: 1, start: 2, end: 12))
        // 22:00–06:00 wraps past midnight.
        for hour in [22, 23, 0, 3, 5] {
            XCTAssertTrue(LichessBotMatchmakingSchedule.isPeak(hour: hour, start: 22, end: 6), "\(hour)")
        }
        for hour in [6, 12, 21] {
            XCTAssertFalse(LichessBotMatchmakingSchedule.isPeak(hour: hour, start: 22, end: 6), "\(hour)")
        }
        for hour in 0..<24 {
            XCTAssertTrue(LichessBotMatchmakingSchedule.isPeak(hour: hour, start: 7, end: 7), "equal hours: every hour is peak")
        }
    }

    func testWraparoundRateAndNextChange() throws {
        let calendar = try chicago
        // 03:00 local in a 22:00–06:00 window: peak, closing at 06:00 local (11:00Z).
        let rate = LichessBotMatchmakingSchedule.rate(settings: settings(start: 22, end: 6), now: threeAMLocal, calendar: calendar)
        XCTAssertTrue(rate.isPeak)
        XCTAssertEqual(rate.changesAt, threeAMLocal.addingTimeInterval(3 * 3600))
        XCTAssertNil(LichessBotMatchmakingSchedule.rate(settings: settings(start: 5, end: 5), now: threeAMLocal, calendar: calendar).changesAt)
    }

    func testScaledCapRoundsToTheNearestChallenge() {
        XCTAssertEqual(LichessBotMatchmakingSchedule.scaledCap(10, percent: 40), 4)
        XCTAssertEqual(LichessBotMatchmakingSchedule.scaledCap(6, percent: 40), 2, "2.4 rounds down")
        XCTAssertEqual(LichessBotMatchmakingSchedule.scaledCap(5, percent: 50), 3, "2.5 rounds up")
        XCTAssertEqual(LichessBotMatchmakingSchedule.scaledCap(1, percent: 40), 1, "any rate above 0% sends at least one an hour")
        XCTAssertEqual(LichessBotMatchmakingSchedule.scaledCap(1, percent: 1), 1)
        XCTAssertEqual(LichessBotMatchmakingSchedule.scaledCap(10, percent: 100), 10)
        XCTAssertEqual(LichessBotMatchmakingSchedule.scaledCap(10, percent: 0), 0)
    }

    /// Spring forward, 2026-03-08: 02:00 CST does not exist; a window
    /// starting at 02:00 opens when the clock reads 03:00 CDT.
    func testSpringForwardSkippedStartHourOpensAtTheNextExistingTime() throws {
        let calendar = try chicago
        let oneThirtyCST = Date(timeIntervalSince1970: 1_772_955_000) // 07:30Z
        let threeCDT = Date(timeIntervalSince1970: 1_772_956_800) // 08:00Z
        let before = LichessBotMatchmakingSchedule.rate(settings: settings(), now: oneThirtyCST, calendar: calendar)
        XCTAssertFalse(before.isPeak)
        XCTAssertEqual(before.changesAt, threeCDT)
        XCTAssertTrue(LichessBotMatchmakingSchedule.rate(settings: settings(), now: threeCDT, calendar: calendar).isPeak)
    }

    /// Fall back, 2026-11-01: 01:30 happens twice (CDT, then CST); both are
    /// local hour 1.
    func testFallBackRepeatedHourIsTheSameLocalHourTwice() throws {
        let calendar = try chicago
        let first = Date(timeIntervalSince1970: 1_793_514_600) // 06:30Z, 01:30 CDT
        let second = Date(timeIntervalSince1970: 1_793_518_200) // 07:30Z, 01:30 CST
        for instant in [first, second] {
            XCTAssertEqual(calendar.component(.hour, from: instant), 1)
            XCTAssertTrue(LichessBotMatchmakingSchedule.rate(settings: settings(start: 1, end: 2), now: instant, calendar: calendar).isPeak)
            XCTAssertFalse(LichessBotMatchmakingSchedule.rate(settings: settings(), now: instant, calendar: calendar).isPeak)
        }
    }

    // MARK: - The rate limiter under the schedule

    func testOffPeakCapStopsSendsSoonerThanThePeakCap() {
        var limiter = LichessBotMatchmakingRateLimiter()
        let start = threeAMLocal
        for minute in 0..<4 {
            limiter.recordAttempt(at: start.addingTimeInterval(Double(minute) * 120), reachedLichess: true)
        }
        let now = start.addingTimeInterval(1000)
        let offPeak = LichessBotMatchmakingSchedule.Rate(hourlyCap: 4, isPeak: false, percent: 40, changesAt: nil)
        let peak = LichessBotMatchmakingSchedule.Rate(hourlyCap: 10, isPeak: true, percent: 100, changesAt: nil)
        XCTAssertEqual(limiter.decision(now: now, rate: offPeak, minimumSpacing: 30), .hourlyCap(until: start.addingTimeInterval(3600), count: 4))
        XCTAssertEqual(limiter.decision(now: now, rate: peak, minimumSpacing: 30), .allowed)
    }

    func testZeroOffPeakRatePausesUntilThePeakWindow() {
        let limiter = LichessBotMatchmakingRateLimiter()
        let paused = LichessBotMatchmakingSchedule.Rate(hourlyCap: 0, isPeak: false, percent: 0, changesAt: nextTwoAMLocal)
        XCTAssertEqual(limiter.decision(now: noonLocal, rate: paused, minimumSpacing: 30), .offPeakPaused(peakStartsAt: nextTwoAMLocal))
        XCTAssertEqual(LichessBotController.nextAutomaticPass(after: .rateLimited(.offPeakPaused(peakStartsAt: nextTwoAMLocal)), now: noonLocal), noonLocal.addingTimeInterval(LichessBotMatchmaking.retryAfterUnproductivePass))
    }

    func testOutOfRangeWindowSettingsAreInvalid() {
        var draft = LichessBotSettings.testBaseline()
        XCTAssertEqual(draft.validationProblems(), [])
        draft.matchmaking.peakStartHour = 24
        XCTAssertFalse(draft.validationProblems().isEmpty)
        draft.matchmaking.peakStartHour = 22
        draft.matchmaking.offPeakRatePercent = 101
        XCTAssertFalse(draft.validationProblems().isEmpty)
        draft.matchmaking.offPeakRatePercent = 0
        XCTAssertEqual(draft.validationProblems(), [])
    }
}
