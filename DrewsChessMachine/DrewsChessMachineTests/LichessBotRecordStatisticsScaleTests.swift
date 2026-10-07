import XCTest
@testable import DrewsChessMachine

/// A guard against quadratic code in the statistics
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §7 P2). Tests run unoptimized, so an
/// absolute time says little about the shipped build; the ratio of two sizes
/// does not depend on the build's speed. Linear is about 4× from 2,500 to
/// 10,000 rows, quadratic about 16×; the bound is 8×, with a loose absolute
/// bound of 10 s at 10,000 rows.
final class LichessBotRecordStatisticsScaleTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private func bestOfThree(_ rows: [LichessBotGameSummary], now: Date) throws -> Double {
        var best = Double.infinity
        for _ in 0..<3 {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)
            best = min(best, Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000)
        }
        return best
    }

    func testComputeScalesLinearly() throws {
        let now = Date(timeIntervalSince1970: 1_791_374_400)
        let small = try Fixtures.syntheticRows(2_500, now: now)
        let large = try Fixtures.syntheticRows(10_000, now: now)
        let smallSeconds = try bestOfThree(small, now: now)
        let largeSeconds = try bestOfThree(large, now: now)
        print("LICHESS-BOT-STATS-SCALE 2500 rows: \(smallSeconds) s, 10000 rows: \(largeSeconds) s, ratio \(largeSeconds / smallSeconds)")
        XCTAssertLessThan(largeSeconds / smallSeconds, 8)
        XCTAssertLessThan(largeSeconds, 10)
    }
}
