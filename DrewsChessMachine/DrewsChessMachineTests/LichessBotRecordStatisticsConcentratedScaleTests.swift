import XCTest
@testable import DrewsChessMachine

/// The statistics stay linear when every game lands in the same groups —
/// one model, one speed, every period (§7 P2's scale guard, sharpened).
///
/// `LichessBotRecordStatisticsScaleTests` spreads its rows over 40 models
/// and many days, so per-group arrays stay small and a per-game copy of
/// them barely shows: it passed (4.2×, under its 8× bound) while the
/// accumulators were being copied out and back for every game. Here every
/// row is one model's, inside the last hour, so every per-group array grows
/// with the whole input, and the size range is wider (1,000 to 16,000 rows:
/// 16× linear, 256× quadratic).
final class LichessBotRecordStatisticsConcentratedScaleTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private func rows(_ count: Int, now: Date) throws -> [LichessBotGameSummary] {
        try (0..<count).map { index in
            try Fixtures.row(
                id: "c\(index)",
                at: now.addingTimeInterval(-Double(index % 3_000) / 10),
                score: [1, 0.5, 0][index % 3],
                speed: "blitz",
                opponentRating: 1300 + index % 400,
                ourRatingBefore: 1500,
                ourRatingDiff: index % 2 == 0 ? 5 : -5,
                facts: Fixtures.facts(generations: [Fixtures.generation("ONE", step: 1000, moves: 20)])
            )
        }
    }

    private func bestOfThree(_ rows: [LichessBotGameSummary], now: Date) throws -> Double {
        var best = Double.infinity
        for _ in 0..<3 {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)
            best = min(best, Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000)
        }
        return best
    }

    func testOneModelInTheLastHourScalesLinearly() throws {
        let now = Date(timeIntervalSince1970: 1_791_374_400)
        let small = try bestOfThree(try rows(1_000, now: now), now: now)
        let large = try bestOfThree(try rows(16_000, now: now), now: now)
        print("LICHESS-BOT-STATS-CONCENTRATED-SCALE 1000 rows: \(small) s, 16000 rows: \(large) s, ratio \(large / small)")
        XCTAssertLessThan(large / small, 48, "linear is about 16×, quadratic about 256×")
        XCTAssertLessThan(large, 10)
    }
}
