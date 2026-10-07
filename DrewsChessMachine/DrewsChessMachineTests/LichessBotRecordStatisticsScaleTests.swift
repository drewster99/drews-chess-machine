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

    /// Synthetic rows shaped like the bot's real ones: several speeds, rated
    /// and casual, ratings spread over 600 points, 40 models with several
    /// checkpoints each, and full facts.
    private func rows(_ count: Int, now: Date) throws -> [LichessBotGameSummary] {
        let speeds = ["bullet", "blitz", "rapid", "classical"]
        return try (0..<count).map { index in
            let score: Double? = index % 37 == 0 ? nil : [1, 0.5, 0, 0][index % 4]
            let decisive: LichessBotDecisivePly
            switch score {
            case .some(1): decisive = .atPly(20 + index % 30)
            case .some(0): decisive = index % 5 == 0 ? .never : .atPly(30 + index % 20)
            default: decisive = .notApplicable
            }
            return try Fixtures.row(
                id: "s\(index)",
                at: now.addingTimeInterval(Double(-index) * 600),
                score: score,
                speed: speeds[index % speeds.count],
                rated: index % 9 != 0,
                color: index % 2 == 0 ? .white : .black,
                opponentRating: 1200 + (index * 7) % 600,
                ourRatingBefore: 1400 + index % 50,
                ourRatingDiff: index % 13 == 0 ? nil : (index % 21) - 10,
                plies: 40 + index % 60,
                facts: Fixtures.facts(
                    checkpoints: [
                        .init(moveNumber: 10, win: 0.5, draw: 0.3, loss: 0.2),
                        .init(moveNumber: 20, win: Float(index % 10) / 10, draw: 0.1, loss: 0.9 - Float(index % 10) / 10),
                    ],
                    buckets: [.init(index: index % 10, positions: 12, sumExpected: Double(index % 10) / 10 * 12)],
                    heldWinStartPly: index % 3 == 0 ? 18 : nil,
                    heldLossStartPly: index % 4 == 0 ? 22 : nil,
                    decisive: decisive,
                    generations: [Fixtures.generation("M\(index % 40)", step: (index / 40) % 25 * 1000, moves: 20)]
                )
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

    func testComputeScalesLinearly() throws {
        let now = Date(timeIntervalSince1970: 1_791_374_400)
        let small = try rows(2_500, now: now)
        let large = try rows(10_000, now: now)
        let smallSeconds = try bestOfThree(small, now: now)
        let largeSeconds = try bestOfThree(large, now: now)
        print("LICHESS-BOT-STATS-SCALE 2500 rows: \(smallSeconds) s, 10000 rows: \(largeSeconds) s, ratio \(largeSeconds / smallSeconds)")
        XCTAssertLessThan(largeSeconds / smallSeconds, 8)
        XCTAssertLessThan(largeSeconds, 10)
    }
}
