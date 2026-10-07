import XCTest
@testable import DrewsChessMachine

/// D1 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §11): results by how the game
/// began, from the controller's origin resolver.
@MainActor
final class LichessBotOriginStatisticsTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private let now = Date(timeIntervalSince1970: 1_791_374_400)

    private func rows() throws -> [LichessBotGameSummary] {
        [
            try Fixtures.row(id: "a", at: now - 100, score: 1, opponentRating: 1600),
            try Fixtures.row(id: "b", at: now - 200, score: 0, opponentRating: 1700),
            try Fixtures.row(id: "c", at: now - 300, score: 0.5, opponentRating: 1500),
            try Fixtures.row(id: "d", at: now - 400, score: 1, opponentRating: 1500),
            try Fixtures.row(id: "e", at: now - 500, score: nil, status: "aborted"),
            try Fixtures.row(id: "f", at: now - 600, score: 0),
        ]
    }

    func testResultsByOriginWithUnknownAsItsOwnRow() throws {
        let origins: [String: LichessBotGameOriginCategory] = [
            "a": .matchmaking, "b": .matchmaking, "c": .incoming, "d": .unknown, "e": .matchmaking,
        ]
        let statistics = try LichessBotRecordStatistics.compute(rows: try rows(), origins: origins, now: now, calendar: Fixtures.utcCalendar)
        let breakdown = try XCTUnwrap(statistics[.all].byPeriod.allTime.later.origins)
        XCTAssertEqual(breakdown.rows.map(\.category), [.incoming, .matchmaking, .unknown], "the category's own order; unknown never folded into a known origin")
        XCTAssertEqual(breakdown.rows[1].tally, LichessBotResultTally(wins: 1, draws: 0, losses: 1, unscored: 0), "scored games only")
        guard case .estimate(let perf) = breakdown.rows[1].performance else { return XCTFail("expected an estimate") }
        XCTAssertEqual(perf, 1650, accuracy: LichessBotEloMath.solverTolerance)
        XCTAssertEqual(breakdown.gamesUnresolved, 1, "game f has no entry in the resolver")
    }

    func testWithoutOriginsThereIsNoBreakdown() throws {
        let statistics = try LichessBotRecordStatistics.compute(rows: try rows(), now: now, calendar: Fixtures.utcCalendar)
        XCTAssertNil(statistics[.all].byPeriod.allTime.later.origins)
    }

    func testThePipelinePassesTheResolversCategoriesAndRecomputesWhenTheyChange() async throws {
        let pipeline = LichessBotRecordStatisticsPipeline(defaults: try makeTemporaryDefaults(), calendar: { Fixtures.utcCalendar })
        addTeardownBlock { @MainActor in await pipeline.shutdown(reason: "test teardown") }
        // Origins before the first index: kept, nothing computed yet.
        pipeline.originsChanged(["a": .incoming])
        XCTAssertEqual(pipeline.requestCount, 0)
        pipeline.indexChanged(rows: try rows())
        await pipeline.latestComputation?.value
        guard case .ready(let first) = pipeline.state else { return XCTFail("expected a snapshot") }
        XCTAssertEqual(first[.all].byPeriod.allTime.later.origins?.rows.map(\.category), [.incoming])
        // A changed resolver recomputes; an unchanged one does not.
        pipeline.originsChanged(["a": .tournament, "b": .tournament])
        XCTAssertEqual(pipeline.requestCount, 2)
        pipeline.originsChanged(["a": .tournament, "b": .tournament])
        XCTAssertEqual(pipeline.requestCount, 2)
        await pipeline.latestComputation?.value
        guard case .ready(let second) = pipeline.state else { return XCTFail("expected a snapshot") }
        XCTAssertEqual(second[.all].byPeriod.allTime.later.origins?.rows.map(\.category), [.tournament])
    }
}
