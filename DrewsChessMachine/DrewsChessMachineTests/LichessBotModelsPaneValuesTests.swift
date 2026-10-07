import XCTest
@testable import DrewsChessMachine

/// The Models pane's pure parts (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.8):
/// its visible rows, checkpoint labels and chart marks.
final class LichessBotModelsPaneValuesTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private let now = Date(timeIntervalSince1970: 1_791_374_400)

    private func models() throws -> LichessBotModelStatistics {
        let sha = String(repeating: "ef", count: 32)
        let rows = [
            try Fixtures.row(id: "a", at: now - 600, score: 1, facts: Fixtures.facts(generations: [Fixtures.generation("NEW", step: 2000, moves: 9)])),
            try Fixtures.row(id: "b", at: now - 700, score: 0, facts: Fixtures.facts(generations: [Fixtures.generation("NEW", step: 1000, moves: 9)])),
            try Fixtures.row(id: "c", at: now - 900, score: 0.5, facts: Fixtures.facts(generations: [Fixtures.generation("OLD", source: .file, step: 500, sha: sha, moves: 9)])),
            try Fixtures.row(id: "d", at: now - 950, score: 1, facts: Fixtures.facts(ourMoveCount: 0, ourMovesWithDecision: 0)),
        ]
        return try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)[.all].byPeriod.allTime.models
    }

    func testRowsExpandGroupsAndEndWithTheNoModelRow() throws {
        let models = try models()
        let collapsed = LichessBotModelTableRow.rows(models, expanded: [])
        XCTAssertEqual(collapsed.map(\.label), ["NEW", "OLD", "No model recorded"])
        XCTAssertEqual(collapsed.map(\.kind), [.group(expanded: false), .group(expanded: false), .noModelRecorded])
        XCTAssertEqual(collapsed.last?.tally, LichessBotResultTally(wins: 1, draws: 0, losses: 0, unscored: 0))
        XCTAssertNil(collapsed.last?.mixedGames)
        XCTAssertNotNil(collapsed.last?.interval)

        let expanded = LichessBotModelTableRow.rows(models, expanded: ["NEW"])
        XCTAssertEqual(expanded.map(\.kind), [.group(expanded: true), .checkpoint, .checkpoint, .group(expanded: false), .noModelRecorded])
        XCTAssertEqual(expanded[1].label, "step \(1000.formatted()) · \(LichessBotModelSourceKind.trainerSnapshot.displayName)")
        XCTAssertEqual(expanded[1].groupModelID, nil)
        XCTAssertEqual(expanded[0].groupModelID, "NEW")
        XCTAssertEqual(Set(expanded.map(\.id)).count, expanded.count, "row IDs are unique")

        let file = LichessBotModelTableRow.rows(models, expanded: ["OLD"])[2]
        XCTAssertEqual(file.label, "step 500 · \(LichessBotModelSourceKind.file.displayName) · efefefef")
    }

    func testNoModelRowOnlyWithGames() throws {
        let rows = [try Fixtures.row(id: "a", at: now - 600, score: 1, facts: Fixtures.facts(generations: [Fixtures.generation("M", step: 1, moves: 3)]))]
        let models = try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)[.all].byPeriod.allTime.models
        XCTAssertEqual(LichessBotModelTableRow.rows(models, expanded: []).map(\.kind), [.group(expanded: false)])
    }

    func testChartMarks() {
        let points = [
            LichessBotProgressionPoint(series: "R", step: 100, stepIsCumulative: false, firstStep: 50, checkpointCount: 2, games: 30, score: 0.5, interval: LichessBotScoreInterval(lower: 0.33, upper: 0.67), performance: .estimate(1501)),
            LichessBotProgressionPoint(series: "R", step: 200, stepIsCumulative: false, firstStep: 150, checkpointCount: 2, games: 30, score: 1, interval: LichessBotScoreInterval(lower: 0.88, upper: 1), performance: .atLeast(1900)),
        ]
        let score = LichessBotProgressChartMark.marks(points, metric: .score)
        XCTAssertEqual(score.map(\.value), [50, 100])
        XCTAssertEqual(score[0].lower ?? -1, 33, accuracy: 1e-9)
        XCTAssertEqual(score[0].upper ?? -1, 67, accuracy: 1e-9)
        // A bound has no value to place on the performance chart.
        let performance = LichessBotProgressChartMark.marks(points, metric: .performance)
        XCTAssertEqual(performance.map(\.value), [1501])
        XCTAssertNil(performance[0].lower)
    }
}
