import XCTest
@testable import DrewsChessMachine

/// Lichess's per-game rating change wherever games are shown: one game's
/// cell (games lists, tiles, the game view), and the net change against an
/// opponent and per model. Every sum goes through `LichessBotRatingChange`,
/// whose classification is `LichessBotGameRatingChange`.
final class LichessBotRatingChangeDisplayTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    /// Wednesday 2026-10-07 12:00 UTC.
    private let now = Date(timeIntervalSince1970: 1_791_374_400)
    private let minus = "\u{2212}"

    // MARK: - One game

    /// Protects: a game's classification — the change (including 0),
    /// casual, rated without a diff, and no result, rated or not.
    /// Fails if: a casual or unscored game shows a number, a rated game
    /// without a diff reads as 0, or a 0 diff reads as missing.
    func testGameClassificationCoversEveryCase() throws {
        let cases: [(name: String, row: LichessBotGameSummary, expected: LichessBotGameRatingChange)] = [
            ("gain", try Fixtures.row(id: "a", at: now, score: 1, ourRatingDiff: 7), .change(7)),
            ("loss", try Fixtures.row(id: "b", at: now, score: 0, ourRatingDiff: -12), .change(-12)),
            ("zero", try Fixtures.row(id: "c", at: now, score: 0.5, ourRatingDiff: 0), .change(0)),
            ("rated without diff", try Fixtures.row(id: "d", at: now, score: 1, ourRatingDiff: nil), .ratedWithoutChange),
            ("casual", try Fixtures.row(id: "e", at: now, score: 1, rated: false, ourRatingDiff: nil), .casual),
            ("casual with a stray diff", try Fixtures.row(id: "f", at: now, score: 1, rated: false, ourRatingDiff: 5), .casual),
            ("rated aborted", try Fixtures.row(id: "g", at: now, score: nil, ourRatingDiff: nil, status: "aborted"), .noResult),
            ("casual aborted", try Fixtures.row(id: "h", at: now, score: nil, rated: false, status: "aborted"), .noResult),
        ]
        for testCase in cases {
            XCTAssertEqual(LichessBotGameRatingChange(testCase.row), testCase.expected, testCase.name)
        }
    }

    /// Protects: the cell text and its tooltip: a true minus, "0" for no
    /// change, "–" with a reason for every game without a number.
    /// Fails if: the cell uses a hyphen-minus, prints "0" for a casual or
    /// missing change, or the tooltips stop telling the cases apart.
    func testGameCellTextAndTooltip() {
        XCTAssertEqual(LichessBotStatsFormat.gameRatingChange(.change(7)), "+7")
        XCTAssertEqual(LichessBotStatsFormat.gameRatingChange(.change(-12)), "\(minus)12")
        XCTAssertEqual(LichessBotStatsFormat.gameRatingChange(.change(0)), "0")
        XCTAssertEqual(LichessBotStatsFormat.gameRatingChange(.casual), "–")
        XCTAssertEqual(LichessBotStatsFormat.gameRatingChange(.ratedWithoutChange), "–")
        XCTAssertEqual(LichessBotStatsFormat.gameRatingChange(.noResult), "–")

        XCTAssertEqual(LichessBotStatsFormat.gameRatingChangeHelp(.casual), "Casual: no rating change")
        XCTAssertEqual(LichessBotStatsFormat.gameRatingChangeHelp(.ratedWithoutChange), "Lichess reported no rating change for this rated game")
        let helps = [LichessBotGameRatingChange.change(1), .casual, .ratedWithoutChange, .noResult].map(LichessBotStatsFormat.gameRatingChangeHelp)
        XCTAssertEqual(Set(helps).count, helps.count, "each case has its own tooltip")
    }

    /// Protects: the cell's color follows the sign only.
    /// Fails if: 0 or a missing change takes the win or loss color.
    func testCellColorFollowsTheSign() {
        XCTAssertEqual(LichessBotStatsStyle.ratingChangeColor(points: 3), LichessBotStatsStyle.win)
        XCTAssertEqual(LichessBotStatsStyle.ratingChangeColor(points: -3), LichessBotStatsStyle.loss)
        XCTAssertEqual(LichessBotStatsStyle.ratingChangeColor(points: 0), LichessBotStatsStyle.neutral)
        XCTAssertEqual(LichessBotStatsStyle.ratingChangeColor(points: nil), LichessBotStatsStyle.neutral)
        XCTAssertNil(LichessBotGameRatingChange.casual.points)
        XCTAssertEqual(LichessBotGameRatingChange.change(0).points, 0)
    }

    // MARK: - Against one opponent

    /// Rows against "alice": +7, −12, rated without a diff, casual with a
    /// stray diff, and a rated abort; one against "bob".
    private func opponentRows() throws -> [LichessBotGameSummary] {
        [
            try Fixtures.row(id: "a1", at: now - 600, score: 1, opponentID: "alice", opponentName: "Alice", ourRatingDiff: 7),
            try Fixtures.row(id: "a2", at: now - 500, score: 0, opponentID: "Alice", opponentName: "Alice", ourRatingDiff: -12),
            try Fixtures.row(id: "a3", at: now - 400, score: 0.5, opponentID: "alice", opponentName: "Alice", ourRatingDiff: nil),
            try Fixtures.row(id: "a4", at: now - 300, score: 1, rated: false, opponentID: "alice", opponentName: "Alice", ourRatingDiff: 9),
            try Fixtures.row(id: "a5", at: now - 200, score: nil, opponentID: "alice", opponentName: "Alice", status: "aborted"),
            try Fixtures.row(id: "b1", at: now - 100, score: 1, opponentID: "bob", opponentName: "Bob", ourRatingDiff: 4),
        ]
    }

    private let aliceChange = LichessBotRatingChange(ratedGames: 3, gamesWithChange: 2, total: -5)

    /// Protects: the net change against an opponent, as the Challenge
    /// sheet's tables, the opponent card and the game view read it.
    /// Fails if: casual or aborted games enter the sum, a rated game
    /// without a diff is dropped from the count, or ids are not folded
    /// case-insensitively.
    func testHeadToHeadSumsRatedScoredGamesPerOpponent() throws {
        let byOpponent = LichessBotRecordSummary.byOpponent(rows: try opponentRows())
        let alice = try XCTUnwrap(byOpponent["alice"])
        XCTAssertEqual(alice.ratingChange, aliceChange)
        XCTAssertEqual(alice.record, LichessBotResultTally(wins: 2, draws: 1, losses: 1, unscored: 1))
        XCTAssertEqual(byOpponent["bob"]?.ratingChange, LichessBotRatingChange(ratedGames: 1, gamesWithChange: 1, total: 4))

        let past = LichessBotRecordSummary.pastOpponents(rows: try opponentRows())
        XCTAssertEqual(past.first { $0.id == "alice" }?.ratingChange, aliceChange)
    }

    /// Protects: the one-line head-to-head text and its count of games
    /// without a change.
    /// Fails if: the line drops the change or its "*" for a missing one.
    func testHeadToHeadText() throws {
        let alice = try XCTUnwrap(LichessBotRecordSummary.byOpponent(rows: try opponentRows())["alice"])
        XCTAssertEqual(LichessBotStatsFormat.headToHead(alice), "2–1–1 · Δ \(minus)5*")
        XCTAssertTrue(LichessBotStatsFormat.headToHeadHelp(alice).contains("2 of 3 rated games"))
        XCTAssertEqual(LichessBotStatsFormat.headToHead(LichessBotHeadToHead()), "0–0–0 · Δ –")
    }

    /// Protects: the Opponents pane's net change per opponent, scoped by
    /// the pane's period and filter like its W–D–L.
    /// Fails if: the change is summed over all time while the tally is
    /// scoped, or the casual game passes the rated filter's sum.
    func testOpponentsPaneChangeFollowsPeriodAndFilter() throws {
        var rows = try opponentRows()
        // Two days earlier: outside "today", inside all time.
        rows.append(try Fixtures.row(id: "a0", at: now - 2 * 86_400, score: 1, opponentID: "alice", opponentName: "Alice", ourRatingDiff: 20))
        let statistics = try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)

        let today = try XCTUnwrap(statistics[.all].byPeriod.today.later.opponents.mostPlayed.first { $0.id == "alice" })
        XCTAssertEqual(today.ratingChange, aliceChange)
        let allTime = try XCTUnwrap(statistics[.all].byPeriod.allTime.later.opponents.mostPlayed.first { $0.id == "alice" })
        XCTAssertEqual(allTime.ratingChange, LichessBotRatingChange(ratedGames: 4, gamesWithChange: 3, total: 15))

        let rated = try XCTUnwrap(statistics[.rated].byPeriod.today.later.opponents.mostPlayed.first { $0.id == "alice" })
        XCTAssertEqual(rated.ratingChange, aliceChange)
    }

    // MARK: - Per model

    /// Protects: the Models table's net change per model, per checkpoint
    /// row, and for the "No model recorded" row.
    /// Fails if: a model's line omits its games' changes, counts its
    /// casual game, or the no-model games' changes are lost.
    func testModelRowsCarryTheirNetChange() throws {
        let generation = { (step: Int) in Fixtures.facts(generations: [Fixtures.generation("M", step: step, moves: 9)]) }
        let rows = [
            try Fixtures.row(id: "m1", at: now - 600, score: 1, ourRatingDiff: 6, facts: generation(1000)),
            try Fixtures.row(id: "m2", at: now - 500, score: 0, ourRatingDiff: -10, facts: generation(2000)),
            try Fixtures.row(id: "m3", at: now - 400, score: 0.5, ourRatingDiff: nil, facts: generation(2000)),
            try Fixtures.row(id: "m4", at: now - 300, score: 1, rated: false, ourRatingDiff: 8, facts: generation(2000)),
            try Fixtures.row(id: "n1", at: now - 200, score: 1, ourRatingDiff: 3, facts: Fixtures.facts(ourMoveCount: 0, ourMovesWithDecision: 0)),
        ]
        let models = try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)[.all].byPeriod.allTime.models
        let table = LichessBotModelTableRow.rows(models, expanded: ["M"])
        XCTAssertEqual(table.map(\.kind), [.group(expanded: true), .checkpoint, .checkpoint, .noModelRecorded])
        XCTAssertEqual(table[0].ratingChange, LichessBotRatingChange(ratedGames: 3, gamesWithChange: 2, total: -4))
        XCTAssertEqual(table[1].ratingChange, LichessBotRatingChange(ratedGames: 1, gamesWithChange: 1, total: 6), "step 1000")
        XCTAssertEqual(table[2].ratingChange, LichessBotRatingChange(ratedGames: 2, gamesWithChange: 1, total: -10), "step 2000")
        XCTAssertEqual(table[3].ratingChange, LichessBotRatingChange(ratedGames: 1, gamesWithChange: 1, total: 3))
    }

    // MARK: - Sorting

    /// Protects: sorting by net change ranks a row without any recorded
    /// change below every row with one, a loss included.
    /// Fails if: a missing change sorts as 0, above a net loss.
    func testSortKeyPutsMissingChangesLast() {
        XCTAssertLessThan(LichessBotRatingChange(ratedGames: 2, gamesWithChange: 0, total: 0).sortKey,
                          LichessBotRatingChange(ratedGames: 1, gamesWithChange: 1, total: -400).sortKey)
        XCTAssertLessThan(LichessBotRatingChange().sortKey, LichessBotRatingChange(ratedGames: 1, gamesWithChange: 1, total: 0).sortKey)
    }
}
