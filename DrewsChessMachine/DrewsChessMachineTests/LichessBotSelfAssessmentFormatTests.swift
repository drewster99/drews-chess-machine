import XCTest
@testable import DrewsChessMachine

/// The Self-assessment pane's text (`LICHESS_BOT_RECORD_STATS_PLAN.md`
/// §3.6): decisive-ply lines, W / D / L triples and held counts.
final class LichessBotSelfAssessmentFormatTests: XCTestCase {

    func testDecisiveLine() {
        let summary = LichessBotDecisiveSummary(games: 46, meanPly: 36.52, medianPly: 25.5, meanLead: 29.43, never: 7, noData: 0)
        XCTAssertEqual(
            LichessBotStatsFormat.decisive(summary, result: "Won"),
            "Won: settled at ply 36.5 on average (median 25.5), 29.4 plies before the end; n 46, never 7, no data 0"
        )
        let none = LichessBotDecisiveSummary(games: 0, meanPly: nil, medianPly: nil, meanLead: nil, never: 3, noData: 2)
        XCTAssertEqual(LichessBotStatsFormat.decisive(none, result: "Lost"), "Lost: no game settled (never 3, no data 2)")
    }

    func testEndingCellsGiveTheShareOfTheirColumn() {
        XCTAssertEqual(LichessBotStatsFormat.countWithShare(42, of: 174), "42 (24.1%)")
        XCTAssertEqual(LichessBotStatsFormat.countWithShare(0, of: 174), "0")
        XCTAssertEqual(LichessBotStatsFormat.countWithShare(0, of: 0), "–", "an empty column has no share")
    }

    func testTriplesAndHeldCounts() {
        XCTAssertEqual(LichessBotStatsFormat.triple(LichessBotOutcomeTriple(win: 0.4123, draw: 0.2, loss: 0.3877)), "0.41/0.20/0.39")
        XCTAssertEqual(LichessBotStatsFormat.triple(nil), "–")
        XCTAssertEqual(LichessBotStatsFormat.held(turned: 0, held: 0, verb: "saved", noun: "held losses"), "No held losses")
        XCTAssertEqual(LichessBotStatsFormat.held(turned: 24, held: 132, verb: "saved", noun: "held losses"), "24 saved of 132 held losses (18.2%)")
    }
}
