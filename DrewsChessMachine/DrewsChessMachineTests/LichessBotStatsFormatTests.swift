import XCTest
@testable import DrewsChessMachine

/// The Record card's number formatting (`LICHESS_BOT_RECORD_STATS_PLAN.md`
/// §4.1): true minus signs, bounds, the missing-value dash, padding.
final class LichessBotStatsFormatTests: XCTestCase {

    func testSignedValuesUseATrueMinus() {
        XCTAssertEqual(LichessBotStatsFormat.signed(12), "+12")
        XCTAssertEqual(LichessBotStatsFormat.signed(-7), "\u{2212}7")
        XCTAssertEqual(LichessBotStatsFormat.signed(0), "0")
        XCTAssertEqual(LichessBotStatsFormat.signed(-36.6), "\u{2212}37")
        XCTAssertFalse(LichessBotStatsFormat.signed(-7).contains("-"), "never a hyphen-minus")
    }

    func testEstimates() {
        XCTAssertEqual(LichessBotStatsFormat.estimate(.none), "–")
        XCTAssertEqual(LichessBotStatsFormat.estimate(.estimate(1734.4)), "1734")
        XCTAssertEqual(LichessBotStatsFormat.estimate(.atLeast(1889.6)), "≥1890")
        XCTAssertEqual(LichessBotStatsFormat.estimate(.atMost(1100.2)), "≤1100")
        XCTAssertEqual(LichessBotStatsFormat.signedEstimate(.estimate(37.2)), "+37")
        XCTAssertEqual(LichessBotStatsFormat.signedEstimate(.atMost(-80)), "≤\u{2212}80")
        XCTAssertEqual(LichessBotStatsFormat.signedEstimate(.none), "–")
    }

    func testRatingChange() {
        XCTAssertEqual(LichessBotStatsFormat.ratingChange(LichessBotRatingChange()), "–")
        XCTAssertEqual(LichessBotStatsFormat.ratingChange(LichessBotRatingChange(ratedGames: 3, gamesWithChange: 0, total: 0)), "–*")
        XCTAssertEqual(LichessBotStatsFormat.ratingChange(LichessBotRatingChange(ratedGames: 3, gamesWithChange: 3, total: 0)), "0")
        XCTAssertEqual(LichessBotStatsFormat.ratingChange(LichessBotRatingChange(ratedGames: 3, gamesWithChange: 2, total: 15)), "+15*")
    }

    func testScoreIsAWholePercentThatNeverRoundsToPerfectOrZero() {
        XCTAssertEqual(LichessBotStatsFormat.score(nil), "–")
        XCTAssertEqual(LichessBotStatsFormat.score(0), "0%")
        XCTAssertEqual(LichessBotStatsFormat.score(1), "100%")
        XCTAssertEqual(LichessBotStatsFormat.score(0.306), "31%")
        XCTAssertEqual(LichessBotStatsFormat.score(0.302), "30%")
        XCTAssertEqual(LichessBotStatsFormat.score(0.625), "63%")
        XCTAssertEqual(LichessBotStatsFormat.score(0.004), "1%")
        XCTAssertEqual(LichessBotStatsFormat.score(0.996), "99%")
    }

    func testPercentagesIntervalsAndDecimals() {
        XCTAssertEqual(LichessBotStatsFormat.percent(nil), "–")
        XCTAssertEqual(LichessBotStatsFormat.percent(0.625), "62.5%")
        XCTAssertEqual(LichessBotStatsFormat.percent(1), "100.0%")
        XCTAssertEqual(LichessBotStatsFormat.interval(LichessBotScoreInterval(lower: 0.4129, upper: 0.8)), "41–80%")
        XCTAssertEqual(LichessBotStatsFormat.interval(nil), "–")
        XCTAssertEqual(LichessBotStatsFormat.decimal(0.12345), "0.123")
        XCTAssertEqual(LichessBotStatsFormat.decimal(-0.5), "\u{2212}0.500")
        XCTAssertEqual(LichessBotStatsFormat.decimal(nil), "–")
        XCTAssertEqual(LichessBotStatsFormat.average(1499.5), "1500")
        XCTAssertEqual(LichessBotStatsFormat.average(nil), "–")
    }

    func testPadding() {
        XCTAssertEqual(LichessBotStatsFormat.padded(7, width: 3), "  7")
        XCTAssertEqual(LichessBotStatsFormat.padded(1234, width: 3), "1234")
        XCTAssertEqual(LichessBotStatsFormat.padded("+5", width: 4), "  +5")
        let tally = LichessBotResultTally(wins: 12, draws: 3, losses: 101, unscored: 4)
        XCTAssertEqual(LichessBotStatsFormat.countWidth([tally]), 3)
        XCTAssertEqual(LichessBotStatsFormat.countWidth([]), 1)
        XCTAssertEqual(LichessBotStatsFormat.tally(tally, width: 3), " 12–  3–101")
    }
}
