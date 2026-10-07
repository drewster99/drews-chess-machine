import XCTest
@testable import DrewsChessMachine

/// The Time controls pane's pure parts (`LICHESS_BOT_RECORD_STATS_PLAN.md`
/// §3.5): its rows, the account rating text and the sparkline series.
final class LichessBotTimeControlStatsTests: XCTestCase {

    private func perf(_ rating: Int?, provisional: Bool? = nil) -> LichessBotPerfRating {
        LichessBotPerfRating(games: 10, rating: rating, rd: 60, prog: 0, prov: provisional)
    }

    func testRowsComeFromRecordsAndTheAccountsSpeedsOnly() {
        // Account pools that are not time controls never become rows.
        XCTAssertEqual(
            LichessBotTimeControlOrder.speeds(recordSpeeds: [], accountSpeeds: ["puzzle", "chess960", "storm", "classical", "bullet"]),
            ["bullet", "classical"]
        )
        // Lichess's order, fastest first, then unknown speeds verbatim.
        XCTAssertEqual(
            LichessBotTimeControlOrder.speeds(recordSpeeds: ["rapid", "zeta", "ultraBullet", "alpha"], accountSpeeds: ["correspondence"]),
            ["ultraBullet", "rapid", "correspondence", "alpha", "zeta"]
        )
        XCTAssertEqual(LichessBotTimeControlOrder.speeds(recordSpeeds: [], accountSpeeds: []), [])
    }

    func testAccountRatingText() {
        XCTAssertEqual(LichessBotStatsFormat.accountRating(perf(1512)), "1512")
        XCTAssertEqual(LichessBotStatsFormat.accountRating(perf(1512, provisional: true)), "1512?")
        XCTAssertEqual(LichessBotStatsFormat.accountRating(perf(1512, provisional: false)), "1512")
        XCTAssertEqual(LichessBotStatsFormat.accountRating(perf(nil)), "–")
        XCTAssertEqual(LichessBotStatsFormat.accountRating(nil), "–", "account not loaded, or no rating in this pool")
    }

    func testSparklineSeriesEndsWithTheCurrentRatingOnlyWhenKnown() {
        let points = [
            LichessBotRatingPoint(gameID: "a", createdAt: Date(timeIntervalSince1970: 1_759_000_000), rating: 1500),
            LichessBotRatingPoint(gameID: "b", createdAt: Date(timeIntervalSince1970: 1_759_100_000), rating: 1480),
        ]
        let withAccount = LichessBotRatingSparklineSeries(points: points, currentRating: 1466)
        XCTAssertEqual(withAccount.ratings, [1500, 1480, 1466])
        XCTAssertTrue(withAccount.help.hasPrefix("1466–1500 over 2 rated games"), withAccount.help)
        XCTAssertTrue(withAccount.help.hasSuffix("the last point is the current rating"), withAccount.help)
        let withoutAccount = LichessBotRatingSparklineSeries(points: points, currentRating: nil)
        XCTAssertEqual(withoutAccount.ratings, [1500, 1480])
        XCTAssertFalse(withoutAccount.help.contains("current rating"))
        let empty = LichessBotRatingSparklineSeries(points: [], currentRating: nil)
        XCTAssertEqual(empty.ratings, [])
        XCTAssertEqual(empty.help, "No rated game with a starting rating")
    }
}
