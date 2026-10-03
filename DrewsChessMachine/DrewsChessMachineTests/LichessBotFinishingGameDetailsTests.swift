import XCTest
@testable import DrewsChessMachine

/// The finishing sheet's per-game text: numbers keep a fixed width so the
/// rows line up, and a value that isn't known yet is said in words rather
/// than shown as a made-up number.
final class LichessBotFinishingGameDetailsTests: XCTestCase {
    private let figureSpace = "\u{2007}"

    func testRatingsKeepTheirWidthAndAMissingOneIsSaid() {
        XCTAssertEqual(LichessBotFinishingGameDetails.ratingText(1500), "1500")
        XCTAssertEqual(LichessBotFinishingGameDetails.ratingText(950), figureSpace + "950")
        XCTAssertEqual(LichessBotFinishingGameDetails.ratingText(nil), "rating not shown")
    }

    func testMaterialLeadIsSignedAndKeepsItsWidth() {
        XCTAssertEqual(LichessBotFinishingGameDetails.signedPadded(3), figureSpace + "+3")
        XCTAssertEqual(LichessBotFinishingGameDetails.signedPadded(-12), "-12")
        XCTAssertEqual(LichessBotFinishingGameDetails.signedPadded(0), figureSpace + "\u{00B1}0")
    }

    func testPercentagesKeepTheirWidth() {
        XCTAssertEqual(LichessBotFinishingGameDetails.percent(0.05), figureSpace + figureSpace + "5%")
        XCTAssertEqual(LichessBotFinishingGameDetails.percent(1), "100%")
    }

    func testNoEstimateIsSaidInWords() {
        XCTAssertEqual(LichessBotFinishingGameDetails.estimateText(nil), "no estimate yet (DCM has not moved)")
    }
}
