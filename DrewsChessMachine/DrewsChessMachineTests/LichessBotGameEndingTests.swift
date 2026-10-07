import XCTest
@testable import DrewsChessMachine

/// §3.9's ending classification: every row, and statuses this build does
/// not name, which are never folded into a known row.
final class LichessBotGameEndingTests: XCTestCase {

    private func classify(_ status: String, winner: String? = nil, score: Double? = 1, draw: ChessDrawCondition? = nil, known: Bool = true) -> LichessBotGameEnding {
        LichessBotGameEnding.classify(status: status, winner: winner, ourScore: score, localDrawCondition: draw, drawConditionKnown: known)
    }

    func testEveryRowOfTheTable() {
        XCTAssertEqual(classify("mate", winner: "white"), .checkmate)
        XCTAssertEqual(classify("resign", winner: "black", score: 0), .resignation)
        XCTAssertEqual(classify("outoftime", winner: "white"), .timeForfeit)
        XCTAssertEqual(classify("timeout", winner: "white"), .leftTheGame)
        XCTAssertEqual(classify("stalemate", score: 0.5), .stalemate)
        XCTAssertEqual(classify("draw", score: 0.5, draw: .threefoldRepetition), .threefoldRepetition)
        XCTAssertEqual(classify("draw", score: 0.5, draw: .fiftyMoveRule), .fiftyMoveRule)
        XCTAssertEqual(classify("draw", score: 0.5, draw: .insufficientMaterial), .insufficientMaterial)
        XCTAssertEqual(classify("insufficientMaterialClaim", score: 0.5), .insufficientMaterialClaim)
        XCTAssertEqual(classify("outoftime", score: 0.5), .timeoutVersusInsufficientMaterial)
        XCTAssertEqual(classify("draw", score: 0.5), .agreedOrOtherDraw)
        XCTAssertEqual(classify("aborted", score: nil), .notCounted)
        XCTAssertEqual(classify("noStart", score: nil), .notCounted)
    }

    func testStatusesTheTableDoesNotNameKeepTheirSpelling() {
        XCTAssertEqual(classify("timeout", score: 0.5), .other(status: "timeout"))
        XCTAssertEqual(classify("noStart", winner: "white"), .other(status: "noStart"))
        XCTAssertEqual(classify("cheat", winner: "white"), .other(status: "cheat"))
        XCTAssertEqual(classify("variantEnd", winner: "black", score: 0), .other(status: "variantEnd"))
        XCTAssertEqual(classify("someNewStatus", score: 0.5), .other(status: "someNewStatus"))
        XCTAssertEqual(classify("someNewStatus", winner: "white"), .other(status: "someNewStatus"))
        XCTAssertEqual(LichessBotGameEnding.other(status: "cheat").label, "Other: cheat")
    }

    func testDrawWithoutFactsIsNotCalledAnAgreement() {
        XCTAssertEqual(classify("draw", score: 0.5, known: false), .drawRuleNotRecorded)
        // Other statuses don't need the draw condition.
        XCTAssertEqual(classify("mate", winner: "white", known: false), .checkmate)
    }

    func testDisplayOrder() {
        let shuffled: [LichessBotGameEnding] = [.notCounted, .other(status: "zz"), .agreedOrOtherDraw, .other(status: "aa"), .checkmate, .stalemate, .resignation]
        XCTAssertEqual(shuffled.sorted(), [.checkmate, .resignation, .stalemate, .agreedOrOtherDraw, .other(status: "aa"), .other(status: "zz"), .notCounted])
    }
}
