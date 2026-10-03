import XCTest
@testable import DrewsChessMachine

/// `position` handling when DCM is the UCI engine: the GUI runs the game and
/// decides how it ends, so a move list that continues past an unclaimed draw
/// must be applied in full, and a list that cannot be applied must be rejected
/// as a whole rather than leaving a partial position behind.
final class UCIPositionTests: XCTestCase {

    private static let knightShuffle = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]

    func testAMoveListPastAnUnclaimedThreefoldIsAppliedInFull() throws {
        let engine = try UCIEngine.engine(forPositionArguments: ["startpos", "moves"] + Self.knightShuffle + ["e2e4", "e7e5"])
        // e4 (row 4, col 4) holds a white pawn, e5 (row 3, col 4) a black pawn,
        // and White is to move: all ten moves were applied.
        let e4 = try XCTUnwrap(engine.state.board[4 * 8 + 4])
        XCTAssertEqual(e4.type, .pawn)
        XCTAssertEqual(e4.color, .white)
        let e5 = try XCTUnwrap(engine.state.board[3 * 8 + 4])
        XCTAssertEqual(e5.type, .pawn)
        XCTAssertEqual(e5.color, .black)
        XCTAssertEqual(engine.state.currentPlayer, .white)
        XCTAssertFalse(engine.currentLegalMoves.isEmpty, "the GUI decides draws; the engine still has moves to offer")
    }

    func testAPositionAtAnUnclaimedThreefoldStillOffersMoves() throws {
        let engine = try UCIEngine.engine(forPositionArguments: ["startpos", "moves"] + Self.knightShuffle)
        XCTAssertNil(engine.result)
        XCTAssertFalse(engine.currentLegalMoves.isEmpty)
    }

    func testAnIllegalMoveRejectsTheWholePosition() {
        XCTAssertThrowsError(try UCIEngine.engine(forPositionArguments: ["startpos", "moves", "e2e4", "e7e5", "e4e6"])) { error in
            guard case UCIEngine.PositionError.moveNotLegal(let token, let ply) = error else {
                return XCTFail("expected moveNotLegal, got \(error)")
            }
            XCTAssertEqual(token, "e4e6")
            XCTAssertEqual(ply, 2)
        }
    }

    func testAMoveAfterCheckmateRejectsTheWholePosition() {
        // Fool's mate, then a further move: no move exists after mate.
        XCTAssertThrowsError(try UCIEngine.engine(forPositionArguments: ["startpos", "moves", "f2f3", "e7e5", "g2g4", "d8h4", "a2a3"])) { error in
            guard case UCIEngine.PositionError.moveNotLegal(let token, let ply) = error else {
                return XCTFail("expected moveNotLegal after mate, got \(error)")
            }
            XCTAssertEqual(token, "a2a3")
            XCTAssertEqual(ply, 4)
        }
    }

    func testMalformedCommandsAreRejected() {
        XCTAssertThrowsError(try UCIEngine.engine(forPositionArguments: []))
        XCTAssertThrowsError(try UCIEngine.engine(forPositionArguments: ["somewhere"]))
        XCTAssertThrowsError(try UCIEngine.engine(forPositionArguments: ["fen", "8/8/8"]))
        XCTAssertThrowsError(try UCIEngine.engine(forPositionArguments: ["startpos", "e2e4"]))
    }
}
