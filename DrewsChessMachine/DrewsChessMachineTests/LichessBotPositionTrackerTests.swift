import XCTest
@testable import DrewsChessMachine

/// `LichessBotPositionTracker` — position as a pure function of the
/// server's move list (Lichess bot plan §8.2, E1–E5, E30).
final class LichessBotPositionTrackerTests: XCTestCase {

    func testStartposAndStandardFENAreAccepted() throws {
        XCTAssertTrue(LichessBotPositionTracker.isStandardStart("startpos"))
        XCTAssertTrue(LichessBotPositionTracker.isStandardStart("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
        XCTAssertTrue(LichessBotPositionTracker.isStandardStart("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 9"), "move counters are ignored")
        XCTAssertFalse(LichessBotPositionTracker.isStandardStart("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"))
        XCTAssertFalse(LichessBotPositionTracker.isStandardStart("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"))
        XCTAssertNoThrow(try LichessBotPositionTracker(initialFen: "startpos"))
    }

    /// E3: a game from any other position is refused.
    func testNonStandardStartThrows() {
        XCTAssertThrowsError(try LichessBotPositionTracker(initialFen: "8/8/8/8/8/8/8/K6k w - - 0 1")) { error in
            XCTAssertEqual(error as? LichessBotPositionError, .unsupportedInitialPosition(fen: "8/8/8/8/8/8/8/K6k w - - 0 1"))
        }
    }

    /// E2 and E4: an empty list is zero moves; a longer list extends.
    func testEmptyThenExtending() throws {
        let tracker = try LichessBotPositionTracker(initialFen: "startpos")
        XCTAssertEqual(try tracker.sync(to: []), .unchanged)
        XCTAssertEqual(tracker.ply, 0)
        XCTAssertEqual(tracker.sideToMove, .white)
        XCTAssertEqual(try tracker.sync(to: ["e2e4"]), .extended(fromPly: 0, toPly: 1))
        XCTAssertEqual(try tracker.sync(to: ["e2e4", "e7e5", "g1f3"]), .extended(fromPly: 1, toPly: 3))
        XCTAssertEqual(try tracker.sync(to: ["e2e4", "e7e5", "g1f3"]), .unchanged)
        XCTAssertEqual(tracker.sideToMove, .black)
    }

    /// E4 and E30: a takeback shortens the list; the position is rebuilt.
    func testShorterListRebuilds() throws {
        let tracker = try LichessBotPositionTracker(initialFen: "startpos")
        try tracker.sync(to: ["e2e4", "e7e5", "g1f3"])
        XCTAssertEqual(try tracker.sync(to: ["e2e4"]), .rebuilt(fromPly: 3, toPly: 1))
        XCTAssertEqual(tracker.ply, 1)
        XCTAssertEqual(tracker.sideToMove, .black)
        XCTAssertEqual(tracker.engine.moveHistory.count, 1)
    }

    /// E5: a list that diverges from the local one is rebuilt, not patched.
    func testDivergentListRebuilds() throws {
        let tracker = try LichessBotPositionTracker(initialFen: "startpos")
        try tracker.sync(to: ["e2e4", "e7e5"])
        XCTAssertEqual(try tracker.sync(to: ["d2d4", "d7d5", "c2c4"]), .rebuilt(fromPly: 2, toPly: 3))
        XCTAssertEqual(tracker.tokens, ["d2d4", "d7d5", "c2c4"])
    }

    /// E1: king-to-rook castling tokens apply, and the raw tokens are kept
    /// exactly as given.
    func testKingToRookCastlingIsAppliedAndKeptAsGiven() throws {
        let tracker = try LichessBotPositionTracker(initialFen: "startpos")
        let tokens = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1h1"]
        try tracker.sync(to: tokens)
        XCTAssertEqual(tracker.tokens, tokens)
        XCTAssertEqual(tracker.moves.last?.uci, "e1g1")
        XCTAssertFalse(tracker.engine.state.whiteKingsideCastle)
    }

    func testIllegalTokenThrows() throws {
        let tracker = try LichessBotPositionTracker(initialFen: "startpos")
        XCTAssertThrowsError(try tracker.sync(to: ["e2e5"])) { error in
            guard case .illegalMove(let token, let ply, _)? = error as? LichessBotPositionError else {
                return XCTFail("expected illegalMove, got \(error)")
            }
            XCTAssertEqual(token, "e2e5")
            XCTAssertEqual(ply, 0)
        }
    }

    /// E8: the tracker's engine never ends a game on a draw rule, so play
    /// continues past a threefold if the server says so.
    func testPlaysPastAThreefold() throws {
        let tracker = try LichessBotPositionTracker(initialFen: "startpos")
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8", "e2e4"]
        try tracker.sync(to: shuffle)
        XCTAssertEqual(tracker.ply, shuffle.count)
        XCTAssertNil(tracker.engine.result)
    }
}
