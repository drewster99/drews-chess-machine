import XCTest
@testable import DrewsChessMachine

/// `ChessGameEngine` adjudication modes (Lichess bot plan §8.2, E8, E10,
/// E12).
///
/// `.automatic` is every existing caller's behavior: the engine ends the
/// game on mate, stalemate, the fifty-move rule, threefold repetition and
/// insufficient material. `.serverAuthoritative` ends it only when the side
/// to move has no legal move, and reports draw conditions through
/// `drawCondition` without acting on them — so a local rule-definition
/// difference can never stop the Lichess bot from moving while the server
/// still considers the game live.
final class ChessGameEngineAdjudicationTests: XCTestCase {

    private func play(_ engine: ChessGameEngine, _ tokens: [String]) throws {
        for token in tokens {
            let move = try XCTUnwrap(ChessMove.parseUCI(token, legal: engine.currentLegalMoves), "illegal token \(token)")
            try engine.applyMoveAndAdvance(move)
        }
    }

    /// Knight shuffle from the start position: the start position recurs
    /// after plies 4 and 8, so ply 8 makes it a threefold.
    private let knightShuffle = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]

    func testAutomaticIsTheDefault() {
        XCTAssertEqual(ChessGameEngine().adjudication, .automatic)
    }

    func testAutomaticEndsOnThreefold() throws {
        let engine = ChessGameEngine()
        try play(engine, Array(knightShuffle.dropLast()))
        XCTAssertNil(engine.result, "no threefold before ply 8")
        try play(engine, [knightShuffle[7]])
        XCTAssertEqual(engine.result, .drawByThreefoldRepetition)
        XCTAssertEqual(engine.drawCondition, .threefoldRepetition)
        let next = try XCTUnwrap(ChessMove.parseUCI("g1f3", legal: engine.currentLegalMoves))
        XCTAssertThrowsError(try engine.applyMoveAndAdvance(next))
    }

    func testServerAuthoritativeReportsThreefoldButPlaysOn() throws {
        let engine = ChessGameEngine(adjudication: .serverAuthoritative)
        try play(engine, knightShuffle)
        XCTAssertNil(engine.result)
        XCTAssertEqual(engine.drawCondition, .threefoldRepetition)
        try play(engine, ["g1f3"])
        XCTAssertNil(engine.result)
    }

    func testAutomaticEndsOnFiftyMoveRule() throws {
        let engine = ChessGameEngine(state: try FENParser.parse("k7/8/8/8/8/8/8/4K2R w - - 99 60"))
        try play(engine, ["h1h2"])
        XCTAssertEqual(engine.result, .drawByFiftyMoveRule)
    }

    func testServerAuthoritativeReportsFiftyMoveRuleButPlaysOn() throws {
        let engine = ChessGameEngine(
            state: try FENParser.parse("k7/8/8/8/8/8/8/4K2R w - - 99 60"),
            adjudication: .serverAuthoritative
        )
        try play(engine, ["h1h2"])
        XCTAssertNil(engine.result)
        XCTAssertEqual(engine.drawCondition, .fiftyMoveRule)
        try play(engine, ["a8b8"])
        XCTAssertNil(engine.result)
    }

    func testAutomaticEndsOnInsufficientMaterial() throws {
        let engine = ChessGameEngine(state: try FENParser.parse("4k3/8/8/8/8/8/3q4/4K3 w - - 0 1"))
        try play(engine, ["e1d2"])
        XCTAssertEqual(engine.result, .drawByInsufficientMaterial)
    }

    func testServerAuthoritativeReportsInsufficientMaterialButPlaysOn() throws {
        let engine = ChessGameEngine(
            state: try FENParser.parse("4k3/8/8/8/8/8/3q4/4K3 w - - 0 1"),
            adjudication: .serverAuthoritative
        )
        try play(engine, ["e1d2"])
        XCTAssertNil(engine.result)
        XCTAssertEqual(engine.drawCondition, .insufficientMaterial)
        try play(engine, ["e8e7"])
        XCTAssertNil(engine.result)
    }

    /// No legal move means no move to make, so checkmate ends the game in
    /// both modes. E12.
    func testCheckmateEndsTheGameInBothModes() throws {
        for mode in [ChessGameAdjudication.automatic, .serverAuthoritative] {
            let engine = ChessGameEngine(adjudication: mode)
            try play(engine, ["f2f3", "e7e5", "g2g4", "d8h4"])
            XCTAssertEqual(engine.result, .checkmate(winner: .black), "\(mode)")
            XCTAssertTrue(engine.currentLegalMoves.isEmpty)
        }
    }

    func testStalemateEndsTheGameInBothModes() throws {
        for mode in [ChessGameAdjudication.automatic, .serverAuthoritative] {
            let engine = ChessGameEngine(
                state: try FENParser.parse("k7/8/1Q6/8/8/8/8/7K w - - 0 1"),
                adjudication: mode
            )
            try play(engine, ["b6c7"])
            XCTAssertEqual(engine.result, .stalemate, "\(mode)")
        }
    }

    func testNoDrawConditionInAnOrdinaryPosition() throws {
        let engine = ChessGameEngine(adjudication: .serverAuthoritative)
        try play(engine, ["e2e4", "e7e5", "g1f3"])
        XCTAssertNil(engine.drawCondition)
        XCTAssertNil(engine.result)
    }
}
