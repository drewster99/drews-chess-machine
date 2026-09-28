import XCTest
@testable import DrewsChessMachine

/// `ChessGameEngine.drawCondition` when the side to move has no legal move.
///
/// Checkmate and stalemate take precedence over every draw rule: a mating or
/// stalemating move that also completes the fifty-move count ends the game in
/// mate or stalemate, and the engine must not also report a draw condition.
/// The Lichess bot logs any reported condition as a local-versus-server
/// disagreement and records it with the finished game.
final class ChessGameEngineDrawConditionPrecedenceTests: XCTestCase {

    private let modes: [ChessGameAdjudication] = [.automatic, .serverAuthoritative]

    private func play(_ engine: ChessGameEngine, _ token: String) throws {
        let move = try XCTUnwrap(ChessMove.parseUCI(token, legal: engine.currentLegalMoves), "illegal token \(token)")
        try engine.applyMoveAndAdvance(move)
    }

    func testCheckmateOnTheFiftyMoveBoundaryIsNotADraw() throws {
        for mode in modes {
            let engine = ChessGameEngine(state: try FENParser.parse("k7/8/1K6/8/8/8/8/7R w - - 99 80"), adjudication: mode)
            try play(engine, "h1h8")
            XCTAssertEqual(engine.state.halfmoveClock, 100, "\(mode)")
            XCTAssertEqual(engine.result, .checkmate(winner: .white), "\(mode)")
            XCTAssertNil(engine.drawCondition, "\(mode)")
        }
    }

    func testStalemateOnTheFiftyMoveBoundaryReportsNoDrawCondition() throws {
        for mode in modes {
            let engine = ChessGameEngine(state: try FENParser.parse("k7/8/1Q6/8/8/8/8/7K w - - 99 80"), adjudication: mode)
            try play(engine, "b6c7")
            XCTAssertEqual(engine.state.halfmoveClock, 100, "\(mode)")
            XCTAssertEqual(engine.result, .stalemate, "\(mode)")
            XCTAssertNil(engine.drawCondition, "\(mode)")
        }
    }

    func testMatedStartingPositionPastTheFiftyMoveCountReportsNoDrawCondition() throws {
        for mode in modes {
            let engine = ChessGameEngine(state: try FENParser.parse("R6k/6pp/8/8/8/8/8/7K b - - 100 80"), adjudication: mode)
            XCTAssertTrue(engine.currentLegalMoves.isEmpty, "\(mode)")
            XCTAssertNil(engine.drawCondition, "\(mode)")
        }
    }
}
