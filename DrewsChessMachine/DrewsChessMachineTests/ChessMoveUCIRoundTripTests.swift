import XCTest
@testable import DrewsChessMachine

/// Round-trip and alias tests for the UCI move helpers in
/// `ChessMove+UCI.swift`.
///
/// Before these tests, nothing covered `parseUCI` / `uci` for castling, en
/// passant or (under)promotion. The Lichess bot depends on all three, and
/// on the king-to-rook castling alias the Lichess Bot API's move lists may
/// use (plan E1, E6, E7).
final class ChessMoveUCIRoundTripTests: XCTestCase {

    private static let perftFENs: [String] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    ]

    private func legalMoves(_ fen: String) throws -> (GameState, [ChessMove]) {
        let state = try FENParser.parse(fen)
        return (state, MoveGenerator.legalMoves(for: state))
    }

    /// Every legal move's `uci` string parses back to the same move through
    /// both parsers. E6, E7.
    func testEveryLegalMoveRoundTripsThroughBothParsers() throws {
        for fen in Self.perftFENs {
            let (state, legal) = try legalMoves(fen)
            for move in legal {
                let token = move.uci
                XCTAssertEqual(ChessMove.parseUCI(token, legal: legal), move, "plain parser, \(token) in \(fen)")
                XCTAssertEqual(ChessMove.parseUCI(token, legal: legal, state: state), move, "state-aware parser, \(token) in \(fen)")
            }
        }
    }

    func testStandardCastlingTokensForBothColors() throws {
        let (white, whiteLegal) = try legalMoves("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        let whiteShort = try XCTUnwrap(ChessMove.parseUCI("e1g1", legal: whiteLegal, state: white))
        let whiteLong = try XCTUnwrap(ChessMove.parseUCI("e1c1", legal: whiteLegal, state: white))
        XCTAssertEqual(whiteShort.uci, "e1g1")
        XCTAssertEqual(whiteLong.uci, "e1c1")

        let (black, blackLegal) = try legalMoves("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
        XCTAssertEqual(ChessMove.parseUCI("e8g8", legal: blackLegal, state: black)?.uci, "e8g8")
        XCTAssertEqual(ChessMove.parseUCI("e8c8", legal: blackLegal, state: black)?.uci, "e8c8")
    }

    /// The king-to-rook form resolves to the same internal castling move as
    /// the standard form. E1.
    func testKingToRookCastlingAliasResolvesForBothColorsAndSides() throws {
        let (white, whiteLegal) = try legalMoves("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        XCTAssertEqual(ChessMove.parseUCI("e1h1", legal: whiteLegal, state: white),
                       ChessMove.parseUCI("e1g1", legal: whiteLegal))
        XCTAssertEqual(ChessMove.parseUCI("e1a1", legal: whiteLegal, state: white),
                       ChessMove.parseUCI("e1c1", legal: whiteLegal))

        let (black, blackLegal) = try legalMoves("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
        XCTAssertEqual(ChessMove.parseUCI("e8h8", legal: blackLegal, state: black),
                       ChessMove.parseUCI("e8g8", legal: blackLegal))
        XCTAssertEqual(ChessMove.parseUCI("e8a8", legal: blackLegal, state: black),
                       ChessMove.parseUCI("e8c8", legal: blackLegal))
    }

    /// The plain parser keeps its existing behavior: the king-to-rook form is
    /// not a move to it. `--uci` depends on that being unchanged.
    func testPlainParserStillRejectsKingToRookForm() throws {
        let (_, legal) = try legalMoves("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        XCTAssertNil(ChessMove.parseUCI("e1h1", legal: legal))
        XCTAssertNil(ChessMove.parseUCI("e1a1", legal: legal))
    }

    /// A rook on e1 travelling toward h1 is not castling, even though a legal
    /// e1→g1 rook move exists. The legal list alone cannot tell the two
    /// apart; the alias must check that the piece on e1 is a king. E1.
    func testAliasDoesNotTurnARookMoveIntoCastling() throws {
        let (state, legal) = try legalMoves("k7/8/8/8/8/8/8/K3R2R w - - 0 1")
        XCTAssertNotNil(ChessMove.parseUCI("e1g1", legal: legal), "precondition: a rook e1→g1 move exists")
        XCTAssertNil(ChessMove.parseUCI("e1h1", legal: legal, state: state))
    }

    /// With no castling right the alias finds no castling move and yields nil.
    func testAliasRequiresTheCastlingMoveToBeLegal() throws {
        let (state, legal) = try legalMoves("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
        XCTAssertNil(ChessMove.parseUCI("e1h1", legal: legal, state: state))
    }

    func testEnPassantRoundTrip() throws {
        let (state, legal) = try legalMoves("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        let move = try XCTUnwrap(ChessMove.parseUCI("e5d6", legal: legal, state: state))
        XCTAssertEqual(move.uci, "e5d6")
        XCTAssertNil(state.board[2 * 8 + 3], "precondition: d6 is empty, so e5d6 is en passant")
    }

    func testPromotionAndUnderpromotionRoundTripInEitherCase() throws {
        let (state, legal) = try legalMoves("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
        for (token, piece) in [("b7b8q", PieceType.queen), ("b7b8r", .rook), ("b7b8b", .bishop), ("b7b8n", .knight)] {
            let lower = try XCTUnwrap(ChessMove.parseUCI(token, legal: legal, state: state), token)
            XCTAssertEqual(lower.promotion, piece)
            XCTAssertEqual(lower.uci, token)
            XCTAssertEqual(ChessMove.parseUCI(token.uppercased(), legal: legal, state: state), lower,
                           "uppercase promotion letter, \(token)")
        }
        XCTAssertNil(ChessMove.parseUCI("b7b8", legal: legal, state: state), "a promotion needs its piece letter")
    }

    func testMalformedTokensAreRejected() throws {
        let (state, legal) = try legalMoves("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        for token in ["", "e2", "e2e", "e2e4e5", "i2i4", "e0e4", "e2e9", "e7e8x"] {
            XCTAssertNil(ChessMove.parseUCI(token, legal: legal, state: state), "token \(token)")
        }
    }
}
