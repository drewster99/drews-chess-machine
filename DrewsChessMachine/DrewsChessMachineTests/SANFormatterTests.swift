import XCTest
@testable import DrewsChessMachine

/// `SANFormatter` — the SAN writer the Lichess bot's PGN records and game
/// detail view use (plan §8.4).
///
/// Validated two ways: specific cases with known SAN, and a round-trip of
/// every legal move in the perft positions (and the positions one ply
/// deeper) through the existing, independently written SAN *parser*,
/// `PGNImporter.resolveLegalSANMove`. That parser rejects ambiguous tokens,
/// so the round-trip also checks that disambiguation is never too short.
final class SANFormatterTests: XCTestCase {

    private func san(_ fen: String, _ token: String) throws -> String {
        let state = try FENParser.parse(fen)
        let legal = MoveGenerator.legalMoves(for: state)
        let move = try XCTUnwrap(ChessMove.parseUCI(token, legal: legal, state: state), "illegal token \(token)")
        return try SANFormatter.san(for: move, in: state, legalMoves: legal)
    }

    private let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    func testPawnAndPieceMoves() throws {
        XCTAssertEqual(try san(start, "e2e4"), "e4")
        XCTAssertEqual(try san(start, "g1f3"), "Nf3")
    }

    func testCastling() throws {
        let fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        XCTAssertEqual(try san(fen, "e1g1"), "O-O")
        XCTAssertEqual(try san(fen, "e1c1"), "O-O-O")
        XCTAssertEqual(try san(fen, "e1h1"), "O-O", "king-to-rook alias formats as castling too")
    }

    func testCaptureAndEnPassant() throws {
        XCTAssertEqual(try san("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"), "exd6")
        XCTAssertEqual(try san("4k3/8/8/3p4/4N3/8/8/4K3 w - - 0 1", "e4d6"), "Nd6+")
        XCTAssertEqual(try san("4k3/8/8/3p4/4N3/8/8/4K3 w - - 0 1", "e4f6"), "Nf6+")
        XCTAssertEqual(try san("4k3/8/8/2p5/4N3/8/8/4K3 w - - 0 1", "e4c5"), "Nxc5")
    }

    func testPromotionAndCapturePromotion() throws {
        XCTAssertEqual(try san("7k/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q"), "b8=Q+")
        XCTAssertEqual(try san("7k/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8n"), "b8=N")
        XCTAssertEqual(try san("r6k/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7a8r"), "bxa8=R+")
    }

    func testDisambiguationByFileByRankAndByBoth() throws {
        let fileCase = "4k3/8/8/8/8/8/8/1N1K1N2 w - - 0 1"
        XCTAssertEqual(try san(fileCase, "b1d2"), "Nbd2")
        XCTAssertEqual(try san(fileCase, "f1d2"), "Nfd2")

        let rankCase = "4k3/8/8/R7/8/8/8/R3K3 w - - 0 1"
        XCTAssertEqual(try san(rankCase, "a1a3"), "R1a3")
        XCTAssertEqual(try san(rankCase, "a5a3"), "R5a3")

        let bothCase = "4k3/8/8/8/8/Q7/8/Q1Q1K3 w - - 0 1"
        XCTAssertEqual(try san(bothCase, "a1b2"), "Qa1b2")
        XCTAssertEqual(try san(bothCase, "a3b2"), "Q3b2")
        XCTAssertEqual(try san(bothCase, "c1b2"), "Qcb2")
    }

    func testCheckmateSuffix() throws {
        XCTAssertEqual(try san("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", "d8h4"), "Qh4#")
    }

    func testIllegalMoveThrows() throws {
        let state = try FENParser.parse(start)
        let illegal = ChessMove(fromRow: 6, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)  // e2e5
        XCTAssertThrowsError(try SANFormatter.san(for: illegal, in: state))
    }

    /// Every legal move, in every perft position and every position one ply
    /// deeper, formats to SAN that the importer's parser resolves back to the
    /// same move.
    func testEveryLegalMoveRoundTripsThroughTheImporterParser() throws {
        let fens = [
            start,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        ]
        var checked = 0
        for fen in fens {
            let root = try FENParser.parse(fen)
            var positions = [root]
            for move in MoveGenerator.legalMoves(for: root) {
                positions.append(MoveGenerator.applyMove(move, to: root))
            }
            for state in positions {
                let legal = MoveGenerator.legalMoves(for: state)
                for move in legal {
                    let text = try SANFormatter.san(for: move, in: state, legalMoves: legal)
                    XCTAssertEqual(PGNImporter.resolveLegalSANMove(text, state: state), move,
                                   "\(text) (\(move.uci)) in \(FENParser.fen(from: state))")
                    checked += 1
                }
            }
        }
        XCTAssertGreaterThan(checked, 5_000, "the round-trip should cover thousands of moves")
    }
}
