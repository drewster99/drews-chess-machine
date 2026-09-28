import XCTest
@testable import DrewsChessMachine

/// `SANFormatter` notation details beyond `SANFormatterTests`: capture
/// markers for every piece kind, en passant for either side, the mate
/// suffix, discovered check, promotion with and without capture and check,
/// castling that gives check, and disambiguation: minimal qualifiers, the
/// full origin square when neither file nor rank alone suffices, a pinned
/// rival that does not count, and a rival of another piece type that does
/// not count.
///
/// The parser round-trip in `SANFormatterTests` cannot catch these: the
/// parser strips check and mate marks, ignores `x`, and accepts a longer
/// qualifier than needed, so each case here pins the one correct string.
final class SANFormatterExactNotationTests: XCTestCase {

    private func san(_ fen: String, _ token: String) throws -> String {
        let state = try FENParser.parse(fen)
        let legal = MoveGenerator.legalMoves(for: state)
        let move = try XCTUnwrap(ChessMove.parseUCI(token, legal: legal, state: state), "illegal token \(token)")
        return try SANFormatter.san(for: move, in: state, legalMoves: legal)
    }

    func testCaptureMarkers() throws {
        XCTAssertEqual(try san("4k3/8/8/8/8/8/r7/R3K3 w - - 0 1", "a1a2"), "Rxa2")
        XCTAssertEqual(try san("4k3/8/8/8/8/8/3p4/4K3 w - - 0 1", "e1d2"), "Kxd2")
        XCTAssertEqual(try san("4k3/8/8/8/8/3p4/4P3/4K3 w - - 0 1", "e2d3"), "exd3")
        XCTAssertEqual(try san("4k3/8/8/8/8/3p4/4P3/4K3 w - - 0 1", "e2e3"), "e3")
    }

    func testEnPassantForEitherSide() throws {
        XCTAssertEqual(try san("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3"), "exd3")
        XCTAssertEqual(try san("8/2k5/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"), "exd6+")
    }

    func testMateSuffixVersusCheckSuffix() throws {
        XCTAssertEqual(try san("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", "a1a8"), "Ra8#")
        XCTAssertEqual(try san("6k1/5p1p/8/8/8/8/8/R3K3 w - - 0 1", "a1a8"), "Ra8+", "the king can leave the back rank")
    }

    func testDiscoveredCheck() throws {
        XCTAssertEqual(try san("4k3/8/8/8/8/8/4N3/4R1K1 w - - 0 1", "e2c3"), "Nc3+")
    }

    func testPromotions() throws {
        XCTAssertEqual(try san("r6k/1P4pp/8/8/8/8/8/4K3 w - - 0 1", "b7a8q"), "bxa8=Q#")
        XCTAssertEqual(try san("r6k/1P4pp/8/8/8/8/8/4K3 w - - 0 1", "b7a8n"), "bxa8=N")
        XCTAssertEqual(try san("8/1P1k4/8/8/8/8/8/4K3 w - - 0 1", "b7b8n"), "b8=N+")
        XCTAssertEqual(try san("8/1P1k4/8/8/8/8/8/4K3 w - - 0 1", "b7b8q"), "b8=Q")
        XCTAssertEqual(try san("4k3/8/8/8/8/8/6p1/4K2R b - - 0 1", "g2h1q"), "gxh1=Q+")
    }

    func testCastling() throws {
        XCTAssertEqual(try san("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", "e8g8"), "O-O")
        XCTAssertEqual(try san("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", "e8c8"), "O-O-O")
        XCTAssertEqual(try san("5k2/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1"), "O-O+")
        XCTAssertEqual(try san("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", "e1c1"), "O-O-O+")
    }

    func testDisambiguation() throws {
        let rooks = "4k3/8/8/8/8/8/8/R4RK1 w - - 0 1"
        XCTAssertEqual(try san(rooks, "a1d1"), "Rad1")
        XCTAssertEqual(try san(rooks, "f1d1"), "Rfd1")

        let knightCaptures = "4k3/8/8/8/8/5N2/3p4/1N5K w - - 0 1"
        XCTAssertEqual(try san(knightCaptures, "b1d2"), "Nbxd2")
        XCTAssertEqual(try san(knightCaptures, "f3d2"), "Nfxd2")

        let sameFile = "4k3/8/8/6N1/8/8/8/4K1N1 w - - 0 1"
        XCTAssertEqual(try san(sameFile, "g1f3"), "N1f3")
        XCTAssertEqual(try san(sameFile, "g5f3"), "N5f3")

        let threeKnights = "4k3/8/8/8/8/1N6/8/1N2KN2 w - - 0 1"
        XCTAssertEqual(try san(threeKnights, "b1d2"), "Nb1d2")
        XCTAssertEqual(try san(threeKnights, "f1d2"), "Nfd2")
        XCTAssertEqual(try san(threeKnights, "b3d2"), "N3d2")
    }

    func testPinnedRivalNeedsNoDisambiguation() throws {
        let fen = "4k3/4r3/8/8/8/8/4N3/1N2K3 w - - 0 1"
        let state = try FENParser.parse(fen)
        XCTAssertNotNil(ChessMove.parseUCI("e2c3", legal: MoveGenerator.pseudoLegalMoves(for: state)), "the pinned knight reaches the square pseudo-legally")
        XCTAssertNil(ChessMove.parseUCI("e2c3", legal: MoveGenerator.legalMoves(for: state)), "but not legally")
        XCTAssertEqual(try san(fen, "b1c3"), "Nc3")
    }

    func testRivalOfAnotherPieceTypeNeedsNoDisambiguation() throws {
        let fen = "4k3/8/8/8/8/3Q4/8/R3K3 w - - 0 1"
        XCTAssertEqual(try san(fen, "a1d1"), "Rd1")
        XCTAssertEqual(try san(fen, "d3d1"), "Qd1")
    }
}
