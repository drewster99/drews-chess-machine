import XCTest
@testable import DrewsChessMachine

final class PGNImporterTests: XCTestCase {

    func testSANTokenizationStripsNumbersCommentsVariationsResult() {
        let movetext = "1. e4 e5 2. Nf3 {a comment} Nc6 (2... d6 3. d4) 3. Bb5 $1 a6 1-0"
        let tokens = PGNImporter.sanTokens(from: movetext)
        XCTAssertEqual(tokens, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
    }

    func testMatchPawnAndKnightFromStart() throws {
        let engine = ChessGameEngine()
        let legal = MoveGenerator.legalMoves(for: engine.state)

        // e4: e2 (row 6,col 4) -> e4 (row 4,col 4) in the row-0-is-rank-8 frame.
        let e4 = try XCTUnwrap(PGNImporter.matchSAN("e4", state: engine.state, legal: legal))
        XCTAssertEqual([e4.fromRow, e4.fromCol, e4.toRow, e4.toCol], [6, 4, 4, 4])
        XCTAssertNil(e4.promotion)

        // Nf3: g1 (row 7,col 6) -> f3 (row 5,col 5).
        let nf3 = try XCTUnwrap(PGNImporter.matchSAN("Nf3", state: engine.state, legal: legal))
        XCTAssertEqual([nf3.fromRow, nf3.fromCol, nf3.toRow, nf3.toCol], [7, 6, 5, 5])
    }

    func testReplayKnownOpening(   ) throws {
        // Ruy Lopez incl. a capture-free line + kingside castling.
        let sans = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]
        let engine = ChessGameEngine()
        var moves: [ChessMove] = []
        for san in sans {
            let legal = MoveGenerator.legalMoves(for: engine.state)
            let move = try XCTUnwrap(PGNImporter.matchSAN(san, state: engine.state, legal: legal),
                                     "failed to match SAN \(san)")
            moves.append(move)
            do {
                try engine.applyMoveAndAdvance(move)
            } catch {
                XCTFail("engine rejected matched move for \(san): \(error)")
                return
            }
        }
        XCTAssertEqual(moves.count, sans.count)
        // The castling move (O-O) is the 9th ply: white king e1 (row7,col4) -> g1 (row7,col6).
        let castle = moves[8]
        XCTAssertEqual([castle.fromRow, castle.fromCol, castle.toRow, castle.toCol], [7, 4, 7, 6])
    }

    func testCaptureSANMatches() throws {
        // 1. e4 d5 2. exd5 — the capture should resolve to the e4 pawn taking d5.
        let engine = ChessGameEngine()
        for san in ["e4", "d5"] {
            let legal = MoveGenerator.legalMoves(for: engine.state)
            let m = try XCTUnwrap(PGNImporter.matchSAN(san, state: engine.state, legal: legal))
            try engine.applyMoveAndAdvance(m)
        }
        let legal = MoveGenerator.legalMoves(for: engine.state)
        let capture = try XCTUnwrap(PGNImporter.matchSAN("exd5", state: engine.state, legal: legal))
        // d5 = row 3, col 3; capturing pawn came from the e-file (col 4).
        XCTAssertEqual([capture.toRow, capture.toCol, capture.fromCol], [3, 3, 4])
    }

    func testTimeControlClassification() {
        XCTAssertEqual(PGNImporter.timeControlClass("60+0"), "bullet")
        XCTAssertEqual(PGNImporter.timeControlClass("180+0"), "blitz")
        XCTAssertEqual(PGNImporter.timeControlClass("600+0"), "rapid")
        XCTAssertEqual(PGNImporter.timeControlClass("1800+0"), "classical")
        XCTAssertEqual(PGNImporter.timeControlClass("300+3"), "blitz") // 300 + 40*3 = 420
    }
}
