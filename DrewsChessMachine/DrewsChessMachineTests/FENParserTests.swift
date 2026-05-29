//
//  FENParserTests.swift
//  DrewsChessMachineTests
//
//  Structural tests for the FEN -> GameState parser. The parser is the
//  bridge between the bundled Lichess probe JSON and the engine's native
//  GameState type, so any drift here silently desyncs the 200-position
//  probe set. Tests cover:
//   - starting-position FEN round-trips to GameState.starting (board,
//     side-to-move, castling rights, en-passant, halfmove clock)
//   - mid-game positions with partial castling rights, set en-passant
//     squares, non-zero halfmove clock, and black-to-move are decoded
//     into the right cells
//   - malformed FENs throw specific ParseError cases rather than
//     silently producing a degenerate GameState
//

import XCTest
@testable import DrewsChessMachine

final class FENParserTests: XCTestCase {

    // MARK: - Starting position

    func testStartingPositionMatchesGameStateStarting() throws {
        let start = try FENParser.parse(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        XCTAssertEqual(start.currentPlayer, .white)
        XCTAssertTrue(start.whiteKingsideCastle)
        XCTAssertTrue(start.whiteQueensideCastle)
        XCTAssertTrue(start.blackKingsideCastle)
        XCTAssertTrue(start.blackQueensideCastle)
        XCTAssertNil(start.enPassantSquare)
        XCTAssertEqual(start.halfmoveClock, 0)

        let reference = GameState.starting
        XCTAssertEqual(start.board.count, 64)
        XCTAssertEqual(reference.board.count, 64)
        for sq in 0..<64 {
            XCTAssertEqual(
                start.board[sq],
                reference.board[sq],
                "square \(sq) differs from GameState.starting"
            )
        }
    }

    // MARK: - Piece placement

    func testBlackKingAtA8AndWhiteKingAtH1() throws {
        let s = try FENParser.parse("k7/8/8/8/8/8/8/7K w - - 0 1")
        XCTAssertEqual(s.board[0], Piece(type: .king, color: .black))
        XCTAssertEqual(s.board[63], Piece(type: .king, color: .white))
        for sq in 1..<63 {
            XCTAssertNil(s.board[sq], "square \(sq) should be empty")
        }
    }

    func testMixedRankRunOfDigits() throws {
        // Rank 5 (row 3): "3R4" -> three empties, white rook on d5, four empties.
        let s = try FENParser.parse("8/8/8/3R4/8/8/8/8 w - - 0 1")
        XCTAssertEqual(s.board[3 * 8 + 3], Piece(type: .rook, color: .white))
        XCTAssertNil(s.board[3 * 8 + 2])
        XCTAssertNil(s.board[3 * 8 + 4])
    }

    // MARK: - Side to move

    func testBlackToMove() throws {
        let s = try FENParser.parse("8/8/8/8/8/8/8/8 b - - 0 1")
        XCTAssertEqual(s.currentPlayer, .black)
    }

    // MARK: - Castling rights

    func testPartialCastlingRights() throws {
        // K only -> white kingside on, white queenside off, black both off.
        let s = try FENParser.parse("8/8/8/8/8/8/8/8 w K - 0 1")
        XCTAssertTrue(s.whiteKingsideCastle)
        XCTAssertFalse(s.whiteQueensideCastle)
        XCTAssertFalse(s.blackKingsideCastle)
        XCTAssertFalse(s.blackQueensideCastle)
    }

    func testNoCastlingRights() throws {
        let s = try FENParser.parse("8/8/8/8/8/8/8/8 w - - 0 1")
        XCTAssertFalse(s.whiteKingsideCastle)
        XCTAssertFalse(s.whiteQueensideCastle)
        XCTAssertFalse(s.blackKingsideCastle)
        XCTAssertFalse(s.blackQueensideCastle)
    }

    func testAllCastlingRights() throws {
        let s = try FENParser.parse("8/8/8/8/8/8/8/8 w KQkq - 0 1")
        XCTAssertTrue(s.whiteKingsideCastle)
        XCTAssertTrue(s.whiteQueensideCastle)
        XCTAssertTrue(s.blackKingsideCastle)
        XCTAssertTrue(s.blackQueensideCastle)
    }

    // MARK: - En passant

    func testEnPassantE3() throws {
        // e3 = file 'e' (col 4), rank 3 (row 5).
        let s = try FENParser.parse("8/8/8/8/8/8/8/8 b - e3 0 1")
        XCTAssertEqual(s.enPassantSquare?.row, 5)
        XCTAssertEqual(s.enPassantSquare?.col, 4)
    }

    func testEnPassantA6() throws {
        // a6 = file 'a' (col 0), rank 6 (row 2).
        let s = try FENParser.parse("8/8/8/8/8/8/8/8 w - a6 0 1")
        XCTAssertEqual(s.enPassantSquare?.row, 2)
        XCTAssertEqual(s.enPassantSquare?.col, 0)
    }

    func testNoEnPassantStaysNil() throws {
        let s = try FENParser.parse("8/8/8/8/8/8/8/8 w - - 0 1")
        XCTAssertNil(s.enPassantSquare)
    }

    // MARK: - Halfmove clock

    func testHalfmoveClockNonZero() throws {
        let s = try FENParser.parse("8/8/8/8/8/8/8/8 w - - 47 1")
        XCTAssertEqual(s.halfmoveClock, 47)
    }

    // MARK: - Real-world sample (one of the curated puzzles)

    func testCuratedHangingPieceSample() throws {
        // From lichess_probes_200.json — hangingPiece bucket id=000lC.
        let s = try FENParser.parse(
            "3r3r/pQNk1ppp/1qnR1n2/1B6/8/8/PPP3PP/5R1K b - - 0 19"
        )
        XCTAssertEqual(s.currentPlayer, .black)
        XCTAssertFalse(s.whiteKingsideCastle)
        XCTAssertFalse(s.whiteQueensideCastle)
        XCTAssertFalse(s.blackKingsideCastle)
        XCTAssertFalse(s.blackQueensideCastle)
        XCTAssertNil(s.enPassantSquare)
        XCTAssertEqual(s.halfmoveClock, 0)

        // Black king on d7 = row 1, col 3.
        XCTAssertEqual(
            s.board[1 * 8 + 3],
            Piece(type: .king, color: .black)
        )
        // White rook on d6 = row 2, col 3.
        XCTAssertEqual(
            s.board[2 * 8 + 3],
            Piece(type: .rook, color: .white)
        )
        // White king on h1 = row 7, col 7.
        XCTAssertEqual(
            s.board[7 * 8 + 7],
            Piece(type: .king, color: .white)
        )
    }

    // MARK: - Malformed input

    func testWrongFieldCountThrows() {
        XCTAssertThrowsError(
            try FENParser.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -")
        ) { error in
            guard let pe = error as? FENParser.ParseError else {
                XCTFail("wrong error type: \(error)"); return
            }
            if case .wrongFieldCount(let n) = pe {
                XCTAssertEqual(n, 5)
            } else {
                XCTFail("expected wrongFieldCount, got \(pe)")
            }
        }
    }

    func testWrongRankCountThrows() {
        XCTAssertThrowsError(
            try FENParser.parse("8/8/8/8/8/8/8 w - - 0 1")
        ) { error in
            if case FENParser.ParseError.wrongRankCount(let n) = error {
                XCTAssertEqual(n, 7)
            } else {
                XCTFail("expected wrongRankCount, got \(error)")
            }
        }
    }

    func testRankSumMismatchThrows() {
        XCTAssertThrowsError(
            try FENParser.parse("8/8/8/8/8/8/8/9 w - - 0 1")
        ) { error in
            guard let pe = error as? FENParser.ParseError else {
                XCTFail("wrong error type: \(error)"); return
            }
            if case .rankFileSumMismatch = pe { /* ok */ } else {
                XCTFail("expected rankFileSumMismatch, got \(pe)")
            }
        }
    }

    func testUnknownPieceCharacterThrows() {
        XCTAssertThrowsError(
            try FENParser.parse("8/8/8/8/8/8/8/X7 w - - 0 1")
        ) { error in
            if case FENParser.ParseError.unknownPieceCharacter(let c) = error {
                XCTAssertEqual(c, "X")
            } else {
                XCTFail("expected unknownPieceCharacter, got \(error)")
            }
        }
    }

    func testBadSideToMoveThrows() {
        XCTAssertThrowsError(
            try FENParser.parse("8/8/8/8/8/8/8/8 X - - 0 1")
        ) { error in
            if case FENParser.ParseError.badSideToMove = error { /* ok */ } else {
                XCTFail("expected badSideToMove, got \(error)")
            }
        }
    }

    func testBadCastlingFieldThrows() {
        XCTAssertThrowsError(
            try FENParser.parse("8/8/8/8/8/8/8/8 w XYZ - 0 1")
        ) { error in
            if case FENParser.ParseError.badCastlingField = error { /* ok */ } else {
                XCTFail("expected badCastlingField, got \(error)")
            }
        }
    }

    func testBadEnPassantFieldThrows() {
        XCTAssertThrowsError(
            try FENParser.parse("8/8/8/8/8/8/8/8 w - e9 0 1")
        ) { error in
            if case FENParser.ParseError.badEnPassantField = error { /* ok */ } else {
                XCTFail("expected badEnPassantField, got \(error)")
            }
        }
    }
}
