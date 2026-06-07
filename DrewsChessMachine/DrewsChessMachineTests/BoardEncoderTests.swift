//
//  BoardEncoderTests.swift
//  DrewsChessMachineTests
//
//  Tensor-shape and plane-content invariant tests for `BoardEncoder`.
//  The encoder produces the input the network sees on every ply, so
//  bugs here mistrain at scale. Tests verify:
//   - tensor length matches the network's expected input size
//   - piece-occupancy planes correctly reflect the starting position
//   - castling planes broadcast as expected
//   - en passant plane sets exactly one cell
//   - halfmove-clock plane uses the /99 normalization
//   - repetition planes 18/19 follow always-fill semantics with proper
//     saturation at count=2
//   - encoder-frame perspective flip works correctly for black to move
//

import XCTest
@testable import DrewsChessMachine

final class BoardEncoderTests: XCTestCase {

    // MARK: - Tensor length

    func testTensorLength() {
        // Every encoding's stride must equal its plane count × 64.
        for encoding in InputEncoding.allCases {
            XCTAssertEqual(BoardEncoder.tensorLength(for: encoding),
                           encoding.planeCount * 8 * 8,
                           "tensorLength(for: .\(encoding)) must equal planeCount × 64")
        }
        let current = NetworkArchitecture.current
        XCTAssertEqual(BoardEncoder.tensorLength(for: current.inputEncoding),
                       current.inputPlanes * 8 * 8,
                       "the current architecture's stride must equal inputPlanes × 64")
        XCTAssertEqual(BoardEncoder.tensorLength(for: .basic30), 1920,
                       "basic30 has 30 planes × 64 = 1920 floats per encoded position (20 baseline + 10 temporal-repetition history)")
    }

    /// Drift guard for the `planeGroups` spec InputEncoding promises ("a unit
    /// test asserts the encoder fills exactly these ranges"). For EVERY
    /// encoding the declared groups must tile `[0, planeCount)` exactly —
    /// start at 0, contiguous, gap-free, non-overlapping — and the
    /// frame/tail/tensor-length derivations must all agree. Catches doc/impl
    /// drift the instant any encoding's plane layout changes (present or future).
    func testPlaneGroupsTileExactlyForAllEncodings() {
        for enc in InputEncoding.allCases {
            let groups = enc.planeGroups.sorted { $0.range.lowerBound < $1.range.lowerBound }
            XCTAssertFalse(groups.isEmpty, "\(enc.rawValue) has no plane groups")
            XCTAssertEqual(groups.first?.range.lowerBound, 0,
                           "\(enc.rawValue) plane groups must start at plane 0")
            var next = 0
            for g in groups {
                XCTAssertEqual(g.range.lowerBound, next,
                    "\(enc.rawValue): gap/overlap at plane \(g.range.lowerBound) (expected \(next))")
                next = g.range.upperBound + 1
            }
            XCTAssertEqual(next, enc.planeCount,
                "\(enc.rawValue): groups cover \(next) planes but planeCount is \(enc.planeCount)")
            XCTAssertEqual(enc.historyFrameCount * enc.planesPerFrame + enc.tailPlaneCount,
                           enc.planeCount,
                           "\(enc.rawValue): frame×perFrame + tail must equal planeCount")
            XCTAssertEqual(BoardEncoder.tensorLength(for: enc), enc.planeCount * 64,
                           "\(enc.rawValue): tensorLength must be planeCount × 64")
        }
    }

    /// Per-encoding content coverage: EVERY encoding's frame 0 (the base
    /// basic20 block, planes 0–19) must equal the `basic20` encode of the same
    /// position — so each tensor type is verified to reuse the (basic30-pinned)
    /// piece / castling / en-passant / halfmove content correctly, not just to
    /// have the right shape. Run across positions that exercise pieces,
    /// castling, en-passant and black-to-move (perspective flip). New
    /// encodings are covered automatically via `allCases`.
    func testEveryEncodingFrameZeroMatchesBasic20() {
        let afterE4 = MoveGenerator.applyMove(
            ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil),
            to: .starting)                                   // sets the en-passant plane
        let afterE4E5 = MoveGenerator.applyMove(
            ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil),
            to: afterE4)                                     // black to move → perspective flip
        let positions: [(String, GameState)] = [
            ("starting", .starting),
            ("after 1.e4 (EP set)", afterE4),
            ("after 1.e4 e5 (black to move)", afterE4E5),
        ]
        let frameZeroFloats = 20 * 64
        for enc in InputEncoding.allCases {
            for (label, pos) in positions {
                let full = BoardEncoder.encode(pos, encoding: enc)
                let basic20 = BoardEncoder.encode(pos, encoding: .basic20)
                XCTAssertGreaterThanOrEqual(full.count, frameZeroFloats)
                XCTAssertEqual(Array(full.prefix(frameZeroFloats)), basic20,
                    "\(enc.rawValue): frame-0 base block must equal basic20 for \(label)")
            }
        }
    }

    func testEncodeStartingPositionProducesCorrectLength() {
        let tensor = BoardEncoder.encode(.starting, encoding: .basic30)
        XCTAssertEqual(tensor.count, BoardEncoder.tensorLength(for: .basic30))
    }

    // MARK: - Piece occupancy

    func testStartingPositionPieceCounts() {
        let tensor = BoardEncoder.encode(.starting, encoding: .basic30)
        // Planes 0-5 = my (white) pieces. Each plane should have the
        // correct count of 1.0s for the starting position:
        //   pawn=8, knight=2, bishop=2, rook=2, queen=1, king=1
        let expectedMyCounts: [Float] = [8, 2, 2, 2, 1, 1]
        for plane in 0..<6 {
            let sum = sumPlane(tensor: tensor, plane: plane)
            XCTAssertEqual(sum, expectedMyCounts[plane],
                           "Plane \(plane) (my piece type \(plane)) should have \(expectedMyCounts[plane]) 1.0 cells")
        }
        // Planes 6-11 = opponent (black) pieces. Same counts.
        for plane in 6..<12 {
            let sum = sumPlane(tensor: tensor, plane: plane)
            XCTAssertEqual(sum, expectedMyCounts[plane - 6],
                           "Plane \(plane) (opp piece type \(plane - 6)) should have \(expectedMyCounts[plane - 6]) 1.0 cells")
        }
    }

    // MARK: - Castling planes

    func testStartingPositionCastlingPlanesAllOnes() {
        let tensor = BoardEncoder.encode(.starting, encoding: .basic30)
        // All four castling planes (12-15) should be all 1.0 at the
        // starting position since both colors have all four rights.
        for plane in 12..<16 {
            let sum = sumPlane(tensor: tensor, plane: plane)
            XCTAssertEqual(sum, 64.0,
                           "Castling plane \(plane) should be entirely 1.0 at starting position (got sum \(sum))")
        }
    }

    func testCastlingPlaneZeroAfterRightsCleared() {
        // Position with white-king-moved (no white castling rights)
        // but black still has both. From white's POV: planes 12-13
        // (my castling) should be zero, planes 14-15 (opp castling)
        // should be all 1.
        var board: [Piece?] = Array(repeating: nil, count: 64)
        board[7 * 8 + 4] = Piece(type: .king, color: .white)
        board[0 * 8 + 0] = Piece(type: .rook, color: .black)
        board[0 * 8 + 4] = Piece(type: .king, color: .black)
        board[0 * 8 + 7] = Piece(type: .rook, color: .black)
        let state = GameState(
            board: board,
            currentPlayer: .white,
            whiteKingsideCastle: false, whiteQueensideCastle: false,
            blackKingsideCastle: true, blackQueensideCastle: true,
            enPassantSquare: nil, halfmoveClock: 0
        )
        let tensor = BoardEncoder.encode(state, encoding: .basic30)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 12), 0.0, "My kingside castle plane should be zero")
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 13), 0.0, "My queenside castle plane should be zero")
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 14), 64.0, "Opp kingside castle plane should be all 1")
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 15), 64.0, "Opp queenside castle plane should be all 1")
    }

    // MARK: - En passant plane

    func testEnPassantPlaneIsZeroAtStartingPosition() {
        let tensor = BoardEncoder.encode(.starting, encoding: .basic30)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 16), 0.0,
                       "En passant plane should be zero at starting position")
    }

    func testEnPassantPlaneSetsExactlyOneCell() {
        // After 1.e4, the en passant target is e3 (row 5, col 4 in
        // absolute coordinates). For black to move, encoder flips
        // rows: e3 → encoder row (7 - 5) = 2.
        let after1e4 = MoveGenerator.applyMove(
            ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil),
            to: .starting
        )
        let tensor = BoardEncoder.encode(after1e4, encoding: .basic30)
        let sum = sumPlane(tensor: tensor, plane: 16)
        XCTAssertEqual(sum, 1.0, "En passant plane should have exactly one 1.0 cell after 1.e4")
        // Verify it's at the right spot: encoder row 2, col 4.
        let cell = tensor[16 * 64 + 2 * 8 + 4]
        XCTAssertEqual(cell, 1.0, "En passant cell should be at encoder row 2, col 4")
    }

    // MARK: - Halfmove clock plane (Leela-style /99)

    func testHalfmoveClockPlaneZeroAtClock0() {
        // Starting position has halfmoveClock = 0, so plane 17 should be all 0.
        let tensor = BoardEncoder.encode(.starting, encoding: .basic30)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 17), 0.0,
                       "Halfmove clock plane should be zero at clock=0")
    }

    func testHalfmoveClockPlaneNormalizesBy99() {
        // Halfmove clock = 99 → all cells = 1.0. Total = 64.
        let state = GameState(
            board: GameState.starting.board,
            currentPlayer: .white,
            whiteKingsideCastle: true, whiteQueensideCastle: true,
            blackKingsideCastle: true, blackQueensideCastle: true,
            enPassantSquare: nil,
            halfmoveClock: 99
        )
        let tensor = BoardEncoder.encode(state, encoding: .basic30)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 17), 64.0,
                       "Halfmove plane saturates at 1.0 when clock=99")
    }

    func testHalfmoveClockPlaneSaturatesAbove99() {
        // Per the encoder, clock is clamped at 99 before division. So
        // clock=100, 200, 999 all produce a fully-1.0 plane.
        for clock in [100, 200, 999] {
            let state = GameState(
                board: GameState.starting.board,
                currentPlayer: .white,
                whiteKingsideCastle: true, whiteQueensideCastle: true,
                blackKingsideCastle: true, blackQueensideCastle: true,
                enPassantSquare: nil,
                halfmoveClock: clock
            )
            let tensor = BoardEncoder.encode(state, encoding: .basic30)
            XCTAssertEqual(sumPlane(tensor: tensor, plane: 17), 64.0,
                           "Halfmove plane should saturate at all-1.0 for clock=\(clock)")
        }
    }

    // MARK: - Repetition planes (18 + 19)

    func testRepetitionPlanesZeroAtCount0() {
        let tensor = BoardEncoder.encode(.starting, encoding: .basic30)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 18), 0.0,
                       "Plane 18 (rep ≥1) should be zero at repetitionCount=0")
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 19), 0.0,
                       "Plane 19 (rep ≥2) should be zero at repetitionCount=0")
    }

    func testRepetitionPlanesAtCount1() {
        let state = GameState.starting.withRepetitionCount(1)
        let tensor = BoardEncoder.encode(state, encoding: .basic30)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 18), 64.0,
                       "Plane 18 should be all-1 at repetitionCount=1")
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 19), 0.0,
                       "Plane 19 should be zero at repetitionCount=1")
    }

    func testRepetitionPlanesAtCount2() {
        let state = GameState.starting.withRepetitionCount(2)
        let tensor = BoardEncoder.encode(state, encoding: .basic30)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 18), 64.0,
                       "Plane 18 should be all-1 at repetitionCount=2")
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 19), 64.0,
                       "Plane 19 should be all-1 at repetitionCount=2")
    }

    func testRepetitionCountSaturatesAtTwo() {
        // The encoder doesn't clamp — it relies on the caller (the
        // game engine) to saturate. But values ≥ 2 should produce the
        // same all-1 result for both planes regardless of how high.
        for count in [2, 3, 5, 100] {
            let state = GameState.starting.withRepetitionCount(count)
            let tensor = BoardEncoder.encode(state, encoding: .basic30)
            XCTAssertEqual(sumPlane(tensor: tensor, plane: 18), 64.0,
                           "Plane 18 should be all-1 for repetitionCount=\(count)")
            XCTAssertEqual(sumPlane(tensor: tensor, plane: 19), 64.0,
                           "Plane 19 should be all-1 for repetitionCount=\(count)")
        }
    }

    // MARK: - Encoder-frame perspective flip

    func testBlackToMoveFlipsRows() {
        // After 1.e4 black is to move. The encoder flips rows so black's
        // POV puts black's pieces at the "bottom" (rows 6-7 in encoder
        // frame). Verify by checking that the black king (which lives
        // at absolute row 0, col 4) appears at encoder row 7, col 4 in
        // the "my king" plane (plane 5).
        let after1e4 = MoveGenerator.applyMove(
            ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil),
            to: .starting
        )
        let tensor = BoardEncoder.encode(after1e4, encoding: .basic30)
        // Plane 5 is "my king." For black to move, that's the black king.
        // Black king at absolute (0, 4) → encoder row (7-0)=7, col 4.
        XCTAssertEqual(tensor[5 * 64 + 7 * 8 + 4], 1.0,
                       "Black king should appear at encoder row 7, col 4 in plane 5 (my king)")
        // Plane 11 is "opp king" — white king at absolute (7, 4) →
        // encoder row (7-7)=0, col 4.
        XCTAssertEqual(tensor[11 * 64 + 0 * 8 + 4], 1.0,
                       "White king should appear at encoder row 0, col 4 in plane 11 (opp king)")
    }

    // MARK: - Helpers

    private func sumPlane(tensor: [Float], plane: Int) -> Float {
        let start = plane * 64
        var sum: Float = 0
        for i in start..<(start + 64) {
            sum += tensor[i]
        }
        return sum
    }
}
