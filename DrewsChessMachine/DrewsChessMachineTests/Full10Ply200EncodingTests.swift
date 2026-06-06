//
//  Full10Ply200EncodingTests.swift
//  DrewsChessMachineTests
//
//  Correctness tests for the `full10ply200` input encoding: 10 stacked
//  `basic20` frames (current + 9 prior plies), all rendered from the ply-N
//  mover's perspective. The load-bearing, easy-to-get-wrong properties are:
//   - frame structure (200 planes = 10 × 20; frame N == basic20)
//   - absent history frames are all-zero (the "no frame here" signal)
//   - history frames are placed at the right plane offsets
//   - PERSPECTIVE: every frame uses the ply-N mover's perspective, NOT the
//     frame's own side-to-move, so an odd (opponent-to-move) prior frame
//     still shows *our* pieces in planes 0–5, oriented to our side — and the
//     whole stack flips when we are Black at N.
//   - the engine's `recentStates` window (data source) is NOT cleared on
//     irreversible moves and caps at `recentStateWindow`.
//

import XCTest
@testable import DrewsChessMachine

final class Full10Ply200EncodingTests: XCTestCase {

    // e2-e4 and e7-e5 in absolute (row 0 = rank 8) coordinates.
    private let e4 = ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
    private let e5 = ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)

    // MARK: - Frame structure

    func testFrameStructureInvariants() {
        XCTAssertEqual(BoardEncoder.tensorLength(for: .full10ply200), 200 * 64,
                       "full10ply200 is 200 planes × 64 = 12800 floats")
        XCTAssertEqual(InputEncoding.full10ply200.planeCount, 200)
        XCTAssertEqual(InputEncoding.full10ply200.historyFrameCount, 10)
        XCTAssertEqual(InputEncoding.full10ply200.planesPerFrame, 20)
        // historyFrameCount × planesPerFrame == planeCount for every encoding.
        for enc in InputEncoding.allCases {
            XCTAssertEqual(enc.historyFrameCount * enc.planesPerFrame, enc.planeCount,
                           "frame-structure invariant must hold for \(enc.rawValue)")
        }
    }

    func testFrameNMatchesBasic20AndAbsentHistoryIsZero() {
        // With no history supplied, frame 0 (planes 0–19) must be byte-equal
        // to a basic20 encoding of the same position, and every history frame
        // (planes 20–199) must be all-zero.
        let full = BoardEncoder.encode(.starting, encoding: .full10ply200)
        let basic20 = BoardEncoder.encode(.starting, encoding: .basic20)
        XCTAssertEqual(full.count, 200 * 64)
        XCTAssertEqual(basic20.count, 20 * 64)
        for i in 0..<(20 * 64) {
            XCTAssertEqual(full[i], basic20[i], "frame-N plane data must match basic20 at float \(i)")
        }
        for i in (20 * 64)..<(200 * 64) {
            XCTAssertEqual(full[i], 0.0, "absent history frame float \(i) must be zero")
        }
    }

    // MARK: - History placement + perspective (White to move at N)

    func testHistoryFrameWhitePerspectiveAtN() throws {
        // 1.e4 e5 → White to move at N. recentStates[0] = after-1.e4 (Black
        // was to move there); it must still render from WHITE's perspective.
        let engine = ChessGameEngine()
        try engine.applyMoveAndAdvance(e4)
        try engine.applyMoveAndAdvance(e5)
        XCTAssertEqual(engine.state.currentPlayer, .white)
        XCTAssertEqual(engine.recentStates.count, 2)

        let t = BoardEncoder.encode(engine.state, history: engine.recentStates, encoding: .full10ply200)

        // Frame 1 = planes 20–39 = the after-1.e4 position. From White's
        // perspective (no flip), the e4 pawn sits in MY-pawns (plane 20) at
        // (row 4, col 4), and the vacated e2 square (row 6, col 4) is empty.
        // If this frame had wrongly used its own (Black) perspective, the
        // Black e-pawn would appear flipped at plane-20 (6,4) instead.
        XCTAssertEqual(t[20 * 64 + 4 * 8 + 4], 1.0, "e4 pawn must be in frame-1 my-pawns (White persp)")
        XCTAssertEqual(t[20 * 64 + 6 * 8 + 4], 0.0, "vacated e2 must be empty (rules out Black-persp render)")
        XCTAssertEqual(sumPlane(t, 20), 8.0, "8 White pawns in frame-1 my-pawns")
        // White king at (7,4), unflipped → frame-1 my-king (plane 25).
        XCTAssertEqual(t[25 * 64 + 7 * 8 + 4], 1.0, "White king in frame-1 my-king (White persp)")

        // Frame 2 = planes 40–59 = the starting position, White persp.
        XCTAssertEqual(sumPlane(t, 40), 8.0, "8 White pawns in frame-2 my-pawns")
        XCTAssertEqual(t[40 * 64 + 6 * 8 + 4], 1.0, "e2 pawn present in frame-2 (start position)")
    }

    // MARK: - History placement + perspective (Black to move at N)

    func testHistoryFrameBlackPerspectiveAtN() throws {
        // After 1.e4 it is Black to move → N's mover is Black, so EVERY frame
        // (including frame 1 = the starting position, whose own mover was
        // White) must render from Black's perspective (flipped, Black in my-).
        let engine = ChessGameEngine()
        try engine.applyMoveAndAdvance(e4)
        XCTAssertEqual(engine.state.currentPlayer, .black)
        XCTAssertEqual(engine.recentStates.count, 1)

        let t = BoardEncoder.encode(engine.state, history: engine.recentStates, encoding: .full10ply200)

        // Frame 0 = current (after-1.e4), Black persp (flipped). Black king
        // at absolute (0,4) → flipped (row 7, col 4) in my-king (plane 5).
        XCTAssertEqual(t[5 * 64 + 7 * 8 + 4], 1.0, "Black king in frame-0 my-king (flipped)")

        // Frame 1 = planes 20–39 = the starting position, rendered from
        // BLACK's perspective. Black pawns (absolute row 1) flip to row 6 in
        // MY-pawns (plane 20); Black king (0,4) → (7,4) in my-king (plane 25).
        XCTAssertEqual(sumPlane(t, 20), 8.0, "8 Black pawns in frame-1 my-pawns (Black persp)")
        XCTAssertEqual(t[20 * 64 + 6 * 8 + 0], 1.0, "Black a7 pawn → flipped (6,0) in frame-1 my-pawns")
        XCTAssertEqual(t[25 * 64 + 7 * 8 + 4], 1.0, "Black king → flipped (7,4) in frame-1 my-king")
    }

    // MARK: - Engine recentStates window (data source for the encoding)

    func testRecentStatesNotClearedOnIrreversibleMove() throws {
        // A pawn move resets the halfmove clock and clears the repetition
        // window (recentPositionKeys), but the history-stack window must NOT
        // clear — a prior frame must still show the true earlier board.
        let engine = ChessGameEngine()
        XCTAssertEqual(engine.recentStates.count, 0)
        try engine.applyMoveAndAdvance(e4) // irreversible (pawn move)
        XCTAssertEqual(engine.recentStates.count, 1,
                       "recentStates must survive an irreversible move")
        XCTAssertEqual(engine.recentStates[0].currentPlayer, .white,
                       "the one retained prior frame is the white-to-move start position")
    }

    func testRecentStatesWindowCap() throws {
        // Playing past the window keeps exactly `recentStateWindow` prior
        // states (the most recent ones). Robust to however the game unfolds.
        let engine = ChessGameEngine()
        var plies = 0
        while plies < 12, engine.result == nil, let move = engine.currentLegalMoves.first {
            try engine.applyMoveAndAdvance(move)
            plies += 1
        }
        XCTAssertEqual(engine.recentStates.count,
                       min(plies, ChessGameEngine.recentStateWindow))
    }

    // MARK: - Helpers

    private func sumPlane(_ tensor: [Float], _ plane: Int) -> Float {
        let start = plane * 64
        var sum: Float = 0
        for i in start..<(start + 64) { sum += tensor[i] }
        return sum
    }
}
