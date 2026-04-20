//
//  RepetitionTrackingTests.swift
//  DrewsChessMachineTests
//
//  Tests for `ChessGameEngine`'s position-history tracking, which is
//  the source of `GameState.repetitionCount` (the value that drives
//  `BoardEncoder` planes 18 and 19). The plan called for Zobrist
//  hashing tests, but we reused the existing PositionKey-based
//  tracking instead — so these tests verify that mechanism instead.
//
//  Tests cover:
//   - first move gets repetitionCount=0
//   - knight-shuffle 3-fold scenario produces correct count escalation
//   - halfmove-clock reset (pawn move / capture) clears history
//   - 3-fold repetition triggers `drawByThreefoldRepetition` result
//

import XCTest
@testable import DrewsChessMachine

final class RepetitionTrackingTests: XCTestCase {

    // MARK: - First move

    func testStartingPositionHasRepCountZero() {
        let engine = ChessGameEngine()
        XCTAssertEqual(engine.state.repetitionCount, 0,
                       "Starting position should have repetitionCount=0 (no prior visits)")
    }

    func testFirstMoveResultsInRepCountZero() throws {
        let engine = ChessGameEngine()
        // 1.e4
        let move = ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
        _ = try engine.applyMoveAndAdvance(move)
        XCTAssertEqual(engine.state.repetitionCount, 0,
                       "After 1.e4, the new position is novel; repetitionCount=0")
    }

    // MARK: - Knight shuffle: 3-fold repetition

    func testKnightShuffleThreefoldRepetition() throws {
        let engine = ChessGameEngine()
        // Move pairs that return to the starting position via knight
        // shuffles. Each WHITE+BLACK round-trip increases the rep count
        // of the starting position by 1.
        //   1. Nf3 Nc6  → starting + 1 ply each side
        //   2. Ng1 Nb8  → back to starting position; 2nd visit total → repCount=1
        //   3. Nf3 Nc6  → 3rd visit (count=2 visits ago + this = ... hmm let me think)
        //   4. Ng1 Nb8  → back to starting; 3rd visit total → repCount=2 → 3-fold
        //
        // The starting position has been visited:
        //   - At time 0 (initial)
        //   - After move 4 (back to starting via Nb8)
        //   - After move 8 (back to starting via second Nb8)
        let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
        let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
        let ng1 = ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil)
        let nb8 = ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)

        // Round 1: Nf3 Nc6
        _ = try engine.applyMoveAndAdvance(nf3)
        _ = try engine.applyMoveAndAdvance(nc6)
        // Now Ng1 Nb8 → back to starting
        _ = try engine.applyMoveAndAdvance(ng1)
        _ = try engine.applyMoveAndAdvance(nb8)
        XCTAssertEqual(engine.state.repetitionCount, 1,
                       "After first return to starting, repCount should be 1 (1 prior visit)")
        XCTAssertNil(engine.result, "1 prior visit is not yet a 3-fold draw")

        // Round 2: Nf3 Nc6 Ng1 Nb8 → back to starting again
        _ = try engine.applyMoveAndAdvance(nf3)
        _ = try engine.applyMoveAndAdvance(nc6)
        _ = try engine.applyMoveAndAdvance(ng1)
        _ = try engine.applyMoveAndAdvance(nb8)
        XCTAssertEqual(engine.state.repetitionCount, 2,
                       "After second return to starting, repCount should be 2 (2 prior visits)")
        // 3-fold should have triggered
        if case .drawByThreefoldRepetition = engine.result {
            // Expected
        } else {
            XCTFail("Expected drawByThreefoldRepetition; got \(String(describing: engine.result))")
        }
    }

    // MARK: - Halfmove-clock reset clears history

    func testPawnMoveClearsRepetitionHistory() throws {
        let engine = ChessGameEngine()
        let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
        let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
        let ng1 = ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil)
        let nb8 = ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)

        // Shuffle once to put starting position in the rep table with count 2.
        _ = try engine.applyMoveAndAdvance(nf3)
        _ = try engine.applyMoveAndAdvance(nc6)
        _ = try engine.applyMoveAndAdvance(ng1)
        _ = try engine.applyMoveAndAdvance(nb8)
        XCTAssertEqual(engine.state.repetitionCount, 1)

        // Now play 1.e4 — pawn move resets halfmove clock and should
        // clear the position-counts table.
        let e4 = ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
        _ = try engine.applyMoveAndAdvance(e4)
        XCTAssertEqual(engine.state.repetitionCount, 0,
                       "Position after 1.e4 is novel after history clear")
        XCTAssertEqual(engine.state.halfmoveClock, 0,
                       "Pawn move resets halfmove clock to 0")
    }
}
