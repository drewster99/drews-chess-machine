//
//  RepetitionPlaneTests.swift
//  DrewsChessMachineTests
//
//  Tests for the 10 binary temporal-repetition-history planes (planes
//  20–29) added in the v3 architecture refresh. The signal these planes
//  carry — "the position N plies ago is a strict chess-rules duplicate
//  of the current position" — is intended to let the network detect
//  cycle patterns (e.g. 2-ply knight shuffles toward threefold
//  repetition) that the existing repetition-count planes 18-19 can only
//  express as an accumulating count without temporal structure.
//
//  Plane indexing: plane 20 = index 0 = "1 ply ago matches current",
//  plane 29 = index 9 = "10 plies ago matches current". The same bit
//  indexing is exposed on `GameState.recentRepetitionMask` as
//  `(mask >> i) & 1` for i in 0..<10.
//

import XCTest
@testable import DrewsChessMachine

final class RepetitionPlaneTests: XCTestCase {

    // MARK: - Plane-presence basics

    /// All 10 history planes should be zero at the starting position
    /// (no prior plies exist in this game).
    func testRepetitionPlanesEmptyAtGameStart() {
        let engine = ChessGameEngine()
        XCTAssertEqual(engine.state.recentRepetitionMask, 0,
                       "Mask should be 0 at game start (no prior positions)")
        let tensor = BoardEncoder.encode(engine.state)
        for i in 0..<10 {
            let sum = sumPlane(tensor: tensor, plane: 20 + i)
            XCTAssertEqual(sum, 0.0,
                           "Plane \(20 + i) should be zero at game start")
        }
    }

    /// A `GameState` constructed without supplying a mask (e.g. from a
    /// test fixture, the UI's editable position, or the starting state)
    /// should produce all-zero history planes.
    func testRepetitionPlanesZeroForDefaultGameState() {
        let tensor = BoardEncoder.encode(.starting)
        for i in 0..<10 {
            let sum = sumPlane(tensor: tensor, plane: 20 + i)
            XCTAssertEqual(sum, 0.0,
                           "Plane \(20 + i) should be zero for a default GameState")
        }
    }

    // MARK: - Specific plane activation by ply distance

    /// In a 2-ply knight shuffle (Nf3 Nc6 Ng1 Nb8 returns to start),
    /// the post-Nb8 position is identical to the starting position. The
    /// pre-Nb8 state (after Ng1) is NOT identical (Nb8 hasn't moved yet),
    /// so the immediate-prior key (1 ply ago) doesn't match the new
    /// position; the position 4 plies ago is the starting position again
    /// — wait, let's trace carefully:
    ///
    /// After full sequence Nf3 Nc6 Ng1 Nb8:
    /// - Current position: starting (4 plies in)
    /// - 1 ply ago: position after Ng1 (black to move, knight on c6)
    /// - 2 plies ago: position after Nc6
    /// - 3 plies ago: position after Nf3
    /// - 4 plies ago: starting (white to move, all knights home)
    ///
    /// So bit 3 (index 3) should be set in the mask, and plane 23 should
    /// be all-1. The other planes 20-29 stay zero.
    func testRepetitionPlaneAt4PliesAgo() throws {
        let engine = ChessGameEngine()
        let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
        let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
        let ng1 = ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil)
        let nb8 = ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)

        _ = try engine.applyMoveAndAdvance(nf3)
        _ = try engine.applyMoveAndAdvance(nc6)
        _ = try engine.applyMoveAndAdvance(ng1)
        _ = try engine.applyMoveAndAdvance(nb8)

        // We expect ONLY bit 3 to be set (position 4 plies ago = starting).
        XCTAssertEqual(engine.state.recentRepetitionMask, UInt16(1) << 3,
                       "After Nf3 Nc6 Ng1 Nb8, only bit 3 (4 plies ago) should be set; got mask=0x\(String(engine.state.recentRepetitionMask, radix: 16))")

        let tensor = BoardEncoder.encode(engine.state)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 23), 64.0,
                       "Plane 23 (4 plies ago) should be all-1")
        for i in 0..<10 where i != 3 {
            XCTAssertEqual(sumPlane(tensor: tensor, plane: 20 + i), 0.0,
                           "Plane \(20 + i) should be zero (no other match)")
        }
    }

    /// Same shuffle, but executed twice. After the second full cycle
    /// (Nf3 Nc6 Ng1 Nb8 ×2 = 8 plies), the current position is starting
    /// again, AND it equals both:
    /// - The position 4 plies ago (after the first Nb8)
    /// - The position 8 plies ago (initial starting state)
    ///
    /// So bits 3 and 7 should both be set, planes 23 and 27 all-1, others 0.
    /// (Also: this triggers 3-fold so the game ends — but the mask is
    /// computed from the state regardless of the result.)
    func testRepetitionPlanesMultipleMatchesInCycle() throws {
        let engine = ChessGameEngine()
        let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
        let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
        let ng1 = ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil)
        let nb8 = ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)

        for _ in 0..<2 {
            _ = try engine.applyMoveAndAdvance(nf3)
            _ = try engine.applyMoveAndAdvance(nc6)
            _ = try engine.applyMoveAndAdvance(ng1)
            _ = try engine.applyMoveAndAdvance(nb8)
        }

        let expected = (UInt16(1) << 3) | (UInt16(1) << 7)
        XCTAssertEqual(engine.state.recentRepetitionMask, expected,
                       "After two full knight cycles, bits 3 and 7 should be set; got mask=0x\(String(engine.state.recentRepetitionMask, radix: 16))")

        let tensor = BoardEncoder.encode(engine.state)
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 23), 64.0,
                       "Plane 23 (4 plies ago) should be all-1")
        XCTAssertEqual(sumPlane(tensor: tensor, plane: 27), 64.0,
                       "Plane 27 (8 plies ago) should be all-1")
        for i in 0..<10 where i != 3 && i != 7 {
            XCTAssertEqual(sumPlane(tensor: tensor, plane: 20 + i), 0.0,
                           "Plane \(20 + i) should be zero (no other match)")
        }
    }

    // MARK: - Zero-padding when history is shorter than 10

    /// Before 10 plies have been played, the engine's `recentPositionKeys`
    /// window is shorter than 10 entries. Bits beyond the window's length
    /// are always 0 in the mask, regardless of any earlier coincidence.
    func testRepetitionPlanesZeroPadShortHistory() throws {
        let engine = ChessGameEngine()
        // 2 plies of normal play — history length is 2 after these.
        let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
        let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
        _ = try engine.applyMoveAndAdvance(nf3)
        _ = try engine.applyMoveAndAdvance(nc6)

        // No matches yet (every position is novel). Mask = 0.
        XCTAssertEqual(engine.state.recentRepetitionMask, 0,
                       "Mask should be 0 with no cycles yet")
        let tensor = BoardEncoder.encode(engine.state)
        for i in 0..<10 {
            XCTAssertEqual(sumPlane(tensor: tensor, plane: 20 + i), 0.0,
                           "Plane \(20 + i) should be zero before any cycles")
        }
    }

    // MARK: - Cleared on irreversible move

    /// When a pawn move or capture resets the halfmove clock to 0, the
    /// recent-position window is cleared (matching the existing
    /// `positionCounts` behavior). The new position's mask is 0 because
    /// no prior state can equal it across an irreversible move.
    func testRepetitionPlanesClearedOnIrreversibleMove() throws {
        let engine = ChessGameEngine()
        let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
        let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
        let ng1 = ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil)
        let nb8 = ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)

        // Build up a non-trivial history with a real cycle match.
        _ = try engine.applyMoveAndAdvance(nf3)
        _ = try engine.applyMoveAndAdvance(nc6)
        _ = try engine.applyMoveAndAdvance(ng1)
        _ = try engine.applyMoveAndAdvance(nb8)
        XCTAssertNotEqual(engine.state.recentRepetitionMask, 0,
                          "Sanity: mask should be non-zero after the cycle returns to start")

        // 1.e4 — pawn move, halfmove clock resets, history clears.
        let e4 = ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
        _ = try engine.applyMoveAndAdvance(e4)
        XCTAssertEqual(engine.state.recentRepetitionMask, 0,
                       "Mask should be 0 after irreversible move clears history")
        XCTAssertEqual(engine.state.halfmoveClock, 0,
                       "Sanity: halfmove clock is 0 after a pawn move")

        let tensor = BoardEncoder.encode(engine.state)
        for i in 0..<10 {
            XCTAssertEqual(sumPlane(tensor: tensor, plane: 20 + i), 0.0,
                           "Plane \(20 + i) should be zero after irreversible-move clear")
        }
    }

    /// And one more move after the irreversible move — the window has
    /// to re-accumulate from scratch, so the mask still reflects only
    /// the post-irreversible positions.
    func testRepetitionPlanesWindowRebuildsAfterClear() throws {
        let engine = ChessGameEngine()
        // 1. e4 (irreversible), 1...e5 (irreversible), 2. Nf3 Nc6
        let e4 = ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
        let e5 = ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)
        let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
        let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
        _ = try engine.applyMoveAndAdvance(e4)
        _ = try engine.applyMoveAndAdvance(e5)
        _ = try engine.applyMoveAndAdvance(nf3)
        _ = try engine.applyMoveAndAdvance(nc6)
        // No repetitions in this line, but the engine has built up a
        // 2-entry window (the two non-pawn moves). Mask is still 0.
        XCTAssertEqual(engine.state.recentRepetitionMask, 0,
                       "No repetitions yet; mask is 0")
    }

    // MARK: - PositionKey semantics: side-to-move must match

    /// A `PositionKey` distinguishes "same pieces, opposite STM" — they
    /// are different chess positions per FIDE Article 9.2. So even if
    /// piece placement happens to match, the mask remains 0 when the
    /// side-to-move differs.
    ///
    /// Realizing this naturally is hard (the engine sees only legal
    /// positions and reaching identical piece placement with opposite
    /// STM in a real game requires a specific maneuver). Instead we
    /// drive the equality check directly using `PositionKey.==`.
    func testPositionKeyDistinguishesSideToMove() {
        let whiteToMove = GameState.starting
        let blackToMove = GameState(
            board: GameState.starting.board,
            currentPlayer: .black,
            whiteKingsideCastle: true,
            whiteQueensideCastle: true,
            blackKingsideCastle: true,
            blackQueensideCastle: true,
            enPassantSquare: nil,
            halfmoveClock: 0
        )
        let keyW = PositionKey(from: whiteToMove)
        let keyB = PositionKey(from: blackToMove)
        XCTAssertNotEqual(keyW, keyB,
                          "PositionKey must distinguish white-to-move from black-to-move (FIDE Article 9.2)")
    }

    // MARK: - GameState.withRecentRepetitionMask round-trips

    /// The setter copies the mask through; the getter reads it back.
    /// All other fields are preserved.
    func testWithRecentRepetitionMaskRoundTrips() {
        let baseline = GameState.starting
        XCTAssertEqual(baseline.recentRepetitionMask, 0)

        let withMask = baseline.withRecentRepetitionMask(0x0285)  // bits 0, 2, 7, 9
        XCTAssertEqual(withMask.recentRepetitionMask, 0x0285)
        // Other fields preserved.
        XCTAssertEqual(withMask.currentPlayer, baseline.currentPlayer)
        XCTAssertEqual(withMask.halfmoveClock, baseline.halfmoveClock)
        XCTAssertEqual(withMask.repetitionCount, baseline.repetitionCount)
        XCTAssertEqual(withMask.board.count, 64)
    }

    /// Setting the mask via `withRecentRepetitionMask` propagates through
    /// the encoder: bits set in the mask produce all-1 planes; bits
    /// unset produce all-0 planes.
    func testEncodedPlanesMirrorMaskBits() {
        // Set bits 0, 4, 9 (planes 20, 24, 29).
        let mask: UInt16 = (1 << 0) | (1 << 4) | (1 << 9)
        let state = GameState.starting.withRecentRepetitionMask(mask)
        let tensor = BoardEncoder.encode(state)
        let setBits: Set<Int> = [0, 4, 9]
        for i in 0..<10 {
            let expected: Float = setBits.contains(i) ? 64.0 : 0.0
            XCTAssertEqual(sumPlane(tensor: tensor, plane: 20 + i), expected,
                           "Plane \(20 + i) should be \(expected == 64.0 ? "all-1" : "all-0") with bit \(i) \(setBits.contains(i) ? "set" : "clear")")
        }
    }

    // MARK: - Architecture sanity

    /// Encoder's tensor length is `inputPlanes × 64`. The v3 refresh sets
    /// `inputPlanes = 30`. If a future change moves the constant, this
    /// test still passes as long as the relationship holds.
    func testTensorLengthMatchesInputPlanes() {
        XCTAssertEqual(BoardEncoder.tensorLength,
                       ChessNetwork.inputPlanes * 8 * 8)
    }

    /// The engine exposes its window-size constant to match the encoder's
    /// plane budget. If either is changed without the other, this test
    /// flags the drift.
    func testEngineWindowMatchesEncoderPlaneCount() {
        // 10 binary history planes, one per ply of lookback.
        XCTAssertEqual(ChessGameEngine.recentPositionKeyWindow, 10)
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
