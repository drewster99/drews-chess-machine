//
//  LegalMoveValidationTests.swift
//  DrewsChessMachineTests
//
//  Tests for `ChessGameEngine`'s authoritative legal-move list and the
//  validation it performs inside `applyMoveAndAdvance`. The engine
//  owns `currentLegalMoves`, computed once at init and refreshed after
//  every successful apply, and rejects any move not in that list with
//  `ChessGameError.illegalMove`. Callers no longer need to track a
//  parallel legal-move list themselves — the engine is the single
//  source of truth.
//
//  Coverage:
//   - currentLegalMoves is populated at init and matches MoveGenerator
//   - a legal move applies successfully and refreshes the list
//   - the refreshed list belongs to the new side-to-move
//   - a structurally invalid move (empty from-square) throws illegalMove
//     instead of trapping inside MoveGenerator.applyMove's force unwrap
//   - an on-board but rule-illegal move (wrong-color, blocked, phantom
//     promotion, non-adjacent king move, etc.) throws illegalMove
//   - a previously-legal move that became illegal after advancing a ply
//     is rejected
//   - illegalMove never mutates state, history, or currentLegalMoves
//   - an illegal move after a result is latched still surfaces
//     gameAlreadyOver (result-guard precedes legality-guard)
//

import XCTest
@testable import DrewsChessMachine

final class LegalMoveValidationTests: XCTestCase {

    // MARK: - Helpers

    /// A move that cannot possibly be legal from the starting position —
    /// the source square (row 4, col 4 = e4) is empty on move 1. Prior
    /// to validation, this would trap on the force-unwrap inside
    /// `MoveGenerator.applyMove`.
    private let emptyFromSquareMove = ChessMove(
        fromRow: 4, fromCol: 4, toRow: 3, toCol: 4, promotion: nil
    )

    /// On-board move that is structurally well-formed but rule-illegal:
    /// white tries to move black's e7 pawn on move 1.
    private let wrongColorPawnMove = ChessMove(
        fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil
    )

    /// Another on-board rule-illegal move: white tries to jump the king
    /// three squares from e1 to h1 — king does not move that far.
    private let phantomKingLeap = ChessMove(
        fromRow: 7, fromCol: 4, toRow: 7, toCol: 7, promotion: nil
    )

    /// A pseudo-promotion move that cannot be legal from the starting
    /// position (white d2 pawn "promoting" to queen on d3 — pawn is not
    /// on the promotion rank).
    private let bogusPromotion = ChessMove(
        fromRow: 6, fromCol: 3, toRow: 5, toCol: 3, promotion: .queen
    )

    /// Standard legal opener: 1. e4.
    private let e4 = ChessMove(
        fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil
    )

    /// Standard legal reply: 1... e5.
    private let e5 = ChessMove(
        fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil
    )

    /// Legal knight opener: 1. Nf3.
    private let nf3 = ChessMove(
        fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil
    )

    // MARK: - currentLegalMoves exposure

    func testEngineExposesCurrentLegalMovesAtInit() {
        let engine = ChessGameEngine()
        let viaGenerator = MoveGenerator.legalMoves(for: engine.state)
        XCTAssertEqual(engine.currentLegalMoves.count, viaGenerator.count,
                       "currentLegalMoves should match MoveGenerator at init")
        XCTAssertEqual(Set(engine.currentLegalMoves), Set(viaGenerator),
                       "currentLegalMoves should be the same set as MoveGenerator.legalMoves")
        XCTAssertEqual(engine.currentLegalMoves.count, 20,
                       "Standard starting position has exactly 20 legal moves")
    }

    func testCurrentLegalMovesRefreshedAfterSuccessfulApply() throws {
        let engine = ChessGameEngine()
        let beforeCount = engine.currentLegalMoves.count
        try engine.applyMoveAndAdvance(e4)

        // After 1.e4 the side to move is black; black also has 20 legal
        // moves from the mirror-symmetric position.
        XCTAssertEqual(engine.state.currentPlayer, .black)
        XCTAssertEqual(engine.currentLegalMoves.count, 20,
                       "Black has 20 legal moves after 1.e4")
        XCTAssertEqual(beforeCount, 20)

        // The refreshed list must match a fresh MoveGenerator call.
        let fresh = MoveGenerator.legalMoves(for: engine.state)
        XCTAssertEqual(Set(engine.currentLegalMoves), Set(fresh),
                       "currentLegalMoves after apply must match MoveGenerator")
    }

    func testApplyMoveAndAdvanceReturnValueMatchesProperty() throws {
        let engine = ChessGameEngine()
        let returned = try engine.applyMoveAndAdvance(e4)
        XCTAssertEqual(Set(returned), Set(engine.currentLegalMoves),
                       "Return value of applyMoveAndAdvance must equal engine.currentLegalMoves")
    }

    // MARK: - Illegal move rejection

    func testEmptyFromSquareMoveThrowsIllegalMove() {
        let engine = ChessGameEngine()
        XCTAssertThrowsError(try engine.applyMoveAndAdvance(emptyFromSquareMove)) { error in
            guard case ChessGameError.illegalMove(let rejected) = error else {
                XCTFail("Expected ChessGameError.illegalMove; got \(error)")
                return
            }
            XCTAssertEqual(rejected, self.emptyFromSquareMove,
                           "Error should carry the rejected move verbatim")
        }
    }

    func testWrongColorPawnMoveThrowsIllegalMove() {
        let engine = ChessGameEngine()
        // On move 1 it's white to move; a black-pawn move must be rejected.
        XCTAssertThrowsError(try engine.applyMoveAndAdvance(wrongColorPawnMove)) { error in
            guard case ChessGameError.illegalMove = error else {
                XCTFail("Expected ChessGameError.illegalMove for wrong-color move; got \(error)")
                return
            }
        }
    }

    func testPhantomKingLeapThrowsIllegalMove() {
        let engine = ChessGameEngine()
        XCTAssertThrowsError(try engine.applyMoveAndAdvance(phantomKingLeap)) { error in
            guard case ChessGameError.illegalMove = error else {
                XCTFail("Expected ChessGameError.illegalMove for phantom king leap; got \(error)")
                return
            }
        }
    }

    func testBogusPromotionThrowsIllegalMove() {
        let engine = ChessGameEngine()
        XCTAssertThrowsError(try engine.applyMoveAndAdvance(bogusPromotion)) { error in
            guard case ChessGameError.illegalMove = error else {
                XCTFail("Expected ChessGameError.illegalMove for bogus promotion; got \(error)")
                return
            }
        }
    }

    func testIllegalMoveDoesNotMutateEngineState() {
        let engine = ChessGameEngine()
        let stateBefore = engine.state
        let historyBefore = engine.moveHistory
        let legalsBefore = engine.currentLegalMoves

        XCTAssertThrowsError(try engine.applyMoveAndAdvance(emptyFromSquareMove))

        XCTAssertEqual(engine.state.currentPlayer, stateBefore.currentPlayer,
                       "currentPlayer must not change on rejected move")
        XCTAssertEqual(engine.state.halfmoveClock, stateBefore.halfmoveClock,
                       "halfmoveClock must not change on rejected move")
        XCTAssertEqual(engine.moveHistory.count, historyBefore.count,
                       "moveHistory must not grow on rejected move")
        XCTAssertEqual(Set(engine.currentLegalMoves), Set(legalsBefore),
                       "currentLegalMoves must not change on rejected move")
        XCTAssertNil(engine.result, "result must remain nil on rejected move")
    }

    func testPreviouslyLegalMoveRejectedAfterAdvancingPly() throws {
        let engine = ChessGameEngine()
        // 1. e4 is legal for white.
        XCTAssertTrue(engine.currentLegalMoves.contains(e4))
        try engine.applyMoveAndAdvance(e4)

        // After 1. e4, it's black's turn. Attempting to play 1. e4
        // again (a white move from an empty e2 square) is illegal and
        // must throw illegalMove rather than trap.
        XCTAssertFalse(engine.currentLegalMoves.contains(e4),
                       "White's e4 opener must not appear in black's legal-move list")
        XCTAssertThrowsError(try engine.applyMoveAndAdvance(e4)) { error in
            guard case ChessGameError.illegalMove = error else {
                XCTFail("Expected ChessGameError.illegalMove for stale move; got \(error)")
                return
            }
        }
    }

    // MARK: - Error precedence: result-guard before legality-guard

    func testGameAlreadyOverTakesPrecedenceOverIllegalMove() throws {
        // Fool's mate: 1. f3 e5 2. g4 Qh4# latches checkmate.
        let engine = ChessGameEngine()
        let f3 = ChessMove(fromRow: 6, fromCol: 5, toRow: 5, toCol: 5, promotion: nil)
        let blackE5 = ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)
        let g4 = ChessMove(fromRow: 6, fromCol: 6, toRow: 4, toCol: 6, promotion: nil)
        let qh4 = ChessMove(fromRow: 0, fromCol: 3, toRow: 4, toCol: 7, promotion: nil)

        try engine.applyMoveAndAdvance(f3)
        try engine.applyMoveAndAdvance(blackE5)
        try engine.applyMoveAndAdvance(g4)
        try engine.applyMoveAndAdvance(qh4)

        guard case .checkmate(winner: .black) = engine.result else {
            XCTFail("Fool's mate setup did not latch checkmate; got \(String(describing: engine.result))")
            return
        }

        // Any subsequent apply — even an obviously legal-looking move —
        // must report gameAlreadyOver, NOT illegalMove, because the
        // result-guard precedes the legality-guard inside
        // applyMoveAndAdvance.
        XCTAssertThrowsError(try engine.applyMoveAndAdvance(nf3)) { error in
            guard case ChessGameError.gameAlreadyOver = error else {
                XCTFail("Expected ChessGameError.gameAlreadyOver after checkmate; got \(error)")
                return
            }
        }

        // Even structurally garbage moves get the same verdict after
        // the game has ended — result-guard wins.
        XCTAssertThrowsError(try engine.applyMoveAndAdvance(emptyFromSquareMove)) { error in
            guard case ChessGameError.gameAlreadyOver = error else {
                XCTFail("Expected ChessGameError.gameAlreadyOver for any post-result call; got \(error)")
                return
            }
        }
    }

    // MARK: - Integration: multi-ply loop against the authoritative list

    func testMultiPlyLoopDrivenByEngineLegalMoves() throws {
        // Ten plies of "play the first legal move the engine reports"
        // must never throw. This exercises the refresh-after-apply
        // invariant: if the engine ever fails to update
        // currentLegalMoves in step with state, a subsequent apply
        // would reject its own reported move.
        let engine = ChessGameEngine()
        for ply in 0..<10 {
            guard engine.result == nil else { break }
            guard let pick = engine.currentLegalMoves.first else {
                XCTFail("Engine reported zero legal moves at ply \(ply) with nil result")
                return
            }
            XCTAssertNoThrow(try engine.applyMoveAndAdvance(pick),
                             "Engine-reported legal move must always apply cleanly (ply \(ply))")
        }
    }
}
