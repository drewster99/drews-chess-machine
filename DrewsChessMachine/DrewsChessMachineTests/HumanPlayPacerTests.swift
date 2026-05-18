import XCTest
@testable import DrewsChessMachine

/// State-machine coverage for `HumanPlayPacer`.
///
/// The pacer was added to fix a UI-pacing bug: under main-actor lag,
/// the AI's pre-move 2-second sleep ran in parallel with the UI
/// rendering of the human's move, so the human's move and the AI's
/// reply could both land in the same throttle window and the AI
/// appeared to move instantly while the human's piece was still
/// mid-slide. The pacer takes explicit ownership of the per-ply
/// sequence: human move → animate → 2s delay → permit AI to think →
/// AI move → animate → next turn. These tests pin the transitions
/// without requiring any Metal/MPSGraph or live game loop.
@MainActor
final class HumanPlayPacerTests: XCTestCase {

    // MARK: - Snapshot fixtures

    private func snapshot(
        moveCount: Int,
        lastMove: ChessMove?,
        sideToMove: PieceColor,
        result: GameResult? = nil
    ) -> GameWatcher.Snapshot {
        var s = GameWatcher.Snapshot()
        s.moveCount = moveCount
        s.lastMove = lastMove
        s.result = result
        s.state = GameState(
            board: GameState.starting.board,
            currentPlayer: sideToMove,
            whiteKingsideCastle: true,
            whiteQueensideCastle: true,
            blackKingsideCastle: true,
            blackQueensideCastle: true,
            enPassantSquare: nil,
            halfmoveClock: 0
        )
        return s
    }

    private func humanWhiteMove() -> ChessMove {
        // e2-e4 — the side to move after this is black.
        ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
    }

    private func aiBlackMove() -> ChessMove {
        // e7-e5 — side to move after this is white.
        ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)
    }

    private func aiWhiteMove() -> ChessMove {
        // d2-d4 — side to move after this is black. Used in human-plays-black tests.
        ChessMove(fromRow: 6, fromCol: 3, toRow: 4, toCol: 3, promotion: nil)
    }

    // MARK: - Start

    func testStartHumanWhiteEntersHumanTurn() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        XCTAssertEqual(pacer.phase, .humanTurn)
        XCTAssertNil(pacer.displayedSnapshot.lastMove)
        XCTAssertEqual(pacer.displayedSnapshot.moveCount, 0)
    }

    func testStartHumanBlackEntersAIDelay() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .black,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        XCTAssertEqual(pacer.phase, .aiDelay)
    }

    // MARK: - Human move advances

    func testHumanMoveSnapshotAdvancesToHumanAnimating() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        let move = humanWhiteMove()
        pacer.ingest(snapshot(moveCount: 1, lastMove: move, sideToMove: .black))
        XCTAssertEqual(pacer.phase, .humanAnimating(move))
        XCTAssertEqual(pacer.displayedSnapshot.lastMove, move)
        XCTAssertEqual(pacer.displayedSnapshot.moveCount, 1)
    }

    func testHumanAnimationCompletionAdvancesToAIDelay() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        pacer.ingest(snapshot(moveCount: 1, lastMove: humanWhiteMove(), sideToMove: .black))
        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .aiDelay)
    }

    // MARK: - Regression: AI snapshot arriving early is rejected

    /// Reproduces the bug condition the pacer fixes. In the old code, an
    /// AI snapshot arriving on the heels of the human's snapshot (the
    /// throttle-collapse scenario) would either be applied as a single
    /// reconcile (both pieces animate simultaneously) or animated
    /// back-to-back so closely that the AI appeared to move instantly.
    /// With the pacer, the AI snapshot is *suppressed* while the pacer
    /// is in `.humanAnimating`/`.aiDelay`/etc.; in production the gated
    /// source prevents the AI's evaluator from running at all in those
    /// phases, so this is also a defensive check against any future
    /// regression that bypasses the gate.
    func testAIMoveSnapshotBeforeAIThinkingIsSuppressed() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        let hm = humanWhiteMove()
        pacer.ingest(snapshot(moveCount: 1, lastMove: hm, sideToMove: .black))
        XCTAssertEqual(pacer.phase, .humanAnimating(hm))

        // Lag scenario: an AI snapshot lands while we're still in
        // .humanAnimating. The pacer must not update displayedSnapshot
        // or transition to .aiAnimating.
        pacer.ingest(snapshot(moveCount: 2, lastMove: aiBlackMove(), sideToMove: .white))
        XCTAssertEqual(pacer.phase, .humanAnimating(hm))
        XCTAssertEqual(pacer.displayedSnapshot.lastMove, hm)
        XCTAssertEqual(pacer.displayedSnapshot.moveCount, 1)
    }

    func testAIMoveSnapshotInAIDelayIsAlsoSuppressed() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        pacer.ingest(snapshot(moveCount: 1, lastMove: humanWhiteMove(), sideToMove: .black))
        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .aiDelay)

        pacer.ingest(snapshot(moveCount: 2, lastMove: aiBlackMove(), sideToMove: .white))
        XCTAssertEqual(pacer.phase, .aiDelay)
        XCTAssertEqual(pacer.displayedSnapshot.moveCount, 1)
    }

    // MARK: - awaitAIPermission

    func testAwaitAIPermissionResumesWhenPacerReachesAIThinking() async throws {
        // Use a short delay so the timer actually fires.
        let pacer = HumanPlayPacer(postHumanDelay: .milliseconds(50))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        pacer.ingest(snapshot(moveCount: 1, lastMove: humanWhiteMove(), sideToMove: .black))
        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .aiDelay)

        let resumed = expectation(description: "awaitAIPermission resumed")
        let task = Task {
            try await pacer.awaitAIPermission()
            resumed.fulfill()
        }

        await fulfillment(of: [resumed], timeout: 2.0)
        XCTAssertEqual(pacer.phase, .aiThinking)
        _ = task
    }

    func testAwaitAIPermissionReturnsImmediatelyIfAlreadyAIThinking() async throws {
        let pacer = HumanPlayPacer(postHumanDelay: .milliseconds(20))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        pacer.ingest(snapshot(moveCount: 1, lastMove: humanWhiteMove(), sideToMove: .black))
        pacer.onAnimationCompleted()

        // Wait long enough for the aiDelay timer to fire.
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertEqual(pacer.phase, .aiThinking)

        // The AI's first call now arrives — should return without parking.
        try await pacer.awaitAIPermission()
        XCTAssertEqual(pacer.phase, .aiThinking)
    }

    func testStopCancelsParkedAwaitAIPermission() async throws {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        pacer.ingest(snapshot(moveCount: 1, lastMove: humanWhiteMove(), sideToMove: .black))
        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .aiDelay)

        let cancelled = expectation(description: "awaitAIPermission threw cancellation")
        let task = Task {
            do {
                try await pacer.awaitAIPermission()
                XCTFail("expected CancellationError")
            } catch is CancellationError {
                cancelled.fulfill()
            } catch {
                XCTFail("unexpected error: \(error)")
            }
        }

        // Give the parked task a moment to install its continuation.
        try await Task.sleep(for: .milliseconds(50))

        pacer.stop()

        await fulfillment(of: [cancelled], timeout: 2.0)
        XCTAssertEqual(pacer.phase, .idle)
        _ = task
    }

    // MARK: - AI move → animation → human turn

    func testAIMoveSnapshotInAIThinkingAdvancesToAIAnimating() async throws {
        let pacer = HumanPlayPacer(postHumanDelay: .milliseconds(20))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        pacer.ingest(snapshot(moveCount: 1, lastMove: humanWhiteMove(), sideToMove: .black))
        pacer.onAnimationCompleted()
        try await Task.sleep(for: .milliseconds(150))
        XCTAssertEqual(pacer.phase, .aiThinking)

        let am = aiBlackMove()
        pacer.ingest(snapshot(moveCount: 2, lastMove: am, sideToMove: .white))
        XCTAssertEqual(pacer.phase, .aiAnimating(am))
        XCTAssertEqual(pacer.displayedSnapshot.lastMove, am)

        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .humanTurn)
    }

    // MARK: - Human-plays-black: AI moves first

    func testHumanBlackAITimerReleasesAIThenAnimates() async throws {
        let pacer = HumanPlayPacer(postHumanDelay: .milliseconds(50))
        pacer.start(
            humanColor: .black,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        XCTAssertEqual(pacer.phase, .aiDelay)

        let permitted = expectation(description: "AI permitted")
        let task = Task {
            try await pacer.awaitAIPermission()
            permitted.fulfill()
        }
        await fulfillment(of: [permitted], timeout: 2.0)
        XCTAssertEqual(pacer.phase, .aiThinking)
        _ = task

        let am = aiWhiteMove()
        pacer.ingest(snapshot(moveCount: 1, lastMove: am, sideToMove: .black))
        XCTAssertEqual(pacer.phase, .aiAnimating(am))

        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .humanTurn)
    }

    // MARK: - Game-over buffering

    func testGameOverDuringHumanAnimationDeferredUntilCompletion() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        let hm = humanWhiteMove()
        pacer.ingest(snapshot(
            moveCount: 1,
            lastMove: hm,
            sideToMove: .black,
            result: .checkmate(winner: .white)
        ))
        // Still animating — gameOver is buffered.
        XCTAssertEqual(pacer.phase, .humanAnimating(hm))

        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .gameOver(.checkmate(winner: .white)))
    }

    // MARK: - Idempotent / out-of-order ingest

    func testDuplicateSnapshotIsNoOp() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        let hm = humanWhiteMove()
        let s1 = snapshot(moveCount: 1, lastMove: hm, sideToMove: .black)
        pacer.ingest(s1)
        XCTAssertEqual(pacer.phase, .humanAnimating(hm))

        // Re-ingest the same snapshot — moveCount didn't advance.
        pacer.ingest(s1)
        XCTAssertEqual(pacer.phase, .humanAnimating(hm))
    }
}
