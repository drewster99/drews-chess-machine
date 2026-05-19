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

    /// Per-test counter that mints monotonic `eventSeq` values so
    /// every fabricated snapshot is "ahead" of the prior one — mirrors
    /// production where `GameWatcher` increments the counter under its
    /// lock for every mutation. Tests that need to deliberately re-
    /// ingest the same snapshot (the "duplicate snapshot is no-op" path)
    /// hold on to the returned value and pass it back unchanged.
    private var nextEventSeq: UInt64 = 0

    private func snapshot(
        moveCount: Int,
        lastMove: ChessMove?,
        sideToMove: PieceColor,
        result: GameResult? = nil
    ) -> GameWatcher.Snapshot {
        nextEventSeq += 1
        var s = GameWatcher.Snapshot()
        s.eventSeq = nextEventSeq
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

    func testTaskCancellationCancelsParkedAwaitAIPermission() async throws {
        // Regression: the `withTaskCancellationHandler` branch of
        // `awaitAIPermission` only fires when the calling Task is
        // cancelled (NOT when the pacer is stopped — that path is
        // already covered by `testStopCancelsParkedAwaitAIPermission`).
        // The onCancel handler must hop back to the main actor and
        // resume the parked continuation with `CancellationError`,
        // without disturbing the pacer's own state.
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        pacer.ingest(snapshot(moveCount: 1, lastMove: humanWhiteMove(), sideToMove: .black))
        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .aiDelay)

        let cancelled = expectation(description: "awaitAIPermission threw CancellationError via Task.cancel")
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

        try await Task.sleep(for: .milliseconds(50))
        XCTAssertEqual(pacer.phase, .aiDelay, "pacer should still be parked while Task awaits")

        task.cancel()

        await fulfillment(of: [cancelled], timeout: 2.0)
        XCTAssertEqual(
            pacer.phase, .aiDelay,
            "pacer's own state is untouched by Task cancellation — only the parked continuation was cancelled"
        )
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
        // The result has not yet propagated into displayedSnapshot —
        // the banner should still show the in-progress state until the
        // animation completes.
        XCTAssertNil(pacer.displayedSnapshot.result)

        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .gameOver(.checkmate(winner: .white)))
        XCTAssertEqual(pacer.displayedSnapshot.result, .checkmate(winner: .white))
    }

    /// Regression: in production main-actor flows, the `didApplyMove`
    /// and `gameEnded` emissions both happen on `ChessMachine`'s
    /// delegate queue back-to-back; by the time the main-queue handler
    /// for the FIRST emission runs, the watcher has already absorbed
    /// the second mutation and `gameWatcher.snapshot()` returns the
    /// post-`gameEnded` state (moveCount=0, result set, lastMove=
    /// final move). The pacer must still recognize the unrecorded
    /// final move in that coalesced snapshot, animate it, and only
    /// then transition to `.gameOver` — otherwise the window stays
    /// on "Waiting…" indefinitely because no `pieces` change fires
    /// `.onChange` and `onAnimationCompleted` never runs.
    ///
    /// The user originally reported this as the post-promotion variant
    /// of the prior "stuck on Waiting…" bug: white promoted a pawn to
    /// queen with check-mate, the engine emitted both events on the
    /// delegate queue in rapid succession, and the window hung with
    /// the pre-promotion board still on screen.
    func testCoalescedFinalMoveSnapshotAnimatesThenEnds() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )

        // Coalesced snapshot: moveCount zeroed by gameEnded, result
        // set, lastMove still reflects the final move. (lastMove != the
        // displayed snapshot's lastMove because the displayed snapshot
        // is the pre-move state.)
        let hm = humanWhiteMove()
        var coalesced = snapshot(
            moveCount: 0,
            lastMove: hm,
            sideToMove: .black,
            result: .checkmate(winner: .white)
        )
        coalesced.lastGameStats = GameStats(
            totalMoves: 1,
            whiteMoves: 1,
            blackMoves: 0,
            whiteThinkingTimeMs: 0,
            blackThinkingTimeMs: 0,
            totalGameTimeMs: 0
        )
        pacer.ingest(coalesced)

        // The final move must animate first — banner stays mid-game
        // until completion fires.
        XCTAssertEqual(pacer.phase, .humanAnimating(hm))
        XCTAssertEqual(pacer.displayedSnapshot.lastMove, hm)
        XCTAssertNil(
            pacer.displayedSnapshot.result,
            "result is deferred — banner must not jump ahead of the slide"
        )
        XCTAssertEqual(
            pacer.history,
            [HumanPlayPacer.HistoryEntry(plyNumber: 1, side: .white, move: hm)],
            "the coalesced final move must still be recorded in history"
        )

        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .gameOver(.checkmate(winner: .white)))
        XCTAssertEqual(pacer.displayedSnapshot.result, .checkmate(winner: .white))
        XCTAssertEqual(pacer.displayedSnapshot.lastGameStats?.totalMoves, 1)
    }

    /// Regression: in production, the `gameEnded` snapshot arrives as a
    /// SEPARATE `GameWatcher.changes` emission AFTER the final move's
    /// `didApplyMove` emission, and the watcher zeroes `moveCount` on
    /// game end. The pacer must therefore stash the result+stats from
    /// the game-end snapshot (whose moveCount=0 won't advance the
    /// display) and promote them onto `displayedSnapshot` when the
    /// final-move animation completes.
    ///
    /// Before this fix, the game-end fields were stashed but never
    /// copied into `displayedSnapshot.result` / `.lastGameStats`, so
    /// the window's banner stayed on "Waiting…" indefinitely after a
    /// natural game end (the user reported this with a live-trainer
    /// game that ended at ply 46).
    func testSeparateGameEndedSnapshotPromotesResultIntoDisplayedSnapshot() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )

        // didApplyMove snapshot: result still nil, moveCount=1, lastMove set.
        let hm = humanWhiteMove()
        pacer.ingest(snapshot(
            moveCount: 1,
            lastMove: hm,
            sideToMove: .black,
            result: nil
        ))
        XCTAssertEqual(pacer.phase, .humanAnimating(hm))

        // gameEnded snapshot: moveCount zeroed by GameWatcher, result set.
        // Carry lastGameStats so the banner+status row can show the final
        // ply count after the game ends.
        var endSnap = snapshot(
            moveCount: 0,
            lastMove: hm,
            sideToMove: .black,
            result: .drawByThreefoldRepetition
        )
        endSnap.lastGameStats = GameStats(
            totalMoves: 1,
            whiteMoves: 1,
            blackMoves: 0,
            whiteThinkingTimeMs: 0,
            blackThinkingTimeMs: 0,
            totalGameTimeMs: 0
        )
        pacer.ingest(endSnap)
        // Still animating — the deferred end-snapshot must not jump the
        // banner ahead of the slide.
        XCTAssertEqual(pacer.phase, .humanAnimating(hm))
        XCTAssertNil(pacer.displayedSnapshot.result)

        pacer.onAnimationCompleted()
        XCTAssertEqual(pacer.phase, .gameOver(.drawByThreefoldRepetition))
        XCTAssertEqual(pacer.displayedSnapshot.result, .drawByThreefoldRepetition)
        XCTAssertEqual(pacer.displayedSnapshot.lastGameStats?.totalMoves, 1)
    }

    // MARK: - Seeded history (Revert to here)

    /// Reverting to a past ply re-launches the game from an
    /// intermediate position; the pacer's `start` accepts the kept
    /// history so the sidebar continues to display the moves that
    /// led to the revert point, and the phase derivation looks at
    /// `initialSnapshot.state.currentPlayer` rather than blindly
    /// assuming white moves first — after an odd-ply revert, black
    /// is on move and a white-playing human enters `.aiDelay` even
    /// though the pacer was just `start(...)`-ed.
    func testStartWithSeededHistoryPreservesEntriesAndPicksPhaseFromCurrentPlayer() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        let hm = humanWhiteMove()
        let seed = [
            HumanPlayPacer.HistoryEntry(plyNumber: 1, side: .white, move: hm)
        ]
        // After ply 1, side to move is black.
        let snap = snapshot(moveCount: 1, lastMove: hm, sideToMove: .black)
        pacer.start(humanColor: .white, initialSnapshot: snap, seedingHistory: seed)

        XCTAssertEqual(pacer.history, seed)
        XCTAssertEqual(pacer.displayedSnapshot.lastMove, hm)
        XCTAssertEqual(
            pacer.phase, .aiDelay,
            "white-playing human + black-to-move snapshot must enter .aiDelay so the AI thinks next"
        )
    }

    func testStartWithSeededHistoryEntersHumanTurnWhenItIsHumansMove() {
        // Even-ply revert: white-to-move next; human plays white →
        // phase should be .humanTurn so the board accepts the next
        // tap immediately.
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        let hm = humanWhiteMove()
        let am = aiBlackMove()
        let seed = [
            HumanPlayPacer.HistoryEntry(plyNumber: 1, side: .white, move: hm),
            HumanPlayPacer.HistoryEntry(plyNumber: 2, side: .black, move: am)
        ]
        let snap = snapshot(moveCount: 2, lastMove: am, sideToMove: .white)
        pacer.start(humanColor: .white, initialSnapshot: snap, seedingHistory: seed)

        XCTAssertEqual(pacer.history, seed)
        XCTAssertEqual(pacer.phase, .humanTurn)
    }

    // MARK: - Move history

    func testHistoryAppendsOneEntryPerIngestedMove() async throws {
        let pacer = HumanPlayPacer(postHumanDelay: .milliseconds(20))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        XCTAssertEqual(pacer.history, [])

        let hm = humanWhiteMove()
        pacer.ingest(snapshot(moveCount: 1, lastMove: hm, sideToMove: .black))
        XCTAssertEqual(pacer.history, [
            HumanPlayPacer.HistoryEntry(plyNumber: 1, side: .white, move: hm)
        ])

        pacer.onAnimationCompleted()
        try await Task.sleep(for: .milliseconds(80))
        XCTAssertEqual(pacer.phase, .aiThinking)

        let am = aiBlackMove()
        pacer.ingest(snapshot(moveCount: 2, lastMove: am, sideToMove: .white))
        XCTAssertEqual(pacer.history, [
            HumanPlayPacer.HistoryEntry(plyNumber: 1, side: .white, move: hm),
            HumanPlayPacer.HistoryEntry(plyNumber: 2, side: .black, move: am)
        ])
    }

    func testHistoryDoesNotRecordSuppressedSnapshots() {
        // If an AI snapshot arrives while we are still in .humanAnimating
        // (the bug the pacer was originally built to prevent), the pacer
        // suppresses the snapshot and must NOT add it to history — only
        // moves the user actually sees on the board are recorded.
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        let hm = humanWhiteMove()
        pacer.ingest(snapshot(moveCount: 1, lastMove: hm, sideToMove: .black))
        XCTAssertEqual(pacer.history.count, 1)

        pacer.ingest(snapshot(moveCount: 2, lastMove: aiBlackMove(), sideToMove: .white))
        XCTAssertEqual(pacer.history.count, 1, "AI snapshot arriving during human animation must not record history")
    }

    func testHistoryResetsOnStart() {
        let pacer = HumanPlayPacer(postHumanDelay: .seconds(10))
        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        pacer.ingest(snapshot(moveCount: 1, lastMove: humanWhiteMove(), sideToMove: .black))
        XCTAssertEqual(pacer.history.count, 1)

        pacer.start(
            humanColor: .white,
            initialSnapshot: snapshot(moveCount: 0, lastMove: nil, sideToMove: .white)
        )
        XCTAssertEqual(pacer.history, [])
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
