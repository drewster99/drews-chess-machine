import Combine
import Foundation
import Observation

/// State machine that paces a single human-vs-network game so the user
/// sees each move in sequence — human's move animates first, then a
/// breathing-room delay, then the AI is permitted to think and apply
/// its reply, then the AI's animation runs — even when the main actor
/// is laggy.
///
/// Before the pacer existed, the AI's pre-move "absorb your move"
/// sleep (`DelayedMoveEvaluationSource`) ran in parallel with the UI
/// rendering of the human's move: the 2-second timer started as soon
/// as the human's continuation resumed, and snapshots from
/// `GameWatcher` flowed into `HumanPlayWindowView` through a
/// `.throttle(0.1)` on the watcher's `PassthroughSubject`. When the
/// main thread was busy (heavy SwiftUI charts, training stats updates,
/// log flushes), both the human-move and AI-move `changes.send()`
/// events could land in the same throttle window — `snapshot()` would
/// already contain the AI's reply by the time the UI got a chance to
/// render, and the user saw the AI's piece appear instantly while the
/// human's piece was still mid-slide (or both moves animated as a
/// single reconcile).
///
/// The pacer takes ownership of the sequence:
///   1. `start(humanColor:initialSnapshot:)` arms the machine. Initial
///      phase is `.humanTurn` when the human plays white, `.aiDelay`
///      (with the breathing-room timer scheduled) when the human plays
///      black.
///   2. Every `GameWatcher.changes` emission is handed to `ingest(_:)`
///      directly (no `.throttle`). The pacer decides whether the
///      snapshot advances `displayedSnapshot` based on which side made
///      the move and which phase it's in. Snapshots that don't fit the
///      current phase are suppressed rather than overwriting the
///      displayed state.
///   3. `HumanPlayBoardView` animates its `tracked` pieces with
///      `withAnimation(_:_:completion:)` and calls
///      `onAnimationCompleted()` from the completion block. The pacer
///      then advances `.humanAnimating → .aiDelay` (scheduling the 2s
///      timer) or `.aiAnimating → .humanTurn`.
///   4. The AI side wraps its `MoveEvaluationSource` in
///      `UIGatedMoveEvaluationSource`, whose `evaluate(...)` parks in
///      `awaitAIPermission()` until the pacer reaches `.aiThinking`.
///      The pacer reaches `.aiThinking` only after the human's
///      animation has completed and the breathing-room timer has
///      elapsed.
///   5. `stop()` (called by `PlayController.stop()`) cancels the
///      delay timer and surfaces `CancellationError` on any parked
///      AI continuation so the game `Task` can unwind cleanly.
///
/// `@MainActor` + `@Observable`: the pacer's state drives SwiftUI
/// re-renders (banner text, board pieces). All mutations happen on
/// the main actor; `awaitAIPermission()` is the only API safe to call
/// from a non-main-actor caller (the game `Task`), and the hop to the
/// main actor is implicit in the `@MainActor` isolation of the method.
@MainActor
@Observable
final class HumanPlayPacer {

    /// Phase of the play loop. Equatable so the value-keyed transitions
    /// in tests and the SwiftUI banner can pattern-match cleanly.
    enum Phase: Equatable {
        /// No game running, or the game has been torn down. The pacer
        /// ignores `ingest(_:)` and `onAnimationCompleted()` in this
        /// phase.
        case idle
        /// Waiting for the human to tap a square. The next ingested
        /// snapshot is expected to carry the human's move.
        case humanTurn
        /// The human's move just landed in `displayedSnapshot` and the
        /// board is animating the slide. `onAnimationCompleted()`
        /// advances to `.aiDelay`.
        case humanAnimating(ChessMove)
        /// Post-human-move breathing room. A delay `Task` is in flight;
        /// on expiry we transition to `.aiThinking` and release any
        /// parked `awaitAIPermission()` caller.
        case aiDelay
        /// The AI has been granted permission to think. Its evaluator's
        /// `awaitAIPermission()` has resumed (or will resume the moment
        /// it parks). When the AI's move snapshot lands in
        /// `ingest(_:)` we move to `.aiAnimating`.
        case aiThinking
        /// The AI's move just landed in `displayedSnapshot` and the
        /// board is animating the slide. `onAnimationCompleted()`
        /// advances to `.humanTurn`.
        case aiAnimating(ChessMove)
        /// Terminal — game has ended. Stays here until `stop()` is
        /// called by `PlayController` (Stop / Reset).
        case gameOver(GameResult)
    }

    /// Current phase. Read by the view's banner ("Your move" / "Network
    /// thinking…") and by `awaitAIPermission()` to decide whether to
    /// park or return immediately.
    private(set) var phase: Phase = .idle

    /// The snapshot the board is currently rendering from. Distinct
    /// from the live `GameWatcher` snapshot — the live one may be one
    /// ply ahead of what the user has had time to see. Tests and the
    /// view read `displayedSnapshot.lastMove`, `.state.board`,
    /// `.moveCount`, etc.
    private(set) var displayedSnapshot: GameWatcher.Snapshot = .init()

    /// Artificial post-human-move "absorb your move" delay before the
    /// AI is permitted to think. Production value is 2 seconds; tests
    /// inject a much smaller value so the suite stays fast.
    private let postHumanDelay: Duration

    /// Color the human is playing as. Set on each `start(...)`. Used
    /// by `ingest(_:)` to attribute each new `lastMove` to a side
    /// (the snapshot's `state.currentPlayer.opposite` is always the
    /// side that just moved, since `currentPlayer` is the side about
    /// to move next).
    private var humanColor: PieceColor = .white

    /// If a game-ending snapshot arrives during an animation, the
    /// `.gameOver` transition is deferred until the animation
    /// completes so the user sees the final move slide in.
    private var pendingResult: GameResult?

    /// Parked `awaitAIPermission()` continuation, if any. The pacer
    /// resumes it (success) when transitioning to `.aiThinking`, or
    /// (cancellation) when `stop()` is called or the calling task is
    /// cancelled.
    private var aiPermissionContinuation: CheckedContinuation<Void, Error>?

    /// Background task running the `postHumanDelay` countdown.
    /// Cancelled by `stop()`; replaced on every entry into `.aiDelay`.
    private var aiDelayTask: Task<Void, Never>?

    init(postHumanDelay: Duration = .seconds(2)) {
        self.postHumanDelay = postHumanDelay
    }

    // MARK: - Lifecycle

    /// Begin a new game. Tears down any prior state, then sets the
    /// initial phase based on which color the human is playing:
    /// - human plays white → `.humanTurn` (wait for human's tap).
    /// - human plays black → `.aiDelay` (schedule the breathing-room
    ///   timer so the AI doesn't snap-move at game start).
    func start(humanColor: PieceColor, initialSnapshot: GameWatcher.Snapshot) {
        stop()
        self.humanColor = humanColor
        self.displayedSnapshot = initialSnapshot
        self.pendingResult = nil
        if humanColor == .white {
            phase = .humanTurn
        } else {
            phase = .aiDelay
            scheduleAIDelay()
        }
    }

    /// Tear down. Cancels the delay timer and resumes any parked AI
    /// continuation with `CancellationError` so the game `Task` can
    /// unwind cleanly. `displayedSnapshot` is intentionally left
    /// untouched — the window may still be visible after `stop()`
    /// (game over banner with the final position) and snapping the
    /// board back to the starting state on stop would be jarring.
    ///
    /// `pendingResult` is dropped on stop. A `.gameOver` snapshot that
    /// arrived mid-animation but hadn't yet been promoted by the
    /// animation-complete callback is silently discarded — `stop()` is
    /// always followed by either a new `start(...)` (which would
    /// overwrite the result anyway) or a window teardown (where
    /// nobody cares).
    func stop() {
        aiDelayTask?.cancel()
        aiDelayTask = nil
        if let cont = aiPermissionContinuation {
            aiPermissionContinuation = nil
            cont.resume(throwing: CancellationError())
        }
        pendingResult = nil
        phase = .idle
    }

    // MARK: - Inputs

    /// Hand in a new `GameWatcher.Snapshot`. Called directly from the
    /// view's `.onReceive(gameWatcher.changes)` — there is no longer
    /// any `.throttle` between the watcher and the pacer.
    ///
    /// The decision tree:
    ///   - If the snapshot carries a game-end `result` and we haven't
    ///     already noted it, stash it in `pendingResult` so the
    ///     `.gameOver` transition can fire after the in-flight
    ///     animation completes.
    ///   - If the snapshot's `moveCount` hasn't advanced past the
    ///     displayed one, there's no new move to animate; return.
    ///   - Otherwise attribute the move to a side via
    ///     `currentPlayer.opposite` (the side that just moved). For
    ///     the matching expected phase (`.humanTurn` for a human
    ///     move, `.aiThinking` for an AI move) advance to the
    ///     corresponding animating phase and update
    ///     `displayedSnapshot`. For any other phase, suppress the
    ///     update — racing the in-flight animation with a stale or
    ///     premature snapshot is exactly the bug the pacer prevents.
    func ingest(_ snapshot: GameWatcher.Snapshot) {
        if snapshot.result != nil, displayedSnapshot.result == nil, pendingResult == nil {
            pendingResult = snapshot.result
        }

        // `moveCount` is the live ply counter from `GameWatcher`. Using
        // it (rather than `lastMove != displayed.lastMove`) handles the
        // rook-oscillation case where the same `(from, to)` repeats
        // later in the game, and the `gameEnded` case where the
        // watcher zeroes `moveCount` while `lastMove` stays put.
        guard snapshot.moveCount > displayedSnapshot.moveCount,
              let move = snapshot.lastMove
        else {
            return
        }

        // `currentPlayer` in the snapshot is the side about to move,
        // i.e., the opponent of whoever just played.
        let mover = snapshot.state.currentPlayer.opposite

        if mover == humanColor {
            if case .humanTurn = phase {
                displayedSnapshot = snapshot
                phase = .humanAnimating(move)
            }
        } else {
            if case .aiThinking = phase {
                displayedSnapshot = snapshot
                phase = .aiAnimating(move)
            }
        }
    }

    /// Called by `HumanPlayBoardView` from its
    /// `withAnimation(_:_:completion:)` completion block once a
    /// piece-slide animation has finished. Other phases are
    /// no-ops — including a completion firing for the initial
    /// `seedTracked` render, which should not change phase.
    func onAnimationCompleted() {
        switch phase {
        case .humanAnimating:
            if let result = pendingResult {
                pendingResult = nil
                phase = .gameOver(result)
            } else {
                phase = .aiDelay
                scheduleAIDelay()
            }
        case .aiAnimating:
            if let result = pendingResult {
                pendingResult = nil
                phase = .gameOver(result)
            } else {
                phase = .humanTurn
            }
        case .idle, .humanTurn, .aiDelay, .aiThinking, .gameOver:
            break
        }
    }

    // MARK: - Outputs

    /// Called by `UIGatedMoveEvaluationSource.evaluate(...)` before it
    /// runs the network forward pass. Returns immediately if the
    /// pacer is already in `.aiThinking`; otherwise parks until the
    /// pacer transitions there. Throws `CancellationError` when the
    /// pacer is `stop()`-ed or the calling task is cancelled.
    func awaitAIPermission() async throws {
        if case .aiThinking = phase {
            return
        }
        try await withTaskCancellationHandler(
            operation: {
                try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                    // Re-check phase after entering the continuation
                    // closure — the body of `withCheckedThrowing-
                    // Continuation` runs synchronously on the calling
                    // executor (main actor here), so this is the
                    // atomic install point.
                    if case .aiThinking = phase {
                        cont.resume()
                        return
                    }
                    // Defensive: at most one AI evaluator should be
                    // waiting per game. If somehow a prior one is
                    // still parked, surface a cancel on it so we
                    // never leak a continuation.
                    if let existing = aiPermissionContinuation {
                        existing.resume(throwing: CancellationError())
                    }
                    aiPermissionContinuation = cont
                }
            },
            onCancel: { [weak self] in
                // The cancellation handler runs on the cancelling
                // thread, not the main actor — hop back to the
                // pacer's isolation before touching its state.
                Task { @MainActor [weak self] in
                    self?.cancelPendingAIPermission()
                }
            }
        )
    }

    // MARK: - Private

    private func scheduleAIDelay() {
        aiDelayTask?.cancel()
        let delay = postHumanDelay
        aiDelayTask = Task { @MainActor [weak self] in
            do {
                try await Task.sleep(for: delay)
            } catch {
                // Cancelled — `stop()` or a new `start(...)` superseded
                // this delay; do not advance phase.
                return
            }
            self?.advanceFromAIDelayIfReady()
        }
    }

    private func advanceFromAIDelayIfReady() {
        guard case .aiDelay = phase else { return }
        phase = .aiThinking
        if let cont = aiPermissionContinuation {
            aiPermissionContinuation = nil
            cont.resume()
        }
    }

    private func cancelPendingAIPermission() {
        if let cont = aiPermissionContinuation {
            aiPermissionContinuation = nil
            cont.resume(throwing: CancellationError())
        }
    }
}
