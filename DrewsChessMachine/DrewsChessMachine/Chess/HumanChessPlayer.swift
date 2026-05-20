import Foundation
import os

/// A `ChessPlayer` whose moves come from a human user interacting with the
/// SwiftUI board, rather than from a neural-network policy.
///
/// `onChooseNextMove` runs on the `ChessMachine` game task. It publishes the
/// turn's legal moves through `onTurnBegin` (the UI consumes that to enable
/// taps + show highlights), then suspends on a `CheckedContinuation` until
/// the UI hands a move back via `submit(_:)`. Cancellation of the parent
/// task — the user pressed Stop, or another play session started — resumes
/// the continuation with `CancellationError` so `beginNewGame` unwinds
/// cleanly instead of stalling forever.
///
/// `@unchecked Sendable` because access to `pending` is guarded by an
/// `OSAllocatedUnfairLock`, matching the rest of the project's small
/// lock-protected helper classes. There's never any expectation of
/// concurrent calls to `onChooseNextMove` (the game loop is serial) — the
/// lock just makes `submit(_:)` / `cancelPendingChoice()` safe to call from
/// arbitrary actors (the main actor, in practice).
final class HumanChessPlayer: ChessPlayer, @unchecked Sendable {
    let identifier: String = UUID().uuidString
    let name: String

    /// Fires on the main actor at the start of each of this player's turns.
    /// The UI uses this to record the legal-move set, light up the board's
    /// tap layer, and display whose turn it is. Cleared after the user
    /// submits a move via `onTurnEnd`.
    private let onTurnBegin: @MainActor ([ChessMove]) -> Void

    /// Fires on the main actor once a submitted move resumed the
    /// continuation (or the game was cancelled). The UI uses it to clear
    /// the legal-move highlights and the selected-from-square.
    private let onTurnEnd: @MainActor () -> Void

    /// Set, under `lock`, while `onChooseNextMove` is parked on a
    /// continuation. `submit(_:)` and `cancelPendingChoice()` consume it.
    private struct Pending {
        let continuation: CheckedContinuation<ChessMove, Error>
        let legalMoves: [ChessMove]
    }

    /// Lock-protected state for the current turn. `cancelled` is set
    /// by `cancelPendingChoice()` and lets the continuation-install
    /// closure detect a cancel that arrived BEFORE the continuation
    /// was stored — without it, the install would happen against a
    /// `nil` slot, `cancelPendingChoice` would no-op, and the just-
    /// installed continuation would never resume (and on scope exit
    /// `withCheckedThrowingContinuation` would assert). The flag is
    /// reset at the start of every `onChooseNextMove` turn.
    private struct State {
        var pending: Pending?
        var cancelled: Bool
    }
    private let lock = OSAllocatedUnfairLock<State>(
        initialState: State(pending: nil, cancelled: false)
    )

    init(
        name: String,
        onTurnBegin: @escaping @MainActor ([ChessMove]) -> Void,
        onTurnEnd: @escaping @MainActor () -> Void
    ) {
        self.name = name
        self.onTurnBegin = onTurnBegin
        self.onTurnEnd = onTurnEnd
    }

    func onNewGame(_ isWhite: Bool) {}

    func onGameEnded(_ result: GameResult, finalState: GameState) {}

    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove {
        guard !legalMoves.isEmpty else {
            throw ChessPlayerError.noLegalMoves
        }
        // Capture the cancellation state synchronously before we suspend.
        // If the parent task is already cancelled, fail fast instead of
        // installing a continuation the UI may never resume.
        try Task.checkCancellation()

        // Fresh turn: clear any stale cancel flag from a prior turn
        // (the previous turn always consumed its continuation, so
        // `pending` is necessarily nil here).
        lock.withLock { state in
            state.cancelled = false
        }

        let movesForUI = legalMoves
        await MainActor.run { onTurnBegin(movesForUI) }

        do {
            let move = try await withTaskCancellationHandler(
                operation: {
                    try await withCheckedThrowingContinuation { cont in
                        // Install the continuation under the lock. If
                        // a cancel already raced ahead of us (parent
                        // task cancelled between `Task.checkCancellation`
                        // above and this line — `cancelPendingChoice`
                        // would have set `cancelled` against an empty
                        // slot), resume-throwing immediately instead
                        // of parking a continuation nothing will wake.
                        let resumeWithCancel = lock.withLock { state -> Bool in
                            if state.cancelled {
                                return true
                            }
                            state.pending = Pending(
                                continuation: cont,
                                legalMoves: legalMoves
                            )
                            return false
                        }
                        if resumeWithCancel {
                            cont.resume(throwing: CancellationError())
                        }
                    }
                },
                onCancel: { [self] in
                    // Triggered if the parent task is cancelled while we
                    // are suspended. Resume with `CancellationError` so
                    // `ChessMachine.runGameLoop` propagates the cancel
                    // upward instead of treating it as a forfeit.
                    self.cancelPendingChoice()
                }
            )
            await MainActor.run { onTurnEnd() }
            return move
        } catch {
            await MainActor.run { onTurnEnd() }
            throw error
        }
    }

    /// Hand a chosen move back to the suspended `onChooseNextMove`. The move
    /// is validated against the legal-move list captured at the start of
    /// the turn; an illegal submission is rejected (returns `false`) and the
    /// continuation stays installed so the UI can correct itself rather
    /// than abort the game. Returns `true` if the move was accepted and
    /// the game loop will advance.
    @discardableResult
    func submit(_ move: ChessMove) -> Bool {
        let resumed: Bool = lock.withLock { state in
            guard let pending = state.pending else { return false }
            guard pending.legalMoves.contains(move) else { return false }
            state.pending = nil
            pending.continuation.resume(returning: move)
            return true
        }
        return resumed
    }

    /// Resume the suspended `onChooseNextMove` with `CancellationError`,
    /// or — if no continuation has been installed yet — latch a
    /// `cancelled` flag that the install closure will consult when it
    /// runs. Idempotent.
    func cancelPendingChoice() {
        lock.withLock { state in
            if let pending = state.pending {
                state.pending = nil
                state.cancelled = true
                pending.continuation.resume(throwing: CancellationError())
                return
            }
            // No continuation yet — record the cancel so the next
            // install bails out immediately.
            state.cancelled = true
        }
    }

    /// Snapshot the currently-pending legal-move list (or nil if no turn
    /// is active). Used by UI sanity checks; the UI's primary path is
    /// `onTurnBegin` / `onTurnEnd`, not polling.
    func pendingLegalMovesSnapshot() -> [ChessMove]? {
        lock.withLock { $0.pending?.legalMoves }
    }
}
