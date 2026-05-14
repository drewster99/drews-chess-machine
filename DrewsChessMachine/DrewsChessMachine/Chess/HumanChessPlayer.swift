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
    private let lock = OSAllocatedUnfairLock<Pending?>(initialState: nil)

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

        let movesForUI = legalMoves
        await MainActor.run { onTurnBegin(movesForUI) }

        do {
            let move = try await withTaskCancellationHandler(
                operation: {
                    try await withCheckedThrowingContinuation { cont in
                        // The UI may try to submit a move BEFORE we get
                        // here — that path stores the continuation and
                        // expects `submit` to consume it. We are the only
                        // installer of a continuation, and the lock
                        // serializes us against `submit` / `cancel`.
                        lock.withLock { slot in
                            slot = Pending(continuation: cont, legalMoves: legalMoves)
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
        let resumed: Bool = lock.withLock { slot in
            guard let pending = slot else { return false }
            guard pending.legalMoves.contains(move) else { return false }
            slot = nil
            pending.continuation.resume(returning: move)
            return true
        }
        return resumed
    }

    /// Resume the suspended `onChooseNextMove` with `CancellationError`.
    /// Idempotent: if no choice is pending, does nothing.
    func cancelPendingChoice() {
        lock.withLock { slot in
            guard let pending = slot else { return }
            slot = nil
            pending.continuation.resume(throwing: CancellationError())
        }
    }

    /// Snapshot the currently-pending legal-move list (or nil if no turn is
    /// active). Used by tests and by UI sanity checks; the UI's primary
    /// path is `onTurnBegin` / `onTurnEnd`, not polling.
    func pendingLegalMovesSnapshot() -> [ChessMove]? {
        lock.withLock { $0?.legalMoves }
    }
}
