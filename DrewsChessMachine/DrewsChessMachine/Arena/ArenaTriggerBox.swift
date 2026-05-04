import Foundation
import os

/// Lock-protected trigger inbox for the arena coordinator task. The
/// training worker fires the trigger when the 30-minute auto cadence
/// elapses; the UI fires it via the Run Arena button. The arena
/// coordinator task polls `consume()` in its main loop and runs an
/// arena whenever the trigger is pending. `recordArenaCompleted()`
/// resets the "last arena" timestamp so the auto-fire math stays
/// accurate across both the automatic and manual paths.
///
/// State protected by `OSAllocatedUnfairLock<State>`; each method
/// holds the lock briefly and never across an `await`.
final class ArenaTriggerBox: @unchecked Sendable {
    private struct State {
        var pending: Bool = false
        var lastArenaTime: Date
    }
    private let lock: OSAllocatedUnfairLock<State>

    init(startTime: Date = Date()) {
        self.lock = OSAllocatedUnfairLock(initialState: State(lastArenaTime: startTime))
    }

    /// Check whether enough wall clock has elapsed since the last
    /// arena to auto-trigger another one. Returns false if a trigger
    /// is already pending, so the training worker doesn't queue
    /// multiple auto-triggers for the same deadline.
    func shouldAutoTrigger(interval: TimeInterval) -> Bool {
        lock.withLock { state in
            if state.pending { return false }
            return Date().timeIntervalSince(state.lastArenaTime) >= interval
        }
    }

    /// Set the pending flag. The arena coordinator's next poll will
    /// consume it and start an arena.
    func trigger() {
        lock.withLock { $0.pending = true }
    }

    /// Poll the trigger. Returns true and clears the pending flag if
    /// a trigger was waiting; returns false otherwise.
    func consume() -> Bool {
        lock.withLock { state in
            if state.pending {
                state.pending = false
                return true
            }
            return false
        }
    }

    /// Record that an arena just finished. Resets the wall-clock
    /// reference for subsequent `shouldAutoTrigger` checks so the
    /// next auto-fire happens `interval` seconds from now, not from
    /// the previous last-arena time. Also clears the pending flag:
    /// the training worker runs in parallel with the arena and can
    /// stamp `pending` mid-arena once elapsed time crosses `interval`
    /// against the stale `lastArenaTime`. Without clearing it here,
    /// that stale trigger would fire a back-to-back arena the instant
    /// the coordinator loops back.
    func recordArenaCompleted() {
        let now = Date()
        lock.withLock { state in
            state.lastArenaTime = now
            state.pending = false
        }
    }

    /// True if a trigger is currently pending (used for UI
    /// disable-while-queued semantics).
    var isPending: Bool {
        lock.withLock { $0.pending }
    }

    /// Wall-clock time of the most recent arena boundary. Read-only
    /// for UI consumers (the status-bar countdown does
    /// `interval - (now - lastArenaTime)`). The internal mutator is
    /// `recordArenaCompleted()`; this accessor only reads.
    var lastArenaTime: Date {
        lock.withLock { $0.lastArenaTime }
    }
}
