import Foundation
import os

/// Trigger inbox for the arena coordinator task. The training worker
/// fires the trigger when the 30-minute auto cadence elapses; the UI
/// fires it via the Run Arena button. The arena coordinator awaits
/// `waitForTrigger()` and runs an arena whenever `consume()` returns
/// true. `recordArenaCompleted()` resets the "last arena" timestamp
/// so the auto-fire math stays accurate across both paths.
///
/// **Wake-up model:** event-driven via `AsyncStream<Void>`. The earlier
/// design polled `consume()` every 500 ms inside a `Task.sleep` loop;
/// that woke the cooperative pool ~28k times per 4-hour session for
/// work that is "false" 99.99% of the time. The stream version sleeps
/// indefinitely until a `trigger()` fires and is therefore both more
/// responsive (no up-to-500 ms polling latency) and pool-friendly.
///
/// State protected by `OSAllocatedUnfairLock<State>`; each method
/// holds the lock briefly and never across an `await`. The
/// `AsyncStream.Continuation` is its own independently-thread-safe
/// channel; we yield on it after releasing the lock.
final class ArenaTriggerBox: @unchecked Sendable {
    private struct State {
        var pending: Bool = false
        var lastArenaTime: Date
    }
    private let lock: OSAllocatedUnfairLock<State>

    /// Single-consumer wake-up channel. `trigger()` yields once;
    /// `waitForTrigger()` consumes one element. Buffer policy is
    /// `bufferingNewest(1)` so multiple yields between consumer wakes
    /// coalesce — the `pending` bool is the source of truth for
    /// "should the arena fire", not the count of yields. Multiple
    /// `trigger()` calls landing during one in-flight arena collapse
    /// to a single subsequent wake-up, matching the prior semantics
    /// where `recordArenaCompleted()` clears `pending` to suppress
    /// stale auto-triggers.
    private let stream: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation

    init(startTime: Date = Date()) {
        self.lock = OSAllocatedUnfairLock(initialState: State(lastArenaTime: startTime))
        var localCont: AsyncStream<Void>.Continuation!
        self.stream = AsyncStream(bufferingPolicy: .bufferingNewest(1)) { cont in
            localCont = cont
        }
        self.continuation = localCont
    }

    deinit {
        // Finish the stream so any consumer still suspended in
        // `waitForTrigger()` exits cleanly when the box is torn down
        // (e.g. session stop tearing down the TaskGroup).
        continuation.finish()
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

    /// Set the pending flag and wake the arena coordinator. The lock
    /// is released before yielding so a re-entrant trigger (e.g. UI
    /// button while the auto-fire is also firing) doesn't dead-lock
    /// against the continuation's internal serialization.
    func trigger() {
        lock.withLock { $0.pending = true }
        continuation.yield()
    }

    /// One of three outcomes from `waitForTrigger()`. The arena
    /// coordinator should `switch` on this — it's the single source
    /// of truth for what to do next, replacing the prior pattern of
    /// `Bool` return + a separate `Task.isCancelled` check after.
    enum WaitResult {
        /// A trigger was pending and has been atomically consumed.
        /// Run an arena, then call `recordArenaCompleted()`.
        case fire
        /// A stream yield arrived but `pending` was already cleared
        /// (auto-fire landed mid-arena and `recordArenaCompleted()`
        /// cleared `pending` before the coordinator returned to
        /// consume it). Loop back and wait again.
        case falseAlarm
        /// The stream has been finished — either by `cancel()` or by
        /// the box being deallocated, both of which the coordinator
        /// should treat as "exit your loop." Task cancellation also
        /// surfaces here: `AsyncStream.Iterator.next()` is
        /// cancellation-aware via its internal task-cancellation
        /// handler, which finishes the stream on cancel.
        case cancelled
    }

    /// Awaits until a `trigger()` fires, the box is `cancel()`ed, or
    /// the consuming Task is cancelled. **Single-consumer**: only
    /// the arena coordinator task should call this.
    ///
    /// On a yield, atomically reads-and-clears `pending` inside the
    /// lock so the return value is consistent with the bool's state
    /// at exactly the moment of consumption.
    func waitForTrigger() async -> WaitResult {
        for await _ in stream {
            return lock.withLock { state in
                if state.pending {
                    state.pending = false
                    return .fire
                }
                return .falseAlarm
            }
        }
        return .cancelled
    }

    /// Wakes any `waitForTrigger()` currently suspended and causes
    /// it to return `.cancelled`, regardless of Task-cancellation
    /// state. Use from a session-stop path that wants to tear down
    /// the coordinator explicitly. Idempotent — `AsyncStream`'s
    /// `finish()` is safe to call multiple times.
    func cancel() {
        lock.withLock { $0.pending = false }
        continuation.finish()
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

    /// Update the last arena timestamp to a specific time (usually
    /// 'now') WITHOUT clearing the pending flag. Used by the
    /// training worker to keep the auto-arena anchor fresh during
    /// the prefill/warmup phase, so the 30-minute clock only
    /// begins once the model is stable.
    func resetLastArenaTime(to date: Date) {
        lock.withLock { $0.lastArenaTime = date }
    }
}
