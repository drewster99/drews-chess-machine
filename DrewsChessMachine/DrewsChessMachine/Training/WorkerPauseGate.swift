import Foundation
import os

/// Request/ack gate used to briefly pause one of the parallel Play and
/// Train workers (self-play or training) while another task needs
/// exclusive access to one of the four shared networks for a few
/// milliseconds — e.g. arena start copying trainer → candidate and
/// champion → arena champion, or arena end copying candidate →
/// champion on promotion. A coordinator calls `pauseAndWait()` to
/// request + wait for the worker to enter its wait state, does its
/// protected work, then calls `resume()`. The worker polls
/// `isRequestedToPause` at natural iteration boundaries (between
/// games for self-play, between SGD steps for training) and spins
/// on a 5 ms sleep until released.
///
/// Cancellation-safe: the worker's spin-wait checks `Task.isCancelled`
/// on every iteration so clicking Stop during a pause exits the
/// wait loop immediately. State protected by `OSAllocatedUnfairLock`,
/// `@unchecked Sendable`. Each public method holds the lock for
/// nanoseconds and never across an `await` (the polling loops in
/// `pauseAndWait` sleep OUTSIDE the lock).
final class WorkerPauseGate: @unchecked Sendable {
    private struct State {
        var requested: Bool = false
        var isWaiting: Bool = false
    }
    private let lock = OSAllocatedUnfairLock<State>(initialState: State())

    /// Polled by the worker at each iteration boundary.
    var isRequestedToPause: Bool {
        lock.withLock { $0.requested }
    }

    /// Called by the worker when it enters its spin-wait state, so
    /// the coordinator knows it's safe to start the protected work.
    func markWaiting() {
        lock.withLock { $0.isWaiting = true }
    }

    /// Called by the worker when it leaves its spin-wait state and
    /// resumes normal iteration.
    func markRunning() {
        lock.withLock { $0.isWaiting = false }
    }

    /// Coordinator: flip the pause request and spin-wait until the
    /// worker has acknowledged by entering its wait state. Returns
    /// once it's safe to perform the protected work.
    func pauseAndWait() async {
        setRequested(true)
        while !Task.isCancelled {
            if readIsWaiting() { return }
            // `try?` is intentional: `Task.sleep` only throws
            // `CancellationError`, and the enclosing
            // `while !Task.isCancelled` loop re-checks cancellation on the
            // very next iteration and exits cleanly. Nothing else can be
            // silently swallowed.
            try? await Task.sleep(for: .milliseconds(5))
        }
    }

    /// Bounded variant of `pauseAndWait` — returns `true` if the
    /// worker entered its wait state within `timeoutMs`, `false`
    /// on timeout (or task cancellation). Used by code paths that
    /// must not deadlock if the worker has exited its loop without
    /// acknowledging the pause (e.g. a Play-and-Train session
    /// ending mid-save via `realTrainingTask.cancel()` — that
    /// cancellation does not propagate to unstructured save
    /// Tasks, so they need their own escape hatch). On timeout
    /// the request flag is cleared so a later-returning worker
    /// doesn't get stuck in a stale pause request.
    func pauseAndWait(timeoutMs: Int) async -> Bool {
        setRequested(true)
        let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000.0)
        while !Task.isCancelled {
            if readIsWaiting() { return true }
            if Date() >= deadline {
                setRequested(false)
                return false
            }
            // `try?` is intentional: see `pauseAndWait()` above. The
            // enclosing `while !Task.isCancelled` loop is the cancellation-
            // handling path; `Task.sleep` can only throw `CancellationError`.
            try? await Task.sleep(for: .milliseconds(5))
        }
        setRequested(false)
        return false
    }

    /// Synchronous helpers so `pauseAndWait()` doesn't hold a lock
    /// across an `await` — each `withLock` is a bounded,
    /// contention-free critical section that returns immediately.
    private func setRequested(_ value: Bool) {
        lock.withLock { $0.requested = value }
    }

    private func readIsWaiting() -> Bool {
        lock.withLock { $0.isWaiting }
    }

    /// Coordinator: release the worker. Clears the request flag so
    /// the worker's next spin-wait iteration sees it and resumes.
    func resume() {
        lock.withLock { $0.requested = false }
    }
}

/// One coordinator's pause of a `WorkerPauseGate`, released at most once.
///
/// The gate is not counted: one `resume()` releases the worker whoever
/// asked for the pause. A coordinator whose release can be reached from
/// more than one place — a session save that lets self-play go as soon as
/// the replay buffer is written, and again on every way the save can end —
/// must not resume twice, because the second call could release a pause
/// another coordinator requested in between. `release()` resumes the gate
/// on its first call only. Lock: the `held` flag, held for one read-and-set.
final class WorkerPauseGateHold: @unchecked Sendable {
    private let gate: WorkerPauseGate
    private let held = OSAllocatedUnfairLock(initialState: true)

    /// A hold on `gate`, which the caller has already paused.
    init(_ gate: WorkerPauseGate) {
        self.gate = gate
    }

    /// Resume the gate, the first time only.
    func release() {
        let wasHeld = held.withLock { isHeld -> Bool in
            let wasHeld = isHeld
            isHeld = false
            return wasHeld
        }
        if wasHeld {
            gate.resume()
        }
    }
}
