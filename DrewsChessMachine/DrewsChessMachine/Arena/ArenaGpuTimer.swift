import Foundation
import os

/// Joint GPU-busy-wall accumulator shared across the candidate and
/// champion `BatchedMoveEvaluationSource` instances during an arena.
///
/// Each batcher reports `fireStarted` immediately before its
/// `network.evaluate(batchBoards:count:)` await and `fireEnded`
/// immediately after. The timer maintains a single integer
/// `inFlight` counter and accumulates wall-clock time between the
/// 0→1 and 1→0 transitions — i.e. wall time during which **at
/// least one** side was on the GPU. The arena callsite reads
/// `totalBusyMs()` at end-of-tournament to emit a single
/// `[ARENA] timing joint` line that's directly comparable to
/// arena wall — `wall - gpuBusy` is exactly the time the GPU was
/// idle (CPU per-ply work, scheduler gaps, continuation-resume
/// backpressure), with no per-side double-counting.
///
/// **Synchronisation:** state is protected by an
/// `OSAllocatedUnfairLock<State>`. The two arena batcher actors
/// call `fireStarted` / `fireEnded` from within their async
/// `fireBatch` methods — but the lock is held only across two
/// integer ops + one `DispatchTime` read, never across an
/// `await`, and is uncontended in practice (fires are sparse
/// compared to lock granularity). An actor would be a closer
/// architectural fit but would introduce two extra suspension
/// points per fire on a hot path; this is the project's documented
/// carve-out for exactly this case.
final class ArenaGpuTimer: @unchecked Sendable {
    private struct State {
        var inFlight: Int = 0
        var busyStartNanos: UInt64 = 0
        var totalBusyNanos: UInt64 = 0
    }
    private let lock = OSAllocatedUnfairLock<State>(initialState: State())

    /// Mark the start of a GPU fire. The timer's `inFlight` count
    /// goes up by one; the busy window opens on the 0→1 transition
    /// and stays open as long as any side is mid-evaluate.
    func fireStarted() {
        lock.withLock { state in
            if state.inFlight == 0 {
                state.busyStartNanos = DispatchTime.now().uptimeNanoseconds
            }
            state.inFlight += 1
        }
    }

    /// Mark the end of a GPU fire. Must pair with a prior
    /// `fireStarted` (failures inside the batcher's `fireBatch` route
    /// through a `gpuTimerEnded` flag that ensures one-end-per-start
    /// even on the throw path). On the 1→0 transition we close the
    /// busy window and add it to the total.
    func fireEnded() {
        lock.withLock { state in
            // Defensive clamp. If for any reason `fireEnded` is called
            // without a prior `fireStarted` (e.g. a future refactor that
            // misses an end-on-throw site) we don't want `inFlight` to
            // wrap below zero — that would silently disable busy-window
            // accumulation for the rest of the arena. Clamp to 0; the
            // log surface for misuse is the unit tests / build.
            if state.inFlight > 0 {
                state.inFlight -= 1
            }
            if state.inFlight == 0 {
                let now = DispatchTime.now().uptimeNanoseconds
                if now > state.busyStartNanos {
                    state.totalBusyNanos &+= (now - state.busyStartNanos)
                }
            }
        }
    }

    /// Cumulative GPU-busy-wall in milliseconds since this timer was
    /// constructed. Safe to call at any time; the arena callsite reads
    /// it once at end-of-tournament to emit the joint timing line.
    func totalBusyMs() -> Double {
        lock.withLock { Double($0.totalBusyNanos) / 1_000_000.0 }
    }
}
