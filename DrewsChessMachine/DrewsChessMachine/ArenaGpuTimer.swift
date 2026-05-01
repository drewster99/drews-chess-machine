import Foundation

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
/// **Synchronisation:** uses a serial `DispatchQueue` to match the
/// project's established pattern for state shared across async call
/// sites (cf. `ReplayBuffer`, `GameDiversityTracker`). The two
/// arena batcher actors call `fireStarted` / `fireEnded` from
/// within their async `fireBatch` methods — but the `queue.sync`
/// critical section is purely synchronous (two integer ops + one
/// `DispatchTime` read), holds nothing across an await, and is
/// uncontended in practice (fires are sparse compared to the
/// queue's mutex granularity). An actor would be a closer
/// architectural fit but would introduce two extra suspension
/// points per fire on a hot path; the serial-queue pattern is
/// the project's documented carve-out for exactly this case.
final class ArenaGpuTimer: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.dcm.arena.gpu-timer")
    private var inFlight: Int = 0
    private var busyStartNanos: UInt64 = 0
    private var totalBusyNanos: UInt64 = 0

    /// Mark the start of a GPU fire. The timer's `inFlight` count
    /// goes up by one; the busy window opens on the 0→1 transition
    /// and stays open as long as any side is mid-evaluate.
    func fireStarted() {
        queue.sync {
            if inFlight == 0 {
                busyStartNanos = DispatchTime.now().uptimeNanoseconds
            }
            inFlight += 1
        }
    }

    /// Mark the end of a GPU fire. Must pair with a prior
    /// `fireStarted` (failures inside the batcher's `fireBatch` route
    /// through a `gpuTimerEnded` flag that ensures one-end-per-start
    /// even on the throw path). On the 1→0 transition we close the
    /// busy window and add it to the total.
    func fireEnded() {
        queue.sync {
            // Defensive clamp. If for any reason `fireEnded` is called
            // without a prior `fireStarted` (e.g. a future refactor that
            // misses an end-on-throw site) we don't want `inFlight` to
            // wrap below zero — that would silently disable busy-window
            // accumulation for the rest of the arena. Clamp to 0; the
            // log surface for misuse is the unit tests / build.
            if inFlight > 0 {
                inFlight -= 1
            }
            if inFlight == 0 {
                let now = DispatchTime.now().uptimeNanoseconds
                if now > busyStartNanos {
                    totalBusyNanos &+= (now - busyStartNanos)
                }
            }
        }
    }

    /// Cumulative GPU-busy-wall in milliseconds since this timer was
    /// constructed. Safe to call at any time; the arena callsite reads
    /// it once at end-of-tournament to emit the joint timing line.
    func totalBusyMs() -> Double {
        return queue.sync {
            return Double(totalBusyNanos) / 1_000_000.0
        }
    }
}
