import Foundation
import os

/// Lock-protected holder for the sweep's shared state across the worker
/// thread and the SwiftUI main thread. The worker writes (progress, row
/// completions, errors); the main-thread heartbeat polls and lifts those
/// values into `@State`. The Stop button writes `cancel()` directly here
/// rather than going through Task cancellation — Swift's unstructured
/// `Task { }` doesn't inherit cancellation, so leaning on Task.isCancelled
/// inside a worker spawned from inside another Task is unreliable.
/// A locked Bool that the worker polls between steps is simpler and works.
///
/// State protected by `OSAllocatedUnfairLock<State>`; each method
/// holds the lock briefly and never across an `await`.
final class CancelBox: @unchecked Sendable {
    private struct State {
        var cancelled: Bool = false
        var progress: SweepProgress?
        var completedRows: [SweepRow] = []
        var rowPeakBytes: UInt64 = 0
    }
    private let lock = OSAllocatedUnfairLock<State>(initialState: State())

    var isCancelled: Bool {
        lock.withLock { $0.cancelled }
    }

    func cancel() {
        lock.withLock { $0.cancelled = true }
    }

    func updateProgress(_ p: SweepProgress) {
        lock.withLock { $0.progress = p }
    }

    var latestProgress: SweepProgress? {
        lock.withLock { $0.progress }
    }

    /// Off-main async getter for `latestProgress`. Lock acquisition
    /// runs on a global executor so the awaiter is never synchronously
    /// blocked.
    func asyncLatestProgress() async -> SweepProgress? {
        await withCheckedContinuation { (cont: CheckedContinuation<SweepProgress?, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(returning: self.latestProgress)
            }
        }
    }

    func appendRow(_ r: SweepRow) {
        lock.withLock { $0.completedRows.append(r) }
    }

    var completedRows: [SweepRow] {
        lock.withLock { $0.completedRows }
    }

    /// Off-main async getter for `completedRows`. Lock acquisition
    /// runs on a global executor so the awaiter is never synchronously
    /// blocked.
    func asyncCompletedRows() async -> [SweepRow] {
        await withCheckedContinuation { (cont: CheckedContinuation<[SweepRow], Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(returning: self.completedRows)
            }
        }
    }

    /// Update the per-row peak with a new sample. The sweep's worker
    /// thread reads and resets this between rows via `takeRowPeak()`.
    /// Called from both the UI heartbeat (every ~100ms) and from the
    /// trainer at row boundaries — whichever produces the higher value
    /// wins for that row.
    func recordPeakSample(_ bytes: UInt64) {
        lock.withLock { state in
            if bytes > state.rowPeakBytes { state.rowPeakBytes = bytes }
        }
    }

    /// Off-main async variant of `recordPeakSample(_:)`. Lock acquisition
    /// runs on a global executor so the awaiter is never synchronously
    /// blocked. Fire-and-forget semantics from the caller's perspective —
    /// awaiting only guarantees the update has been applied.
    func asyncRecordPeakSample(_ bytes: UInt64) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                self.recordPeakSample(bytes)
                cont.resume()
            }
        }
    }

    /// Read the peak observed during the just-finished row and reset the
    /// accumulator for the next one.
    func takeRowPeak() -> UInt64 {
        lock.withLock { state in
            let peak = state.rowPeakBytes
            state.rowPeakBytes = 0
            return peak
        }
    }
}
