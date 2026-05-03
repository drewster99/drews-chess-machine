import Foundation

/// Lock-protected holder for the sweep's shared state across the worker
/// thread and the SwiftUI main thread. The worker writes (progress, row
/// completions, errors); the main-thread heartbeat polls and lifts those
/// values into `@State`. The Stop button writes `cancel()` directly here
/// rather than going through Task cancellation — Swift's unstructured
/// `Task { }` doesn't inherit cancellation, so leaning on Task.isCancelled
/// inside a worker spawned from inside another Task is unreliable.
/// A locked Bool that the worker polls between steps is simpler and works.
final class CancelBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.cancelbox.serial")
    private var _cancelled = false
    private var _progress: SweepProgress?
    private var _completedRows: [SweepRow] = []
    private var _rowPeakBytes: UInt64 = 0

    var isCancelled: Bool {
        queue.sync { _cancelled }
    }

    func cancel() {
        queue.async { [weak self] in self?._cancelled = true }
    }

    func updateProgress(_ p: SweepProgress) {
        queue.async { [weak self] in self?._progress = p }
    }

    var latestProgress: SweepProgress? {
        queue.sync { _progress }
    }

    func appendRow(_ r: SweepRow) {
        queue.async { [weak self] in self?._completedRows.append(r) }
    }

    var completedRows: [SweepRow] {
        queue.sync { _completedRows }
    }

    /// Update the per-row peak with a new sample. The sweep's worker
    /// thread reads and resets this between rows via `takeRowPeak()`.
    /// Called from both the UI heartbeat (every ~100ms) and from the
    /// trainer at row boundaries — whichever produces the higher value
    /// wins for that row.
    func recordPeakSample(_ bytes: UInt64) {
        queue.async { [weak self] in
            guard let self else { return }
            if bytes > self._rowPeakBytes { self._rowPeakBytes = bytes }
        }
    }

    /// Read the peak observed during the just-finished row and reset the
    /// accumulator for the next one.
    func takeRowPeak() -> UInt64 {
        queue.sync {
            let peak = _rowPeakBytes
            _rowPeakBytes = 0
            return peak
        }
    }
}
