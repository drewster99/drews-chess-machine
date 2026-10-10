import Foundation
import os

/// Every GPU fault this process has learned about, from either detection
/// layer, in the order it learned of them — the one source the fault policy
/// reads (GPU fault forensics plan, A3–A5).
///
/// Two layers write here:
///
/// - `GPUSubmission.verify` — a submission whose reachable command
///   buffers or completion handler reported a failure. Known the moment the
///   submission is checked.
/// - `GPUFaultMonitor` — macOS's own fault messages for this process
///   (`kIOGPUCommandBufferCallbackError…`), which also cover command buffers
///   MPSGraph created out of the app's reach. Known up to one poll interval
///   late; `time` is when macOS logged the fault, not when the monitor read
///   it.
///
/// One GPU reset usually shows up in both layers (and as several system-log
/// entries, one per discarded buffer). The ledger keeps every report;
/// consumers ask "has anything new been recorded since sequence N", so
/// duplicates never cause a second action.
///
/// Consumers keep the `latestSequence` they last acted on and compare:
/// a training loop stops on any new fault, an arena compares fault times
/// with its own start and end.
///
/// The shared ledger also has `GPUFlightRecorder` log what was on the GPU
/// around each fault it records (`[GPU-INFLIGHT]`): every fault passes
/// through here, from either layer, with the time it happened.
final class GPUFaultLedger: @unchecked Sendable {
    static let shared = GPUFaultLedger(flightRecorder: .shared)

    enum Source: Sendable, Equatable {
        /// A checked submission failed; `detail` is its `[GPU-ERR]` account.
        case submission(stage: String, detail: String)
        /// macOS logged a command-buffer fault in this process.
        case systemLog(message: String)
    }

    struct Fault: Sendable, Equatable {
        /// 1, 2, 3, … in the order the ledger recorded them.
        let sequence: Int
        /// When the fault happened as far as the reporter knows: the check
        /// time for a submission, macOS's log time for a system-log entry.
        let time: Date
        let source: Source

        var summary: String {
            switch source {
            case .submission(let stage, let detail):
                return "submission stage=\(stage) \(detail)"
            case .systemLog(let message):
                return "system log: \(message)"
            }
        }
    }

    /// Faults kept for `faults(after:)`. A run that sees more than this
    /// has stopped long before; the count keeps the memory bounded.
    static let retainedFaultCount = 1_000

    private struct State {
        var faults: [Fault] = []
        var latestSequence = 0
    }

    private let state = OSAllocatedUnfairLock<State>(initialState: State())
    /// Accounts for each recorded fault; nil for a test ledger.
    private let flightRecorder: GPUFlightRecorder?

    /// A ledger of its own, for tests (no `[GPU-INFLIGHT]` accounts);
    /// production code uses `shared`.
    init() {
        self.flightRecorder = nil
    }

    private init(flightRecorder: GPUFlightRecorder) {
        self.flightRecorder = flightRecorder
    }

    /// Records one fault and returns it with its sequence number.
    @discardableResult
    func record(_ source: Source, at time: Date = Date()) -> Fault {
        let fault = state.withLock { state in
            state.latestSequence += 1
            let fault = Fault(sequence: state.latestSequence, time: time, source: source)
            state.faults.append(fault)
            if state.faults.count > Self.retainedFaultCount {
                state.faults.removeFirst(state.faults.count - Self.retainedFaultCount)
            }
            return fault
        }
        flightRecorder?.logAccount(for: fault)
        return fault
    }

    /// The sequence number of the newest fault recorded; 0 before any.
    var latestSequence: Int {
        state.withLock { $0.latestSequence }
    }

    /// Every retained fault recorded after `sequence`, oldest first.
    func faults(after sequence: Int) -> [Fault] {
        state.withLock { state in
            state.faults.filter { $0.sequence > sequence }
        }
    }
}
