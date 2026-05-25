import Foundation

/// Lock-protected current-N holder shared between the SwiftUI
/// Stepper (which mutates the value on the main actor) and the
/// concurrent self-play worker tasks (which poll it between games).
/// Workers above the current count idle in their pause loop until
/// either the count grows enough to include them or the session
/// stops. Decoupling the box from `trainingParams.selfPlayConcurrency` is
/// what lets the value cross the actor boundary without forcing
/// every worker to hop back to the main actor on each game.
///
/// Backed by a `SyncBox<Int>` (an `OSAllocatedUnfairLock`); reads
/// and writes are sub-microsecond and never queue behind any other
/// work. Lower bound is `>= 0` so tests, cold-start states, and a
/// transient drain-to-zero (e.g. before tearing down a session) are
/// all expressible. The UI Stepper's `range` constant is what
/// enforces the user-facing floor; the upper bound is similarly
/// enforced by the Stepper and the spawn loop's
/// `UpperContentView.absoluteMaxSelfPlayWorkers` constant, not here.
final class WorkerCountBox: @unchecked Sendable {
    private let _count: SyncBox<Int>

    init(initial: Int) {
        precondition(initial >= 0, "WorkerCountBox initial count must be >= 0")
        _count = SyncBox<Int>(initial)
    }

    var count: Int {
        _count.value
    }

    /// Set the active worker count. Lower-bound symmetric with init
    /// (`>= 0`); UI/caller enforces any non-zero floor.
    func set(_ value: Int) {
        precondition(value >= 0, "WorkerCountBox.set: value must be >= 0")
        _count.value = value
    }
}
