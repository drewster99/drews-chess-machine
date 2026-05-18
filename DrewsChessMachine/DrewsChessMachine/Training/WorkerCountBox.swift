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
/// work. The setter's clamp-at-1 lower bound is a UX guard (a stuck
/// Stepper or sloppy caller can never zero out self-play once
/// running); the upper bound is enforced by the Stepper and the
/// spawn loop's `absoluteMaxSelfPlayWorkers` constant, not here.
///
/// **Init accepts 0** so tests and cold-start states can begin with
/// no active workers and grow into them; the setter still clamps to 1
/// once the user-driven control flow takes over.
final class WorkerCountBox: @unchecked Sendable {
    private let _count: SyncBox<Int>

    init(initial: Int) {
        precondition(initial >= 0, "WorkerCountBox initial count must be >= 0")
        _count = SyncBox<Int>(initial)
    }

    var count: Int {
        _count.value
    }

    /// Set the active worker count. Clamped at the bottom to 1 so a
    /// stuck Stepper or a sloppy caller can never zero out self-play
    /// (the upper bound is enforced by the Stepper and the spawn
    /// loop's `absoluteMaxSelfPlayWorkers` constant, not here).
    func set(_ value: Int) {
        _count.value = max(1, value)
    }
}
