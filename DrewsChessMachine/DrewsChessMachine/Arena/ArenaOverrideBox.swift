import Foundation
import os

/// Lock-protected user-override inbox for an in-flight arena
/// tournament. Exactly two user actions can end a tournament early:
/// `abort()` ends it with no promotion regardless of the score, and
/// `promote()` ends it early and forces promotion regardless of the
/// score. The decision is set-once — whichever of the two buttons
/// lands first wins, and the second is a no-op — so rapid conflicting
/// clicks can't produce contradictory state. `runArenaParallel`
/// clears the box at the start of every tournament and consumes
/// the decision once the driver returns.
///
/// Backed by `OSAllocatedUnfairLock<Decision?>`; reads and writes are
/// sub-microsecond. Each public method holds the lock for the
/// duration of a single read or single read-modify-write — never
/// across an `await`, never recursively.
final class ArenaOverrideBox: @unchecked Sendable {
    enum Decision: Sendable {
        case abort
        case promote
    }

    private let lock = OSAllocatedUnfairLock<Decision?>(initialState: nil)

    /// Request abort: end the current tournament early with no
    /// promotion. No-op if a decision (abort or promote) is already
    /// set for this tournament.
    func abort() {
        lock.withLock { decision in
            if decision == nil {
                decision = .abort
            }
        }
    }

    /// Request forced promotion: end the current tournament early
    /// and promote the candidate unconditionally. No-op if a decision
    /// (abort or promote) is already set for this tournament.
    func promote() {
        lock.withLock { decision in
            if decision == nil {
                decision = .promote
            }
        }
    }

    /// True once either `abort()` or `promote()` has been called,
    /// until `consume()` resets the box. Polled by the tournament
    /// driver's `isCancelled` closure so the game loop breaks out
    /// between games the moment the user clicks one of the buttons.
    var isActive: Bool {
        lock.withLock { $0 != nil }
    }

    /// Read-and-clear the decision. Returns `nil` if no override
    /// was set (normal tournament completion), or the decision the
    /// user made. Called once by `runArenaParallel` after the
    /// driver returns, both to branch on the decision and to reset
    /// the box for the next tournament.
    func consume() -> Decision? {
        lock.withLock { decision in
            let d = decision
            decision = nil
            return d
        }
    }
}
