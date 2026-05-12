import Foundation
import os

/// Lock-protected user-override inbox for an in-flight arena
/// tournament. There is exactly one user action that can end a
/// tournament early: `abort()` ends it with no promotion regardless
/// of the score. (Promotion is decided solely by the score-threshold
/// check at the end of `runArenaParallel` — there is no "force
/// promote this candidate" button; if you want the current *trainer*
/// promoted right now without an arena, use `Engine ▸ Promote Trainee
/// Now`, which is `SessionController.promoteTrainerNow()`.) The flag
/// is set-once and idempotent — repeated `abort()` clicks are a
/// no-op. `runArenaParallel` clears the box at the start of every
/// tournament and consumes the decision once the driver returns.
///
/// Backed by `OSAllocatedUnfairLock<Bool>`; reads and writes are
/// sub-microsecond. Each public method holds the lock for the
/// duration of a single read or single read-modify-write — never
/// across an `await`, never recursively.
final class ArenaOverrideBox: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock<Bool>(initialState: false)

    /// Request abort: end the current tournament early with no
    /// promotion. Idempotent — a no-op if abort is already set for
    /// this tournament.
    func abort() {
        lock.withLock { $0 = true }
    }

    /// True once `abort()` has been called, until `consume()` resets
    /// the box. Polled by the tournament driver's `isCancelled`
    /// closure so the game loop breaks out between games the moment
    /// the user clicks Abort.
    var isActive: Bool {
        lock.withLock { $0 }
    }

    /// Read-and-clear the abort flag. Returns `true` if the user
    /// requested abort during this tournament, `false` on normal
    /// completion. Called once by `runArenaParallel` after the driver
    /// returns, both to branch on the decision and to reset the box
    /// for the next tournament.
    func consume() -> Bool {
        lock.withLock { aborted in
            let v = aborted
            aborted = false
            return v
        }
    }
}
