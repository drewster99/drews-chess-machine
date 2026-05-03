import Foundation

/// Lock-protected user-override inbox for an in-flight arena
/// tournament. Exactly two user actions can end a tournament early:
/// `abort()` ends it with no promotion regardless of the score, and
/// `promote()` ends it early and forces promotion regardless of the
/// score. The decision is set-once — whichever of the two buttons
/// lands first wins, and the second is a no-op — so rapid conflicting
/// clicks can't produce contradictory state. `runArenaParallel`
/// clears the box at the start of every tournament and consumes
/// the decision once the driver returns.
final class ArenaOverrideBox: @unchecked Sendable {
    enum Decision: Sendable {
        case abort
        case promote
    }

    private let queue = DispatchQueue(label: "drewschess.arenaoverridebox.serial")
    private var _decision: Decision?

    /// Request abort: end the current tournament early with no
    /// promotion. No-op if a decision (abort or promote) is already
    /// set for this tournament.
    func abort() {
        queue.async { [weak self] in
            guard let self else { return }
            if self._decision == nil {
                self._decision = .abort
            }
        }
    }

    /// Request forced promotion: end the current tournament early
    /// and promote the candidate unconditionally. No-op if a decision
    /// (abort or promote) is already set for this tournament.
    func promote() {
        queue.async { [weak self] in
            guard let self else { return }
            if self._decision == nil {
                self._decision = .promote
            }
        }
    }

    /// True once either `abort()` or `promote()` has been called,
    /// until `consume()` resets the box. Polled by the tournament
    /// driver's `isCancelled` closure so the game loop breaks out
    /// between games the moment the user clicks one of the buttons.
    var isActive: Bool {
        queue.sync { _decision != nil }
    }

    /// Read-and-clear the decision. Returns `nil` if no override
    /// was set (normal tournament completion), or the decision the
    /// user made. Called once by `runArenaParallel` after the
    /// driver returns, both to branch on the decision and to reset
    /// the box for the next tournament.
    func consume() -> Decision? {
        queue.sync {
            let d = _decision
            _decision = nil
            return d
        }
    }
}
