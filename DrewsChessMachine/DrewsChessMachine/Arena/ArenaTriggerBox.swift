import Foundation

/// Lock-protected trigger inbox for the arena coordinator task. The
/// training worker fires the trigger when the 30-minute auto cadence
/// elapses; the UI fires it via the Run Arena button. The arena
/// coordinator task polls `consume()` in its main loop and runs an
/// arena whenever the trigger is pending. `recordArenaCompleted()`
/// resets the "last arena" timestamp so the auto-fire math stays
/// accurate across both the automatic and manual paths.
final class ArenaTriggerBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.arenatriggerbox.serial")
    private var _pending = false
    private var _lastArenaTime: Date

    init(startTime: Date = Date()) {
        self._lastArenaTime = startTime
    }

    /// Check whether enough wall clock has elapsed since the last
    /// arena to auto-trigger another one. Returns false if a trigger
    /// is already pending, so the training worker doesn't queue
    /// multiple auto-triggers for the same deadline.
    func shouldAutoTrigger(interval: TimeInterval) -> Bool {
        queue.sync {
            if _pending { return false }
            return Date().timeIntervalSince(_lastArenaTime) >= interval
        }
    }

    /// Set the pending flag. The arena coordinator's next poll will
    /// consume it and start an arena.
    func trigger() {
        queue.async { [weak self] in self?._pending = true }
    }

    /// Poll the trigger. Returns true and clears the pending flag if
    /// a trigger was waiting; returns false otherwise.
    func consume() -> Bool {
        queue.sync {
            if _pending {
                _pending = false
                return true
            }
            return false
        }
    }

    /// Record that an arena just finished. Resets the wall-clock
    /// reference for subsequent `shouldAutoTrigger` checks so the
    /// next auto-fire happens `interval` seconds from now, not from
    /// the previous last-arena time. Also clears the pending flag:
    /// the training worker runs in parallel with the arena and can
    /// stamp `_pending` mid-arena once elapsed time crosses `interval`
    /// against the stale `_lastArenaTime`. Without clearing it here,
    /// that stale trigger would fire a back-to-back arena the instant
    /// the coordinator loops back.
    func recordArenaCompleted() {
        let now = Date()
        queue.async { [weak self] in
            guard let self else { return }
            self._lastArenaTime = now
            self._pending = false
        }
    }

    /// True if a trigger is currently pending (used for UI
    /// disable-while-queued semantics).
    var isPending: Bool {
        queue.sync { _pending }
    }

    /// Wall-clock time of the most recent arena boundary. Read-only
    /// for UI consumers (the status-bar countdown does
    /// `interval - (now - lastArenaTime)`). The internal mutator is
    /// `recordArenaCompleted()`; this accessor only reads.
    var lastArenaTime: Date {
        queue.sync { _lastArenaTime }
    }
}
