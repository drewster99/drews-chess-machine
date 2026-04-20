import Foundation

/// Pure-logic scheduler for the 4-hour periodic session autosave.
///
/// The controller tracks a single deadline relative to the most
/// recent successful session save. It is armed while Play-and-Train
/// is active and disarmed otherwise. On a deadline crossing the
/// caller is supposed to trigger a save; if the arena is running at
/// that moment the decision is **deferred** — the controller holds
/// a "pending fire" flag until the arena ends, at which point the
/// next call to `decide(now:)` either:
///
/// - swallows the pending fire if a (post-promotion) successful
///   save happened during the arena-deferred window, since the
///   arena's own promotion path already wrote the session and a
///   second save on top would be redundant; or
/// - fires immediately if no save came in during that window, so
///   the 4-hour cadence is only a little late instead of skipping
///   an entire interval.
///
/// The controller is **not** a timer — it holds no `Task` or
/// `Timer`. The caller polls `decide(now:)` at whatever cadence
/// makes sense (the UI already runs a main-actor heartbeat;
/// piggy-backing on that avoids a second timer). All state
/// transitions are driven by explicit method calls, which makes
/// the whole thing trivially unit-testable with an injected clock.
///
/// Thread-safety: `@MainActor`-isolated. Not designed for cross-
/// thread access — the caller funnels updates through the main
/// actor (same pattern `ContentView.swift` uses for its other
/// scheduler boxes).
@MainActor
final class PeriodicSaveController {

    /// Interval between scheduled saves while Play-and-Train is
    /// armed. 4 hours per the spec; the controller does not know
    /// about this from anywhere else, so the caller passes it in
    /// once at construction for ease of testing with a shorter
    /// cadence.
    let interval: TimeInterval

    /// `true` while Play-and-Train is running. The deadline clock
    /// is only meaningful when `armed == true`. Disarming cancels
    /// any pending fire so the next re-arm starts from a clean
    /// deadline.
    private(set) var armed: Bool = false

    /// `true` while an arena tournament is in flight. Deadlines
    /// that cross while an arena runs are held as `pendingFire`
    /// and checked again on arena end.
    private(set) var arenaRunning: Bool = false

    /// Wall-clock deadline for the next scheduled fire. `nil` when
    /// the controller is disarmed. Updated on arm, on every
    /// successful save (any trigger), and after a fire is
    /// dispatched. Monotonic within a single arm — never rewound.
    private(set) var nextFireAt: Date?

    /// Set to `true` when `decide(now:)` observes the deadline has
    /// crossed but an arena is running. Consumed on the next
    /// `decide(now:)` call after the arena ends. Cleared if a
    /// successful save arrives first (the arena's promotion autosave
    /// already wrote a session, so the pending periodic save is
    /// redundant and swallowed).
    private(set) var pendingFire: Bool = false

    /// The last time any successful session save was observed.
    /// Useful for diagnostics / logging; the scheduler itself only
    /// reads `nextFireAt`.
    private(set) var lastSuccessfulSaveAt: Date?

    /// Construct a controller with a given cadence.
    /// - Parameter interval: Wall-clock seconds between scheduled
    ///   periodic saves. Must be > 0.
    init(interval: TimeInterval) {
        precondition(interval > 0, "PeriodicSaveController interval must be > 0; got \(interval)")
        self.interval = interval
    }

    // MARK: - Lifecycle

    /// Arm the controller when Play-and-Train starts. Sets the
    /// first deadline `interval` seconds from now. Calling this
    /// while already armed resets the deadline — intentional so
    /// that a user-visible "session just started" always resets
    /// the next save to a full interval away.
    func arm(now: Date) {
        armed = true
        nextFireAt = now.addingTimeInterval(interval)
        pendingFire = false
    }

    /// Disarm the controller when Play-and-Train stops. Clears
    /// the deadline and any pending fire so that a stop-then-start
    /// does not immediately fire a save.
    func disarm() {
        armed = false
        nextFireAt = nil
        pendingFire = false
        // Deliberately preserve `lastSuccessfulSaveAt` and
        // `arenaRunning` across arm/disarm cycles — they reflect
        // facts about the world outside the controller's armed
        // state, and a fresh arm() will overwrite the deadline
        // regardless.
    }

    // MARK: - External event notifications

    /// Notify that an arena tournament has started. While `true`,
    /// a deadline crossing produces a pending fire instead of a
    /// firing decision.
    func noteArenaBegan() {
        arenaRunning = true
    }

    /// Notify that an arena tournament has ended. The next
    /// `decide(now:)` call will then consult `pendingFire` and
    /// either fire (if no save arrived during the arena window)
    /// or swallow (if one did).
    func noteArenaEnded() {
        arenaRunning = false
    }

    /// Notify the controller that a session save completed
    /// successfully. Resets the deadline to `now + interval` and
    /// clears any pending fire, regardless of trigger — a manual,
    /// post-promotion, or periodic save all count as "the session
    /// is freshly on disk, next scheduled save is a full interval
    /// from now".
    ///
    /// Safe to call while disarmed (e.g. the arena's promotion
    /// autosave completes just after Stop — the call still
    /// updates `lastSuccessfulSaveAt` for diagnostics but does
    /// not re-arm the schedule).
    func noteSuccessfulSave(at now: Date) {
        lastSuccessfulSaveAt = now
        pendingFire = false
        if armed {
            nextFireAt = now.addingTimeInterval(interval)
        }
    }

    // MARK: - Scheduling decision

    /// Outcome of polling the scheduler at `now`. One of:
    /// - `.idle` — nothing to do.
    /// - `.fire` — the caller should trigger a periodic save.
    ///
    /// Calling `decide` never mutates `nextFireAt` when returning
    /// `.fire`; the caller is expected to start the save and, on
    /// success, call `noteSuccessfulSave(at:)` which slides the
    /// next deadline forward. On save failure the caller should
    /// not call `noteSuccessfulSave` — the controller's next
    /// `decide(now:)` will then re-fire (subject to the same
    /// arena-deferral rule), so a flaky save does not burn an
    /// interval.
    enum Decision {
        case idle
        case fire
    }

    /// Evaluate the scheduler state against the current time and
    /// return what the caller should do. The caller must invoke
    /// this periodically (e.g. once a second from the main-actor
    /// heartbeat) while the app is alive.
    ///
    /// The method mutates `pendingFire` to record the
    /// arena-deferral state, so repeated calls with no change in
    /// inputs converge: a deadline crossing inside an arena sets
    /// `pendingFire = true` once and subsequent calls stay idle
    /// until the arena ends.
    func decide(now: Date) -> Decision {
        guard armed else {
            // Disarmed: no save scheduled. Drop any stale pending
            // flag so a subsequent arm() starts clean.
            pendingFire = false
            return .idle
        }

        // Has the deadline crossed since we last looked?
        let deadlineCrossed: Bool
        if let fireAt = nextFireAt {
            deadlineCrossed = now >= fireAt
        } else {
            // Armed but no deadline — shouldn't happen given arm()
            // always sets one, but treat as "not yet" rather than
            // asserting.
            deadlineCrossed = false
        }

        if arenaRunning {
            // During an arena: turn a deadline crossing into a
            // pending fire. We do not fire; we will evaluate again
            // once the arena reports end.
            if deadlineCrossed {
                pendingFire = true
            }
            return .idle
        }

        // Arena not running. Two reasons to fire right now:
        //   1) a pending fire is queued from an arena-deferred
        //      deadline, AND no successful save arrived in the
        //      meantime (which would have cleared pendingFire).
        //   2) the deadline crossed "just now" (outside any arena).
        // Both reduce to "fire and let the caller reset the
        // deadline on success via noteSuccessfulSave(at:)".
        if pendingFire {
            // We intentionally do NOT clear pendingFire here —
            // `noteSuccessfulSave` will clear it on the save's
            // success path, or a subsequent failure will let the
            // next tick re-fire if still past the (stale) deadline.
            // But to avoid immediate re-firing on the very next
            // tick (while the save task is in flight), clear it
            // now — the caller is expected to follow up with
            // noteSuccessfulSave. On failure the next deadline
            // (unchanged) will still be in the past, so the next
            // decide() call will return .fire again.
            pendingFire = false
            return .fire
        }
        if deadlineCrossed {
            return .fire
        }
        return .idle
    }
}
