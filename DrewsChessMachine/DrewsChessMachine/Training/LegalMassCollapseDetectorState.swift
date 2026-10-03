import Foundation

/// The legal-mass-collapse detector's running state as a session saves it
/// (determinism plan C1 #20): the recent probes' legal mass and how much of
/// the grace period training had already used. A resume continues both, so
/// the detector neither restarts its grace period nor forgets the probes
/// that were building toward (or away from) a collapse.
struct LegalMassCollapseDetectorState: Codable, Equatable, Sendable {
    /// The most recent probes' legal mass, oldest first.
    var legalMassWindow: [Double]
    /// Training seconds the grace period had counted at save time —
    /// measured from the first probe that saw an SGD step — or nil when no
    /// probe had seen one yet.
    var graceElapsedSec: Double?

    enum CodingKeys: String, CodingKey {
        case legalMassWindow = "legal_mass_window"
        case graceElapsedSec = "grace_elapsed_sec"
    }
}

/// The live legal-mass-collapse detector state, shared between the detector
/// task (which updates it once per probe) and session saves (which read it).
/// Lock discipline: every read and write goes through `state`, a `SyncBox`
/// (`OSAllocatedUnfairLock`); nothing here blocks or awaits while holding it.
///
/// The grace anchor is a point in time, but what a save records is how much
/// grace training had used, and a resume carries that amount forward: the
/// anchor is re-set only when the resumed run's first SGD step is observed,
/// to that moment minus the carried amount. Refilling the replay buffer
/// after a resume therefore spends none of the grace period, exactly as the
/// first fill of a fresh session spends none.
final class LegalMassCollapseDetectorBox: @unchecked Sendable {
    private struct Live {
        var legalMassWindow: [Double]
        /// When the grace countdown started, or nil while no SGD step has
        /// been observed in this process.
        var trainingStartAt: Date?
        /// Grace already used before this process (from a resumed session).
        var carriedGraceSec: Double
    }

    private let state = SyncBox(Live(legalMassWindow: [], trainingStartAt: nil, carriedGraceSec: 0))

    /// The state a session save records at `now`.
    func snapshot(now: Date) -> LegalMassCollapseDetectorState {
        state.mutate { live in
            let graceElapsedSec: Double?
            if let start = live.trainingStartAt {
                graceElapsedSec = now.timeIntervalSince(start)
            } else {
                graceElapsedSec = live.carriedGraceSec > 0 ? live.carriedGraceSec : nil
            }
            return LegalMassCollapseDetectorState(legalMassWindow: live.legalMassWindow,
                                                  graceElapsedSec: graceElapsedSec)
        }
    }

    /// Continue a saved state. The grace already used is carried until the
    /// resumed run's first observed SGD step re-anchors it.
    func restore(_ saved: LegalMassCollapseDetectorState) {
        state.mutate { live in
            live.legalMassWindow = saved.legalMassWindow
            live.trainingStartAt = nil
            live.carriedGraceSec = saved.graceElapsedSec ?? 0
        }
    }

    /// A probe saw no SGD step yet: the grace countdown has not started (or
    /// has stopped, for a new session's trainer).
    func noteNoTrainingStepsYet() {
        state.mutate { $0.trainingStartAt = nil }
    }

    /// The grace seconds used so far at `now`, anchoring the countdown on the
    /// first call after an SGD step has been observed.
    func graceElapsed(observingTrainingAt now: Date) -> TimeInterval {
        state.mutate { live in
            let start: Date
            if let existing = live.trainingStartAt {
                start = existing
            } else {
                start = now.addingTimeInterval(-live.carriedGraceSec)
                live.trainingStartAt = start
            }
            return now.timeIntervalSince(start)
        }
    }

    /// Append one probe's legal mass and keep the newest `capacity` readings;
    /// returns the window, oldest first.
    func append(legalMass: Double, capacity: Int) -> [Double] {
        precondition(capacity >= 1, "the legal-mass window holds at least one probe (got \(capacity))")
        return state.mutate { live in
            live.legalMassWindow.append(legalMass)
            if live.legalMassWindow.count > capacity {
                live.legalMassWindow.removeFirst(live.legalMassWindow.count - capacity)
            }
            return live.legalMassWindow
        }
    }
}
