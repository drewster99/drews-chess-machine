import Foundation

/// Why a GUI Play-and-Train run's training is suspended, if it is: the one
/// source of truth for the suspension and for every gate it closes (the
/// alarms plan R3, owner decision OD-5). It replaces the old
/// `trainingSuspendedByDivergence: Bool`, which could not say why, so the
/// gates that differ between the two causes could not be told apart.
///
/// The run itself is never torn down by a suspension: self-play, the
/// heartbeat and the stats ticker keep running, the banner or the health
/// list says why, and Stop (then Start) clears it.
///
/// | Gate | `.divergence` | `.healthAlarm` |
/// |---|---|---|
/// | arena runs | skipped | skipped (a damaged trainer must not become a candidate) |
/// | periodic autosave | skipped (NaN weights) | runs (finite weights; useful for forensics) |
/// | heartbeat alarm evaluation | skipped (frozen metrics) | runs |
/// | legal-mass banner | not raised over it | not raised over it |
/// | Train ▸ Promote Trainee Now | refused | refused |
enum TrainingSuspension: Sendable, Equatable {
    /// A training step diverged (non-finite loss, GPU failure, gradient
    /// blow-up). The trainer worker has returned; the weights may be
    /// non-finite.
    case divergence(reason: String)
    /// A training-health alarm whose rule's action stops the run is active.
    /// The trainer worker is parked (it still services pause requests, so
    /// saves and arenas already running complete); the weights are finite.
    case healthAlarm(rule: TrainingHealthRule, detail: String)

    /// What the `[ARENA] skipped — training suspended (…)` line names.
    var arenaSkipLabel: String {
        switch self {
        case .divergence: return "divergence"
        case .healthAlarm(let rule, _): return "health alarm \(rule.rawValue)"
        }
    }

    /// Why a menu action refused, for `onRefuseMenuAction`.
    var refusalReason: String {
        switch self {
        case .divergence(let reason):
            return "Training is suspended after a divergence (\(reason)). Stop, then reload an earlier checkpoint."
        case .healthAlarm(let rule, let detail):
            return "Training is suspended by the training-health alarm \(rule.rawValue) (\(detail)). "
                + "Stop first; to keep training it, set that rule's action to Log on the Health tab."
        }
    }

    /// Whether the periodic autosave skips its tick.
    var skipsPeriodicAutosave: Bool {
        switch self {
        case .divergence: return true
        case .healthAlarm: return false
        }
    }

    /// Whether the heartbeat skips the banner detectors' evaluation.
    var skipsHeartbeatAlarmEvaluation: Bool {
        switch self {
        case .divergence: return true
        case .healthAlarm: return false
        }
    }
}
