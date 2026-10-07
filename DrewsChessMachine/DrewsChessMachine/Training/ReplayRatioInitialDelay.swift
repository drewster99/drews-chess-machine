import Foundation

/// The training-step delay a Play-and-Train start builds its replay-ratio
/// controller with, and where it came from (plan B6, owner decision O-20).
///
/// **Why this exists.** With auto-adjust on, a start used to seed the
/// controller from the last auto-computed delay, a `UserDefaults` value read
/// with a silent `?? 50` when none was stored, and no file recorded which
/// value a run started from. The selection is now one pure function: a saved
/// auto delay when auto-adjust is on and one exists, else the declared
/// `training_step_delay_ms` (what manual mode always used), each logged and
/// recorded in the lineage configuration's `replay_ratio.starts` under the
/// same source name.
enum ReplayRatioInitialDelay {
    struct Resolved: Sendable, Equatable {
        let autoAdjust: Bool
        let delayMs: Int
        let source: LineageRecord.ReplayRatio.InitialDelaySource

        /// The `[REPLAY-RATIO]` line a start logs for this choice.
        var logLine: String {
            switch source {
            case .lastAutoComputedDelayMs:
                return "[REPLAY-RATIO] initial delay: saved auto-computed delay \(delayMs) ms"
            case .trainingStepDelayMs:
                if autoAdjust {
                    return "[REPLAY-RATIO] initial delay: no saved auto-computed delay; starting from training_step_delay_ms=\(delayMs)"
                }
                return "[REPLAY-RATIO] initial delay: training_step_delay_ms=\(delayMs) (auto-adjust off)"
            }
        }
    }

    enum StoredDelayError: Error, CustomStringConvertible, LocalizedError {
        case notAnInteger(storedType: String)

        var description: String {
            switch self {
            case .notAnInteger(let storedType):
                return "lastAutoComputedDelayMs: stored value is not an integer (it is a \(storedType))"
            }
        }

        var errorDescription: String? { description }
    }

    static func resolve(autoAdjust: Bool, savedAutoDelayMs: Int?, trainingStepDelayMs: Int) -> Resolved {
        if autoAdjust, let saved = savedAutoDelayMs {
            return Resolved(autoAdjust: true, delayMs: saved, source: .lastAutoComputedDelayMs)
        }
        return Resolved(autoAdjust: autoAdjust, delayMs: trainingStepDelayMs, source: .trainingStepDelayMs)
    }
}

/// How a Play-and-Train start got its run seed (gap 9b): a Continue after
/// Stop keeps the run's seed and adds no seed entry; every other start
/// resolved (or inherited) one, which a start that keeps the segment ("New
/// Session, keep trainer") records as a new entry.
enum RunSeedStartKind: Sendable, Equatable {
    case continued
    case resolvedOrInherited
}
