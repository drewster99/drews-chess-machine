import Foundation

/// Applies `TrainingParameterResolution` to the live settings while a session
/// resumes, and logs every decision.
///
/// For each key it writes the resolved value onto `TrainingParameters` with
/// the persistence that value deserves:
/// - a value the checkpoint carried is restored through
///   `restoreFromSession` (saved to app settings, as resume always has;
///   held for the run only if it lies outside today's declared range);
/// - a declared pre-feature value is held for this run only — it describes
///   the old run, not the user's preference;
/// - an operational knob or a not-exact training parameter keeps the live
///   setting untouched.
///
/// Each call emits exactly one `[RESUME-DIFF]` line (saved, applied, current,
/// reason), so a resume can never change a parameter silently. Training
/// parameters with no recorded value are collected in `notExactParameterIDs`
/// for the resume's NOT EXACT summary. The trainer is then configured from
/// the resulting settings through the same `TrainerHyperparameters` path a
/// fresh start uses.
@MainActor
final class SessionParameterResume {
    let parameters: TrainingParameters
    private let log: (String) -> Void
    private(set) var notExactParameterIDs: [String] = []

    init(parameters: TrainingParameters, log: @escaping (String) -> Void) {
        self.parameters = parameters
        self.log = log
    }

    /// Resolve and apply one key held on the settings as its own value type.
    @discardableResult
    func restore<K: TrainingParameterKey>(
        _ key: K.Type,
        saved: K.Value?,
        into keyPath: ReferenceWritableKeyPath<TrainingParameters, K.Value>
    ) -> K.Value {
        restore(
            K.self,
            saved: saved,
            current: parameters[keyPath: keyPath],
            describe: { "\($0)" },
            write: { value, source in
                switch source {
                case .session:
                    parameters.restoreFromSession(K.self, value, into: keyPath)
                case .preFeature:
                    parameters.holdForThisRun(K.self, value, into: keyPath)
                case .currentSetting, .notExact:
                    break
                }
            }
        )
    }

    /// Resolve and apply one `Double` key whose session copy is a `Float`
    /// (the trainer's hyperparameters). The saved float is widened through
    /// its shortest decimal text, so a saved 0.1 is restored as 0.1.
    @discardableResult
    func restore<K: TrainingParameterKey>(
        _ key: K.Type,
        savedFloat: Float?,
        into keyPath: ReferenceWritableKeyPath<TrainingParameters, Double>
    ) -> Double where K.Value == Double {
        restore(K.self, saved: savedFloat.map(TrainingParameters.doubleFromSavedFloat), into: keyPath)
    }

    /// Resolve one key and hand the decision to `write`, for a key the
    /// settings expose as a richer type than its stored value (an enum over
    /// an `Int` raw value). `describe` renders a value for the log line.
    /// `write` is called for every source; it decides how to apply each
    /// (`holdForThisRun` for `.preFeature`, nothing for the live-setting
    /// sources).
    @discardableResult
    func restore<K: TrainingParameterKey>(
        _ key: K.Type,
        saved: K.Value?,
        current: K.Value,
        describe: (K.Value) -> String,
        write: (K.Value, TrainingParameterResolution.Source) -> Void
    ) -> K.Value {
        let resolved = TrainingParameterResolution.resolve(K.self, saved: saved, current: current)
        log(TrainingParameterResolution.diffLine(K.self, resolved, describe: describe))
        if resolved.source == .notExact {
            notExactParameterIDs.append(K.id)
        }
        write(resolved.applied, resolved.source)
        return resolved.applied
    }

    /// A saved value that was unusable and that the user, reviewing the
    /// session at load, chose to replace with the current setting. Nothing is
    /// written; the line records the replacement.
    func logReplacedAtLoad<K: TrainingParameterKey>(
        _ key: K.Type,
        savedDescription: String,
        currentDescription: String,
        why: String
    ) {
        log("[RESUME-DIFF] \(K.id): saved=\(savedDescription) applied=\(currentDescription) current=\(currentDescription) "
            + "(saved value \(why); replaced with the current setting, accepted by the user at load)")
    }

    /// The resume's NOT EXACT summary for parameters the session did not
    /// record, or nothing when every training parameter was resolved.
    func notExactSummary() -> String? {
        guard !notExactParameterIDs.isEmpty else { return nil }
        return "[RESUME] NOT EXACT: parameters absent from the session (current settings used): "
            + notExactParameterIDs.joined(separator: ", ")
    }
}
