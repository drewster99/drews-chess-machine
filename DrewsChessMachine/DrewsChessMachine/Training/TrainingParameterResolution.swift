import Foundation
import TrainingParametersMacroSupport

/// The one rule for what a resumed session runs with, per training
/// parameter.
///
/// A checkpoint either carries a value for a key or it does not. When it
/// does, that value is the run's own and is applied. When it does not, the
/// key's declared `absentValue` decides — never an ad-hoc branch at the call
/// site. The resume paths used to hand-write one branch per parameter, and
/// most fell through to the live setting when the checkpoint was silent; that
/// is how a session saved before channel dropout existed was resumed with the
/// live dropout rate 0.7 and trained with dropout it had never used. Keeping
/// the decision here, driven by declarations, makes every absent-key choice
/// explicit, reviewable in one table (the `@TrainingParameter` declarations),
/// and identical across resume paths.
///
/// Pure and nonisolated: it decides; `SessionParameterResume` applies the
/// decision to the live settings and logs it.
enum TrainingParameterResolution {

    /// Why the applied value is what it is.
    enum Source: Equatable, Sendable {
        /// The checkpoint carried the value.
        case session
        /// Absent; the run predates the parameter, so its declared
        /// pre-feature value (or the range maximum standing in for
        /// "unbounded") applies.
        case preFeature
        /// Absent; an operational knob, so the live setting applies.
        case currentSetting
        /// Absent; the parameter changes training math and no value the run
        /// used is known. The live setting applies and the resume is not
        /// exact for this parameter.
        case notExact
    }

    struct Resolved<Value: Sendable & Equatable>: Equatable, Sendable {
        let applied: Value
        let source: Source
        let saved: Value?
        let current: Value
    }

    static func resolve<K: TrainingParameterKey>(
        _ key: K.Type,
        saved: K.Value?,
        current: K.Value
    ) -> Resolved<K.Value> {
        if let saved {
            return Resolved(applied: saved, source: .session, saved: saved, current: current)
        }
        switch K.absentValue {
        case .preFeature(let value):
            return Resolved(applied: value, source: .preFeature, saved: nil, current: current)
        case .declaredRangeMaximum:
            return Resolved(applied: declaredMaximum(of: K.self), source: .preFeature, saved: nil, current: current)
        case .currentSetting:
            return Resolved(applied: current, source: .currentSetting, saved: nil, current: current)
        case .refuseExact:
            return Resolved(applied: current, source: .notExact, saved: nil, current: current)
        }
    }

    /// The value a session resume applies for `key` when the checkpoint is
    /// silent and the decision does not depend on the live setting. Traps for
    /// a key whose absence means "the live setting" — asking for a fixed value
    /// there is a programmer error.
    static func absentValue<K: TrainingParameterKey>(of key: K.Type) -> K.Value {
        switch K.absentValue {
        case .preFeature(let value):
            return value
        case .declaredRangeMaximum:
            return declaredMaximum(of: K.self)
        case .currentSetting, .refuseExact:
            preconditionFailure("\(K.id) has no fixed absent value; its absence resolves to the live setting")
        }
    }

    /// One `[RESUME-DIFF]` line: saved vs applied vs current, and why.
    /// `describe` renders a value (an enum token for an enum-backed key).
    static func diffLine<K: TrainingParameterKey>(
        _ key: K.Type,
        _ resolved: Resolved<K.Value>,
        describe: (K.Value) -> String
    ) -> String {
        let savedText = resolved.saved.map(describe) ?? "absent"
        let reason: String
        switch resolved.source {
        case .session:
            reason = "from session"
        case .preFeature:
            reason = "the session predates the parameter; its pre-feature value applies, held for this run and not saved to app settings"
        case .currentSetting:
            reason = "operational setting not in the session; the current setting applies"
        case .notExact:
            reason = "NOT EXACT: the session does not record this training parameter; the current setting applies"
        }
        return "[RESUME-DIFF] \(K.id): saved=\(savedText) applied=\(describe(resolved.applied)) "
            + "current=\(describe(resolved.current)) (\(reason))"
    }

    private static func declaredMaximum<K: TrainingParameterKey>(of key: K.Type) -> K.Value {
        let raw: ParameterValue
        if let range = K.definition.intRange {
            raw = .int(range.max)
        } else if let range = K.definition.doubleRange {
            raw = .double(range.max)
        } else {
            preconditionFailure("\(K.id) declares absentValue .declaredRangeMaximum but has no range")
        }
        do {
            return try K.decode(raw)
        } catch {
            preconditionFailure("\(K.id): its declared range maximum does not decode as its own type: \(error)")
        }
    }
}
