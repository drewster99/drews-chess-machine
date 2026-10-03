/// What a session resume applies for a training parameter whose saved
/// checkpoint carries no value for it — a checkpoint written before the
/// parameter (or before its field in the checkpoint) existed.
///
/// Every `@TrainingParameter` declares one through `absentValue:`, so the
/// answer lives with the parameter rather than in a hand-written branch of
/// each resume path. Choosing the live setting for a training-math parameter
/// is how an old run once resumed with dropout it never used; this type makes
/// that choice explicit and reviewable per key.
public enum TrainingParameterAbsence<Value: Sendable & Equatable>: Sendable, Equatable {
    /// The run that wrote the checkpoint factually trained with this value,
    /// because the feature the parameter controls did not exist yet (no
    /// dropout ⇒ rate 0, plain SGD ⇒ momentum 0, …). Resume reproduces it.
    case preFeature(Value)

    /// The pre-feature behavior was "unbounded", which the declared range no
    /// longer expresses; its maximum is the closest representable value.
    /// Only for numeric parameters declared with a range.
    case declaredRangeMaximum

    /// An operational knob (intervals, alarms, autosave, concurrency, arena
    /// scheduling) that does not change what the trainer computes; the
    /// resumed run uses the live setting.
    case currentSetting

    /// The parameter changes training math, and there is no value the saved
    /// run is known to have used. An exact resume cannot proceed without it;
    /// a resume that is not exact by design (the GUI) uses the live setting
    /// and reports the parameter as not exact.
    case refuseExact
}
