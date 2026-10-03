// MARK: - @TrainingParameter

/// Attaches to an empty `enum` declaration and synthesizes the boilerplate that
/// makes it a `TrainingParameterKey`. Consumers conform manually:
///
/// ```swift
/// @TrainingParameter(
///     name: "Replay Buffer Capacity",
///     description: "Maximum number of moves retained.",
///     default: 1_000_000,
///     range: 1_000...10_000_000,
///     category: "Replay Buffer"
/// )
/// public enum ReplayBufferCapacity: TrainingParameterKey {}
/// ```
///
/// Expands to: `id`, `definition`, `encode(_:)`, `decode(_:)`, and — when
/// `absentValue:` is given — `absentValue`.
///
/// `absentValue:` is what a session resume applies when the saved checkpoint
/// carries no value for the key (see `TrainingParameterAbsence`). It is
/// optional here only so the macro can still expand a declaration without
/// it; the app's `TrainingParameterKey` protocol requires `absentValue`, so a
/// key declared without one does not compile.
@attached(member, names: named(id), named(definition), named(encode), named(decode), named(absentValue))
public macro TrainingParameter(
    name: String,
    description: String,
    default: Double,
    range: ClosedRange<Double>,
    category: String,
    id: String? = nil,
    liveTunable: Bool = false,
    absentValue: TrainingParameterAbsence<Double>? = nil
) = #externalMacro(module: "TrainingParametersMacroPlugin", type: "TrainingParameterMacro")

@attached(member, names: named(id), named(definition), named(encode), named(decode), named(absentValue))
public macro TrainingParameter(
    name: String,
    description: String,
    default: Int,
    range: ClosedRange<Int>,
    category: String,
    id: String? = nil,
    liveTunable: Bool = false,
    absentValue: TrainingParameterAbsence<Int>? = nil
) = #externalMacro(module: "TrainingParametersMacroPlugin", type: "TrainingParameterMacro")

@attached(member, names: named(id), named(definition), named(encode), named(decode), named(absentValue))
public macro TrainingParameter(
    name: String,
    description: String,
    default: Bool,
    category: String,
    id: String? = nil,
    liveTunable: Bool = false,
    absentValue: TrainingParameterAbsence<Bool>? = nil
) = #externalMacro(module: "TrainingParametersMacroPlugin", type: "TrainingParameterMacro")
