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
/// Expands to: `id`, `definition`, `encode(_:)`, `decode(_:)`.
@attached(member, names: named(id), named(definition), named(encode), named(decode))
public macro TrainingParameter(
    name: String,
    description: String,
    default: Double,
    range: ClosedRange<Double>,
    category: String,
    id: String? = nil,
    liveTunable: Bool = false
) = #externalMacro(module: "TrainingParametersMacroPlugin", type: "TrainingParameterMacro")

@attached(member, names: named(id), named(definition), named(encode), named(decode))
public macro TrainingParameter(
    name: String,
    description: String,
    default: Int,
    range: ClosedRange<Int>,
    category: String,
    id: String? = nil,
    liveTunable: Bool = false
) = #externalMacro(module: "TrainingParametersMacroPlugin", type: "TrainingParameterMacro")

@attached(member, names: named(id), named(definition), named(encode), named(decode))
public macro TrainingParameter(
    name: String,
    description: String,
    default: Bool,
    category: String,
    id: String? = nil,
    liveTunable: Bool = false
) = #externalMacro(module: "TrainingParametersMacroPlugin", type: "TrainingParameterMacro")
