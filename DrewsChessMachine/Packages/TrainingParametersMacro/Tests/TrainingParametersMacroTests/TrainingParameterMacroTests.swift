import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest
@testable import TrainingParametersMacroPlugin

final class TrainingParameterMacroTests: XCTestCase {
    let macros: [String: Macro.Type] = [
        "TrainingParameter": TrainingParameterMacro.self
    ]

    func test_doubleParameter_withRange() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "Entropy Bonus",
                description: "Entropy regularization coefficient.",
                default: 0.0025,
                range: 0.0...0.1,
                category: "Optimizer"
            )
            public enum EntropyBonus: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum EntropyBonus: TrainingParameterKey {

                public static let id: String = "entropy_bonus"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Entropy Bonus",
                    description: "Entropy regularization coefficient.",
                    type: .double,
                    defaultValue: .double(0.0025),
                    doubleRange: NumericRange(min: 0.0, max: 0.1),
                    category: "Optimizer",
                    liveTunable: false
                )

                public static func encode(_ value: Double) -> ParameterValue {
                    .double(value)
                }

                public static func decode(_ value: ParameterValue) throws -> Double {
                    switch value {
                    case .double(let x):
                        return x
                    case .int(let x):
                        return Double(x)
                    default:
                        throw TrainingConfigError.wrongType(id: id)
                    }
                }
            }
            """,
            macros: macros
        )
    }

    func test_intParameter_withRange() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "Self-Play Concurrency",
                description: "Parallel self-play game count.",
                default: 6,
                range: 1...256,
                category: "Training Window",
                liveTunable: true
            )
            public enum SelfPlayConcurrency: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum SelfPlayConcurrency: TrainingParameterKey {

                public static let id: String = "self_play_concurrency"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Self-Play Concurrency",
                    description: "Parallel self-play game count.",
                    type: .int,
                    defaultValue: .int(6),
                    intRange: NumericRange(min: 1, max: 256),
                    category: "Training Window",
                    liveTunable: true
                )

                public static func encode(_ value: Int) -> ParameterValue {
                    .int(value)
                }

                public static func decode(_ value: ParameterValue) throws -> Int {
                    guard case .int(let x) = value else {
                        throw TrainingConfigError.wrongType(id: id)
                    }
                    return x
                }
            }
            """,
            macros: macros
        )
    }

    func test_boolParameter() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "Replay Ratio Auto Adjust",
                description: "Whether to auto-adjust replay ratio.",
                default: true,
                category: "Replay Buffer",
                liveTunable: true
            )
            public enum ReplayRatioAutoAdjust: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum ReplayRatioAutoAdjust: TrainingParameterKey {

                public static let id: String = "replay_ratio_auto_adjust"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Replay Ratio Auto Adjust",
                    description: "Whether to auto-adjust replay ratio.",
                    type: .bool,
                    defaultValue: .bool(true),
                    category: "Replay Buffer",
                    liveTunable: true
                )

                public static func encode(_ value: Bool) -> ParameterValue {
                    .bool(value)
                }

                public static func decode(_ value: ParameterValue) throws -> Bool {
                    guard case .bool(let x) = value else {
                        throw TrainingConfigError.wrongType(id: id)
                    }
                    return x
                }
            }
            """,
            macros: macros
        )
    }

    /// Acronyms (consecutive uppercase letters) must be preserved as a
    /// single word in the generated snake_case id, not split per-letter.
    /// `lrWarmupSteps` → `lr_warmup_steps`, NOT `l_r_warmup_steps`.
    func test_acronymSnakeCasing_leadingAcronym() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "LR Warmup Steps",
                description: "Warmup steps.",
                default: 100,
                range: 0...100000,
                category: "Optimizer"
            )
            public enum LRWarmupSteps: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum LRWarmupSteps: TrainingParameterKey {

                public static let id: String = "lr_warmup_steps"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "LR Warmup Steps",
                    description: "Warmup steps.",
                    type: .int,
                    defaultValue: .int(100),
                    intRange: NumericRange(min: 0, max: 100000),
                    category: "Optimizer",
                    liveTunable: false
                )

                public static func encode(_ value: Int) -> ParameterValue {
                    .int(value)
                }

                public static func decode(_ value: ParameterValue) throws -> Int {
                    switch value {
                    case .int(let x):
                        return x
                    case .double(let x):
                        return Int(x)
                    default:
                        throw TrainingConfigError.wrongType(id: id)
                    }
                }
            }
            """,
            macros: macros
        )
    }

    /// Acronym at the END of the name. `sqrtBatchScalingLR` →
    /// `sqrt_batch_scaling_lr`, NOT `sqrt_batch_scaling_l_r`.
    func test_acronymSnakeCasing_trailingAcronym() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "Sqrt Batch Scaling LR",
                description: "Scale lr by sqrt(batch).",
                default: true,
                category: "Optimizer"
            )
            public enum SqrtBatchScalingLR: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum SqrtBatchScalingLR: TrainingParameterKey {

                public static let id: String = "sqrt_batch_scaling_lr"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Sqrt Batch Scaling LR",
                    description: "Scale lr by sqrt(batch).",
                    type: .bool,
                    defaultValue: .bool(true),
                    category: "Optimizer",
                    liveTunable: false
                )

                public static func encode(_ value: Bool) -> ParameterValue {
                    .bool(value)
                }

                public static func decode(_ value: ParameterValue) throws -> Bool {
                    switch value {
                    case .bool(let x):
                        return x
                    default:
                        throw TrainingConfigError.wrongType(id: id)
                    }
                }
            }
            """,
            macros: macros
        )
    }

    func test_idOverride() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "Policy Scale K",
                description: "Policy scale.",
                default: 5.0,
                range: 0.1...20.0,
                category: "Optimizer",
                id: "K"
            )
            public enum PolicyScaleK: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum PolicyScaleK: TrainingParameterKey {

                public static let id: String = "K"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Policy Scale K",
                    description: "Policy scale.",
                    type: .double,
                    defaultValue: .double(5.0),
                    doubleRange: NumericRange(min: 0.1, max: 20.0),
                    category: "Optimizer",
                    liveTunable: false
                )

                public static func encode(_ value: Double) -> ParameterValue {
                    .double(value)
                }

                public static func decode(_ value: ParameterValue) throws -> Double {
                    switch value {
                    case .double(let x):
                        return x
                    case .int(let x):
                        return Double(x)
                    default:
                        throw TrainingConfigError.wrongType(id: id)
                    }
                }
            }
            """,
            macros: macros
        )
    }

    /// `absentValue:` declares what a resume applies when a checkpoint carries
    /// no value for the key; the macro surfaces it as `absentValue`, typed by
    /// the parameter's value type.
    func test_absentValue_isEmittedWhenDeclared() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "Dropout Rate",
                description: "Channel dropout.",
                default: 0.0,
                range: 0.0...0.95,
                category: "Regularization",
                absentValue: .preFeature(0.0)
            )
            public enum DropoutRate: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum DropoutRate: TrainingParameterKey {

                public static let id: String = "dropout_rate"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Dropout Rate",
                    description: "Channel dropout.",
                    type: .double,
                    defaultValue: .double(0.0),
                    doubleRange: NumericRange(min: 0.0, max: 0.95),
                    category: "Regularization",
                    liveTunable: false
                )

                public static func encode(_ value: Double) -> ParameterValue {
                    .double(value)
                }

                public static func decode(_ value: ParameterValue) throws -> Double {
                    switch value {
                    case .double(let x):
                        return x
                    case .int(let x):
                        return Double(x)
                    default:
                        throw TrainingConfigError.wrongType(id: id)
                    }
                }

                public static let absentValue: TrainingParameterAbsence<Double> = .preFeature(0.0)
            }
            """,
            macros: macros
        )
    }

    func test_absentValue_currentSetting() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "Arena Auto Interval",
                description: "Seconds between automatic arenas.",
                default: 900.0,
                range: 60.0...86400.0,
                category: "Arena",
                liveTunable: true,
                absentValue: .currentSetting
            )
            public enum ArenaAutoIntervalSec: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum ArenaAutoIntervalSec: TrainingParameterKey {

                public static let id: String = "arena_auto_interval_sec"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Arena Auto Interval",
                    description: "Seconds between automatic arenas.",
                    type: .double,
                    defaultValue: .double(900.0),
                    doubleRange: NumericRange(min: 60.0, max: 86400.0),
                    category: "Arena",
                    liveTunable: true
                )

                public static func encode(_ value: Double) -> ParameterValue {
                    .double(value)
                }

                public static func decode(_ value: ParameterValue) throws -> Double {
                    switch value {
                    case .double(let x):
                        return x
                    case .int(let x):
                        return Double(x)
                    default:
                        throw TrainingConfigError.wrongType(id: id)
                    }
                }

                public static let absentValue: TrainingParameterAbsence<Double> = .currentSetting
            }
            """,
            macros: macros
        )
    }

    /// A `UInt64(<literal>)` default selects the full-range unsigned kind
    /// (a random seed), with its own range and decode.
    func test_uint64Parameter_withRange() {
        assertMacroExpansion(
            """
            @TrainingParameter(
                name: "Random Seed",
                description: "Master seed.",
                default: UInt64(0),
                range: 0...UInt64.max,
                category: "Reproducibility",
                id: "random_seed",
                absentValue: .refuseExact
            )
            public enum RandomSeed: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum RandomSeed: TrainingParameterKey {

                public static let id: String = "random_seed"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Random Seed",
                    description: "Master seed.",
                    type: .uint64,
                    defaultValue: .uint64(UInt64(0)),
                    uint64Range: NumericRange(min: 0, max: UInt64.max),
                    category: "Reproducibility",
                    liveTunable: false
                )

                public static func encode(_ value: UInt64) -> ParameterValue {
                    .uint64(value)
                }

                public static func decode(_ value: ParameterValue) throws -> UInt64 {
                    switch value {
                    case .uint64(let x):
                        return x
                    case .int(let x) where x >= 0:
                        return UInt64(x)
                    default:
                        throw TrainingConfigError.wrongType(id: id)
                    }
                }

                public static let absentValue: TrainingParameterAbsence<UInt64> = .refuseExact
            }
            """,
            macros: macros
        )
    }
}
