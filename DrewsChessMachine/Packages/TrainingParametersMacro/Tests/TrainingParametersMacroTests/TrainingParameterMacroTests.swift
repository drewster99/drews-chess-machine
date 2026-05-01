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
                name: "Self-Play Workers",
                description: "Parallel self-play game count.",
                default: 6,
                range: 1...256,
                category: "Training Window",
                liveTunable: true
            )
            public enum SelfPlayWorkers: TrainingParameterKey {}
            """,
            expandedSource: """
            public enum SelfPlayWorkers: TrainingParameterKey {

                public static let id: String = "self_play_workers"

                public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
                    id: id,
                    name: "Self-Play Workers",
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
}
