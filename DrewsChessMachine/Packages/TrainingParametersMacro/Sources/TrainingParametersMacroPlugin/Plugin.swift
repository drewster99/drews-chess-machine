import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct TrainingParametersMacroPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        TrainingParameterMacro.self
    ]
}
