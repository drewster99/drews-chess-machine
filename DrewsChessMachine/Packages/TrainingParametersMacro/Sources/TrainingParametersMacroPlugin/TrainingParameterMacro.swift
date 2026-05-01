import SwiftSyntax
import SwiftSyntaxMacros

public struct TrainingParameterMacro: MemberMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingMembersOf declaration: some DeclGroupSyntax,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        guard let enumDecl = declaration.as(EnumDeclSyntax.self) else {
            throw MacroError.notAnEnum
        }

        let typeName = enumDecl.name.text
        let args = try parseArguments(node)
        let id = args.id ?? snakeCase(from: typeName)
        let valueKind = args.valueKind
        let swiftTypeName = valueKind.swiftTypeName

        // static let id: String
        let idDecl: DeclSyntax = """
        public static let id: String = \(literal: id)
        """

        // static let definition: TrainingParameterDefinition
        let defaultValueFragment: String
        let parameterTypeCase: String
        var rangeArgLine: String = ""
        switch valueKind {
        case .double(let lo, let hi):
            defaultValueFragment = ".double(\(args.defaultExpr))"
            parameterTypeCase = ".double"
            rangeArgLine = "    doubleRange: NumericRange(min: \(lo), max: \(hi)),\n"
        case .int(let lo, let hi):
            defaultValueFragment = ".int(\(args.defaultExpr))"
            parameterTypeCase = ".int"
            rangeArgLine = "    intRange: NumericRange(min: \(lo), max: \(hi)),\n"
        case .bool:
            defaultValueFragment = ".bool(\(args.defaultExpr))"
            parameterTypeCase = ".bool"
        }

        let definitionDecl: DeclSyntax = """
        public static let definition: TrainingParameterDefinition = TrainingParameterDefinition(
            id: id,
            name: \(raw: args.nameExpr),
            description: \(raw: args.descriptionExpr),
            type: \(raw: parameterTypeCase),
            defaultValue: \(raw: defaultValueFragment),
        \(raw: rangeArgLine)    category: \(raw: args.categoryExpr),
            liveTunable: \(raw: args.liveTunableExpr)
        )
        """

        // static func encode(_ value: Value) -> ParameterValue
        let encodeDecl: DeclSyntax = """
        public static func encode(_ value: \(raw: swiftTypeName)) -> ParameterValue {
            \(raw: parameterTypeCase)(value)
        }
        """

        // static func decode(_ value: ParameterValue) throws -> Value
        let decodeBody: String
        switch valueKind {
        case .double:
            decodeBody = """
            switch value {
            case .double(let x): return x
            case .int(let x): return Double(x)
            default: throw TrainingConfigError.wrongType(id: id)
            }
            """
        case .int:
            decodeBody = """
            guard case .int(let x) = value else {
                throw TrainingConfigError.wrongType(id: id)
            }
            return x
            """
        case .bool:
            decodeBody = """
            guard case .bool(let x) = value else {
                throw TrainingConfigError.wrongType(id: id)
            }
            return x
            """
        }

        let decodeDecl: DeclSyntax = """
        public static func decode(_ value: ParameterValue) throws -> \(raw: swiftTypeName) {
            \(raw: decodeBody)
        }
        """

        return [idDecl, definitionDecl, encodeDecl, decodeDecl]
    }
}

// MARK: - Argument parsing

private struct ParsedArgs {
    var nameExpr: String
    var descriptionExpr: String
    var categoryExpr: String
    var defaultExpr: String
    var liveTunableExpr: String
    var id: String?
    var valueKind: ValueKind
}

private enum ValueKind {
    case double(min: String, max: String)
    case int(min: String, max: String)
    case bool

    var swiftTypeName: String {
        switch self {
        case .double: return "Double"
        case .int: return "Int"
        case .bool: return "Bool"
        }
    }
}

private enum MacroError: Error, CustomStringConvertible {
    case notAnEnum
    case missingArgument(String)
    case rangeRequired
    case unknownDefaultType

    var description: String {
        switch self {
        case .notAnEnum:
            return "@TrainingParameter can only be attached to an enum declaration"
        case .missingArgument(let name):
            return "@TrainingParameter missing required argument '\(name)'"
        case .rangeRequired:
            return "@TrainingParameter requires 'range:' for numeric parameters"
        case .unknownDefaultType:
            return "@TrainingParameter could not infer the parameter type from 'default:' (must be a Double, Int, or Bool literal)"
        }
    }
}

private func parseArguments(_ node: AttributeSyntax) throws -> ParsedArgs {
    guard let arguments = node.arguments?.as(LabeledExprListSyntax.self) else {
        throw MacroError.missingArgument("name")
    }

    var byLabel: [String: ExprSyntax] = [:]
    for element in arguments {
        guard let label = element.label?.text else { continue }
        byLabel[label] = element.expression
    }

    guard let nameExpr = byLabel["name"] else { throw MacroError.missingArgument("name") }
    guard let descriptionExpr = byLabel["description"] else { throw MacroError.missingArgument("description") }
    guard let categoryExpr = byLabel["category"] else { throw MacroError.missingArgument("category") }
    guard let defaultExpr = byLabel["default"] else { throw MacroError.missingArgument("default") }

    let liveTunableExpr = byLabel["liveTunable"].map { $0.description } ?? "false"

    let idLiteral: String? = byLabel["id"].flatMap { stringLiteralValue($0) }

    let kind = try valueKind(forDefault: defaultExpr, range: byLabel["range"])

    return ParsedArgs(
        nameExpr: nameExpr.description,
        descriptionExpr: descriptionExpr.description,
        categoryExpr: categoryExpr.description,
        defaultExpr: defaultExpr.description,
        liveTunableExpr: liveTunableExpr,
        id: idLiteral,
        valueKind: kind
    )
}

private func valueKind(forDefault defaultExpr: ExprSyntax, range: ExprSyntax?) throws -> ValueKind {
    if defaultExpr.is(BooleanLiteralExprSyntax.self) {
        return .bool
    }
    if defaultExpr.is(FloatLiteralExprSyntax.self) {
        guard let r = range else { throw MacroError.rangeRequired }
        let (lo, hi) = try parseRange(r)
        return .double(min: lo, max: hi)
    }
    if defaultExpr.is(IntegerLiteralExprSyntax.self) {
        // Without a 'range:' we can't tell Int from Double — but range is required
        // for numeric parameters, so this is fine.
        guard let r = range else { throw MacroError.rangeRequired }
        let (lo, hi, looksFloat) = try parseRangeWithFloatHint(r)
        if looksFloat {
            return .double(min: lo, max: hi)
        } else {
            return .int(min: lo, max: hi)
        }
    }
    // Negation of a numeric literal (e.g. `-1.0`, `-5`) shows up as PrefixOperatorExpr.
    if let prefix = defaultExpr.as(PrefixOperatorExprSyntax.self),
       prefix.operator.text == "-" {
        if prefix.expression.is(FloatLiteralExprSyntax.self) {
            guard let r = range else { throw MacroError.rangeRequired }
            let (lo, hi) = try parseRange(r)
            return .double(min: lo, max: hi)
        }
        if prefix.expression.is(IntegerLiteralExprSyntax.self) {
            guard let r = range else { throw MacroError.rangeRequired }
            let (lo, hi, looksFloat) = try parseRangeWithFloatHint(r)
            return looksFloat ? .double(min: lo, max: hi) : .int(min: lo, max: hi)
        }
    }
    throw MacroError.unknownDefaultType
}

private func parseRange(_ rangeExpr: ExprSyntax) throws -> (String, String) {
    if let seq = rangeExpr.as(SequenceExprSyntax.self) {
        let elements = Array(seq.elements)
        if elements.count == 3,
           let op = elements[1].as(BinaryOperatorExprSyntax.self),
           op.operator.text == "..." {
            return (elements[0].description, elements[2].description)
        }
    }
    if let infix = rangeExpr.as(InfixOperatorExprSyntax.self),
       let op = infix.operator.as(BinaryOperatorExprSyntax.self),
       op.operator.text == "..." {
        return (infix.leftOperand.description, infix.rightOperand.description)
    }
    throw MacroError.rangeRequired
}

private func parseRangeWithFloatHint(_ rangeExpr: ExprSyntax) throws -> (String, String, Bool) {
    let (lo, hi) = try parseRange(rangeExpr)
    let looksFloat = lo.contains(".") || hi.contains(".") || lo.contains("e") || hi.contains("e")
    return (lo, hi, looksFloat)
}

private func stringLiteralValue(_ expr: ExprSyntax) -> String? {
    guard let lit = expr.as(StringLiteralExprSyntax.self) else { return nil }
    var result = ""
    for segment in lit.segments {
        if let text = segment.as(StringSegmentSyntax.self) {
            result += text.content.text
        } else {
            return nil
        }
    }
    return result
}

private func snakeCase(from camel: String) -> String {
    var result = ""
    for (i, ch) in camel.enumerated() {
        if ch.isUppercase {
            if i > 0 { result.append("_") }
            result.append(Character(ch.lowercased()))
        } else {
            result.append(ch)
        }
    }
    return result
}
