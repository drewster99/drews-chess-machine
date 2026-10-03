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
        case .uint64(let lo, let hi):
            defaultValueFragment = ".uint64(\(args.defaultExpr))"
            parameterTypeCase = ".uint64"
            rangeArgLine = "    uint64Range: NumericRange(min: \(lo), max: \(hi)),\n"
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
        case .uint64:
            // A non-negative JSON integer is accepted as well as the decimal
            // string the app writes (it is exact up to `Int.max`); a negative
            // one, a fraction or anything else is the wrong type.
            decodeBody = """
            switch value {
            case .uint64(let x): return x
            case .int(let x) where x >= 0: return UInt64(x)
            default: throw TrainingConfigError.wrongType(id: id)
            }
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

        var members = [idDecl, definitionDecl, encodeDecl, decodeDecl]
        if let absentValueExpr = args.absentValueExpr {
            let absentValueDecl: DeclSyntax = """
            public static let absentValue: TrainingParameterAbsence<\(raw: swiftTypeName)> = \(raw: absentValueExpr)
            """
            members.append(absentValueDecl)
        }
        return members
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
    var absentValueExpr: String?
}

private enum ValueKind {
    case double(min: String, max: String)
    case int(min: String, max: String)
    /// Declared with `default: UInt64(<literal>)` — the explicit conversion is
    /// how the declaration says the full unsigned 64-bit range is meant (an
    /// integer literal alone reads as `Int`).
    case uint64(min: String, max: String)
    case bool

    var swiftTypeName: String {
        switch self {
        case .double: return "Double"
        case .int: return "Int"
        case .uint64: return "UInt64"
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
            return "@TrainingParameter could not infer the parameter type from 'default:' (must be a Double, Int, or Bool literal, or UInt64(<integer literal>))"
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
        valueKind: kind,
        absentValueExpr: byLabel["absentValue"].map { $0.description }
    )
}

private func valueKind(forDefault defaultExpr: ExprSyntax, range: ExprSyntax?) throws -> ValueKind {
    if defaultExpr.is(BooleanLiteralExprSyntax.self) {
        return .bool
    }
    if let call = defaultExpr.as(FunctionCallExprSyntax.self),
       let callee = call.calledExpression.as(DeclReferenceExprSyntax.self),
       callee.baseName.text == "UInt64" {
        guard let r = range else { throw MacroError.rangeRequired }
        let (lo, hi) = try parseRange(r)
        return .uint64(min: lo, max: hi)
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
    // Snake-case conversion that preserves acronyms as single words.
    // Insert `_` before an uppercase character only at:
    //   1) a lower→upper boundary (end of a previous word), OR
    //   2) an upper→upper→lower boundary (last upper of an acronym
    //      that introduces a new lowercase-starting word).
    // Examples: `learningRate`→`learning_rate`, `lrWarmupSteps`→
    // `lr_warmup_steps`, `sqrtBatchScalingLR`→`sqrt_batch_scaling_lr`,
    // `parseXMLFile`→`parse_xml_file`, `XMLParser`→`xml_parser`.
    let chars = Array(camel)
    var result = ""
    for i in 0..<chars.count {
        let ch = chars[i]
        if ch.isUppercase && i > 0 {
            let prev = chars[i - 1]
            let nextIsLower = (i + 1) < chars.count && chars[i + 1].isLowercase
            if prev.isLowercase || (prev.isUppercase && nextIsLower) {
                result.append("_")
            }
        }
        result.append(Character(ch.lowercased()))
    }
    return result
}
