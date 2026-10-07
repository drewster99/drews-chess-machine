import Foundation

/// `DrewsChessMachine --replay-health-log <log> [<log> …]
/// [--learning-grace-steps N] [--lr-warmup-steps N]
/// [--segment-step-as-trainer-step]`: run the real training-health monitor
/// and evaluator over saved session logs (`TrainingHealthLogReplay`, the
/// alarms plan D4) and print what they would have raised. No GUI, no GPU, no
/// writes: the logs are only read.
///
/// The output is the replay's header (the limitations of the logs given),
/// every line the monitors wrote — in the live `[HEALTH]` / `[ALARM] health`
/// formats, so the same greps work on it — and a per-rule summary table.
///
/// The settings are the declared defaults (never the user's saved settings,
/// so a replay is reproducible anywhere) with `--learning-grace-steps` /
/// `--lr-warmup-steps` overriding the two the logs do not carry; every
/// action is `log`, since nothing here can stop.
///
/// Exit status 0 on success; 2 on a usage error, an unreadable log, a
/// malformed or unsupported log line (named with its file and line), or a
/// log with neither step rows nor `[LAYER-HEALTH]` lines.
enum TrainingHealthReplayCLI {

    static let flag = "--replay-health-log"
    static let learningGraceFlag = "--learning-grace-steps"
    static let warmupFlag = "--lr-warmup-steps"
    static let segmentStepFlag = "--segment-step-as-trainer-step"

    struct Arguments: Equatable {
        let logPaths: [String]
        let learningGraceSteps: Int?
        let lrWarmupSteps: Int?
        let segmentStepAsTrainerStep: Bool
    }

    enum UsageError: LocalizedError, Equatable {
        case noLogs
        case unknownFlag(String)
        case missingValue(String)
        case notAnInteger(flag: String, value: String)
        case repeatedFlag(String)

        var errorDescription: String? {
            switch self {
            case .noLogs:
                return "\(TrainingHealthReplayCLI.flag) requires at least one session log path"
            case .unknownFlag(let flag):
                return "\(TrainingHealthReplayCLI.flag) does not accept '\(flag)'"
            case .missingValue(let flag):
                return "\(flag) requires a value"
            case .notAnInteger(let flag, let value):
                return "\(flag) takes a whole number of trainer steps, got '\(value)'"
            case .repeatedFlag(let flag):
                return "\(flag) given more than once"
            }
        }
    }

    /// Parse the arguments after the executable path. Every token that is
    /// not a flag or a flag's value is a log path, in order.
    static func parse(_ arguments: [String]) throws -> Arguments {
        var paths: [String] = []
        var grace: Int?
        var warmup: Int?
        var segmentStep = false
        var sawReplayFlag = false
        var index = 0
        func integer(after flag: String) throws -> Int {
            let valueIndex = index + 1
            guard valueIndex < arguments.count, !arguments[valueIndex].hasPrefix("--") else {
                throw UsageError.missingValue(flag)
            }
            let text = arguments[valueIndex]
            guard let value = Int(text), value >= 0 else {
                throw UsageError.notAnInteger(flag: flag, value: text)
            }
            index = valueIndex
            return value
        }
        while index < arguments.count {
            let token = arguments[index]
            switch token {
            case flag:
                guard !sawReplayFlag else { throw UsageError.repeatedFlag(flag) }
                sawReplayFlag = true
            case learningGraceFlag:
                guard grace == nil else { throw UsageError.repeatedFlag(token) }
                grace = try integer(after: token)
            case warmupFlag:
                guard warmup == nil else { throw UsageError.repeatedFlag(token) }
                warmup = try integer(after: token)
            case segmentStepFlag:
                guard !segmentStep else { throw UsageError.repeatedFlag(token) }
                segmentStep = true
            default:
                guard !token.hasPrefix("--") else { throw UsageError.unknownFlag(token) }
                paths.append(token)
            }
            index += 1
        }
        guard !paths.isEmpty else { throw UsageError.noLogs }
        return Arguments(
            logPaths: paths, learningGraceSteps: grace, lrWarmupSteps: warmup,
            segmentStepAsTrainerStep: segmentStep)
    }

    /// The config a replay runs under: the declared defaults (alarms on,
    /// every action `log`) with the two overrides. Validated like any
    /// parameters file, so an out-of-range value is refused.
    static func config(for arguments: Arguments) throws -> TrainingHealthConfig {
        var overrides: [String: ParameterValue] = [:]
        if let grace = arguments.learningGraceSteps {
            overrides[TrainingHealthLearningGraceSteps.id] = TrainingHealthLearningGraceSteps.encode(grace)
        }
        if let warmup = arguments.lrWarmupSteps {
            overrides[LRWarmupSteps.id] = LRWarmupSteps.encode(warmup)
        }
        return try TrainingHealthConfig(TrainingParametersSnapshot.declaredDefaults(overriding: overrides))
    }

    /// Read the logs and replay them. Throws on an unreadable log and on
    /// everything `TrainingHealthLogReplay.run` refuses.
    static func replay(_ arguments: Arguments) throws -> TrainingHealthLogReplay.Output {
        let sources = try arguments.logPaths.map { path in
            TrainingHealthLogReplay.Source(
                name: (path as NSString).lastPathComponent,
                text: try String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8))
        }
        return try TrainingHealthLogReplay.run(
            sources,
            options: TrainingHealthLogReplay.Options(
                segmentStepAsTrainerStep: arguments.segmentStepAsTrainerStep,
                config: try config(for: arguments)))
    }

    /// The per-rule summary: raises, the first raise, the highest severity
    /// reached, clears, and the severity still active at the end (`-` when
    /// none), one aligned row per rule.
    static func summaryLines(_ output: TrainingHealthLogReplay.Output) -> [String] {
        var lines = ["[HEALTH] replay summary"]
        let header = ["rule", "raises", "first_raise", "highest", "clears", "active_at_end"]
        var rows: [[String]] = [header]
        for rule in TrainingHealthRule.allCases {
            let events = output.events.filter { $0.rule == rule }
            let raises = events.filter { $0.kind == .raise }
            let clears = events.filter { $0.kind == .clear }
            let raisedOrEscalated = events.filter { $0.kind == .raise || $0.kind == .escalate }
            let highest = raisedOrEscalated.max { $0.severity.healthRank < $1.severity.healthRank }
            var activeSeverity: TrainingAlarm.Severity?
            for event in events {
                switch event.kind {
                case .raise, .escalate, .worsen, .active:
                    activeSeverity = event.severity
                case .clear:
                    activeSeverity = nil
                case .stop:
                    break
                }
            }
            rows.append([
                rule.rawValue,
                String(raises.count),
                raises.first.map { String($0.trainerStep) } ?? "-",
                highest?.severity.rawValue ?? "-",
                String(clears.count),
                activeSeverity?.rawValue ?? "-",
            ])
        }
        let widths = header.indices.map { column in rows.map { $0[column].count }.max() ?? 0 }
        for row in rows {
            let cells = row.enumerated().map { column, cell in
                column == 0
                    ? cell.padding(toLength: widths[column], withPad: " ", startingAt: 0)
                    : String(repeating: " ", count: widths[column] - cell.count) + cell
            }
            lines.append("  " + cells.joined(separator: "  "))
        }
        return lines
    }

    /// Entry point from the launch pre-flight; never returns.
    static func runAndExit(arguments rawArguments: [String]) -> Never {
        do {
            let arguments = try parse(rawArguments)
            let output = try replay(arguments)
            var text = output.header.map { "# \($0)" }
            text.append(contentsOf: output.lines.map(\.text))
            text.append(contentsOf: summaryLines(output))
            print(text.joined(separator: "\n"))
            Darwin.exit(0)
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            Darwin.exit(2)
        }
    }
}
