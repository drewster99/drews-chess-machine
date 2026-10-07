import Foundation

/// Parameters JSON loader for the `--parameters <file>` flag.
///
/// The file is a flat snake_case JSON object. Most keys map to
/// `TrainingParameters` registered ids and pass through validation
/// against the corresponding `TrainingParameterDefinition`. One key
/// — `training_time_limit` — is the session-time budget (a CLI launch
/// concern, not a training tunable), so it's surfaced separately.
///
/// Unknown keys are passed through to `TrainingParameters.apply`,
/// which throws `TrainingConfigError.unknownParameter` for any
/// unrecognized id. This loader does not pre-validate; the apply
/// path is strict (a file written by a newer build with a new key
/// will fail to load on an older build, and a typo in a recognized
/// id will likewise surface as an `unknownParameter` error at apply
/// time).
struct CliTrainingConfig: Sendable {
    /// Map of `TrainingParameters` ids to typed `ParameterValue`
    /// payloads. Feed this to `TrainingParameters.shared.apply(_:)`
    /// to populate the singleton; per-field validation happens there.
    var trainingParameters: [String: ParameterValue]

    /// Session wall-clock budget. Only takes effect when `--output`
    /// is also supplied (per the existing CLI semantics). Nil when
    /// the params file did not include `training_time_limit`.
    var trainingTimeLimitSec: Double?

    /// Session step budget: stop (snapshot + exit, identical dance to
    /// the time limit) once the trainer's completed-step counter reaches
    /// this value. Same `--output` gating as the time limit. Both limits
    /// may be set — whichever fires first wins. Nil when the params file
    /// did not include `training_step_limit`.
    var trainingStepLimit: Int?

    /// The two budget keys a parameters file may carry beside the
    /// registered parameters.
    static let trainingTimeLimitKey = "training_time_limit"
    static let trainingStepLimitKey = "training_step_limit"

    /// Load and decode a parameters JSON file from disk. Throws on
    /// I/O failure or malformed JSON.
    ///
    /// The file is read through `ParameterValue.parametersObject(fromJSON:)`,
    /// so every number is exactly the value its text spells — a defaults
    /// file written by `--create-parameters-file` reads back to the
    /// declared defaults bit for bit.
    ///
    /// `training_time_limit` and `training_step_limit` are pulled out of
    /// the values map before it gets handed to `TrainingParameters.apply(_:)`,
    /// since those ids are not registered parameters. The time limit is any
    /// number of seconds; the step limit a whole number (a value with a
    /// fraction, or true/false, is refused rather than truncated or read as
    /// 1).
    static func load(from url: URL) throws -> CliTrainingConfig {
        var values = try ParameterValue.parametersObject(fromJSON: try Data(contentsOf: url))
        var trainingTimeLimitSec: Double?
        var trainingStepLimit: Int?

        if let raw = values.removeValue(forKey: trainingTimeLimitKey) {
            switch raw {
            case .int(let seconds): trainingTimeLimitSec = Double(seconds)
            case .double(let seconds): trainingTimeLimitSec = seconds
            case .bool, .uint64: throw TrainingConfigError.wrongType(id: trainingTimeLimitKey)
            }
        }
        if let raw = values.removeValue(forKey: trainingStepLimitKey) {
            switch raw {
            case .int(let steps):
                trainingStepLimit = steps
            case .double(let steps):
                guard let whole = Int(exactly: steps) else {
                    throw TrainingConfigError.wrongType(id: trainingStepLimitKey)
                }
                trainingStepLimit = whole
            case .bool, .uint64:
                throw TrainingConfigError.wrongType(id: trainingStepLimitKey)
            }
        }

        return CliTrainingConfig(
            trainingParameters: values,
            trainingTimeLimitSec: trainingTimeLimitSec,
            trainingStepLimit: trainingStepLimit
        )
    }

    /// Why corpus replay refuses this file, or nil when it does not: replay
    /// enforces no wall-clock limit (owner decision O-21, gap 11) — its runs
    /// are budgeted in steps and epochs, so a run's end never depends on
    /// machine speed — and a `training_time_limit` it silently ignored would
    /// read as one it honoured. (A JSON `null` for the key is already refused
    /// by the loader, as no parameter kind reads it.)
    func corpusReplayRefusal(parametersPath: String) -> CLIRunRefusal? {
        guard trainingTimeLimitSec != nil else { return nil }
        return CLIRunRefusal(message: "corpus replay enforces no wall-clock limit; remove \(Self.trainingTimeLimitKey) "
            + "from \(parametersPath) (use training_step_limit, --training-step-limit or --epochs)")
    }

    /// Human-readable single-line summary for the `[APP]` banner —
    /// shows what the runtime actually picked up, so a typo in a
    /// key name (which would currently stay silent until apply time)
    /// is at least visible alongside the recognized values.
    func summaryString() -> String {
        var parts: [String] = []
        let sortedIds = trainingParameters.keys.sorted()
        for id in sortedIds {
            guard let value = trainingParameters[id] else { continue }
            parts.append("\(id)=\(value.displayText)")
        }
        if let t = trainingTimeLimitSec {
            parts.append("\(Self.trainingTimeLimitKey)=\(t)")
        }
        if let s = trainingStepLimit {
            parts.append("\(Self.trainingStepLimitKey)=\(s)")
        }
        return parts.isEmpty ? "(empty)" : parts.joined(separator: " ")
    }
}

extension CliTrainingConfig {
    /// Load the `--parameters` file at `path` (tilde-expanded) and apply it to
    /// `TrainingParameters.shared` as a transient, this-process-only override
    /// — the one load-and-apply path shared by the headless runners.
    ///
    /// Persistence is suppressed for the apply: a `--parameters` override must
    /// not become the GUI's next-launch default, and a headless process shares
    /// the GUI's bundle id (same `UserDefaults` domain). `apply(_:)` validates
    /// the whole file against the declared ranges before assigning anything,
    /// so a rejected file changes nothing. Errors are `TrainingConfigError`
    /// (whose `localizedDescription` names the parameter and value) or the
    /// underlying I/O / JSON error.
    @MainActor
    static func loadAndApplyTransiently(path: String) throws -> CliTrainingConfig {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let config = try load(from: url)
        TrainingParameters.suppressPersistence = true
        defer { TrainingParameters.suppressPersistence = false }
        try TrainingParameters.shared.apply(config.trainingParameters)
        return config
    }
}
