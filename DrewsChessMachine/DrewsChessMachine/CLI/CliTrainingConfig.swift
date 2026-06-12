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

    /// Load and decode a parameters JSON file from disk. Throws on
    /// I/O failure or malformed JSON.
    ///
    /// `training_time_limit` is pulled out of the values map before
    /// it gets handed to `TrainingParameters.apply(_:)`, since that
    /// id is not a registered parameter.
    static func load(from url: URL) throws -> CliTrainingConfig {
        let data = try Data(contentsOf: url)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TrainingConfigError.wrongType(id: "<root>")
        }

        var values: [String: ParameterValue] = [:]
        var trainingTimeLimitSec: Double?
        var trainingStepLimit: Int?

        for (id, anyValue) in dict {
            if id == "training_time_limit" {
                if let n = anyValue as? NSNumber {
                    trainingTimeLimitSec = n.doubleValue
                } else {
                    throw TrainingConfigError.wrongType(id: id)
                }
                continue
            }
            if id == "training_step_limit" {
                if let n = anyValue as? NSNumber {
                    trainingStepLimit = n.intValue
                } else {
                    throw TrainingConfigError.wrongType(id: id)
                }
                continue
            }

            // NSNumber bridging is treacherous: `as? Bool` succeeds for any
            // NSNumber (so `1` parses as `true`). Disambiguate via objCType
            // — true/false are char-typed ("c"/"B"), JSON ints are "q",
            // JSON doubles are "d". The cast-to-Bool path only fires when
            // JSONSerialization actually emitted a Bool.
            if let n = anyValue as? NSNumber {
                let typeChar = String(cString: n.objCType)
                switch typeChar {
                case "c", "B":
                    values[id] = .bool(n.boolValue)
                case "d", "f":
                    values[id] = .double(n.doubleValue)
                default:
                    values[id] = .int(n.intValue)
                }
            } else {
                throw TrainingConfigError.wrongType(id: id)
            }
        }

        return CliTrainingConfig(
            trainingParameters: values,
            trainingTimeLimitSec: trainingTimeLimitSec,
            trainingStepLimit: trainingStepLimit
        )
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
            switch value {
            case .bool(let x): parts.append("\(id)=\(x)")
            case .int(let x): parts.append("\(id)=\(x)")
            case .double(let x): parts.append("\(id)=\(x)")
            }
        }
        if let t = trainingTimeLimitSec {
            parts.append("training_time_limit=\(t)")
        }
        if let s = trainingStepLimit {
            parts.append("training_step_limit=\(s)")
        }
        return parts.isEmpty ? "(empty)" : parts.joined(separator: " ")
    }
}
