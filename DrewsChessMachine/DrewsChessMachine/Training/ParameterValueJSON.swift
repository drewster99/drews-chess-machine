import Foundation
import TrainingParametersMacroSupport

extension ParameterValue {

    /// A flat JSON object of parameter values — a `parameters.json`, a saved
    /// settings file, a lineage record's parameter snapshot — read as
    /// `[id: ParameterValue]`, every number exactly the value its text
    /// spells. The one reader of parameter JSON text: `--parameters`
    /// (`CliTrainingConfig.load`), the settings load
    /// (`TrainingParameters.load(from:)`) and the CLI resume comparison
    /// (`differences(fromLineage:)`) all go through it.
    ///
    /// **Why two parsers.** Every writer of parameter JSON uses
    /// `JSONSerialization`, which spells a `Double` with every significant
    /// digit (`0.0003` as `0.00029999999999999997`). `JSONSerialization`'s
    /// own parser does not read every such spelling back to the same
    /// `Double` (`weight_decay` 0.0003 came back as 0.0002999999999999999),
    /// so a run started from an untouched defaults file recorded, hashed and
    /// resume-compared a value nobody set. `JSONDecoder` reads every `Double`
    /// exactly, but cannot tell `42.0` from `42`, a distinction the kinds
    /// rest on: `ParameterValue(jsonValue:)` reads a number written with a
    /// fraction or exponent as `.double` (so `42.0` for an integer
    /// parameter is refused as the wrong type) and true/false as `.bool`
    /// (never `1`). So the kind of each value comes from `JSONSerialization`,
    /// exactly as before, and the value of every `.double` from `JSONDecoder`.
    ///
    /// The LR keyframe list's numbers are read the same way: kind and shape
    /// from `JSONSerialization`, every value from `JSONDecoder`.
    ///
    /// **Previous LR schedule keys.** Text written before LR keyframes
    /// carries the single-decay schedule's nine keys (`lr_cycle_min` …
    /// `momentum_follow_end_high`). They are converted here, in the one
    /// reader, to `lr_cycle_keyframes` (`LegacyLRScheduleKeys.converting`),
    /// so every old `parameters.json`, settings file and lineage snapshot
    /// reads as the schedule it described, bit for bit (owner decision Q-4).
    /// `currentValue` supplies a key the conversion needs that the text
    /// omits — a partial `parameters.json` leaves it to the current setting,
    /// as `--parameters` always has; the default reads the persisted setting.
    ///
    /// Throws `TrainingConfigError.wrongType(id: "<root>")` when the text is
    /// not a JSON object, `wrongType(id:)` for a value no parameter kind
    /// reads, and the underlying error for text that is not JSON.
    static func parametersObject(
        fromJSON data: Data,
        currentValue: (String) -> ParameterValue? = LegacyLRScheduleKeys.persistedSetting
    ) throws -> [String: ParameterValue] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TrainingConfigError.wrongType(id: "<root>")
        }
        let exactNumbers = try JSONDecoder().decode([String: ExactJSONNumber].self, from: data)
        var values: [String: ParameterValue] = [:]
        for (id, jsonValue) in object {
            let value = try ParameterValue(jsonValue: jsonValue, id: id)
            if case .double = value {
                // The same text, read by the exact parser. Both parsers read
                // the same object, so a key one has and the other lacks, or
                // a number one reads and the other does not, is a parser
                // disagreement — an error, never a silent pick of either.
                guard case .number(let exact) = exactNumbers[id] else {
                    throw TrainingConfigError.wrongType(id: id)
                }
                values[id] = .double(exact)
            } else if case .lrKeyframes = value {
                guard case .keyframes(let exact) = exactNumbers[id] else {
                    throw TrainingConfigError.wrongType(id: id)
                }
                values[id] = .lrKeyframes(exact)
            } else {
                values[id] = value
            }
        }
        let converted = try LegacyLRScheduleKeys.converting(values, currentValue: currentValue)
        if !converted.notes.isEmpty {
            LegacyLRScheduleKeys.logNote("[PARAM] legacy LR schedule keys converted: " + converted.notes.joined(separator: "; "))
        }
        return converted.values
    }
}

/// One value of a parameter JSON object as `JSONDecoder` reads it: a number,
/// read exactly as a `Double`; an LR keyframe list, its numbers read exactly;
/// or anything else (true/false, a string, null, another array or object),
/// which `ParameterValue.parametersObject(fromJSON:)` takes from
/// `JSONSerialization` instead.
private enum ExactJSONNumber: Decodable {
    case number(Double)
    case keyframes(LRKeyframeSchedule)
    case notANumber

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .notANumber
            return
        }
        do {
            self = .number(try container.decode(Double.self))
        } catch DecodingError.typeMismatch {
            // Not a number: a keyframe list, or a kind `JSONSerialization`
            // supplies (a mismatch here is classification, not an error; a
            // keyframe value that fails to decode is refused by the caller's
            // `guard case .keyframes`).
            do {
                self = .keyframes(try container.decode(LRKeyframeSchedule.self))
            } catch is DecodingError {
                self = .notANumber
            }
        }
    }
}
