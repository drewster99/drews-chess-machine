import Foundation

enum LichessBotSettingsStoreError: LocalizedError, Equatable {
    /// The stored settings exist but can't be decoded (plan §17.1: an
    /// error state, never a silent return to defaults).
    case unreadable(detail: String)
    case invalid(problems: [String])
    /// This build's defaults don't encode as a JSON object, so saved
    /// settings can't be overlaid onto them: a defect in the build, not in
    /// the saved data, which is left as it is.
    case defaultsNotAJSONObject(typeName: String)

    var errorDescription: String? {
        switch self {
        case .unreadable(let detail):
            return "Saved Lichess bot settings can't be read: \(detail). Reset them in Settings to continue."
        case .invalid(let problems):
            return "Lichess bot settings are invalid: " + problems.joined(separator: "; ")
        case .defaultsNotAJSONObject(let typeName):
            return "This build's default \(typeName) don't encode as a JSON object, so the saved Lichess bot settings can't be read with them. The saved settings were left unchanged."
        }
    }
}

/// Persists `LichessBotSettings` as one JSON blob in `UserDefaults` (plan
/// §12.1). Not a `TrainingParameters` entry, and never part of a
/// `.dcmsession`.
enum LichessBotSettingsStore {
    static let defaultsKey = "lichess_bot_settings"

    /// What `loadReporting(from:)` read, and what it had to supply itself.
    struct LoadResult: Equatable {
        let settings: LichessBotSettings
        /// Dotted paths (e.g. `alerts`, `challenge.minimumInitialSeconds`) of
        /// non-optional settings the saved blob predates, filled from the
        /// current defaults. Empty when the saved blob is complete.
        let filledFromDefaults: [String]
        /// Dotted paths in the saved blob that the current settings no longer
        /// have (removed or renamed fields); ignored.
        let ignoredSavedKeys: [String]
    }

    /// The saved settings. With nothing saved yet (first run), the defaults;
    /// saved settings that don't decode, or decode but fail validation,
    /// throw. Settings saved before a field existed still load: see
    /// `loadReporting(from:)`.
    static func load(from defaults: UserDefaults) throws -> LichessBotSettings {
        try loadReporting(from: defaults).settings
    }

    /// Like `load(from:)`, and also reports any field filled from the current
    /// defaults because the saved settings predate it, and any saved field the
    /// current settings no longer have. The caller logs both.
    ///
    /// Why: settings are one JSON blob decoded with the synthesized
    /// `Decodable`, which throws `keyNotFound` for any non-optional field the
    /// blob lacks. Adding a field (as `alerts` in `c5542b8`) therefore made
    /// every earlier save unreadable and forced a Reset to defaults, losing
    /// the operator's configuration. Settings must survive additions, so a
    /// missing non-optional field is filled from today's default and reported
    /// — never silently.
    ///
    /// How: the saved JSON object is overlaid onto the JSON of
    /// `LichessBotSettings()` — saved values win, nested objects merge key by
    /// key, arrays are taken whole — and the merged object is decoded. A key
    /// absent from the save is filled only if it is non-optional. Optional
    /// fields encode nothing when nil, so an absent optional may be the
    /// operator's deliberate nil; it is left absent and decodes as nil. A key
    /// is optional exactly when removing it from the full default object still
    /// decodes. A saved value of the wrong type still fails to decode and is
    /// reported as unreadable, as before.
    static func loadReporting(from defaults: UserDefaults) throws -> LoadResult {
        guard let data = defaults.data(forKey: defaultsKey) else {
            return LoadResult(settings: LichessBotSettings(), filledFromDefaults: [], ignoredSavedKeys: [])
        }
        let settings: LichessBotSettings
        var filled: [String] = []
        var ignored: [String] = []
        do {
            guard let saved = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw LichessBotSettingsStoreError.unreadable(detail: "the saved settings are not a JSON object")
            }
            let defaultObject = try defaultSettingsObject()
            let merged = try overlay(
                saved: saved, onto: defaultObject, path: [], defaultRoot: defaultObject,
                filled: &filled, ignored: &ignored)
            settings = try JSONDecoder().decode(
                LichessBotSettings.self, from: JSONSerialization.data(withJSONObject: merged))
        } catch let error as LichessBotSettingsStoreError {
            throw error
        } catch {
            throw LichessBotSettingsStoreError.unreadable(detail: String(describing: error))
        }
        let problems = settings.validationProblems()
        guard problems.isEmpty else {
            throw LichessBotSettingsStoreError.invalid(problems: problems)
        }
        return LoadResult(settings: settings, filledFromDefaults: filled.sorted(), ignoredSavedKeys: ignored.sorted())
    }

    /// `LichessBotSettings()` as a JSON object.
    private static func defaultSettingsObject() throws -> [String: Any] {
        try jsonObject(encoding: LichessBotSettings())
    }

    /// `value` encoded as a JSON object.
    static func jsonObject<Value: Encodable>(encoding value: Value) throws -> [String: Any] {
        let data = try JSONEncoder().encode(value)
        guard let object = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) as? [String: Any] else {
            throw LichessBotSettingsStoreError.defaultsNotAJSONObject(typeName: String(describing: Value.self))
        }
        return object
    }

    /// `saved` overlaid onto `defaults` (the default object at `path`).
    private static func overlay(
        saved: [String: Any], onto defaults: [String: Any], path: [String], defaultRoot: [String: Any],
        filled: inout [String], ignored: inout [String]
    ) throws -> [String: Any] {
        var result: [String: Any] = [:]
        for (key, savedValue) in saved {
            let keyPath = path + [key]
            guard let defaultValue = defaults[key] else {
                // Not in today's defaults: a removed or renamed field, or an
                // optional whose default is nil. Keep it — the decoder ignores
                // unknown keys and reads a present optional — and report it
                // only if the current type has no such field at all.
                result[key] = savedValue
                if try !isKnownOptionalKey(keyPath, defaultRoot: defaultRoot) {
                    ignored.append(keyPath.joined(separator: "."))
                }
                continue
            }
            // Merge nested objects key by key, but only when they share a key.
            // A struct and its older save always do; an enum with associated
            // values encodes as a one-key object named after its case, so a
            // saved case different from the default case shares none and must
            // be taken whole — merging would hand the decoder two cases. An
            // empty saved object is a struct whose fields were all nil
            // optionals (an enum case object is never empty), so it merges.
            if let savedObject = savedValue as? [String: Any], let defaultObject = defaultValue as? [String: Any],
               savedObject.isEmpty || !Set(savedObject.keys).isDisjoint(with: defaultObject.keys) {
                result[key] = try overlay(
                    saved: savedObject, onto: defaultObject, path: keyPath, defaultRoot: defaultRoot,
                    filled: &filled, ignored: &ignored)
            } else {
                result[key] = savedValue
            }
        }
        for (key, defaultValue) in defaults where saved[key] == nil {
            let keyPath = path + [key]
            if try isOptional(keyPath, defaultRoot: defaultRoot) {
                continue
            }
            result[key] = defaultValue
            filled.append(keyPath.joined(separator: "."))
        }
        return result
    }

    /// Whether the field at `keyPath` (present in the default object) is
    /// optional: the default object minus that key still decodes.
    private static func isOptional(_ keyPath: [String], defaultRoot: [String: Any]) throws -> Bool {
        let reduced = removing(keyPath, from: defaultRoot)
        let data = try JSONSerialization.data(withJSONObject: reduced)
        do {
            _ = try JSONDecoder().decode(LichessBotSettings.self, from: data)
            return true
        } catch DecodingError.keyNotFound {
            return false
        }
    }

    /// Whether `keyPath`, absent from the default object, is still a field of
    /// the current settings — an optional whose default is nil. Decoding the
    /// default object with a deliberately wrong-typed value at that path fails
    /// exactly when the decoder reads that key: with a type mismatch for a
    /// scalar, or, for a struct-typed optional, with a missing key inside the
    /// probe object — the decoder entered the probe, so the field exists. A
    /// missing key anywhere else is a real error and propagates.
    private static func isKnownOptionalKey(_ keyPath: [String], defaultRoot: [String: Any]) throws -> Bool {
        let probe = setting(keyPath, to: ["__probe__": true], in: defaultRoot)
        let data = try JSONSerialization.data(withJSONObject: probe)
        do {
            _ = try JSONDecoder().decode(LichessBotSettings.self, from: data)
            return false
        } catch DecodingError.typeMismatch {
            return true
        } catch DecodingError.dataCorrupted {
            return true
        } catch DecodingError.keyNotFound(_, let context) where context.codingPath.map(\.stringValue) == keyPath {
            return true
        }
    }

    private static func removing(_ keyPath: [String], from object: [String: Any]) -> [String: Any] {
        guard let first = keyPath.first else { return object }
        var copy = object
        if keyPath.count == 1 {
            copy.removeValue(forKey: first)
        } else if let child = object[first] as? [String: Any] {
            copy[first] = removing(Array(keyPath.dropFirst()), from: child)
        }
        return copy
    }

    private static func setting(_ keyPath: [String], to value: Any, in object: [String: Any]) -> [String: Any] {
        guard let first = keyPath.first else { return object }
        var copy = object
        if keyPath.count == 1 {
            copy[first] = value
        } else {
            let child = object[first] as? [String: Any] ?? [:]
            copy[first] = setting(Array(keyPath.dropFirst()), to: value, in: child)
        }
        return copy
    }

    /// Validate, then save. Invalid settings are rejected as a whole.
    static func save(_ settings: LichessBotSettings, to defaults: UserDefaults) throws {
        let problems = settings.validationProblems()
        guard problems.isEmpty else {
            throw LichessBotSettingsStoreError.invalid(problems: problems)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        defaults.set(try encoder.encode(settings), forKey: defaultsKey)
    }

    /// Replace unreadable saved settings with the defaults — the operator's
    /// explicit "Reset" in the error state.
    static func reset(in defaults: UserDefaults) throws {
        try save(LichessBotSettings(), to: defaults)
    }
}
