import Foundation

enum LichessBotSettingsStoreError: LocalizedError, Equatable {
    /// The stored settings exist but can't be decoded (plan §17.1: an
    /// error state, never a silent return to defaults).
    case unreadable(detail: String)
    case invalid(problems: [String])

    var errorDescription: String? {
        switch self {
        case .unreadable(let detail):
            return "Saved Lichess bot settings can't be read: \(detail). Reset them in Settings to continue."
        case .invalid(let problems):
            return "Lichess bot settings are invalid: " + problems.joined(separator: "; ")
        }
    }
}

/// Persists `LichessBotSettings` as one JSON blob in `UserDefaults` (plan
/// §12.1). Not a `TrainingParameters` entry, and never part of a
/// `.dcmsession`.
enum LichessBotSettingsStore {
    static let defaultsKey = "lichess_bot_settings"

    /// The saved settings. With nothing saved yet (first run), the defaults;
    /// saved settings that don't decode, or decode but fail validation,
    /// throw.
    static func load(from defaults: UserDefaults) throws -> LichessBotSettings {
        guard let data = defaults.data(forKey: defaultsKey) else {
            return LichessBotSettings()
        }
        let settings: LichessBotSettings
        do {
            settings = try JSONDecoder().decode(LichessBotSettings.self, from: data)
        } catch {
            throw LichessBotSettingsStoreError.unreadable(detail: String(describing: error))
        }
        let problems = settings.validationProblems()
        guard problems.isEmpty else {
            throw LichessBotSettingsStoreError.invalid(problems: problems)
        }
        return settings
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
