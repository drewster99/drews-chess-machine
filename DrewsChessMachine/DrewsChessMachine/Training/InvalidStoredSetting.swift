import Foundation

/// A stored or saved setting that could not be used as found.
///
/// Settings come from two places that can hold a value the app cannot use: the
/// user's saved preferences (`UserDefaults`, one entry per training parameter)
/// and a session file being resumed (`session.json`). Either can be corrupt,
/// hand-edited, or written by a build with different ranges. The rule for both
/// is the same: nothing is silently replaced. Each unusable value is logged,
/// listed in the UI with what was found and why it cannot be used, and
/// replaced only when the user chooses the offered valid value.
public struct InvalidStoredSetting: Identifiable, Equatable, Sendable {
    /// Stable key: the parameter id for a stored preference, or the session
    /// field's id for a saved session value.
    public let id: String
    /// What the user knows the setting as.
    public let name: String
    /// The value as found, rendered as text.
    public let found: String
    /// Why it cannot be used.
    public let problem: String
    /// The valid value offered in its place, rendered as text.
    public let replacement: String

    public init(id: String, name: String, found: String, problem: String, replacement: String) {
        self.id = id
        self.name = name
        self.found = found
        self.problem = problem
        self.replacement = replacement
    }
}

extension ParameterValue {
    /// The value as it appears in `parameters.json`.
    var displayText: String {
        switch self {
        case .bool(let b): return b ? "true" : "false"
        case .int(let n): return String(n)
        case .double(let d): return "\(d)"
        }
    }
}
