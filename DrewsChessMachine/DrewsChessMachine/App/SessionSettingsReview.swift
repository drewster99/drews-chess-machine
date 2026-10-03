import Foundation

/// A session load waiting on the user: the session's `session.json` holds
/// settings that cannot be used as found, and nothing is resumed until the
/// user chooses. Carries everything needed to repeat the load with the
/// offered replacements accepted.
struct SessionSettingsReview: Identifiable, Sendable {
    let id = UUID()
    let sessionURL: URL
    let startAfterLoad: Bool
    let forceFloat32: Bool
    let findings: [InvalidStoredSetting]
}
