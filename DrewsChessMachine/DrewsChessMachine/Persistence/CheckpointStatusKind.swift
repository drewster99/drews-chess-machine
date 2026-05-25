import Foundation

/// Kind of ephemeral checkpoint status message shown in the status
/// row. Determines the leading icon (none / green check / red error
/// glyph), the text color, and the auto-clear lifetime.
///
/// Success messages get a green checkmark and a longer dwell time
/// so they're hard to miss — a durable confirmation of success
/// without resorting to a modal alert.
enum CheckpointStatusKind: Sendable {
    case progress
    /// Save has been running longer than the watchdog deadline.
    /// Visually distinct from `.progress` (amber tint, clock icon)
    /// so a stalled save catches the user's eye without being
    /// promoted to an outright error — the save may still complete
    /// successfully, it's just taking longer than expected.
    case slowProgress
    case success
    case error
}
