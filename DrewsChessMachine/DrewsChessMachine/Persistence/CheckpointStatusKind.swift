import Foundation

/// Kind of ephemeral checkpoint status message shown in the status
/// row. Determines the leading icon (none / green check / red error
/// glyph), the text color, and the auto-clear lifetime.
///
/// Success messages are visually distinct and linger longer because
/// the original flow — "Saving session (manual)…" then a same-styled
/// gray "Saved <filename>" that cleared after 6 seconds — was easy to
/// miss, leaving the user unsure whether the save had actually
/// completed. A green checkmark plus a longer dwell time gives a
/// durable confirmation of success without resorting to a modal
/// alert.
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
