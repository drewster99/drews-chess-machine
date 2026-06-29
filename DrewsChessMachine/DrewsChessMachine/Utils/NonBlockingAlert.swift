import AppKit
import Foundation

/// Presents `NSAlert`s **without blocking the main run loop**.
///
/// The hazard this exists to avoid: `NSAlert.runModal()` spins a nested
/// modal run loop that owns the main thread until the user dismisses the
/// panel. That starves Swift's MainActor job queue — and the self-play
/// and trainer loops hop to the MainActor every iteration to read
/// live-tunable parameters (`BatchedSelfPlayDriver`'s per-cycle
/// `TrainingParameters.shared` reads; the trainer loop's
/// `fireCandidateProbeIfNeeded()` / `TrainingParameters.shared.snapshot()`
/// calls in `SessionController+Training`). So a result dialog left open
/// silently halts ALL training for as long as the panel stays up. This
/// was observed in the field: a "Run All Analyses" result dialog left
/// open ~4 hours advanced the trainer by 11 steps total — effectively a
/// full stop — and resumed the instant it was dismissed.
///
/// `beginSheetModal(for:)` presents the same alert as a window-attached
/// sheet and returns immediately, so the run loop keeps servicing the
/// MainActor and the (intentionally decoupled) training pipeline never
/// stalls behind a piece of telemetry the user may have walked away
/// from.
@MainActor
enum NonBlockingAlert {

    /// Window a sheet hangs from. The app is single-window, so
    /// key → main → first window covers focus that currently lives on a
    /// popover or panel. Nil only before any window exists (early
    /// bring-up), which is never a moment any of these alerts fire from —
    /// they are all reachable solely through menu items and on-screen
    /// buttons, i.e. with the main window already up.
    private static func hostWindow() -> NSWindow? {
        NSApp.keyWindow ?? NSApp.mainWindow ?? NSApp.windows.first
    }

    /// Informational result alert with an OK button, plus a "Reveal in
    /// Finder" button when `revealURL` is non-nil (selects the file in
    /// Finder on click). Returns immediately; the run loop is never
    /// blocked.
    static func presentInformational(
        title: String,
        message: String,
        revealURL: URL? = nil
    ) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        if revealURL != nil {
            alert.addButton(withTitle: "Reveal in Finder")
        }
        present(alert) { response in
            if let url = revealURL, response == .alertSecondButtonReturn {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
        }
    }

    /// Warning alert with a single OK button. Returns immediately.
    static func presentWarning(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        present(alert) { _ in }
    }

    /// Attach `alert` as a non-blocking sheet on the host window and run
    /// `completion` with the user's response when it's dismissed.
    ///
    /// If no host window exists the sheet has nothing to attach to. We
    /// log that condition and present the alert window-less via
    /// `runModal()` as a genuine last resort — this path is only
    /// reachable before the GUI is up, when no training session can be
    /// running, so its blocking is harmless. It is deliberately the
    /// only `runModal()` left in the alert path; if it ever shows up in
    /// the log during a session, the window-resolution assumption above
    /// has been violated and should be revisited.
    private static func present(
        _ alert: NSAlert,
        completion: @escaping (NSApplication.ModalResponse) -> Void
    ) {
        guard let window = hostWindow() else {
            SessionLogger.shared.log(
                "[ALERT] no host window for sheet (\"\(alert.messageText)\"); "
                + "presenting window-less — training should not be running at this point"
            )
            completion(alert.runModal())
            return
        }
        alert.beginSheetModal(for: window, completionHandler: completion)
    }
}
