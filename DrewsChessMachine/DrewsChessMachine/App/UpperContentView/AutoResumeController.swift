import SwiftUI

/// Owns the launch-time auto-resume flow, lifted out of `UpperContentView`.
///
/// On app launch (`maybePresentSheet`), if a `LastSessionPointer` still names an
/// on-disk `.dcmsession`, the sheet is shown with a 30-second countdown; if the
/// user neither resumes nor dismisses, the countdown fires the resume
/// automatically. After dismiss, the File-menu "Resume training from autosave"
/// command (`resumeFromPointer`) covers the same flow for the rest of the
/// launch. The actual load-and-start chain lives on `UpperContentView`
/// (`loadSessionFrom(url:startAfterLoad:)`); the controller reaches it through
/// the injected `onResume` closure and is told the chain finished via
/// `markResumeFinished()`.
@MainActor
@Observable
final class AutoResumeController {

    /// Drives the sheet presentation. `var` (not `private(set)`) so it can back
    /// a SwiftUI `.sheet(isPresented:)` binding.
    var sheetShowing = false

    /// Pointer the sheet / menu is offering to resume. Captured when presented
    /// so the resume action uses the exact pointer the user saw.
    private(set) var pointer: LastSessionPointer?

    /// Lightweight peek of the target session's `session.json` (powers the rich
    /// sheet body); `nil` when the peek failed (sheet falls back to a minimal
    /// pointer-only layout).
    private(set) var summary: SessionResumeSummary?

    /// Seconds left on the countdown; ticks down once per second while the
    /// sheet is showing.
    private(set) var countdownRemaining = 0

    /// True while a resume load is in flight (disables the File-menu item and
    /// guards against a double resume). Cleared via `markResumeFinished()`.
    private(set) var inFlight = false

    private var countdownTask: Task<Void, Never>?

    /// Initial value of the auto-resume countdown in seconds.
    nonisolated static let countdownStartSec: Int = 30

    /// Wired by `UpperContentView` to chain into the load-and-start path
    /// (`loadSessionFrom(url: …, startAfterLoad: true)`). The controller logs
    /// the `[RESUME] Starting auto-resume of …` line before calling this.
    var onResume: (LastSessionPointer) -> Void = { _ in }

    /// True iff a last-saved-session pointer exists and still names an on-disk
    /// directory and no resume is in flight / no sheet is up. `UpperContentView`
    /// AND-s this with `!realTraining` for the File-menu item's enabled state.
    var canResume: Bool {
        !inFlight && !sheetShowing && (LastSessionPointer.read()?.directoryExists ?? false)
    }

    /// Composite scalar version of the two gating flags so a single `.onChange`
    /// on the view body can drive `syncMenuCommandHubState()` when either flips.
    var stateVersion: Int {
        (inFlight ? 1 : 0) | (sheetShowing ? 2 : 0)
    }

    /// Call when the load-and-start chain finishes (success or failure).
    func markResumeFinished() {
        inFlight = false
    }

    /// Present the launch-time sheet if a valid last-session pointer is on disk.
    /// A no-op under XCTest, if no pointer exists, if the target was deleted
    /// externally, or if training is already active (`isTrainingActive`, the
    /// live `realTraining` flag — should never trip at launch, but cheap).
    func maybePresentSheet(isTrainingActive: Bool) {
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
            SessionLogger.shared.log("[RESUME] Skipping auto-resume sheet — running under XCTest")
            return
        }
        guard !isTrainingActive, !sheetShowing else { return }
        guard let p = LastSessionPointer.read() else {
            SessionLogger.shared.log("[RESUME] No last-session pointer found — skipping auto-resume prompt")
            return
        }
        guard p.directoryExists else {
            SessionLogger.shared.log(
                "[RESUME] Last-session pointer names missing folder \(p.directoryPath) — clearing stale pointer"
            )
            LastSessionPointer.clear()
            return
        }
        pointer = p
        // Best-effort peek of the session's metadata file. A peek failure
        // leaves `summary` nil and the sheet falls back to a minimal layout —
        // we never suppress the prompt over a peek failure.
        do {
            summary = try CheckpointManager.peekSessionMetadata(at: p.directoryURL)
        } catch {
            summary = nil
            SessionLogger.shared.log(
                "[RESUME] session.json peek failed for \(p.sessionID): "
                + "\(error.localizedDescription) — sheet will use minimal layout"
            )
        }
        countdownRemaining = Self.countdownStartSec
        sheetShowing = true
        let savedAgo = max(0, Int(Date().timeIntervalSince1970) - Int(p.savedAtUnix))
        SessionLogger.shared.log(
            "[RESUME] Presenting auto-resume sheet for \(p.sessionID) (\(p.trigger), saved \(savedAgo)s ago)"
        )
        startCountdownTask()
    }

    private func startCountdownTask() {
        countdownTask?.cancel()
        countdownTask = Task { @MainActor in
            while sheetShowing && countdownRemaining > 0 {
                do {
                    try await Task.sleep(for: .seconds(1))
                } catch {
                    // Cancelled (user dismissed). Nothing to do.
                    return
                }
                if Task.isCancelled { return }
                countdownRemaining -= 1
            }
            // Only fire if the sheet is still up — a dismiss path may have
            // zeroed the counter on its way out.
            if sheetShowing {
                performResume()
            }
        }
    }

    /// Dismiss the sheet without resuming. The File-menu item stays available.
    func dismiss() {
        countdownTask?.cancel()
        countdownTask = nil
        sheetShowing = false
        summary = nil
        SessionLogger.shared.log("[RESUME] User dismissed auto-resume sheet")
    }

    /// Resume the session the sheet is currently offering (Resume button /
    /// countdown fire).
    func performResume() {
        guard let p = pointer else {
            dismiss()
            return
        }
        startResume(p)
    }

    /// Resume a specific pointer (File-menu "Resume training from autosave").
    /// `UpperContentView` owns the `!realTraining` / pointer-exists guards and
    /// the refuse-message UX; this just runs the resume.
    func resumeFromPointer(_ p: LastSessionPointer) {
        guard !inFlight else { return }
        startResume(p)
    }

    private func startResume(_ p: LastSessionPointer) {
        pointer = p
        countdownTask?.cancel()
        countdownTask = nil
        sheetShowing = false
        summary = nil
        inFlight = true
        SessionLogger.shared.log(
            "[RESUME] Starting auto-resume of \(p.sessionID) from \(p.directoryURL.lastPathComponent)"
        )
        onResume(p)
    }
}
