import SwiftUI

/// Owns the in-app checkpoint subsystem, lifted out of `UpperContentView`.
///
/// **This is the first half of the extraction.** Stage 3c part 1 (this file)
/// hosts the checkpoint **status display** (`checkpointStatusMessage` /
/// `checkpointStatusKind` + the auto-clear timer + the `[CHECKPOINT-ERR]` echo)
/// and the **slow-save watchdog** that flips a still-running save to a
/// `.slowProgress` row after 10 s. The big save / load / segments / periodic-
/// save logic is still on `UpperContentView` and calls into this controller via
/// `checkpoint.setCheckpointStatus(_:kind:)` / `.startSlowSaveWatchdog(label:)`
/// / `.cancelSlowSaveWatchdog()` / `.checkpointSaveInFlight = …`. Part 2 of the
/// extraction will move the save/load methods into this same controller.
@MainActor
@Observable
final class CheckpointController {

    /// Status row driven by the save/load paths. Auto-clears after a
    /// kind-dependent lifetime via the `Task { … Task.sleep }` in `setCheckpointStatus`.
    private(set) var checkpointStatusMessage: String?
    private(set) var checkpointStatusKind: CheckpointStatusKind = .progress

    /// True while a save is in flight, so the slow-save watchdog can tell a
    /// completed-fast save from one that's actually stuck on disk. Set / cleared
    /// by the save paths (still on `UpperContentView`) at their entry / exit
    /// points. `var` (not `private(set)`) so those external sites can write it.
    var checkpointSaveInFlight: Bool = false

    /// Handle to the in-flight slow-save watchdog `Task`. `private` — the only
    /// way to start / stop one is through the `startSlowSaveWatchdog(label:)` /
    /// `cancelSlowSaveWatchdog()` methods, which keep the cancel + nil-out
    /// invariants together.
    private var slowSaveWatchdogTask: Task<Void, Never>?

    /// Slow-save watchdog deadline. If a save has not completed
    /// within this many seconds of starting, the status row flips to
    /// `.slowProgress` and a `[CHECKPOINT-WARN]` line is logged
    /// exactly once per save (no progressive warnings — completion
    /// will eventually flip the row to success/error and restore
    /// normal styling). Calibrated to the typical save cost: a
    /// healthy session save (two ~10 MB `.dcmmodel` files plus a 35
    /// MB replay buffer at 500k positions) takes well under a second
    /// on SSD; 10 s leaves headroom for the post-promotion path's
    /// `.utility`-priority detached task to be scheduled under load
    /// without firing false-positive warnings, while still surfacing
    /// genuinely stuck saves promptly.
    nonisolated static let slowSaveWatchdogSeconds: Int = 10

    /// Surface a status message on the checkpoint status row, auto-clearing
    /// after a kind-dependent lifetime so a transient save success line doesn't
    /// linger past usefulness. Errors are also echoed to the session log
    /// (`[CHECKPOINT-ERR]`) so a 12-s on-screen line that auto-clears is still
    /// recoverable from the persistent log file. Success lifetime is 20 s — long
    /// enough for the user to glance up and confirm the save actually landed —
    /// versus 6 s for progress lines and 12 s for errors.
    func setCheckpointStatus(_ message: String, kind: CheckpointStatusKind) {
        checkpointStatusMessage = message
        checkpointStatusKind = kind
        // Always echo errors to the session log so a transient on-screen
        // error message that auto-clears in 12 seconds is still
        // recoverable from the persistent log file. (Some callsites
        // also log their own more-detailed [CHECKPOINT] line — minor
        // duplication is fine; visibility is the priority.)
        if kind == .error {
            SessionLogger.shared.log("[CHECKPOINT-ERR] \(message)")
        }
        // Auto-clear after a kind-dependent lifetime. Grabs the
        // current message at schedule time so a later message isn't
        // wiped out by an earlier one's timer.
        let snapshotMessage = message
        let lifetimeSeconds: Int
        switch kind {
        case .progress: lifetimeSeconds = 6
        // Slow-save status persists noticeably longer than a normal
        // progress line — the user is presumably waiting on it, and a
        // 6-second auto-clear in the middle of a stuck save would just
        // leave them confused about whether anything is still happening.
        case .slowProgress: lifetimeSeconds = 120
        case .success: lifetimeSeconds = 20
        case .error: lifetimeSeconds = 12
        }
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(lifetimeSeconds))
            if self.checkpointStatusMessage == snapshotMessage {
                self.checkpointStatusMessage = nil
                self.checkpointStatusKind = .progress
            }
        }
    }

    /// Start a watchdog that warns the user if the save tagged
    /// `label` has not completed within `slowSaveWatchdogSeconds`.
    /// Every save path's completion branch must call `cancelSlowSaveWatchdog()`
    /// so a fast save's watchdog body never runs. Calling this while a previous
    /// watchdog is still pending cancels the previous one — only one save can
    /// be in flight at a time, and the most recent label is what should appear
    /// if it stalls.
    func startSlowSaveWatchdog(label: String) {
        slowSaveWatchdogTask?.cancel()
        let deadline = Self.slowSaveWatchdogSeconds
        slowSaveWatchdogTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(deadline))
            } catch {
                // The save completed before the deadline — its completion path
                // called `cancelSlowSaveWatchdog()`, which cancelled this Task.
                // `Task.sleep` throws `CancellationError`. Exit silently; the
                // fast-save case is the common one.
                return
            }
            if Task.isCancelled { return }
            // If the save already finished and emitted a final success/error
            // status, don't clobber it. We only flip to .slowProgress if the
            // row still shows the original "Saving…" line.
            guard self.checkpointSaveInFlight else { return }
            SessionLogger.shared.log(
                "[CHECKPOINT-WARN] \(label) still running after \(deadline)s — disk busy or replay buffer large?"
            )
            self.setCheckpointStatus(
                "Saving \(label)… (still running, \(deadline)s+)",
                kind: .slowProgress
            )
        }
    }

    /// Cancel the slow-save watchdog if any. Safe to call on any
    /// completion path — including success, error, and timeout
    /// branches that don't involve `slowSaveWatchdogTask` directly.
    func cancelSlowSaveWatchdog() {
        slowSaveWatchdogTask?.cancel()
        slowSaveWatchdogTask = nil
    }
}
