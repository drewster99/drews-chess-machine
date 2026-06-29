import AppKit
import Foundation

/// Single-flight guard + in-progress signal for the user-triggered Debug-menu
/// analyses (network weights, value head, replay buffer, run-all). Each of
/// those spawns a detached task and surfaces a completion alert; without a
/// shared running flag, nothing stopped a second press from stacking another
/// concurrent run (several of them read live network weights, so overlapping
/// runs during training are undesirable). `beginAnalysis` is the gate;
/// `endAnalysis` clears it from the detached task's `defer`.
extension SessionController {

    /// Begin a user-triggered analysis. Returns `false` (and shows a
    /// non-blocking "busy" alert) if another analysis is already running.
    /// On success, records the run and pushes an "Analyzing …" status message.
    @discardableResult
    func beginAnalysis(_ label: String) -> Bool {
        if let running = runningAnalysisLabel {
            NonBlockingAlert.presentInformational(
                title: "Analysis Already Running",
                message: "“\(running)” is still running — wait for it to finish before starting another analysis.",
                revealURL: nil
            )
            return false
        }
        runningAnalysisLabel = label
        checkpoint?.setCheckpointStatus("Analyzing \(label)…", kind: .progress)
        SessionLogger.shared.log("[ANALYSIS-GUARD] begin: \(label)")
        return true
    }

    /// End the current analysis so the next one can start. Idempotent. The
    /// "Analyzing…" status auto-clears on its own; each analysis's own
    /// completion alert is the success confirmation, so we don't post another.
    func endAnalysis() {
        guard let label = runningAnalysisLabel else { return }
        runningAnalysisLabel = nil
        SessionLogger.shared.log("[ANALYSIS-GUARD] end: \(label)")
    }
}
