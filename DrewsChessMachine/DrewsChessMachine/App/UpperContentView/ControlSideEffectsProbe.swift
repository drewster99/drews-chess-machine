import SwiftUI

/// Hidden zero-sized view that funnels every "user adjusted a
/// control" side-effect chain into one place. Color.clear is the
/// SwiftUI carrier — the only point of the view is to host the
/// `.onChange` modifiers below, so the parent's `body` doesn't
/// devolve into a forest of side-effect modifiers attached to
/// arbitrary visible content.
///
/// Each `.onChange` reacts to one observable input, logs the
/// transition (so the session log captures every parameter edit),
/// and pushes the new value through to the live training-loop
/// objects (`workerCountBox`, `replayRatioController`, `trainer`)
/// as well as the `@Binding`-bound view state (e.g. flipping the
/// candidate-test mode also bumps the overlay so policy arrows
/// render immediately).
struct ControlSideEffectsProbe: View {
    @Binding var playAndTrainBoardMode: PlayAndTrainBoardMode
    @Binding var probeNetworkTarget: ProbeNetworkTarget
    @Binding var candidateProbeDirty: Bool
    @Binding var selectedOverlay: Int
    /// Re-sync the Training popover's LR-warmup edit text after a CLI /
    /// parameters-file override changes `trainingParams.lrWarmupSteps`
    /// behind the user's back (so the popover, if opened, shows the new
    /// value rather than the stale pre-override one). The popover's edit
    /// text now lives on `TrainingSettingsPopoverModel`, so this is a
    /// closure into the model rather than a direct `@Binding`.
    let resyncLrWarmupText: (String) -> Void
    @Binding var effectiveReplayRatioTarget: Double?
    @Binding var lastReplayRatioCompensatorAt: Date?
    @Bindable var trainingParams: TrainingParameters
    let workerCountBox: WorkerCountBox?
    let trainer: ChessTrainer?
    let replayRatioController: ReplayRatioController?
    /// Snap-to-rung helper for the shared discrete delay policy.
    /// Forwarded in so the probe doesn't have to duplicate the
    /// parent's valid-delay rung list.
    let snapDelayToNearestValidRung: (Int) -> Int

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onChange(of: playAndTrainBoardMode) { _, newValue in
                // Flipping to Candidate-test marks the probe dirty so
                // the driver fires an immediate forward pass on the
                // next gap — otherwise the user would wait up to 15s
                // for the interval probe to trigger on the new mode.
                if newValue == .candidateTest {
                    candidateProbeDirty = true
                }
            }
            .onChange(of: probeNetworkTarget) { _, _ in
                // Flipping the probe target marks the probe dirty so
                // the next driver gap fires a fresh evaluation against
                // the newly-selected network instead of showing stale
                // results.
                candidateProbeDirty = true
            }
            .onChange(of: trainingParams.selfPlayWorkers) { oldValue, newValue in
                // Stepper's `in:` range clamps; `.onChange` only fires
                // on real change, so log + box update happen only when
                // N actually shifts. `workerCountBox` is nil between
                // sessions, so out-of-session writes just update the
                // @State and take effect on the next session start.
                SessionLogger.shared.log("[PARAM] selfPlayWorkers: \(oldValue) -> \(newValue)")
                workerCountBox?.set(newValue)
                // Game-run mode is a placeholder card with N > 1 (the
                // live board hides itself behind a "N concurrent games"
                // overlay), and the Game-run / Candidate-test picker is
                // hidden in that case — so silently switch the mode to
                // Candidate test on the transition into multi-worker
                // mode. The user can switch back when they drop to N=1.
                if newValue > 1, oldValue <= 1, playAndTrainBoardMode == .gameRun {
                    playAndTrainBoardMode = .candidateTest
                }
            }
            .onChange(of: trainingParams.replayRatioTarget) { oldValue, newValue in
                SessionLogger.shared.log(
                    String(format: "[PARAM] replayRatioTarget: %.2f -> %.2f", oldValue, newValue)
                )
                replayRatioController?.targetRatio = newValue
                // Reset the outer integral compensator on user edit
                // — its accumulated drift was relative to the
                // previous user target and is no longer meaningful.
                // The next heartbeat tick will re-seed `T_eff` from
                // the new user value.
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
            }
            .onChange(of: trainingParams.sqrtBatchScalingLR) { oldValue, newValue in
                SessionLogger.shared.log("[PARAM] sqrtBatchScalingForLR: \(oldValue) -> \(newValue)")
                trainer?.sqrtBatchScalingForLR = newValue
            }
            .onChange(of: trainingParams.lrWarmupSteps) { oldValue, newValue in
                SessionLogger.shared.log("[PARAM] lrWarmupSteps: \(oldValue) -> \(newValue)")
                trainer?.lrWarmupSteps = newValue
                // Re-sync the popover's edit-text mirror so it reflects
                // CLI-driven overrides (applyCliConfigOverrides writes
                // `trainingParams` AFTER the popover model's last seed).
                // Without this, the training loop sees the correct value
                // but the popover (when opened) shows the stale one.
                resyncLrWarmupText(String(newValue))
            }
            .onChange(of: trainingParams.replayRatioAutoAdjust) { oldValue, newValue in
                SessionLogger.shared.log("[PARAM] replayRatioAutoAdjust: \(oldValue) -> \(newValue)")
                replayRatioController?.autoAdjust = newValue
                // Reset outer integral compensator on either
                // transition. Auto-OFF: there is no SP throttle to
                // compensate. Auto-ON: the inner controller restarts
                // from a different equilibrium and the previous
                // `T_eff` is stale.
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
                // Also push the user's target back to the controller
                // so the on-resume base point is the user's intent
                // rather than wherever the prior `T_eff` had drifted
                // to.
                replayRatioController?.targetRatio = trainingParams.replayRatioTarget
                if newValue {
                    // Auto ON: seed the computed delay from the current
                    // manual delay so the display doesn't jump.
                    replayRatioController?.computedDelayMs = trainingParams.trainingStepDelayMs
                } else {
                    // Auto OFF: inherit the last auto-computed delay
                    // as the new manual value, snapped to the nearest
                    // ladder rung so the Stepper binding doesn't crash
                    // on an off-ladder value.
                    let lastAuto = replayRatioController?.computedDelayMs ?? trainingParams.trainingStepDelayMs
                    let snapped = snapDelayToNearestValidRung(lastAuto)
                    trainingParams.trainingStepDelayMs = snapped
                    replayRatioController?.manualDelayMs = snapped
                    // Symmetric inherit on the SP side: pick up the
                    // last auto-computed SP per-game delay (the value
                    // the controller was about to apply at the moment
                    // of toggle), snap to the ladder, and write it
                    // through to both the training parameter (for
                    // persistence + Stepper display) and the
                    // controller's manual slot (so the worker sees the
                    // new value on its next game without waiting for a
                    // session restart). If the controller hadn't
                    // produced a non-zero auto sp delay yet (e.g.
                    // training was the slow side) we fall back to the
                    // user's prior `selfPlayDelayMs` setting rather
                    // than collapsing to 0.
                    let priorSP = trainingParams.selfPlayDelayMs
                    let lastAutoSP = replayRatioController?.smoothedSelfPlayDelayMs ?? priorSP
                    let snappedSP = snapDelayToNearestValidRung(lastAutoSP > 0 ? lastAutoSP : priorSP)
                    trainingParams.selfPlayDelayMs = snappedSP
                    replayRatioController?.manualSelfPlayDelayMs = snappedSP
                }
            }
    }
}
