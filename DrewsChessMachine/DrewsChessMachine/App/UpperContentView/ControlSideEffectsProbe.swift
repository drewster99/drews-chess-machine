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
    @Binding var lrWarmupStepsEditText: String
    @Binding var effectiveReplayRatioTarget: Double?
    @Binding var lastReplayRatioCompensatorAt: Date?
    @Bindable var trainingParams: TrainingParameters
    let workerCountBox: WorkerCountBox?
    let trainer: ChessTrainer?
    let replayRatioController: ReplayRatioController?
    /// Snap-to-rung helper for the discrete training-step delay
    /// ladder. Forwarded in so the probe doesn't have to
    /// duplicate the parent's `stepDelayLadder` constant.
    let snapDelayToLadder: (Int) -> Int

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
                    // The default mini-board overlay is -1 (plain
                    // board) so right-arrow walks the user through
                    // Top Moves → channel views. But a freshly-entered
                    // Candidate-test mode is *expected* to show the
                    // policy arrows on the board — that's the whole
                    // point of the mode. Bump the overlay from the
                    // plain-board default to Top Moves (0) so the
                    // arrows render immediately. Leaves any deeper
                    // selection alone so a user who right-arrowed
                    // into channels and toggled CT off and on keeps
                    // their place.
                    if selectedOverlay < 0 {
                        selectedOverlay = 0
                    }
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
                // Re-sync the TextField's mirror state so the UI
                // reflects CLI-driven overrides (applyCliConfigOverrides
                // writes @AppStorage AFTER the view's onAppear has
                // already copied the pre-override value into the edit
                // text). Without this, the training loop sees the
                // correct value but the field shows the stale one.
                lrWarmupStepsEditText = String(newValue)
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
                    let snapped = snapDelayToLadder(lastAuto)
                    trainingParams.trainingStepDelayMs = snapped
                    replayRatioController?.manualDelayMs = snapped
                }
            }
    }
}
