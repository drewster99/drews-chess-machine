import SwiftUI

/// Transactional scratch state for `TrainingSettingsPopover`, lifted out of
/// `UpperContentView`.
///
/// Holds every editable field as a `String`/`Bool` (the raw control contents)
/// plus a matching `*Error` flag that drives the red invalid-input overlay.
/// Editing a field clears its own error via `didSet` (this replaces the ~19
/// per-field `.onChange` handlers that previously had to be split into an
/// `AnyView` chain to stay under the type-checker's per-expression budget).
/// `save()` parses every field, writes valid values back to
/// `TrainingParameters.shared` (logging each `[PARAM]` transition), mirrors the
/// optimizer-touching params onto the live `trainer`, pushes the freshly-edited
/// self-play τ-schedule into the live `samplingScheduleBox` (via the injected
/// `pushSelfPlaySchedule` closure), and dismisses the popover only if every
/// field parsed.
///
/// **Live-propagation exception for the replay-ratio fields.** The three
/// replay-ratio control fields (`selfPlayDelayMs`, `trainingStepDelayMs`,
/// `replayRatioAutoAdjust`) plus `replayRatioTarget` write through to
/// `TrainingParameters.shared` *immediately on change* via the `applyLive…`
/// methods rather than waiting for Save, so the user can watch the live ratio
/// display respond. On Cancel (or outside-click dismiss) those four are
/// reverted from the stash captured in `seedFromParams()`, matching the
/// "edit → cancel discards" mental model even though the live writes already
/// landed.
///
/// The live `trainer` / `ReplayRatioController` and the self-play-schedule push
/// are reached through injected closures so this model carries no dependency on
/// `UpperContentView`. The numeric clamp limits are injected as ints.
@MainActor
@Observable
final class TrainingSettingsPopoverModel {
    /// Drives the chip's popover presentation. Replaces the old
    /// `showTrainingPopover` `@State` on `UpperContentView`.
    var isPresented = false

    // MARK: - Optimizer tab

    var lrText = "" { didSet { lrError = false } }
    var warmupText = "" { didSet { warmupError = false } }
    var momentumText = "" { didSet { momentumError = false } }
    var sqrtBatchScalingValue = true
    var entropyText = "" { didSet { entropyError = false } }
    var illegalMassWeightText = "" { didSet { illegalMassWeightError = false } }
    var gradClipText = "" { didSet { gradClipError = false } }
    var weightDecayText = "" { didSet { weightDecayError = false } }
    var policyLossWeightText = "" { didSet { policyLossWeightError = false } }
    var valueLossWeightText = "" { didSet { valueLossWeightError = false } }
    var valueLabelSmoothingText = "" { didSet { valueLabelSmoothingError = false } }
    var drawPenaltyText = "" { didSet { drawPenaltyError = false } }
    var trainingBatchSizeText = "" { didSet { trainingBatchSizeError = false } }

    private(set) var lrError = false
    private(set) var warmupError = false
    private(set) var momentumError = false
    private(set) var entropyError = false
    private(set) var illegalMassWeightError = false
    private(set) var gradClipError = false
    private(set) var weightDecayError = false
    private(set) var policyLossWeightError = false
    private(set) var valueLossWeightError = false
    private(set) var valueLabelSmoothingError = false
    private(set) var drawPenaltyError = false
    private(set) var trainingBatchSizeError = false

    // MARK: - Self Play tab

    var selfPlayWorkersText = "" { didSet { selfPlayWorkersError = false } }
    var selfPlayStartTauText = "" { didSet { selfPlayStartTauError = false } }
    var selfPlayDecayPerPlyText = "" { didSet { selfPlayDecayPerPlyError = false } }
    var selfPlayFloorTauText = "" { didSet { selfPlayFloorTauError = false } }

    private(set) var selfPlayWorkersError = false
    private(set) var selfPlayStartTauError = false
    private(set) var selfPlayDecayPerPlyError = false
    private(set) var selfPlayFloorTauError = false

    // MARK: - Replay tab

    var replayBufferCapacityText = "" { didSet { replayBufferCapacityError = false } }
    var replayBufferMinPositionsText = "" { didSet { replayBufferMinPositionsError = false } }
    var replayRatioTargetText = "" { didSet { replayRatioTargetError = false } }
    var replaySelfPlayDelayText = "" { didSet { replaySelfPlayDelayError = false } }
    var replayTrainingStepDelayText = "" { didSet { replayTrainingStepDelayError = false } }
    var replayRatioAutoAdjust = true

    private(set) var replayBufferCapacityError = false
    private(set) var replayBufferMinPositionsError = false
    private(set) var replayRatioTargetError = false
    private(set) var replaySelfPlayDelayError = false
    private(set) var replayTrainingStepDelayError = false

    // MARK: - Cancel stash (for the live-propagated replay-ratio fields)

    private var originalReplayRatioTarget: Double = 1.0
    private var originalReplaySelfPlayDelayMs: Int = 0
    private var originalReplayTrainingStepDelayMs: Int = 0
    private var originalReplayRatioAutoAdjust: Bool = true

    // MARK: - Injected dependencies

    private let selfPlayDelayMaxMs: Int
    private let stepDelayMaxMs: Int
    private let maxSelfPlayWorkers: Int

    /// Returns the live trainer (if a session is running) so `save()` can
    /// mirror the optimizer-touching parameters onto it. Nil between sessions
    /// — those params then take effect at the next session start.
    var trainerProvider: () -> ChessTrainer? = { nil }
    /// Returns the live replay-ratio controller so the `applyLive…` delay
    /// methods can push their changes through immediately.
    var replayRatioControllerProvider: () -> ReplayRatioController? = { nil }
    /// Pushes the freshly-edited self-play τ-schedule into the live
    /// `samplingScheduleBox`. No-op before the first session.
    var pushSelfPlaySchedule: () -> Void = {}

    init(
        selfPlayDelayMaxMs: Int,
        stepDelayMaxMs: Int,
        maxSelfPlayWorkers: Int
    ) {
        self.selfPlayDelayMaxMs = selfPlayDelayMaxMs
        self.stepDelayMaxMs = stepDelayMaxMs
        self.maxSelfPlayWorkers = maxSelfPlayWorkers
        seedFromParams()
    }

    /// Re-sync the LR-warmup edit text from an external source — used by
    /// `ControlSideEffectsProbe` when a CLI / parameters-file override changes
    /// `trainingParams.lrWarmupSteps` behind the user's back so the popover (if
    /// opened) shows the new value rather than the stale pre-override one.
    func resyncLrWarmupText(_ s: String) {
        warmupText = s
    }

    // MARK: - Seed / cancel

    /// Seed the edit fields from the live `trainingParams` snapshot. Called
    /// when the popover opens so it always reflects current state, even if a
    /// CLI / parameters-file override changed something since the last open.
    func seedFromParams() {
        let p = TrainingParameters.shared
        // --- Optimizer tab ---
        lrText = String(format: "%.2e", p.learningRate)
        warmupText = String(p.lrWarmupSteps)
        momentumText = String(format: "%.3f", p.momentumCoeff)
        sqrtBatchScalingValue = p.sqrtBatchScalingLR
        entropyText = String(format: "%.2e", p.entropyBonus)
        illegalMassWeightText = String(format: "%.2f", p.illegalMassWeight)
        gradClipText = String(format: "%.1f", p.gradClipMaxNorm)
        weightDecayText = String(format: "%.2e", p.weightDecay)
        policyLossWeightText = String(format: "%.2f", p.policyLossWeight)
        valueLossWeightText = String(format: "%.2f", p.valueLossWeight)
        valueLabelSmoothingText = String(format: "%.3f", p.valueLabelSmoothingEpsilon)
        drawPenaltyText = String(format: "%.3f", p.drawPenalty)
        trainingBatchSizeText = String(p.trainingBatchSize)
        // --- Self Play tab ---
        selfPlayWorkersText = String(p.selfPlayWorkers)
        selfPlayStartTauText = String(format: "%.2f", p.selfPlayStartTau)
        selfPlayDecayPerPlyText = String(format: "%.3f", p.selfPlayTauDecayPerPly)
        selfPlayFloorTauText = String(format: "%.2f", p.selfPlayTargetTau)
        // --- Replay tab ---
        replayBufferCapacityText = String(p.replayBufferCapacity)
        replayBufferMinPositionsText = String(p.replayBufferMinPositionsBeforeTraining)
        replayRatioTargetText = String(format: "%.2f", p.replayRatioTarget)
        replaySelfPlayDelayText = String(p.selfPlayDelayMs)
        replayTrainingStepDelayText = String(p.trainingStepDelayMs)
        replayRatioAutoAdjust = p.replayRatioAutoAdjust
        // Stash pre-edit values for the four replay-ratio control fields. The
        // Replay tab live-propagates changes to those fields; if the user hits
        // Cancel we restore from this stash, matching the standard
        // "Cancel discards" mental model even though the live writes already
        // reached `trainingParams`.
        originalReplayRatioTarget = p.replayRatioTarget
        originalReplaySelfPlayDelayMs = p.selfPlayDelayMs
        originalReplayTrainingStepDelayMs = p.trainingStepDelayMs
        originalReplayRatioAutoAdjust = p.replayRatioAutoAdjust
        // Reset every error flag — a fresh open should never carry red overlays
        // from a previously-cancelled bad input.
        lrError = false
        warmupError = false
        momentumError = false
        entropyError = false
        illegalMassWeightError = false
        gradClipError = false
        weightDecayError = false
        policyLossWeightError = false
        valueLossWeightError = false
        valueLabelSmoothingError = false
        drawPenaltyError = false
        trainingBatchSizeError = false
        selfPlayWorkersError = false
        selfPlayStartTauError = false
        selfPlayDecayPerPlyError = false
        selfPlayFloorTauError = false
        replayBufferCapacityError = false
        replayBufferMinPositionsError = false
        replayRatioTargetError = false
        replaySelfPlayDelayError = false
        replayTrainingStepDelayError = false
    }

    /// Restore the four live-propagated replay-ratio control fields from the
    /// stash captured in `seedFromParams()`, then dismiss. Matches the
    /// user-facing "Cancel discards changes" pattern even though the underlying
    /// `trainingParams` writes already happened during the edit session. No
    /// `[PARAM]` log on revert (the live-update writes were not logged either —
    /// see `save()` for the commit-time logging). Idempotent: `save()` updates
    /// the stash before closing, so a Save → onDisappear sequence finds nothing
    /// to revert.
    func cancel() {
        let p = TrainingParameters.shared
        if abs(p.replayRatioTarget - originalReplayRatioTarget) > Double.ulpOfOne {
            p.replayRatioTarget = originalReplayRatioTarget
            // The `ControlSideEffectsProbe` watches `replayRatioTarget` and
            // pushes the new value into the live `ReplayRatioController`, so
            // this single write is sufficient — no direct controller call here.
        }
        if p.selfPlayDelayMs != originalReplaySelfPlayDelayMs {
            p.selfPlayDelayMs = originalReplaySelfPlayDelayMs
            replayRatioControllerProvider()?.manualSelfPlayDelayMs = originalReplaySelfPlayDelayMs
        }
        if p.trainingStepDelayMs != originalReplayTrainingStepDelayMs {
            p.trainingStepDelayMs = originalReplayTrainingStepDelayMs
            replayRatioControllerProvider()?.manualDelayMs = originalReplayTrainingStepDelayMs
        }
        if p.replayRatioAutoAdjust != originalReplayRatioAutoAdjust {
            p.replayRatioAutoAdjust = originalReplayRatioAutoAdjust
        }
        isPresented = false
    }

    // MARK: - Live-propagation (Replay tab)

    /// Live-propagate the replay-ratio-target edit straight to
    /// `trainingParams.replayRatioTarget`. The `ControlSideEffectsProbe`
    /// watches that property and forwards the new value into the live
    /// `ReplayRatioController.targetRatio`, so this single write suffices.
    /// Snapped to the parameter's `[0.1, 5.0]` range.
    func applyLiveReplayRatioTarget(_ newValue: Double) {
        guard newValue.isFinite else { return }
        let snapped = max(0.1, min(5.0, newValue))
        let p = TrainingParameters.shared
        if abs(p.replayRatioTarget - snapped) > Double.ulpOfOne {
            p.replayRatioTarget = snapped
        }
    }

    /// Live-propagate the self-play-delay edit to `trainingParams.selfPlayDelayMs`
    /// and the live `ReplayRatioController`.
    func applyLiveSelfPlayDelay(_ newValue: Int) {
        let snapped = max(0, min(selfPlayDelayMaxMs, newValue))
        let p = TrainingParameters.shared
        if p.selfPlayDelayMs != snapped {
            p.selfPlayDelayMs = snapped
            replayRatioControllerProvider()?.manualSelfPlayDelayMs = snapped
        }
    }

    /// Live-propagate the train-step-delay edit. Also writes through to
    /// `replayRatioController.manualDelayMs` because that's what
    /// `recordTrainingBatchAndGetDelay` reads each training step.
    func applyLiveTrainingStepDelay(_ newValue: Int) {
        let snapped = max(0, min(stepDelayMaxMs, newValue))
        let p = TrainingParameters.shared
        if p.trainingStepDelayMs != snapped {
            p.trainingStepDelayMs = snapped
            replayRatioControllerProvider()?.manualDelayMs = snapped
        }
    }

    /// Live-propagate the auto-control checkbox toggle. The
    /// `ControlSideEffectsProbe` watches `replayRatioAutoAdjust` and on the OFF
    /// transition writes inherited last-auto values into
    /// `trainingParams.trainingStepDelayMs` / `selfPlayDelayMs` — that runs
    /// after this setter returns. We defer a re-seed of the two delay text
    /// fields to the next main-actor tick so the editable rows that appear when
    /// auto goes OFF show the inherited values rather than the pre-toggle stash.
    func applyLiveReplayRatioAutoAdjust(_ newValue: Bool) {
        let p = TrainingParameters.shared
        if p.replayRatioAutoAdjust != newValue {
            p.replayRatioAutoAdjust = newValue
            if !newValue {
                Task { @MainActor in
                    let q = TrainingParameters.shared
                    self.replaySelfPlayDelayText = String(q.selfPlayDelayMs)
                    self.replayTrainingStepDelayText = String(q.trainingStepDelayMs)
                }
            }
        }
    }

    // MARK: - Save

    /// Validate every field against its parameter range and write valid values
    /// back to `trainingParams` (and mirror the optimizer-touching ones onto
    /// the live `trainer`). On any parse failure the affected field's
    /// red-overlay flag is set and the popover stays open. On full success the
    /// popover dismisses. `[PARAM] name: old -> new` log line on every actual
    /// change, no log when unchanged.
    func save() {
        let p = TrainingParameters.shared
        let trainer = trainerProvider()
        var anyError = false

        // LR — Double in [1e-7, 1.0].
        if let v = Double(lrText.trimmingCharacters(in: .whitespaces)),
           v >= 1e-7, v <= 1.0, v.isFinite {
            lrError = false
            if abs(v - p.learningRate) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] learningRate: %.3e -> %.3e", p.learningRate, v)
                )
                p.learningRate = v
                trainer?.learningRate = Float(v)
            }
        } else {
            lrError = true
            anyError = true
        }

        // LR Warmup steps — Int in [0, 100_000].
        if let n = Int(warmupText.trimmingCharacters(in: .whitespaces)),
           n >= 0, n <= 100_000 {
            warmupError = false
            if n != p.lrWarmupSteps {
                SessionLogger.shared.log("[PARAM] lrWarmupSteps: \(p.lrWarmupSteps) -> \(n)")
                p.lrWarmupSteps = n
                trainer?.lrWarmupSteps = n
            }
        } else {
            warmupError = true
            anyError = true
        }

        // Momentum — Double in [0, 0.99].
        if let v = Double(momentumText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 0.99, v.isFinite {
            momentumError = false
            if abs(v - p.momentumCoeff) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] momentumCoeff: %.3f -> %.3f", p.momentumCoeff, v)
                )
                p.momentumCoeff = v
                trainer?.momentumCoeff = Float(v)
            }
        } else {
            momentumError = true
            anyError = true
        }

        // √batch scaling toggle — Bool, cannot fail to parse.
        if sqrtBatchScalingValue != p.sqrtBatchScalingLR {
            SessionLogger.shared.log(
                "[PARAM] sqrtBatchScalingLR: \(p.sqrtBatchScalingLR) -> \(sqrtBatchScalingValue)"
            )
            p.sqrtBatchScalingLR = sqrtBatchScalingValue
        }

        // Entropy regularization — Double in [0, 0.1].
        if let v = Double(entropyText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 0.1, v.isFinite {
            entropyError = false
            if abs(v - p.entropyBonus) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] entropyBonus: %.3e -> %.3e", p.entropyBonus, v)
                )
                p.entropyBonus = v
                trainer?.entropyRegularizationCoeff = Float(v)
            }
        } else {
            entropyError = true
            anyError = true
        }

        // Illegal mass penalty — Double in [0, 100].
        if let v = Double(illegalMassWeightText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 100.0, v.isFinite {
            illegalMassWeightError = false
            if abs(v - p.illegalMassWeight) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] illegalMassWeight: %.2f -> %.2f", p.illegalMassWeight, v)
                )
                p.illegalMassWeight = v
                trainer?.illegalMassPenaltyWeight = Float(v)
            }
        } else {
            illegalMassWeightError = true
            anyError = true
        }

        // Grad clip — Double in [0.1, 1000].
        if let v = Double(gradClipText.trimmingCharacters(in: .whitespaces)),
           v >= 0.1, v <= 1000.0, v.isFinite {
            gradClipError = false
            if abs(v - p.gradClipMaxNorm) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] gradClipMaxNorm: %.2f -> %.2f", p.gradClipMaxNorm, v)
                )
                p.gradClipMaxNorm = v
                trainer?.gradClipMaxNorm = Float(v)
            }
        } else {
            gradClipError = true
            anyError = true
        }

        // Weight decay — Double in [0, 0.1].
        if let v = Double(weightDecayText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 0.1, v.isFinite {
            weightDecayError = false
            if abs(v - p.weightDecay) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] weightDecay: %.3e -> %.3e", p.weightDecay, v)
                )
                p.weightDecay = v
                trainer?.weightDecayC = Float(v)
            }
        } else {
            weightDecayError = true
            anyError = true
        }

        // Policy loss weight — Double in [0, 20].
        if let v = Double(policyLossWeightText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 20.0, v.isFinite {
            policyLossWeightError = false
            if abs(v - p.policyLossWeight) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] policyLossWeight: %.2f -> %.2f", p.policyLossWeight, v)
                )
                p.policyLossWeight = v
                trainer?.policyLossWeight = Float(v)
            }
        } else {
            policyLossWeightError = true
            anyError = true
        }

        // Value loss weight — Double in [0, 20].
        if let v = Double(valueLossWeightText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 20.0, v.isFinite {
            valueLossWeightError = false
            if abs(v - p.valueLossWeight) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] valueLossWeight: %.2f -> %.2f", p.valueLossWeight, v)
                )
                p.valueLossWeight = v
                trainer?.valueLossWeight = Float(v)
            }
        } else {
            valueLossWeightError = true
            anyError = true
        }

        // Value-head label smoothing ε — Double in [0, 0.5].
        if let v = Double(valueLabelSmoothingText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 0.5, v.isFinite {
            valueLabelSmoothingError = false
            if abs(v - p.valueLabelSmoothingEpsilon) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] valueLabelSmoothingEpsilon: %.3f -> %.3f", p.valueLabelSmoothingEpsilon, v)
                )
                p.valueLabelSmoothingEpsilon = v
                trainer?.valueLabelSmoothingEpsilon = Float(v)
            }
        } else {
            valueLabelSmoothingError = true
            anyError = true
        }

        // Draw penalty — Double in [-1, 1].
        if let v = Double(drawPenaltyText.trimmingCharacters(in: .whitespaces)),
           v >= -1.0, v <= 1.0, v.isFinite {
            drawPenaltyError = false
            if abs(v - p.drawPenalty) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] drawPenalty: %.3f -> %.3f", p.drawPenalty, v)
                )
                p.drawPenalty = v
                trainer?.drawPenalty = Float(v)
            }
        } else {
            drawPenaltyError = true
            anyError = true
        }

        // Training batch size — Int in [32, 32_768]. Snapshot-only; the live
        // trainer rebuilds its feed cache lazily on the next batch shape.
        if let n = Int(trainingBatchSizeText.trimmingCharacters(in: .whitespaces)),
           n >= 32, n <= 32_768 {
            trainingBatchSizeError = false
            if n != p.trainingBatchSize {
                SessionLogger.shared.log("[PARAM] trainingBatchSize: \(p.trainingBatchSize) -> \(n)")
                p.trainingBatchSize = n
            }
        } else {
            trainingBatchSizeError = true
            anyError = true
        }

        // Self-play workers — Int in [1, maxSelfPlayWorkers]. Live-tunable: the
        // BatchedSelfPlayDriver reconcile loop picks up the new count.
        if let n = Int(selfPlayWorkersText.trimmingCharacters(in: .whitespaces)),
           n >= 1, n <= maxSelfPlayWorkers {
            selfPlayWorkersError = false
            if n != p.selfPlayWorkers {
                SessionLogger.shared.log("[PARAM] selfPlayWorkers: \(p.selfPlayWorkers) -> \(n)")
                p.selfPlayWorkers = n
            }
        } else {
            selfPlayWorkersError = true
            anyError = true
        }

        // Self-play τ schedule — Doubles in [0.01, 5.0] (start / floor) and
        // [0.0, 1.0] (decay). Rebuilt by `buildSelfPlaySchedule()` next time
        // the schedule box is constructed; mid-session changes don't
        // retroactively alter games already in progress.
        if let v = Double(selfPlayStartTauText.trimmingCharacters(in: .whitespaces)),
           v >= 0.01, v <= 5.0, v.isFinite {
            selfPlayStartTauError = false
            if abs(v - p.selfPlayStartTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayStartTau: %.2f -> %.2f", p.selfPlayStartTau, v)
                )
                p.selfPlayStartTau = v
            }
        } else {
            selfPlayStartTauError = true
            anyError = true
        }
        if let v = Double(selfPlayDecayPerPlyText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 1.0, v.isFinite {
            selfPlayDecayPerPlyError = false
            if abs(v - p.selfPlayTauDecayPerPly) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayTauDecayPerPly: %.3f -> %.3f", p.selfPlayTauDecayPerPly, v)
                )
                p.selfPlayTauDecayPerPly = v
            }
        } else {
            selfPlayDecayPerPlyError = true
            anyError = true
        }
        if let v = Double(selfPlayFloorTauText.trimmingCharacters(in: .whitespaces)),
           v >= 0.01, v <= 5.0, v.isFinite {
            selfPlayFloorTauError = false
            if abs(v - p.selfPlayTargetTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayTargetTau: %.2f -> %.2f", p.selfPlayTargetTau, v)
                )
                p.selfPlayTargetTau = v
            }
        } else {
            selfPlayFloorTauError = true
            anyError = true
        }
        // Push the freshly-edited self-play schedule into the live
        // `samplingScheduleBox` so the next self-play game on each worker slot
        // picks up the new τ curve. Safe to call unconditionally — the box's
        // `setSelfPlay` is a no-op before the first session.
        pushSelfPlaySchedule()

        // Replay buffer capacity — Int in [1024, 100_000_000]. Snapshot-only:
        // the live ring cannot resize mid-session.
        if let n = Int(replayBufferCapacityText.trimmingCharacters(in: .whitespaces)),
           n >= 1024, n <= 100_000_000 {
            replayBufferCapacityError = false
            if n != p.replayBufferCapacity {
                SessionLogger.shared.log("[PARAM] replayBufferCapacity: \(p.replayBufferCapacity) -> \(n)")
                p.replayBufferCapacity = n
            }
        } else {
            replayBufferCapacityError = true
            anyError = true
        }

        // Pre-train fill threshold — Int in [0, 100_000_000]. Live-tunable.
        if let n = Int(replayBufferMinPositionsText.trimmingCharacters(in: .whitespaces)),
           n >= 0, n <= 100_000_000 {
            replayBufferMinPositionsError = false
            if n != p.replayBufferMinPositionsBeforeTraining {
                SessionLogger.shared.log(
                    "[PARAM] replayBufferMinPositionsBeforeTraining: \(p.replayBufferMinPositionsBeforeTraining) -> \(n)"
                )
                p.replayBufferMinPositionsBeforeTraining = n
            }
        } else {
            replayBufferMinPositionsError = true
            anyError = true
        }

        // Replay-ratio control fields are live-propagated during edits via
        // `applyLive…` — the writes already reached `trainingParams`. Save
        // validates the current text values for red-overlay display only; no
        // parameter writes here.
        if let v = Double(replayRatioTargetText.trimmingCharacters(in: .whitespaces)),
           v >= 0.1, v <= 5.0, v.isFinite {
            replayRatioTargetError = false
        } else {
            replayRatioTargetError = true
            anyError = true
        }
        if let n = Int(replaySelfPlayDelayText.trimmingCharacters(in: .whitespaces)),
           n >= 0, n <= selfPlayDelayMaxMs {
            replaySelfPlayDelayError = false
        } else {
            replaySelfPlayDelayError = true
            anyError = true
        }
        if let n = Int(replayTrainingStepDelayText.trimmingCharacters(in: .whitespaces)),
           n >= 0, n <= 10_000 {
            replayTrainingStepDelayError = false
        } else {
            replayTrainingStepDelayError = true
            anyError = true
        }

        if !anyError {
            // Commit-time [PARAM] log lines for the four live-propagated
            // replay-ratio fields. The live writes during the edit session are
            // intentionally silent (a log per keystroke would be noise); this
            // is the single authoritative log line per Save.
            if abs(p.replayRatioTarget - originalReplayRatioTarget) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] replayRatioTarget: %.2f -> %.2f", originalReplayRatioTarget, p.replayRatioTarget)
                )
            }
            if p.selfPlayDelayMs != originalReplaySelfPlayDelayMs {
                SessionLogger.shared.log(
                    "[PARAM] selfPlayDelayMs: \(originalReplaySelfPlayDelayMs) -> \(p.selfPlayDelayMs)"
                )
            }
            if p.trainingStepDelayMs != originalReplayTrainingStepDelayMs {
                SessionLogger.shared.log(
                    "[PARAM] trainingStepDelayMs: \(originalReplayTrainingStepDelayMs) -> \(p.trainingStepDelayMs)"
                )
            }
            if p.replayRatioAutoAdjust != originalReplayRatioAutoAdjust {
                SessionLogger.shared.log(
                    "[PARAM] replayRatioAutoAdjust: \(originalReplayRatioAutoAdjust) -> \(p.replayRatioAutoAdjust)"
                )
            }
            // On successful save the stash that backs Cancel becomes the new
            // pre-edit baseline — closing the popover with Save commits the
            // live ratio writes.
            originalReplayRatioTarget = p.replayRatioTarget
            originalReplaySelfPlayDelayMs = p.selfPlayDelayMs
            originalReplayTrainingStepDelayMs = p.trainingStepDelayMs
            originalReplayRatioAutoAdjust = p.replayRatioAutoAdjust
            isPresented = false
        }
    }
}
