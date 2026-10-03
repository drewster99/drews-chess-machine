import SwiftUI

/// Transactional scratch state for `TrainingSettingsPopover`, lifted out of
/// `UpperContentView`.
///
/// Holds every editable field as a `String`/`Bool` (the raw control contents)
/// plus a matching `*Error` flag that drives the red invalid-input overlay.
/// Editing a field clears its own error via `didSet`.
/// `save()` parses every field, writes valid values back to
/// `TrainingParameters.shared` (logging each `[PARAM]` transition), mirrors the
/// optimizer-touching params onto the live `trainer`, pushes the freshly-edited
/// self-play τ-schedule into the live `samplingScheduleBox` (via the injected
/// `pushSelfPlaySchedule` closure), and dismisses the popover only if every
/// field parsed.
///
/// **Live-propagation exception for selected Replay-tab fields.** Seven
/// fields write through to `TrainingParameters.shared` *immediately on
/// change* via the `applyLive…` methods rather than waiting for Save, so
/// the user can watch the live ratio display and the live sampling-
/// composition readout respond. On Cancel (or outside-click dismiss) all
/// seven are reverted from the stash captured in `seedFromParams()`,
/// matching the "edit → cancel discards" mental model even though the
/// live writes already landed:
///
///   - Replay-ratio control: `replayRatioTarget`, `selfPlayDelayMs`,
///     `trainingStepDelayMs`, `replayRatioAutoAdjust`.
///   - Sampling constraints: `maxPliesFromAnyOneGame`,
///     `targetSampledGameLengthPlies`, `maxDrawPercentPerBatch`.
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
    var signedAdvantageComplementCEValue = true
    var entropyText = "" { didSet { entropyError = false } }
    var illegalMassWeightText = "" { didSet { illegalMassWeightError = false } }
    var gradClipText = "" { didSet { gradClipError = false } }
    var weightDecayText = "" { didSet { weightDecayError = false } }
    var dropoutRateText = "" { didSet { dropoutRateError = false } }
    var policyLossWeightText = "" { didSet { policyLossWeightError = false } }
    var valueLossWeightText = "" { didSet { valueLossWeightError = false } }
    var valueLabelSmoothingText = "" { didSet { valueLabelSmoothingError = false } }
    /// Pending policy-smoothing mode. Decides which of the three policy
    /// smoothing fields below are shown as in effect and which `save()`
    /// validates and commits: ε in fixed-total mode, δ and the cap in
    /// per-move mode. The fields of the inactive mode are left untouched on
    /// Save, so an edit made to them before switching modes is discarded
    /// rather than committed unseen.
    var policyLabelSmoothingModeValue: PolicyLabelSmoothingMode = .fixedTotal
    var policyLabelSmoothingText = "" { didSet { policyLabelSmoothingError = false } }
    var policyLabelSmoothingPerMoveText = "" { didSet { policyLabelSmoothingPerMoveError = false } }
    var policyLabelSmoothingPerMoveCapText = "" { didSet { policyLabelSmoothingPerMoveCapError = false } }
    var drawPenaltyText = "" { didSet { drawPenaltyError = false } }
    var trainingBatchSizeText = "" { didSet { trainingBatchSizeError = false } }

    private(set) var lrError = false
    private(set) var warmupError = false
    private(set) var momentumError = false
    private(set) var entropyError = false
    private(set) var illegalMassWeightError = false
    private(set) var gradClipError = false
    private(set) var weightDecayError = false
    private(set) var dropoutRateError = false
    private(set) var policyLossWeightError = false
    private(set) var valueLossWeightError = false
    private(set) var valueLabelSmoothingError = false
    private(set) var policyLabelSmoothingError = false
    private(set) var policyLabelSmoothingPerMoveError = false
    private(set) var policyLabelSmoothingPerMoveCapError = false
    private(set) var drawPenaltyError = false
    private(set) var trainingBatchSizeError = false

    // MARK: - Cycling tab
    //
    // LR / momentum cycling (TRAINING_DYNAMICS_PLAN.md §3). Same edit-text +
    // validate-on-Save pattern as the Optimizer tab — these are NOT live-
    // propagated, so there is no Cancel stash for them (Cancel just re-seeds on
    // the next open). The two `…Enabled` toggles also gate the Optimizer tab's
    // LR / momentum text fields (disabled while the matching cycle is on); that
    // cross-tab disable reads these pending model values, so it responds the
    // instant the toggle flips, before Save.

    var lrCycleEnabledValue = false
    var lrCycleMinText = "" { didSet { lrCycleMinError = false } }
    var lrCycleMaxText = "" { didSet { lrCycleMaxError = false } }
    var lrCyclePeriodText = "" { didSet { lrCyclePeriodError = false } }
    var lrCycleCountText = "" { didSet { lrCycleCountError = false } }
    var lrCycleInvertValue = false
    var momentumCycleEnabledValue = false
    var momentumCycleMinText = "" { didSet { momentumCycleMinError = false } }
    var momentumCycleMaxText = "" { didSet { momentumCycleMaxError = false } }
    var momentumCyclePeriodText = "" { didSet { momentumCyclePeriodError = false } }
    var momentumCycleCountText = "" { didSet { momentumCycleCountError = false } }
    var momentumCycleInvertValue = true
    var lrCyclePeakEndText = "" { didSet { lrCyclePeakEndError = false } }
    var lrCycleTroughEndText = "" { didSet { lrCycleTroughEndError = false } }
    var lrCycleDecayHorizonText = "" { didSet { lrCycleDecayHorizonError = false } }
    var momentumFollowsLRCycleValue = true
    var momentumFollowStartLowText = "" { didSet { momentumFollowStartLowError = false } }
    var momentumFollowStartHighText = "" { didSet { momentumFollowStartHighError = false } }
    var momentumFollowEndLowText = "" { didSet { momentumFollowEndLowError = false } }
    var momentumFollowEndHighText = "" { didSet { momentumFollowEndHighError = false } }

    private(set) var lrCycleMinError = false
    private(set) var lrCycleMaxError = false
    private(set) var lrCyclePeriodError = false
    private(set) var lrCycleCountError = false
    private(set) var momentumCycleMinError = false
    private(set) var momentumCycleMaxError = false
    private(set) var momentumCyclePeriodError = false
    private(set) var momentumCycleCountError = false
    private(set) var lrCyclePeakEndError = false
    private(set) var lrCycleTroughEndError = false
    private(set) var lrCycleDecayHorizonError = false
    private(set) var momentumFollowStartLowError = false
    private(set) var momentumFollowStartHighError = false
    private(set) var momentumFollowEndLowError = false
    private(set) var momentumFollowEndHighError = false

    // MARK: - Self Play tab

    var selfPlayConcurrencyText = "" { didSet { selfPlayConcurrencyError = false } }
    var selfPlayStartTauText = "" { didSet { selfPlayStartTauError = false } }
    var selfPlayDecayPerPlyText = "" { didSet { selfPlayDecayPerPlyError = false } }
    var selfPlayFloorTauText = "" { didSet { selfPlayFloorTauError = false } }
    var selfPlayDrawKeepFractionText = "" { didSet { selfPlayDrawKeepFractionError = false } }
    var selfPlayMaxPliesPerGameText = "" { didSet { selfPlayMaxPliesPerGameError = false } }
    var drawWatchPDrawThresholdText = "" { didSet { drawWatchPDrawThresholdError = false } }
    var drawWatchTerminateGames: Bool = false
    var drawWatchStreakLengthText = "" { didSet { drawWatchStreakLengthError = false } }

    private(set) var selfPlayConcurrencyError = false
    private(set) var selfPlayStartTauError = false
    private(set) var selfPlayDecayPerPlyError = false
    private(set) var selfPlayFloorTauError = false
    private(set) var selfPlayDrawKeepFractionError = false
    private(set) var selfPlayMaxPliesPerGameError = false
    private(set) var drawWatchPDrawThresholdError = false
    private(set) var drawWatchStreakLengthError = false

    // MARK: - Replay tab

    var replayBufferCapacityText = "" { didSet { replayBufferCapacityError = false } }
    var replayBufferMinPositionsText = "" { didSet { replayBufferMinPositionsError = false } }
    var replayRatioTargetText = "" { didSet { replayRatioTargetError = false } }
    var replaySelfPlayDelayText = "" { didSet { replaySelfPlayDelayError = false } }
    var replayTrainingStepDelayText = "" { didSet { replayTrainingStepDelayError = false } }
    var replayRatioAutoAdjust = true
    var maxPliesFromAnyOneGameText = "" { didSet { maxPliesFromAnyOneGameError = false } }
    var targetSampledGameLengthPliesText = "" { didSet { targetSampledGameLengthPliesError = false } }
    var maxDrawPercentPerBatchText = "" { didSet { maxDrawPercentPerBatchError = false } }
    /// Bound to the Replay tab's "Stratify training batches by game phase"
    /// checkbox. Live-propagates to
    /// `TrainingParameters.shared.replayBufferStratifyByMaterial` and,
    /// via `ControlSideEffectsProbe`, to the live replay buffer's
    /// `SamplingConstraints.materialBucketWeights`. Cancel reverts from
    /// `originalReplayBufferStratifyByMaterial`.
    var replayBufferStratifyByMaterial = false

    private(set) var replayBufferCapacityError = false
    private(set) var replayBufferMinPositionsError = false
    private(set) var replayRatioTargetError = false
    private(set) var replaySelfPlayDelayError = false
    private(set) var replayTrainingStepDelayError = false
    private(set) var maxPliesFromAnyOneGameError = false
    private(set) var targetSampledGameLengthPliesError = false
    private(set) var maxDrawPercentPerBatchError = false

    // MARK: - Sessions tab
    //
    // Autosave-policy knobs. Not live-propagated like the Replay-tab fields:
    // the periodic-save interval is reconciled mid-session by the heartbeat
    // (which reads `periodicAutosaveIntervalSec` off the singleton and
    // re-anchors the running `PeriodicSaveController`), and the retention cap
    // is read at prune time after each periodic or post-promotion save — so a
    // plain commit-on-Save write to `TrainingParameters.shared` is all that's
    // needed, with no Cancel stash. The interval is edited in MINUTES for legibility (the parameter
    // stores seconds); seeding rounds to the nearest minute.
    var periodicAutosaveIntervalMinutesText = "" { didSet { periodicAutosaveIntervalError = false } }
    var maxPeriodicAutosavesKeptText = "" { didSet { maxPeriodicAutosavesKeptError = false } }
    /// `automatic_save_pruning_enabled`. Commit-on-Save like the rest of the
    /// tab: it is read live after each periodic or post-promotion save.
    var automaticSavePruningEnabledValue = false
    var klProbeIntervalText = "" { didSet { klProbeIntervalError = false } }

    private(set) var periodicAutosaveIntervalError = false
    private(set) var maxPeriodicAutosavesKeptError = false
    private(set) var klProbeIntervalError = false

    /// Pending `random_seed_mode`. Commit-on-Save; read only when a run
    /// starts (`RunRandomSeed.resolve`), so an edit made during a run takes
    /// effect at the next start, and a continue after Stop keeps its seed.
    /// The seed field is in effect only in `seeded` mode: in `unseeded` mode
    /// Save leaves `random_seed` untouched, like the inactive policy-smoothing
    /// fields, so an edit there is discarded rather than committed unseen.
    var randomSeedModeValue: RandomSeedMode = .unseeded
    var randomSeedText = "" { didSet { randomSeedError = false } }
    private(set) var randomSeedError = false

    /// Make the given run's seed the configured one: `seeded` mode with that
    /// seed in the field, committed by Save like any edit. Repeats an
    /// unseeded run without retyping its seed.
    func useRunSeed(_ runSeed: RunRandomSeed) {
        randomSeedModeValue = .seeded
        randomSeedText = String(runSeed.masterSeed)
    }

    /// The minutes ↔ seconds factor for the autosave-interval field, which
    /// is edited in minutes while `periodic_autosave_interval_sec` stores
    /// seconds.
    private static let secondsPerMinute: Double = 60

    /// Minutes shown in the autosave-interval field for a stored interval of
    /// `seconds`, rounded to the nearest minute. The one seconds → minutes
    /// conversion, shared by the seeded text, the placeholder and the
    /// stepper's fallback so the three cannot disagree.
    static func periodicAutosaveIntervalMinutes(fromSeconds seconds: Double) -> Int {
        Int((seconds / secondsPerMinute).rounded())
    }

    /// The declared default interval, in the field's minutes.
    static var periodicAutosaveIntervalDefaultMinutes: Int {
        periodicAutosaveIntervalMinutes(fromSeconds: PeriodicAutosaveIntervalSec.declaredDefault)
    }

    /// The declared interval range, in whole minutes that stay inside it
    /// (the lower bound rounds up, the upper bound rounds down), for the
    /// field's stepper.
    static var periodicAutosaveIntervalMinutesRange: ClosedRange<Int> {
        let seconds = PeriodicAutosaveIntervalSec.declaredClosedRange
        let lowestMinutes = Int((seconds.lowerBound / secondsPerMinute).rounded(.up))
        let highestMinutes = Int((seconds.upperBound / secondsPerMinute).rounded(.down))
        return lowestMinutes...highestMinutes
    }

    // MARK: - Cancel stash (for the live-propagated replay-ratio fields)

    private var originalReplayRatioTarget: Double = 1.0
    private var originalReplaySelfPlayDelayMs: Int = 0
    private var originalReplayTrainingStepDelayMs: Int = 0
    private var originalReplayRatioAutoAdjust: Bool = true

    // MARK: - Cancel stash (for the live-propagated sampling-constraint fields)

    private var originalMaxPliesFromAnyOneGame: Int = 10
    private var originalTargetSampledGameLengthPlies: Int = 0
    private var originalMaxDrawPercentPerBatch: Int = 100
    private var originalReplayBufferStratifyByMaterial: Bool = false

    // MARK: - Cancel stash (for the live-propagated self-play draw-keep field)

    private var originalSelfPlayDrawKeepFraction: Double = 1.0
    private var originalSelfPlayMaxPliesPerGame: Int = 150
    private var originalDrawWatchPDrawThreshold: Double = 0.95
    private var originalDrawWatchTerminateGames: Bool = false
    private var originalDrawWatchStreakLength: Int = 8

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

    /// The `Double` edit fields whose seeded text is a rounded rendering of
    /// the live value (`%.2e`, `%.3f`, …). Save compares each against what
    /// seeding wrote, so an untouched field never writes its rounded text
    /// back over a more precise value (a `parameters.json` δ of 1/300 shown
    /// as "0.0033" must stay 1/300), and an untouched field holding a session
    /// value outside today's declared range never blocks Save.
    private static let roundedTextFields: [ReferenceWritableKeyPath<TrainingSettingsPopoverModel, String>] = [
        \.lrText, \.momentumText,
        \.lrCycleMinText, \.lrCycleMaxText, \.momentumCycleMinText, \.momentumCycleMaxText,
        \.lrCyclePeakEndText, \.lrCycleTroughEndText,
        \.momentumFollowStartLowText, \.momentumFollowStartHighText,
        \.momentumFollowEndLowText, \.momentumFollowEndHighText,
        \.entropyText, \.illegalMassWeightText, \.gradClipText, \.weightDecayText,
        \.dropoutRateText, \.policyLossWeightText, \.valueLossWeightText,
        \.valueLabelSmoothingText, \.drawPenaltyText,
        \.policyLabelSmoothingText, \.policyLabelSmoothingPerMoveText, \.policyLabelSmoothingPerMoveCapText,
        \.selfPlayStartTauText, \.selfPlayDecayPerPlyText, \.selfPlayFloorTauText,
    ]

    /// Each `roundedTextFields` entry's text as `seedFromParams` wrote it.
    /// Empty until the first seed, in which case every field is parsed.
    @ObservationIgnored private var seededText: [ReferenceWritableKeyPath<TrainingSettingsPopoverModel, String>: String] = [:]

    /// What Save takes from a rounded `Double` field: the live value itself
    /// when the field still reads exactly what seeding wrote (nothing was
    /// edited, so nothing is written), otherwise the parsed text — nil when
    /// it is not a finite number inside the declared range.
    private func editedValue<K: TrainingParameterKey>(
        _ key: K.Type,
        _ field: ReferenceWritableKeyPath<TrainingSettingsPopoverModel, String>,
        current: Double
    ) -> Double? where K.Value == Double {
        if let seeded = seededText[field], seeded == self[keyPath: field] {
            return current
        }
        return K.parsedInDeclaredRange(self[keyPath: field])
    }

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
        signedAdvantageComplementCEValue = p.signedAdvantageComplementCE
        entropyText = String(format: "%.2e", p.entropyBonus)
        illegalMassWeightText = String(format: "%.2f", p.illegalMassWeight)
        gradClipText = String(format: "%.1f", p.gradClipMaxNorm)
        weightDecayText = String(format: "%.2e", p.weightDecay)
        dropoutRateText = String(format: "%.2f", p.dropoutRate)
        policyLossWeightText = String(format: "%.2f", p.policyLossWeight)
        valueLossWeightText = String(format: "%.2f", p.valueLossWeight)
        valueLabelSmoothingText = String(format: "%.3f", p.valueLabelSmoothingEpsilon)
        policyLabelSmoothingModeValue = p.policyLabelSmoothingMode
        policyLabelSmoothingText = String(format: "%.3f", p.policyLabelSmoothingEpsilon)
        policyLabelSmoothingPerMoveText = String(format: "%.4f", p.policyLabelSmoothingPerMove)
        policyLabelSmoothingPerMoveCapText = String(format: "%.2f", p.policyLabelSmoothingPerMoveCap)
        drawPenaltyText = String(format: "%.3f", p.drawPenalty)
        trainingBatchSizeText = String(p.trainingBatchSize)
        // --- Cycling tab ---
        lrCycleEnabledValue = p.lrCycleEnabled
        lrCycleMinText = String(format: "%.2e", p.lrCycleMin)
        lrCycleMaxText = String(format: "%.2e", p.lrCycleMax)
        lrCyclePeriodText = String(p.lrCyclePeriodSteps)
        lrCycleCountText = String(p.lrCycleCount)
        lrCycleInvertValue = p.lrCycleInvert
        momentumCycleEnabledValue = p.momentumCycleEnabled
        momentumCycleMinText = String(format: "%.2f", p.momentumCycleMin)
        momentumCycleMaxText = String(format: "%.2f", p.momentumCycleMax)
        momentumCyclePeriodText = String(p.momentumCyclePeriodSteps)
        momentumCycleCountText = String(p.momentumCycleCount)
        momentumCycleInvertValue = p.momentumCycleInvert
        lrCyclePeakEndText = String(format: "%.2e", p.lrCyclePeakEnd)
        lrCycleTroughEndText = String(format: "%.2e", p.lrCycleTroughEnd)
        lrCycleDecayHorizonText = String(p.lrCycleDecayHorizonSteps)
        momentumFollowsLRCycleValue = p.momentumFollowsLRCycle
        momentumFollowStartLowText = String(format: "%.3f", p.momentumFollowStartLow)
        momentumFollowStartHighText = String(format: "%.3f", p.momentumFollowStartHigh)
        momentumFollowEndLowText = String(format: "%.3f", p.momentumFollowEndLow)
        momentumFollowEndHighText = String(format: "%.3f", p.momentumFollowEndHigh)
        // --- Self Play tab ---
        selfPlayConcurrencyText = String(p.selfPlayConcurrency)
        selfPlayStartTauText = String(format: "%.2f", p.selfPlayStartTau)
        selfPlayDecayPerPlyText = String(format: "%.3f", p.selfPlayTauDecayPerPly)
        selfPlayFloorTauText = String(format: "%.2f", p.selfPlayTargetTau)
        selfPlayDrawKeepFractionText = String(format: "%.2f", p.selfPlayDrawKeepFraction)
        selfPlayMaxPliesPerGameText = String(p.selfPlayMaxPliesPerGame)
        drawWatchPDrawThresholdText = String(format: "%.3f", p.drawWatchPDrawThreshold)
        drawWatchTerminateGames = p.drawWatchTerminateGames
        drawWatchStreakLengthText = String(p.drawWatchStreakLength)
        // --- Replay tab ---
        replayBufferCapacityText = String(p.replayBufferCapacity)
        replayBufferMinPositionsText = String(p.replayBufferMinPositionsBeforeTraining)
        replayRatioTargetText = String(format: "%.2f", p.replayRatioTarget)
        replaySelfPlayDelayText = String(p.selfPlayDelayMs)
        replayTrainingStepDelayText = String(p.trainingStepDelayMs)
        replayRatioAutoAdjust = p.replayRatioAutoAdjust
        maxPliesFromAnyOneGameText = String(p.maxPliesFromAnyOneGame)
        targetSampledGameLengthPliesText = String(p.targetSampledGameLengthPlies)
        maxDrawPercentPerBatchText = String(p.maxDrawPercentPerBatch)
        replayBufferStratifyByMaterial = p.replayBufferStratifyByMaterial
        // --- Sessions tab ---
        periodicAutosaveIntervalMinutesText = String(
            Self.periodicAutosaveIntervalMinutes(fromSeconds: p.periodicAutosaveIntervalSec)
        )
        maxPeriodicAutosavesKeptText = String(p.maxPeriodicAutosavesKept)
        automaticSavePruningEnabledValue = p.automaticSavePruningEnabled
        klProbeIntervalText = String(p.klProbeInterval)
        randomSeedModeValue = p.randomSeedMode
        randomSeedText = String(p.randomSeed)
        // Stash pre-edit values for the four replay-ratio control fields. The
        // Replay tab live-propagates changes to those fields; if the user hits
        // Cancel we restore from this stash, matching the standard
        // "Cancel discards" mental model even though the live writes already
        // reached `trainingParams`.
        originalReplayRatioTarget = p.replayRatioTarget
        originalReplaySelfPlayDelayMs = p.selfPlayDelayMs
        originalReplayTrainingStepDelayMs = p.trainingStepDelayMs
        originalReplayRatioAutoAdjust = p.replayRatioAutoAdjust
        // Same stash-for-revert pattern for the three sampling-constraint
        // fields. The buffer's `Composition` readout updates live with the
        // current values so the operator can watch a stepper change and see
        // the realized batch composition shift in real time; if they Cancel
        // (or click outside the popover) we revert to these snapshots.
        originalMaxPliesFromAnyOneGame = p.maxPliesFromAnyOneGame
        originalTargetSampledGameLengthPlies = p.targetSampledGameLengthPlies
        originalMaxDrawPercentPerBatch = p.maxDrawPercentPerBatch
        originalReplayBufferStratifyByMaterial = p.replayBufferStratifyByMaterial
        originalSelfPlayDrawKeepFraction = p.selfPlayDrawKeepFraction
        originalSelfPlayMaxPliesPerGame = p.selfPlayMaxPliesPerGame
        originalDrawWatchPDrawThreshold = p.drawWatchPDrawThreshold
        originalDrawWatchTerminateGames = p.drawWatchTerminateGames
        originalDrawWatchStreakLength = p.drawWatchStreakLength
        seededText = Dictionary(uniqueKeysWithValues: Self.roundedTextFields.map { ($0, self[keyPath: $0]) })
        // Reset every error flag — a fresh open should never carry red overlays
        // from a previously-cancelled bad input.
        lrError = false
        warmupError = false
        momentumError = false
        entropyError = false
        illegalMassWeightError = false
        gradClipError = false
        weightDecayError = false
        dropoutRateError = false
        policyLossWeightError = false
        valueLossWeightError = false
        valueLabelSmoothingError = false
        policyLabelSmoothingError = false
        policyLabelSmoothingPerMoveError = false
        policyLabelSmoothingPerMoveCapError = false
        drawPenaltyError = false
        trainingBatchSizeError = false
        lrCycleMinError = false
        lrCycleMaxError = false
        lrCyclePeriodError = false
        lrCycleCountError = false
        momentumCycleMinError = false
        momentumCycleMaxError = false
        momentumCyclePeriodError = false
        momentumCycleCountError = false
        lrCyclePeakEndError = false
        lrCycleTroughEndError = false
        lrCycleDecayHorizonError = false
        momentumFollowStartLowError = false
        momentumFollowStartHighError = false
        momentumFollowEndLowError = false
        momentumFollowEndHighError = false
        selfPlayConcurrencyError = false
        selfPlayStartTauError = false
        selfPlayDecayPerPlyError = false
        selfPlayFloorTauError = false
        selfPlayDrawKeepFractionError = false
        selfPlayMaxPliesPerGameError = false
        drawWatchPDrawThresholdError = false
        drawWatchStreakLengthError = false
        replayBufferCapacityError = false
        replayBufferMinPositionsError = false
        replayRatioTargetError = false
        replaySelfPlayDelayError = false
        replayTrainingStepDelayError = false
        maxPliesFromAnyOneGameError = false
        targetSampledGameLengthPliesError = false
        maxDrawPercentPerBatchError = false
        periodicAutosaveIntervalError = false
        maxPeriodicAutosavesKeptError = false
        klProbeIntervalError = false
        randomSeedError = false
    }

    /// Restore the seven live-propagated Replay-tab fields (four replay-ratio
    /// control + three sampling constraints) from the stash captured in
    /// `seedFromParams()`, then dismiss. Matches the user-facing "Cancel
    /// discards changes" pattern even though the underlying `trainingParams`
    /// writes already happened during the edit session. No `[PARAM]` log on
    /// revert (the live-update writes were not logged either — see `save()`
    /// for the commit-time logging). Idempotent: `save()` updates the stash
    /// before closing, so a Save → onDisappear sequence finds nothing to
    /// revert.
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
        // Sampling-constraint fields: same revert pattern.
        // `ControlSideEffectsProbe`'s `.onChange` handlers push the
        // restored value into `ReplayBuffer.setSamplingConstraints`
        // reactively, so no direct buffer call is needed here.
        if p.maxPliesFromAnyOneGame != originalMaxPliesFromAnyOneGame {
            p.maxPliesFromAnyOneGame = originalMaxPliesFromAnyOneGame
        }
        if p.targetSampledGameLengthPlies != originalTargetSampledGameLengthPlies {
            p.targetSampledGameLengthPlies = originalTargetSampledGameLengthPlies
        }
        if p.maxDrawPercentPerBatch != originalMaxDrawPercentPerBatch {
            p.maxDrawPercentPerBatch = originalMaxDrawPercentPerBatch
        }
        if p.replayBufferStratifyByMaterial != originalReplayBufferStratifyByMaterial {
            p.replayBufferStratifyByMaterial = originalReplayBufferStratifyByMaterial
        }
        if abs(p.selfPlayDrawKeepFraction - originalSelfPlayDrawKeepFraction) > Double.ulpOfOne {
            p.selfPlayDrawKeepFraction = originalSelfPlayDrawKeepFraction
        }
        if p.selfPlayMaxPliesPerGame != originalSelfPlayMaxPliesPerGame {
            p.selfPlayMaxPliesPerGame = originalSelfPlayMaxPliesPerGame
        }
        if abs(p.drawWatchPDrawThreshold - originalDrawWatchPDrawThreshold) > Double.ulpOfOne {
            p.drawWatchPDrawThreshold = originalDrawWatchPDrawThreshold
        }
        if p.drawWatchTerminateGames != originalDrawWatchTerminateGames {
            p.drawWatchTerminateGames = originalDrawWatchTerminateGames
        }
        if p.drawWatchStreakLength != originalDrawWatchStreakLength {
            p.drawWatchStreakLength = originalDrawWatchStreakLength
        }
        isPresented = false
    }

    // MARK: - Live-propagation (Replay tab)

    /// Live-propagate the replay-ratio-target edit straight to
    /// `trainingParams.replayRatioTarget`. The `ControlSideEffectsProbe`
    /// watches that property and forwards the new value into the live
    /// `ReplayRatioController.targetRatio`, so this single write suffices.
    /// Snapped to the parameter's declared range.
    func applyLiveReplayRatioTarget(_ newValue: Double) {
        guard newValue.isFinite else { return }
        let snapped = ReplayRatioTarget.snappedToDeclaredRange(newValue)
        let p = TrainingParameters.shared
        if abs(p.replayRatioTarget - snapped) > Double.ulpOfOne {
            p.replayRatioTarget = snapped
        }
    }

    /// Live-propagate the self-play-delay edit to `trainingParams.selfPlayDelayMs`
    /// and the live `ReplayRatioController`.
    func applyLiveSelfPlayDelay(_ newValue: Int) {
        let snapped = min(selfPlayDelayMaxMs, SelfPlayDelayMs.snappedToDeclaredRange(newValue))
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
        let snapped = min(stepDelayMaxMs, TrainingStepDelayMs.snappedToDeclaredRange(newValue))
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

    /// Live-propagate the max-plies-per-game edit to
    /// `trainingParams.maxPliesFromAnyOneGame`.
    /// `ControlSideEffectsProbe`'s `.onChange(of: maxPliesFromAnyOneGame)`
    /// handler pushes the new value into
    /// `ReplayBuffer.setSamplingConstraints(.fromCurrentParameters())`
    /// reactively, so this single write is sufficient — the next
    /// training batch picks up the new K cap and the popover's
    /// Composition readout reflects the change on the next heartbeat.
    /// Snapped to the parameter's declared range.
    func applyLiveMaxPliesFromAnyOneGame(_ newValue: Int) {
        let snapped = MaxPliesFromAnyOneGame.snappedToDeclaredRange(newValue)
        let p = TrainingParameters.shared
        if p.maxPliesFromAnyOneGame != snapped {
            p.maxPliesFromAnyOneGame = snapped
        }
    }

    /// Live-propagate the target-sampled-game-length edit. Snapped to
    /// the parameter's declared range; zero disables the length tilt.
    func applyLiveTargetSampledGameLengthPlies(_ newValue: Int) {
        let snapped = TargetSampledGameLengthPlies.snappedToDeclaredRange(newValue)
        let p = TrainingParameters.shared
        if p.targetSampledGameLengthPlies != snapped {
            p.targetSampledGameLengthPlies = snapped
        }
    }

    /// Live-propagate the max-draw-percent-per-batch edit. Snapped to
    /// the parameter's declared range; its maximum disables the draw cap.
    func applyLiveMaxDrawPercentPerBatch(_ newValue: Int) {
        let snapped = MaxDrawPercentPerBatch.snappedToDeclaredRange(newValue)
        let p = TrainingParameters.shared
        if p.maxDrawPercentPerBatch != snapped {
            p.maxDrawPercentPerBatch = snapped
        }
    }

    /// Live-propagate the "Stratify training batches by game phase"
    /// checkbox edit straight to
    /// `TrainingParameters.shared.replayBufferStratifyByMaterial`.
    /// `ControlSideEffectsProbe`'s `.onChange(of: replayBufferStratifyByMaterial)`
    /// handler pushes the new value into
    /// `ReplayBuffer.setSamplingConstraints(.fromCurrentParameters())`
    /// reactively, so a single write here is enough to flip the
    /// running trainer's sampling path on the next minibatch.
    func applyLiveReplayBufferStratifyByMaterial(_ newValue: Bool) {
        let p = TrainingParameters.shared
        if p.replayBufferStratifyByMaterial != newValue {
            p.replayBufferStratifyByMaterial = newValue
        }
    }

    /// Live-propagate the self-play draw-keep-fraction edit straight
    /// to `trainingParams.selfPlayDrawKeepFraction`. The slot driver
    /// reads `TrainingParameters.shared.selfPlayDrawKeepFraction`
    /// at the end of every self-play game, so a mid-session edit
    /// takes effect on the next completed game on every worker slot
    /// without further plumbing. Snapped to the parameter's
    /// declared range; non-finite inputs are ignored.
    func applyLiveSelfPlayDrawKeepFraction(_ newValue: Double) {
        guard newValue.isFinite else { return }
        let snapped = SelfPlayDrawKeepFraction.snappedToDeclaredRange(newValue)
        let p = TrainingParameters.shared
        if abs(p.selfPlayDrawKeepFraction - snapped) > Double.ulpOfOne {
            p.selfPlayDrawKeepFraction = snapped
        }
    }

    /// Live-propagate the max-plies-per-game edit. The self-play
    /// driver reads `TrainingParameters.shared.selfPlayMaxPliesPerGame` at the
    /// start of every game, so a mid-session edit takes effect on the
    /// next game spawned by each worker slot. Snapped to the
    /// parameter's declared range.
    func applyLiveMaxPliesPerGame(_ newValue: Int) {
        let snapped = SelfPlayMaxPliesPerGame.snappedToDeclaredRange(newValue)
        let p = TrainingParameters.shared
        if p.selfPlayMaxPliesPerGame != snapped {
            p.selfPlayMaxPliesPerGame = snapped
        }
    }

    /// Live-propagate the draw-watch pDraw threshold edit. The
    /// self-play driver reads `TrainingParameters.shared.drawWatchPDrawThreshold`
    /// at the start of every tick (one MainActor hop per tick),
    /// so a mid-session edit takes effect on the next ply. Snapped
    /// to the parameter's declared range; non-finite inputs are
    /// ignored.
    func applyLiveDrawWatchPDrawThreshold(_ newValue: Double) {
        guard newValue.isFinite else { return }
        let snapped = DrawWatchPDrawThreshold.snappedToDeclaredRange(newValue)
        let p = TrainingParameters.shared
        if abs(p.drawWatchPDrawThreshold - snapped) > Double.ulpOfOne {
            p.drawWatchPDrawThreshold = snapped
        }
    }

    /// Live-propagate the draw-watch terminate-games toggle. The
    /// self-play driver reads `TrainingParameters.shared.drawWatchTerminateGames`
    /// alongside the threshold on every tick, so a toggle flip
    /// takes effect on the next ply. ON: games are dropped the
    /// instant their N-ply pDraw streak completes (same drop path
    /// as ply-cap). OFF (default): purely observational.
    func applyLiveDrawWatchTerminateGames(_ newValue: Bool) {
        let p = TrainingParameters.shared
        if p.drawWatchTerminateGames != newValue {
            p.drawWatchTerminateGames = newValue
        }
    }

    /// Live-propagate the draw-watch streak-length edit. The
    /// self-play driver reads `TrainingParameters.shared.drawWatchStreakLength`
    /// every tick. Snapped to the parameter's declared range.
    func applyLiveDrawWatchStreakLength(_ newValue: Int) {
        let snapped = DrawWatchStreakLength.snappedToDeclaredRange(newValue)
        let p = TrainingParameters.shared
        if p.drawWatchStreakLength != snapped {
            p.drawWatchStreakLength = snapped
        }
    }

    /// Write the pending policy-smoothing mode to `p`, logging the change.
    /// Called by `save()` only after the fields that mode reads validated.
    private func commitPolicyLabelSmoothingMode(to p: TrainingParameters) {
        guard policyLabelSmoothingModeValue != p.policyLabelSmoothingMode else { return }
        SessionLogger.shared.log(
            "[PARAM] policyLabelSmoothingMode: \(p.policyLabelSmoothingMode.logToken) -> \(policyLabelSmoothingModeValue.logToken)"
        )
        p.policyLabelSmoothingMode = policyLabelSmoothingModeValue
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

        // LR — Double in the declared range.
        if let v = editedValue(LearningRate.self, \.lrText, current: p.learningRate) {
            lrError = false
            if abs(v - p.learningRate) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] learningRate: %.6g -> %.6g", p.learningRate, v)
                )
                p.learningRate = v
            }
        } else {
            lrError = true
            anyError = true
        }

        // LR Warmup steps — Int in the declared range.
        if let n = LRWarmupSteps.parsedInDeclaredRange(warmupText) {
            warmupError = false
            if n != p.lrWarmupSteps {
                SessionLogger.shared.log("[PARAM] lrWarmupSteps: \(p.lrWarmupSteps) -> \(n)")
                p.lrWarmupSteps = n
            }
        } else {
            warmupError = true
            anyError = true
        }

        // Momentum — Double in the declared range.
        if let v = editedValue(MomentumCoeff.self, \.momentumText, current: p.momentumCoeff) {
            momentumError = false
            if abs(v - p.momentumCoeff) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] momentumCoeff: %.6g -> %.6g", p.momentumCoeff, v)
                )
                p.momentumCoeff = v
            }
        } else {
            momentumError = true
            anyError = true
        }

        // --- Cycling tab ---
        // LR / momentum cycling params. Validated against the same ranges as
        // the underlying @TrainingParameters and written to the singleton here.
        // The per-change [PARAM] audit line AND the push onto the live trainer
        // are both handled once, as a bundle, by `ControlSideEffectsProbe`'s
        // `.onChange(of: trainingParams.lrMomentumCycle)` forwarder (which fires
        // when any of these writes lands) — the same "write singleton → probe
        // reacts" split the Replay tab's sampling constraints use, avoiding a
        // 12-line-per-Save log spew.
        if lrCycleEnabledValue != p.lrCycleEnabled { p.lrCycleEnabled = lrCycleEnabledValue }
        if lrCycleInvertValue != p.lrCycleInvert { p.lrCycleInvert = lrCycleInvertValue }
        if let v = editedValue(LRCycleMin.self, \.lrCycleMinText, current: p.lrCycleMin) {
            lrCycleMinError = false
            if abs(v - p.lrCycleMin) > Double.ulpOfOne { p.lrCycleMin = v }
        } else {
            lrCycleMinError = true
            anyError = true
        }
        if let v = editedValue(LRCycleMax.self, \.lrCycleMaxText, current: p.lrCycleMax) {
            lrCycleMaxError = false
            if abs(v - p.lrCycleMax) > Double.ulpOfOne { p.lrCycleMax = v }
        } else {
            lrCycleMaxError = true
            anyError = true
        }
        // Cross-field: LR max must be >= min. The cycle guards itself
        // (`learningRate(forStep:)` returns nil and falls back to the static
        // LR when lrMax < lrMin), so an inverted pair would silently make the
        // cycle inert while the UI reads "enabled" — flag the max field so the
        // misconfiguration is visible instead.
        if !lrCycleMinError, !lrCycleMaxError, p.lrCycleMax < p.lrCycleMin {
            lrCycleMaxError = true
            anyError = true
        }
        if let n = LRCyclePeriodSteps.parsedInDeclaredRange(lrCyclePeriodText) {
            lrCyclePeriodError = false
            if n != p.lrCyclePeriodSteps { p.lrCyclePeriodSteps = n }
        } else {
            lrCyclePeriodError = true
            anyError = true
        }
        if let n = LRCycleCount.parsedInDeclaredRange(lrCycleCountText) {
            lrCycleCountError = false
            if n != p.lrCycleCount { p.lrCycleCount = n }
        } else {
            lrCycleCountError = true
            anyError = true
        }
        if momentumCycleEnabledValue != p.momentumCycleEnabled { p.momentumCycleEnabled = momentumCycleEnabledValue }
        if momentumCycleInvertValue != p.momentumCycleInvert { p.momentumCycleInvert = momentumCycleInvertValue }
        if let v = editedValue(MomentumCycleMin.self, \.momentumCycleMinText, current: p.momentumCycleMin) {
            momentumCycleMinError = false
            if abs(v - p.momentumCycleMin) > Double.ulpOfOne { p.momentumCycleMin = v }
        } else {
            momentumCycleMinError = true
            anyError = true
        }
        if let v = editedValue(MomentumCycleMax.self, \.momentumCycleMaxText, current: p.momentumCycleMax) {
            momentumCycleMaxError = false
            if abs(v - p.momentumCycleMax) > Double.ulpOfOne { p.momentumCycleMax = v }
        } else {
            momentumCycleMaxError = true
            anyError = true
        }
        // Cross-field: momentum max must be >= min. Unlike LR, `momentum(forStep:)`
        // has NO min<=max guard — an inverted pair would silently run the
        // schedule reversed with no signal — so flagging it here is the only
        // safeguard against a silent misconfiguration.
        if !momentumCycleMinError, !momentumCycleMaxError, p.momentumCycleMax < p.momentumCycleMin {
            momentumCycleMaxError = true
            anyError = true
        }
        if let n = MomentumCyclePeriodSteps.parsedInDeclaredRange(momentumCyclePeriodText) {
            momentumCyclePeriodError = false
            if n != p.momentumCyclePeriodSteps { p.momentumCyclePeriodSteps = n }
        } else {
            momentumCyclePeriodError = true
            anyError = true
        }
        if let n = MomentumCycleCount.parsedInDeclaredRange(momentumCycleCountText) {
            momentumCycleCountError = false
            if n != p.momentumCycleCount { p.momentumCycleCount = n }
        } else {
            momentumCycleCountError = true
            anyError = true
        }
        // Decay envelope. Ranges mirror the `@TrainingParameter` declarations.
        if let v = editedValue(LRCyclePeakEnd.self, \.lrCyclePeakEndText, current: p.lrCyclePeakEnd) {
            lrCyclePeakEndError = false
            if abs(v - p.lrCyclePeakEnd) > Double.ulpOfOne { p.lrCyclePeakEnd = v }
        } else {
            lrCyclePeakEndError = true
            anyError = true
        }
        if let v = editedValue(LRCycleTroughEnd.self, \.lrCycleTroughEndText, current: p.lrCycleTroughEnd) {
            lrCycleTroughEndError = false
            if abs(v - p.lrCycleTroughEnd) > Double.ulpOfOne { p.lrCycleTroughEnd = v }
        } else {
            lrCycleTroughEndError = true
            anyError = true
        }
        // Cross-field: the end peak must not sit below the end trough. The
        // schedule refuses such an envelope (falls back to the static LR), so
        // without this flag the cycle would go inert while reading "enabled".
        if !lrCyclePeakEndError, !lrCycleTroughEndError, p.lrCyclePeakEnd < p.lrCycleTroughEnd {
            lrCyclePeakEndError = true
            anyError = true
        }
        if let n = LRCycleDecayHorizonSteps.parsedInDeclaredRange(lrCycleDecayHorizonText) {
            lrCycleDecayHorizonError = false
            if n != p.lrCycleDecayHorizonSteps { p.lrCycleDecayHorizonSteps = n }
        } else {
            lrCycleDecayHorizonError = true
            anyError = true
        }
        // Momentum-follow bounds.
        if momentumFollowsLRCycleValue != p.momentumFollowsLRCycle { p.momentumFollowsLRCycle = momentumFollowsLRCycleValue }
        if let v = editedValue(MomentumFollowStartLow.self, \.momentumFollowStartLowText, current: p.momentumFollowStartLow) {
            momentumFollowStartLowError = false
            if abs(v - p.momentumFollowStartLow) > Double.ulpOfOne { p.momentumFollowStartLow = v }
        } else {
            momentumFollowStartLowError = true
            anyError = true
        }
        if let v = editedValue(MomentumFollowStartHigh.self, \.momentumFollowStartHighText, current: p.momentumFollowStartHigh) {
            momentumFollowStartHighError = false
            if abs(v - p.momentumFollowStartHigh) > Double.ulpOfOne { p.momentumFollowStartHigh = v }
        } else {
            momentumFollowStartHighError = true
            anyError = true
        }
        if let v = editedValue(MomentumFollowEndLow.self, \.momentumFollowEndLowText, current: p.momentumFollowEndLow) {
            momentumFollowEndLowError = false
            if abs(v - p.momentumFollowEndLow) > Double.ulpOfOne { p.momentumFollowEndLow = v }
        } else {
            momentumFollowEndLowError = true
            anyError = true
        }
        if let v = editedValue(MomentumFollowEndHigh.self, \.momentumFollowEndHighText, current: p.momentumFollowEndHigh) {
            momentumFollowEndHighError = false
            if abs(v - p.momentumFollowEndHigh) > Double.ulpOfOne { p.momentumFollowEndHigh = v }
        } else {
            momentumFollowEndHighError = true
            anyError = true
        }
        // Cross-field: each follow pair's high must be >= its low. The
        // schedule reports no momentum (static fallback) for an inverted
        // pair, so flag it rather than let following silently switch off.
        if !momentumFollowStartLowError, !momentumFollowStartHighError,
           p.momentumFollowStartHigh < p.momentumFollowStartLow {
            momentumFollowStartHighError = true
            anyError = true
        }
        if !momentumFollowEndLowError, !momentumFollowEndHighError,
           p.momentumFollowEndHigh < p.momentumFollowEndLow {
            momentumFollowEndHighError = true
            anyError = true
        }
        // √batch scaling toggle — Bool, cannot fail to parse.
        if sqrtBatchScalingValue != p.sqrtBatchScalingLR {
            SessionLogger.shared.log(
                "[PARAM] sqrtBatchScalingLR: \(p.sqrtBatchScalingLR) -> \(sqrtBatchScalingValue)"
            )
            p.sqrtBatchScalingLR = sqrtBatchScalingValue
        }

        // Signed-advantage complement-CE toggle — Bool.
        if signedAdvantageComplementCEValue != p.signedAdvantageComplementCE {
            SessionLogger.shared.log(
                "[PARAM] signedAdvantageComplementCE: \(p.signedAdvantageComplementCE) -> \(signedAdvantageComplementCEValue)"
            )
            p.signedAdvantageComplementCE = signedAdvantageComplementCEValue
        }

        // Entropy regularization — Double in the declared range.
        if let v = editedValue(EntropyBonus.self, \.entropyText, current: p.entropyBonus) {
            entropyError = false
            if abs(v - p.entropyBonus) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] entropyBonus: %.6g -> %.6g", p.entropyBonus, v)
                )
                p.entropyBonus = v
            }
        } else {
            entropyError = true
            anyError = true
        }

        // Illegal mass penalty — Double in the declared range.
        if let v = editedValue(IllegalMassWeight.self, \.illegalMassWeightText, current: p.illegalMassWeight) {
            illegalMassWeightError = false
            if abs(v - p.illegalMassWeight) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] illegalMassWeight: %.6g -> %.6g", p.illegalMassWeight, v)
                )
                p.illegalMassWeight = v
            }
        } else {
            illegalMassWeightError = true
            anyError = true
        }

        // Grad clip — Double in the declared range.
        if let v = editedValue(GradClipMaxNorm.self, \.gradClipText, current: p.gradClipMaxNorm) {
            gradClipError = false
            if abs(v - p.gradClipMaxNorm) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] gradClipMaxNorm: %.6g -> %.6g", p.gradClipMaxNorm, v)
                )
                p.gradClipMaxNorm = v
            }
        } else {
            gradClipError = true
            anyError = true
        }

        // Weight decay — Double in the declared range.
        if let v = editedValue(WeightDecay.self, \.weightDecayText, current: p.weightDecay) {
            weightDecayError = false
            if abs(v - p.weightDecay) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] weightDecay: %.6g -> %.6g", p.weightDecay, v)
                )
                p.weightDecay = v
            }
        } else {
            weightDecayError = true
            anyError = true
        }

        // Dropout rate — Double in the declared range. Drop probability
        // (PyTorch/Keras convention); 0 disables. Pushed onto the live
        // trainer via the graph-variable assign (`ChessTrainer.dropoutRate`).
        if let v = editedValue(DropoutRate.self, \.dropoutRateText, current: p.dropoutRate) {
            dropoutRateError = false
            if abs(v - p.dropoutRate) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] dropoutRate: %.6g -> %.6g", p.dropoutRate, v)
                )
                p.dropoutRate = v
            }
        } else {
            dropoutRateError = true
            anyError = true
        }

        // Policy loss weight — Double in the declared range.
        if let v = editedValue(PolicyLossWeight.self, \.policyLossWeightText, current: p.policyLossWeight) {
            policyLossWeightError = false
            if abs(v - p.policyLossWeight) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] policyLossWeight: %.6g -> %.6g", p.policyLossWeight, v)
                )
                p.policyLossWeight = v
            }
        } else {
            policyLossWeightError = true
            anyError = true
        }

        // Value loss weight — Double in the declared range.
        if let v = editedValue(ValueLossWeight.self, \.valueLossWeightText, current: p.valueLossWeight) {
            valueLossWeightError = false
            if abs(v - p.valueLossWeight) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] valueLossWeight: %.6g -> %.6g", p.valueLossWeight, v)
                )
                p.valueLossWeight = v
            }
        } else {
            valueLossWeightError = true
            anyError = true
        }

        // Value-head label smoothing ε — Double in the declared range.
        if let v = editedValue(ValueLabelSmoothingEpsilon.self, \.valueLabelSmoothingText, current: p.valueLabelSmoothingEpsilon) {
            valueLabelSmoothingError = false
            if abs(v - p.valueLabelSmoothingEpsilon) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] valueLabelSmoothingEpsilon: %.6g -> %.6g", p.valueLabelSmoothingEpsilon, v)
                )
                p.valueLabelSmoothingEpsilon = v
            }
        } else {
            valueLabelSmoothingError = true
            anyError = true
        }

        // Policy label smoothing — only the fields the pending mode reads are
        // validated and written; the inactive mode's fields are left as they
        // are (see `policyLabelSmoothingModeValue`) and their stale error
        // flags cleared, so a dimmed field can never hold Save disabled. The
        // mode itself is committed only once every field it reads has
        // validated: switching to per-move with a bad δ must not start
        // training under per-move with the previous δ.
        switch policyLabelSmoothingModeValue {
        case .fixedTotal:
            policyLabelSmoothingPerMoveError = false
            policyLabelSmoothingPerMoveCapError = false
            let epsilon = editedValue(PolicyLabelSmoothingEpsilon.self, \.policyLabelSmoothingText, current: p.policyLabelSmoothingEpsilon)
            policyLabelSmoothingError = epsilon == nil
            if let epsilon {
                if abs(epsilon - p.policyLabelSmoothingEpsilon) > Double.ulpOfOne {
                    SessionLogger.shared.log(
                        String(format: "[PARAM] policyLabelSmoothingEpsilon: %.6g -> %.6g", p.policyLabelSmoothingEpsilon, epsilon)
                    )
                    p.policyLabelSmoothingEpsilon = epsilon
                }
                commitPolicyLabelSmoothingMode(to: p)
            } else {
                anyError = true
            }
        case .perMove:
            policyLabelSmoothingError = false
            let perMove = editedValue(PolicyLabelSmoothingPerMove.self, \.policyLabelSmoothingPerMoveText, current: p.policyLabelSmoothingPerMove)
            let perMoveCap = editedValue(PolicyLabelSmoothingPerMoveCap.self, \.policyLabelSmoothingPerMoveCapText, current: p.policyLabelSmoothingPerMoveCap)
            policyLabelSmoothingPerMoveError = perMove == nil
            policyLabelSmoothingPerMoveCapError = perMoveCap == nil
            if let perMove, let perMoveCap {
                if abs(perMove - p.policyLabelSmoothingPerMove) > Double.ulpOfOne {
                    SessionLogger.shared.log(
                        String(format: "[PARAM] policyLabelSmoothingPerMove: %.6g -> %.6g", p.policyLabelSmoothingPerMove, perMove)
                    )
                    p.policyLabelSmoothingPerMove = perMove
                }
                if abs(perMoveCap - p.policyLabelSmoothingPerMoveCap) > Double.ulpOfOne {
                    SessionLogger.shared.log(
                        String(format: "[PARAM] policyLabelSmoothingPerMoveCap: %.6g -> %.6g", p.policyLabelSmoothingPerMoveCap, perMoveCap)
                    )
                    p.policyLabelSmoothingPerMoveCap = perMoveCap
                }
                commitPolicyLabelSmoothingMode(to: p)
            } else {
                anyError = true
            }
        }

        // Draw penalty — Double in the declared range.
        if let v = editedValue(DrawPenalty.self, \.drawPenaltyText, current: p.drawPenalty) {
            drawPenaltyError = false
            if abs(v - p.drawPenalty) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] drawPenalty: %.6g -> %.6g", p.drawPenalty, v)
                )
                p.drawPenalty = v
            }
        } else {
            drawPenaltyError = true
            anyError = true
        }

        // Training batch size — Int in the declared range. Snapshot-only; the live
        // trainer rebuilds its feed cache lazily on the next batch shape.
        if let n = TrainingBatchSize.parsedInDeclaredRange(trainingBatchSizeText) {
            trainingBatchSizeError = false
            if n != p.trainingBatchSize {
                SessionLogger.shared.log("[PARAM] trainingBatchSize: \(p.trainingBatchSize) -> \(n)")
                p.trainingBatchSize = n
            }
        } else {
            trainingBatchSizeError = true
            anyError = true
        }

        // Self-play workers — Int in the declared range. Live-tunable: the
        // BatchedSelfPlayDriver reconcile loop picks up the new count.
        if let n = SelfPlayConcurrency.parsedInDeclaredRange(selfPlayConcurrencyText),
           n <= maxSelfPlayWorkers {
            selfPlayConcurrencyError = false
            if n != p.selfPlayConcurrency {
                SessionLogger.shared.log("[PARAM] selfPlayConcurrency: \(p.selfPlayConcurrency) -> \(n)")
                p.selfPlayConcurrency = n
            }
        } else {
            selfPlayConcurrencyError = true
            anyError = true
        }

        // Self-play τ schedule — start, decay and floor, each in its
        // declared range. Rebuilt by `buildSelfPlaySchedule()` next time
        // the schedule box is constructed; mid-session changes don't
        // retroactively alter games already in progress.
        if let v = editedValue(SelfPlayStartTau.self, \.selfPlayStartTauText, current: p.selfPlayStartTau) {
            selfPlayStartTauError = false
            if abs(v - p.selfPlayStartTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayStartTau: %.6g -> %.6g", p.selfPlayStartTau, v)
                )
                p.selfPlayStartTau = v
            }
        } else {
            selfPlayStartTauError = true
            anyError = true
        }
        if let v = editedValue(SelfPlayTauDecayPerPly.self, \.selfPlayDecayPerPlyText, current: p.selfPlayTauDecayPerPly) {
            selfPlayDecayPerPlyError = false
            if abs(v - p.selfPlayTauDecayPerPly) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayTauDecayPerPly: %.6g -> %.6g", p.selfPlayTauDecayPerPly, v)
                )
                p.selfPlayTauDecayPerPly = v
            }
        } else {
            selfPlayDecayPerPlyError = true
            anyError = true
        }
        if let v = editedValue(SelfPlayTargetTau.self, \.selfPlayFloorTauText, current: p.selfPlayTargetTau) {
            selfPlayFloorTauError = false
            if abs(v - p.selfPlayTargetTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayTargetTau: %.6g -> %.6g", p.selfPlayTargetTau, v)
                )
                p.selfPlayTargetTau = v
            }
        } else {
            selfPlayFloorTauError = true
            anyError = true
        }
        // Self-Play Draw Keep Fraction — Double in the declared range.
        // Live-propagated to `TrainingParameters.shared` during the
        // edit via `applyLiveSelfPlayDrawKeepFraction`, so save() only
        // needs to validate the current edit text for the red-overlay
        // display; the new committed value is captured by the next
        // `seedFromParams()` call into `originalSelfPlayDrawKeepFraction`
        // so a subsequent Cancel after Save reverts to "what Save
        // committed," not the original pre-edit value.
        // Empty text is treated as the parameter's default (1.0 =
        // keep every drawn game) — matches the popover's placeholder
        // and the `.onChange` handler's "blank means default" branch
        // (which has already fired the live propagation by the time
        // we get here).
        let drawKeepTrimmed = selfPlayDrawKeepFractionText.trimmingCharacters(in: .whitespaces)
        if drawKeepTrimmed.isEmpty {
            selfPlayDrawKeepFractionError = false
        } else if SelfPlayDrawKeepFraction.parsedInDeclaredRange(drawKeepTrimmed) != nil {
            selfPlayDrawKeepFractionError = false
        } else {
            selfPlayDrawKeepFractionError = true
            anyError = true
        }
        // Max plies per game — Int in the declared range. Same live-propagated
        // pattern as draw-keep fraction: the slot driver reads
        // `TrainingParameters.shared.selfPlayMaxPliesPerGame` at the start of
        // each game, so save() just validates the current text and
        // logs a [PARAM] line if the committed value changed.
        let selfPlayMaxPliesPerGameTrimmed = selfPlayMaxPliesPerGameText.trimmingCharacters(in: .whitespaces)
        if selfPlayMaxPliesPerGameTrimmed.isEmpty {
            selfPlayMaxPliesPerGameError = false
        } else if SelfPlayMaxPliesPerGame.parsedInDeclaredRange(selfPlayMaxPliesPerGameTrimmed) != nil {
            selfPlayMaxPliesPerGameError = false
        } else {
            selfPlayMaxPliesPerGameError = true
            anyError = true
        }
        // Draw-watch pDraw threshold — Double in the declared range. Same
        // live-propagated pattern: the driver re-reads
        // `TrainingParameters.shared.drawWatchPDrawThreshold` at the
        // start of every tick, so save() just validates the current
        // text and logs a [PARAM] line if the committed value
        // changed.
        let drawWatchPDrawThresholdTrimmed = drawWatchPDrawThresholdText.trimmingCharacters(in: .whitespaces)
        if drawWatchPDrawThresholdTrimmed.isEmpty {
            drawWatchPDrawThresholdError = false
        } else if DrawWatchPDrawThreshold.parsedInDeclaredRange(drawWatchPDrawThresholdTrimmed) != nil {
            drawWatchPDrawThresholdError = false
        } else {
            drawWatchPDrawThresholdError = true
            anyError = true
        }
        // Draw-watch streak length — Int in the declared range. Same live-
        // propagated pattern; driver re-reads each tick.
        let drawWatchStreakLengthTrimmed = drawWatchStreakLengthText.trimmingCharacters(in: .whitespaces)
        if drawWatchStreakLengthTrimmed.isEmpty {
            drawWatchStreakLengthError = false
        } else if DrawWatchStreakLength.parsedInDeclaredRange(drawWatchStreakLengthTrimmed) != nil {
            drawWatchStreakLengthError = false
        } else {
            drawWatchStreakLengthError = true
            anyError = true
        }
        // Push the freshly-edited self-play schedule into the live
        // `samplingScheduleBox` so the next self-play game on each worker slot
        // picks up the new τ curve. Safe to call unconditionally — the box's
        // `setSelfPlay` is a no-op before the first session.
        pushSelfPlaySchedule()

        // Replay buffer capacity — Int in the declared range. Snapshot-only:
        // the live ring cannot resize mid-session.
        if let n = ReplayBufferCapacity.parsedInDeclaredRange(replayBufferCapacityText) {
            replayBufferCapacityError = false
            if n != p.replayBufferCapacity {
                SessionLogger.shared.log("[PARAM] replayBufferCapacity: \(p.replayBufferCapacity) -> \(n)")
                p.replayBufferCapacity = n
            }
        } else {
            replayBufferCapacityError = true
            anyError = true
        }

        // Pre-train fill threshold — Int in the declared range. Live-tunable.
        if let n = ReplayBufferMinPositionsBeforeTraining.parsedInDeclaredRange(replayBufferMinPositionsText) {
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

        // Replay-sampling constraints — Int, live-propagated during edits
        // via `applyLiveMaxPliesFromAnyOneGame` / `applyLiveTargetSampledGameLengthPlies`
        // / `applyLiveMaxDrawPercentPerBatch` so the buffer's Composition
        // readout updates in real time. Save validates the current edit
        // text against each parameter's range for the red-overlay display;
        // the [PARAM] log line fires below if the committed value differs
        // from the pre-edit stash (mirrors the four replay-ratio fields).
        // Empty text on each live-propagated field is treated as the
        // parameter's default — matches the popover's placeholder
        // and the `.onChange` handler's "blank means default" branch
        // (which has already fired the live propagation by the time
        // we get here). Without this, Save would error-flag the
        // field for blank input even though the live readout below
        // the field already shows the default value.
        let maxPliesTrimmed = maxPliesFromAnyOneGameText.trimmingCharacters(in: .whitespaces)
        if maxPliesTrimmed.isEmpty {
            maxPliesFromAnyOneGameError = false
        } else if MaxPliesFromAnyOneGame.parsedInDeclaredRange(maxPliesTrimmed) != nil {
            maxPliesFromAnyOneGameError = false
        } else {
            maxPliesFromAnyOneGameError = true
            anyError = true
        }
        let targetLenTrimmed = targetSampledGameLengthPliesText.trimmingCharacters(in: .whitespaces)
        if targetLenTrimmed.isEmpty {
            targetSampledGameLengthPliesError = false
        } else if TargetSampledGameLengthPlies.parsedInDeclaredRange(targetLenTrimmed) != nil {
            targetSampledGameLengthPliesError = false
        } else {
            targetSampledGameLengthPliesError = true
            anyError = true
        }
        let maxDrawTrimmed = maxDrawPercentPerBatchText.trimmingCharacters(in: .whitespaces)
        if maxDrawTrimmed.isEmpty {
            maxDrawPercentPerBatchError = false
        } else if MaxDrawPercentPerBatch.parsedInDeclaredRange(maxDrawTrimmed) != nil {
            maxDrawPercentPerBatchError = false
        } else {
            maxDrawPercentPerBatchError = true
            anyError = true
        }

        // Replay-ratio control fields are live-propagated during edits via
        // `applyLive…` — the writes already reached `trainingParams`. Save
        // validates the current text values for red-overlay display only; no
        // parameter writes here.
        if ReplayRatioTarget.parsedInDeclaredRange(replayRatioTargetText) != nil {
            replayRatioTargetError = false
        } else {
            replayRatioTargetError = true
            anyError = true
        }
        if let n = SelfPlayDelayMs.parsedInDeclaredRange(replaySelfPlayDelayText),
           n <= selfPlayDelayMaxMs {
            replaySelfPlayDelayError = false
        } else {
            replaySelfPlayDelayError = true
            anyError = true
        }
        if TrainingStepDelayMs.parsedInDeclaredRange(replayTrainingStepDelayText) != nil {
            replayTrainingStepDelayError = false
        } else {
            replayTrainingStepDelayError = true
            anyError = true
        }

        // --- Sessions tab ---
        // Autosave interval — edited in minutes, stored as seconds, and
        // validated in seconds against the parameter's declared range.
        // Commit-on-Save: the heartbeat re-anchors the running controller from
        // the new value, so no live trainer write is needed here.
        if let mins = Int(periodicAutosaveIntervalMinutesText.trimmingCharacters(in: .whitespaces)),
           PeriodicAutosaveIntervalSec.isWithinDeclaration(Double(mins) * Self.secondsPerMinute) {
            periodicAutosaveIntervalError = false
            let secs = Double(mins) * Self.secondsPerMinute
            if abs(secs - p.periodicAutosaveIntervalSec) > 0.5 {
                SessionLogger.shared.log(
                    "[PARAM] periodicAutosaveIntervalSec: \(Int(p.periodicAutosaveIntervalSec)) -> \(Int(secs))"
                )
                p.periodicAutosaveIntervalSec = secs
            }
        } else {
            periodicAutosaveIntervalError = true
            anyError = true
        }
        // Max periodic autosaves kept — Int in the declared range; 0 = unlimited.
        // Read live at prune time (after each periodic or post-promotion
        // save), so a plain singleton write is all that's required.
        if let n = MaxPeriodicAutosavesKept.parsedInDeclaredRange(maxPeriodicAutosavesKeptText) {
            maxPeriodicAutosavesKeptError = false
            if n != p.maxPeriodicAutosavesKept {
                SessionLogger.shared.log("[PARAM] maxPeriodicAutosavesKept: \(p.maxPeriodicAutosavesKept) -> \(n)")
                p.maxPeriodicAutosavesKept = n
            }
        } else {
            maxPeriodicAutosavesKeptError = true
            anyError = true
        }
        // Automatic-save pruning toggle — Bool, cannot fail to parse. Read
        // live after each periodic or post-promotion save, so a plain
        // singleton write is all that's required. Stored even while the
        // build forces pruning off, so the preference is there when it is
        // lifted.
        if automaticSavePruningEnabledValue != p.automaticSavePruningEnabled {
            SessionLogger.shared.log(
                "[PARAM] automaticSavePruningEnabled: \(p.automaticSavePruningEnabled) -> \(automaticSavePruningEnabledValue)"
            )
            p.automaticSavePruningEnabled = automaticSavePruningEnabledValue
        }
        // KL probe interval — Int in the declared range; 0 = off. `liveTunable`, and
        // the training loop reconciles it against the running trainer on its
        // poll, so a plain singleton write is all that is required here.
        if let n = KLProbeInterval.parsedInDeclaredRange(klProbeIntervalText) {
            klProbeIntervalError = false
            if n != p.klProbeInterval {
                SessionLogger.shared.log("[PARAM] klProbeInterval: \(p.klProbeInterval) -> \(n)")
                p.klProbeInterval = n
            }
        } else {
            klProbeIntervalError = true
            anyError = true
        }
        // Run seed — the mode, then (in seeded mode only) a UInt64 seed.
        // Read when a run starts, so a plain singleton write is all that's
        // required. The seed is validated before either is written, so a bad
        // seed never leaves `seeded` committed over the previous seed.
        let pendingSeed: UInt64?
        switch randomSeedModeValue {
        case .seeded:
            if let seed = RandomSeed.parsedInDeclaredRange(randomSeedText) {
                randomSeedError = false
                pendingSeed = seed
            } else {
                randomSeedError = true
                anyError = true
                pendingSeed = nil
            }
        case .unseeded:
            randomSeedError = false
            pendingSeed = nil
        }
        if !randomSeedError {
            if randomSeedModeValue != p.randomSeedMode {
                SessionLogger.shared.log(
                    "[PARAM] randomSeedMode: \(p.randomSeedMode.logToken) -> \(randomSeedModeValue.logToken)"
                )
                p.randomSeedMode = randomSeedModeValue
            }
            if let pendingSeed, pendingSeed != p.randomSeed {
                SessionLogger.shared.log("[PARAM] randomSeed: \(p.randomSeed) -> \(pendingSeed)")
                p.randomSeed = pendingSeed
            }
        }

        // Push the committed configuration onto the live trainer in one call,
        // through the same `TrainerHyperparameters` path that configures a
        // trainer at session start and in the CLI runners, so the popover can
        // never update a subset the other paths disagree with. Runs whether or
        // not some field failed: every field that did validate has already been
        // written to the singleton above, and an invalid one left its previous
        // value in place. The trainer reads the cycle from its own `SyncBox`,
        // so this push (not only `ControlSideEffectsProbe`'s `.onChange`, which
        // still owns the bundled `[PARAM] lr_momentum_cycle` line) is what makes
        // a cycling edit take effect even if the probe is not mounted.
        if let trainer {
            TrainerHyperparameters(p.snapshot()).apply(to: trainer)
        }

        if !anyError {
            // Commit-time [PARAM] log lines for the seven live-propagated
            // Replay-tab fields. The live writes during the edit session are
            // intentionally silent (a log per keystroke would be noise); this
            // is the single authoritative log line per Save.
            if abs(p.replayRatioTarget - originalReplayRatioTarget) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] replayRatioTarget: %.6g -> %.6g", originalReplayRatioTarget, p.replayRatioTarget)
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
            // Same commit-time logging for the three live-propagated
            // sampling-constraint fields.
            if p.maxPliesFromAnyOneGame != originalMaxPliesFromAnyOneGame {
                SessionLogger.shared.log(
                    "[PARAM] maxPliesFromAnyOneGame: \(originalMaxPliesFromAnyOneGame) -> \(p.maxPliesFromAnyOneGame)"
                )
            }
            if p.targetSampledGameLengthPlies != originalTargetSampledGameLengthPlies {
                SessionLogger.shared.log(
                    "[PARAM] targetSampledGameLengthPlies: \(originalTargetSampledGameLengthPlies) -> \(p.targetSampledGameLengthPlies)"
                )
            }
            if p.maxDrawPercentPerBatch != originalMaxDrawPercentPerBatch {
                SessionLogger.shared.log(
                    "[PARAM] maxDrawPercentPerBatch: \(originalMaxDrawPercentPerBatch) -> \(p.maxDrawPercentPerBatch)"
                )
            }
            if abs(p.selfPlayDrawKeepFraction - originalSelfPlayDrawKeepFraction) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(
                        format: "[PARAM] selfPlayDrawKeepFraction: %.2f -> %.2f",
                        originalSelfPlayDrawKeepFraction,
                        p.selfPlayDrawKeepFraction
                    )
                )
            }
            if p.selfPlayMaxPliesPerGame != originalSelfPlayMaxPliesPerGame {
                SessionLogger.shared.log(
                    "[PARAM] selfPlayMaxPliesPerGame: \(originalSelfPlayMaxPliesPerGame) -> \(p.selfPlayMaxPliesPerGame)"
                )
            }
            if abs(p.drawWatchPDrawThreshold - originalDrawWatchPDrawThreshold) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] drawWatchPDrawThreshold: %.6g -> %.6g",
                        originalDrawWatchPDrawThreshold,
                        p.drawWatchPDrawThreshold
                    )
                )
            }
            if p.drawWatchTerminateGames != originalDrawWatchTerminateGames {
                SessionLogger.shared.log(
                    "[PARAM] drawWatchTerminateGames: \(originalDrawWatchTerminateGames) -> \(p.drawWatchTerminateGames)"
                )
            }
            if p.drawWatchStreakLength != originalDrawWatchStreakLength {
                SessionLogger.shared.log(
                    "[PARAM] drawWatchStreakLength: \(originalDrawWatchStreakLength) -> \(p.drawWatchStreakLength)"
                )
            }
            // On successful save the stash that backs Cancel becomes the new
            // pre-edit baseline — closing the popover with Save commits the
            // live writes. Missing this line for `selfPlayDrawKeepFraction`
            // was the bug behind the reported "edit saved 0.50 then reopened
            // and it's 1.00" symptom: the popover dismissed via Save →
            // onDisappear → cancel() saw a stash that still held the
            // pre-edit value and reverted the just-committed change.
            originalReplayRatioTarget = p.replayRatioTarget
            originalReplaySelfPlayDelayMs = p.selfPlayDelayMs
            originalReplayTrainingStepDelayMs = p.trainingStepDelayMs
            originalReplayRatioAutoAdjust = p.replayRatioAutoAdjust
            originalMaxPliesFromAnyOneGame = p.maxPliesFromAnyOneGame
            originalTargetSampledGameLengthPlies = p.targetSampledGameLengthPlies
            originalMaxDrawPercentPerBatch = p.maxDrawPercentPerBatch
            originalReplayBufferStratifyByMaterial = p.replayBufferStratifyByMaterial
            originalSelfPlayDrawKeepFraction = p.selfPlayDrawKeepFraction
            originalSelfPlayMaxPliesPerGame = p.selfPlayMaxPliesPerGame
            originalDrawWatchPDrawThreshold = p.drawWatchPDrawThreshold
            originalDrawWatchTerminateGames = p.drawWatchTerminateGames
            originalDrawWatchStreakLength = p.drawWatchStreakLength
            isPresented = false
        }
    }
}
