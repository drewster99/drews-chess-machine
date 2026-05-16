import SwiftUI

/// Tabbed editor for every live-tunable parameter that affects how
/// Play-and-Train runs. Pulled out of the inline stats text panel
/// so the main screen stays focused on telemetry.
///
/// Three tabs:
///   - **Optimizer**: LR + warmup + momentum + √-batch, entropy
///     bonus + clip + decay + K, draw penalty, training batch size.
///     The historical content of this popover.
///   - **Self Play**: concurrency + self-play move delay + temperature
///     schedule (start / decay / floor with "reached at ply N" hint).
///   - **Replay**: buffer capacity (with auto-GB readout), pre-train
///     fill threshold (with auto-% readout), and the replay-ratio
///     control panel (live current-ratio readout, self-play delay,
///     train step delay, auto-control checkbox).
///
/// All transactional state — edit-text fields, per-field error flags, the
/// validate-and-write-back logic, and the live-propagation of the replay-ratio
/// fields — lives on the `TrainingSettingsPopoverModel` (`@MainActor @Observable`),
/// which this view receives as `@Bindable model`. Keeping it on a model object
/// (rather than `@State` on `UpperContentView`) keeps in-progress text alive
/// across renders, lets the sub-tab views bind directly off `$model`, and means
/// editing a field clears its own error via the model's `didSet` — so no
/// `.onChange` chain is needed here. Save/Cancel/onAppearSeed call
/// `model.save()` / `model.cancel()` / `model.seedFromParams()`.
///
/// **Live-propagation exception for replay-ratio fields.** The replay-ratio
/// control fields (`replayRatioTarget`, `selfPlayDelayMs`, `trainingStepDelayMs`,
/// `replayRatioAutoAdjust`) write through to `trainingParams` *immediately on
/// change* via `model.applyLive…` rather than waiting for Save, so the user can
/// watch the live ratio display respond. On Cancel (and on outside-click
/// dismiss, routed through `model.cancel()`) those are reverted from the stash
/// captured in `model.seedFromParams()`, matching the standard
/// "edit → cancel discards" pattern from the user's POV.
fileprivate enum Tab: String, CaseIterable, Identifiable {
    case optimizer = "Optimizer"
    case selfPlay = "Self Play"
    case replay = "Replay"
    var id: String { rawValue }
}

struct TrainingSettingsPopover: View {

    /// All transactional state (edit-text fields, per-field error flags,
    /// validation + write-back, live-propagation of the replay-ratio fields).
    @Bindable var model: TrainingSettingsPopoverModel

    /// Trainer model ID at session start, displayed in the header.
    /// "—" when no trainer exists yet (build-pre-session).
    let modelID: String
    /// Session start wall-clock time, displayed in the header so the
    /// user has a stable "this is the run you're configuring" anchor.
    let sessionStart: Date

    /// Live "current ratio (target X.XX)" snapshot for the Replay
    /// tab's status line. Refreshes with the parent's heartbeat;
    /// nil before the controller has produced its first sample.
    let replayRatioCurrent: Double?
    /// Live auto-computed training-step delay from the controller.
    /// Displayed in place of the editable Train-step-delay field
    /// when `model.replayRatioAutoAdjust == true`. Nil before the
    /// controller has produced its first sample.
    let replayRatioComputedDelayMs: Int?
    /// Live auto-computed self-play delay from the controller.
    let replayRatioComputedSelfPlayDelayMs: Int?
    /// Bytes-per-position for the auto-GB readout (`ReplayBuffer.bytesPerPosition`).
    let bytesPerPosition: Int
    /// Live resident-set composition of the replay buffer, for the
    /// "Replay sampling" section's pre-constraint readout. `nil` outside
    /// a Play-and-Train session / before the first heartbeat.
    let bufferComposition: ReplayBuffer.CompositionSnapshot?

    /// Most recent `ReplayBuffer.sample(...)` achievement report — drives
    /// the "Last batch" column of the two-column composition readout in
    /// the Replay tab. `nil` until the first batch has landed in the
    /// current Play-and-Train session, at which point the heartbeat
    /// mirrors `replayBuffer.lastSamplingResult()` into it.
    let lastSamplingResult: ReplayBuffer.SamplingResult?

    /// Live parallel-worker stats snapshot. Drives the Self Play tab's
    /// "Emitted games" readout (played vs emitted plies-per-hour, played
    /// vs emitted W/L/D shares). `nil` before the first heartbeat tick
    /// in the current Play-and-Train session.
    let parallelStats: ParallelWorkerStatsBox.Snapshot?

    // MARK: - Tab selection

    /// Local tab selection. Resets to `.optimizer` each time the
    /// popover re-instantiates (i.e. each time the chip is clicked
    /// open). Persisting it across opens would surprise the user
    /// — the natural mental model is "always start on Optimizer."
    @State private var selectedTab: Tab = .optimizer

    // Aggregated per-tab error flags, derived from the per-field `*Error`
    // properties on the model. These drive (a) the red-dot indicator on each
    // tab in the segmented control and (b) the Save button's `.disabled(...)` —
    // once Save has set any error flag, the user clears it by editing the
    // offending field (the model's `didSet` does that), at which point Save
    // re-enables and the next click re-validates the full form transactionally.
    private var optimizerHasError: Bool {
        model.lrError
            || model.warmupError
            || model.momentumError
            || model.entropyError
            || model.gradClipError
            || model.weightDecayError
            || model.policyLossWeightError
            || model.valueLossWeightError
            || model.valueLabelSmoothingError
            || model.drawPenaltyError
            || model.trainingBatchSizeError
    }

    private var selfPlayHasError: Bool {
        model.selfPlayConcurrencyError
            || model.selfPlayStartTauError
            || model.selfPlayDecayPerPlyError
            || model.selfPlayFloorTauError
            || model.selfPlayDrawKeepFractionError
            || model.selfPlayMaxPliesPerGameError
    }

    private var replayHasError: Bool {
        model.replayBufferCapacityError
            || model.replayBufferMinPositionsError
            || model.replayRatioTargetError
            || model.replaySelfPlayDelayError
            || model.replayTrainingStepDelayError
            || model.maxPliesFromAnyOneGameError
            || model.targetSampledGameLengthPliesError
            || model.maxDrawPercentPerBatchError
    }

    private var anyTabHasError: Bool {
        optimizerHasError || selfPlayHasError || replayHasError
    }

    var body: some View {
        let dateFmt: DateFormatter = {
            let f = DateFormatter()
            f.dateStyle = .short
            f.timeStyle = .medium
            return f
        }()

        VStack(alignment: .leading, spacing: 12) {
            // Header — display-only model ID + session-start timestamp.
            VStack(alignment: .leading, spacing: 4) {
                Text("Training")
                    .font(.headline)
                Text(modelID)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                Text(dateFmt.string(from: sessionStart))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            // Tab bar — a small custom segmented control rather than
            // `Picker(.segmented)` because the macOS native segmented
            // control does not allow per-segment colored content. We
            // render a 6-pt red dot next to a tab's label whenever
            // any field on that tab is currently flagged with a
            // validation error, so the user can see at a glance which
            // tab(s) need attention even when looking at a different
            // tab. Visual styling is intentionally close to (not
            // pixel-perfect with) `NSSegmentedControl`.
            TrainingSettingsTabBar(
                selectedTab: $selectedTab,
                optimizerHasError: optimizerHasError,
                selfPlayHasError: selfPlayHasError,
                replayHasError: replayHasError
            )

            Divider()

            // Per-tab content. Each tab is its own subview struct so
            // the parent body stays small and the Stepper/binding
            // helpers can live close to the fields they drive. The
            // sub-views keep their existing `@Binding` interface; the
            // bindings are projected off `model` here.
            //
            // Wrapped in a `ScrollView` because the Replay tab can
            // get long (replay-ratio control + composition readout +
            // sampling section), and on smaller screens the
            // Cancel / Save row at the bottom would otherwise fall
            // off the visible region. The outer `.frame(maxHeight:)`
            // below caps the popover so the ScrollView has a known
            // bounded vertical region to scroll within.
            ScrollView(.vertical, showsIndicators: true) {
            switch selectedTab {
            case .optimizer:
                OptimizerTab(
                    lrText: $model.lrText,
                    warmupText: $model.warmupText,
                    momentumText: $model.momentumText,
                    sqrtBatchScalingValue: $model.sqrtBatchScalingValue,
                    entropyText: $model.entropyText,
                    illegalMassWeightText: $model.illegalMassWeightText,
                    gradClipText: $model.gradClipText,
                    weightDecayText: $model.weightDecayText,
                    policyLossWeightText: $model.policyLossWeightText,
                    valueLossWeightText: $model.valueLossWeightText,
                    valueLabelSmoothingText: $model.valueLabelSmoothingText,
                    drawPenaltyText: $model.drawPenaltyText,
                    trainingBatchSizeText: $model.trainingBatchSizeText,
                    lrError: model.lrError,
                    warmupError: model.warmupError,
                    momentumError: model.momentumError,
                    entropyError: model.entropyError,
                    illegalMassWeightError: model.illegalMassWeightError,
                    gradClipError: model.gradClipError,
                    weightDecayError: model.weightDecayError,
                    policyLossWeightError: model.policyLossWeightError,
                    valueLossWeightError: model.valueLossWeightError,
                    valueLabelSmoothingError: model.valueLabelSmoothingError,
                    drawPenaltyError: model.drawPenaltyError,
                    trainingBatchSizeError: model.trainingBatchSizeError
                )
            case .selfPlay:
                SelfPlayTab(
                    selfPlayConcurrencyText: $model.selfPlayConcurrencyText,
                    selfPlayStartTauText: $model.selfPlayStartTauText,
                    selfPlayDecayPerPlyText: $model.selfPlayDecayPerPlyText,
                    selfPlayFloorTauText: $model.selfPlayFloorTauText,
                    selfPlayDrawKeepFractionText: $model.selfPlayDrawKeepFractionText,
                    selfPlayMaxPliesPerGameText: $model.selfPlayMaxPliesPerGameText,
                    selfPlayConcurrencyError: model.selfPlayConcurrencyError,
                    selfPlayStartTauError: model.selfPlayStartTauError,
                    selfPlayDecayPerPlyError: model.selfPlayDecayPerPlyError,
                    selfPlayFloorTauError: model.selfPlayFloorTauError,
                    selfPlayDrawKeepFractionError: model.selfPlayDrawKeepFractionError,
                    selfPlayMaxPliesPerGameError: model.selfPlayMaxPliesPerGameError,
                    onLiveSelfPlayDrawKeepFractionChange: { model.applyLiveSelfPlayDrawKeepFraction($0) },
                    onLiveMaxPliesPerGameChange: { model.applyLiveMaxPliesPerGame($0) },
                    parallelStats: parallelStats
                )
            case .replay:
                ReplayTab(
                    replayBufferCapacityText: $model.replayBufferCapacityText,
                    replayBufferMinPositionsText: $model.replayBufferMinPositionsText,
                    replayRatioTargetText: $model.replayRatioTargetText,
                    replaySelfPlayDelayText: $model.replaySelfPlayDelayText,
                    replayTrainingStepDelayText: $model.replayTrainingStepDelayText,
                    replayRatioAutoAdjust: $model.replayRatioAutoAdjust,
                    maxPliesFromAnyOneGameText: $model.maxPliesFromAnyOneGameText,
                    targetSampledGameLengthPliesText: $model.targetSampledGameLengthPliesText,
                    maxDrawPercentPerBatchText: $model.maxDrawPercentPerBatchText,
                    replayBufferCapacityError: model.replayBufferCapacityError,
                    replayBufferMinPositionsError: model.replayBufferMinPositionsError,
                    replayRatioTargetError: model.replayRatioTargetError,
                    replaySelfPlayDelayError: model.replaySelfPlayDelayError,
                    replayTrainingStepDelayError: model.replayTrainingStepDelayError,
                    maxPliesFromAnyOneGameError: model.maxPliesFromAnyOneGameError,
                    targetSampledGameLengthPliesError: model.targetSampledGameLengthPliesError,
                    maxDrawPercentPerBatchError: model.maxDrawPercentPerBatchError,
                    replayRatioCurrent: replayRatioCurrent,
                    replayRatioComputedDelayMs: replayRatioComputedDelayMs,
                    replayRatioComputedSelfPlayDelayMs: replayRatioComputedSelfPlayDelayMs,
                    bytesPerPosition: bytesPerPosition,
                    bufferComposition: bufferComposition,
                    lastSamplingResult: lastSamplingResult,
                    onLiveReplayRatioTargetChange: { model.applyLiveReplayRatioTarget($0) },
                    onLiveSelfPlayDelayChange: { model.applyLiveSelfPlayDelay($0) },
                    onLiveTrainingStepDelayChange: { model.applyLiveTrainingStepDelay($0) },
                    onLiveReplayRatioAutoAdjustChange: { model.applyLiveReplayRatioAutoAdjust($0) },
                    onLiveMaxPliesFromAnyOneGameChange: { model.applyLiveMaxPliesFromAnyOneGame($0) },
                    onLiveTargetSampledGameLengthPliesChange: { model.applyLiveTargetSampledGameLengthPlies($0) },
                    onLiveMaxDrawPercentPerBatchChange: { model.applyLiveMaxDrawPercentPerBatch($0) }
                )
            }
            }

            HStack {
                Spacer()
                Button("Cancel") { model.cancel() }
                    .keyboardShortcut(.cancelAction)
                // Save stays disabled while any field is currently
                // marked invalid. The matching error flag is cleared
                // by the model's `didSet` when the user edits the
                // offending field, at which point Save re-enables and
                // the next click re-validates the full form transactionally.
                Button("Save") { model.save() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(anyTabHasError)
            }
        }
        .padding(16)
        .frame(width: 700)
        // Hard ceiling on the popover height. Pairs with the
        // `ScrollView` around the tab-content `switch` above so a
        // tall tab (Replay tab grows the most via the composition
        // readout) keeps the Cancel / Save row visible on small
        // screens. Fits on a 1024×768 macOS minimum after accounting
        // for the menu bar and dock; on larger screens the popover
        // stops growing past this ceiling and the tab content scrolls
        // instead.
        .frame(maxHeight: 805)
        .background(.thickMaterial)
        .onAppear { model.seedFromParams() }
        .onDisappear {
            // macOS popovers dismiss on outside-click without ever
            // calling the Cancel button's action. The Replay tab
            // live-propagates values to `trainingParams` during edits,
            // so an outside-click would silently commit those changes
            // against the user's intent. By routing every dismissal
            // through `model.cancel()` we get "Cancel reverts, Save
            // commits, outside-click reverts" — and `cancel()` is
            // idempotent because Save updates the pre-edit stash before
            // closing, so a Save → onDisappear sequence finds nothing
            // to revert.
            model.cancel()
        }
    }

}

/// Custom segmented control. One button per `Tab` case, with an
/// optional 6-pt red dot trailing the label when that tab's
/// `*HasError` input is true. The selected tab is filled with a
/// light accent-color tint; unselected tabs render with a
/// secondary foreground for the label.
///
/// The three tabs are unrolled (rather than `ForEach(Tab.allCases)`)
/// so there is no per-render `Array(Tab.allCases.enumerated())`
/// allocation and no `if idx > 0 { Divider() }` conditional that
/// would change the view tree shape across re-evals. SwiftUI sees a
/// stable five-child HStack: button, divider, button, divider,
/// button.
fileprivate struct TrainingSettingsTabBar: View {
    @Binding var selectedTab: Tab
    let optimizerHasError: Bool
    let selfPlayHasError: Bool
    let replayHasError: Bool

    var body: some View {
        HStack(spacing: 0) {
            TrainingSettingsTabButton(
                tab: .optimizer,
                selectedTab: $selectedTab,
                hasError: optimizerHasError
            )
            Divider().frame(height: 18)
            TrainingSettingsTabButton(
                tab: .selfPlay,
                selectedTab: $selectedTab,
                hasError: selfPlayHasError
            )
            Divider().frame(height: 18)
            TrainingSettingsTabButton(
                tab: .replay,
                selectedTab: $selectedTab,
                hasError: replayHasError
            )
        }
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.secondary.opacity(0.35), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

/// One tab button. The error dot is always present in the tree;
/// its size and opacity flip together so SwiftUI never sees the
/// button's child count change. This avoids the
/// `if hasError { Circle() }` conditional that would otherwise
/// rebuild the AppKit-side view when validation flips.
fileprivate struct TrainingSettingsTabButton: View {
    let tab: Tab
    @Binding var selectedTab: Tab
    let hasError: Bool

    var body: some View {
        Button {
            selectedTab = tab
        } label: {
            HStack(spacing: 6) {
                Text(tab.rawValue)
                    .font(.system(size: 13))
                    .foregroundStyle(tab == selectedTab ? Color.primary : Color.secondary)
                Circle()
                    .fill(Color.red)
                    .frame(width: hasError ? 6 : 0, height: hasError ? 6 : 0)
                    .opacity(hasError ? 1 : 0)
                    .accessibilityHidden(!hasError)
                    .accessibilityLabel("Errors on \(tab.rawValue) tab")
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 4)
            .background(
                tab == selectedTab
                    ? Color.accentColor.opacity(0.18)
                    : Color.clear
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Live ratio badge

/// Live current-ratio badge. Color-coded: green within ±10% of
/// target, orange between 10% and 30%, red beyond 30% — matches
/// the existing replay-ratio chart's color semantics. Target
/// value reads from the live text-edit binding so the badge
/// updates in lock-step with the user typing into the Target
/// ratio field.
///
/// View tree is stable: a single `Text` view always — the value
/// portion and the "(target …)" portion are concatenated via
/// `Text + Text`, which produces one inline-styled Text rather
/// than an HStack. When `current` is nil the value run shows "--"
/// with secondary styling; when present, it shows the formatted
/// value with the banded color. SwiftUI never sees the structure
/// swap, so the AppKit bridge stays put — and the visual matches
/// the original single-Text rendering exactly (no HStack spacing
/// delta).
fileprivate struct LiveRatioBadge: View {
    let current: Double?
    let targetText: String

    var body: some View {
        let target = Double(targetText.trimmingCharacters(in: .whitespaces)) ?? 1.10
        let valuePart: Text = {
            guard let cur = current else {
                return Text("--").foregroundStyle(.secondary)
            }
            let delta = abs(cur - target) / max(0.001, target)
            let color: Color
            if delta < 0.10 {
                color = .green
            } else if delta < 0.30 {
                color = .orange
            } else {
                color = .red
            }
            return Text(String(format: "%.2f", cur)).foregroundStyle(color)
        }()
        return (valuePart
                + Text(" ")
                + Text(String(format: "(target %.2f)", target))
                    .foregroundStyle(.secondary))
            .font(.system(.caption, design: .monospaced))
    }
}

// MARK: - Optimizer tab

private struct OptimizerTab: View {
    @Binding var lrText: String
    @Binding var warmupText: String
    @Binding var momentumText: String
    @Binding var sqrtBatchScalingValue: Bool
    @Binding var entropyText: String
    @Binding var illegalMassWeightText: String
    @Binding var gradClipText: String
    @Binding var weightDecayText: String
    @Binding var policyLossWeightText: String
    @Binding var valueLossWeightText: String
    @Binding var valueLabelSmoothingText: String
    @Binding var drawPenaltyText: String
    @Binding var trainingBatchSizeText: String

    let lrError: Bool
    let warmupError: Bool
    let momentumError: Bool
    let entropyError: Bool
    let illegalMassWeightError: Bool
    let gradClipError: Bool
    let weightDecayError: Bool
    let policyLossWeightError: Bool
    let valueLossWeightError: Bool
    let valueLabelSmoothingError: Bool
    let drawPenaltyError: Bool
    let trainingBatchSizeError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // LR section — top-level LR, indented modifiers.
            VStack(alignment: .leading, spacing: 6) {
                PopoverRow(
                    label: "LR:",
                    text: $lrText,
                    error: lrError,
                    placeholder: "5.00e-05"
                ) {
                    // Half-decade (×√10 ≈ ×3.162) ladder: two
                    // clicks span one order of magnitude. Lets the
                    // user walk LR with finer granularity than a
                    // pure ×10 step while still climbing a decade
                    // in only two presses.
                    Stepper(
                        "",
                        onIncrement: { stepLRBy(factor: sqrt(10.0)) },
                        onDecrement: { stepLRBy(factor: 1.0 / sqrt(10.0)) }
                    )
                }
                VStack(alignment: .leading, spacing: 6) {
                    PopoverRow(
                        label: "Warm-up steps:",
                        text: $warmupText,
                        error: warmupError,
                        placeholder: "100"
                    ) {
                        Stepper(
                            "",
                            value: PopoverBindings.intBinding(text: $warmupText, fallback: 100),
                            in: 0...100_000,
                            step: 100
                        )
                    }
                    PopoverRow(
                        label: "Momentum:",
                        text: $momentumText,
                        error: momentumError,
                        placeholder: "0.000"
                    ) {
                        Stepper(
                            "",
                            value: PopoverBindings.doubleBinding(
                                text: $momentumText,
                                fallback: 0.0,
                                format: "%.3f"
                            ),
                            in: 0.0...0.99,
                            step: 0.05
                        )
                    }
                    HStack(spacing: 8) {
                        Text("")
                            .frame(width: 160, alignment: .trailing)
                        Toggle("Batch scaling", isOn: $sqrtBatchScalingValue)
                            .toggleStyle(.checkbox)
                        Spacer()
                    }
                }
                .padding(.leading, 20)
            }

            Divider()

            // Entropy section — top-level entropy, indented
            // regularization companions (clip, decay, K).
            VStack(alignment: .leading, spacing: 6) {
                PopoverRow(
                    label: "Entropy reg. bonus:",
                    text: $entropyText,
                    error: entropyError,
                    placeholder: "1.00e-03"
                ) {
                    Stepper(
                        "",
                        value: PopoverBindings.doubleBinding(
                            text: $entropyText,
                            fallback: 1e-3,
                            format: "%.2e"
                        ),
                        in: 0.0...0.1,
                        step: 1e-3
                    )
                }
                VStack(alignment: .leading, spacing: 6) {
                    PopoverRow(
                        label: "Illegal mass penalty:",
                        text: $illegalMassWeightText,
                        error: illegalMassWeightError,
                        placeholder: "1.00"
                    ) {
                        Stepper(
                            "",
                            value: PopoverBindings.doubleBinding(
                                text: $illegalMassWeightText,
                                fallback: 1.0,
                                format: "%.2f"
                            ),
                            in: 0.0...100.0,
                            step: 0.5
                        )
                    }
                    PopoverRow(
                        label: "Clip:",
                        text: $gradClipText,
                        error: gradClipError,
                        placeholder: "30.0"
                    ) {
                        Stepper(
                            "",
                            value: PopoverBindings.doubleBinding(
                                text: $gradClipText,
                                fallback: 30.0,
                                format: "%.1f"
                            ),
                            in: 0.1...1000.0,
                            step: 1.0
                        )
                    }
                    PopoverRow(
                        label: "Decay:",
                        text: $weightDecayText,
                        error: weightDecayError,
                        placeholder: "1.00e-04"
                    ) {
                        Stepper(
                            "",
                            value: PopoverBindings.doubleBinding(
                                text: $weightDecayText,
                                fallback: 1e-4,
                                format: "%.2e"
                            ),
                            in: 0.0...0.1,
                            step: 1e-4
                        )
                    }
                    PopoverRow(
                        label: "Policy loss weight:",
                        text: $policyLossWeightText,
                        error: policyLossWeightError,
                        placeholder: "1.00"
                    ) {
                        Stepper(
                            "",
                            value: PopoverBindings.doubleBinding(
                                text: $policyLossWeightText,
                                fallback: 1.0,
                                format: "%.2f"
                            ),
                            in: 0.0...20.0,
                            step: 0.5
                        )
                    }
                    PopoverRow(
                        label: "Value loss weight:",
                        text: $valueLossWeightText,
                        error: valueLossWeightError,
                        placeholder: "1.00"
                    ) {
                        Stepper(
                            "",
                            value: PopoverBindings.doubleBinding(
                                text: $valueLossWeightText,
                                fallback: 1.0,
                                format: "%.2f"
                            ),
                            in: 0.0...20.0,
                            step: 0.5
                        )
                    }
                    PopoverRow(
                        label: "Value label smoothing:",
                        text: $valueLabelSmoothingText,
                        error: valueLabelSmoothingError,
                        placeholder: "0.000",
                        hint: "ε (W/D/L CE)",
                        info: { ValueLabelSmoothingInfoButton() }
                    ) {
                        Stepper(
                            "",
                            value: PopoverBindings.doubleBinding(
                                text: $valueLabelSmoothingText,
                                fallback: 0.0,
                                format: "%.3f"
                            ),
                            in: 0.0...0.5,
                            step: 0.05
                        )
                    }
                }
                .padding(.leading, 20)
            }

            Divider()

            // Standalone — draw penalty, training batch size.
            PopoverRow(
                label: "Draw penalty:",
                text: $drawPenaltyText,
                error: drawPenaltyError,
                placeholder: "0.100"
            ) {
                Stepper(
                    "",
                    value: PopoverBindings.doubleBinding(
                        text: $drawPenaltyText,
                        fallback: 0.1,
                        format: "%.3f"
                    ),
                    in: -1.0...1.0,
                    step: 0.05
                )
            }
            PopoverRow(
                label: "Training batch size:",
                text: $trainingBatchSizeText,
                error: trainingBatchSizeError,
                placeholder: "4096"
            ) {
                Stepper(
                    "",
                    value: PopoverBindings.intBinding(text: $trainingBatchSizeText, fallback: 4096),
                    in: 32...32_768,
                    step: 256
                )
            }
        }
    }

    /// Multiply the current LR text by `factor` (`√10` for `+`,
    /// `1/√10` for `-`), clamp into `TrainingParameters` range
    /// `[1e-7, 1.0]`, and write back. Half-decade log ladder so
    /// two presses move exactly one order of magnitude.
    private func stepLRBy(factor: Double) {
        let trimmed = lrText.trimmingCharacters(in: .whitespaces)
        let current = Double(trimmed) ?? 5e-5
        let next = max(1e-7, min(1.0, current * factor))
        lrText = String(format: "%.2e", next)
    }

}

// MARK: - Self Play tab

private struct SelfPlayTab: View {
    @Binding var selfPlayConcurrencyText: String
    @Binding var selfPlayStartTauText: String
    @Binding var selfPlayDecayPerPlyText: String
    @Binding var selfPlayFloorTauText: String
    @Binding var selfPlayDrawKeepFractionText: String
    @Binding var selfPlayMaxPliesPerGameText: String

    let selfPlayConcurrencyError: Bool
    let selfPlayStartTauError: Bool
    let selfPlayDecayPerPlyError: Bool
    let selfPlayFloorTauError: Bool
    let selfPlayDrawKeepFractionError: Bool
    let selfPlayMaxPliesPerGameError: Bool

    /// Live-propagate handler. Fires on every edit-text change that
    /// parses to a valid `[0, 1]` Double — writes through to
    /// `TrainingParameters.shared.selfPlayDrawKeepFraction`
    /// immediately so the running slot driver picks up the new
    /// value at the next game-end. Cancel on the popover reverts
    /// via the model's stash; Save updates the stash to the
    /// committed value so a subsequent Cancel is a no-op.
    let onLiveSelfPlayDrawKeepFractionChange: (Double) -> Void

    /// Live-propagate handler for max-plies. Each edit flows
    /// through to `TrainingParameters.shared.selfPlayMaxPliesPerGame`; the
    /// next self-play game started by any worker slot reads the
    /// new value at game start. Cancel reverts via the model's
    /// stash; Save updates the stash.
    let onLiveMaxPliesPerGameChange: (Int) -> Void

    /// Live snapshot of the parallel-worker stats box; drives the
    /// "Emitted games" readout's W/L/D + plies-per-hour rows. `nil`
    /// before the first heartbeat tick (e.g. popover opened during
    /// app-launch before Play-and-Train has started). The Self Play
    /// tab tolerates `nil` by rendering dashes in the readout cells.
    let parallelStats: ParallelWorkerStatsBox.Snapshot?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Self Play")
                    .font(.subheadline.weight(.semibold))
                PopoverRow(
                    label: "Concurrency:",
                    text: $selfPlayConcurrencyText,
                    error: selfPlayConcurrencyError,
                    placeholder: "8"
                ) {
                    Stepper(
                        "",
                        value: PopoverBindings.intBinding(text: $selfPlayConcurrencyText, fallback: 8),
                        in: 1...256,
                        step: 1
                    )
                }
            }

            Divider()

            // Self-play temperature schedule. Same layout as the
            // arena tau section, including the live "reached at
            // ply N" hint computed from whatever the user has
            // typed so far.
            VStack(alignment: .leading, spacing: 6) {
                Text("Self-play temperature (τ)")
                    .font(.subheadline.weight(.semibold))
                PopoverRow(
                    label: "Start of game:",
                    text: $selfPlayStartTauText,
                    error: selfPlayStartTauError,
                    placeholder: "1.00",
                    hint: nil
                ) {
                    Stepper(
                        "",
                        value: PopoverBindings.doubleBinding(
                            text: $selfPlayStartTauText,
                            fallback: 1.0,
                            format: "%.2f"
                        ),
                        in: 0.05...5.0,
                        step: 0.05
                    )
                }
                PopoverRow(
                    label: "Decay:",
                    text: $selfPlayDecayPerPlyText,
                    error: selfPlayDecayPerPlyError,
                    placeholder: "0.030",
                    hint: "per ply"
                ) {
                    // Finer step than the other τ fields — at the
                    // 0.0–1.0 range, 0.05/ply would decay a starting τ
                    // of 1.0 to floor in 20 plies and miss anything in
                    // between. 0.005 lets the user dial in the slow
                    // decay (~0.007 is the current working value)
                    // without round-tripping through the text field.
                    Stepper(
                        "",
                        value: PopoverBindings.doubleBinding(
                            text: $selfPlayDecayPerPlyText,
                            fallback: 0.03,
                            format: "%.3f"
                        ),
                        in: 0.0...1.0,
                        step: 0.005
                    )
                }
                PopoverRow(
                    label: "Floor:",
                    text: $selfPlayFloorTauText,
                    error: selfPlayFloorTauError,
                    placeholder: "0.40",
                    hint: tauReachedAtHint
                ) {
                    Stepper(
                        "",
                        value: PopoverBindings.doubleBinding(
                            text: $selfPlayFloorTauText,
                            fallback: 0.40,
                            format: "%.2f"
                        ),
                        in: 0.05...5.0,
                        step: 0.05
                    )
                }
                // Soft advisory: the popover validates start/floor in
                // [0.01, 5.0] but values that pass validation can
                // still produce near-uniform sampling at every ply
                // (e.g. floor=5.0 keeps softmax flat regardless of
                // decay). Warn when either the steady-state floor or
                // the post-20-ply effective tau is high enough that
                // the network's logit advantage is largely lost.
                // Decay here is additive per ply (not multiplicative),
                // so effective tau at ply N is
                //     max(start - N*decay, floor).
                if showsTauWarning {
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                        Text(
                            "Sampling will be near-uniform: τ is high enough that "
                                + "the network's logit advantage will be largely lost. "
                                + "Consider lowering Floor or increasing Decay."
                        )
                        .foregroundStyle(.red)
                        .font(.system(size: 12))
                        .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.top, 4)
                }
            }

            Divider()

            // Per-game draw-keep filter. Live-propagated via
            // `onLiveSelfPlayDrawKeepFractionChange` so that mid-
            // edit changes affect the next completed game on every
            // worker slot without waiting for Save; Cancel reverts
            // through the model's `originalSelfPlayDrawKeepFraction`
            // stash captured at popover open.
            VStack(alignment: .leading, spacing: 6) {
                Text("Emitted games")
                    .font(.subheadline.weight(.semibold))
                // Max plies per game — hard ceiling on self-play game
                // length. Games hitting this cap are dropped (not
                // emitted) and counted as "Drop" in the Played row of
                // the Live snapshot below. Live-propagated to
                // `TrainingParameters.shared.selfPlayMaxPliesPerGame`; each
                // slot reads it at the start of its next game.
                PopoverRow(
                    label: "Max plies per game:",
                    text: $selfPlayMaxPliesPerGameText,
                    error: selfPlayMaxPliesPerGameError,
                    placeholder: "1000",
                    hint: maxPliesHint
                ) {
                    Stepper(
                        "",
                        value: liveIntBinding(
                            text: $selfPlayMaxPliesPerGameText,
                            fallback: 1000,
                            onChange: onLiveMaxPliesPerGameChange
                        ),
                        in: 25...1000,
                        step: 25
                    )
                }
                .onChange(of: selfPlayMaxPliesPerGameText) { _, newValue in
                    let trimmed = newValue.trimmingCharacters(in: .whitespaces)
                    if trimmed.isEmpty {
                        onLiveMaxPliesPerGameChange(1000)
                    } else if let n = Int(trimmed), n >= 25, n <= 1000 {
                        onLiveMaxPliesPerGameChange(n)
                    }
                }
                PopoverRow(
                    label: "Draw keep fraction:",
                    text: $selfPlayDrawKeepFractionText,
                    error: selfPlayDrawKeepFractionError,
                    placeholder: "1.00",
                    hint: drawKeepHint
                ) {
                    // Live-propagated stepper: each click flows through
                    // `onLiveSelfPlayDrawKeepFractionChange` (sibling
                    // wires direct text edits via the `.onChange(of:)`
                    // handler below), so the slot driver picks up the
                    // new keep-fraction on the next completed game
                    // without waiting for Save.
                    Stepper(
                        "",
                        value: liveDoubleBinding(
                            text: $selfPlayDrawKeepFractionText,
                            fallback: 1.0,
                            format: "%.2f",
                            onChange: onLiveSelfPlayDrawKeepFractionChange
                        ),
                        in: 0.0...1.0,
                        step: 0.05
                    )
                }
                .onChange(of: selfPlayDrawKeepFractionText) { _, newValue in
                    drawKeepHint = Self.makeDrawKeepHint(from: newValue)
                    let trimmed = newValue.trimmingCharacters(in: .whitespaces)
                    if trimmed.isEmpty {
                        onLiveSelfPlayDrawKeepFractionChange(1.0)
                    } else if let v = Double(trimmed),
                              v >= 0.0, v <= 1.0, v.isFinite {
                        onLiveSelfPlayDrawKeepFractionChange(v)
                    }
                }
                liveEmittedGamesReadout
            }
        }
        .onAppear {
            // Seed the cached hint from whatever text the model
            // currently holds. Subsequent edits update it via the
            // `.onChange(of: selfPlayDrawKeepFractionText)` above.
            drawKeepHint = Self.makeDrawKeepHint(from: selfPlayDrawKeepFractionText)
        }
    }

    /// Plain-English hint for the current max-plies-per-game cap.
    /// At 1000 the cap is effectively disabled (no legitimate
    /// chess game reaches 1000 plies — the 50-move rule and 3-fold
    /// repetition end any sane game well before that). Lower values
    /// say "drop games hitting this many plies."
    private var maxPliesHint: String {
        guard let n = Int(selfPlayMaxPliesPerGameText.trimmingCharacters(in: .whitespaces)),
              n >= 25, n <= 1000 else {
            return "25–1000"
        }
        if n >= 1000 {
            return "effectively disabled"
        }
        return "drop games at \(n) plies"
    }

    /// Stepper binding that writes its new Int through `onChange`
    /// (in addition to updating the text field). Mirrors the
    /// `liveDoubleBinding` helper used by the keep-fraction stepper —
    /// each click of the stepper arrow flows through to
    /// `TrainingParameters.shared.selfPlayMaxPliesPerGame` immediately.
    private func liveIntBinding(
        text: Binding<String>,
        fallback: Int,
        onChange: @escaping (Int) -> Void
    ) -> Binding<Int> {
        Binding<Int>(
            get: {
                Int(text.wrappedValue.trimmingCharacters(in: .whitespaces)) ?? fallback
            },
            set: { newValue in
                text.wrappedValue = String(newValue)
                onChange(newValue)
            }
        )
    }

    /// Plain-English explanation of the current keep-fraction value,
    /// shown next to the text field so the operator can read the
    /// effect of an edit without having to reason about the math.
    /// Stored as `@State` rather than a computed-on-every-body
    /// getter: kept in sync with `selfPlayDrawKeepFractionText` via
    /// the body's `.onAppear` + `.onChange(of:)` hooks so SwiftUI
    /// only re-parses on actual text edits, not on every body
    /// re-evaluation triggered by sibling state changes.
    @State private var drawKeepHint: String = "0.0–1.0"

    /// Pure helper that derives the plain-English hint from the
    /// raw text. Called from `.onAppear` (initial state) and
    /// `.onChange(of: selfPlayDrawKeepFractionText)` (live edits).
    private static func makeDrawKeepHint(from text: String) -> String {
        guard let v = Double(text.trimmingCharacters(in: .whitespaces)),
              v.isFinite, v >= 0.0, v <= 1.0 else {
            return "0.0–1.0"
        }
        if v >= 0.999 {
            return "keep every drawn game"
        }
        if v <= 0.001 {
            return "drop every drawn game (decisive only)"
        }
        let pctDrop = Int(((1.0 - v) * 100).rounded())
        return "drop ~\(pctDrop)% of drawn games"
    }

    /// Heuristic predicate for the soft tau warning. Fires when either
    ///   - the floor itself is ≥ 2.0 (steady-state too flat to recover
    ///     even with maximal decay), or
    ///   - effective τ after 20 plies of additive decay
    ///     (`max(start − 20·decay, floor)`) is still ≥ 2.0.
    /// 2.0 is a soft threshold: at τ ≈ 2 a ±2-nat logit gap (a fairly
    ///  confident network call) is compressed to ~e^1 ≈ 2.7× odds, which
    /// is most of the network's signal washed out.
    private var showsTauWarning: Bool {
        guard let start = Double(selfPlayStartTauText), start.isFinite,
              let floor = Double(selfPlayFloorTauText), floor.isFinite,
              let decay = Double(selfPlayDecayPerPlyText), decay.isFinite,
              decay >= 0 else {
            return false
        }
        if floor >= 2.0 { return true }
        let after20 = max(start - 20.0 * decay, floor)
        if after20 >= 2.0 { return true }
        return false
    }

    /// Live "reached at N plies" hint computed from the current
    /// edit-text values. Mirrors the helper in `ArenaSettingsPopover`.
    private var tauReachedAtHint: String {
        guard let start = Double(selfPlayStartTauText), start > 0,
              let decay = Double(selfPlayDecayPerPlyText), decay >= 0,
              let floor = Double(selfPlayFloorTauText), floor > 0 else {
            return "(reached at —)"
        }
        guard decay > 0 else { return "(no decay; floor unreached)" }
        guard floor < start else { return "(reached at ply 0)" }
        let plies = Int(((start - floor) / decay).rounded(.up))
        return "(reached at ply \(plies))"
    }

    /// Live "played vs emitted" readout under the Draw keep fraction
    /// field. Two columns ("Played" / "Emitted") × two rows
    /// (plies/hour, W/D/L breakdown). The 1-minute rolling window
    /// (`ParallelWorkerStatsBox.recentWindow`) drives the plies-per-
    /// hour numbers. Outcome shares come straight from the lifetime
    /// totals — decisive games are always kept by the keep-fraction
    /// filter, so emitted W/L equal played W/L and the only
    /// difference between the two columns is the drawn-game count
    /// (`emittedGames - W - L`).
    @ViewBuilder
    private var liveEmittedGamesReadout: some View {
        let s = parallelStats
        let dash = "—"
        let hasStats = (s?.selfPlayGames ?? 0) > 0
        // Rolling rates (plies / hour) over the box's `recentWindow`
        // (1-minute) window. Stays at "—" while `recentWindowSeconds`
        // is 0 (first ~minute of a session before the window fills).
        let recentWindow = s?.recentWindowSeconds ?? 0
        let playedRate: Double = hasStats && recentWindow > 0
            ? Double(s?.recentMoves ?? 0) / recentWindow * 3600
            : 0
        let emittedRate: Double = hasStats && recentWindow > 0
            ? Double(s?.recentEmittedPositions ?? 0) / recentWindow * 3600
            : 0
        // Lifetime totals for the "games:" row.
        let playedTotal = (s?.selfPlayGames ?? 0)
        let emittedTotal = (s?.emittedGames ?? 0)

        // Rolling-window per-outcome counts for the W/D/L row. These
        // are the only W/D/L numbers worth showing here: lifetime
        // per-outcome counts are dominated by whatever happened
        // before the operator last touched `selfPlayDrawKeepFraction`,
        // so the Played vs Emitted columns look identical at any
        // realistic display precision until the filter has been
        // running long enough to move lifetime totals by ≥ 0.1%.
        // The 1-minute rolling window matches the `plies / hour (1m)`
        // row's denominator so the two rows describe the same
        // recent slice of self-play.
        let recentPlayedTotal = s?.recentGames ?? 0
        let recentEmittedTotal = s?.recentEmittedGames ?? 0
        let recentPlayedW = s?.recentWhiteCheckmates ?? 0
        let recentPlayedL = s?.recentBlackCheckmates ?? 0
        let recentPlayedD = s?.recentDraws ?? 0
        // Dropped count only meaningful on the Played side — dropped
        // games never reach the emit stage, so the Emitted column
        // stays a 3-way W/D/L %. Played row sums W/D/L/Drop to 100%.
        let recentPlayedDrop = s?.recentMaxPliesDropped ?? 0
        let recentEmittedW = s?.recentEmittedWhiteCheckmates ?? 0
        let recentEmittedL = s?.recentEmittedBlackCheckmates ?? 0
        let recentEmittedD = s?.recentEmittedDraws ?? 0

        VStack(alignment: .leading, spacing: 2) {
            Text("Live snapshot")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.top, 6)
                .padding(.bottom, 2)
            HStack(spacing: 8) {
                Text("")
                    .frame(width: 140, alignment: .trailing)
                Text("Played")
                    .frame(width: 165, alignment: .leading)
                Text("Emitted")
                    .frame(width: 165, alignment: .leading)
                Spacer()
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            twoColRow(
                label: "plies / hour (1m):",
                playedValue: hasStats && recentWindow > 0
                    ? Self.numberString(Int(playedRate.rounded()))
                    : dash,
                emittedValue: hasStats && recentWindow > 0
                    ? Self.numberString(Int(emittedRate.rounded()))
                    : dash
            )
            // Game-total counts. The Played vs Emitted gap is exactly
            // what the keep-fraction filter has dropped — the
            // headline number for "is my filter doing what I want."
            // The W/D/L breakdown lives one row down as percentages.
            twoColRow(
                label: "games:",
                playedValue: hasStats ? Self.numberString(playedTotal) : dash,
                emittedValue: hasStats ? Self.numberString(emittedTotal) : dash
            )
            // Dropped-for-max-plies on its own line so the W/D/L row
            // below can keep its three-way layout. The four percentages
            // across these two rows still sum to 100 — both rows use
            // `recentPlayedTotal` (played including dropped) as the
            // denominator. The Emitted column is "—" on the dropped
            // line because dropped games never reach the emit stage.
            twoColRow(
                label: "dropped - max plies (1m):",
                playedValue: hasStats && recentPlayedTotal > 0
                    ? String(format: "%.1f%%", Double(recentPlayedDrop) / Double(recentPlayedTotal) * 100)
                    : dash,
                emittedValue: dash
            )
            twoColRow(
                label: "W / D / L (1m):",
                playedValue: hasStats && recentPlayedTotal > 0
                    ? Self.pctTriple(w: recentPlayedW, d: recentPlayedD, l: recentPlayedL, total: recentPlayedTotal)
                    : dash,
                emittedValue: hasStats && recentEmittedTotal > 0
                    ? Self.pctTriple(w: recentEmittedW, d: recentEmittedD, l: recentEmittedL, total: recentEmittedTotal)
                    : dash
            )
        }
    }

    /// One right-aligned label + two left-aligned value cells, same
    /// shape as the Replay tab's `twoColRow` helper but with
    /// "Played" / "Emitted" columns. Typography matches the
    /// `SelfPlayStatsCard` / `ResultsCard` and the upper status bar's
    /// `StatusBarCell` (caption2-secondary labels + callout-monospaced-
    /// semibold values).
    @ViewBuilder
    private func twoColRow(
        label: String,
        playedValue: String,
        emittedValue: String
    ) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 140, alignment: .trailing)
            Text(playedValue)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: 165, alignment: .leading)
            Text(emittedValue)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: 165, alignment: .leading)
            Spacer()
        }
    }

    private static func numberString(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? String(n)
    }

    /// Format a W/D/L triple as "ww.w% / dd.d% / ll.l%" given the
    /// counts and a total. Returns the formatted string; caller is
    /// responsible for the `hasStats` guard.
    private static func pctTriple(w: Int, d: Int, l: Int, total: Int) -> String {
        guard total > 0 else { return "—" }
        let den = Double(total)
        let wp = Double(w) / den * 100
        let dp = Double(d) / den * 100
        let lp = Double(l) / den * 100
        return String(format: "%.1f%% / %.1f%% / %.1f%%", wp, dp, lp)
    }


    /// Stepper-binding for the live-update Draw keep fraction field.
    /// Mirrors the helper in `ReplayTab` — kept tab-local rather than
    /// shared on `PopoverBindings` because the two tabs each carry
    /// their own private set of helpers and a project-wide refactor
    /// of those isn't in scope here. Routes the post-step value
    /// through `onChange` so the parent's `applyLive…` writes through
    /// to `TrainingParameters.shared` immediately; direct text edits
    /// hit the same `onChange` via the surrounding
    /// `.onChange(of: textBinding)` modifier at the call site.
    private func liveDoubleBinding(
        text: Binding<String>,
        fallback: Double,
        format: String,
        onChange: @escaping (Double) -> Void
    ) -> Binding<Double> {
        Binding<Double>(
            get: {
                let trimmed = text.wrappedValue.trimmingCharacters(in: .whitespaces)
                return Double(trimmed) ?? fallback
            },
            set: { newValue in
                text.wrappedValue = String(format: format, newValue)
                onChange(newValue)
            }
        )
    }
}

// MARK: - Replay tab

private struct ReplayTab: View {
    @Binding var replayBufferCapacityText: String
    @Binding var replayBufferMinPositionsText: String
    @Binding var replayRatioTargetText: String
    @Binding var replaySelfPlayDelayText: String
    @Binding var replayTrainingStepDelayText: String
    @Binding var replayRatioAutoAdjust: Bool
    @Binding var maxPliesFromAnyOneGameText: String
    @Binding var targetSampledGameLengthPliesText: String
    @Binding var maxDrawPercentPerBatchText: String

    let replayBufferCapacityError: Bool
    let replayBufferMinPositionsError: Bool
    let replayRatioTargetError: Bool
    let replaySelfPlayDelayError: Bool
    let replayTrainingStepDelayError: Bool
    let maxPliesFromAnyOneGameError: Bool
    let targetSampledGameLengthPliesError: Bool
    let maxDrawPercentPerBatchError: Bool

    let replayRatioCurrent: Double?
    let replayRatioComputedDelayMs: Int?
    let replayRatioComputedSelfPlayDelayMs: Int?
    let bytesPerPosition: Int
    let bufferComposition: ReplayBuffer.CompositionSnapshot?
    let lastSamplingResult: ReplayBuffer.SamplingResult?

    let onLiveReplayRatioTargetChange: (Double) -> Void
    let onLiveSelfPlayDelayChange: (Int) -> Void
    let onLiveTrainingStepDelayChange: (Int) -> Void
    let onLiveReplayRatioAutoAdjustChange: (Bool) -> Void
    let onLiveMaxPliesFromAnyOneGameChange: (Int) -> Void
    let onLiveTargetSampledGameLengthPliesChange: (Int) -> Void
    let onLiveMaxDrawPercentPerBatchChange: (Int) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // --- Replay buffer ---
            VStack(alignment: .leading, spacing: 6) {
                Text("Replay buffer")
                    .font(.subheadline.weight(.semibold))
                PopoverRow(
                    label: "Capacity (plies):",
                    text: $replayBufferCapacityText,
                    error: replayBufferCapacityError,
                    placeholder: "1000000",
                    hint: capacityGBHint
                ) {
                    Stepper(
                        "",
                        value: PopoverBindings.intBinding(
                            text: $replayBufferCapacityText,
                            fallback: 1_000_000
                        ),
                        in: 1024...100_000_000,
                        step: 100_000
                    )
                }
                PopoverRow(
                    label: "Pre-train fill:",
                    text: $replayBufferMinPositionsText,
                    error: replayBufferMinPositionsError,
                    placeholder: "50000",
                    hint: preTrainFillPctHint
                ) {
                    Stepper(
                        "",
                        value: PopoverBindings.intBinding(
                            text: $replayBufferMinPositionsText,
                            fallback: 50_000
                        ),
                        in: 0...100_000_000,
                        step: 10_000
                    )
                }
            }

            Divider()

            // --- Replay ratio control ---
            //
            // Live current-ratio readout; three editable fields
            // (self-play delay, train step delay, auto checkbox)
            // that propagate to the live trainingParams /
            // ReplayRatioController immediately so the user can
            // watch their change take effect on the same panel.
            // On Cancel, the parent restores the values it stashed
            // when this popover opened.
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Replay ratio control")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    LiveRatioBadge(current: replayRatioCurrent, targetText: replayRatioTargetText)
                }
                let autoOn = replayRatioAutoAdjust
                PopoverRow(
                    label: "Target ratio:",
                    text: $replayRatioTargetText,
                    error: replayRatioTargetError,
                    placeholder: "1.10",
                    hint: nil
                ) {
                    Stepper(
                        "",
                        value: liveDoubleBinding(
                            text: $replayRatioTargetText,
                            fallback: 1.10,
                            format: "%.2f",
                            onChange: onLiveReplayRatioTargetChange
                        ),
                        in: 0.1...5.0,
                        step: 0.05
                    )
                }
                .onChange(of: replayRatioTargetText) { _, newValue in
                    if let v = Double(newValue.trimmingCharacters(in: .whitespaces)),
                       v >= 0.1, v <= 5.0, v.isFinite {
                        onLiveReplayRatioTargetChange(v)
                    }
                }
                if autoOn {
                    liveDelayRow(
                        label: "Self-play delay:",
                        valueMs: replayRatioComputedSelfPlayDelayMs
                    )
                    liveDelayRow(
                        label: "Train step delay:",
                        valueMs: replayRatioComputedDelayMs
                    )
                } else {
                    PopoverRow(
                        label: "Self-play delay:",
                        text: $replaySelfPlayDelayText,
                        error: replaySelfPlayDelayError,
                        placeholder: "0",
                        hint: "ms"
                    ) {
                        Stepper(
                            "",
                            value: liveIntBinding(
                                text: $replaySelfPlayDelayText,
                                fallback: 0,
                                onChange: onLiveSelfPlayDelayChange
                            ),
                            in: 0...3000,
                            step: 5
                        )
                    }
                    .onChange(of: replaySelfPlayDelayText) { _, newValue in
                        if let v = Int(newValue.trimmingCharacters(in: .whitespaces)),
                           v >= 0, v <= 3000 {
                            onLiveSelfPlayDelayChange(v)
                        }
                    }
                    PopoverRow(
                        label: "Train step delay:",
                        text: $replayTrainingStepDelayText,
                        error: replayTrainingStepDelayError,
                        placeholder: "0",
                        hint: "ms"
                    ) {
                        Stepper(
                            "",
                            value: liveIntBinding(
                                text: $replayTrainingStepDelayText,
                                fallback: 0,
                                onChange: onLiveTrainingStepDelayChange
                            ),
                            in: 0...10_000,
                            step: 5
                        )
                    }
                    .onChange(of: replayTrainingStepDelayText) { _, newValue in
                        if let v = Int(newValue.trimmingCharacters(in: .whitespaces)),
                           v >= 0, v <= 10_000 {
                            onLiveTrainingStepDelayChange(v)
                        }
                    }
                }
                HStack(spacing: 8) {
                    Text("")
                        .frame(width: 160, alignment: .trailing)
                    Toggle("Enable automatic control", isOn: Binding(
                        get: { replayRatioAutoAdjust },
                        set: { newValue in
                            replayRatioAutoAdjust = newValue
                            onLiveReplayRatioAutoAdjustChange(newValue)
                        }
                    ))
                    .toggleStyle(.checkbox)
                    Spacer()
                }
            }

            Divider()

            // --- Replay sampling ---
            //
            // Per-training-batch composition constraints, plus the
            // current (pre-constraint) composition of the resident
            // buffer. The post-constraint distribution of an actual
            // sampled batch shows up on the [BATCH-STATS] log line.
            VStack(alignment: .leading, spacing: 6) {
                Text("Replay sampling")
                    .font(.subheadline.weight(.semibold))
                // Three sampling-constraint fields are live-propagated:
                // each stepper write *and* each direct text edit
                // (covered by the trailing `.onChange(of:)` handlers)
                // flows through `onLive…Change`, which writes to
                // `TrainingParameters.shared` immediately so the
                // `Composition` readout below can refresh on the next
                // heartbeat. Cancel / outside-click reverts to the
                // snapshot taken at popover open (see
                // `TrainingSettingsPopoverModel.cancel`).
                PopoverRow(
                    label: "Max plies per game:",
                    text: $maxPliesFromAnyOneGameText,
                    error: maxPliesFromAnyOneGameError,
                    placeholder: "10",
                    hint: "plies from any 1 game"
                ) {
                    Stepper(
                        "",
                        value: liveIntBinding(
                            text: $maxPliesFromAnyOneGameText,
                            fallback: 10,
                            onChange: onLiveMaxPliesFromAnyOneGameChange
                        ),
                        in: 1...400,
                        step: 1
                    )
                }
                .onChange(of: maxPliesFromAnyOneGameText) { _, newValue in
                    // Empty (or whitespace-only) text is treated as
                    // the parameter's default and live-propagated as
                    // such — the placeholder text shows that default
                    // so clearing the field reads visually as "use
                    // the default" and the live readout below
                    // matches what's about to be saved.
                    let trimmed = newValue.trimmingCharacters(in: .whitespaces)
                    if trimmed.isEmpty {
                        onLiveMaxPliesFromAnyOneGameChange(10)
                    } else if let v = Int(trimmed), v >= 1, v <= 400 {
                        onLiveMaxPliesFromAnyOneGameChange(v)
                    }
                }
                PopoverRow(
                    label: "Target avg game plies:",
                    text: $targetSampledGameLengthPliesText,
                    error: targetSampledGameLengthPliesError,
                    placeholder: "0",
                    hint: "plies"
                ) {
                    Stepper(
                        "",
                        value: liveIntBinding(
                            text: $targetSampledGameLengthPliesText,
                            fallback: 0,
                            onChange: onLiveTargetSampledGameLengthPliesChange
                        ),
                        in: 0...10_000,
                        step: 10
                    )
                }
                .onChange(of: targetSampledGameLengthPliesText) { _, newValue in
                    let trimmed = newValue.trimmingCharacters(in: .whitespaces)
                    if trimmed.isEmpty {
                        onLiveTargetSampledGameLengthPliesChange(0)
                    } else if let v = Int(trimmed), v >= 0, v <= 10_000 {
                        onLiveTargetSampledGameLengthPliesChange(v)
                    }
                }
                PopoverRow(
                    label: "Max draws % per batch:",
                    text: $maxDrawPercentPerBatchText,
                    error: maxDrawPercentPerBatchError,
                    placeholder: "100",
                    hint: maxDrawPercentHint
                ) {
                    Stepper(
                        "",
                        value: liveIntBinding(
                            text: $maxDrawPercentPerBatchText,
                            fallback: 100,
                            onChange: onLiveMaxDrawPercentPerBatchChange
                        ),
                        in: 0...100,
                        step: 5
                    )
                }
                .onChange(of: maxDrawPercentPerBatchText) { _, newValue in
                    let trimmed = newValue.trimmingCharacters(in: .whitespaces)
                    if trimmed.isEmpty {
                        onLiveMaxDrawPercentPerBatchChange(100)
                    } else if let v = Int(trimmed), v >= 0, v <= 100 {
                        onLiveMaxDrawPercentPerBatchChange(v)
                    }
                }
                replayCompositionReadout
            }
        }
    }

    /// Pre-constraint resident-buffer composition next to the post-
    /// constraint composition of the most-recent sampled training batch.
    /// Layout: a right-aligned label column + two left-aligned value
    /// columns ("Buffer" | "Last batch"). Rows that have no counterpart
    /// in one of the columns render `—` in that cell so the row spacing
    /// stays uniform.
    ///
    /// Buffer-side facts (left column):
    /// - "games": estimated resident game count from the length histogram
    ///   (worker game IDs can collide after resume, so the exact distinct
    ///   ID count is only used internally for the per-batch cap).
    /// - "avg game length": simple per-game mean (each resident game
    ///   contributes once). No batch analog.
    /// - "avg sampled game length": position-weighted mean
    ///   `E[L²]/E[L]` — the expected game length of the game a
    ///   randomly-drawn position came from. The gap between the two
    ///   is the buffer's game-length dispersion.
    /// - W / D / L: position-share of each outcome in the resident set.
    /// - "decisive split": within-decisive `+z / −z` share; coloured
    ///   orange when it slips outside [45, 55] as a sign-assignment
    ///   smell.
    ///
    /// Batch-side facts (right column, post-constraint achievement):
    /// - "games": `distinctGamesInBatch / batchSize`.
    /// - "avg sampled game length": Σ game-length / batch size.
    /// - "samples / game (avg)": batchSize / distinctGamesInBatch.
    /// - "samples / game (max)": `maxPerGame` achieved this batch.
    /// - W / D / L: achieved outcome shares (also coloured by
    ///   sign skew within decisive).
    @ViewBuilder
    private var replayCompositionReadout: some View {
        // Always render the same row tree regardless of whether the
        // inputs are yet populated — empty cells collapse to "—".
        // Keeps the SwiftUI view tree shape stable across the nil-to-
        // populated transition.
        let c = bufferComposition
        let sr = lastSamplingResult
        let dash = "—"

        // --- Buffer side ---
        let bufHas = (c?.storedCount ?? 0) > 0
        let bufDecisive = (c?.winPositions ?? 0) + (c?.lossPositions ?? 0)
        let bufPlusZ = bufDecisive > 0 ? Double(c?.winPositions ?? 0) / Double(bufDecisive) * 100 : 0.0
        let bufMinusZ = bufDecisive > 0 ? Double(c?.lossPositions ?? 0) / Double(bufDecisive) * 100 : 0.0
        let bufSkewed = bufDecisive > 0 && (bufPlusZ < 45 || bufPlusZ > 55)

        // --- Batch side ---
        let batchHas = (sr?.batchSize ?? 0) > 0
        let batchDecisive = (sr?.achievedWinCount ?? 0) + (sr?.achievedLossCount ?? 0)
        let batchPlusZ = batchDecisive > 0 ? Double(sr?.achievedWinCount ?? 0) / Double(batchDecisive) * 100 : 0.0
        let batchMinusZ = batchDecisive > 0 ? Double(sr?.achievedLossCount ?? 0) / Double(batchDecisive) * 100 : 0.0
        let batchSkewed = batchDecisive > 0 && (batchPlusZ < 45 || batchPlusZ > 55)

        // --- Constraint-vs-achievement indicators ---
        // `wasConstrainedPath == false` means the no-op fast path was
        // taken (no K cap, no D cap, no length tilt active) — no point
        // showing constraint context in that case.
        let constraintsActive = sr?.wasConstrainedPath ?? false

        // Draw-share gap: paint the merged W/D/L value cell red
        // whenever the constraint is active and the achieved-vs-
        // requested gap exceeds the same slop the sampler uses to set
        // `wasDegraded`. Red signals overshoot (buffer composition +
        // K cap forced more draws into the batch than the operator
        // asked for) or undershoot (not enough resident draws to
        // support the cap). The configured cap stays visible in the
        // "Max draws % per batch:" field above, so we don't duplicate
        // it inline (would push the row over the popover's 540 pt
        // width budget).
        let drawCountSlop = max(1, (sr?.batchSize ?? 0) / 100)
        let drawCapDegraded = constraintsActive
            && abs((sr?.achievedDrawCount ?? 0) - (sr?.requestedDrawCount ?? 0)) > drawCountSlop

        // Attempt-budget hit: the loop fell through to the
        // unconstrained uniform fill — all three caps were silently
        // dropped for the remainder of the batch. After the K-aware
        // sizing fix this should essentially never fire; if it does,
        // either the buffer is in a jointly-pathological state (very
        // small K with too few resident games on both sides) or
        // there's a regression.
        let budgetHit = sr?.attemptBudgetHit ?? false

        // Pre-compute the per-letter percent triples for the merged
        // "avg W/D/L" row. Buffer = position-weighted resident shares
        // (winFraction/drawFraction/lossFraction). Batch = achieved
        // per-letter shares — the per-batch draw cap context is
        // surfaced via the red value color when achieved drifts off
        // requested (see `drawCapDegraded` above). Pre-computed via the
        // private helpers `Self.formatBufferWdl(_:)` and
        // `Self.formatBatchWdl(_:)` so the expressions inside this
        // `@ViewBuilder` body remain single expressions (an
        // `if`/`else`-driven `let` here would be mis-parsed as a
        // `View` by ViewBuilder and fail to compile).
        let bufWdlText = Self.formatBufferWdl(c, dash: dash)
        let batchWdlText = Self.formatBatchWdl(sr, dash: dash)

        VStack(alignment: .leading, spacing: 2) {
            Text("Composition")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.bottom, 2)
            // Header strip: two column titles aligned with the value
            // cells below. Empty label cell on the left so the
            // titles sit directly above their data.
            HStack(spacing: 8) {
                Text("")
                    .frame(width: 140, alignment: .trailing)
                Text("Buffer")
                    .frame(width: 200, alignment: .leading)
                Text("Last batch")
                    .frame(width: 200, alignment: .leading)
                Spacer()
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            // First row: storage size context. Buffer = current resident
            // position count (the W/D/L percentages below are share-of-
            // these). Batch = the trainer's batch size (positions
            // sampled per SGD step).
            twoColRow(
                label: "plies:",
                bufferValue: bufHas ? numberString(c?.storedCount ?? 0) : dash,
                batchValue: batchHas ? numberString(sr?.batchSize ?? 0) : dash
            )
            // Both columns are counts of distinct games — the row label
            // is renamed accordingly so the right-hand value can't be
            // misread as a fill ratio ("X of Y" looked like a partial
            // fill).
            twoColRow(
                label: "distinct games:",
                bufferValue: bufHas
                    ? numberString(Int((c?.gameWeightedResidentGameCount ?? 0).rounded()))
                    : dash,
                batchValue: batchHas
                    ? numberString(sr?.distinctGamesInBatch ?? 0)
                    : dash
            )
            // Decisive *plies* (W + L positions) on both sides — the
            // batch column is the per-position outcome histogram (W +
            // L counts), so the buffer column matches by reading the
            // resident `winPositions + lossPositions` rather than the
            // distinct decisive *game* count. Position-units on both
            // sides keeps the comparison honest (a 4096-ply batch can
            // legitimately draw thousands of decisive positions from
            // a few hundred decisive games at K plies per game; the
            // ratio is meaningless without matching units).
            twoColRow(
                label: "decisive plies:",
                bufferValue: bufHas
                    ? numberString((c?.winPositions ?? 0) + (c?.lossPositions ?? 0))
                    : dash,
                batchValue: batchHas
                    ? numberString(batchDecisive)
                    : dash
            )
            // Buffer game-mean (each resident game contributes once);
            // no batch analog (the batch is sampled positions, not
            // games).
            twoColRow(
                label: "avg game length:",
                bufferValue: c.map { String(format: "%.0f plies", $0.meanGameLengthPerGame) } ?? dash,
                batchValue: dash
            )
            // Position-weighted mean: for the buffer this is E[L²]/E[L]
            // (the expected length of the game a randomly-drawn
            // position came from); for the batch it's the mean over
            // emitted positions of the contributing game's length.
            twoColRow(
                label: "avg sampled game length:",
                bufferValue: c.map { String(format: "%.0f plies", $0.meanGameLengthPerSampledPosition) } ?? dash,
                batchValue: batchHas
                    ? String(format: "%.0f plies", sr?.achievedMeanGameLength ?? 0)
                    : dash
            )
            // For the batch only: positions / distinctGames, i.e. how
            // many plies (on average) each contributing game donated.
            // For the buffer the equivalent is "avg game length" two
            // rows up, so this row's Buffer cell is dashed rather than
            // duplicated.
            twoColRow(
                label: "avg plies / game:",
                bufferValue: dash,
                batchValue: batchHas
                    ? String(format: "%.2f", sr?.achievedMeanSamplesPerGame ?? 0)
                    : dash
            )
            // Single merged W/D/L row — replaces the prior W: / D: / L:
            // triple. Format mirrors the Self Play tab's "W / D / L %"
            // readout. Buffer side is *position-weighted* (each ply
            // counts once); the row below reports the *game-weighted*
            // draw share for direct comparison against the Self Play
            // tab's per-game W/D/L. Batch cell turns red when the
            // achieved-vs-requested gap exceeds the sampler's slop.
            twoColRow(
                label: "avg W / D / L:",
                bufferValue: bufWdlText,
                batchValue: batchWdlText,
                batchValueColor: drawCapDegraded ? .red : nil
            )
            // Game-weighted draw share for the buffer — distinct
            // resident draw games / distinct resident games. Higher
            // when draws are short relative to decisives, lower when
            // they're long. Pairs with the position-weighted row
            // above so the reader can see the bias in one glance:
            // typically draws are *longer* than decisives, so the
            // position-weighted D% above is HIGHER than this row's
            // game-weighted D%.
            twoColRow(
                label: "draws (games):",
                bufferValue: bufferGameWeightedDrawText(c, dash: dash),
                batchValue: dash
            )
            // Within-decisive split: of the W+L positions, how many were
            // wins vs losses. Healthy ≈ 50/50; orange when outside
            // [45, 55] as a sign-assignment smell. Label carries the
            // W/L distinction so values render plain.
            twoColRow(
                label: "decisive split W / L:",
                bufferValue: bufHas
                    ? String(format: "%.0f%% / %.0f%%", bufPlusZ, bufMinusZ)
                    : dash,
                bufferValueColor: bufSkewed ? .orange : nil,
                batchValue: batchHas
                    ? String(format: "%.0f%% / %.0f%%", batchPlusZ, batchMinusZ)
                    : dash,
                batchValueColor: batchSkewed ? .orange : nil
            )
            // Attempt-budget-hit banner. Always rendered (view-type
            // stable per the SwiftUI rules — the Text node is the same
            // identity whether the banner is visible or not), collapsed
            // to zero height + zero opacity when not active so it
            // leaves no visual trace. When active, padded slightly off
            // the data above so the warning has breathing room.
            HStack(alignment: .top, spacing: 6) {
                Text("⚠")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.red)
                Text("Attempt budget hit — sampler fell back to uniform; K, D and length-target caps all dropped this batch")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.top, budgetHit ? 6 : 0)
            .frame(height: budgetHit ? nil : 0)
            .opacity(budgetHit ? 1 : 0)
            .accessibilityHidden(!budgetHit)
            .accessibilityLabel("Sampler attempt budget hit, all sampling caps dropped this batch")
        }
        .padding(.top, 4)
    }

    /// One right-aligned label + two left-aligned value cells. Label
    /// column rendered in `.caption2 .secondary` to match the section
    /// title and column headers; values rendered in
    /// `.system(.callout, design: .monospaced).weight(.semibold)`
    /// to match the Self Play / Results card values + the upper
    /// status bar's `StatusBarCell`. `*ValueColor` is applied
    /// unconditionally — nil resolves to `.primary` so values read
    /// full-contrast. The view tree never branches on the color
    /// (matches the project's view-stability rule).
    @ViewBuilder
    private func twoColRow(
        label: String,
        bufferValue: String,
        bufferValueColor: Color? = nil,
        batchValue: String,
        batchValueColor: Color? = nil
    ) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 140, alignment: .trailing)
            Text(bufferValue)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(bufferValueColor ?? Color.primary)
                .lineLimit(1)
                .frame(width: 200, alignment: .leading)
            Text(batchValue)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(batchValueColor ?? Color.primary)
                .lineLimit(1)
                .frame(width: 200, alignment: .leading)
            Spacer()
        }
    }

    /// Game-weighted buffer draw share — `(distinctResidentGames -
    /// residentDecisiveGameCount) / distinctResidentGames` rendered
    /// as a percentage. Companion to the position-weighted W/D/L
    /// row: when draws are longer than decisives (the typical case),
    /// the position-weighted D% is higher than this game-weighted
    /// D%. Both numerator and denominator come from the same gameID-
    /// tracked counter (`distinctResidentGames`) rather than the
    /// histogram-derived estimate, so the percentage is internally
    /// consistent even on a resumed session where the two counters
    /// can drift.
    private func bufferGameWeightedDrawText(
        _ c: ReplayBuffer.CompositionSnapshot?,
        dash: String
    ) -> String {
        guard let c, c.distinctResidentGames > 0 else { return dash }
        let drawGames = c.distinctResidentGames - c.residentDecisiveGameCount
        let pct = Double(max(0, drawGames)) / Double(c.distinctResidentGames) * 100
        return String(format: "%.1f%%", pct)
    }

    /// Buffer-side W / D / L triple. Position-weighted resident
    /// shares; renders as `dash` when no positions are stored yet.
    private static func formatBufferWdl(
        _ c: ReplayBuffer.CompositionSnapshot?,
        dash: String
    ) -> String {
        guard let c, c.storedCount > 0 else { return dash }
        return String(
            format: "%.1f%% / %.1f%% / %.1f%%",
            c.winFraction * 100, c.drawFraction * 100, c.lossFraction * 100
        )
    }

    /// Last-batch W / D / L triple. Achieved per-letter shares of the
    /// emitted batch. The per-batch draw-cap context (configured value,
    /// achieved-vs-requested gap) is conveyed via the red value color
    /// when the constraint is active and degraded — see
    /// `drawCapDegraded` in `replayCompositionReadout`. The configured
    /// cap itself is always visible in the "Max draws % per batch:"
    /// field above, so an inline annotation would duplicate it and
    /// blow out the popover's fixed 540 pt width.
    private static func formatBatchWdl(
        _ sr: ReplayBuffer.SamplingResult?,
        dash: String
    ) -> String {
        guard let sr, sr.batchSize > 0 else { return dash }
        return String(
            format: "%.1f%% / %.1f%% / %.1f%%",
            sr.achievedWinPercent, sr.achievedDrawPercent, sr.achievedLossPercent
        )
    }

    /// Format an Int with thousands separators ("2,722" not "2722").
    private func numberString(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? String(n)
    }

    private var maxDrawPercentHint: String {
        let t = maxDrawPercentPerBatchText.trimmingCharacters(in: .whitespaces)
        if let n = Int(t), n >= 100 { return "off — no draw cap" }
        return "% per batch"
    }

    /// Live "≈ X.X GB" hint for the capacity field. Updates as the
    /// user types because it's a computed property re-evaluated on
    /// every body re-render.
    private var capacityGBHint: String {
        let trimmed = replayBufferCapacityText.trimmingCharacters(in: .whitespaces)
        guard let plies = Int(trimmed), plies > 0 else { return "" }
        let bytes = Double(plies) * Double(bytesPerPosition)
        let gb = bytes / 1_000_000_000.0
        if gb >= 100 {
            return String(format: "≈ %.0f GB", gb)
        } else if gb >= 1 {
            return String(format: "≈ %.1f GB", gb)
        } else {
            return String(format: "≈ %.0f MB", bytes / 1_000_000.0)
        }
    }

    /// Live "(N.N %)" hint for the pre-train fill field. Computed
    /// from the live capacity edit-text and pre-train edit-text so
    /// the percentage updates in real time as either is typed.
    private var preTrainFillPctHint: String {
        let capTrim = replayBufferCapacityText.trimmingCharacters(in: .whitespaces)
        let fillTrim = replayBufferMinPositionsText.trimmingCharacters(in: .whitespaces)
        guard let cap = Int(capTrim), cap > 0,
              let fill = Int(fillTrim), fill >= 0 else {
            return ""
        }
        let pct = Double(fill) / Double(cap) * 100.0
        if pct >= 10 {
            return String(format: "(%.0f %%)", pct)
        } else {
            return String(format: "(%.1f %%)", pct)
        }
    }

    /// Read-only row used for the two delay fields when
    /// auto-control is on. The auto controller drives the values
    /// directly, so the editable TextField/Stepper would just
    /// display stale typed values; instead we render a label +
    /// "<N> ms (auto)" readout that refreshes via the parent's
    /// heartbeat (`replayRatioComputed*` are passed in fresh on
    /// every body re-eval). Layout matches `PopoverRow`'s right-
    /// aligned label and trailing-spacer columns so the rows line
    /// up with the editable form.
    @ViewBuilder
    private func liveDelayRow(label: String, valueMs: Int?) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .frame(width: 160, alignment: .trailing)
            if let v = valueMs {
                Text("\(v)")
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 110, alignment: .leading)
            } else {
                Text("--")
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 110, alignment: .leading)
            }
            Text("ms (auto)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    /// Stepper-binding for the live-update int fields. Same shape
    /// as `PopoverBindings.intBinding` but routes the post-step
    /// value through an `onChange` callback so the parent can
    /// propagate the change to `trainingParams` immediately. This
    /// is the path the +/- arrow buttons take; direct text edits
    /// flow through the surrounding `.onChange(of: textBinding)`
    /// handler at the call site.
    private func liveIntBinding(
        text: Binding<String>,
        fallback: Int,
        onChange: @escaping (Int) -> Void
    ) -> Binding<Int> {
        Binding<Int>(
            get: {
                let trimmed = text.wrappedValue.trimmingCharacters(in: .whitespaces)
                return Int(trimmed) ?? fallback
            },
            set: { newValue in
                text.wrappedValue = String(newValue)
                onChange(newValue)
            }
        )
    }

    /// Stepper-binding for the live-update double fields (currently
    /// only the replay-ratio target). Same shape as `liveIntBinding`
    /// but for `Binding<Double>`.
    private func liveDoubleBinding(
        text: Binding<String>,
        fallback: Double,
        format: String,
        onChange: @escaping (Double) -> Void
    ) -> Binding<Double> {
        Binding<Double>(
            get: {
                let trimmed = text.wrappedValue.trimmingCharacters(in: .whitespaces)
                return Double(trimmed) ?? fallback
            },
            set: { newValue in
                text.wrappedValue = String(format: format, newValue)
                onChange(newValue)
            }
        )
    }
}

// MARK: - Shared row + binding helpers

/// One value row used by every tab: right-aligned label,
/// monospaced text field with red error overlay, optional trailing
/// stepper, optional trailing hint string. Pulled out of the per-
/// tab subviews so the row layout stays consistent across tabs.
private struct PopoverRow<Stepper: View, Info: View>: View {
    let label: String
    @Binding var text: String
    let error: Bool
    let placeholder: String
    var hint: String? = nil
    var disabled: Bool = false
    /// Optional `ⓘ`-style button rendered immediately to the right of the
    /// editable value (before the Stepper). `EmptyView` for the vast
    /// majority of rows — see the `where Info == EmptyView` convenience
    /// initializer below, which is what every plain call site binds to.
    @ViewBuilder let info: () -> Info
    @ViewBuilder let stepper: () -> Stepper

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .frame(width: 160, alignment: .trailing)
            // Use ParameterTextField (not bare TextField). On macOS a
            // bare `TextField(text:)` only commits its in-progress
            // edit to the binding on Return or focus loss — clicking
            // Save while a field still has first-responder status
            // would silently throw away typed/stepped values, even
            // though the field visually showed them. ParameterTextField
            // attaches a `@FocusState` and an `onChange(of: isFocused)`
            // that drives the macOS commit on every focus change so
            // the popover's Save handler sees the actual value the
            // user just entered. The `onCommit` closure here is a
            // no-op because validation/parsing happens transactionally
            // in `TrainingSettingsPopoverModel.save()` — we just need
            // the binding to be current by the time Save reads it.
            ParameterTextField(
                placeholder: placeholder,
                text: $text,
                width: 110
            ) { _ in /* no-op: Save handler does parse + apply */ }
                .font(.system(.body, design: .monospaced))
                .disabled(disabled)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.red, lineWidth: error ? 2 : 0)
                )
            info()
            stepper()
                .labelsHidden()
            if let hint {
                Text(hint)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }
}

/// Convenience initializer for the common case: a row with no `ⓘ` info
/// button. Keeps every existing `PopoverRow(...) { Stepper(...) }` call
/// site compiling unchanged (`Info` is inferred as `EmptyView`).
extension PopoverRow where Info == EmptyView {
    init(
        label: String,
        text: Binding<String>,
        error: Bool,
        placeholder: String,
        hint: String? = nil,
        disabled: Bool = false,
        @ViewBuilder stepper: @escaping () -> Stepper
    ) {
        self.init(
            label: label,
            text: text,
            error: error,
            placeholder: placeholder,
            hint: hint,
            disabled: disabled,
            info: { EmptyView() },
            stepper: stepper
        )
    }
}

/// Stepper-binding helpers shared across the optimizer / self-play
/// tabs. Each helper preserves the popover's transactional model:
/// the displayed text moves with the Stepper, but the actual write
/// to `trainingParams` still happens on Save in the parent.
private enum PopoverBindings {

    /// `Binding<Double>` that reads the current edit text, parses
    /// it (falling back to `fallback` on parse failure), and writes
    /// `String(format:)` back when the Stepper increments.
    static func doubleBinding(
        text: Binding<String>,
        fallback: Double,
        format: String
    ) -> Binding<Double> {
        Binding<Double>(
            get: {
                let trimmed = text.wrappedValue.trimmingCharacters(in: .whitespaces)
                return Double(trimmed) ?? fallback
            },
            set: { newValue in
                text.wrappedValue = String(format: format, newValue)
            }
        )
    }

    /// Same idea as `doubleBinding` but for Int-typed Steppers.
    static func intBinding(
        text: Binding<String>,
        fallback: Int
    ) -> Binding<Int> {
        Binding<Int>(
            get: {
                let trimmed = text.wrappedValue.trimmingCharacters(in: .whitespaces)
                return Int(trimmed) ?? fallback
            },
            set: { newValue in
                text.wrappedValue = String(newValue)
            }
        )
    }
}
