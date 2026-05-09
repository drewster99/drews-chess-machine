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
/// Edit-text bindings + error flags are owned by `UpperContentView`
/// (same pattern as `ArenaSettingsPopover`) — keeps in-progress
/// text alive across renders. Save validates and writes back to
/// `trainingParams` via the parent's `trainingPopoverSave()` helper;
/// Cancel discards in-progress text.
///
/// **Live-propagation exception for replay-ratio fields.** The three
/// replay-ratio control fields (`selfPlayDelayMs`, `trainingStepDelayMs`,
/// `replayRatioAutoAdjust`) write through to `trainingParams`
/// *immediately on change* via the `onLive...Change` callbacks rather
/// than waiting for Save. This lets the user watch the live ratio
/// display respond to their change. On Cancel, the parent reverts
/// these three fields to the values it stashed in `onAppearSeed`,
/// matching the standard "edit → cancel discards" pattern from the
/// user's POV.
fileprivate enum Tab: String, CaseIterable, Identifiable {
    case optimizer = "Optimizer"
    case selfPlay = "Self Play"
    case replay = "Replay"
    var id: String { rawValue }
}

struct TrainingSettingsPopover: View {

    /// Trainer model ID at session start, displayed in the header.
    /// "—" when no trainer exists yet (build-pre-session).
    let modelID: String
    /// Session start wall-clock time, displayed in the header so the
    /// user has a stable "this is the run you're configuring" anchor.
    let sessionStart: Date

    // MARK: - Optimizer-tab bindings

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
    let drawPenaltyError: Bool
    let trainingBatchSizeError: Bool

    // MARK: - Self-Play-tab bindings

    @Binding var selfPlayWorkersText: String
    @Binding var selfPlayStartTauText: String
    @Binding var selfPlayDecayPerPlyText: String
    @Binding var selfPlayFloorTauText: String

    let selfPlayWorkersError: Bool
    let selfPlayStartTauError: Bool
    let selfPlayDecayPerPlyError: Bool
    let selfPlayFloorTauError: Bool

    // MARK: - Replay-tab bindings

    @Binding var replayBufferCapacityText: String
    @Binding var replayBufferMinPositionsText: String
    @Binding var replayRatioTargetText: String
    @Binding var replaySelfPlayDelayText: String
    @Binding var replayTrainingStepDelayText: String
    @Binding var replayRatioAutoAdjust: Bool

    let replayBufferCapacityError: Bool
    let replayBufferMinPositionsError: Bool
    let replayRatioTargetError: Bool
    let replaySelfPlayDelayError: Bool
    let replayTrainingStepDelayError: Bool

    /// Live "current ratio (target X.XX)" snapshot for the Replay
    /// tab's status line. Refreshes with the parent's heartbeat;
    /// nil before the controller has produced its first sample.
    let replayRatioCurrent: Double?
    /// Live auto-computed training-step delay from the controller.
    /// Displayed in place of the editable Train-step-delay field
    /// when `replayRatioAutoAdjust == true` so the user can see
    /// what the auto-controller is currently doing. Nil before the
    /// controller has produced its first sample.
    let replayRatioComputedDelayMs: Int?
    /// Live auto-computed self-play delay from the controller.
    /// Same role as `replayRatioComputedDelayMs` but for the
    /// Self-play-delay field.
    let replayRatioComputedSelfPlayDelayMs: Int?
    /// Bytes-per-position for the auto-GB readout. Sourced from
    /// `ReplayBuffer.bytesPerPosition` so any future schema change
    /// updates the displayed estimate without further plumbing.
    let bytesPerPosition: Int

    // MARK: - Live-update callbacks (Replay-ratio fields only)
    //
    // These callbacks fire on EVERY change while the popover is
    // open — text-field commit, stepper press, checkbox toggle.
    // The parent writes the value through to `trainingParams` and
    // (for the delay values) to the live `ReplayRatioController`
    // immediately. On Cancel, the parent restores from its stash.

    let onLiveReplayRatioTargetChange: (Double) -> Void
    let onLiveSelfPlayDelayChange: (Int) -> Void
    let onLiveTrainingStepDelayChange: (Int) -> Void
    let onLiveReplayRatioAutoAdjustChange: (Bool) -> Void

    // MARK: - Lifecycle callbacks

    let onCancel: () -> Void
    let onSave: () -> Void
    let onAppearSeed: () -> Void

    // MARK: - Tab selection

    /// Local tab selection. Resets to `.optimizer` each time the
    /// popover re-instantiates (i.e. each time the chip is clicked
    /// open). Persisting it across opens would surprise the user
    /// — the natural mental model is "always start on Optimizer."
    @State private var selectedTab: Tab = .optimizer

    // Aggregated per-tab error flags, derived from the existing
    // per-field `*Error: Bool` properties already passed in by the
    // parent. These drive (a) the red-dot indicator on each tab in
    // the segmented control below and (b) the Save button's
    // `.disabled(...)` modifier — once Save has set any error flag,
    // the user has to clear it (by editing the offending field) before
    // Save re-enables. The parent's `.onChange` handlers next to the
    // popover construction site clear individual error flags as the
    // user edits the matching text field.
    private var optimizerHasError: Bool {
        lrError
            || warmupError
            || momentumError
            || entropyError
            || gradClipError
            || weightDecayError
            || policyLossWeightError
            || valueLossWeightError
            || drawPenaltyError
            || trainingBatchSizeError
    }

    private var selfPlayHasError: Bool {
        selfPlayWorkersError
            || selfPlayStartTauError
            || selfPlayDecayPerPlyError
            || selfPlayFloorTauError
    }

    private var replayHasError: Bool {
        replayBufferCapacityError
            || replayBufferMinPositionsError
            || replayRatioTargetError
            || replaySelfPlayDelayError
            || replayTrainingStepDelayError
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
            // helpers can live close to the fields they drive.
            switch selectedTab {
            case .optimizer:
                OptimizerTab(
                    lrText: $lrText,
                    warmupText: $warmupText,
                    momentumText: $momentumText,
                    sqrtBatchScalingValue: $sqrtBatchScalingValue,
                    entropyText: $entropyText,
                    illegalMassWeightText: $illegalMassWeightText,
                    gradClipText: $gradClipText,
                    weightDecayText: $weightDecayText,
                    policyLossWeightText: $policyLossWeightText,
                    valueLossWeightText: $valueLossWeightText,
                    drawPenaltyText: $drawPenaltyText,
                    trainingBatchSizeText: $trainingBatchSizeText,
                    lrError: lrError,
                    warmupError: warmupError,
                    momentumError: momentumError,
                    entropyError: entropyError,
                    illegalMassWeightError: illegalMassWeightError,
                    gradClipError: gradClipError,
                    weightDecayError: weightDecayError,
                    policyLossWeightError: policyLossWeightError,
                    valueLossWeightError: valueLossWeightError,
                    drawPenaltyError: drawPenaltyError,
                    trainingBatchSizeError: trainingBatchSizeError
                )
            case .selfPlay:
                SelfPlayTab(
                    selfPlayWorkersText: $selfPlayWorkersText,
                    selfPlayStartTauText: $selfPlayStartTauText,
                    selfPlayDecayPerPlyText: $selfPlayDecayPerPlyText,
                    selfPlayFloorTauText: $selfPlayFloorTauText,
                    selfPlayWorkersError: selfPlayWorkersError,
                    selfPlayStartTauError: selfPlayStartTauError,
                    selfPlayDecayPerPlyError: selfPlayDecayPerPlyError,
                    selfPlayFloorTauError: selfPlayFloorTauError
                )
            case .replay:
                ReplayTab(
                    replayBufferCapacityText: $replayBufferCapacityText,
                    replayBufferMinPositionsText: $replayBufferMinPositionsText,
                    replayRatioTargetText: $replayRatioTargetText,
                    replaySelfPlayDelayText: $replaySelfPlayDelayText,
                    replayTrainingStepDelayText: $replayTrainingStepDelayText,
                    replayRatioAutoAdjust: $replayRatioAutoAdjust,
                    replayBufferCapacityError: replayBufferCapacityError,
                    replayBufferMinPositionsError: replayBufferMinPositionsError,
                    replayRatioTargetError: replayRatioTargetError,
                    replaySelfPlayDelayError: replaySelfPlayDelayError,
                    replayTrainingStepDelayError: replayTrainingStepDelayError,
                    replayRatioCurrent: replayRatioCurrent,
                    replayRatioComputedDelayMs: replayRatioComputedDelayMs,
                    replayRatioComputedSelfPlayDelayMs: replayRatioComputedSelfPlayDelayMs,
                    bytesPerPosition: bytesPerPosition,
                    onLiveReplayRatioTargetChange: onLiveReplayRatioTargetChange,
                    onLiveSelfPlayDelayChange: onLiveSelfPlayDelayChange,
                    onLiveTrainingStepDelayChange: onLiveTrainingStepDelayChange,
                    onLiveReplayRatioAutoAdjustChange: onLiveReplayRatioAutoAdjustChange
                )
            }

            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
                // Save stays disabled while any field is currently
                // marked invalid. The matching error flag is cleared
                // by the parent's `.onChange` handler when the user
                // edits the offending field, at which point Save
                // re-enables and the next click re-validates the
                // full form transactionally.
                Button("Save", action: onSave)
                    .keyboardShortcut(.defaultAction)
                    .disabled(anyTabHasError)
            }
        }
        .padding(16)
        .frame(width: 480)
        .onAppear { onAppearSeed() }
        .onDisappear {
            // macOS popovers dismiss on outside-click without ever
            // calling the Cancel button's action. The Replay tab
            // live-propagates three values to `trainingParams`
            // during edits, so an outside-click would silently
            // commit those changes against the user's intent. By
            // routing every dismissal through `onCancel` we get
            // "Cancel reverts, Save commits, outside-click reverts"
            // — and `onCancel` (== `trainingPopoverCancel` in the
            // parent) is idempotent because Save updates the parent's
            // pre-edit stash before closing, so a Save → onDisappear
            // sequence finds nothing to revert.
            onCancel()
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
    @Binding var selfPlayWorkersText: String
    @Binding var selfPlayStartTauText: String
    @Binding var selfPlayDecayPerPlyText: String
    @Binding var selfPlayFloorTauText: String

    let selfPlayWorkersError: Bool
    let selfPlayStartTauError: Bool
    let selfPlayDecayPerPlyError: Bool
    let selfPlayFloorTauError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Self Play")
                    .font(.subheadline.weight(.semibold))
                PopoverRow(
                    label: "Concurrency:",
                    text: $selfPlayWorkersText,
                    error: selfPlayWorkersError,
                    placeholder: "8"
                ) {
                    Stepper(
                        "",
                        value: PopoverBindings.intBinding(text: $selfPlayWorkersText, fallback: 8),
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
                    EmptyView()
                }
                PopoverRow(
                    label: "Decay:",
                    text: $selfPlayDecayPerPlyText,
                    error: selfPlayDecayPerPlyError,
                    placeholder: "0.030",
                    hint: "per ply"
                ) {
                    EmptyView()
                }
                PopoverRow(
                    label: "Floor:",
                    text: $selfPlayFloorTauText,
                    error: selfPlayFloorTauError,
                    placeholder: "0.40",
                    hint: tauReachedAtHint
                ) {
                    EmptyView()
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
        }
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
}

// MARK: - Replay tab

private struct ReplayTab: View {
    @Binding var replayBufferCapacityText: String
    @Binding var replayBufferMinPositionsText: String
    @Binding var replayRatioTargetText: String
    @Binding var replaySelfPlayDelayText: String
    @Binding var replayTrainingStepDelayText: String
    @Binding var replayRatioAutoAdjust: Bool

    let replayBufferCapacityError: Bool
    let replayBufferMinPositionsError: Bool
    let replayRatioTargetError: Bool
    let replaySelfPlayDelayError: Bool
    let replayTrainingStepDelayError: Bool

    let replayRatioCurrent: Double?
    let replayRatioComputedDelayMs: Int?
    let replayRatioComputedSelfPlayDelayMs: Int?
    let bytesPerPosition: Int

    let onLiveReplayRatioTargetChange: (Double) -> Void
    let onLiveSelfPlayDelayChange: (Int) -> Void
    let onLiveTrainingStepDelayChange: (Int) -> Void
    let onLiveReplayRatioAutoAdjustChange: (Bool) -> Void

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
        }
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
private struct PopoverRow<Stepper: View>: View {
    let label: String
    @Binding var text: String
    let error: Bool
    let placeholder: String
    var hint: String? = nil
    var disabled: Bool = false
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
            // in `trainingPopoverSave()` — we just need the binding
            // to be current by the time Save reads it.
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
