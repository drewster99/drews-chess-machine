import SwiftUI

/// Optimizer-knobs editor opened from the top-bar `TrainingSettingsChip`.
/// Pulls all live-tunable optimizer math out of the inline stats text
/// panel into a single focused dialog. Layout mirrors the user's
/// sketched hierarchy: LR + LR-modifiers (warmup / momentum / √batch),
/// Entropy bonus + regularization companions (clip / decay / K policy
/// scalar), Draw penalty standalone.
///
/// Edit-text bindings are owned by `UpperContentView` (same pattern as
/// `ArenaSettingsPopover` — keeps in-progress text alive across renders
/// and lets the parent reuse the existing `learningRateEditText` /
/// `entropyRegularizationEditText` / etc. that previously backed the
/// inline rows). Save validates and writes through `arenaPopoverSave`-
/// style logic in the parent; Cancel discards the in-progress text.
///
/// Each numeric row is a label + monospaced text field + Stepper. The
/// Steppers update the displayed text directly (via numeric `Binding`
/// helpers below) so a `+` press shows the new value immediately;
/// the actual write to `trainingParams` still happens on Save. LR uses
/// a log-ladder (×10 / ÷10) Stepper because the value spans 7 orders
/// of magnitude and a linear step would be unusable.
struct TrainingSettingsPopover: View {
    /// Trainer model ID at session start, displayed in the header.
    /// "—" when no trainer exists yet (build-pre-session).
    let modelID: String
    /// Session start wall-clock time, displayed in the header so the
    /// user has a stable "this is the run you're configuring" anchor.
    let sessionStart: Date

    @Binding var lrText: String
    @Binding var warmupText: String
    @Binding var momentumText: String
    @Binding var sqrtBatchScalingValue: Bool
    @Binding var entropyText: String
    @Binding var gradClipText: String
    @Binding var weightDecayText: String
    @Binding var policyKText: String
    @Binding var drawPenaltyText: String

    let lrError: Bool
    let warmupError: Bool
    let momentumError: Bool
    let entropyError: Bool
    let gradClipError: Bool
    let weightDecayError: Bool
    let policyKError: Bool
    let drawPenaltyError: Bool

    let onCancel: () -> Void
    let onSave: () -> Void
    let onAppearSeed: () -> Void

    var body: some View {
        let dateFmt: DateFormatter = {
            let f = DateFormatter()
            f.dateStyle = .short
            f.timeStyle = .medium
            return f
        }()

        VStack(alignment: .leading, spacing: 16) {
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

            Divider()

            // LR section — top-level LR, indented modifiers.
            VStack(alignment: .leading, spacing: 6) {
                row(
                    label: "LR:",
                    text: $lrText,
                    error: lrError,
                    placeholder: "5.00e-05"
                ) {
                    Stepper(
                        "",
                        onIncrement: { stepLRBy(factor: 10) },
                        onDecrement: { stepLRBy(factor: 0.1) }
                    )
                }
                VStack(alignment: .leading, spacing: 6) {
                    row(
                        label: "Warm-up steps:",
                        text: $warmupText,
                        error: warmupError,
                        placeholder: "100"
                    ) {
                        Stepper(
                            "",
                            value: intBinding(text: $warmupText, fallback: 100),
                            in: 0...100_000,
                            step: 100
                        )
                    }
                    row(
                        label: "Momentum:",
                        text: $momentumText,
                        error: momentumError,
                        placeholder: "0.000"
                    ) {
                        Stepper(
                            "",
                            value: doubleBinding(
                                text: $momentumText,
                                fallback: 0.0,
                                format: "%.3f"
                            ),
                            in: 0.0...0.99,
                            step: 0.05
                        )
                    }
                    HStack(spacing: 8) {
                        // Empty leading cell to line the checkbox up
                        // with the text-field column above.
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
                row(
                    label: "Entropy regularization:",
                    text: $entropyText,
                    error: entropyError,
                    placeholder: "1.00e-03"
                ) {
                    Stepper(
                        "",
                        value: doubleBinding(
                            text: $entropyText,
                            fallback: 1e-3,
                            format: "%.2e"
                        ),
                        in: 0.0...0.1,
                        step: 1e-3
                    )
                }
                VStack(alignment: .leading, spacing: 6) {
                    row(
                        label: "Clip:",
                        text: $gradClipText,
                        error: gradClipError,
                        placeholder: "30.0"
                    ) {
                        Stepper(
                            "",
                            value: doubleBinding(
                                text: $gradClipText,
                                fallback: 30.0,
                                format: "%.1f"
                            ),
                            in: 0.1...1000.0,
                            step: 1.0
                        )
                    }
                    row(
                        label: "Decay:",
                        text: $weightDecayText,
                        error: weightDecayError,
                        placeholder: "1.00e-04"
                    ) {
                        Stepper(
                            "",
                            value: doubleBinding(
                                text: $weightDecayText,
                                fallback: 1e-4,
                                format: "%.2e"
                            ),
                            in: 0.0...0.1,
                            step: 1e-4
                        )
                    }
                    row(
                        label: "K policy scalar:",
                        text: $policyKText,
                        error: policyKError,
                        placeholder: "5.00"
                    ) {
                        Stepper(
                            "",
                            value: doubleBinding(
                                text: $policyKText,
                                fallback: 5.0,
                                format: "%.2f"
                            ),
                            in: 0.1...20.0,
                            step: 0.5
                        )
                    }
                }
                .padding(.leading, 20)
            }

            Divider()

            // Draw penalty — top-level, no indent (sketch-level peer
            // to LR / Entropy, but logically standalone).
            row(
                label: "Draw penalty:",
                text: $drawPenaltyText,
                error: drawPenaltyError,
                placeholder: "0.100"
            ) {
                Stepper(
                    "",
                    value: doubleBinding(
                        text: $drawPenaltyText,
                        fallback: 0.1,
                        format: "%.3f"
                    ),
                    in: -1.0...1.0,
                    step: 0.05
                )
            }

            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Button("Save", action: onSave)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(width: 460)
        .onAppear { onAppearSeed() }
    }

    // MARK: - Row + binding helpers

    /// One value row: right-aligned label, monospaced text field with
    /// red error overlay, trailing Stepper. The Stepper view is
    /// supplied by the caller because each row binds it to a
    /// different parameter / range / format.
    @ViewBuilder
    private func row<S: View>(
        label: String,
        text: Binding<String>,
        error: Bool,
        placeholder: String,
        @ViewBuilder stepper: () -> S
    ) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .frame(width: 160, alignment: .trailing)
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
                .frame(width: 110)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.red, lineWidth: error ? 2 : 0)
                )
            stepper()
                .labelsHidden()
            Spacer()
        }
    }

    /// `Binding<Double>` that reads the current edit text, parses it
    /// (falling back to `fallback` on parse failure), and writes
    /// `String(format:)` back when the Stepper increments. This
    /// preserves the popover's transactional model — the displayed
    /// text moves with the Stepper, but the actual write to
    /// `trainingParams` still happens on Save in the parent.
    private func doubleBinding(
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
    private func intBinding(
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

    /// Multiply the current LR text by `factor` (10 for `+`, 0.1 for
    /// `-`), clamp into `TrainingParameters` range `[1e-7, 1.0]`, and
    /// write back. The log ladder is necessary because LR spans
    /// seven orders of magnitude — a linear `step:` would be either
    /// useless at the small end or jump past the working range at
    /// the large end.
    private func stepLRBy(factor: Double) {
        let trimmed = lrText.trimmingCharacters(in: .whitespaces)
        let current = Double(trimmed) ?? 5e-5
        let next = max(1e-7, min(1.0, current * factor))
        lrText = String(format: "%.2e", next)
    }
}
