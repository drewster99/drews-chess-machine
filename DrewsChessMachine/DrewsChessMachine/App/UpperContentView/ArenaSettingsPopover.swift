import SwiftUI

/// Two-section editor opened from the countdown chip. Top section
/// shows the next-arena timestamp + Run Arena Now button. Bottom
/// section edits the three arena knobs (#games, concurrency, and
/// interval — interval accepts 15m/500s/7d/etc.). Cancel discards
/// edits; Save validates all three and writes them back to
/// `trainingParams` (validation + write happen in the supplied
/// `onSave` callback so the popover stays decoupled from the
/// parameter store).
struct ArenaSettingsPopover: View {
    /// Wall-clock time when the next auto-arena will fire, or `nil`
    /// when there is no live session (in which case the row reads
    /// "Next session" instead of a timestamp).
    let nextArenaDate: Date?
    let isArenaRunning: Bool
    let realTraining: Bool
    @Binding var gamesText: String
    @Binding var concurrencyText: String
    @Binding var intervalText: String
    let gamesError: Bool
    let concurrencyError: Bool
    let intervalError: Bool
    let onRunNow: () -> Void
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
            Text("Arena")
                .font(.headline)

            // --- Next Arena section ---
            VStack(alignment: .leading, spacing: 8) {
                Text("Next Arena")
                    .font(.subheadline.weight(.semibold))
                HStack {
                    if let nextArenaDate, realTraining {
                        Text(dateFmt.string(from: nextArenaDate))
                            .font(.system(.body, design: .monospaced))
                    } else {
                        Text("Next session")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button {
                        guard !isArenaRunning, realTraining else { return }
                        onRunNow()
                    } label: {
                        Label("Run Arena Now", systemImage: "flag.checkered")
                    }
                    .disabled(isArenaRunning || !realTraining)
                }
                Text("(countdown is shown in the chip above)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Divider()

            // --- Options section ---
            VStack(alignment: .leading, spacing: 8) {
                Text("Options")
                    .font(.subheadline.weight(.semibold))
                ArenaPopoverField(
                    label: "# of games:",
                    text: $gamesText,
                    error: gamesError,
                    placeholder: "200",
                    width: 100
                )
                ArenaPopoverField(
                    label: "Concurrency:",
                    text: $concurrencyText,
                    error: concurrencyError,
                    placeholder: "200",
                    width: 100
                )
                ArenaPopoverField(
                    label: "Interval:",
                    text: $intervalText,
                    error: intervalError,
                    placeholder: "15m",
                    width: 100,
                    hint: "(e.g. 15m, 500s, 7d, 90)"
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
        .frame(width: 360)
        .onAppear { onAppearSeed() }
    }
}
