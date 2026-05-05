import SwiftUI

/// Three-section editor opened from the countdown chip. The top
/// "Next Arena" section shows the next-arena timestamp; a "Run
/// Now" button shares the section header line and triggers an
/// immediate arena. The middle "Last Arena" section shows the
/// most recent tournament's date/time, W-L-D, AlphaZero score,
/// and kept/promoted verdict, with a "More" link in the section
/// header that opens the full `ArenaHistoryView` sheet. The
/// bottom "Options" section edits the three arena knobs
/// (#games, concurrency, interval — interval accepts
/// 15m/500s/7d/etc.). Cancel discards edits; Save validates all
/// three and writes them back to `trainingParams` (validation +
/// write happen in the supplied `onSave` callback so the popover
/// stays decoupled from the parameter store).
struct ArenaSettingsPopover: View {
    /// Wall-clock time when the next auto-arena will fire, or `nil`
    /// when there is no live session (in which case the row reads
    /// "Next session" instead of a timestamp).
    let nextArenaDate: Date?
    /// Most recent tournament. `nil` when no arena has run yet in
    /// the current session and no resumed history was loaded.
    let lastArena: TournamentRecord?
    let isArenaRunning: Bool
    let realTraining: Bool
    @Binding var gamesText: String
    @Binding var concurrencyText: String
    @Binding var intervalText: String
    let gamesError: Bool
    let concurrencyError: Bool
    let intervalError: Bool
    let onRunNow: () -> Void
    let onShowHistory: () -> Void
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
                HStack {
                    Text("Next Arena")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Button {
                        guard !isArenaRunning, realTraining else { return }
                        onRunNow()
                    } label: {
                        Label("Run Now", systemImage: "flag.checkered")
                    }
                    .disabled(isArenaRunning || !realTraining)
                }
                if let nextArenaDate, realTraining {
                    Text(dateFmt.string(from: nextArenaDate))
                        .font(.system(.body, design: .monospaced))
                } else {
                    Text("Next session")
                        .foregroundStyle(.secondary)
                }
                Text("(countdown is shown in the chip above)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Divider()

            // --- Last Arena section ---
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Last Arena")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Button("More", action: onShowHistory)
                        .buttonStyle(.link)
                        .disabled(lastArena == nil)
                }
                if let lastArena {
                    lastArenaSummary(lastArena, dateFmt: dateFmt)
                } else {
                    Text("No arenas yet")
                        .foregroundStyle(.secondary)
                }
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
        .frame(width: 380)
        .onAppear { onAppearSeed() }
    }

    /// Two-row last-arena summary. Extracted into a `@ViewBuilder`
    /// helper because the original inline form mixed `if let` /
    /// `switch` into a `String`-typed `let` chain inside the parent
    /// `VStack`'s ViewBuilder body, which the compiler tried to
    /// resolve as `buildEither` branches and failed (`type '()'
    /// cannot conform to 'View'`). Computing the strings up front
    /// in normal Swift control flow side-steps the issue and keeps
    /// the view-tree decisions inside a single concrete HStack.
    @ViewBuilder
    private func lastArenaSummary(
        _ lastArena: TournamentRecord,
        dateFmt: DateFormatter
    ) -> some View {
        let dateText: String = {
            if let dt = lastArena.finishedAt {
                return dateFmt.string(from: dt)
            }
            return "—"
        }()
        let wdl = "\(lastArena.candidateWins)–\(lastArena.draws)–\(lastArena.championWins)"
        let scoreText = String(format: "%.3f", lastArena.score)
        let (verdict, verdictColor): (String, Color) = {
            switch lastArena.promotionKind {
            case .automatic: return ("PROMOTED (auto)", .green)
            case .manual:    return ("PROMOTED (manual)", .green)
            case .none:      return ("kept", .secondary)
            }
        }()

        Text(dateText)
            .font(.system(.body, design: .monospaced))
        HStack(spacing: 8) {
            Text("W–D–L \(wdl)")
                .font(.system(.caption, design: .monospaced))
            Text("•")
                .foregroundStyle(.secondary)
            Text("score \(scoreText)")
                .font(.system(.caption, design: .monospaced))
            Text("•")
                .foregroundStyle(.secondary)
            Text(verdict)
                .font(.caption.weight(.semibold))
                .foregroundStyle(verdictColor)
        }
    }
}
