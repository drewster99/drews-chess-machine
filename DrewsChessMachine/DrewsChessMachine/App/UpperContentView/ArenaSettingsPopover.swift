import SwiftUI

/// Three-section editor opened from the countdown chip. The top
/// "Next Arena" section shows the next-arena timestamp; a "Run
/// Now" button shares the section header line and triggers an
/// immediate arena. The middle "Last Arena" section shows the
/// most recent tournament's date/time, W-L-D, AlphaZero score,
/// and kept/promoted verdict, with a "More" link in the section
/// header that opens the full `ArenaHistoryView` sheet. The
/// bottom "Options" section edits the arena knobs (#games,
/// concurrency, interval, promote threshold) plus the τ schedule.
/// Cancel discards edits; Save validates everything and writes it
/// back to `trainingParams` — all of which lives on the
/// `ArenaSettingsPopoverModel` so this view stays a thin shell.
struct ArenaSettingsPopover: View {
    /// All editable / transactional state.
    @Bindable var model: ArenaSettingsPopoverModel

    /// Wall-clock time when the next auto-arena will fire, or `nil`
    /// when there is no live session (in which case the row reads
    /// "Next session" instead of a timestamp).
    let nextArenaDate: Date?
    /// Most recent tournament. `nil` when no arena has run yet in
    /// the current session and no resumed history was loaded.
    let lastArena: TournamentRecord?
    let isArenaRunning: Bool
    let realTraining: Bool
    let onRunNow: () -> Void
    let onShowHistory: () -> Void

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
                HStack(alignment: .firstTextBaseline) {
                    Text("Last Arena")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Button("More history", action: onShowHistory)
                        .buttonStyle(.link)
                        .font(.subheadline)
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

            // --- Match temperature section ---
            //
            // τ schedule controls the per-ply softmax temperature
            // for arena games. Linear decay: tau(ply) = max(floor,
            // start - decay * ply). The "reached at N plies" hint
            // shows when the floor first activates, computed live
            // from whatever the user has typed so far.
            VStack(alignment: .leading, spacing: 8) {
                Text("Match temperature (τ)")
                    .font(.subheadline.weight(.semibold))
                ArenaPopoverField(
                    label: "Start of game:",
                    text: $model.tauStartText,
                    error: model.tauStartError,
                    placeholder: "2.00",
                    width: 100
                )
                ArenaPopoverField(
                    label: "Decay:",
                    text: $model.tauDecayText,
                    error: model.tauDecayError,
                    placeholder: "0.015",
                    width: 100,
                    hint: "per ply"
                )
                ArenaPopoverField(
                    label: "Floor:",
                    text: $model.tauFloorText,
                    error: model.tauFloorError,
                    placeholder: "0.50",
                    width: 100,
                    hint: model.tauReachedAtHint
                )
            }

            Divider()

            // --- Options section ---
            VStack(alignment: .leading, spacing: 8) {
                Text("Options")
                    .font(.subheadline.weight(.semibold))
                ArenaPopoverField(
                    label: "# of games:",
                    text: $model.gamesText,
                    error: model.gamesError,
                    placeholder: "200",
                    width: 100
                )
                ArenaPopoverField(
                    label: "Concurrency:",
                    text: $model.concurrencyText,
                    error: model.concurrencyError,
                    placeholder: "200",
                    width: 100
                )
                ArenaPopoverField(
                    label: "Interval:",
                    text: $model.intervalText,
                    error: model.intervalError,
                    placeholder: "15m",
                    width: 100,
                    hint: "(e.g. 15m, 500s, 7d, 90)"
                )
                // Lowering this is the way to "semi-force" an early
                // promotion from a candidate that's been hovering
                // just above 0.50 — e.g. set 0.510 to promote on
                // any clearly-positive arena.
                ArenaPopoverField(
                    label: "Promote threshold:",
                    text: $model.promoteThresholdText,
                    error: model.promoteThresholdError,
                    placeholder: "0.550",
                    width: 100,
                    hint: "(score in [0.5, 1.0])"
                )
            }

            HStack {
                Spacer()
                Button("Cancel") { model.cancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") { model.save() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(width: 380)
        .onAppear { model.seedFromParams() }
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
