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
    @Binding var promoteThresholdText: String
    @Binding var tauStartText: String
    @Binding var tauDecayText: String
    @Binding var tauFloorText: String
    let gamesError: Bool
    let concurrencyError: Bool
    let intervalError: Bool
    let promoteThresholdError: Bool
    let tauStartError: Bool
    let tauDecayError: Bool
    let tauFloorError: Bool
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
            // tau (τ) schedule controls the per-ply softmax temperature
            // for arena games. Linear decay: tau(ply) = max(floor,
            // start - decay * ply). The "reached at N plies" hint
            // shows when the floor first activates, computed live
            // from whatever the user has typed so far.
            VStack(alignment: .leading, spacing: 8) {
                Text("Match temperature (τ)")
                    .font(.subheadline.weight(.semibold))
                ArenaPopoverField(
                    label: "Start of game:",
                    text: $tauStartText,
                    error: tauStartError,
                    placeholder: "2.00",
                    width: 100
                )
                ArenaPopoverField(
                    label: "Decay:",
                    text: $tauDecayText,
                    error: tauDecayError,
                    placeholder: "0.015",
                    width: 100,
                    hint: "per ply"
                )
                ArenaPopoverField(
                    label: "Floor:",
                    text: $tauFloorText,
                    error: tauFloorError,
                    placeholder: "0.50",
                    width: 100,
                    hint: tauReachedAtHint
                )
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
                // Lowering this is the way to "semi-force" an early
                // promotion from a candidate that's been hovering
                // just above 0.50 — e.g. set 0.510 to promote on
                // any clearly-positive arena.
                ArenaPopoverField(
                    label: "Promote threshold:",
                    text: $promoteThresholdText,
                    error: promoteThresholdError,
                    placeholder: "0.550",
                    width: 100,
                    hint: "(score in [0.5, 1.0])"
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

    /// Live "reached at N plies" hint for the floor field. Reads
    /// the *current* parsed values of all three tau fields and
    /// recomputes on every render — typing into any field updates
    /// the hint immediately. Returns "—" when any field is invalid
    /// or the math is degenerate (decay = 0, floor >= start).
    private var tauReachedAtHint: String {
        guard let start = Double(tauStartText), start > 0,
              let decay = Double(tauDecayText), decay >= 0,
              let floor = Double(tauFloorText), floor > 0 else {
            return "(reached at —)"
        }
        guard decay > 0 else { return "(no decay; floor unreached)" }
        guard floor < start else { return "(reached at ply 0)" }
        let plies = Int(((start - floor) / decay).rounded(.up))
        return "(reached at ply \(plies))"
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
