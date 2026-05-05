import SwiftUI

/// Sheet that displays the full per-session arena history in a
/// 2-line tabular layout. One row per completed tournament,
/// newest at the top so the most recent result is visible without
/// scrolling.
///
/// Row layout:
///   Line 1 — `#index`, finished-at datetime,
///            `N games (mm:ss)`, Elo with 95% CI, verdict
///            (kept / PROMOTED).
///   Line 2 — `step N,NNN` (column-aligned under Line 1's
///            datetime), then `Trainer <cand_id> (W: N%) -vs-
///            <champ_id> W:N D:N(d%) L:N` (column-aligned
///            under Line 1's "N games" cell), with a `⇒
///            <promoted_id>` tail when applicable.
///
/// Outliers (promotions, scores ≥ promoteThreshold even when not
/// promoted, scores ≤ a "lost-by-a-lot" floor) are highlighted
/// with color so a quick scan picks them out.
struct ArenaHistoryView: View {
    let history: [TournamentRecord]
    /// Total games configured for tournaments at the time the
    /// sheet was opened — used in the "games X/Y" display so it
    /// matches what the rest of the UI reports.
    let configuredGamesPerTournament: Int
    /// Promote threshold from `TrainingParameters`. Records at or
    /// above this score are highlighted as a "high score" outlier,
    /// even when not promoted (e.g. mid-arena abort or bug).
    let promoteThreshold: Double
    let onClose: () -> Void
    /// Optional recovery callback — when present and at least one
    /// row has nil `finishedAt` / `candidateID` / `championID`,
    /// the header shows a "Recover from logs" button that
    /// triggers a one-shot scan of `~/Library/Logs/DrewsChessMachine/`
    /// to backfill missing fields. Owner is responsible for
    /// triggering a session save after a successful recovery so
    /// the recovered data persists across resumes.
    var onRecoverFromLogs: (() -> Void)?
    /// Set true while a recovery is in progress so the button can
    /// disable itself and the header can show a spinner.
    var recoveryInProgress: Bool = false

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .medium
        return f
    }()

    /// True iff at least one row is missing the fields a recovery
    /// pass could fill in. When false, the recovery button stays
    /// hidden — there's nothing to backfill.
    private var hasMissingFields: Bool {
        history.contains { record in
            record.finishedAt == nil
            || record.candidateID == nil
            || record.championID == nil
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Arena History")
                    .font(.title2.weight(.semibold))
                Spacer()
                Text("\(history.count) tournament\(history.count == 1 ? "" : "s")")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                if let onRecoverFromLogs, hasMissingFields {
                    if recoveryInProgress {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Button("Recover from logs", action: onRecoverFromLogs)
                        .disabled(recoveryInProgress)
                }
                Button("Close", action: onClose)
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            Divider()

            if history.isEmpty {
                VStack {
                    Spacer()
                    Text("No arenas yet")
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        // Newest first so the recent picture is
                        // visible without scrolling. Reverse the
                        // backing array; the original `id` from
                        // `TournamentRecord` is stable so SwiftUI
                        // can diff updates.
                        ForEach(Array(history.enumerated().reversed()), id: \.element.id) { idx, record in
                            ArenaHistoryRow(
                                index: idx + 1,
                                record: record,
                                configuredGamesPerTournament: configuredGamesPerTournament,
                                promoteThreshold: promoteThreshold,
                                dateFormatter: Self.dateFmt
                            )
                            Divider()
                        }
                    }
                }
            }
        }
        .frame(minWidth: 1380, idealWidth: 1560, minHeight: 380, idealHeight: 560)
    }
}

/// Single row in the arena history table. Keeps its own layout
/// closure so the parent `LazyVStack` only re-renders this row
/// when its inputs change.
private struct ArenaHistoryRow: View {
    let index: Int
    let record: TournamentRecord
    let configuredGamesPerTournament: Int
    let promoteThreshold: Double
    let dateFormatter: DateFormatter

    // Column widths shared between Row 1 and Row 2 so columns
    // stay aligned. Both rows MUST use the same `colSpacing` so
    // the column arithmetic below stays valid.
    //
    //   Row 1: [#N (idx)] [datetime] [N games (mm:ss)] [Elo CI]
    //   Row 2:            [step N]   [Trainer … vs … W:N D:N L:N]
    //
    // Row 2's `step` cell sits directly under Row 1's datetime
    // cell, and Row 2's Trainer block sits directly under
    // Row 1's games cell. The two rows share the index-cell
    // width but diverge on what they put in cols 2 and 3.
    private static let colIndexW: CGFloat = 50
    private static let colDateW: CGFloat = 180
    private static let colGamesW: CGFloat = 170
    private static let colEloW: CGFloat = 140
    private static let colSpacing: CGFloat = 12

    var body: some View {
        let elo = record.eloSummary
        let eloCI = ArenaEloStats.formatEloWithCI(elo)
        let dateText = record.finishedAt.map { dateFormatter.string(from: $0) } ?? "—"

        let stepText = "step \(record.finishedAtStep.formatted(.number.grouping(.automatic)))"
        let durationText = ArenaLogFormatter.formatDuration(record.durationSec)
        let gamesText = "\(record.gamesPlayed) games (\(durationText))"

        let scoreOutlier = scoreOutlierKind()
        let verdict = ArenaLogFormatter.formatVerdict(record: record)
        let verdictColor: Color = record.promoted ? .green : .secondary

        // Inline-init to a tuple-typed closure: avoids the
        // ViewBuilder treating a top-level `switch` as a view-tree
        // branch (`type '()' cannot conform to 'View'`).
        let eloColor: Color = {
            switch scoreOutlier {
            case .high: return .green
            case .low:  return .red
            case .none: return .secondary
            }
        }()

        // Row 2 strings. Trainer (= candidate at arena time)
        // gets its overall AlphaZero score in parentheses;
        // champion gets the W/D/L breakdown plus draw-rate
        // percentage.
        let trainerID = record.candidateID?.description ?? "—"
        let championID = record.championID?.description ?? "—"
        let trainerScorePct = String(format: "%.0f%%", record.score * 100)
        let drawRatePct: String = {
            guard record.gamesPlayed > 0 else { return "—" }
            let frac = Double(record.draws) / Double(record.gamesPlayed)
            return String(format: "%.0f%%", frac * 100)
        }()

        VStack(alignment: .leading, spacing: 3) {
            // Row 1 — headline: index, datetime, games+duration,
            // Elo CI, verdict. Step number lives in Row 2, under
            // the datetime, so the headline reads as a calendar
            // entry first and a counter second.
            HStack(alignment: .firstTextBaseline, spacing: Self.colSpacing) {
                Text("#\(index)")
                    .font(.system(.body, design: .monospaced).weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: Self.colIndexW, alignment: .leading)
                Text(dateText)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: Self.colDateW, alignment: .leading)
                Text(gamesText)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: Self.colGamesW, alignment: .leading)
                Text(eloCI)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(eloColor)
                    .frame(width: Self.colEloW, alignment: .leading)
                Spacer(minLength: 8)
                Text(verdict)
                    .font(.system(.body, design: .monospaced).weight(.semibold))
                    .foregroundStyle(verdictColor)
            }
            // Row 2 — `step N,NNN` directly under the datetime,
            // then `Trainer … -vs- … W:N D:N(d%) L:N` directly
            // under Row 1's games column. Same font size as Row 1
            // for tabular feel; lighter weight + secondary
            // foreground for visual differentiation.
            //
            // Layout: [empty(idx)] [stepText sized to colDateW]
            //         [trainer block sized to colGamesW + rest].
            // The two HStack-managed `colSpacing` gaps between
            // those three cells exactly match Row 1's gaps.
            HStack(alignment: .firstTextBaseline, spacing: Self.colSpacing) {
                Color.clear
                    .frame(width: Self.colIndexW, height: 1)
                Text(stepText)
                    .frame(width: Self.colDateW, alignment: .leading)
                HStack(spacing: 6) {
                    Text("Trainer")
                    Text(trainerID)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text("(W: \(trainerScorePct))")
                    Text("-vs-")
                    Text(championID)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text("W:\(record.candidateWins)")
                    Text("D:\(record.draws)(\(drawRatePct))")
                    Text("L:\(record.championWins)")
                    if record.promoted, let pid = record.promotedID {
                        Text("⇒")
                        Text(pid.description)
                            .foregroundStyle(.green)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Spacer(minLength: 8)
                }
            }
            .font(.system(.body, design: .monospaced).weight(.light))
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(rowBackground(scoreOutlier: scoreOutlier))
    }

    private enum ScoreOutlier { case high, low, none }

    private func scoreOutlierKind() -> ScoreOutlier {
        guard record.gamesPlayed > 0 else { return .none }
        // Symmetric "blowout" floor below 0.5: if the candidate
        // scored at least as far below 0.5 as `promoteThreshold` is
        // above it, flag it as a low-side outlier so a quick scan
        // picks out arenas where the candidate got crushed.
        let highCutoff = promoteThreshold
        let lowCutoff = max(0, 1.0 - promoteThreshold)
        if record.score >= highCutoff { return .high }
        if record.score <= lowCutoff { return .low }
        return .none
    }

    @ViewBuilder
    private func rowBackground(scoreOutlier: ScoreOutlier) -> some View {
        if record.promoted {
            Color.green.opacity(0.10)
        } else {
            switch scoreOutlier {
            case .high: Color.green.opacity(0.05)
            case .low:  Color.red.opacity(0.05)
            case .none: Color.clear
            }
        }
    }
}
