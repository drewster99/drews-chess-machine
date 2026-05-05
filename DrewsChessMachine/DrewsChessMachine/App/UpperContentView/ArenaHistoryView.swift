import SwiftUI

/// Sheet that displays the full per-session arena history in a
/// 2-line tabular layout. One row per completed tournament,
/// newest at the top so the most recent result is visible without
/// scrolling.
///
/// Row layout:
///   Line 1 — index, finished-at datetime, W–D–L, AlphaZero score,
///            Elo with 95% CI, verdict (kept / PROMOTED).
///   Line 2 — finishedAtStep, games played, duration, per-side
///            white/black scores, candidate→champion IDs (with a
///            promoted-ID tail when applicable).
///
/// Outliers (promotions, scores ≥ promoteThreshold even when not
/// promoted, scores ≤ a "lost-by-a-lot" floor) are highlighted with
/// color so a quick scan picks them out.
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

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .medium
        return f
    }()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Arena History")
                    .font(.title2.weight(.semibold))
                Spacer()
                Text("\(history.count) tournament\(history.count == 1 ? "" : "s")")
                    .font(.callout)
                    .foregroundStyle(.secondary)
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
        .frame(minWidth: 860, idealWidth: 980, minHeight: 380, idealHeight: 560)
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

    var body: some View {
        let elo = record.eloSummary
        let scoreCI = ArenaEloStats.formatScorePercentWithCI(elo)
        let eloCI = ArenaEloStats.formatEloWithCI(elo)
        let dateText = record.finishedAt.map { dateFormatter.string(from: $0) } ?? "—"
        let wdl = "\(record.candidateWins)–\(record.draws)–\(record.championWins)"

        let gamesText = "\(record.gamesPlayed)/\(configuredGamesPerTournament)"
        let stepText = record.finishedAtStep.formatted(.number.grouping(.automatic))
        let durationText = ArenaLogFormatter.formatDuration(record.durationSec)

        let whiteN = record.candidateWinsAsWhite + record.candidateLossesAsWhite + record.candidateDrawsAsWhite
        let blackN = record.candidateWinsAsBlack + record.candidateLossesAsBlack + record.candidateDrawsAsBlack
        let whiteScoreStr: String = whiteN > 0
            ? String(format: "W %.0f%%", record.candidateScoreAsWhite * 100)
            : "W —"
        let blackScoreStr: String = blackN > 0
            ? String(format: "B %.0f%%", record.candidateScoreAsBlack * 100)
            : "B —"

        let scoreOutlier = scoreOutlierKind()
        let verdict = ArenaLogFormatter.formatVerdict(record: record)
        let verdictColor: Color = record.promoted ? .green : .secondary

        // Inline-init to a tuple-typed closure: avoids the
        // ViewBuilder treating a top-level `switch` as a view-tree
        // branch (`type '()' cannot conform to 'View'`).
        let scoreColor: Color = {
            switch scoreOutlier {
            case .high: return .green
            case .low:  return .red
            case .none: return .primary
            }
        }()

        let candidateText = record.candidateID?.description ?? "—"
        let championText = record.championID?.description ?? "—"

        VStack(alignment: .leading, spacing: 3) {
            // Line 1 — headline: index, datetime, W-D-L, score CI,
            // Elo CI, verdict.
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text("#\(index)")
                    .font(.system(.body, design: .monospaced).weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 48, alignment: .leading)
                Text(dateText)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 168, alignment: .leading)
                Text(wdl)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 78, alignment: .leading)
                Text(scoreCI)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(scoreColor)
                    .frame(width: 200, alignment: .leading)
                Text(eloCI)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                Text(verdict)
                    .font(.system(.body, design: .monospaced).weight(.semibold))
                    .foregroundStyle(verdictColor)
            }
            // Line 2 — supporting metadata: step, games, duration,
            // per-side scores, candidate→champion IDs (with a
            // promoted-ID tail when applicable).
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("step \(stepText)")
                    .frame(width: 144, alignment: .leading)
                Text("games \(gamesText)")
                    .frame(width: 92, alignment: .leading)
                Text("dur \(durationText)")
                    .frame(width: 70, alignment: .leading)
                Text("\(whiteScoreStr)  \(blackScoreStr)")
                    .frame(width: 130, alignment: .leading)
                Text("\(candidateText) → \(championText)")
                    .lineLimit(1)
                    .truncationMode(.middle)
                if record.promoted, let pid = record.promotedID {
                    Text("⇒")
                    Text(pid.description)
                        .foregroundStyle(.green)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 8)
            }
            .font(.system(.caption, design: .monospaced))
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
