import SwiftUI

/// Self Play stats card — Overall vs Kept side-by-side. Replaces the
/// per-line text block that used to live inside `playAndTrainStatsText`.
/// Visual style matches the upper status bar (`StatusBarCell` typography
/// inside a `Color.secondary.opacity(0.10)` rounded-rect shell).
///
/// The "Kept" column reflects what the draw-keep filter
/// (`selfPlayDrawKeepFraction`) actually pushed into the replay buffer:
/// Games and Moves diverge from Overall whenever the filter is < 1.0;
/// Avg move / Avg game share the same wall-time denominator as Overall
/// so the per-emitted-move and per-emitted-game costs read as the
/// effective production cost (slower per kept unit because some plies
/// were thrown away).
///
/// Rates are 1-minute rolling — `ParallelWorkerStatsBox.recentWindow`
/// is 60 s, with the Played counters feeding "Overall" and the
/// Emitted counters feeding "Kept."
struct SelfPlayStatsCard: View {
    let snapshot: ParallelWorkerStatsBox.Snapshot?
    let modelID: String

    private static let dash = "—"

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Self Play [\(modelID)]")
                .font(.caption2)
                .foregroundStyle(.secondary)

            // Header strip — empty label cell + two right-aligned
            // column titles aligned with the numeric value cells below.
            HStack(spacing: 0) {
                Text("")
                    .frame(width: Self.labelWidth, alignment: .trailing)
                Text("Overall")
                    .frame(width: Self.valueWidth, alignment: .trailing)
                Text("Kept")
                    .frame(width: Self.valueWidth, alignment: .trailing)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .padding(.bottom, 1)

            row(label: "Games", overall: gamesOverall, kept: gamesKept)
            row(label: "Moves", overall: movesOverall, kept: movesKept)
            row(label: "Avg move", overall: avgMoveOverall, kept: avgMoveKept)
            row(label: "Avg game", overall: avgGameOverall, kept: avgGameKept)
            row(label: "Moves/hr", overall: movesPerHourOverall, kept: movesPerHourKept)
            row(label: "Games/hr", overall: gamesPerHourOverall, kept: gamesPerHourKept)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color.secondary.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Layout

    private static let labelWidth: CGFloat = 80
    private static let valueWidth: CGFloat = 92

    @ViewBuilder
    private func row(label: String, overall: String, kept: String) -> some View {
        HStack(spacing: 0) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: Self.labelWidth, alignment: .trailing)
            Text(overall)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.valueWidth, alignment: .trailing)
            Text(kept)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.valueWidth, alignment: .trailing)
        }
    }

    // MARK: - Cell content

    private var gamesOverall: String {
        guard let s = snapshot, s.selfPlayGames > 0 else { return Self.dash }
        return numberString(s.selfPlayGames)
    }
    private var gamesKept: String {
        guard let s = snapshot, s.emittedGames > 0 else { return Self.dash }
        return numberString(s.emittedGames)
    }
    private var movesOverall: String {
        guard let s = snapshot, s.selfPlayPositions > 0 else { return Self.dash }
        return numberString(s.selfPlayPositions)
    }
    private var movesKept: String {
        guard let s = snapshot, s.emittedPositions > 0 else { return Self.dash }
        return numberString(s.emittedPositions)
    }

    /// "Overall" Avg move = wall time / total moves. "Kept" Avg move =
    /// same wall time / kept moves. Both share the elapsed-since-
    /// session-start denominator so "Kept" naturally reads larger
    /// (the system spent the same wall time but a smaller fraction
    /// of plies survived the keep filter).
    private var avgMoveOverall: String {
        guard let s = snapshot, s.selfPlayPositions > 0,
              let elapsedMs = elapsedMs(for: s) else { return Self.dash }
        return msString(elapsedMs / Double(s.selfPlayPositions))
    }
    private var avgMoveKept: String {
        guard let s = snapshot, s.emittedPositions > 0,
              let elapsedMs = elapsedMs(for: s) else { return Self.dash }
        return msString(elapsedMs / Double(s.emittedPositions))
    }
    private var avgGameOverall: String {
        guard let s = snapshot, s.selfPlayGames > 0,
              let elapsedMs = elapsedMs(for: s) else { return Self.dash }
        return msString(elapsedMs / Double(s.selfPlayGames))
    }
    private var avgGameKept: String {
        guard let s = snapshot, s.emittedGames > 0,
              let elapsedMs = elapsedMs(for: s) else { return Self.dash }
        return msString(elapsedMs / Double(s.emittedGames))
    }

    /// 1-minute rolling rate using the box's `recentWindow` (currently
    /// 60 s). Stays at "—" until the window has at least one entry.
    private var movesPerHourOverall: String {
        guard let s = snapshot, s.recentWindowSeconds > 0,
              s.recentMoves > 0 else { return Self.dash }
        return numberString(Int((Double(s.recentMoves) / s.recentWindowSeconds * 3600).rounded()))
    }
    private var movesPerHourKept: String {
        guard let s = snapshot, s.recentWindowSeconds > 0,
              s.recentEmittedPositions > 0 else { return Self.dash }
        return numberString(Int((Double(s.recentEmittedPositions) / s.recentWindowSeconds * 3600).rounded()))
    }
    private var gamesPerHourOverall: String {
        guard let s = snapshot, s.recentWindowSeconds > 0,
              s.recentGames > 0 else { return Self.dash }
        return numberString(Int((Double(s.recentGames) / s.recentWindowSeconds * 3600).rounded()))
    }
    private var gamesPerHourKept: String {
        guard let s = snapshot, s.recentWindowSeconds > 0,
              s.recentEmittedGames > 0 else { return Self.dash }
        return numberString(Int((Double(s.recentEmittedGames) / s.recentWindowSeconds * 3600).rounded()))
    }

    // MARK: - Formatters

    /// Wall-clock milliseconds since `sessionStart` — the denominator
    /// shared by Overall and Kept Avg-move/Avg-game so the two columns
    /// describe "the same wall time, divided by a different numerator."
    private func elapsedMs(for s: ParallelWorkerStatsBox.Snapshot) -> Double? {
        let elapsed = Date().timeIntervalSince(s.sessionStart)
        guard elapsed > 0 else { return nil }
        return elapsed * 1000
    }

    private func numberString(_ n: Int) -> String {
        n.formatted(.number.grouping(.automatic))
    }

    /// "0.47 ms" / "103.9 ms" / "1,234 ms" — picks the precision that
    /// keeps a stable column width across the realistic range of
    /// per-move (sub-ms) and per-game (ms-to-multi-second) values.
    private func msString(_ ms: Double) -> String {
        if ms < 10 {
            return String(format: "%.2f ms", ms)
        }
        if ms < 1000 {
            return String(format: "%.1f ms", ms)
        }
        let intMs = Int(ms.rounded())
        return "\(intMs.formatted(.number.grouping(.automatic))) ms"
    }
}
