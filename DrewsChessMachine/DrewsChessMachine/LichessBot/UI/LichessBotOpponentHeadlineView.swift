import SwiftUI

/// The detail window's headline: DCM's overall record against the opponent,
/// current and longest streaks, the highest-rated win, and when DCM first
/// and last played them. Every line is text, hidden when it has nothing to
/// say, so the view needs no `some View` helpers or branches in `body`.
struct LichessBotOpponentHeadlineView: View {
    let stats: LichessBotOpponentProfileStats?

    var body: some View {
        GroupBox("DCM vs them") {
            VStack(alignment: .leading, spacing: 4) {
                Text(summaryLine)
                    .font(.callout.weight(.medium))
                    .help(headToHeadHelp)
                Text(ratingLine)
                    .help(ratingHelp)
                    .shown(hasGames)
                Text(streakLine)
                    .foregroundStyle(.secondary)
                    .shown(hasGames)
                Text(notableLine)
                    .foregroundStyle(.secondary)
                    .shown(stats?.highestRatedWin != nil)
                Text(playedLine)
                    .foregroundStyle(.secondary)
                    .shown(hasGames)
            }
            .font(.callout)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private var hasGames: Bool { stats?.games.isEmpty == false }

    private var summaryLine: String {
        guard let stats else { return "Loading games…" }
        guard hasGames else { return "No games filed against this opponent yet." }
        return LichessBotStatsFormat.headToHead(stats.headToHead)
    }

    private var headToHeadHelp: String {
        stats.map { LichessBotStatsFormat.headToHeadHelp($0.headToHead) } ?? ""
    }

    private var ratingLine: String {
        guard let stats else { return "" }
        return "Rating Δ: \(LichessBotStatsFormat.ratingChange(stats.headToHead.ratingChange))"
    }

    private var ratingHelp: String {
        stats.map { LichessBotStatsFormat.ratingChangeHelp($0.headToHead.ratingChange) } ?? ""
    }

    private var streakLine: String {
        guard let stats else { return "" }
        return "Current: \(LichessBotStatsFormat.streak(stats.currentStreak)) · longest win \(stats.longestWinStreak) · longest loss \(stats.longestLossStreak)"
    }

    private var notableLine: String {
        guard let win = stats?.highestRatedWin else { return "" }
        return "Best win: beat \(win.opponentName ?? "them") at \(win.rating), \(win.createdAt.formatted(date: .abbreviated, time: .omitted))"
    }

    private var playedLine: String {
        guard let first = stats?.firstPlayedAt, let last = stats?.lastPlayedAt else { return "" }
        return "First played \(first.formatted(date: .abbreviated, time: .omitted)) · last \(last.formatted(.relative(presentation: .named)))"
    }
}
