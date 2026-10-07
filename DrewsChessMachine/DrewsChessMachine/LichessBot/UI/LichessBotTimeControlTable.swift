import SwiftUI

/// Item 2 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.5): one row per time
/// control — the account's current rating there, the rating change today
/// and this week, the selected period's games, W–D–L, score and performance
/// rating, and a sparkline of the recorded ratings. Each speed is its own
/// Lichess rating pool, so these are the per-pool numbers the all-speed
/// period table mixes.
struct LichessBotTimeControlTable: View {
    let statistics: LichessBotRecordStatistics
    let filter: LichessBotStatsFilter
    let period: LichessBotStatsPeriod
    /// Nil while the account is not loaded (no token, or before the first
    /// fetch).
    let account: LichessBotAccount?

    static let titles = ["", "Rating", "Today ±", "Week ±", "Games", "W–D–L", "Score", "Perf", "Trend"]

    var body: some View {
        let filtered = statistics[filter]
        let selected = filtered.byPeriod[period].timeControls
        let speeds = LichessBotTimeControlOrder.speeds(recordSpeeds: statistics.recordSpeeds, accountSpeeds: Set(account?.perfs?.keys.map { $0 } ?? []))
        let countWidth = LichessBotStatsFormat.countWidth(selected.values.map(\.tally))
        Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
            LichessBotStatsHeaderRow(titles: Self.titles)
            ForEach(speeds, id: \.self) { speed in
                LichessBotTimeControlRow(
                    speed: speed,
                    selected: selected[speed],
                    today: filtered.byPeriod.today.timeControls[speed],
                    week: filtered.byPeriod.thisWeek.timeControls[speed],
                    trend: statistics.ratingTrends[speed] ?? [],
                    accountLoaded: account != nil,
                    perf: account?.perfs?[speed],
                    countWidth: countWidth
                )
            }
        }
    }
}
