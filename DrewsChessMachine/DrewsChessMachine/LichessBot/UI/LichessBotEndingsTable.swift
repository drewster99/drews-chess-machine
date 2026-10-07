import SwiftUI

/// Item 5 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.9): how games ended, one
/// row per ending, with DCM's wins, draws and losses in it and each count's
/// share of its column. Only endings with games are listed; statuses this
/// build does not name keep Lichess's spelling ("Other: …"); games without a
/// result are the "Not counted" row.
struct LichessBotEndingsTable: View {
    let endings: LichessBotEndingStatistics

    static let titles = ["", "DCM won", "Drew", "DCM lost", "Not counted"]

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
            LichessBotStatsHeaderRow(titles: Self.titles)
            ForEach(endings.rows) { row in
                LichessBotEndingRowView(row: row, totals: endings)
            }
        }
    }
}
