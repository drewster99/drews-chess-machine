import SwiftUI

/// D1 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §11): DCM's results by how the
/// game began — accepted challenges, the Challenge sheet, the queue,
/// matchmaking, tournaments, unknown — from the controller's one origin
/// resolver, which also covers games played before origins were recorded.
struct LichessBotOriginsPane: View {
    /// Nil until the origins have loaded.
    let origins: LichessBotOriginStatistics?

    static let titles = ["Origin", "Games", "W–D–L", "Score", "Perf"]

    var body: some View {
        let rows = origins?.rows ?? []
        let countWidth = LichessBotStatsFormat.countWidth(rows.map(\.tally))
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            LichessBotPaneEmptyNote(text: "Game origins are not loaded yet")
                .shown(origins == nil)
            ScrollView(.horizontal) {
                Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                    LichessBotStatsHeaderRow(titles: Self.titles)
                    ForEach(rows) { row in
                        LichessBotOriginRowView(row: row, countWidth: countWidth)
                    }
                }
            }
            LichessBotPaneEmptyNote(text: "\(origins?.gamesUnresolved ?? 0) scored game(s) had no resolved origin")
                .shown(origins.map { $0.gamesUnresolved > 0 } == true)
        }
    }
}
