import SwiftUI

/// Item 4 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.8): DCM's record per model
/// ID (one CLI process, one lineage segment), newest last game first, each
/// group expanding into its checkpoints by training step. A game belongs to
/// the model that chose most of DCM's moves in it; "Mixed" counts the games
/// another model also played. Games in which no DCM move names a model have
/// their own row at the bottom.
struct LichessBotModelRecordTable: View {
    let models: LichessBotModelStatistics
    /// Model IDs whose checkpoints are shown. Kept here, so it survives
    /// recomputes and tab switches (the panel keeps every pane mounted).
    @State private var expanded: Set<String> = []

    static let titles = ["", "Model", "Games", "W–D–L", "Score (95%)", "Perf", "Opp avg", "Mixed", "First", "Last"]

    var body: some View {
        let rows = LichessBotModelTableRow.rows(models, expanded: expanded)
        let countWidth = LichessBotStatsFormat.countWidth(models.groups.map(\.line.tally) + [models.noModelRecorded])
        Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
            LichessBotStatsHeaderRow(titles: Self.titles)
            ForEach(rows) { row in
                LichessBotModelRecordRow(row: row, countWidth: countWidth, expanded: $expanded)
            }
        }
    }
}
