import SwiftUI

/// L6 (§11): the bot's own health over every game of the period, scored or
/// not — anomalies the record builder noted, moves Lichess refused, game
/// stream reconnections, and games whose journal the export corrected or
/// that were filed without an export.
struct LichessBotBotHealthPane: View {
    let health: LichessBotHealthStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                LichessBotHealthRow(label: "Games", value: health.games, help: "Every filed game in the period, scored or not")
                LichessBotHealthRow(label: "Anomalies", value: health.anomalies, help: "Notes the record builder made (\(health.gamesWithAnomalies) game(s) have any)")
                LichessBotHealthRow(label: "Rejected moves", value: health.rejectedMoves, help: "Moves Lichess refused")
                LichessBotHealthRow(label: "Stream reconnects", value: health.streamReconnects, help: "Game-stream connections after the first")
                LichessBotHealthRow(label: "Corrected by the export", value: health.reconciliationCorrected, help: "Games whose journal disagreed with Lichess's export; the export's values were used")
                LichessBotHealthRow(label: "Filed without an export", value: health.exportUnavailable, help: "Games Lichess kept no export for; the journal is the only source")
            }
            LichessBotPaneEmptyNote(text: "\(health.rowsWithoutFacts) game\(health.rowsWithoutFacts == 1 ? "" : "s") without facts: their rejected moves and reconnects are not counted")
                .shown(health.rowsWithoutFacts > 0)
        }
    }
}
