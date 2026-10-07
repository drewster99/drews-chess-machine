import SwiftUI

/// The selected pane, for the selected period and filter. Every pane stays
/// in the hierarchy, shown only when selected, so switching tabs keeps each
/// pane's state (the Models table's expanded groups).
struct LichessBotRecordPanel: View {
    /// Nil while the account is not loaded.
    let account: LichessBotAccount?
    let pipeline: LichessBotRecordStatisticsPipeline
    let statistics: LichessBotRecordStatistics

    var body: some View {
        let filter = pipeline.rememberedFilter
        let period = pipeline.rememberedPeriod
        let pane = pipeline.rememberedPane
        let hasGames = statistics[filter].periodRows[period].record.all.games > 0
        let breakdowns = statistics[filter].byPeriod[period]
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            // One "empty" line for every pane: a period without a single
            // game (scored or not) has nothing to break down.
            LichessBotPaneEmptyNote(text: "No games in this period")
                .shown(!hasGames)
            ZStack(alignment: .topLeading) {
                // Time controls also lists speeds the account is rated in,
                // so it shows even in an empty period.
                ScrollView(.horizontal) {
                    LichessBotTimeControlTable(statistics: statistics, filter: filter, period: period, account: account)
                }
                .shown(pane == .timeControls)
                LichessBotModelsPane(models: breakdowns.models)
                    .shown(pane == .models)
                LichessBotSelfAssessmentPane(assessment: breakdowns.selfAssessment)
                    .shown(pane == .selfAssessment)
            }
        }
    }
}
