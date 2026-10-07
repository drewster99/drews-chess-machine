import SwiftUI

/// The statistics for the selected filter: the period table, its footnote,
/// the panel's pickers, and the selected pane. The table, footnote and
/// header take their ideal heights; the pane takes the rest of the card's
/// fixed height and scrolls, so a tall pane never overflows the card.
struct LichessBotRecordStatisticsContent: View {
    let controller: LichessBotController
    let pipeline: LichessBotRecordStatisticsPipeline
    let statistics: LichessBotRecordStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            LichessBotRecordPeriodTable(rows: statistics[pipeline.rememberedFilter].periodRows)
            LichessBotRecordFootnote(statistics: statistics[pipeline.rememberedFilter])
            LichessBotRecordPanelHeader(pipeline: pipeline)
            ScrollView(.vertical) {
                LichessBotRecordPanel(controller: controller, pipeline: pipeline, statistics: statistics)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        }
    }
}
