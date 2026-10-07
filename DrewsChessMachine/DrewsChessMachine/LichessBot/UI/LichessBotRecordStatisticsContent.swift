import SwiftUI

/// The statistics for the selected filter: the period table, its footnote,
/// the panel's pickers, and the selected pane. The table, footnote and
/// header take their ideal heights; the pane takes the rest of the card's
/// height and scrolls, so a tall pane never overflows the card. The pane is
/// never shorter than `LichessBotStatsStyle.paneMinimumHeight`: the card
/// grows instead, so the pickers always show what they select.
struct LichessBotRecordStatisticsContent: View {
    /// Nil while the account is not loaded.
    let account: LichessBotAccount?
    let pipeline: LichessBotRecordStatisticsPipeline
    let statistics: LichessBotRecordStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            LichessBotRecordPeriodTable(rows: statistics[pipeline.rememberedFilter].periodRows, widths: statistics.periodColumnWidths)
            LichessBotRecordFootnote(statistics: statistics[pipeline.rememberedFilter])
            LichessBotRecordPanelHeader(pipeline: pipeline)
            ScrollView(.vertical) {
                LichessBotRecordPanel(account: account, pipeline: pipeline, statistics: statistics)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(minHeight: LichessBotStatsStyle.paneMinimumHeight)
        }
    }
}
