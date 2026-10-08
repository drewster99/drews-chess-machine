import SwiftUI

/// The statistics for the selected filter: the period table, its footnote,
/// the panel's pickers, and the selected pane. Every part takes its full
/// height: a pane taller than the card's dragged height grows the card
/// (`LichessBotAtLeastHeightLayout`; the Overview itself scrolls) instead of
/// scrolling inside it, so nothing is cut off where the card ends (owner
/// decision 2026-10-08: a pane's own scroll area cut the Opponent strength
/// chart off below its tick labels). The pane is never shorter than
/// `LichessBotStatsStyle.paneMinimumHeight`, so the pickers always show what
/// they select; a card dragged taller leaves the extra space below it.
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
            // At least the pane minimum, and taller when the pane needs it.
            // Not `.frame(minHeight:)`: offered less than its content, such
            // a frame reports the offered height and lets the pane spill
            // over whatever is laid out below it (the recent games).
            LichessBotAtLeastHeightLayout(minimumHeight: LichessBotStatsStyle.paneMinimumHeight) {
                LichessBotRecordPanel(account: account, pipeline: pipeline, statistics: statistics)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        }
    }
}
