import SwiftUI

/// The selected pane, for the selected period and filter. Every pane stays
/// in the hierarchy, shown only when selected, so switching tabs keeps each
/// pane's state (the Models table's expanded groups).
struct LichessBotRecordPanel: View {
    let controller: LichessBotController
    let pipeline: LichessBotRecordStatisticsPipeline
    let statistics: LichessBotRecordStatistics

    var body: some View {
        let filtered = statistics[pipeline.rememberedFilter]
        ZStack(alignment: .topLeading) {
            // One "empty" line for every pane: a period without a single
            // game (scored or not) has nothing to break down.
            LichessBotPaneEmptyNote(text: "No games in this period")
                .shown(filtered.periodRows[pipeline.rememberedPeriod].record.all.games == 0)
        }
    }
}
