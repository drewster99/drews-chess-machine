import SwiftUI

/// Item 1 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.3) inside a horizontal
/// scroll view, so the table never clips at the narrowest window.
struct LichessBotRecordPeriodTable: View {
    let rows: LichessBotPeriodValues<LichessBotPeriodStatistics>
    let widths: LichessBotRecordPeriodColumnWidths

    var body: some View {
        ScrollView(.horizontal) {
            LichessBotRecordPeriodGrid(rows: rows, widths: widths)
                .padding(.bottom, 2)
        }
    }
}
