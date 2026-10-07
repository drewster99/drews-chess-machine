import SwiftUI

/// The Record card's statistics column in its three states: loading, the
/// statistics, or why they can't be computed. Each is its own child shown
/// for its state (no `switch` in the body), so the statistics keep their
/// identity — and the panes their state — across recomputes.
struct LichessBotRecordStatisticsColumn: View {
    /// Nil while the account is not loaded.
    let account: LichessBotAccount?
    let pipeline: LichessBotRecordStatisticsPipeline

    var body: some View {
        ZStack(alignment: .topLeading) {
            Text("Loading the game records…")
                .foregroundStyle(LichessBotStatsStyle.neutral)
                .shown(pipeline.state.isLoading)
            Text(pipeline.state.failureText ?? "")
                .foregroundStyle(LichessBotStatsStyle.failure)
                .shown(pipeline.state.failureText != nil)
            ForEach(pipeline.state.readyItems) { item in
                LichessBotRecordStatisticsContent(account: account, pipeline: pipeline, statistics: item.statistics)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
