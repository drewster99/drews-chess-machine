import SwiftUI

/// The Record card's content: the statistics column and the recent games,
/// side by side or stacked (`LichessBotRecordCardLayout`).
struct LichessBotRecordCardContent: View {
    let controller: LichessBotController

    var body: some View {
        LichessBotRecordCardLayout<LichessBotRecordStatisticsColumn, LichessBotRecentGamesSection>(
            statistics: LichessBotRecordStatisticsColumn(account: controller.account, pipeline: controller.recordStatistics),
            recentGames: LichessBotRecentGamesSection(controller: controller)
        )
    }
}
