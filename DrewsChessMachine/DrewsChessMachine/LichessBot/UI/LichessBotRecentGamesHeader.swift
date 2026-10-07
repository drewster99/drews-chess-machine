import SwiftUI

/// The recent games' title, with "More…", which opens every game in a
/// sortable window.
struct LichessBotRecentGamesHeader: View {
    let controller: LichessBotController

    var body: some View {
        HStack {
            Text("Recent games")
                .font(LichessBotStatsStyle.headerFont)
                .foregroundStyle(LichessBotStatsStyle.neutral)
            Spacer()
            Button("More…") {
                LichessBotAllGamesWindowController.open(controller: controller)
            }
            .font(LichessBotStatsStyle.noteFont)
        }
    }
}
