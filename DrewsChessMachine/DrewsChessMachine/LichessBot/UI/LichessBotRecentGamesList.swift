import SwiftUI

/// DCM's most recent filed games, newest first: result chip, color, the
/// opponent and their kind and rating, the game's speed, and when. It
/// scrolls within the card's height; "More…" opens every game in a
/// sortable window.
struct LichessBotRecentGamesList: View {
    /// A bound on what the card lists; the window lists every game.
    static let maximumRows = 200
    let controller: LichessBotController
    let rows: [LichessBotGameSummary]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Recent games")
                    .font(LichessBotStatsStyle.headerFont)
                    .foregroundStyle(LichessBotStatsStyle.neutral)
                Spacer()
                Button("More…") {
                    LichessBotAllGamesWindowController.open(controller: controller)
                }
                .font(.caption)
            }
            Text("No games filed yet")
                .foregroundStyle(LichessBotStatsStyle.neutral)
                .shown(rows.isEmpty)
            ScrollView {
                Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 4) {
                    ForEach(rows, id: \.gameID) { row in
                        LichessBotRecentGameRow(controller: controller, row: row)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}
