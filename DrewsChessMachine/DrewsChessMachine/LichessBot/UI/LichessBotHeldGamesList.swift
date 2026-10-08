import SwiftUI

/// The most recent blown wins (or saves): games in which the value head
/// held a win (or a loss) at ≥ 0.80 for two consecutive DCM moves and the
/// result went the other way (§3.6, OD-14), each with a button that opens
/// it on Lichess.
struct LichessBotHeldGamesList: View {
    let title: String
    /// What the list holds, for the title's tooltip.
    let help: String
    let games: [LichessBotHeldGame]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(LichessBotStatsStyle.sectionFont)
                .help(help)
            LichessBotPaneEmptyNote(text: "None")
                .shown(games.isEmpty)
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 4) {
                ForEach(games) { game in
                    LichessBotHeldGameRow(game: game)
                }
            }
        }
    }
}
