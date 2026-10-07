import SwiftUI

/// One short loss (a `GridRow`): opponent, plies, when, and a button that
/// opens it on Lichess.
struct LichessBotShortGameRow: View {
    let game: LichessBotShortGame

    var body: some View {
        GridRow {
            LichessBotResultChip(ourScore: 0)
            Text(game.opponentName ?? "Lichess AI")
                .lineLimit(1)
            Text("\(game.plies) plies")
                .font(LichessBotStatsStyle.numberFont)
                .foregroundStyle(LichessBotStatsStyle.neutral)
            Text(game.createdAt.formatted(date: .numeric, time: .shortened))
                .font(LichessBotStatsStyle.noteFont)
                .foregroundStyle(LichessBotStatsStyle.neutral)
            Button("Open") {
                LichessBotLinks.openGame(game.gameID)
            }
            .font(LichessBotStatsStyle.noteFont)
            .help("Open \(game.gameID) on Lichess")
        }
        .font(.callout)
        .lineLimit(1)
        .fixedSize()
    }
}
