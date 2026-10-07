import SwiftUI

/// One held game (a `GridRow`): its result, opponent, the ply where the
/// hold began, when, and a button that opens it on Lichess.
struct LichessBotHeldGameRow: View {
    let game: LichessBotHeldGame

    var body: some View {
        GridRow {
            LichessBotResultChip(ourScore: game.ourScore)
            Text(game.opponentName ?? "Lichess AI")
                .lineLimit(1)
            Text("from ply \(game.startPly)")
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
        .font(LichessBotStatsStyle.rowFont)
        .lineLimit(1)
        .fixedSize()
    }
}
