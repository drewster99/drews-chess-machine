import SwiftUI

/// One held game (a `GridRow`): its result, opponent, the ply where the
/// hold began, when, and a button that opens it on Lichess.
struct LichessBotHeldGameRow: View {
    let game: LichessBotHeldGame

    var body: some View {
        GridRow {
            LichessBotResultChip(ourScore: game.ourScore)
                .help("DCM's result in this game")
            Text(game.opponentName ?? "Lichess AI")
                .lineLimit(1)
            Text("from ply \(game.startPly)")
                .font(LichessBotStatsStyle.numberFont)
                .foregroundStyle(LichessBotStatsStyle.neutral)
                .help(Self.startPlyHelp(game.startPly))
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

    /// Where the hold began, and which move that ply is (plies count both
    /// sides' moves from 0: White moves on even plies, and ply p is move
    /// p ÷ 2 + 1).
    static func startPlyHelp(_ ply: Int) -> String {
        let side = ply.isMultiple(of: 2) ? "White" : "Black"
        return "Where the network first held this result: the first of two consecutive DCM moves at which the value head gave it ≥ 80%. "
            + "A ply is one move by either side; ply \(ply) is \(side)'s move \(ply / 2 + 1)."
    }
}
