import SwiftUI

/// One origin's row of the Origins table (a `GridRow`): the origin's glyph
/// and short label as every other view draws them
/// (`LichessBotGameOriginStyle`), then DCM's results in those games.
struct LichessBotOriginRowView: View {
    let row: LichessBotOriginRow
    let countWidth: Int

    var body: some View {
        GridRow {
            Label(LichessBotGameOriginStyle.shortLabel(for: row.category), systemImage: LichessBotGameOriginStyle.systemImage(for: row.category))
                .font(LichessBotStatsStyle.rowLabelFont)
                .gridColumnAlignment(.leading)
                .help(LichessBotGameOriginStyle.longLabel(for: row.category))
            Text("\(row.tally.scored)")
            LichessBotTallyText(tally: row.tally, countWidth: countWidth)
            Text(LichessBotStatsFormat.score(row.tally.score))
            Text(LichessBotStatsFormat.estimate(row.performance))
                .help("Performance rating over \(row.ratedOpponentGames) game(s) with a rated opponent")
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
