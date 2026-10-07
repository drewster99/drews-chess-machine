import SwiftUI

/// One row of the Models table (a `GridRow`): a group with its disclosure
/// button, a checkpoint (indented), or the "No model recorded" row.
struct LichessBotModelRecordRow: View {
    let row: LichessBotModelTableRow
    let countWidth: Int
    @Binding var expanded: Set<String>

    var body: some View {
        GridRow {
            LichessBotModelDisclosureButton(groupModelID: row.groupModelID, expanded: $expanded)
            Text(row.label)
                .font(row.kind == .checkpoint ? LichessBotStatsStyle.noteFont : LichessBotStatsStyle.rowLabelFont)
                .padding(.leading, row.kind == .checkpoint ? 14 : 0)
                .gridColumnAlignment(.leading)
            Text("\(row.tally.scored)")
            LichessBotTallyText(tally: row.tally, countWidth: countWidth)
            Text("\(LichessBotStatsFormat.percent(row.tally.score)) (\(LichessBotStatsFormat.interval(row.interval)))")
                .help("Score, with its 95% Wilson interval (draws count half)")
            Text(LichessBotStatsFormat.estimate(row.performance))
                .help("Performance rating over \(row.ratedOpponentGames) game(s) with a rated opponent")
            Text(LichessBotStatsFormat.average(row.opponentAverage))
            Text(row.mixedGames.map(String.init) ?? LichessBotStatsFormat.missing)
                .help("Games in which another model also chose DCM's moves; each game counts for the model that chose most of them")
            Text(row.firstGame.map { $0.formatted(date: .numeric, time: .omitted) } ?? LichessBotStatsFormat.missing)
            Text(row.lastGame.map { $0.formatted(date: .numeric, time: .omitted) } ?? LichessBotStatsFormat.missing)
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
