import SwiftUI

/// One period's row of the period table (a `GridRow`). Every number is
/// padded to its column's width over all filters, so the columns keep their
/// width when the filter changes.
struct LichessBotRecordPeriodRow: View {
    let period: LichessBotStatsPeriod
    let row: LichessBotPeriodStatistics
    let widths: LichessBotRecordPeriodColumnWidths

    var body: some View {
        GridRow {
            LichessBotRowLabel(text: period.label)
                .help(period.help)
            Text(LichessBotStatsFormat.padded(row.record.all.scored, width: widths.games))
            LichessBotTallyText(tally: row.record.all, countWidth: widths.count)
            Text(LichessBotStatsFormat.padded(LichessBotStatsFormat.percent(row.record.all.score), width: widths.score))
            Text(LichessBotStatsFormat.padded(LichessBotStatsFormat.estimate(row.performance), width: widths.performance))
                .help("Performance rating (Elo maximum likelihood) over \(row.ratedOpponentGames) game(s) with a rated opponent; ≥ / ≤ for a perfect / zero score. Mixes Lichess' per-speed rating pools; Time controls has each pool's own.")
            Text(LichessBotStatsFormat.padded(LichessBotStatsFormat.average(row.opponentAverage), width: widths.opponentAverage))
                .help("Average opponent rating over \(row.ratedOpponentGames) game(s); Lichess AI games have no rating and are left out")
            LichessBotTallyText(tally: row.record.versusBots, countWidth: widths.count)
            LichessBotTallyText(tally: row.record.versusHumans, countWidth: widths.count)
            LichessBotTallyText(tally: row.record.asWhite, countWidth: widths.count)
            LichessBotTallyText(tally: row.record.asBlack, countWidth: widths.count)
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
