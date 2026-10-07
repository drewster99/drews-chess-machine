import SwiftUI

/// One period's row of the period table (a `GridRow`).
struct LichessBotRecordPeriodRow: View {
    let period: LichessBotStatsPeriod
    let row: LichessBotPeriodStatistics
    let countWidth: Int

    var body: some View {
        GridRow {
            Text(period.label)
                .font(LichessBotStatsStyle.rowLabelFont)
                .gridColumnAlignment(.leading)
                .help(period == .lastHour ? "The 60 minutes before now (not the clock hour), so just after midnight it can hold games Today does not" : "Games started since the start of this period, in this Mac's calendar and time zone")
            Text("\(row.record.all.scored)")
            LichessBotTallyText(tally: row.record.all, countWidth: countWidth)
            Text(LichessBotStatsFormat.percent(row.record.all.score))
            Text(LichessBotStatsFormat.estimate(row.performance))
                .help("Performance rating (Elo maximum likelihood) over \(row.ratedOpponentGames) game(s) with a rated opponent; ≥ / ≤ for a perfect / zero score. Mixes Lichess's per-speed rating pools; Time controls has each pool's own.")
            Text(LichessBotStatsFormat.average(row.opponentAverage))
                .help("Average opponent rating over \(row.ratedOpponentGames) game(s); Lichess AI games have no rating and are left out")
            Text(LichessBotStatsFormat.ratingChange(row.ratingChange))
                .help(LichessBotStatsFormat.ratingChangeHelp(row.ratingChange))
            LichessBotTallyText(tally: row.record.versusBots, countWidth: countWidth)
            LichessBotTallyText(tally: row.record.versusHumans, countWidth: countWidth)
            LichessBotTallyText(tally: row.record.asWhite, countWidth: countWidth)
            LichessBotTallyText(tally: row.record.asBlack, countWidth: countWidth)
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
