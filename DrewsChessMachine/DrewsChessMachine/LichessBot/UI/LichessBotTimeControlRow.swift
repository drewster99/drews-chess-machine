import SwiftUI

/// One speed's row of the Time controls table (a `GridRow`). A speed with no
/// game in the selected period (nil `selected`) shows 0 games and "–" for
/// its ratios; one with no rated game today or this week shows "–" there.
struct LichessBotTimeControlRow: View {
    let speed: String
    let selected: LichessBotTimeControlStatistics?
    let today: LichessBotTimeControlStatistics?
    let week: LichessBotTimeControlStatistics?
    let trend: [LichessBotRatingPoint]
    let accountLoaded: Bool
    let perf: LichessBotPerfRating?
    let countWidth: Int

    var body: some View {
        let tally = selected?.tally ?? LichessBotResultTally()
        GridRow {
            LichessBotRowLabel(text: speed)
            Text(LichessBotStatsFormat.accountRating(perf))
                .help(accountLoaded ? "Lichess' current rating in \(speed)\(perf?.prov == true ? " (provisional)" : "")" : "Account not loaded")
            Text(LichessBotStatsFormat.ratingChange(today?.ratingChange ?? LichessBotRatingChange()))
                .help(LichessBotStatsFormat.ratingChangeHelp(today?.ratingChange ?? LichessBotRatingChange()))
            Text(LichessBotStatsFormat.ratingChange(week?.ratingChange ?? LichessBotRatingChange()))
                .help(LichessBotStatsFormat.ratingChangeHelp(week?.ratingChange ?? LichessBotRatingChange()))
            Text("\(tally.scored)")
            LichessBotTallyText(tally: tally, countWidth: countWidth)
            Text(LichessBotStatsFormat.score(tally.score))
            Text(LichessBotStatsFormat.estimate(selected?.performance ?? .none))
                .help("Performance rating over \(selected?.ratedOpponentGames ?? 0) game(s) with a rated opponent, within this pool")
            Text(LichessBotStatsFormat.ratingChange(selected?.ratingChange ?? LichessBotRatingChange()))
                .help(LichessBotStatsFormat.ratingChangeHelp(selected?.ratingChange ?? LichessBotRatingChange()))
            LichessBotRatingSparkline(series: LichessBotRatingSparklineSeries(points: trend, currentRating: accountLoaded ? perf?.rating : nil))
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
