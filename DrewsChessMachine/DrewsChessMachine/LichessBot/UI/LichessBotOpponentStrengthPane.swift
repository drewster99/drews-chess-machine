import SwiftUI

/// Item 3 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.7): DCM's score by the
/// opponent's rating gap (opponent minus DCM, at the game's start, within
/// its own speed), against the score Elo expects, and the gap at which DCM
/// scores 50% — 0 when its rating is accurate.
struct LichessBotOpponentStrengthPane: View {
    let strength: LichessBotOpponentStrengthStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            Text(LichessBotStatsFormat.fiftyPercentPoint(strength.fiftyPercentPoint, games: strength.games))
                .font(LichessBotStatsStyle.sectionFont)
                .help("The rating gap Δ at which Elo's expected scores add up to DCM's actual score: DCM scores 50% against opponents rated Δ above its own rating. ≥ / ≤ when every game was won / lost.")
            LichessBotRatingBandChart(bands: strength.bands)
                .shown(!strength.bands.isEmpty)
            ScrollView(.horizontal) {
                LichessBotRatingBandTable(bands: strength.bands)
            }
            LichessBotPaneEmptyNote(text: "\(strength.gamesWithoutRatings) scored game\(strength.gamesWithoutRatings == 1 ? "" : "s") without both ratings (Lichess AI, or a missing rating) are left out")
                .shown(strength.gamesWithoutRatings > 0)
        }
    }
}
