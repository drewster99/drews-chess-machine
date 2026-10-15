import SwiftUI

/// One filed game's rating change ("+7", "−12", "0", "–"), colored by sign
/// with `LichessBotStatsStyle`'s win and loss colors, and a tooltip saying
/// why a "–" has no number. Nil — a game not filed yet, whose record (and
/// so Lichess's change) has not arrived — shows nothing. The caller sets
/// the font, so the cell matches the numbers around it.
struct LichessBotGameRatingChangeText: View {
    let change: LichessBotGameRatingChange?

    var body: some View {
        Text(change.map(LichessBotStatsFormat.gameRatingChange) ?? "")
            .foregroundStyle(LichessBotStatsStyle.ratingChangeColor(points: change?.points))
            .help(change.map(LichessBotStatsFormat.gameRatingChangeHelp) ?? "")
            .shown(change != nil)
    }
}
