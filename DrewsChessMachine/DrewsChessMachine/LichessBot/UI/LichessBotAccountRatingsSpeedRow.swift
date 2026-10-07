import SwiftUI

/// One speed's row in the ratings grid.
struct LichessBotAccountRatingsSpeedRow: View {
    let row: LichessBotAccountRatingRow
    let counts: LichessBotAccountGameCounts.Reading

    var body: some View {
        GridRow {
            Text(row.speed)
                .font(.callout)
                .gridColumnAlignment(.leading)
            Text(row.rating)
            Text(row.ratedGames)
            Text(counts.text(\.unrated, speed: row.speed))
            Text(counts.text(\.today, speed: row.speed))
            Text(counts.text(\.lastDay, speed: row.speed))
        }
        .font(.system(.callout, design: .monospaced))
    }
}
