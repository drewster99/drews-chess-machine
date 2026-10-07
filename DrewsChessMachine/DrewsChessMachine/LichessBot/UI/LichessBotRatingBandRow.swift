import SwiftUI

/// One band's row of the rating-band table (a `GridRow`).
struct LichessBotRatingBandRow: View {
    let band: LichessBotRatingBand

    var body: some View {
        GridRow {
            Text(LichessBotRatingBand.label(index: band.index))
                .gridColumnAlignment(.leading)
            Text("\(band.games)")
            Text(LichessBotStatsFormat.percent(band.actualScore))
                .foregroundStyle(LichessBotStatsStyle.actual)
            Text(LichessBotStatsFormat.interval(band.interval))
                .foregroundStyle(LichessBotStatsStyle.interval)
            Text(LichessBotStatsFormat.percent(band.meanExpected))
                .foregroundStyle(LichessBotStatsStyle.expected)
            Text(LichessBotStatsFormat.signedPoints(band.actualScore - band.meanExpected))
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
