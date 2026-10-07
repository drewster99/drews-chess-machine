import SwiftUI

/// The rating bands as numbers: games, DCM's actual score with its 95%
/// Wilson interval, Elo's mean expected score, and the difference in
/// percentage points (§3.7, R-1). Read the difference against the interval:
/// when the expected score lies inside it, the band is consistent with the
/// rating.
struct LichessBotRatingBandTable: View {
    let bands: [LichessBotRatingBand]

    static let titles = ["Gap", "Games", "Actual", "95%", "Expected", "Actual − expected"]

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
            LichessBotStatsHeaderRow(titles: Self.titles)
            ForEach(bands) { band in
                LichessBotRatingBandRow(band: band)
            }
        }
    }
}
