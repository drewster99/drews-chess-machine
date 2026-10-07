import SwiftUI

/// L3 (§11): game length in plies by result, and the most recent short
/// losses (under 30 plies), each with a button that opens it on Lichess.
struct LichessBotGameLengthPane: View {
    let gameLength: LichessBotGameLengthStatistics

    static let titles = ["", "Games", "Mean plies", "Median plies"]

    var body: some View {
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                LichessBotStatsHeaderRow(titles: Self.titles)
                ForEach(gameLength.rows) { row in
                    LichessBotGameLengthRowView(row: row)
                }
            }
            Text("Short losses (under \(LichessBotGameLengthStatistics.shortLossPlies) plies): \(gameLength.shortLossCount)")
                .font(LichessBotStatsStyle.sectionFont)
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 4) {
                ForEach(gameLength.shortLosses) { game in
                    LichessBotShortGameRow(game: game)
                }
            }
        }
    }
}
