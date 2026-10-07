import SwiftUI

/// L4 (§11): DCM's results by opening family (the Lichess opening name
/// before its first colon, with the ECO codes it spans), as White and as
/// Black, most played first. Openings come from Lichess's export, so a game
/// filed without one has none.
struct LichessBotOpeningsPane: View {
    let openings: LichessBotOpeningStatistics

    static let titles = ["Opening", "ECO", "as White", "Score", "as Black", "Score"]

    var body: some View {
        let countWidth = LichessBotStatsFormat.countWidth(openings.rows.flatMap { [$0.asWhite, $0.asBlack] })
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            ScrollView(.horizontal) {
                Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                    LichessBotStatsHeaderRow(titles: Self.titles)
                    ForEach(openings.rows) { row in
                        LichessBotOpeningRowView(row: row, countWidth: countWidth)
                    }
                }
            }
            LichessBotPaneEmptyNote(text: "\(openings.gamesWithoutOpening) scored game\(openings.gamesWithoutOpening == 1 ? "" : "s") without an opening from Lichess")
                .shown(openings.gamesWithoutOpening > 0)
        }
    }
}
