import SwiftUI

/// L5 (§11): the opponents — how many, the current and longest streaks, the
/// highest-rated win, and the most played opponents with DCM's record
/// against each.
struct LichessBotOpponentsPane: View {
    let opponents: LichessBotOpponentsStatistics

    static let titles = ["Opponent", "Kind", "Games", "W–D–L", "Score", "Last played"]

    var body: some View {
        let countWidth = LichessBotStatsFormat.countWidth(opponents.mostPlayed.map(\.tally))
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            LichessBotOpponentsFigures(opponents: opponents)
            ScrollView(.horizontal) {
                Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                    LichessBotStatsHeaderRow(titles: Self.titles)
                    ForEach(opponents.mostPlayed) { row in
                        LichessBotOpponentRecordRowView(row: row, countWidth: countWidth)
                    }
                }
            }
        }
    }
}
