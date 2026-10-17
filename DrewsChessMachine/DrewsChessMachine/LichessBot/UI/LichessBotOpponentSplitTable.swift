import SwiftUI

/// A per-opponent split table: one row per group (a time control, or a
/// model), with DCM's record and net rating change over that group. Shared
/// by the By time control and By model sections.
struct LichessBotOpponentSplitTable: View {
    let rows: [LichessBotOpponentSplitRow]

    static let titles = ["", "Games", "W–D–L", "Score", "Rating Δ"]

    var body: some View {
        let countWidth = LichessBotStatsFormat.countWidth(rows.map(\.tally))
        ScrollView(.horizontal) {
            Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                LichessBotStatsHeaderRow(titles: Self.titles)
                ForEach(rows) { row in
                    GridRow {
                        LichessBotRowLabel(text: row.label)
                        Text("\(row.tally.games)")
                        LichessBotTallyText(tally: row.tally, countWidth: countWidth)
                        Text(LichessBotStatsFormat.score(row.tally.score))
                        Text(LichessBotStatsFormat.ratingChange(row.ratingChange))
                            .help(LichessBotStatsFormat.ratingChangeHelp(row.ratingChange))
                    }
                    .font(LichessBotStatsStyle.numberFont)
                    .lineLimit(1)
                    .fixedSize()
                }
            }
        }
    }
}
