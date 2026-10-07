import SwiftUI

/// L2 (§11): DCM's clock per time control — mean think time per move (from
/// consecutive server clocks plus the increment), mean time left after its
/// last move, and games lost on time. Games without a clock (correspondence)
/// have no row.
struct LichessBotClockPane: View {
    let rows: [LichessBotClockRow]

    static let titles = ["", "Games", "Think / move", "Left at end", "Lost on time"]

    var body: some View {
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            ScrollView(.horizontal) {
                Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                    LichessBotStatsHeaderRow(titles: Self.titles)
                    ForEach(rows) { row in
                        LichessBotClockRowView(row: row)
                    }
                }
            }
            LichessBotPaneEmptyNote(text: "Think time is the previous clock plus the increment minus the clock after the move; time the opponent gave DCM counts against it.")
        }
    }
}
