import SwiftUI

/// One opening family's row of the Openings table (a `GridRow`).
struct LichessBotOpeningRowView: View {
    let row: LichessBotOpeningRow
    let countWidth: Int

    var body: some View {
        GridRow {
            Text(row.family)
                .font(LichessBotStatsStyle.rowLabelFont)
                .gridColumnAlignment(.leading)
            Text(row.ecoRange)
            LichessBotTallyText(tally: row.asWhite, countWidth: countWidth)
            Text(LichessBotStatsFormat.percent(row.asWhite.score))
            LichessBotTallyText(tally: row.asBlack, countWidth: countWidth)
            Text(LichessBotStatsFormat.percent(row.asBlack.score))
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
