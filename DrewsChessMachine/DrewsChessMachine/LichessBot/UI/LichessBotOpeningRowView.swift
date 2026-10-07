import SwiftUI

/// One opening family's row of the Openings table (a `GridRow`).
struct LichessBotOpeningRowView: View {
    let row: LichessBotOpeningRow
    let countWidth: Int

    var body: some View {
        GridRow {
            LichessBotRowLabel(text: row.family)
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
