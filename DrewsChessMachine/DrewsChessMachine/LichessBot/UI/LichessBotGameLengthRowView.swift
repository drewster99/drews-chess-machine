import SwiftUI

/// One result's row of the Game length table (a `GridRow`).
struct LichessBotGameLengthRowView: View {
    let row: LichessBotGameLengthRow

    var body: some View {
        GridRow {
            Text(row.label)
                .font(LichessBotStatsStyle.rowLabelFont)
                .gridColumnAlignment(.leading)
            Text("\(row.games)")
            Text(LichessBotStatsFormat.oneDecimal(row.meanPlies))
            Text(LichessBotStatsFormat.oneDecimal(row.medianPlies))
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
