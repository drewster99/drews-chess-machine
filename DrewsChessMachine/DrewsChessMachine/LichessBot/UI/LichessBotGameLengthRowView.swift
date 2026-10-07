import SwiftUI

/// One result's row of the Game length table (a `GridRow`).
struct LichessBotGameLengthRowView: View {
    let row: LichessBotGameLengthRow

    var body: some View {
        GridRow {
            LichessBotRowLabel(text: row.label)
            Text("\(row.games)")
            Text(LichessBotStatsFormat.oneDecimal(row.meanPlies))
            Text(LichessBotStatsFormat.oneDecimal(row.medianPlies))
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
