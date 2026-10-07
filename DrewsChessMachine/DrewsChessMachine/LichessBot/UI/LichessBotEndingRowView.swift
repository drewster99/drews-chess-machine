import SwiftUI

/// One ending's row of the Endings table (a `GridRow`). The "Not counted"
/// row has no result columns ("–"); every other row has no not-counted
/// games ("–" there).
struct LichessBotEndingRowView: View {
    let row: LichessBotEndingRow
    let totals: LichessBotEndingStatistics

    var body: some View {
        let counted = row.ending != .notCounted
        GridRow {
            Text(row.ending.label)
                .font(LichessBotStatsStyle.rowLabelFont)
                .gridColumnAlignment(.leading)
            Text(counted ? LichessBotStatsFormat.countWithShare(row.wins, of: totals.wins) : LichessBotStatsFormat.missing)
                .foregroundStyle(LichessBotStatsStyle.win)
            Text(counted ? LichessBotStatsFormat.countWithShare(row.draws, of: totals.draws) : LichessBotStatsFormat.missing)
            Text(counted ? LichessBotStatsFormat.countWithShare(row.losses, of: totals.losses) : LichessBotStatsFormat.missing)
                .foregroundStyle(LichessBotStatsStyle.loss)
            Text(counted ? LichessBotStatsFormat.missing : "\(row.unscored)")
                .foregroundStyle(LichessBotStatsStyle.neutral)
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
