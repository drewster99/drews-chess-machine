import SwiftUI

/// One time control's row of the Clock table (a `GridRow`).
struct LichessBotClockRowView: View {
    let row: LichessBotClockRow

    var body: some View {
        GridRow {
            LichessBotRowLabel(text: row.speed)
            Text("\(row.games)")
            Text(LichessBotStatsFormat.seconds(row.meanThinkSeconds))
                .help("Over \(row.thinkTimeMoves) moves whose clocks were recorded")
            Text(LichessBotStatsFormat.seconds(row.meanFinalClockSeconds))
                .help("DCM's clock after its last move with a recorded clock, over \(row.gamesWithFinalClock) games")
            Text("\(row.flagged)")
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
