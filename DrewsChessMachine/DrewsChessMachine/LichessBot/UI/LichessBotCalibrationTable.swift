import SwiftUI

/// The value head against the results at DCM's 10th, 20th and 40th moves
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.6, OD-13): the mean predicted
/// expected score and W / D / L against what happened, the 3-class Brier
/// score (0 perfect, 2 worst), and its skill against knowing only the same
/// games' result frequencies (positive is better than the base rates).
struct LichessBotCalibrationTable: View {
    let rows: [LichessBotCalibrationRow]

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
            LichessBotStatsHeaderRow(columns: LichessBotCalibrationColumn.allCases.map { .init(title: $0.title, help: $0.help) })
            ForEach(rows, id: \.moveNumber) { row in
                LichessBotCalibrationRowView(row: row)
            }
        }
    }
}
