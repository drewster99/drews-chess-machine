import SwiftUI

/// One checkpoint's row of the calibration table (a `GridRow`).
struct LichessBotCalibrationRowView: View {
    let row: LichessBotCalibrationRow

    var body: some View {
        GridRow {
            LichessBotRowLabel(text: "\(row.moveNumber)")
                .help("Games that reached DCM's move \(row.moveNumber); games that ended earlier are not in this row")
            Text("\(row.games)")
            Text("\(row.missing)")
                .help("Games that reached this move with no recorded decision there; left out")
            Text(LichessBotStatsFormat.percent(row.meanPredictedExpected))
                .foregroundStyle(LichessBotStatsStyle.expected)
                .help("Mean expected score the value head gave (win + ½ draw)")
            Text(LichessBotStatsFormat.percent(row.meanActualScore))
                .foregroundStyle(LichessBotStatsStyle.actual)
            Text(LichessBotStatsFormat.triple(row.meanPredicted))
            Text(LichessBotStatsFormat.triple(row.actualFrequencies))
            Text(LichessBotStatsFormat.decimal(row.brier))
                .help("Mean 3-class Brier score: 0 is perfect, 2 the worst")
            Text(LichessBotStatsFormat.decimal(row.skill, places: 2))
                .help("1 − Brier / Brier of always predicting these games' own result frequencies; positive beats the base rates, \"–\" when every game had the same result")
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
