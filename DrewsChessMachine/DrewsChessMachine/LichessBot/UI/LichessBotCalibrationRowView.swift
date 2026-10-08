import SwiftUI

/// One checkpoint's row of the calibration table (a `GridRow`).
struct LichessBotCalibrationRowView: View {
    let row: LichessBotCalibrationRow

    var body: some View {
        GridRow {
            LichessBotRowLabel(text: "\(row.moveNumber)")
                .help(LichessBotCalibrationColumn.move.help)
            Text("\(row.games)")
                .help(LichessBotCalibrationColumn.games.help)
            Text("\(row.missing)")
                .help(LichessBotCalibrationColumn.missing.help)
            Text(LichessBotStatsFormat.percent(row.meanPredictedExpected))
                .foregroundStyle(LichessBotStatsStyle.expected)
                .help(LichessBotCalibrationColumn.predicted.help)
            Text(LichessBotStatsFormat.percent(row.meanActualScore))
                .foregroundStyle(LichessBotStatsStyle.actual)
                .help(LichessBotCalibrationColumn.actual.help)
            Text(LichessBotStatsFormat.triple(row.meanPredicted))
                .help(LichessBotCalibrationColumn.predictedTriple.help)
            Text(LichessBotStatsFormat.triple(row.actualFrequencies))
                .help(LichessBotCalibrationColumn.actualTriple.help)
            Text(LichessBotStatsFormat.decimal(row.brier))
                .help(LichessBotCalibrationColumn.brier.help)
            Text(LichessBotStatsFormat.decimal(row.skill, places: 2))
                .help(LichessBotCalibrationColumn.skill.help)
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
