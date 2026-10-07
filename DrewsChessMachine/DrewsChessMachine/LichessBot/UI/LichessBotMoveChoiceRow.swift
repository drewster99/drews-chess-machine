import SwiftUI

/// One row of the Move choice table (a `GridRow`).
struct LichessBotMoveChoiceRow: View {
    let label: String
    let line: LichessBotMoveChoiceLine

    var body: some View {
        GridRow {
            Text(label)
                .font(LichessBotStatsStyle.rowLabelFont)
                .gridColumnAlignment(.leading)
            Text("\(line.decisions)")
            Text(LichessBotStatsFormat.percent(line.topMoveShare))
                .help("Share of DCM's decisions (\(line.decisionsWithTopMoves) recorded the policy's top moves) that played the policy's own top move")
            Text(LichessBotStatsFormat.decimal(line.meanChosenProbability))
                .help("Mean probability the sampling distribution (after temperature) gave the move DCM played")
            Text("\(line.randomish)")
                .help("Decisions whose distribution after temperature was essentially uniform: close to a random pick")
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }
}
