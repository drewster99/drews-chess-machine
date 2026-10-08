import SwiftUI

/// L1 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §11): how DCM's moves related to
/// its own policy — how often it played the policy's top move, the mean
/// probability its sampling gave the move it chose, and how many moves were
/// close to a random pick — over all games and per model.
struct LichessBotMoveChoicePane: View {
    let moveChoice: LichessBotMoveChoiceStatistics

    static let columns: [LichessBotStatsHeaderRow.Column] = [
        .init(title: "", help: nil),
        .init(title: "Decisions", help: "DCM's moves with a recorded decision (the network's policy and the sampled move)."),
        .init(title: "Top move", help: "Share of those moves where DCM played the move its policy ranked first (temperature 1, legal moves only). The rest were sampled lower-ranked moves."),
        .init(title: "Mean p(chosen)", help: "Mean probability the sampling distribution (the policy after the bot's temperature) gave the move DCM played. Near 1: after temperature, nearly all the probability was on the move played; lower: it was still spread over several moves."),
        .init(title: "Randomish", help: "Moves where the distribution DCM sampled from was nearly flat: every move below 1.5× the uniform probability (1 ÷ legal moves). Such a move is close to a random pick, not a network opinion; a forced move never counts."),
    ]

    var body: some View {
        ScrollView(.horizontal) {
            Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                LichessBotStatsHeaderRow(columns: Self.columns)
                LichessBotMoveChoiceRow(label: "All games", line: moveChoice.overall)
                ForEach(moveChoice.byModel) { model in
                    LichessBotMoveChoiceRow(label: model.modelID, line: model.line)
                }
            }
        }
    }
}
