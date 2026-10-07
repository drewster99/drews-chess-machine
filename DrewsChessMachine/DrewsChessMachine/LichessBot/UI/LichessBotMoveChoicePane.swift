import SwiftUI

/// L1 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §11): how DCM's moves related to
/// its own policy — how often it played the policy's top move, the mean
/// probability its sampling gave the move it chose, and how many moves were
/// close to a random pick — over all games and per model.
struct LichessBotMoveChoicePane: View {
    let moveChoice: LichessBotMoveChoiceStatistics

    static let titles = ["", "Decisions", "Top move", "Mean p(chosen)", "Randomish"]

    var body: some View {
        ScrollView(.horizontal) {
            Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
                LichessBotStatsHeaderRow(titles: Self.titles)
                LichessBotMoveChoiceRow(label: "All games", line: moveChoice.overall)
                ForEach(moveChoice.byModel) { model in
                    LichessBotMoveChoiceRow(label: model.modelID, line: model.line)
                }
            }
        }
    }
}
