import SwiftUI

/// The Opponents pane's summary lines: opponent count, streaks and the
/// highest-rated win (with a button that opens it on Lichess).
struct LichessBotOpponentsFigures: View {
    let opponents: LichessBotOpponentsStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(opponents.opponentCount) opponent\(opponents.opponentCount == 1 ? "" : "s") with an account")
            Text("Current streak: \(LichessBotStatsFormat.streak(opponents.currentStreak)) · longest: \(LichessBotStatsFormat.streak(LichessBotStreak(ourScore: 1, length: opponents.longestWinStreak))), \(LichessBotStatsFormat.streak(LichessBotStreak(ourScore: 0, length: opponents.longestLossStreak)))")
                .help("Runs of one result in consecutive scored games, in the order they started; a draw ends a run of wins or losses")
            HStack(spacing: 8) {
                Text(Self.bestWinText(opponents.highestRatedWin))
                Button("Open") {
                    if let gameID = opponents.highestRatedWin?.gameID {
                        LichessBotLinks.openGame(gameID)
                    }
                }
                .font(LichessBotStatsStyle.noteFont)
                .opacity(opponents.highestRatedWin == nil ? 0 : 1)
                .disabled(opponents.highestRatedWin == nil)
            }
        }
        .font(LichessBotStatsStyle.figureFont)
    }

    static func bestWinText(_ win: LichessBotNotableWin?) -> String {
        guard let win else { return "Highest-rated win: none" }
        return "Highest-rated win: \(win.opponentName) (\(win.rating)), \(win.createdAt.formatted(date: .abbreviated, time: .omitted))"
    }
}
