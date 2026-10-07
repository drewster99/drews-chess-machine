import SwiftUI

/// A pane's line for "nothing to show" ("No games in this period").
struct LichessBotPaneEmptyNote: View {
    let text: String

    var body: some View {
        Text(text)
            .font(LichessBotStatsStyle.noteFont)
            .foregroundStyle(LichessBotStatsStyle.neutral)
    }
}
