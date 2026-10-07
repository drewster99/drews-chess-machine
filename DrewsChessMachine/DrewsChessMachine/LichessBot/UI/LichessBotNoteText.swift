import SwiftUI

/// Secondary text in a table cell: the note font in the neutral color.
struct LichessBotNoteText: View {
    let text: String

    var body: some View {
        Text(text)
            .font(LichessBotStatsStyle.noteFont)
            .foregroundStyle(LichessBotStatsStyle.neutral)
    }
}
