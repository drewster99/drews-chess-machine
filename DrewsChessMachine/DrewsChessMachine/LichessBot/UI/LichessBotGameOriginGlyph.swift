import SwiftUI

/// A game's origin as its glyph alone, with the full explanation as help
/// (challenge-log plan §3.9): for tables and tight rows. Nil is a live game
/// whose origin is not decided yet.
struct LichessBotGameOriginGlyph: View {
    let display: LichessBotGameOriginDisplay?

    var body: some View {
        let presentation = LichessBotGameOriginStyle.presentation(of: display)
        Image(systemName: presentation.systemImage)
            .foregroundStyle(presentation.color)
            .frame(width: LichessBotGameOriginStyle.glyphWidth)
            .help(presentation.help)
            .accessibilityLabel(presentation.markedShortLabel)
    }
}
