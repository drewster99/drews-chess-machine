import SwiftUI

/// A game's origin as its glyph and short label, with the basis marker
/// ("≈" for an inferred one) and the full explanation as help
/// (challenge-log plan §3.9). Nil is a live game whose origin is not decided
/// yet.
struct LichessBotGameOriginLabel: View {
    let display: LichessBotGameOriginDisplay?

    var body: some View {
        let presentation = LichessBotGameOriginStyle.presentation(of: display)
        HStack(spacing: 4) {
            Image(systemName: presentation.systemImage)
                .frame(width: LichessBotGameOriginStyle.glyphWidth)
            Text(presentation.markedShortLabel)
                .lineLimit(1)
        }
        .foregroundStyle(presentation.color)
        .help(presentation.help)
        .accessibilityElement(children: .combine)
    }
}
