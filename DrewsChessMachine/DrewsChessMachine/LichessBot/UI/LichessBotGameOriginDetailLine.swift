import SwiftUI

/// The game detail's "Origin" row (challenge-log plan §3.9): the glyph, the
/// long label with its basis marker, and the detail. Nil is a live game
/// whose origin is not decided yet.
struct LichessBotGameOriginDetailLine: View {
    let display: LichessBotGameOriginDisplay?

    var body: some View {
        let presentation = LichessBotGameOriginStyle.presentation(of: display)
        HStack(spacing: 6) {
            Text("Origin")
                .foregroundStyle(.secondary)
            Image(systemName: presentation.systemImage)
                .foregroundStyle(presentation.color)
                .frame(width: LichessBotGameOriginStyle.glyphWidth)
            Text(presentation.markedLongLabel)
                .foregroundStyle(presentation.color)
            Text(display?.detail ?? "")
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .shown(display != nil)
        }
        .font(.callout)
        .lineLimit(1)
        .help(presentation.help)
    }
}
