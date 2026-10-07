import SwiftUI

/// A statistics table's header row (a `GridRow` of its grid): the titles in
/// the header font, one per column. Every pane's table uses it, so headers
/// look the same everywhere.
struct LichessBotStatsHeaderRow: View {
    let titles: [String]

    var body: some View {
        GridRow {
            ForEach(Array(titles.enumerated()), id: \.offset) { _, title in
                Text(title)
            }
        }
        .font(LichessBotStatsStyle.headerFont)
        .foregroundStyle(LichessBotStatsStyle.neutral)
        .lineLimit(1)
        .fixedSize()
    }
}
