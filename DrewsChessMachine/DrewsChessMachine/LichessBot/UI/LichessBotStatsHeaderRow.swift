import SwiftUI

/// A statistics table's header row (a `GridRow` of its grid): the titles in
/// the header font, one per column, each with its tooltip when the table
/// gives one. Every pane's table uses it, so headers look the same
/// everywhere.
struct LichessBotStatsHeaderRow: View {
    /// A column's title and, when the table explains it, its tooltip.
    struct Column {
        let title: String
        let help: String?
    }

    let columns: [Column]

    /// Titles without tooltips.
    init(titles: [String]) {
        columns = titles.map { Column(title: $0, help: nil) }
    }

    init(columns: [Column]) {
        self.columns = columns
    }

    var body: some View {
        GridRow {
            ForEach(Array(columns.enumerated()), id: \.offset) { _, column in
                Text(column.title)
                    .help(column.help ?? "")
            }
        }
        .font(LichessBotStatsStyle.headerFont)
        .foregroundStyle(LichessBotStatsStyle.neutral)
        .lineLimit(1)
        .fixedSize()
    }
}
