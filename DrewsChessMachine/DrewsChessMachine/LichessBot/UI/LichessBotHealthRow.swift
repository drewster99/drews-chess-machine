import SwiftUI

/// One count of the Bot health pane (a `GridRow`): its name and value.
struct LichessBotHealthRow: View {
    let label: String
    let value: Int
    let help: String

    var body: some View {
        GridRow {
            Text(label)
                .font(LichessBotStatsStyle.rowLabelFont)
                .gridColumnAlignment(.leading)
            Text("\(value)")
                .font(LichessBotStatsStyle.numberFont)
        }
        .lineLimit(1)
        .fixedSize()
        .help(help)
    }
}
