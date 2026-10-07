import SwiftUI

/// A table row's name cell: the row-label font, aligned to the leading edge
/// of its grid column.
struct LichessBotRowLabel: View {
    let text: String

    var body: some View {
        Text(text)
            .font(LichessBotStatsStyle.rowLabelFont)
            .gridColumnAlignment(.leading)
    }
}
