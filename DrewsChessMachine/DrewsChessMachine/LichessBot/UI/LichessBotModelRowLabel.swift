import SwiftUI

/// A Models table row's name cell: a group's model ID in the row-label
/// font, a checkpoint's label smaller and indented under its group.
struct LichessBotModelRowLabel: View {
    let row: LichessBotModelTableRow

    static let checkpointIndent: CGFloat = 14

    var body: some View {
        Text(row.label)
            .font(row.kind == .checkpoint ? LichessBotStatsStyle.noteFont : LichessBotStatsStyle.rowLabelFont)
            .padding(.leading, row.kind == .checkpoint ? Self.checkpointIndent : 0)
            .gridColumnAlignment(.leading)
    }
}
