import SwiftUI

/// A W–D–L tally: counts in the primary color, each left-padded to
/// `countWidth` digits (spaces are digit-wide in the monospaced font the
/// tables set), and faint dashes between them.
struct LichessBotTallyText: View {
    let tally: LichessBotResultTally
    let countWidth: Int

    var body: some View {
        let dash = Text("–").foregroundStyle(.tertiary)
        Text("\(padded(tally.wins))\(dash)\(padded(tally.draws))\(dash)\(padded(tally.losses))")
    }

    private func padded(_ count: Int) -> String {
        LichessBotStatsFormat.padded(count, width: countWidth)
    }
}
