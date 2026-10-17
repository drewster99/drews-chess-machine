import SwiftUI

/// The detail window's two split tables side by side where there is room:
/// DCM's record against the opponent broken out by time control and by the
/// model that played.
struct LichessBotOpponentSplitsView: View {
    let stats: LichessBotOpponentProfileStats?

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 14) {
                LichessBotOpponentSplitBox(title: "By time control", rows: stats?.bySpeed ?? [])
                LichessBotOpponentSplitBox(title: "By model", rows: stats?.byModel ?? [])
            }
            VStack(alignment: .leading, spacing: 14) {
                LichessBotOpponentSplitBox(title: "By time control", rows: stats?.bySpeed ?? [])
                LichessBotOpponentSplitBox(title: "By model", rows: stats?.byModel ?? [])
            }
        }
        .shown(stats?.games.isEmpty == false)
    }
}

/// One titled split table in a group box.
struct LichessBotOpponentSplitBox: View {
    let title: String
    let rows: [LichessBotOpponentSplitRow]

    var body: some View {
        GroupBox(title) {
            LichessBotOpponentSplitTable(rows: rows)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
        }
    }
}
