import SwiftUI

/// Places the statistics column, a separator and the recent games side by
/// side when the card is at least `LichessBotStatsStyle.wideCardWidth`
/// wide, and stacked (recent games below, at a fixed height) when it is
/// narrower (`LICHESS_BOT_RECORD_STATS_PLAN.md` §5.1, R-2).
///
/// One view with `AnyLayout`, not `ViewThatFits`: `ViewThatFits` chooses by
/// each child's ideal width, which here depends on the data (the longest
/// opponent name, digit counts, the widest pane), so the arrangement would
/// flip as games arrive; and its variants are different view types, so each
/// flip would discard the panes' state and scroll positions. `AnyLayout`
/// keeps the children's identity across the switch, and the switch depends
/// only on the card's width, read with `onGeometryChange` (not a
/// `GeometryReader`).
///
/// The separator is a 1-point rectangle sized per arrangement rather than a
/// `Divider`, whose orientation follows the enclosing stack type and is not
/// documented for `AnyLayout`.
struct LichessBotRecordCardLayout<Statistics: View, RecentGames: View>: View {
    let statistics: Statistics
    let recentGames: RecentGames
    @State private var isWide = false

    var body: some View {
        let layout = isWide
            ? AnyLayout(HStackLayout(alignment: .top, spacing: 16))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 10))
        layout {
            statistics
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Rectangle()
                .fill(LichessBotStatsStyle.separator)
                .frame(width: isWide ? 1 : nil, height: isWide ? nil : 1)
            recentGames
                .modifier(LichessBotRecentGamesFrame(isWide: isWide))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onGeometryChange(
            for: Bool.self,
            of: { proxy in proxy.size.width >= LichessBotStatsStyle.wideCardWidth },
            action: { wide in isWide = wide }
        )
    }
}
