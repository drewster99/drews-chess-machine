import SwiftUI

/// The recent games' frame in each arrangement of the Record card: beside
/// the statistics, between a minimum and a maximum width and the card's
/// full height; below them, the full width at a fixed height.
struct LichessBotRecentGamesFrame: ViewModifier {
    let isWide: Bool

    func body(content: Content) -> some View {
        content
            .frame(
                minWidth: isWide ? LichessBotStatsStyle.recentGamesMinimumWidth : nil,
                maxWidth: isWide ? LichessBotStatsStyle.recentGamesMaximumWidth : .infinity,
                maxHeight: .infinity,
                alignment: .topLeading
            )
            .frame(height: isWide ? nil : LichessBotStatsStyle.narrowRecentGamesHeight)
    }
}
