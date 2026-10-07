import SwiftUI

/// The recent games, or "Loading…" until the games index has loaded. Read
/// from the index, not the statistics, so the list shows as soon as the
/// index does.
struct LichessBotRecentGamesSection: View {
    let controller: LichessBotController

    var body: some View {
        ZStack(alignment: .topLeading) {
            Text("Loading the game records…")
                .foregroundStyle(LichessBotStatsStyle.neutral)
                .shown(!Self.isLoaded(controller.index))
            // Re-evaluated each minute so the relative times stay current.
            // The index rows are already newest first
            // (`LichessBotIndex.sorted`), so this takes a prefix rather than
            // re-sorting every row on the main actor each minute.
            TimelineView(.everyMinute) { _ in
                LichessBotRecentGamesList(
                    controller: controller,
                    rows: Self.recentRows(controller.index)
                )
            }
            .shown(Self.isLoaded(controller.index))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private static func isLoaded(_ index: LichessBotIndex.File?) -> Bool {
        index != nil
    }

    /// The newest `maximumRows` rows; none before the index loads (the list
    /// is hidden then).
    private static func recentRows(_ index: LichessBotIndex.File?) -> [LichessBotGameSummary] {
        guard let index else { return [] }
        return Array(index.rows.prefix(LichessBotRecentGamesList.maximumRows))
    }
}
