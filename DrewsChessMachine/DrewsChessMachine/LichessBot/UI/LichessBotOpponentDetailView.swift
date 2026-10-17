import SwiftUI

/// One opponent's full breakdown (plan §14.3c). The stats are recomputed off
/// `body` into state whenever the games index changes, so the view never
/// folds the records during layout. The live profile (ratings, crosstable)
/// is fetched by the embedded `LichessBotOpponentCard`.
struct LichessBotOpponentDetailView: View {
    let controller: LichessBotController
    let opponentID: String

    @State private var stats: LichessBotOpponentProfileStats?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                LichessBotOpponentCard(controller: controller, username: displayName)
                LichessBotOpponentHeadlineView(stats: stats)
                LichessBotOpponentSplitsView(stats: stats)
                LichessBotOpponentTrendChartsView(stats: stats)
                LichessBotOpponentHistoryTable(controller: controller, games: stats?.games ?? [])
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onChange(of: controller.index?.rows, initial: true) {
            Task { @MainActor in recompute() }
        }
    }

    /// The best name the records recorded, falling back to the id so the
    /// profile card can still fetch.
    private var displayName: String {
        stats?.name ?? opponentID
    }

    private func recompute() {
        stats = LichessBotOpponentProfileStats(opponentID: opponentID, rows: controller.games(againstOpponent: opponentID))
    }
}
