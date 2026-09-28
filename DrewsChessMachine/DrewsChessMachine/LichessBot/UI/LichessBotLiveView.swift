import SwiftUI

/// The live games section (plan §14.3a): one game large — by default the
/// first-started game still in progress — or every game in a grid, with
/// finished games kept for a while. Any game can pop out into its own
/// window.
struct LichessBotLiveView: View {
    @Bindable var controller: LichessBotController
    /// Whether this section is on screen; its arrow-key shortcuts are
    /// claimed only then, so they never steal keys from other sections.
    let isVisible: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                Picker("", selection: $controller.showsGrid) {
                    Label("Single", systemImage: "square").tag(false)
                    Label("Grid", systemImage: "square.grid.2x2").tag(true)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 160)
                Picker("Game", selection: $controller.focusedGameID) {
                    Text("First in progress").tag(String?.none)
                    ForEach(controller.games) { game in
                        Text(LichessBotLiveView.menuTitle(for: game)).tag(String?.some(game.id))
                    }
                }
                .frame(maxWidth: 320)
                .shown(!controller.showsGrid)
                Spacer()
                Text("\(controller.gamesInProgress.count) in progress · \(controller.games.count - controller.gamesInProgress.count) finished")
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                Button("Clear Finished") {
                    controller.dismissFinishedGames()
                }
                .disabled(controller.games.count == controller.gamesInProgress.count)
            }
            .padding(10)
            Divider()
            ZStack {
                LichessBotLiveSingleView(controller: controller, claimsKeyboardShortcuts: isVisible && !controller.showsGrid)
                    .shown(!controller.showsGrid)
                LichessBotGameGridView(controller: controller)
                    .shown(controller.showsGrid)
            }
        }
    }

    static func menuTitle(for game: LichessBotLiveGame) -> String {
        let opponent = game.opponent?.name ?? "?"
        return "\(game.isFinished ? "✓" : "●") \(opponent) · \(game.id)"
    }
}

/// The single large game, or an empty-state message.
struct LichessBotLiveSingleView: View {
    let controller: LichessBotController
    let claimsKeyboardShortcuts: Bool

    var body: some View {
        ZStack {
            Text(controller.isRunning ? "No games yet. Games appear here as they start." : "The bot is offline.")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .shown(controller.displayedGame == nil)
            ForEach(controller.displayedGame.map { [$0] } ?? []) { game in
                LichessBotGameDetailView(
                    controller: controller,
                    game: game,
                    headToHead: controller.headToHead(against: game.opponent?.id),
                    onPopOut: { LichessBotGameWindowLauncher.open(game: game, controller: controller) },
                    claimsKeyboardShortcuts: claimsKeyboardShortcuts
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

/// Every game as a tile.
struct LichessBotGameGridView: View {
    let controller: LichessBotController

    /// Games in progress first, oldest start first (the order they began);
    /// then finished games, most recently finished first.
    static func ordered(_ games: [LichessBotLiveGame]) -> [LichessBotLiveGame] {
        let live = games.filter { !$0.isFinished }.sorted { $0.startedAt < $1.startedAt }
        let finished = games
            .compactMap { game in game.finishedAt.map { (game: game, finishedAt: $0) } }
            .sorted { $0.finishedAt > $1.finishedAt }
            .map(\.game)
        return live + finished
    }

    var body: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 240, maximum: 340), spacing: 12)], spacing: 12) {
                ForEach(Self.ordered(controller.games)) { game in
                    LichessBotGameTileView(
                        controller: controller,
                        game: game,
                        onOpen: { LichessBotGameWindowLauncher.open(game: game, controller: controller) },
                        onDismiss: { controller.dismissGame(game.id) }
                    )
                }
            }
            .padding(12)
        }
    }
}
