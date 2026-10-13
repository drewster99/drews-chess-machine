import SwiftUI

/// The operator's **Abort** for one game, on its tile and in its game view.
/// Shown only while the game is in progress and a side has yet to make its
/// first move (`LichessBotLiveGame.isAbortable`): Lichess refuses an abort
/// after that, and a game that has started is ended by resigning instead.
/// Asks first, since an abort can't be undone. The game's result arrives on
/// its stream like any other ending; a refusal is logged by the session
/// manager.
struct LichessBotAbortGameButton: View {
    let controller: LichessBotController
    let game: LichessBotLiveGame

    @State private var isConfirming = false

    var body: some View {
        Button(
            role: .destructive,
            action: { isConfirming = true },
            label: { Label("Abort", systemImage: "xmark.octagon") }
        )
        .font(.callout)
        .help("Abort this game on Lichess. Allowed until both sides have made their first move; rating-neutral.")
        .confirmationDialog(
            "Abort game \(game.id)?",
            isPresented: $isConfirming,
            titleVisibility: .visible,
            actions: {
                Button("Abort Game", role: .destructive) {
                    let gameID = game.id
                    Task { await controller.abortGame(id: gameID) }
                }
                Button("Cancel", role: .cancel) {}
            },
            message: { Text("Lichess ends the game without a result. It can't be undone.") }
        )
        .shown(game.isAbortable)
    }
}
