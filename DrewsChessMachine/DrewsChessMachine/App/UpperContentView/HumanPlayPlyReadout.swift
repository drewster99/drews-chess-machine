import Foundation

/// Pure helper backing the human-play status bar's "Ply" readout.
///
/// Lives apart from the `fileprivate` `HumanPlayWindowView` so it can be
/// unit-tested. The decision it encodes is small but easy to get wrong:
/// the "Ply" readout must show the **displayed game length** — the number
/// of plies in the move-list `history`, which includes any reverted-in
/// prefix — and must **not** show `GameStats.totalMoves`.
///
/// `GameStats.totalMoves` counts only the plies the most recent
/// `ChessMachine.runGameLoop` actually played. After a Revert, the engine
/// is reseeded at the reverted-to position (`beginNewGame(initialState:)`)
/// and replays forward, so its per-loop counters restart at 0 and the
/// total excludes the seeded prefix — under-reporting the game the user is
/// looking at. The move list, board, and live counter are all seeded with
/// that prefix, so the engine's per-loop total is the one number on screen
/// that would disagree. See the `revertToHistoryPly` / `seededHistory`
/// path in `PlayController` and the `GameStats` construction in
/// `ChessMachine.runGameLoop`.
enum HumanPlayPlyReadout {

    /// The ply number to display in the status bar.
    ///
    /// Always returns `displayedHistoryPlies` (the move-list length).
    /// `engineLoopTotalMoves` — a `GameStats.totalMoves`, non-nil only
    /// once the game is over — is accepted purely to make explicit, and
    /// to test-guard, that it is deliberately **not** used: showing it
    /// would make "Ply" drop below the move list on any reverted game.
    static func displayedPlyCount(
        displayedHistoryPlies: Int,
        engineLoopTotalMoves: Int?
    ) -> Int {
        displayedHistoryPlies
    }
}
