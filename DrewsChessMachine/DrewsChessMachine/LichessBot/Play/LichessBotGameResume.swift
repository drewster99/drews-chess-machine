import Foundation

/// What a game session carries over from earlier sessions of the same game:
/// the per-game allowances and streaks that must not start over when the
/// app is relaunched (or goes offline and back) while the game is live.
struct LichessBotGameSessionCarryover: Sendable, Equatable {
    /// The greeting was already handled (sent, or deliberately not sent) for
    /// this game: a resumed session must not greet the opponent again.
    let greeted: Bool

    /// A game this runtime is the first to see.
    static let newGame = LichessBotGameSessionCarryover(greeted: false)

    /// A game resumed from a journal left by an earlier runtime. It was
    /// greeted then, if it was ever going to be: never greeted twice.
    static let resumedGame = LichessBotGameSessionCarryover(greeted: true)
}
