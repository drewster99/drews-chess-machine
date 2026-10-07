import Foundation

/// Hands the sequential test (`ArenaSPRT.Monitor`) its games in the order
/// they were *started*, not the order they finished.
///
/// **Why.** An arena runs hundreds of games at once, and a game's length is
/// correlated with its result: short threefold draws end first, decisive
/// games later. Fed in finishing order, the test's first hundred games are
/// the hundred shortest — almost all draws — and a stopping rule decides on
/// that biased sample before most decisive games have ended. On 2026-10-07
/// arenas #2, #4 and #5 rejected at games 182, 125 and 56 on W/D/L
/// 0/181/1, 0/124/1 and 0/55/1 with 400 games in flight; #4's full 524
/// games were W239/D223/L62, Elo +122 [+100, +145].
///
/// A game's start index does not depend on how long it lasts, so every
/// prefix of games 0, 1, 2, … in start order is an unbiased sample, and the
/// colors alternate along it. This keeps the results that finished ahead of
/// an unfinished earlier game and releases them only once the run of
/// finished games from 0 is unbroken. The verdict therefore waits for the
/// slower games of that run; the games played meanwhile are the price of a
/// correct sample.
///
/// The tally here is the test's sample — games `0..<nextGameIndex` — which
/// is not the tournament's total (the driver keeps that, over every game
/// finished in any order), so it is not a second copy of the driver's.
struct ArenaSPRTStartOrderFeed: Sendable {
    /// One game's result from the candidate's side.
    enum Outcome: Sendable, Equatable {
        case win
        case draw
        case loss
    }

    /// The test's sample after one more game in start order.
    struct Tally: Sendable, Equatable {
        let wins: Int
        let draws: Int
        let losses: Int
    }

    /// The first game not yet released to the test.
    private(set) var nextGameIndex = 0
    private(set) var wins = 0
    private(set) var draws = 0
    private(set) var losses = 0
    /// Finished games whose start index is past `nextGameIndex`.
    private var finishedAhead: [Int: Outcome] = [:]

    /// Games finished but not yet released, because an earlier one is still
    /// being played.
    var heldCount: Int { finishedAhead.count }

    /// Record that game `gameIndex` finished with `outcome`. Returns the
    /// tallies this releases, oldest first, one per game, each to be fed to
    /// the test in turn; empty while an earlier game is unfinished.
    mutating func record(_ outcome: Outcome, gameIndex: Int) -> [Tally] {
        // A game index is spawned once and finishes once; a repeat or one
        // already released is a driver bug, never a result to merge.
        precondition(gameIndex >= nextGameIndex && finishedAhead[gameIndex] == nil,
                     "game \(gameIndex) recorded twice (next to release: \(nextGameIndex))")
        finishedAhead[gameIndex] = outcome
        var released: [Tally] = []
        while let next = finishedAhead.removeValue(forKey: nextGameIndex) {
            switch next {
            case .win: wins += 1
            case .draw: draws += 1
            case .loss: losses += 1
            }
            nextGameIndex += 1
            released.append(Tally(wins: wins, draws: draws, losses: losses))
        }
        return released
    }
}
