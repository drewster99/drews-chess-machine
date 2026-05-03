import Foundation

/// Live view of an in-progress arena tournament. Updated by the driver
/// task after each finished game via a lock-protected `TournamentLiveBox`
/// and mirrored into `@State` by the UI heartbeat. "Candidate" and
/// "champion" here refer to the two networks being compared — not to
/// any chess color. Colors alternate every game.
struct TournamentProgress: Sendable {
    let currentGame: Int
    let totalGames: Int
    let candidateWins: Int
    let championWins: Int
    let draws: Int
    /// Wall-clock time the tournament started, captured once in
    /// `runArenaTournament` before the first game and carried through
    /// every per-game update so the busy label can compute live
    /// elapsed time via `Date().timeIntervalSince(startTime)` on each
    /// render without needing a separate ticking timer.
    let startTime: Date
    /// Number of arena games running concurrently for this tournament.
    /// 1 = sequential (the historical default). > 1 enables the
    /// busy-label suffix "(×K concurrent)" so the user can see why
    /// the tournament is finishing faster than before.
    let concurrency: Int

    init(
        currentGame: Int,
        totalGames: Int,
        candidateWins: Int,
        championWins: Int,
        draws: Int,
        startTime: Date,
        concurrency: Int = 1
    ) {
        self.currentGame = currentGame
        self.totalGames = totalGames
        self.candidateWins = candidateWins
        self.championWins = championWins
        self.draws = draws
        self.startTime = startTime
        self.concurrency = concurrency
    }

    /// AlphaZero-style score: (wins + 0.5 * draws) / games_played.
    /// Pure draws → 0.5. Candidate sweeping every decisive game with
    /// zero losses → 1.0. Used with a 0.55 promotion threshold.
    var candidateScore: Double {
        let played = currentGame > 0 ? currentGame : 1
        return (Double(candidateWins) + 0.5 * Double(draws)) / Double(played)
    }
}
