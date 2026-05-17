import Foundation

// MARK: - Tournament Stats

/// Accumulated results from a multi-game tournament. Per-side
/// counters are from player A's perspective — at the arena call
/// site player A is the candidate / trainer network being
/// evaluated against the champion. With colors alternating game to
/// game (`gameIndex % 2 == 0 → A is white`), roughly half of N
/// games end up in each side bucket and the split per-side score
/// tells the reader whether a candidate that beat the champion did
/// so evenly, or is really just strong at one color.
struct TournamentStats: Sendable {
    let gamesPlayed: Int
    let playerAWins: Int
    let playerBWins: Int
    let draws: Int

    // Per-side W/L/D, player-A-perspective. "Wins as white" means A
    // won when it was playing white. Draws are the same in both
    // directions (there's no "A-won draw") so they're bucketed by
    // which side A was on.
    let playerAWinsAsWhite: Int
    let playerAWinsAsBlack: Int
    let playerALossesAsWhite: Int
    let playerALossesAsBlack: Int
    let playerADrawsAsWhite: Int
    let playerADrawsAsBlack: Int

    var playerAWinRate: Double {
        gamesPlayed > 0 ? Double(playerAWins) / Double(gamesPlayed) : 0
    }

    var playerBWinRate: Double {
        gamesPlayed > 0 ? Double(playerBWins) / Double(gamesPlayed) : 0
    }

    var drawRate: Double {
        gamesPlayed > 0 ? Double(draws) / Double(gamesPlayed) : 0
    }

    /// Games player A played as white, summed across all outcomes.
    var playerAWhiteGames: Int {
        playerAWinsAsWhite + playerALossesAsWhite + playerADrawsAsWhite
    }

    /// Games player A played as black, summed across all outcomes.
    var playerABlackGames: Int {
        playerAWinsAsBlack + playerALossesAsBlack + playerADrawsAsBlack
    }

    /// AlphaZero-style score for A's white games only, `(W + 0.5·D) / N`.
    /// 0 if A didn't play any white games in this run.
    var playerAScoreAsWhite: Double {
        let n = playerAWhiteGames
        guard n > 0 else { return 0 }
        return (Double(playerAWinsAsWhite) + 0.5 * Double(playerADrawsAsWhite)) / Double(n)
    }

    /// AlphaZero-style score for A's black games only.
    var playerAScoreAsBlack: Double {
        let n = playerABlackGames
        guard n > 0 else { return 0 }
        return (Double(playerAWinsAsBlack) + 0.5 * Double(playerADrawsAsBlack)) / Double(n)
    }
}

/// One completed game's worth of data harvested by the tournament
/// driver. Includes the captured `moveHistory` so the post-arena
/// validity sweep can replay every game through a fresh engine.
struct TournamentGameRecord: Sendable {
    let gameIndex: Int
    let aIsWhite: Bool
    let result: GameResult
    let moveHistory: [ChessMove]
}

/// Outcome of a post-tournament validity sweep — replays every
/// captured game through a fresh `ChessGameEngine` and verifies
/// that each move was legal in the position that preceded it.
///
/// This is a belt-and-suspenders check originally added to catch
/// the hypothetical "batched evaluation handed game A's policy to
/// game B" failure mode under the legacy concurrent-arena
/// task-per-game driver. Each `ChessMachine.applyMove` already
/// validates legality at apply time, so a successful sweep is
/// mostly confirmation that the captured `moveHistory` is
/// internally consistent.
struct TournamentValidityReport: Sendable {
    let gamesChecked: Int
    let totalMovesChecked: Int
    /// Index into the source records array of the first game that
    /// failed validation (if any). Nil = all games passed.
    let firstFailingRecordIndex: Int?
    let failureDescription: String?

    var passed: Bool { firstFailingRecordIndex == nil }
}
