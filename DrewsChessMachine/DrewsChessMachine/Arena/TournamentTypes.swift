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

/// One per-ply value-head readback captured during an arena game,
/// always from the candidate (trainer-snapshot) network's perspective.
/// Recorded only at plies where the candidate is to move — the value
/// scalar is then the candidate's own assessment of its position
/// (`p_win - p_loss ∈ [-1, +1]`, side-to-move = candidate).
struct CandidateValueSample: Sendable {
    /// 0-indexed absolute ply about to be played when this evaluation
    /// fired. (`ply == 0` ⇒ the very first half-move of the game.)
    let ply: Int
    /// Candidate network's value-head scalar at that ply, in `[-1, +1]`.
    /// Positive = candidate thinks it's winning.
    let value: Float
    /// Probability mass the candidate's policy placed on the move it
    /// actually played at this ply — the post-temperature legal-only
    /// softmax value at the chosen move. A calibration companion to
    /// `value`: `value` is how good the candidate *thinks* the
    /// position is; `policyProbability` is how committed it was to
    /// the move it chose.
    let policyProbability: Float
}

/// One completed game's worth of data harvested by the tournament
/// driver. Includes the captured `moveHistory` so the post-arena
/// validity sweep can replay every game through a fresh engine, and
/// the per-ply candidate value samples for the post-arena summary
/// breakdowns (mean strength by absolute ply / by game progress).
struct TournamentGameRecord: Sendable {
    let gameIndex: Int
    let aIsWhite: Bool
    let result: GameResult
    let moveHistory: [ChessMove]
    /// One sample per candidate-to-move ply in this game. Roughly
    /// half the length of `moveHistory` — exactly `ceil(N/2)` if the
    /// candidate is white, `floor(N/2)` if it's black, where
    /// `N = moveHistory.count`. Empty for legacy records and for
    /// games that ended before the candidate ever moved.
    let candidateValueSamples: [CandidateValueSample]
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
