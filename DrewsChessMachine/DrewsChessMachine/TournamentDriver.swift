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

// MARK: - Tournament Driver

/// Plays a series of games between two player factories, alternating colors,
/// and collects statistics and training data.
///
/// Usage:
/// ```
/// let network = try ChessMPSNetwork(.randomWeights)
/// let driver = TournamentDriver()
/// let stats = await driver.run(
///     playerA: { MPSChessPlayer(name: "Net", source: DirectMoveEvaluationSource(network: network)) },
///     playerB: { RandomPlayer() },
///     games: 100
/// )
/// print("Net wins: \(stats.playerAWins), Random wins: \(stats.playerBWins)")
/// ```
final class TournamentDriver {
    weak var delegate: (any ChessMachineDelegate)?

    /// Run a tournament of N games between two player factories.
    ///
    /// Colors alternate each game: game 0 → A is white, game 1 → A is black, etc.
    /// Player factories are called once per game to create fresh instances.
    ///
    /// Training positions recorded by `MPSChessPlayer` during a tournament
    /// are flushed directly into the player's attached `ReplayBuffer` (if
    /// any) at game end — the tournament driver does not re-collect them.
    /// Arena evaluation uses `MPSChessPlayer` instances without a replay
    /// buffer so no position data leaks into the self-play replay mix.
    ///
    /// - Parameters:
    ///   - playerA: Factory creating player A for each game.
    ///   - playerB: Factory creating player B for each game.
    ///   - games: Total number of games to play.
    ///   - onGameCompleted: Optional callback invoked after each finished
    ///     game with the running totals `(gameIndex, aWins, bWins,
    ///     draws)` (gameIndex is 1-based — "games completed so far").
    ///     Used by the arena-evaluation caller to push live progress
    ///     into a lock-protected box the UI heartbeat polls.
    /// - Returns: Aggregated win/loss/draw statistics.
    func run(
        playerA: @Sendable () -> any ChessPlayer,
        playerB: @Sendable () -> any ChessPlayer,
        games: Int,
        diversityTracker: GameDiversityTracker? = nil,
        isCancelled: (@Sendable () -> Bool)? = nil,
        onGameCompleted: (@Sendable (Int, Int, Int, Int) -> Void)? = nil
    ) async -> TournamentStats {
        var aWins = 0
        var bWins = 0
        var draws = 0
        // Per-side tallies — see the comment on `TournamentStats`
        // for why these matter in arena analysis.
        var aWinsAsWhite = 0
        var aWinsAsBlack = 0
        var aLossesAsWhite = 0
        var aLossesAsBlack = 0
        var aDrawsAsWhite = 0
        var aDrawsAsBlack = 0

        for gameIndex in 0..<games {
            // Two cancellation checks: the caller's own task (if this
            // run() is called from a non-detached context) AND an
            // externally-provided flag (used by arena callers that run
            // inside a detached task and therefore can't rely on
            // Task.isCancelled propagating from the outer driver).
            guard !Task.isCancelled else { break }
            if isCancelled?() == true { break }

            let aIsWhite = gameIndex % 2 == 0
            let a = playerA()
            let b = playerB()

            let machine = ChessMachine()
            machine.delegate = delegate

            let white: any ChessPlayer = aIsWhite ? a : b
            let black: any ChessPlayer = aIsWhite ? b : a

            // `beginNewGame` can throw `alreadyPlaying` (impossible here
            // — this machine was just constructed) or `CancellationError`
            // if the caller's Task is cancelled mid-game. On cancellation
            // we bail out of the tournament with whatever results we've
            // tallied so far rather than counting the partial game as a
            // draw — the outer loop's `Task.isCancelled` / `isCancelled`
            // guards would break on the next iteration anyway.
            let result: GameResult
            do {
                result = try await machine.beginNewGame(white: white, black: black)
            } catch is CancellationError {
                break
            } catch {
                // Non-cancellation error — engine hiccup, player
                // logic error, etc. Count as a draw for both the
                // side-agnostic total AND the per-side counter
                // corresponding to whichever side player A was
                // on. Skipping the per-side update would break
                // the identity `aDrawsAsWhite + aDrawsAsBlack ==
                // draws` that all downstream consumers rely on.
                draws += 1
                if aIsWhite { aDrawsAsWhite += 1 } else { aDrawsAsBlack += 1 }
                continue
            }

            diversityTracker?.recordGame(moves: machine.moveHistory)

            // Tally result
            switch result {
            case .checkmate(let winner):
                let aWon = (winner == .white && aIsWhite) || (winner == .black && !aIsWhite)
                if aWon {
                    aWins += 1
                    if aIsWhite { aWinsAsWhite += 1 } else { aWinsAsBlack += 1 }
                } else {
                    bWins += 1
                    if aIsWhite { aLossesAsWhite += 1 } else { aLossesAsBlack += 1 }
                }
            case .stalemate,
                 .drawByFiftyMoveRule,
                 .drawByInsufficientMaterial,
                 .drawByThreefoldRepetition:
                draws += 1
                if aIsWhite { aDrawsAsWhite += 1 } else { aDrawsAsBlack += 1 }
            }

            onGameCompleted?(gameIndex + 1, aWins, bWins, draws)
        }

        return TournamentStats(
            gamesPlayed: aWins + bWins + draws,
            playerAWins: aWins,
            playerBWins: bWins,
            draws: draws,
            playerAWinsAsWhite: aWinsAsWhite,
            playerAWinsAsBlack: aWinsAsBlack,
            playerALossesAsWhite: aLossesAsWhite,
            playerALossesAsBlack: aLossesAsBlack,
            playerADrawsAsWhite: aDrawsAsWhite,
            playerADrawsAsBlack: aDrawsAsBlack
        )
    }
}
