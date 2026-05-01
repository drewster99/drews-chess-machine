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

/// One completed game's worth of data harvested by a slot task and
/// merged by the parent task. Includes the captured `moveHistory`
/// so the post-arena validity sweep can replay every game through
/// a fresh engine.
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
/// This is a belt-and-suspenders check intended specifically to
/// catch the hypothetical "batched evaluation handed game A's
/// policy to game B" failure mode under concurrent arena play.
/// Each `ChessMachine.applyMove` already validates legality at
/// apply time, so a successful sweep is mostly confirmation that
/// the captured `moveHistory` is internally consistent. A failure
/// here would point at either the batcher cross-talk hypothesis
/// or a regression in `ChessMachine`'s legality plumbing.
struct TournamentValidityReport: Sendable {
    let gamesChecked: Int
    let totalMovesChecked: Int
    /// Index into the source records array of the first game that
    /// failed validation (if any). Nil = all games passed.
    let firstFailingRecordIndex: Int?
    let failureDescription: String?

    var passed: Bool { firstFailingRecordIndex == nil }
}

/// Replay every record's `moveHistory` through a fresh engine.
/// Cheap (a few legal-move generations per move) and catches the
/// "batching cross-talk" failure mode by surfacing any move that
/// isn't legal in its preceding position. Stops at the first
/// failing game so the report can carry diagnostic detail without
/// running the entire batch on a known-broken run.
func validateTournamentRecords(_ records: [TournamentGameRecord]) -> TournamentValidityReport {
    var totalMoves = 0
    for (idx, record) in records.enumerated() {
        let engine = ChessGameEngine()
        for (ply, move) in record.moveHistory.enumerated() {
            do {
                _ = try engine.applyMoveAndAdvance(move)
                totalMoves += 1
            } catch {
                let detail = "game gameIndex=\(record.gameIndex) ply=\(ply) move=\(move): \(error)"
                return TournamentValidityReport(
                    gamesChecked: idx + 1,
                    totalMovesChecked: totalMoves,
                    firstFailingRecordIndex: idx,
                    failureDescription: detail
                )
            }
        }
    }
    return TournamentValidityReport(
        gamesChecked: records.count,
        totalMovesChecked: totalMoves,
        firstFailingRecordIndex: nil,
        failureDescription: nil
    )
}

// MARK: - Tournament Driver

/// Plays a series of games between two player factories, alternating colors,
/// and collects statistics and training data.
///
/// `concurrency` controls how many games run in parallel. K=1 runs
/// games sequentially (one slot task at a time). K>1 runs K games
/// concurrently using `withThrowingTaskGroup`; each slot task pulls
/// the next gameIndex from the parent's serial counter, runs one
/// `ChessMachine.beginNewGame`, and returns a `TournamentGameRecord`.
/// The parent task tallies serially as records arrive — there is no
/// shared mutable state across slot tasks, no lock anywhere.
///
/// Color alternation by `gameIndex % 2` is preserved across all K:
/// the parent assigns the index at spawn time before adding the slot
/// task, so even with K games completing out of order the full set
/// of N games still produces an even white/black split for player A.
///
/// Usage:
/// ```
/// let network = try ChessMPSNetwork(.randomWeights)
/// let driver = TournamentDriver()
/// let stats = try await driver.run(
///     playerA: { MPSChessPlayer(name: "Net", source: DirectMoveEvaluationSource(network: network)) },
///     playerB: { RandomPlayer() },
///     games: 100,
///     concurrency: 1
/// )
/// print("Net wins: \(stats.playerAWins), Random wins: \(stats.playerBWins)")
/// ```
final class TournamentDriver {

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
    ///   - concurrency: Number of slot tasks running games in parallel.
    ///     Clamped to `[1, games]`.
    ///   - diversityTracker: Optional rolling-window tracker; each
    ///     completed game's move list is fed in serially from the
    ///     parent task as records arrive.
    ///   - isCancelled: Optional flag the driver checks between record
    ///     harvests; on a positive return the driver stops spawning
    ///     new games and the loop exits as in-flight slots return.
    ///   - onGameCompleted: Optional callback invoked from the parent
    ///     task after each finished game with `(completedSoFar, aWins,
    ///     bWins, draws)`. The first argument is the count of games
    ///     completed up to and including this one (1-based).
    ///   - onSlotExited: Optional async callback invoked from inside
    ///     each slot task right before it returns its game record.
    ///     Used by the parallel-arena callsite to decrement each
    ///     batcher's `expectedSlotCount` as games finish so the
    ///     count barrier remains achievable.
    /// - Returns: Aggregated win/loss/draw statistics. Per-game
    ///   records (used by callers that want to run post-tournament
    ///   checks) flow out via the `onGameRecorded` callback, fired
    ///   serially from the parent harvest loop as each slot returns.
    /// - Throws: Any non-cancellation error raised while running a game.
    func run(
        playerA: @escaping @Sendable () -> any ChessPlayer,
        playerB: @escaping @Sendable () -> any ChessPlayer,
        games: Int,
        concurrency: Int = 1,
        diversityTracker: GameDiversityTracker? = nil,
        isCancelled: (@Sendable () -> Bool)? = nil,
        onGameCompleted: (@Sendable (Int, Int, Int, Int) -> Void)? = nil,
        onSlotExited: (@Sendable () async -> Void)? = nil,
        onGameRecorded: (@Sendable (TournamentGameRecord) -> Void)? = nil
    ) async throws -> TournamentStats {
        let effectiveConcurrency = max(1, min(concurrency, games))

        // Parent-task accumulators. None of these are shared across
        // slot tasks — slot tasks return values, the parent merges
        // them serially as each `group.next()` resolves. No lock,
        // no actor, no DispatchQueue.
        var aWins = 0
        var bWins = 0
        var draws = 0
        var aWinsAsWhite = 0
        var aWinsAsBlack = 0
        var aLossesAsWhite = 0
        var aLossesAsBlack = 0
        var aDrawsAsWhite = 0
        var aDrawsAsBlack = 0
        var completed = 0
        var nextGameIndex = 0

        try await withThrowingTaskGroup(
            of: TournamentGameRecord?.self
        ) { group in
            // Slot-task closure shared between the initial fan-out
            // and each per-completion replenishment.
            //
            // Returns `nil` on cancellation so the parent's tally
            // loop can simply `continue` past it; a non-nil return
            // is an authoritative completed game.
            func spawnTask(forGameIndex gameIndex: Int) {
                let aIsWhite = gameIndex % 2 == 0
                group.addTask {
                    if Task.isCancelled { return nil }
                    if isCancelled?() == true { return nil }

                    let a = playerA()
                    let b = playerB()
                    let machine = ChessMachine()
                    // Tournament games run without a delegate — live
                    // animation only makes sense for single-game UI
                    // paths, not for K parallel slots.

                    let white: any ChessPlayer = aIsWhite ? a : b
                    let black: any ChessPlayer = aIsWhite ? b : a

                    let result: GameResult
                    do {
                        result = try await machine.beginNewGame(
                            white: white, black: black
                        )
                    } catch is CancellationError {
                        await onSlotExited?()
                        return nil
                    } catch {
                        await onSlotExited?()
                        throw error
                    }

                    await onSlotExited?()

                    return TournamentGameRecord(
                        gameIndex: gameIndex,
                        aIsWhite: aIsWhite,
                        result: result,
                        moveHistory: machine.moveHistory
                    )
                }
            }

            // Prime the first wave of K concurrent slots.
            let initialSpawn = min(effectiveConcurrency, games)
            for _ in 0..<initialSpawn {
                spawnTask(forGameIndex: nextGameIndex)
                nextGameIndex += 1
            }

            while let maybeRecord = try await group.next() {
                if Task.isCancelled || isCancelled?() == true {
                    // Stop fanning out; let in-flight slots drain.
                    continue
                }

                guard let record = maybeRecord else {
                    // Slot was cancelled. Spawn a replacement only
                    // if we still have games to play AND we haven't
                    // been asked to stop.
                    if nextGameIndex < games,
                       !Task.isCancelled,
                       isCancelled?() != true {
                        spawnTask(forGameIndex: nextGameIndex)
                        nextGameIndex += 1
                    }
                    continue
                }

                completed += 1
                let aIsWhite = record.aIsWhite
                switch record.result {
                case .checkmate(let winner):
                    let aWon = (winner == .white && aIsWhite)
                        || (winner == .black && !aIsWhite)
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

                diversityTracker?.recordGame(moves: record.moveHistory)
                onGameRecorded?(record)
                onGameCompleted?(completed, aWins, bWins, draws)

                // Refill the slot pool with the next game.
                if nextGameIndex < games {
                    spawnTask(forGameIndex: nextGameIndex)
                    nextGameIndex += 1
                }
            }
        }

        let stats = TournamentStats(
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
        return stats
    }
}
