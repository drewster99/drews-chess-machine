import Foundation

// MARK: - Tournament Stats

/// Accumulated results from a multi-game tournament.
struct TournamentStats: Sendable {
    let gamesPlayed: Int
    let playerAWins: Int
    let playerBWins: Int
    let draws: Int

    /// Collected training positions from all games (both players).
    let trainingPositions: [TrainingPosition]

    var playerAWinRate: Double {
        gamesPlayed > 0 ? Double(playerAWins) / Double(gamesPlayed) : 0
    }

    var playerBWinRate: Double {
        gamesPlayed > 0 ? Double(playerBWins) / Double(gamesPlayed) : 0
    }

    var drawRate: Double {
        gamesPlayed > 0 ? Double(draws) / Double(gamesPlayed) : 0
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
///     playerA: { MPSChessPlayer(name: "Net", network: network) },
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
    /// Player factories are called once per game to create fresh instances (so each
    /// game gets clean state for training data recording).
    ///
    /// - Parameters:
    ///   - playerA: Factory creating player A for each game.
    ///   - playerB: Factory creating player B for each game.
    ///   - games: Total number of games to play.
    ///   - collectTrainingPositions: Whether to gather `TrainingPosition`s
    ///     off each MPS player after their game ends. Default `true` for
    ///     compatibility with the training-data use case; pass `false`
    ///     from the arena-evaluation path to avoid holding ~184 MB of
    ///     useless tensors (arena games are played by two different
    ///     networks and the resulting positions don't match the self-
    ///     play replay-buffer distribution, so collecting them would
    ///     waste memory for nothing).
    ///   - onGameCompleted: Optional callback invoked after each finished
    ///     game with the running totals `(gameIndex, aWins, bWins,
    ///     draws)` (gameIndex is 1-based — "games completed so far").
    ///     Used by the arena-evaluation caller to push live progress
    ///     into a lock-protected box the UI heartbeat polls.
    /// - Returns: Aggregated statistics and training positions.
    func run(
        playerA: @Sendable () -> any ChessPlayer,
        playerB: @Sendable () -> any ChessPlayer,
        games: Int,
        collectTrainingPositions: Bool = true,
        isCancelled: (@Sendable () -> Bool)? = nil,
        onGameCompleted: (@Sendable (Int, Int, Int, Int) -> Void)? = nil
    ) async -> TournamentStats {
        var aWins = 0
        var bWins = 0
        var draws = 0
        var allPositions: [TrainingPosition] = []

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

            // beginNewGame only throws if a game is already in progress, and
            // this machine was just constructed — so the throw cannot happen
            // here. If it ever does, treat the game as a draw and continue.
            let task: Task<GameResult, Never>
            do {
                task = try machine.beginNewGame(white: white, black: black)
            } catch {
                draws += 1
                continue
            }
            let result = await task.value

            // Tally result
            switch result {
            case .checkmate(let winner):
                let aWon = (winner == .white && aIsWhite) || (winner == .black && !aIsWhite)
                if aWon { aWins += 1 } else { bWins += 1 }
            case .stalemate,
                 .drawByFiftyMoveRule,
                 .drawByInsufficientMaterial,
                 .drawByThreefoldRepetition:
                draws += 1
            }

            if collectTrainingPositions {
                if let mpsPlayer = a as? MPSChessPlayer {
                    allPositions.append(contentsOf: mpsPlayer.gamePositions)
                }
                if let mpsPlayer = b as? MPSChessPlayer {
                    allPositions.append(contentsOf: mpsPlayer.gamePositions)
                }
            }

            onGameCompleted?(gameIndex + 1, aWins, bWins, draws)
        }

        return TournamentStats(
            gamesPlayed: aWins + bWins + draws,
            playerAWins: aWins,
            playerBWins: bWins,
            draws: draws,
            trainingPositions: allPositions
        )
    }
}
