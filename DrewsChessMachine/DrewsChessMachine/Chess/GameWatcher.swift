import Foundation
import os

/// Holds live game state mutated by the ChessMachine delegate queue.
///
/// **Not @Observable.** SwiftUI doesn't observe its mutations directly —
/// instead, ContentView polls `snapshot()` on the heartbeat timer and copies
/// the values into local @State. That decouples UI redraw frequency from game
/// throughput: continuous self-play can run hundreds of moves per second
/// while the UI updates only on the heartbeat, and the game loop never waits
/// for SwiftUI invalidation.
///
/// Mutations come from two sources:
/// 1. The ChessMachine delegate queue (didApplyMove, gameEnded, playerErrored).
/// 2. Direct calls from ContentView actions (resetCurrentGame, markPlaying).
/// Both go through `lock`, so reads from any thread are safe.
final class GameWatcher: ChessMachineDelegate, @unchecked Sendable {
    /// Snapshot of all displayable values, taken atomically. Sendable so it
    /// can flow from any thread to the main actor.
    struct Snapshot: Sendable {
        var state: GameState = .starting
        var result: GameResult?
        var moveCount = 0
        var isPlaying = false
        var lastGameStats: GameStats?

        var totalGames = 0
        var totalMoves = 0
        var totalGameTimeMs: Double = 0
        var totalWhiteThinkMs: Double = 0
        var totalBlackThinkMs: Double = 0

        /// Cumulative wall-clock seconds spent in active play (between
        /// markPlaying(true) and the matching markPlaying(false) /
        /// gameEnded / playerErrored). Idle time between Play clicks
        /// and the small inter-game pauses in continuous play are excluded.
        var activePlaySeconds: Double = 0
        /// Set to the wall-clock time at which play started; cleared on stop.
        /// While non-nil, the live "active seconds" includes (now - this).
        var currentPlayStartTime: CFAbsoluteTime?

        var whiteCheckmates = 0
        var blackCheckmates = 0
        var stalemates = 0
        var fiftyMoveDraws = 0
        var insufficientMaterialDraws = 0
        var threefoldRepetitionDraws = 0
    }

    private let lock = OSAllocatedUnfairLock<Snapshot>(initialState: Snapshot())

    func snapshot() -> Snapshot {
        lock.withLock { $0 }
    }

    /// Off-main async variant of `snapshot()`. Lock acquisition runs on
    /// a global executor so the awaiter (typically the main actor) is
    /// never synchronously blocked on `lock.withLock`. The continuation
    /// is checked rather than unsafe so a buggy refactor surfaces a
    /// runtime warning instead of a silent hang.
    func asyncSnapshot() async -> Snapshot {
        await withCheckedContinuation { (cont: CheckedContinuation<Snapshot, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(returning: self.snapshot())
            }
        }
    }

    func resetCurrentGame() {
        lock.withLock { s in
            s.state = .starting
            s.result = nil
            s.moveCount = 0
            // Keep lastGameStats — show previous game until the next one ends
        }
    }

    func resetAll() {
        lock.withLock { $0 = Snapshot() }
    }

    func markPlaying(_ playing: Bool) {
        let now = CFAbsoluteTimeGetCurrent()
        lock.withLock { s in
            Self.setPlayingLocked(&s, playing: playing, now: now)
        }
    }

    /// Toggle isPlaying and update the active-play stopwatch. Caller
    /// must already hold the lock and pass a `now` value captured at
    /// the original call site — the per-call timestamp reflects the
    /// moment the caller decided to start/stop play. Idempotent:
    /// calling with the same value twice is a no-op for the stopwatch.
    private static func setPlayingLocked(_ s: inout Snapshot, playing: Bool, now: CFAbsoluteTime) {
        if playing {
            if s.currentPlayStartTime == nil {
                s.currentPlayStartTime = now
            }
        } else if let start = s.currentPlayStartTime {
            s.activePlaySeconds += max(0, now - start)
            s.currentPlayStartTime = nil
        }
        s.isPlaying = playing
    }

    // MARK: - Delegate (called on ChessMachine.delegateQueue, never main)

    func chessMachine(_ machine: ChessMachine, didApplyMove move: ChessMove, newState: GameState) {
        lock.withLock { s in
            s.state = newState
            s.moveCount += 1
        }
    }

    func chessMachine(
        _ machine: ChessMachine,
        gameEndedWith result: GameResult,
        finalState: GameState,
        stats: GameStats
    ) {
        let now = CFAbsoluteTimeGetCurrent()
        lock.withLock { s in
            s.result = result
            s.state = finalState
            s.lastGameStats = stats
            Self.setPlayingLocked(&s, playing: false, now: now)

            s.totalGames += 1
            s.totalMoves += stats.totalMoves
            s.totalGameTimeMs += stats.totalGameTimeMs
            s.totalWhiteThinkMs += stats.whiteThinkingTimeMs
            s.totalBlackThinkMs += stats.blackThinkingTimeMs
            // Move counting handed off to totalMoves; zero the per-game counter
            // atomically so display helpers using `totalMoves + moveCount` don't
            // double-count between gameEnded and the next resetCurrentGame call.
            s.moveCount = 0

            switch result {
            case .checkmate(let winner):
                if winner == .white {
                    s.whiteCheckmates += 1
                } else {
                    s.blackCheckmates += 1
                }
            case .stalemate:
                s.stalemates += 1
            case .drawByFiftyMoveRule:
                s.fiftyMoveDraws += 1
            case .drawByInsufficientMaterial:
                s.insufficientMaterialDraws += 1
            case .drawByThreefoldRepetition:
                s.threefoldRepetitionDraws += 1
            }
        }
    }

    func chessMachine(_ machine: ChessMachine, playerErrored player: any ChessPlayer, error: any Error) {
        let now = CFAbsoluteTimeGetCurrent()
        lock.withLock { s in
            Self.setPlayingLocked(&s, playing: false, now: now)
        }
    }
}

// MARK: - Snapshot Display Helpers

extension GameWatcher.Snapshot {
    var avgMoveTimeMs: Double {
        totalMoves > 0 ? (totalWhiteThinkMs + totalBlackThinkMs) / Double(totalMoves) : 0
    }

    var avgGameTimeMs: Double {
        totalGames > 0 ? totalGameTimeMs / Double(totalGames) : 0
    }

    /// Discrete throughput: based only on completed-game time. Equivalent to
    /// 3600 / avgGameTimeSec. Updates only when a game ends, so it never
    /// drifts mid-game.
    func gamesPerHour() -> Double {
        guard totalGames > 0, totalGameTimeMs > 0 else { return 0 }
        let totalGameSec = totalGameTimeMs / 1000
        return Double(totalGames) / (totalGameSec / 3600)
    }

    /// Live throughput: includes the in-progress game's moves so the rate
    /// updates smoothly. Denominator is active play time, so idle gaps
    /// between Play clicks don't depress the value.
    func movesPerHour(now: CFAbsoluteTime) -> Double {
        let moves = totalMoves + moveCount
        let secs = activeSeconds(now: now)
        guard moves > 0, secs > 0 else { return 0 }
        return Double(moves) / (secs / 3600)
    }

    /// Live cumulative seconds spent in active play.
    func activeSeconds(now: CFAbsoluteTime) -> Double {
        var sec = activePlaySeconds
        if let start = currentPlayStartTime {
            sec += max(0, now - start)
        }
        return sec
    }

    /// True once any play has started — used to decide whether to show
    /// "--" for time and rates.
    var hasSession: Bool {
        activePlaySeconds > 0 || currentPlayStartTime != nil
    }

    /// Fixed-layout stats text. Every section is always present — values show
    /// "--" when no data exists. Structure never changes so layout stays stable.
    func statsText(continuousPlay: Bool) -> String {
        let dash = "--"
        let now = CFAbsoluteTimeGetCurrent()

        // Status line
        let status: String
        if isPlaying {
            let turn = state.currentPlayer == .white ? "White" : "Black"
            let check = MoveGenerator.isInCheck(state, color: state.currentPlayer) ? " CHECK" : ""
            status = "\(turn) to move (move \(moveCount + 1))\(check)"
        } else if let result {
            switch result {
            case .checkmate(let winner):
                status = "\(winner == .white ? "White" : "Black") wins by checkmate"
            case .stalemate:
                status = "Draw by stalemate"
            case .drawByFiftyMoveRule:
                status = "Draw by fifty-move rule"
            case .drawByInsufficientMaterial:
                status = "Draw by insufficient material"
            case .drawByThreefoldRepetition:
                status = "Draw by threefold repetition"
            }
        } else {
            status = dash
        }

        // Session — totalMoves only counts completed games' moves; add the
        // in-progress game's moveCount so the displayed count and rate update
        // smoothly instead of jumping at game-end.
        let liveMoves = totalMoves + moveCount
        let sGames = totalGames > 0 ? totalGames.formatted(.number.grouping(.automatic)) : dash
        let sMoves = liveMoves > 0 ? liveMoves.formatted(.number.grouping(.automatic)) : dash
        let sTime = hasSession ? Self.formatHMS(seconds: activeSeconds(now: now)) : dash
        let sAvgMove = totalMoves > 0 ? String(format: "%.2f ms", avgMoveTimeMs) : dash
        let sAvgGame = totalGames > 0 ? String(format: "%.1f ms", avgGameTimeMs) : dash
        let sMovesHr = liveMoves > 0
        ? Int(movesPerHour(now: now).rounded()).formatted(.number.grouping(.automatic))
        : dash
        let sGamesHr = totalGames > 0
        ? Int(gamesPerHour().rounded()).formatted(.number.grouping(.automatic))
        : dash

        // Last Game — only shown when not in active continuous play. During
        // continuous play it has no value (it's already part of session totals
        // and changes too fast to read).
        let lastGameSection: String
        if continuousPlay {
            lastGameSection = ""
        } else if let stats = lastGameStats {
            let lgMoves = String(format: "%d (%dW + %dB)", stats.totalMoves, stats.whiteMoves, stats.blackMoves)
            let lgTime = String(format: "%.1f ms", stats.totalGameTimeMs)
            let lgAvg = String(format: "%.2f ms", stats.avgMoveTimeMs)
            let lgWAvg = String(format: "%.2f ms", stats.avgWhiteMoveTimeMs)
            let lgBAvg = String(format: "%.2f ms", stats.avgBlackMoveTimeMs)
            lastGameSection = """
                Last Game
                  Moves:     \(lgMoves)
                  Time:      \(lgTime)
                  Avg move:  \(lgAvg)
                  White avg: \(lgWAvg)
                  Black avg: \(lgBAvg)


                """
        } else {
            lastGameSection = ""
        }

        return """
            Status: \(status)

            \(lastGameSection)Session
              Games:     \(sGames)
              Moves:     \(sMoves)
              Time:      \(sTime)
              Avg move:  \(sAvgMove)
              Avg game:  \(sAvgGame)
              Moves/hr:  \(sMovesHr)
              Games/hr:  \(sGamesHr)

            Results
              Checkmate:      \(whiteCheckmates + blackCheckmates)\(pct(whiteCheckmates + blackCheckmates))
                White wins:     \(whiteCheckmates)\(pct(whiteCheckmates))
                Black wins:     \(blackCheckmates)\(pct(blackCheckmates))
              Stalemate:      \(stalemates)\(pct(stalemates))
              50-move draw:   \(fiftyMoveDraws)\(pct(fiftyMoveDraws))
              Threefold rep:  \(threefoldRepetitionDraws)\(pct(threefoldRepetitionDraws))
              Insufficient:   \(insufficientMaterialDraws)\(pct(insufficientMaterialDraws))
            """
    }

    private func pct(_ count: Int) -> String {
        guard totalGames > 0 else { return "" }
        return String(format: "  (%.1f%%)", Double(count) / Double(totalGames) * 100)
    }

    static func formatHMS(seconds: Double) -> String {
        let totalSeconds = Int(seconds.rounded(.down))
        let h = totalSeconds / 3600
        let m = (totalSeconds % 3600) / 60
        let s = totalSeconds % 60
        return String(format: "%02d:%02d:%02d", h, m, s)
    }
}
