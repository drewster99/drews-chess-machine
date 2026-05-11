import Foundation
import os

/// Lock-protected rolling counters for the parallel self-play and
/// training workers. Each worker increments its own counters after
/// finishing one unit of work (one game for self-play, one SGD step
/// for training), and the UI heartbeat reads both to compute the
/// live positions-per-second rates shown in the busy label. Values
/// are monotonic for the life of the Play and Train session; wall-
/// clock rate is computed against `sessionStart`.
final class ParallelWorkerStatsBox: @unchecked Sendable {
    /// Rolling-window length in seconds for the "recent" rate column
    /// shown next to the lifetime rates in the Session panel. Fixed at
    /// 10 minutes so short-term throughput shifts (e.g. after an arena
    /// pause, or when training/self-play contention changes) become
    /// visible without having to watch the lifetime number drift.
    static let recentWindow: TimeInterval = 600

    /// One completed game, stored in the rolling window. Drops out of
    /// the window once its `timestamp` is more than `recentWindow`
    /// seconds behind `Date()`. Storage is O(games in the last 10
    /// minutes); at typical self-play rates this is a few thousand
    /// records max.
    private struct GameRecord {
        let timestamp: Date
        let moves: Int
        let durationMs: Double
    }

    private let lock = OSAllocatedUnfairLock()
    private var _totalGames: Int = 0
    private var _totalMoves: Int = 0
    private var _totalGameWallMs: Double = 0
    private var _whiteCheckmates: Int = 0
    private var _blackCheckmates: Int = 0
    private var _stalemates: Int = 0
    private var _fiftyMoveDraws: Int = 0
    private var _threefoldRepetitionDraws: Int = 0
    private var _insufficientMaterialDraws: Int = 0
    private var _trainingSteps: Int = 0
    private var _recentGames: [GameRecord] = []
    /// Fixed-capacity ring of recent game lengths (plies), used to
    /// compute p50/p95 in `Snapshot`. Sized for a few hundred games
    /// — plenty for a meaningful percentile on the 10-minute
    /// cadence without carrying the rolling-window memory back to
    /// a per-game scan. Lengths-only (no timestamps) because
    /// percentile semantics don't age the samples out by time.
    private var _recentGameLengths: [Int] = []
    private var _recentGameLengthsHead: Int = 0
    private static let gameLengthRingCapacity: Int = 512
    /// Wall-clock anchor used as the denominator for every session
    /// rate the UI shows (games/hr, moves/hr, steps/sec, avg move ms,
    /// "Total session time", ...). Initially set to the moment the
    /// box is created, then advanced by `markWorkersStarted()` to
    /// the moment the worker task group actually begins — Play-and-
    /// Train setup (network builds, trainer reset, weight copies)
    /// would otherwise inflate the denominator by several seconds
    /// and pull every rate down proportionally.
    private var _sessionStart: Date

    init(sessionStart: Date = Date()) {
        self._sessionStart = sessionStart
    }

    /// Seeded init for session resume. All counters pick up where
    /// the saved session left off so the UI shows continuity.
    init(
        sessionStart: Date,
        totalGames: Int,
        totalMoves: Int,
        totalGameWallMs: Double,
        whiteCheckmates: Int,
        blackCheckmates: Int,
        stalemates: Int,
        fiftyMoveDraws: Int,
        threefoldRepetitionDraws: Int,
        insufficientMaterialDraws: Int,
        trainingSteps: Int
    ) {
        self._sessionStart = sessionStart
        self._totalGames = totalGames
        self._totalMoves = totalMoves
        self._totalGameWallMs = totalGameWallMs
        self._whiteCheckmates = whiteCheckmates
        self._blackCheckmates = blackCheckmates
        self._stalemates = stalemates
        self._fiftyMoveDraws = fiftyMoveDraws
        self._threefoldRepetitionDraws = threefoldRepetitionDraws
        self._insufficientMaterialDraws = insufficientMaterialDraws
        self._trainingSteps = trainingSteps
    }

    /// Advance `sessionStart` to `Date()`. Called once from inside
    /// the Play-and-Train task, immediately before the worker group
    /// is spawned, so that rate denominators only cover the window
    /// in which workers are actually running. Captures `Date()` at
    /// the call site (not inside the closure) so the timestamp
    /// reflects the moment the caller invoked this method, not the
    /// moment the lock acquisition lands.
    func markWorkersStarted() {
        let now = Date()
        lock.withLock {
            self._sessionStart = now
        }
    }

    /// Reset game-play counters so post-promotion stats reflect
    /// only the newly-promoted champion's self-play performance.
    /// Training step count and sessionStart are NOT reset so
    /// training-rate display stays continuous.
    func resetGameStats() {
        lock.withLock {
            self._totalGames = 0
            self._totalMoves = 0
            self._totalGameWallMs = 0
            self._whiteCheckmates = 0
            self._blackCheckmates = 0
            self._stalemates = 0
            self._fiftyMoveDraws = 0
            self._threefoldRepetitionDraws = 0
            self._insufficientMaterialDraws = 0
            self._recentGames.removeAll()
            self._recentGameLengths.removeAll(keepingCapacity: true)
            self._recentGameLengthsHead = 0
        }
    }

    /// Record one completed self-play game. Called from every worker
    /// at game-end with the game's total moves, wall-clock duration,
    /// and final result. Bumps lifetime totals, the per-outcome
    /// counters, and the rolling 10-minute window. Thread-safe via
    /// the box's lock. `Date()` is captured at the call site — the
    /// game-end timestamp feeds the rolling-window rate stats, so
    /// it must reflect when the game actually finished rather than
    /// when the lock acquisition lands.
    func recordCompletedGame(moves: Int, durationMs: Double, result: GameResult) {
        let now = Date()
        lock.withLock {
            self._totalGames += 1
            self._totalMoves += moves
            self._totalGameWallMs += durationMs

            switch result {
            case .checkmate(let winner):
                if winner == .white {
                    self._whiteCheckmates += 1
                } else {
                    self._blackCheckmates += 1
                }
            case .stalemate:
                self._stalemates += 1
            case .drawByFiftyMoveRule:
                self._fiftyMoveDraws += 1
            case .drawByInsufficientMaterial:
                self._insufficientMaterialDraws += 1
            case .drawByThreefoldRepetition:
                self._threefoldRepetitionDraws += 1
            }

            self._recentGames.append(GameRecord(timestamp: now, moves: moves, durationMs: durationMs))
            self.pruneRecentLocked(now: now)

            // Percentile ring. Pre-sized lazily; FIFO overwrite once
            // full. `moves` is plies (not half-moves pairs), matching
            // every other game-length counter the app exposes.
            if self._recentGameLengths.count < Self.gameLengthRingCapacity {
                self._recentGameLengths.append(moves)
            } else {
                self._recentGameLengths[self._recentGameLengthsHead] = moves
                self._recentGameLengthsHead = (self._recentGameLengthsHead + 1) % Self.gameLengthRingCapacity
            }
        }
    }

    func recordTrainingStep() {
        lock.withLock {
            self._trainingSteps += 1
        }
    }

    /// Drop rolling-window entries older than `now - recentWindow`.
    /// Caller must already hold `lock`. Records are appended in
    /// monotonic timestamp order so this is a prefix removal —
    /// O(k) where k is the expired count.
    private func pruneRecentLocked(now: Date) {
        let cutoff = now.addingTimeInterval(-Self.recentWindow)
        while let first = _recentGames.first, first.timestamp < cutoff {
            _recentGames.removeFirst()
        }
    }

    struct Snapshot: Sendable {
        // Lifetime totals
        let selfPlayGames: Int
        let selfPlayPositions: Int
        let totalGameWallMs: Double
        let whiteCheckmates: Int
        let blackCheckmates: Int
        let stalemates: Int
        let fiftyMoveDraws: Int
        let threefoldRepetitionDraws: Int
        let insufficientMaterialDraws: Int
        let trainingSteps: Int
        let sessionStart: Date
        // Rolling 10-minute window aggregates
        let recentGames: Int
        let recentMoves: Int
        let recentGameWallMs: Double
        /// Effective denominator for rolling rate calculations, in
        /// seconds: `min(recentWindow, now - oldest-entry-timestamp)`.
        /// Starts at 0 and grows to `recentWindow` over the first 10
        /// minutes of a session. `0` means the window holds no games.
        let recentWindowSeconds: Double
        /// Percentiles of recent game lengths (plies) from the
        /// fixed-capacity length ring. `nil` while the ring is empty.
        /// Pairs with `avgLen` in the `[STATS]` line to expose the
        /// draw/mate bimodality that a mean alone hides.
        let gameLenP50: Int?
        let gameLenP95: Int?
    }

    /// Off-main async variant of `snapshot()`. Lock acquisition runs
    /// on a global executor so the awaiter (typically the main actor)
    /// is never synchronously blocked on `lock.withLock`.
    func asyncSnapshot() async -> Snapshot {
        await withCheckedContinuation { (cont: CheckedContinuation<Snapshot, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(returning: self.snapshot())
            }
        }
    }

    func snapshot() -> Snapshot {
        lock.withLock {
            let now = Date()
            pruneRecentLocked(now: now)

            var recentMoves = 0
            var recentGameWallMs: Double = 0
            for r in _recentGames {
                recentMoves += r.moves
                recentGameWallMs += r.durationMs
            }
            let recentWindowSec: Double
            if let oldest = _recentGames.first?.timestamp {
                recentWindowSec = min(Self.recentWindow, now.timeIntervalSince(oldest))
            } else {
                recentWindowSec = 0
            }

            let (p50, p95): (Int?, Int?) = {
                guard !_recentGameLengths.isEmpty else { return (nil, nil) }
                var sorted = _recentGameLengths
                sorted.sort()
                let n = sorted.count
                func idx(_ p: Double) -> Int {
                    max(0, min(n - 1, Int((p * Double(n - 1)).rounded())))
                }
                return (sorted[idx(0.50)], sorted[idx(0.95)])
            }()

            return Snapshot(
                selfPlayGames: _totalGames,
                selfPlayPositions: _totalMoves,
                totalGameWallMs: _totalGameWallMs,
                whiteCheckmates: _whiteCheckmates,
                blackCheckmates: _blackCheckmates,
                stalemates: _stalemates,
                fiftyMoveDraws: _fiftyMoveDraws,
                threefoldRepetitionDraws: _threefoldRepetitionDraws,
                insufficientMaterialDraws: _insufficientMaterialDraws,
                trainingSteps: _trainingSteps,
                sessionStart: _sessionStart,
                recentGames: _recentGames.count,
                recentMoves: recentMoves,
                recentGameWallMs: recentGameWallMs,
                recentWindowSeconds: recentWindowSec,
                gameLenP50: p50,
                gameLenP95: p95
            )
        }
    }
}
