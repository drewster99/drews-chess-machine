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
    /// Games that survived the per-game keep/drop filter in the
    /// self-play driver (i.e. `selfPlayDrawKeepFraction`) and got
    /// bulk-flushed into the replay buffer. A subset of `_totalGames`:
    /// decisive games always increment both, drawn games increment
    /// only `_totalGames` when dropped. With `selfPlayDrawKeepFraction
    /// = 1.0` (default) this equals `_totalGames`.
    private var _emittedGames: Int = 0
    /// Total plies emitted into the replay buffer across the session.
    /// Sums to `whitePlies + blackPlies` for every emitted game, so
    /// at default-keepFraction matches `_totalMoves`; under filtering
    /// this is the EFFECTIVE production-side position count (which is
    /// what `[STATS]` rates and the replay-ratio controller target).
    private var _emittedPositions: Int = 0
    private var _recentGames: [GameRecord] = []
    private var _recentGamesHead: Int = 0
    private var _recentGamesRunningMoves: Int = 0
    private var _recentGamesRunningWallMs: Double = 0.0
    /// Rolling 10-minute window of emitted games. Each entry is one
    /// game that was kept (passed the draw-keep filter) — its
    /// timestamp drives the rolling-rate display for
    /// "spRateEm" / "spGamesEmHr" in `[STATS]`. We don't share storage
    /// with `_recentGames` because the kept/dropped subsets diverge
    /// when filtering is active.
    private struct EmittedGameRecord {
        let timestamp: Date
        let positions: Int
    }
    private var _recentEmittedGames: [EmittedGameRecord] = []
    private var _recentEmittedGamesHead: Int = 0
    private var _recentEmittedGamesRunningPositions: Int = 0
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
    ///
    /// `emittedGames` / `emittedPositions` are Optional and back-compat
    /// only — sessions saved before the draw-keep filter existed had
    /// emitted == played, and the loader can leave these nil (they
    /// will default to the same totals as `totalGames` / `totalMoves`).
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
        trainingSteps: Int,
        emittedGames: Int? = nil,
        emittedPositions: Int? = nil
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
        self._emittedGames = emittedGames ?? totalGames
        self._emittedPositions = emittedPositions ?? totalMoves
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
            self._emittedGames = 0
            self._emittedPositions = 0
            self._recentGames.removeAll()
            self._recentGamesHead = 0
            self._recentGamesRunningMoves = 0
            self._recentGamesRunningWallMs = 0
            self._recentGameLengths.removeAll(keepingCapacity: true)
            self._recentGameLengthsHead = 0
            self._recentEmittedGames.removeAll()
            self._recentEmittedGamesHead = 0
            self._recentEmittedGamesRunningPositions = 0
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
            self._recentGamesRunningMoves += moves
            self._recentGamesRunningWallMs += durationMs
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

    /// Record one self-play game whose recorded plies were just
    /// flushed into the replay buffer. Called from the slot driver
    /// after the per-game keep/drop filter decided "keep." The driver
    /// always calls this *in addition* to `recordCompletedGame`
    /// (which counts every completed game regardless of filter), so
    /// `emittedGames <= totalGames`.
    func recordEmittedGame(positions: Int) {
        let now = Date()
        lock.withLock {
            self._emittedGames += 1
            self._emittedPositions += positions
            self._recentEmittedGames.append(
                EmittedGameRecord(timestamp: now, positions: positions)
            )
            self._recentEmittedGamesRunningPositions += positions
            self.pruneRecentEmittedLocked(now: now)
        }
    }

    /// Drop emitted-game rolling-window entries older than `now -
    /// recentWindow`. Caller must already hold `lock`. Same prefix-
    /// removal pattern as `pruneRecentLocked`.
    private func pruneRecentEmittedLocked(now: Date) {
        let cutoff = now.addingTimeInterval(-Self.recentWindow)
        while _recentEmittedGamesHead < _recentEmittedGames.count,
              _recentEmittedGames[_recentEmittedGamesHead].timestamp < cutoff {
            _recentEmittedGamesRunningPositions -= _recentEmittedGames[_recentEmittedGamesHead].positions
            _recentEmittedGamesHead += 1
        }
        if _recentEmittedGamesHead > _recentEmittedGames.count / 2 && _recentEmittedGamesHead > 100 {
            _recentEmittedGames.removeSubrange(0..<_recentEmittedGamesHead)
            _recentEmittedGamesHead = 0
        }
    }

    /// Drop rolling-window entries older than `now - recentWindow`.
    /// Caller must already hold `lock`. Records are appended in
    /// monotonic timestamp order so this is a prefix removal —
    /// O(k) where k is the expired count.
    ///
    /// `_recentGamesRunningMoves` (Int) is decremented exactly; the
    /// `Double` `_recentGamesRunningWallMs` accumulates a tiny amount
    /// of floating-point error per add/subtract, so it's re-summed
    /// from the surviving records whenever the backing array is
    /// compacted (which during active self-play happens regularly).
    /// That bounds the drift to "since the last compaction" rather
    /// than "since the session started".
    private func pruneRecentLocked(now: Date) {
        let cutoff = now.addingTimeInterval(-Self.recentWindow)
        while _recentGamesHead < _recentGames.count, _recentGames[_recentGamesHead].timestamp < cutoff {
            let r = _recentGames[_recentGamesHead]
            _recentGamesRunningMoves -= r.moves
            _recentGamesRunningWallMs -= r.durationMs
            _recentGamesHead += 1
        }

        if _recentGamesHead > _recentGames.count / 2 && _recentGamesHead > 100 {
            _recentGames.removeSubrange(0..<_recentGamesHead)
            _recentGamesHead = 0
            // Re-derive the Double accumulator exactly from the
            // surviving window, shedding any add/subtract round-off.
            _recentGamesRunningWallMs = _recentGames.reduce(0.0) { $0 + $1.durationMs }
        }
    }

    struct Snapshot: Sendable {
        // Lifetime totals
        let selfPlayGames: Int
        let selfPlayPositions: Int
        /// Lifetime total of self-play games that survived the
        /// per-game keep/drop filter (`selfPlayDrawKeepFraction`) and
        /// were bulk-flushed into the replay buffer. `<= selfPlayGames`.
        /// Equal at default keepFraction = 1.0.
        let emittedGames: Int
        /// Lifetime total of plies emitted into the replay buffer
        /// (white + black recorded plies summed across all kept games).
        /// `<= selfPlayPositions`; equal at default keepFraction.
        let emittedPositions: Int
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
        /// Rolling 10-minute window aggregates for emitted games — the
        /// kept subset of `recentGames`. Pair with `recentWindowSeconds`
        /// to derive emitted-games/hour and emitted-positions/sec.
        let recentEmittedGames: Int
        let recentEmittedPositions: Int
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
    ///
    /// Emits a `[DISPATCH-LATENCY]` line to stdout (Xcode console only,
    /// not the session log) when the round-trip exceeds 50 ms, broken
    /// into pre-continuation / queue-wait / lock-work components so a
    /// multi-second heartbeat tick can be attributed to a specific
    /// off-main snapshot read. Matches the instrumentation on
    /// `TournamentLiveBox.asyncSnapshot` / `GameDiversityTracker.asyncSnapshot`.
    func asyncSnapshot() async -> Snapshot {
        let start = Date()
        return await withCheckedContinuation { (cont: CheckedContinuation<Snapshot, Never>) in
            let inContinuation = Date()
            DispatchQueue.global(qos: .userInitiated).async {
                let dispatched = Date()
                let result = self.snapshot()
                let now = Date()
                let total = now.timeIntervalSince(start)
                if total > 0.05 {
                    let pre = inContinuation.timeIntervalSince(start)
                    let queue = dispatched.timeIntervalSince(inContinuation)
                    let work = now.timeIntervalSince(dispatched)
                    print(String(format: "[DISPATCH-LATENCY] ParallelWorkerStatsBox.asyncSnapshot: total=%.2fms (pre=%.2fms queue=%.2fms work=%.2fms)", total * 1000, pre * 1000, queue * 1000, work * 1000))
                }
                cont.resume(returning: result)
            }
        }
    }

    func snapshot() -> Snapshot {
        lock.withLock {
            let now = Date()
            pruneRecentLocked(now: now)
            pruneRecentEmittedLocked(now: now)

            let recentMoves = _recentGamesRunningMoves
            let recentGameWallMs = _recentGamesRunningWallMs
            let recentWindowSec: Double
            if _recentGamesHead < _recentGames.count {
                let oldest = _recentGames[_recentGamesHead].timestamp
                recentWindowSec = min(Self.recentWindow, now.timeIntervalSince(oldest))
            } else {
                recentWindowSec = 0
            }
            let recentEmittedGameCount = _recentEmittedGames.count - _recentEmittedGamesHead
            let recentEmittedPositionsValue = _recentEmittedGamesRunningPositions

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
                emittedGames: _emittedGames,
                emittedPositions: _emittedPositions,
                totalGameWallMs: _totalGameWallMs,
                whiteCheckmates: _whiteCheckmates,
                blackCheckmates: _blackCheckmates,
                stalemates: _stalemates,
                fiftyMoveDraws: _fiftyMoveDraws,
                threefoldRepetitionDraws: _threefoldRepetitionDraws,
                insufficientMaterialDraws: _insufficientMaterialDraws,
                trainingSteps: _trainingSteps,
                sessionStart: _sessionStart,
                recentGames: _recentGames.count - _recentGamesHead,
                recentMoves: recentMoves,
                recentGameWallMs: recentGameWallMs,
                recentEmittedGames: recentEmittedGameCount,
                recentEmittedPositions: recentEmittedPositionsValue,
                recentWindowSeconds: recentWindowSec,
                gameLenP50: p50,
                gameLenP95: p95
            )
        }
    }
}
