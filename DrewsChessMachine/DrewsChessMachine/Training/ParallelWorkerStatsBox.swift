import Foundation
import os

// MARK: - Phase histograms (shared between flush + box)

/// 5-bucket histogram used for both ply-phase and material-phase
/// position counts. Bucket cutoffs match `ReplayBuffer.computeBatchStats`
/// — open/early/mid/late/end — so emit-window stats line up
/// semantically with per-batch stats. Tuple-backed (no heap) and
/// `Sendable` so the per-game record can move freely between actors.
public struct PhaseHistogram: Sendable, Equatable {
    public var open: Int
    public var early: Int
    public var mid: Int
    public var late: Int
    public var end: Int

    public static let zero = PhaseHistogram(open: 0, early: 0, mid: 0, late: 0, end: 0)

    public var total: Int { open + early + mid + late + end }

    public static func + (a: Self, b: Self) -> Self {
        PhaseHistogram(
            open: a.open + b.open,
            early: a.early + b.early,
            mid: a.mid + b.mid,
            late: a.late + b.late,
            end: a.end + b.end
        )
    }

    public static func += (a: inout Self, b: Self) {
        a.open += b.open
        a.early += b.early
        a.mid += b.mid
        a.late += b.late
        a.end += b.end
    }

    /// Bucket a **game-total** ply index (0-indexed half-move count
    /// from the start of the game — same scale across both sides)
    /// using the same cutoffs as `ReplayBuffer.computeBatchStats`'s
    /// per-batch phase_by_ply.
    /// 0 = open (≤15), 1 = early (16–35), 2 = mid (36–75),
    /// 3 = late (76–125), 4 = end (126+).
    public static func plyBucket(ply: Int) -> Int {
        if ply <= 15 { return 0 }
        if ply <= 35 { return 1 }
        if ply <= 75 { return 2 }
        if ply <= 125 { return 3 }
        return 4
    }

    /// Bucket a non-pawn piece count using the same cutoffs as
    /// `ReplayBuffer.computeBatchStats`'s per-batch phase_by_material.
    /// 0 = open (≥14), 1 = early (12–13), 2 = mid (8–11),
    /// 3 = late (4–7), 4 = end (≤3).
    public static func materialBucket(materialCount: Int) -> Int {
        if materialCount >= 14 { return 0 }
        if materialCount >= 12 { return 1 }
        if materialCount >= 8 { return 2 }
        if materialCount >= 4 { return 3 }
        return 4
    }
}

/// Per-game flush summary — what one game's flushed plies contributed
/// to the replay buffer. Returned per-side by `ActiveGame.flush(buffer:result:)`
/// (white half + black half summed) and handed by `BatchedSelfPlayDriver`
/// to `ParallelWorkerStatsBox.recordEmittedGame`.
public struct FlushedGameStats: Sendable {
    public let positions: Int
    public let phaseByPly: PhaseHistogram
    public let phaseByMaterial: PhaseHistogram

    public static let empty = FlushedGameStats(
        positions: 0, phaseByPly: .zero, phaseByMaterial: .zero
    )

    public static func + (a: Self, b: Self) -> Self {
        FlushedGameStats(
            positions: a.positions + b.positions,
            phaseByPly: a.phaseByPly + b.phaseByPly,
            phaseByMaterial: a.phaseByMaterial + b.phaseByMaterial
        )
    }
}

/// Lock-protected rolling counters for the parallel self-play and
/// training workers. Each worker increments its own counters after
/// finishing one unit of work (one game for self-play, one SGD step
/// for training), and the UI heartbeat reads both to compute the
/// live positions-per-second rates shown in the busy label. Values
/// are monotonic for the life of the Play and Train session; wall-
/// clock rate is computed against `sessionStart`.
final class ParallelWorkerStatsBox: @unchecked Sendable {
    /// Rolling-window length in seconds for the "recent" rate columns
    /// shown next to the lifetime rates in the Session panel. Fixed at
    /// 1 minute so the displayed rate reacts quickly to throughput
    /// shifts (e.g. after an arena pause, or when training/self-play
    /// contention changes) without lagging behind by minutes.
    static let recentWindow: TimeInterval = 60

    /// One completed game, stored in the rolling window. Drops out of
    /// the window once its `timestamp` is more than `recentWindow`
    /// seconds behind `Date()`. Storage is O(games in the last minute);
    /// at typical self-play rates this is a few hundred records max.
    /// `result` is held so `snapshot()` can tally per-outcome counts
    /// over the rolling window for the popover Live snapshot's
    /// recent-window W/D/L/Drop breakdown — without it the per-outcome
    /// display could only show lifetime totals, which dominate any
    /// short recent change to the draw-keep filter or the max-plies
    /// cap.
    ///
    /// `result` is a `RawGameResult` rather than a `GameResult` so
    /// max-plies-dropped games can share this ring with natural-
    /// termination games — the rolling moves/plies accumulators
    /// then already include them without parallel bookkeeping.
    /// `snapshot()` switches on the case to tally either the natural
    /// per-outcome counter or the dropped counter.
    private struct GameRecord {
        let timestamp: Date
        let moves: Int
        let durationMs: Double
        let result: RawGameResult
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
    /// Lifetime count of self-play games that hit the configured
    /// `selfPlayMaxPliesPerGame` cap before terminating naturally and were
    /// dropped (never emitted to the replay buffer). Increments
    /// alongside `_totalGames` and `_totalMoves` — the game WAS
    /// played, just discarded after the fact.
    private var _maxPliesDropped: Int = 0
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
    /// Per-outcome lifetime tallies of games that survived the draw-keep
    /// filter and were flushed into the replay buffer. Decisive emitted
    /// counters (`_emittedWhiteCheckmates`, `_emittedBlackCheckmates`)
    /// always equal their played-side counterparts because decisives
    /// bypass the filter; only the four draw-type emitted counters can
    /// fall below their played-side equivalents.
    private var _emittedWhiteCheckmates: Int = 0
    private var _emittedBlackCheckmates: Int = 0
    private var _emittedStalemates: Int = 0
    private var _emittedFiftyMoveDraws: Int = 0
    private var _emittedThreefoldRepetitionDraws: Int = 0
    private var _emittedInsufficientMaterialDraws: Int = 0
    private var _recentGames: [GameRecord] = []
    private var _recentGamesHead: Int = 0
    private var _recentGamesRunningMoves: Int = 0
    private var _recentGamesRunningWallMs: Double = 0.0
    /// Rolling-window storage of emitted games (window length =
    /// `recentWindow`, currently 1 minute). Each entry is one
    /// game that was kept (passed the draw-keep filter) — its
    /// timestamp drives the rolling-rate display for
    /// "spRateEm" / "spGamesEmHr" in `[STATS]`. We don't share storage
    /// with `_recentGames` because the kept/dropped subsets diverge
    /// when filtering is active. `result` is recorded so `snapshot()`
    /// can tally per-outcome counts for the popover Live snapshot's
    /// Emitted-side recent W/D/L breakdown.
    private struct EmittedGameRecord {
        let timestamp: Date
        let positions: Int
        let result: GameResult
        /// Per-game ply-phase histogram (open/early/mid/late/end
        /// position counts). Sums to `positions`. Aggregated across
        /// the rolling window in `snapshot()` for the View > Emit
        /// Window panel.
        let phaseByPly: PhaseHistogram
        /// Per-game material-phase histogram. Same shape as
        /// `phaseByPly` but bucketed by non-pawn piece count.
        let phaseByMaterial: PhaseHistogram
    }
    private var _recentEmittedGames: [EmittedGameRecord] = []
    private var _recentEmittedGamesHead: Int = 0
    private var _recentEmittedGamesRunningPositions: Int = 0
    /// Fixed-capacity ring of recent game lengths (plies), used to
    /// compute p50/p95 in `Snapshot`. Sized for a few hundred games
    /// — plenty for a meaningful percentile without carrying the
    /// rolling-window memory back to a per-game scan. Lengths-only
    /// (no timestamps) because percentile semantics don't age the
    /// samples out by time.
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
        maxPliesDropped: Int? = nil,
        trainingSteps: Int,
        emittedGames: Int? = nil,
        emittedPositions: Int? = nil,
        emittedWhiteCheckmates: Int? = nil,
        emittedBlackCheckmates: Int? = nil,
        emittedStalemates: Int? = nil,
        emittedFiftyMoveDraws: Int? = nil,
        emittedThreefoldRepetitionDraws: Int? = nil,
        emittedInsufficientMaterialDraws: Int? = nil
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
        // Sessions saved before the max-plies feature had no dropped
        // games (the cap didn't exist), so absent values default to 0.
        self._maxPliesDropped = maxPliesDropped ?? 0
        self._trainingSteps = trainingSteps
        self._emittedGames = emittedGames ?? totalGames
        self._emittedPositions = emittedPositions ?? totalMoves
        // Sessions saved before per-outcome emitted counters existed
        // had emitted == played at all outcome levels (the filter was
        // either disabled or absent), so default the missing values to
        // the played-side counterparts.
        self._emittedWhiteCheckmates = emittedWhiteCheckmates ?? whiteCheckmates
        self._emittedBlackCheckmates = emittedBlackCheckmates ?? blackCheckmates
        self._emittedStalemates = emittedStalemates ?? stalemates
        self._emittedFiftyMoveDraws = emittedFiftyMoveDraws ?? fiftyMoveDraws
        self._emittedThreefoldRepetitionDraws = emittedThreefoldRepetitionDraws ?? threefoldRepetitionDraws
        self._emittedInsufficientMaterialDraws = emittedInsufficientMaterialDraws ?? insufficientMaterialDraws
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
            self._maxPliesDropped = 0
            self._emittedGames = 0
            self._emittedPositions = 0
            self._emittedWhiteCheckmates = 0
            self._emittedBlackCheckmates = 0
            self._emittedStalemates = 0
            self._emittedFiftyMoveDraws = 0
            self._emittedThreefoldRepetitionDraws = 0
            self._emittedInsufficientMaterialDraws = 0
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
    /// counters, and the rolling-window aggregates. Thread-safe via
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

            self._recentGames.append(GameRecord(
                timestamp: now,
                moves: moves,
                durationMs: durationMs,
                result: .terminatedNormally(result)
            ))
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

    /// Record one self-play game that hit the configured `selfPlayMaxPliesPerGame`
    /// cap and was dropped before reaching the replay buffer. Bumps
    /// `_totalGames`, `_totalMoves`, `_totalGameWallMs`, and the
    /// dedicated `_maxPliesDropped` lifetime counter — the game WAS
    /// played, so it counts toward generation stats and the rolling
    /// plies/hour rate, but it never contributes to per-outcome W/D/L
    /// tallies (it's its own "dropped" category). The rolling-window
    /// entry uses `RawGameResult.terminatedEarly` so `snapshot()` can
    /// tally `recentMaxPliesDropped` alongside the W/D/L counts.
    func recordDroppedGame(moves: Int, durationMs: Double) {
        let now = Date()
        lock.withLock {
            self._totalGames += 1
            self._totalMoves += moves
            self._totalGameWallMs += durationMs
            self._maxPliesDropped += 1

            self._recentGames.append(GameRecord(
                timestamp: now,
                moves: moves,
                durationMs: durationMs,
                result: .terminatedEarly
            ))
            self._recentGamesRunningMoves += moves
            self._recentGamesRunningWallMs += durationMs
            self.pruneRecentLocked(now: now)

            // Percentile ring — dropped games still contribute their
            // length so the p50/p95 reflect everything the workers
            // produced, not just the kept subset.
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
    /// `emittedGames <= totalGames`. The `result` argument increments
    /// the per-outcome emitted counter that mirrors the played-side
    /// counter; the difference between played and emitted at each
    /// outcome category is what the draw-keep filter actually dropped.
    /// `flushed` carries the combined-from-both-players phase
    /// histograms — the rolling-window record stores them so
    /// `snapshot()` can aggregate the View > Emit Window panel.
    func recordEmittedGame(result: GameResult, flushed: FlushedGameStats) {
        let now = Date()
        lock.withLock {
            self._emittedGames += 1
            self._emittedPositions += flushed.positions
            switch result {
            case .checkmate(let winner):
                if winner == .white {
                    self._emittedWhiteCheckmates += 1
                } else {
                    self._emittedBlackCheckmates += 1
                }
            case .stalemate:
                self._emittedStalemates += 1
            case .drawByFiftyMoveRule:
                self._emittedFiftyMoveDraws += 1
            case .drawByInsufficientMaterial:
                self._emittedInsufficientMaterialDraws += 1
            case .drawByThreefoldRepetition:
                self._emittedThreefoldRepetitionDraws += 1
            }
            self._recentEmittedGames.append(
                EmittedGameRecord(
                    timestamp: now,
                    positions: flushed.positions,
                    result: result,
                    phaseByPly: flushed.phaseByPly,
                    phaseByMaterial: flushed.phaseByMaterial
                )
            )
            self._recentEmittedGamesRunningPositions += flushed.positions
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
        /// Lifetime count of games dropped for hitting the `selfPlayMaxPliesPerGame`
        /// cap. These ARE included in `selfPlayGames` and `selfPlayPositions`
        /// (they were played) but are NOT in any per-outcome counter
        /// (they're their own "dropped" category, never in W/D/L).
        let maxPliesDropped: Int
        /// Per-outcome lifetime tallies of games that survived the
        /// draw-keep filter. Decisive emitted counters always equal
        /// their played-side counterparts; only the four draw types
        /// can fall below.
        let emittedWhiteCheckmates: Int
        let emittedBlackCheckmates: Int
        let emittedStalemates: Int
        let emittedFiftyMoveDraws: Int
        let emittedThreefoldRepetitionDraws: Int
        let emittedInsufficientMaterialDraws: Int
        let trainingSteps: Int
        let sessionStart: Date
        // Rolling-window aggregates (window length = `recentWindow`).
        let recentGames: Int
        let recentMoves: Int
        let recentGameWallMs: Double
        /// Rolling-window aggregates for emitted games — the kept
        /// subset of `recentGames`. Pair with `recentWindowSeconds`
        /// to derive emitted-games/hour and emitted-positions/sec.
        let recentEmittedGames: Int
        let recentEmittedPositions: Int
        /// Rolling-window per-outcome counts on the *played* side
        /// (every completed game in the window, regardless of the
        /// draw-keep filter). Sum to `recentGames`. Used by the
        /// popover Live snapshot to render the W/D/L row over the
        /// recent window — the lifetime per-outcome counters dwarf
        /// any short post-filter-change interval and visually look
        /// identical between Played and Emitted columns until the
        /// filter has been running long enough to move lifetime
        /// totals by ≥ 0.1%.
        let recentWhiteCheckmates: Int
        let recentBlackCheckmates: Int
        let recentDraws: Int
        /// Rolling-window count of self-play games that hit the
        /// `selfPlayMaxPliesPerGame` cap and were dropped. Sums with the W/D/L
        /// counts above to equal `recentGames`. Drives the "Drop %"
        /// column of the popover Live snapshot's Played row.
        let recentMaxPliesDropped: Int
        /// Same per-outcome breakdown over the rolling window for the
        /// *emitted* side (games that survived the keep filter).
        /// `recentEmittedWhiteCheckmates` and `recentEmittedBlackCheckmates`
        /// equal their played-side counterparts because decisive games
        /// bypass the filter; only `recentEmittedDraws` ever falls
        /// below `recentDraws`.
        let recentEmittedWhiteCheckmates: Int
        let recentEmittedBlackCheckmates: Int
        let recentEmittedDraws: Int

        /// Aggregated emit-window phase histograms — one slot per
        /// (ply-phase or material-phase) bucket × outcome (W/D/L).
        /// Sum of position counts across all kept games in the
        /// rolling window. Drives the View > Emit Window panel.
        let emitWindowPhaseByPlyW: PhaseHistogram
        let emitWindowPhaseByPlyD: PhaseHistogram
        let emitWindowPhaseByPlyL: PhaseHistogram
        let emitWindowPhaseByMaterialW: PhaseHistogram
        let emitWindowPhaseByMaterialD: PhaseHistogram
        let emitWindowPhaseByMaterialL: PhaseHistogram
        /// Effective denominator for rolling rate calculations, in
        /// seconds: `min(recentWindow, now - oldest-entry-timestamp)`.
        /// Starts at 0 and grows to `recentWindow` over the first
        /// `recentWindow` seconds of a session. `0` means the window
        /// holds no games.
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

            // Per-outcome rolling-window tallies. One pass over the
            // surviving prefix-trimmed window per side. At a 1-minute
            // window and typical self-play rates the window holds
            // O(100) entries, so the cost is negligible compared to
            // the percentile sort below — and avoids carrying six
            // running counters that have to be kept in sync with
            // every append + prune.
            var recW = 0
            var recL = 0
            var recD = 0
            var recDropped = 0
            for i in _recentGamesHead..<_recentGames.count {
                switch _recentGames[i].result {
                case .terminatedNormally(.checkmate(let winner)):
                    if winner == .white { recW += 1 } else { recL += 1 }
                case .terminatedNormally(.stalemate),
                     .terminatedNormally(.drawByFiftyMoveRule),
                     .terminatedNormally(.drawByInsufficientMaterial),
                     .terminatedNormally(.drawByThreefoldRepetition):
                    recD += 1
                case .terminatedEarly:
                    recDropped += 1
                }
            }
            var recEmW = 0
            var recEmL = 0
            var recEmD = 0
            // Per-outcome × per-bucket aggregates for the View > Emit
            // Window panel. Each emitted game contributes its full
            // phase histogram into the bucket matching its outcome
            // (W = white checkmate, L = black checkmate, D = any of
            // the four draw types). `total` of `winPly + drawPly +
            // lossPly` equals the rolling-window emitted-positions
            // total; same for material. Iteration cost is the same
            // walk we're already doing for the per-outcome counts —
            // no second pass.
            var winPly = PhaseHistogram.zero
            var drawPly = PhaseHistogram.zero
            var lossPly = PhaseHistogram.zero
            var winMat = PhaseHistogram.zero
            var drawMat = PhaseHistogram.zero
            var lossMat = PhaseHistogram.zero
            for i in _recentEmittedGamesHead..<_recentEmittedGames.count {
                let rec = _recentEmittedGames[i]
                switch rec.result {
                case .checkmate(let winner):
                    if winner == .white {
                        recEmW += 1
                        winPly += rec.phaseByPly
                        winMat += rec.phaseByMaterial
                    } else {
                        recEmL += 1
                        lossPly += rec.phaseByPly
                        lossMat += rec.phaseByMaterial
                    }
                case .stalemate, .drawByFiftyMoveRule,
                     .drawByInsufficientMaterial, .drawByThreefoldRepetition:
                    recEmD += 1
                    drawPly += rec.phaseByPly
                    drawMat += rec.phaseByMaterial
                }
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
                emittedGames: _emittedGames,
                emittedPositions: _emittedPositions,
                totalGameWallMs: _totalGameWallMs,
                whiteCheckmates: _whiteCheckmates,
                blackCheckmates: _blackCheckmates,
                stalemates: _stalemates,
                fiftyMoveDraws: _fiftyMoveDraws,
                threefoldRepetitionDraws: _threefoldRepetitionDraws,
                insufficientMaterialDraws: _insufficientMaterialDraws,
                maxPliesDropped: _maxPliesDropped,
                emittedWhiteCheckmates: _emittedWhiteCheckmates,
                emittedBlackCheckmates: _emittedBlackCheckmates,
                emittedStalemates: _emittedStalemates,
                emittedFiftyMoveDraws: _emittedFiftyMoveDraws,
                emittedThreefoldRepetitionDraws: _emittedThreefoldRepetitionDraws,
                emittedInsufficientMaterialDraws: _emittedInsufficientMaterialDraws,
                trainingSteps: _trainingSteps,
                sessionStart: _sessionStart,
                recentGames: _recentGames.count - _recentGamesHead,
                recentMoves: recentMoves,
                recentGameWallMs: recentGameWallMs,
                recentEmittedGames: recentEmittedGameCount,
                recentEmittedPositions: recentEmittedPositionsValue,
                recentWhiteCheckmates: recW,
                recentBlackCheckmates: recL,
                recentDraws: recD,
                recentMaxPliesDropped: recDropped,
                recentEmittedWhiteCheckmates: recEmW,
                recentEmittedBlackCheckmates: recEmL,
                recentEmittedDraws: recEmD,
                emitWindowPhaseByPlyW: winPly,
                emitWindowPhaseByPlyD: drawPly,
                emitWindowPhaseByPlyL: lossPly,
                emitWindowPhaseByMaterialW: winMat,
                emitWindowPhaseByMaterialD: drawMat,
                emitWindowPhaseByMaterialL: lossMat,
                recentWindowSeconds: recentWindowSec,
                gameLenP50: p50,
                gameLenP95: p95
            )
        }
    }
}
