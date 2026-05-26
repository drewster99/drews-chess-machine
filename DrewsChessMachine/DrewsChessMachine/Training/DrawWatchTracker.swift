import Foundation
import os

/// One flag-fire event: emitted when a self-play game's W/D/L head
/// reports `pDraw ≥ DrawWatchTracker.flagThresholdPDraw` for
/// `DrawWatchTracker.flagStreakLength` consecutive plies. **Observation
/// only — does not influence the game in any way; the game continues
/// playing to its natural conclusion.**
///
/// `plyIndex` is the 0-indexed half-move count at which the streak
/// reached its trigger length (i.e. the 8th ply of the streak when
/// `flagStreakLength == 8`). `streakStartPly = plyIndex - (flagStreakLength - 1)`.
struct DrawWatchFlag: Sendable, Equatable {
    let workerId: UInt16
    let intraWorkerGameIndex: UInt32
    let plyIndex: UInt16
    let streakStartPly: UInt16
    let pDrawAtFire: Float
    let timestamp: Date
}

/// Sendable, value-typed snapshot of `DrawWatchTracker`'s state for the
/// UI heartbeat to mirror into `@State`. Captures both raw counters and
/// derived metrics so consumers don't have to recompute.
struct DrawWatchSnapshot: Sendable, Equatable {
    let flags: [DrawWatchFlag]
    let totalGamesObserved: Int
    let totalPliesObserved: Int
    let totalPliesInFlaggedStreaks: Int
    let flaggedGamesObserved: Int
    /// Subset of `flaggedGamesObserved` that did NOT hit the ply cap
    /// (cap-terminated games are excluded from precision math — see
    /// `DRAW_WATCH_PLAN.md` locked decision #5).
    let flaggedGamesEligibleForPrecision: Int
    let flaggedGamesEndedInDraw: Int
    let flaggedGamesEndedInDecisive: Int
    /// Histogram of flag-fire `plyIndex` values, bucketed into fixed
    /// 20-ply buckets `[0,20), [20,40), ..., [(B-1)*20, B*20)`. Length
    /// equals `DrawWatchTracker.histogramBucketCount`. Entries past the
    /// last bucket's right edge are clamped into the last bucket so a
    /// future `selfPlayMaxPliesPerGame` raise doesn't silently drop them.
    let plyBucketHistogram: [Int]

    /// `flaggedGames / totalGames` as a fraction in `[0, 1]`. `nil` when
    /// no games have completed yet.
    var fractionOfGamesFlagged: Double? {
        guard totalGamesObserved > 0 else { return nil }
        return Double(flaggedGamesObserved) / Double(totalGamesObserved)
    }

    /// `pliesInFlaggedStreaks / totalPlies` as a fraction in `[0, 1]`.
    /// `nil` when no plies have been observed yet.
    var fractionOfPliesInFlaggedStreaks: Double? {
        guard totalPliesObserved > 0 else { return nil }
        return Double(totalPliesInFlaggedStreaks) / Double(totalPliesObserved)
    }

    /// "Of the games we flagged (excluding cap-terminated), what
    /// fraction actually ended in a draw?" — the v1 calibration metric.
    /// `nil` until at least one non-cap flagged game has completed.
    var flagDrawAccuracy: Double? {
        guard flaggedGamesEligibleForPrecision > 0 else { return nil }
        return Double(flaggedGamesEndedInDraw) / Double(flaggedGamesEligibleForPrecision)
    }
}

/// Session-wide aggregator of `DrawWatchFlag` events emitted by
/// self-play workers, plus the per-game completion observations needed
/// to compute the flag's draw-precision metric.
///
/// Designed to be hit from off-main worker threads — each
/// `BatchedSelfPlayDriver` task may call `recordFlag`, `recordStreakEnded`,
/// and `recordGameCompleted` while the heartbeat reads via the off-main
/// `asyncSnapshot()`. State is protected by an internal
/// `OSAllocatedUnfairLock`, matching the project's
/// `ParallelWorkerStatsBox` / `GameDiversityTracker` convention.
///
/// **Stealth-mode v1 — no game termination.** Recording a flag does
/// NOT signal the worker to end the game; the game plays out as normal
/// and its eventual outcome is later fed back via `recordGameCompleted`.
/// See `DRAW_WATCH_PLAN.md` for the full design.
final class DrawWatchTracker: @unchecked Sendable {

    // MARK: - Public constants (locked design)

    /// pDraw threshold a single ply must clear to count toward the
    /// running streak.
    static let flagThresholdPDraw: Float = 0.95

    /// Number of consecutive plies above the threshold required to
    /// raise one flag.
    static let flagStreakLength: Int = 8

    /// Fixed bucket width (in plies) for the histogram. 20 plies per
    /// bucket survives any future raise of `selfPlayMaxPliesPerGame`
    /// without re-bucketing logic (the last bucket clamps overflow).
    static let histogramBucketWidthPlies: Int = 20

    /// Number of buckets in `plyBucketHistogram`. Covers `0..<(width × count)`
    /// plies natively; values at or past `width × count` clamp into the
    /// last bucket. Sized at 10 buckets (0–200 plies) — comfortably
    /// covers the current 150-ply self-play cap with headroom for a
    /// future raise.
    static let histogramBucketCount: Int = 10

    /// Per-session cap on retained flag history. Older flags drop
    /// oldest-first on overflow so a many-hour session doesn't grow
    /// the snapshot's `flags` array unboundedly.
    static let maxRetainedFlags: Int = 10_000

    // MARK: - Locked state

    private let lock = OSAllocatedUnfairLock()
    private var _flags: [DrawWatchFlag] = []
    private var _flagsHead: Int = 0   // ring head for oldest-drop overflow
    private var _totalGamesObserved: Int = 0
    private var _totalPliesObserved: Int = 0
    private var _totalPliesInFlaggedStreaks: Int = 0
    private var _flaggedGamesObserved: Int = 0
    private var _flaggedGamesEligibleForPrecision: Int = 0
    private var _flaggedGamesEndedInDraw: Int = 0
    private var _flaggedGamesEndedInDecisive: Int = 0
    private var _plyBucketHistogram: [Int]

    // MARK: - Init

    init() {
        _plyBucketHistogram = [Int](repeating: 0, count: Self.histogramBucketCount)
    }

    // MARK: - Mutating API (called from worker threads)

    /// Append a flag-fire event. If the ring is at `maxRetainedFlags`
    /// capacity, the oldest flag is dropped first so length stays
    /// bounded. Bucket histogram is updated unconditionally — dropping
    /// a flag from the ring does NOT decrement its bucket; the
    /// histogram reflects all flags ever observed this session, not
    /// just the resident ring.
    func recordFlag(_ flag: DrawWatchFlag) {
        lock.withLock {
            // Bucket update (independent of ring trim — see doc).
            let raw = Int(flag.plyIndex) / Self.histogramBucketWidthPlies
            let bucket = min(raw, Self.histogramBucketCount - 1)
            _plyBucketHistogram[bucket] += 1

            // Ring append + oldest-drop overflow.
            _flags.append(flag)
            if _flags.count - _flagsHead > Self.maxRetainedFlags {
                _flagsHead += 1
                // Compact occasionally so the underlying Array doesn't
                // grow without bound while head walks forward.
                if _flagsHead > Self.maxRetainedFlags {
                    _flags.removeFirst(_flagsHead)
                    _flagsHead = 0
                }
            }
        }
    }

    /// Report that a streak (which already triggered a flag, or never
    /// did) ended at length `streakLength`. The caller passes the FULL
    /// streak length, not the threshold. Bumps
    /// `totalPliesInFlaggedStreaks` only when `streakLength >= flagStreakLength`
    /// — sub-threshold streaks are not "flagged streaks" and do not
    /// contribute to the "% of plies in flagged streaks" metric.
    ///
    /// Called by the worker whenever:
    ///   * an above-threshold streak drops below threshold mid-game, OR
    ///   * the game ends while a streak is still active (call with the
    ///     streak length at that moment).
    func recordStreakEnded(streakLength: Int) {
        guard streakLength >= Self.flagStreakLength else { return }
        lock.withLock { _totalPliesInFlaggedStreaks += streakLength }
    }

    /// Submit a completed game's observability data. `wasFlagged` is
    /// `true` iff the game raised at least one flag during play (sticky
    /// across multiple streaks within the same game). `wasCapTerminated`
    /// is `true` iff the game ended because it hit the ply cap (those
    /// games are excluded from the flag-precision ratio — see
    /// `DRAW_WATCH_PLAN.md` locked decision #5). `outcome` is the same
    /// convention as `ReplayBuffer.append`: `+1` win / `0` draw / `-1`
    /// loss; the project's `|outcome| < 0.5` test classifies draws.
    func recordGameCompleted(
        plyCount: Int,
        wasFlagged: Bool,
        wasCapTerminated: Bool,
        outcome: Float
    ) {
        lock.withLock {
            _totalGamesObserved += 1
            _totalPliesObserved += plyCount
            guard wasFlagged else { return }
            _flaggedGamesObserved += 1
            guard !wasCapTerminated else { return }
            _flaggedGamesEligibleForPrecision += 1
            if abs(outcome) < 0.5 {
                _flaggedGamesEndedInDraw += 1
            } else {
                _flaggedGamesEndedInDecisive += 1
            }
        }
    }

    // MARK: - Read API

    /// Synchronous snapshot. Cheap (one lock + array copy) — safe to
    /// call from anywhere off-main. Use `asyncSnapshot()` from the
    /// main actor to avoid blocking it.
    func snapshot() -> DrawWatchSnapshot {
        lock.withLock {
            DrawWatchSnapshot(
                flags: Array(_flags[_flagsHead..<_flags.count]),
                totalGamesObserved: _totalGamesObserved,
                totalPliesObserved: _totalPliesObserved,
                totalPliesInFlaggedStreaks: _totalPliesInFlaggedStreaks,
                flaggedGamesObserved: _flaggedGamesObserved,
                flaggedGamesEligibleForPrecision: _flaggedGamesEligibleForPrecision,
                flaggedGamesEndedInDraw: _flaggedGamesEndedInDraw,
                flaggedGamesEndedInDecisive: _flaggedGamesEndedInDecisive,
                plyBucketHistogram: _plyBucketHistogram
            )
        }
    }

    /// Off-main snapshot for the UI heartbeat. Hops via
    /// `DispatchQueue.global` + `CheckedContinuation` so the main
    /// actor never waits synchronously on `lock` while a worker
    /// holds it. Mirrors `ParallelWorkerStatsBox.asyncSnapshot()` /
    /// `ReplayBuffer.asyncCompositionSnapshot()`.
    func asyncSnapshot() async -> DrawWatchSnapshot {
        let start = Date()
        return await withCheckedContinuation { (cont: CheckedContinuation<DrawWatchSnapshot, Never>) in
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
                    print(String(format: "[DISPATCH-LATENCY] DrawWatchTracker.asyncSnapshot: total=%.2fms (pre=%.2fms queue=%.2fms work=%.2fms)", total * 1000, pre * 1000, queue * 1000, work * 1000))
                }
                cont.resume(returning: result)
            }
        }
    }

    /// Drop all accumulated state. Called only at session start.
    func reset() {
        lock.withLock {
            _flags.removeAll(keepingCapacity: true)
            _flagsHead = 0
            _totalGamesObserved = 0
            _totalPliesObserved = 0
            _totalPliesInFlaggedStreaks = 0
            _flaggedGamesObserved = 0
            _flaggedGamesEligibleForPrecision = 0
            _flaggedGamesEndedInDraw = 0
            _flaggedGamesEndedInDecisive = 0
            for i in _plyBucketHistogram.indices { _plyBucketHistogram[i] = 0 }
        }
    }
}
