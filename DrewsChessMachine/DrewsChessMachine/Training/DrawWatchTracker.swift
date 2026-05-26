import Foundation
import os

/// One completed-game observation submitted to `DrawWatchTracker` by
/// the self-play driver at game-end. Carries everything the rolling
/// window needs to recompute the chart tile's bars and metrics on
/// each snapshot.
///
/// `firstFlagPlyIndex` is the 0-indexed game-total ply at which the
/// 8-ply streak first completed for this game — i.e. the 8th
/// consecutive ply meeting the pDraw threshold. `nil` when the game
/// never reached an 8-ply streak (sub-threshold games never enter the
/// histogram). `wasCapTerminated` is `true` iff the game ended
/// because it hit `selfPlayMaxPliesPerGame` rather than via a chess-
/// rule terminator — cap-terminated games are excluded from the
/// flag→draw precision denominator (locked decision #5 in
/// DRAW_WATCH_PLAN.md).
struct DrawWatchGameObservation: Sendable, Equatable {
    let timestamp: Date
    let plyCount: Int
    let firstFlagPlyIndex: UInt16?
    let wasCapTerminated: Bool
    /// Outcome encoding matches `ReplayBuffer.append`: `|outcome| < 0.5`
    /// ⇒ draw. We only ever distinguish draw vs not-draw downstream
    /// (the W-vs-L split isn't useful for the precision metric), but
    /// store the original Float so the rule stays in one place.
    let outcome: Float
}

/// Sendable, value-typed snapshot of `DrawWatchTracker`'s rolling-
/// window aggregates. Recomputed on every `snapshot()` call from the
/// current resident observations after pruning anything older than
/// `DrawWatchTracker.windowSec`.
struct DrawWatchSnapshot: Sendable, Equatable {
    /// Total completed games observed in the active window.
    let totalGames: Int
    /// Subset of `totalGames` whose `firstFlagPlyIndex != nil` —
    /// i.e. the game raised a flag (8-ply streak completed) at least
    /// once.
    let flaggedGames: Int
    /// Subset of `flaggedGames` whose final outcome counts toward the
    /// precision ratio (i.e. `!wasCapTerminated`). Snapshot denominator
    /// for `flagDrawAccuracy` and per-bucket precision; reported
    /// separately so consumers can show the selection-effect.
    let flaggedGamesEligible: Int
    /// Subset of `flaggedGamesEligible` whose final outcome was a
    /// draw (`|outcome| < 0.5`).
    let flaggedGamesDrawn: Int
    /// Per-bucket flagged-game count. Bucket `i` covers
    /// `[i * bucketWidth, (i+1) * bucketWidth)` plies; the last
    /// bucket clamps anything `>= histogramBucketCount * bucketWidth`.
    /// Length equals `DrawWatchTracker.histogramBucketCount`.
    let gamesFlaggedByBucket: [Int]
    /// Per-bucket eligible-game count (the precision denominator for
    /// that bucket).
    let gamesEligibleByBucket: [Int]
    /// Per-bucket drawn-game count. `gamesEligibleByBucket - this` is
    /// the bucket's decisive-game count.
    let gamesDrawnByBucket: [Int]
    /// Length of the active rolling window, in seconds. Surfaced so
    /// the chart header can label "(last 30 min)" without hard-coding
    /// the value at the call site.
    let windowSec: Double

    /// `flaggedGames / totalGames` as a fraction in `[0, 1]`. `nil`
    /// when the window holds no completed games yet.
    var fractionOfGamesFlagged: Double? {
        guard totalGames > 0 else { return nil }
        return Double(flaggedGames) / Double(totalGames)
    }

    /// Of the games we flagged (excluding cap-terminated), the
    /// fraction that actually ended in a draw — the v1 calibration
    /// metric. `nil` until at least one eligible flagged game has
    /// completed inside the window.
    var flagDrawAccuracy: Double? {
        guard flaggedGamesEligible > 0 else { return nil }
        return Double(flaggedGamesDrawn) / Double(flaggedGamesEligible)
    }

    /// Per-bucket draw-precision fraction. `nil` for buckets whose
    /// eligible count is zero (rendered as "--" by the chart). Other
    /// buckets return `gamesDrawnByBucket[i] / gamesEligibleByBucket[i]`.
    func drawAccuracyForBucket(_ i: Int) -> Double? {
        guard gamesEligibleByBucket.indices.contains(i),
              gamesEligibleByBucket[i] > 0 else { return nil }
        return Double(gamesDrawnByBucket[i]) / Double(gamesEligibleByBucket[i])
    }
}

/// Rolling-window aggregator of self-play game-end observations for
/// the stealth-mode draw-watch monitor. Designed to be hit from
/// off-main worker threads — each `BatchedSelfPlayDriver` task calls
/// `recordGameCompleted` at game-end while the heartbeat reads via
/// `asyncSnapshot()`. Internal `OSAllocatedUnfairLock` matches the
/// project's `ParallelWorkerStatsBox` / `GameDiversityTracker`
/// convention. The snapshot performs the prune + per-bucket
/// aggregation on each read; the per-game ring stays small enough
/// (max ~110k games/hr × 30-min window ≈ 55k entries at peak self-
/// play rates) that the O(N) walk is negligible against the
/// heartbeat's 5-second cadence.
///
/// **Stealth-mode v1 — no game termination.** Recording an
/// observation does NOT signal the worker to end the game. Both the
/// observation submission and the in-flight per-game streak counter
/// (on `ActiveGame`) are purely diagnostic; the game plays out as
/// normal and its eventual outcome is reported here at game-end.
/// See `DRAW_WATCH_PLAN.md`.
final class DrawWatchTracker: @unchecked Sendable {

    // MARK: - Public constants (locked design)

    /// Default pDraw threshold a single ply must clear to count
    /// toward the running streak — overridable per-tick by reading
    /// `TrainingParameters.shared.drawWatchPDrawThreshold` (which
    /// `BatchedSelfPlayDriver` does at the top of each tick). This
    /// constant is the value the parameter ships at and the value
    /// the tests pin against.
    static let defaultFlagThresholdPDraw: Float = 0.95

    /// Number of consecutive plies above the threshold required to
    /// raise one flag.
    static let flagStreakLength: Int = 8

    /// Fixed bucket width (in plies) for the histogram. 40 plies per
    /// bucket; the last bucket clamps anything past
    /// `histogramBucketCount * 40` so a future raise of
    /// `selfPlayMaxPliesPerGame` doesn't silently drop overflow.
    static let histogramBucketWidthPlies: Int = 40

    /// Number of buckets in `gamesFlaggedByBucket`. Covers
    /// `0..<(width × count)` plies natively; values past clamp into
    /// the last bucket. 10 buckets × 40 plies = 0–400 plies of
    /// natural coverage with headroom for a longer `maxPliesPerGame`
    /// down the road.
    static let histogramBucketCount: Int = 10

    /// Length of the rolling window the snapshot covers, in seconds.
    /// 30 minutes is the v1 default — long enough to smooth tick-by-
    /// tick noise across self-play barrier dynamics, short enough to
    /// see the impact of a parameter edit within a few snapshots.
    static let windowSec: TimeInterval = 30 * 60

    // MARK: - Locked state

    private let lock = OSAllocatedUnfairLock()
    /// Ring of completed-game observations within the rolling window.
    /// `_recentGamesHead` walks forward as the prune pass drops
    /// out-of-window observations; the underlying Array is compacted
    /// every so often so it doesn't grow without bound between
    /// snapshots.
    private var _recentGames: [DrawWatchGameObservation] = []
    private var _recentGamesHead: Int = 0

    // MARK: - Init

    init() {}

    // MARK: - Mutating API (called from worker threads)

    /// Submit a completed game's observability data. `firstFlagPlyIndex`
    /// is the ply at which the game's 8-ply streak first completed
    /// (i.e. the 8th consecutive above-threshold ply); `nil` when
    /// the game never reached an 8-ply streak. `wasCapTerminated` is
    /// `true` iff the game ended at the ply cap; per-plan-decision
    /// #5 those games are counted in the histogram but excluded from
    /// the flag→draw precision denominator. `outcome` follows
    /// `ReplayBuffer.append`'s convention (`|x| < 0.5` ⇒ draw).
    func recordGameCompleted(
        plyCount: Int,
        firstFlagPlyIndex: UInt16?,
        wasCapTerminated: Bool,
        outcome: Float
    ) {
        let obs = DrawWatchGameObservation(
            timestamp: Date(),
            plyCount: plyCount,
            firstFlagPlyIndex: firstFlagPlyIndex,
            wasCapTerminated: wasCapTerminated,
            outcome: outcome
        )
        lock.withLock { _recentGames.append(obs) }
    }

    // MARK: - Read API

    /// Synchronous snapshot. Prunes out-of-window observations,
    /// recomputes per-bucket counters, and returns a Sendable struct.
    /// O(window size) per call — at ~55k peak resident this is sub-
    /// millisecond, well below the heartbeat cadence.
    func snapshot() -> DrawWatchSnapshot {
        lock.withLock {
            let now = Date()
            pruneRecentLocked(now: now)

            let bucketCount = Self.histogramBucketCount
            let bucketWidth = Self.histogramBucketWidthPlies
            var totalGames = 0
            var flaggedGames = 0
            var eligibleFlaggedGames = 0
            var drawnFlaggedGames = 0
            var flaggedByBucket = [Int](repeating: 0, count: bucketCount)
            var eligibleByBucket = [Int](repeating: 0, count: bucketCount)
            var drawnByBucket = [Int](repeating: 0, count: bucketCount)
            for i in _recentGamesHead..<_recentGames.count {
                let g = _recentGames[i]
                totalGames += 1
                guard let ply = g.firstFlagPlyIndex else { continue }
                flaggedGames += 1
                let bucket = min(Int(ply) / bucketWidth, bucketCount - 1)
                flaggedByBucket[bucket] += 1
                guard !g.wasCapTerminated else { continue }
                eligibleFlaggedGames += 1
                eligibleByBucket[bucket] += 1
                if abs(g.outcome) < 0.5 {
                    drawnFlaggedGames += 1
                    drawnByBucket[bucket] += 1
                }
            }
            return DrawWatchSnapshot(
                totalGames: totalGames,
                flaggedGames: flaggedGames,
                flaggedGamesEligible: eligibleFlaggedGames,
                flaggedGamesDrawn: drawnFlaggedGames,
                gamesFlaggedByBucket: flaggedByBucket,
                gamesEligibleByBucket: eligibleByBucket,
                gamesDrawnByBucket: drawnByBucket,
                windowSec: Self.windowSec
            )
        }
    }

    /// Off-main snapshot for the UI heartbeat — hops via
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
            _recentGames.removeAll(keepingCapacity: true)
            _recentGamesHead = 0
        }
    }

    // MARK: - Private

    /// Advance `_recentGamesHead` past every observation older than
    /// `now - windowSec`. Underlying Array is compacted when the head
    /// has walked past more than half its length to keep memory in
    /// check for a multi-hour session.
    private func pruneRecentLocked(now: Date) {
        let cutoff = now.addingTimeInterval(-Self.windowSec)
        while _recentGamesHead < _recentGames.count, _recentGames[_recentGamesHead].timestamp < cutoff {
            _recentGamesHead += 1
        }
        if _recentGamesHead > 0 && _recentGamesHead * 2 >= _recentGames.count {
            _recentGames.removeFirst(_recentGamesHead)
            _recentGamesHead = 0
        }
    }
}
