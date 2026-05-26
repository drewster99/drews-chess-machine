import XCTest
@testable import DrewsChessMachine

/// Unit tests for `DrawWatchTracker`. Covers the streak / flag / draw-
/// precision contract documented in `DRAW_WATCH_PLAN.md` — the
/// concurrency/threading model is exercised at runtime by the
/// `BatchedSelfPlayDriver` integration; here we pin the pure logic.
final class DrawWatchTrackerTests: XCTestCase {

    // MARK: - Helpers

    /// Drive a synthetic per-game pDraw series through a tracker the
    /// same way `BatchedSelfPlayDriver`'s consume closure would: for
    /// each ply, check threshold + streak length, fire on the 7→8
    /// transition (once per streak), record streak-end when the
    /// series drops back below threshold. Returns the per-game state
    /// the driver would carry (sticky `everFired`, current streak).
    @discardableResult
    private func simulateGame(
        tracker: DrawWatchTracker,
        workerId: UInt16 = 1,
        gameIndex: UInt32 = 1,
        pDrawSeries: [Float]
    ) -> (everFired: Bool, finalStreak: Int) {
        var streak = 0
        var firedThisStreak = false
        var everFired = false
        let threshold = DrawWatchTracker.flagThresholdPDraw
        let triggerLen = DrawWatchTracker.flagStreakLength
        for (idx, pDraw) in pDrawSeries.enumerated() {
            if pDraw >= threshold {
                streak += 1
                if streak == triggerLen && !firedThisStreak {
                    firedThisStreak = true
                    everFired = true
                    let plyIdx = UInt16(idx)
                    let streakStart = plyIdx &- UInt16(triggerLen - 1)
                    tracker.recordFlag(DrawWatchFlag(
                        workerId: workerId,
                        intraWorkerGameIndex: gameIndex,
                        plyIndex: plyIdx,
                        streakStartPly: streakStart,
                        pDrawAtFire: pDraw,
                        timestamp: Date()
                    ))
                }
            } else if streak > 0 {
                tracker.recordStreakEnded(streakLength: streak)
                streak = 0
                firedThisStreak = false
            }
        }
        return (everFired, streak)
    }

    // MARK: - Streak / flag math

    func testStreakFiresAtEight() {
        let tracker = DrawWatchTracker()
        let (everFired, _) = simulateGame(
            tracker: tracker,
            pDrawSeries: [Float](repeating: 0.99, count: 8)
        )
        XCTAssertTrue(everFired)
        let snap = tracker.snapshot()
        XCTAssertEqual(snap.flags.count, 1)
        XCTAssertEqual(snap.flags.first?.plyIndex, 7)        // 0-indexed: 8th ply has index 7
        XCTAssertEqual(snap.flags.first?.streakStartPly, 0)
        XCTAssertEqual(snap.flags.first?.pDrawAtFire, 0.99)
    }

    func testNoFireAtSevenConsecutive() {
        let tracker = DrawWatchTracker()
        simulateGame(tracker: tracker, pDrawSeries: [Float](repeating: 0.99, count: 7))
        XCTAssertEqual(tracker.snapshot().flags.count, 0)
    }

    func testStreakResetOnDip() {
        let tracker = DrawWatchTracker()
        var series = [Float](repeating: 0.99, count: 7)
        series.append(0.5)
        series.append(contentsOf: [Float](repeating: 0.99, count: 8))
        let (everFired, _) = simulateGame(tracker: tracker, pDrawSeries: series)
        XCTAssertTrue(everFired)
        // The dip resets the streak; only the second 8-ply run fires.
        XCTAssertEqual(tracker.snapshot().flags.count, 1)
    }

    func testNoRefireWithinSameStreak() {
        let tracker = DrawWatchTracker()
        simulateGame(tracker: tracker, pDrawSeries: [Float](repeating: 0.99, count: 20))
        XCTAssertEqual(tracker.snapshot().flags.count, 1)
    }

    func testSecondFlagAfterStreakBreaks() {
        let tracker = DrawWatchTracker()
        var series = [Float](repeating: 0.99, count: 8)
        series.append(0.5)
        series.append(contentsOf: [Float](repeating: 0.99, count: 8))
        simulateGame(tracker: tracker, pDrawSeries: series)
        XCTAssertEqual(tracker.snapshot().flags.count, 2)
    }

    func testJustBelowThresholdDoesNotCount() {
        let tracker = DrawWatchTracker()
        // 0.949 < 0.95
        simulateGame(tracker: tracker, pDrawSeries: [Float](repeating: 0.949, count: 100))
        XCTAssertEqual(tracker.snapshot().flags.count, 0)
    }

    // MARK: - Per-game completion + precision math

    func testRecordGameCompletedCountersAdvance() {
        let tracker = DrawWatchTracker()
        for _ in 0..<10 {
            tracker.recordGameCompleted(
                plyCount: 100, wasFlagged: false, wasCapTerminated: false, outcome: 0.0
            )
        }
        let snap = tracker.snapshot()
        XCTAssertEqual(snap.totalGamesObserved, 10)
        XCTAssertEqual(snap.totalPliesObserved, 1000)
        XCTAssertEqual(snap.flaggedGamesObserved, 0)
        XCTAssertNil(snap.flagDrawAccuracy)
    }

    func testFlagDrawAccuracyExcludesCapTerminatedGames() {
        let tracker = DrawWatchTracker()
        // 4 flagged games — 1 ended as draw, 1 decisive, 2 cap-terminated
        // → eligible = 2, drew = 1, precision = 0.5
        tracker.recordGameCompleted(plyCount: 50, wasFlagged: true, wasCapTerminated: false, outcome: 0.0)
        tracker.recordGameCompleted(plyCount: 60, wasFlagged: true, wasCapTerminated: false, outcome: 1.0)
        tracker.recordGameCompleted(plyCount: 150, wasFlagged: true, wasCapTerminated: true, outcome: 0.0)
        tracker.recordGameCompleted(plyCount: 150, wasFlagged: true, wasCapTerminated: true, outcome: 1.0)
        let snap = tracker.snapshot()
        XCTAssertEqual(snap.flaggedGamesObserved, 4)
        XCTAssertEqual(snap.flaggedGamesEligibleForPrecision, 2)
        XCTAssertEqual(snap.flaggedGamesEndedInDraw, 1)
        XCTAssertEqual(snap.flaggedGamesEndedInDecisive, 1)
        XCTAssertEqual(snap.flagDrawAccuracy, 0.5)
    }

    func testFlagDrawAccuracyDrawIsBelowAbsThreshold() {
        // The plan specifies the same `|outcome| < 0.5` draw test
        // `ReplayBuffer.append` uses. A value of +0.4 should classify
        // as a draw, +0.6 as decisive.
        let tracker = DrawWatchTracker()
        tracker.recordGameCompleted(plyCount: 50, wasFlagged: true, wasCapTerminated: false, outcome: 0.4)
        tracker.recordGameCompleted(plyCount: 50, wasFlagged: true, wasCapTerminated: false, outcome: 0.6)
        let snap = tracker.snapshot()
        XCTAssertEqual(snap.flaggedGamesEndedInDraw, 1)
        XCTAssertEqual(snap.flaggedGamesEndedInDecisive, 1)
    }

    func testFlagDrawAccuracyNilUntilFirstEligibleGame() {
        let tracker = DrawWatchTracker()
        tracker.recordGameCompleted(plyCount: 50, wasFlagged: false, wasCapTerminated: false, outcome: 0.0)
        tracker.recordGameCompleted(plyCount: 150, wasFlagged: true, wasCapTerminated: true, outcome: 0.0)
        XCTAssertNil(tracker.snapshot().flagDrawAccuracy)
    }

    // MARK: - Streak-length aggregation

    func testStreakLengthAccumulatesAtOrAboveTrigger() {
        let tracker = DrawWatchTracker()
        tracker.recordStreakEnded(streakLength: 8)
        tracker.recordStreakEnded(streakLength: 15)
        tracker.recordStreakEnded(streakLength: 100)
        let snap = tracker.snapshot()
        XCTAssertEqual(snap.totalPliesInFlaggedStreaks, 8 + 15 + 100)
    }

    func testStreakLengthIgnoredBelowTrigger() {
        let tracker = DrawWatchTracker()
        tracker.recordStreakEnded(streakLength: 1)
        tracker.recordStreakEnded(streakLength: 7)
        XCTAssertEqual(tracker.snapshot().totalPliesInFlaggedStreaks, 0)
    }

    // MARK: - Histogram bucketing

    func testHistogramBucketingFixedTwentyPlies() {
        let tracker = DrawWatchTracker()
        // Fire flags at carefully-chosen ply indices
        let plies: [Int] = [
            0, 5, 19,   // bucket 0 (0-19)
            20, 39,     // bucket 1 (20-39)
            100, 119,   // bucket 5 (100-119)
            200, 999    // overflow into last bucket (9, since 200/20==10 clamps to 9)
        ]
        for (i, p) in plies.enumerated() {
            tracker.recordFlag(DrawWatchFlag(
                workerId: 1,
                intraWorkerGameIndex: UInt32(i + 1),
                plyIndex: UInt16(min(p, Int(UInt16.max))),
                streakStartPly: UInt16(min(max(p - 7, 0), Int(UInt16.max))),
                pDrawAtFire: 0.99,
                timestamp: Date()
            ))
        }
        let hist = tracker.snapshot().plyBucketHistogram
        XCTAssertEqual(hist.count, DrawWatchTracker.histogramBucketCount)
        XCTAssertEqual(hist[0], 3)
        XCTAssertEqual(hist[1], 2)
        XCTAssertEqual(hist[5], 2)
        XCTAssertEqual(hist[hist.count - 1], 2)
    }

    // MARK: - Ring bounding

    func testFlagRingBoundedAtCap() {
        let tracker = DrawWatchTracker()
        let n = DrawWatchTracker.maxRetainedFlags + 250
        for i in 0..<n {
            tracker.recordFlag(DrawWatchFlag(
                workerId: 1,
                intraWorkerGameIndex: UInt32(i + 1),
                plyIndex: 10,
                streakStartPly: 3,
                pDrawAtFire: 0.99,
                timestamp: Date()
            ))
        }
        let snap = tracker.snapshot()
        XCTAssertLessThanOrEqual(snap.flags.count, DrawWatchTracker.maxRetainedFlags)
        // Histogram is independent of the ring trim — should reflect
        // ALL events ever submitted this session.
        let bucket = 10 / DrawWatchTracker.histogramBucketWidthPlies
        XCTAssertEqual(snap.plyBucketHistogram[bucket], n)
    }

    // MARK: - Reset

    func testResetClearsEverything() {
        let tracker = DrawWatchTracker()
        tracker.recordFlag(DrawWatchFlag(
            workerId: 1, intraWorkerGameIndex: 1, plyIndex: 8,
            streakStartPly: 1, pDrawAtFire: 0.99, timestamp: Date()
        ))
        tracker.recordStreakEnded(streakLength: 8)
        tracker.recordGameCompleted(plyCount: 50, wasFlagged: true, wasCapTerminated: false, outcome: 0.0)
        tracker.reset()
        let snap = tracker.snapshot()
        XCTAssertEqual(snap.flags.count, 0)
        XCTAssertEqual(snap.totalGamesObserved, 0)
        XCTAssertEqual(snap.totalPliesObserved, 0)
        XCTAssertEqual(snap.totalPliesInFlaggedStreaks, 0)
        XCTAssertEqual(snap.flaggedGamesObserved, 0)
        XCTAssertEqual(snap.flaggedGamesEligibleForPrecision, 0)
        XCTAssertEqual(snap.flaggedGamesEndedInDraw, 0)
        XCTAssertEqual(snap.flaggedGamesEndedInDecisive, 0)
        XCTAssertEqual(snap.plyBucketHistogram, [Int](repeating: 0, count: DrawWatchTracker.histogramBucketCount))
        XCTAssertNil(snap.flagDrawAccuracy)
    }
}
