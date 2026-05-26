import XCTest
@testable import DrewsChessMachine

/// Unit tests for `DrawWatchTracker`'s rolling-window aggregator. Pins
/// the per-game observation contract documented in `DRAW_WATCH_PLAN.md`
/// — the per-ply streak math itself lives on `ActiveGame` / the
/// driver and is exercised at runtime via `BatchedSelfPlayDriver`.
final class DrawWatchTrackerTests: XCTestCase {

    // MARK: - Empty / no-data behavior

    func testEmptySnapshotIsAllZero() {
        let tracker = DrawWatchTracker()
        let snap = tracker.snapshot()
        XCTAssertEqual(snap.totalGames, 0)
        XCTAssertEqual(snap.flaggedGames, 0)
        XCTAssertEqual(snap.flaggedGamesEligible, 0)
        XCTAssertEqual(snap.flaggedGamesDrawn, 0)
        XCTAssertEqual(snap.gamesFlaggedByBucket.count, DrawWatchTracker.histogramBucketCount)
        XCTAssertEqual(snap.gamesFlaggedByBucket, Array(repeating: 0, count: DrawWatchTracker.histogramBucketCount))
        XCTAssertNil(snap.fractionOfGamesFlagged)
        XCTAssertNil(snap.flagDrawAccuracy)
    }

    func testTotalGamesIncrementsForEveryCompletion() {
        let tracker = DrawWatchTracker()
        for _ in 0..<7 {
            tracker.recordGameCompleted(
                plyCount: 50, firstFlagPlyIndex: nil, excludedFromPrecision: false, outcome: 1.0
            )
        }
        XCTAssertEqual(tracker.snapshot().totalGames, 7)
        XCTAssertEqual(tracker.snapshot().flaggedGames, 0)
    }

    // MARK: - Flagged / draw / cap-terminated math

    func testFlaggedGamesAndDrawPrecision() {
        // 5 games:
        //  - 2 flagged + drew + not-cap
        //  - 1 flagged + decisive + not-cap
        //  - 1 flagged + cap-terminated (excluded from precision)
        //  - 1 not-flagged
        // Expected: totalGames=5, flaggedGames=4, eligible=3, drawn=2,
        // precision = 2/3 ≈ 0.667
        let tracker = DrawWatchTracker()
        tracker.recordGameCompleted(plyCount: 60, firstFlagPlyIndex: 25, excludedFromPrecision: false, outcome: 0.0)
        tracker.recordGameCompleted(plyCount: 70, firstFlagPlyIndex: 35, excludedFromPrecision: false, outcome: 0.0)
        tracker.recordGameCompleted(plyCount: 80, firstFlagPlyIndex: 45, excludedFromPrecision: false, outcome: 1.0)
        tracker.recordGameCompleted(plyCount: 150, firstFlagPlyIndex: 90, excludedFromPrecision: true, outcome: 0.0)
        tracker.recordGameCompleted(plyCount: 30, firstFlagPlyIndex: nil, excludedFromPrecision: false, outcome: 1.0)
        let s = tracker.snapshot()
        XCTAssertEqual(s.totalGames, 5)
        XCTAssertEqual(s.flaggedGames, 4)
        XCTAssertEqual(s.flaggedGamesEligible, 3)
        XCTAssertEqual(s.flaggedGamesDrawn, 2)
        XCTAssertEqual(s.flagDrawAccuracy ?? -1, 2.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(s.fractionOfGamesFlagged ?? -1, 4.0 / 5.0, accuracy: 1e-9)
    }

    func testDrawClassifierMatchesReplayBufferConvention() {
        // `|outcome| < 0.5` ⇒ draw — same as `ReplayBuffer.append`.
        // +0.4 is a draw, +0.6 is decisive; the boundary value 0.5
        // itself is decisive (the test in `recordGameCompleted` is
        // strict `<`, mirroring the replay buffer's `> 0.5` /
        // `< -0.5` win/loss tests).
        let tracker = DrawWatchTracker()
        tracker.recordGameCompleted(plyCount: 60, firstFlagPlyIndex: 25, excludedFromPrecision: false, outcome: 0.4)
        tracker.recordGameCompleted(plyCount: 60, firstFlagPlyIndex: 25, excludedFromPrecision: false, outcome: 0.6)
        tracker.recordGameCompleted(plyCount: 60, firstFlagPlyIndex: 25, excludedFromPrecision: false, outcome: 0.5)
        let s = tracker.snapshot()
        XCTAssertEqual(s.flaggedGamesDrawn, 1)
    }

    func testCapTerminatedExcludedEvenWhenDrawn() {
        let tracker = DrawWatchTracker()
        tracker.recordGameCompleted(plyCount: 150, firstFlagPlyIndex: 100, excludedFromPrecision: true, outcome: 0.0)
        let s = tracker.snapshot()
        XCTAssertEqual(s.flaggedGames, 1)
        XCTAssertEqual(s.flaggedGamesEligible, 0)
        XCTAssertEqual(s.flaggedGamesDrawn, 0)
        XCTAssertNil(s.flagDrawAccuracy)
    }

    func testNilFirstFlagMeansNotFlagged() {
        // Even a draw outcome doesn't count as flagged if
        // firstFlagPlyIndex is nil.
        let tracker = DrawWatchTracker()
        tracker.recordGameCompleted(plyCount: 80, firstFlagPlyIndex: nil, excludedFromPrecision: false, outcome: 0.0)
        let s = tracker.snapshot()
        XCTAssertEqual(s.totalGames, 1)
        XCTAssertEqual(s.flaggedGames, 0)
    }

    // MARK: - Per-bucket bucketing + precision

    func testBucketingFixedFortyPlies() {
        // Buckets: 0-40, 40-80, 80-120, 120-160, 160-200, 200-240,
        //          240-280, 280-320, 320-360, 360+
        let tracker = DrawWatchTracker()
        let pliesAndOutcomes: [(UInt16, Float, Bool)] = [
            (0,   0.0, false),   // bucket 0, draw
            (39,  1.0, false),   // bucket 0, decisive
            (40,  0.0, false),   // bucket 1, draw
            (79,  0.0, false),   // bucket 1, draw
            (100, 0.0, true),    // bucket 2 — CAP-TERMINATED, excluded from bucket 2 precision
            (200, 1.0, false),   // bucket 5, decisive
            (999, 0.0, false),   // overflow → bucket 9, draw
        ]
        for (ply, outcome, cap) in pliesAndOutcomes {
            tracker.recordGameCompleted(plyCount: Int(ply), firstFlagPlyIndex: ply, excludedFromPrecision: cap, outcome: outcome)
        }
        let s = tracker.snapshot()
        XCTAssertEqual(s.gamesFlaggedByBucket[0], 2)
        XCTAssertEqual(s.gamesFlaggedByBucket[1], 2)
        XCTAssertEqual(s.gamesFlaggedByBucket[2], 1)
        XCTAssertEqual(s.gamesFlaggedByBucket[5], 1)
        XCTAssertEqual(s.gamesFlaggedByBucket[s.gamesFlaggedByBucket.count - 1], 1)

        XCTAssertEqual(s.drawAccuracyForBucket(0) ?? -1, 0.5, accuracy: 1e-9)
        XCTAssertEqual(s.drawAccuracyForBucket(1) ?? -1, 1.0, accuracy: 1e-9)
        XCTAssertNil(s.drawAccuracyForBucket(2)) // cap-terminated → eligible=0
        XCTAssertEqual(s.drawAccuracyForBucket(5) ?? -1, 0.0, accuracy: 1e-9)
        XCTAssertEqual(s.drawAccuracyForBucket(s.gamesFlaggedByBucket.count - 1) ?? -1, 1.0, accuracy: 1e-9)
    }

    // MARK: - Rolling window

    func testResetClearsEverything() {
        let tracker = DrawWatchTracker()
        for _ in 0..<10 {
            tracker.recordGameCompleted(plyCount: 50, firstFlagPlyIndex: 10, excludedFromPrecision: false, outcome: 0.0)
        }
        tracker.reset()
        let s = tracker.snapshot()
        XCTAssertEqual(s.totalGames, 0)
        XCTAssertEqual(s.flaggedGames, 0)
        XCTAssertEqual(s.flaggedGamesEligible, 0)
        XCTAssertEqual(s.flaggedGamesDrawn, 0)
        XCTAssertEqual(s.gamesFlaggedByBucket, Array(repeating: 0, count: DrawWatchTracker.histogramBucketCount))
    }

    func testWindowSecExposed() {
        // Sanity — the snapshot carries the window length so the
        // chart can label it without hard-coding the constant.
        let tracker = DrawWatchTracker()
        XCTAssertEqual(tracker.snapshot().windowSec, DrawWatchTracker.windowSec, accuracy: 1e-9)
    }

    // MARK: - Constants surface area

    // MARK: - Imposed-outcome exclusion (cap OR draw-watch-terminated)

    func testImposedOutcomeExcludedRegardlessOfSource() {
        // The tracker doesn't distinguish "cap-terminated" from
        // "draw-watch-terminated" — both flow through
        // `excludedFromPrecision: true` and are excluded from the
        // precision denominator. The bucket histogram still counts
        // them (they did flag). Verifies the rename's intent.
        let tracker = DrawWatchTracker()
        // Two flagged games, both with imposed outcomes:
        tracker.recordGameCompleted(plyCount: 175, firstFlagPlyIndex: 100, excludedFromPrecision: true, outcome: 0.0)
        tracker.recordGameCompleted(plyCount:  45, firstFlagPlyIndex:  40, excludedFromPrecision: true, outcome: 0.0)
        let s = tracker.snapshot()
        XCTAssertEqual(s.flaggedGames, 2)
        XCTAssertEqual(s.flaggedGamesEligible, 0)
        XCTAssertNil(s.flagDrawAccuracy)
        // Both bucket counts present:
        XCTAssertEqual(s.gamesFlaggedByBucket[1], 1) // 40-80
        XCTAssertEqual(s.gamesFlaggedByBucket[2], 1) // 80-120
    }

    // MARK: - Constants surface area

    func testLockedConstants() {
        // These are the locked decisions documented in DRAW_WATCH_PLAN.md.
        // Changing them is a behavior change worth a deliberate commit.
        XCTAssertEqual(DrawWatchTracker.defaultFlagThresholdPDraw, 0.95, accuracy: 1e-9)
        XCTAssertEqual(DrawWatchTracker.defaultFlagStreakLength, 8)
        XCTAssertEqual(DrawWatchTracker.histogramBucketWidthPlies, 40)
        XCTAssertEqual(DrawWatchTracker.histogramBucketCount, 10)
        XCTAssertEqual(DrawWatchTracker.windowSec, 5 * 60, accuracy: 1e-9)
    }
}
