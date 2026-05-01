//
//  BatchSizeStatsTests.swift
//  DrewsChessMachineTests
//
//  Pure-data tests for `BatchSizeStats`, the post-arena telemetry
//  payload `BatchedMoveEvaluationSource` exposes for the arena's
//  "how much concurrency did we end up with" log line. The full
//  batcher requires a Metal-backed `ChessMPSNetwork` so the
//  end-to-end timer + barrier paths are verified through manual
//  arena runs (see the plan's verification section), but the format
//  helper is pure-Swift and worth pinning here so a future log-line
//  refactor catches regressions before they hit a session log.
//

import XCTest
@testable import DrewsChessMachine

final class BatchSizeStatsTests: XCTestCase {

    func testEmptyStatsFormatsAsNoBatches() {
        XCTAssertEqual(BatchSizeStats.empty.formatLogLine(), "no batches fired")
    }

    func testNonEmptyStatsRendersHistogramSorted() {
        // Histogram keys (batch sizes) should be sorted ascending in
        // the rendered output so a log reader can see the
        // distribution at a glance.
        let stats = BatchSizeStats(
            totalBatches: 6,
            totalPositions: 22,
            minBatch: 1,
            maxBatch: 8,
            mean: 22.0 / 6.0,
            histogram: [8: 1, 1: 2, 4: 3],
            fireReasonCounts: [:],
            expectedDriftCount: 0,
            expectedDriftMaxDelta: 0
        )
        let line = stats.formatLogLine()
        XCTAssertTrue(line.contains("mean=3.67"), "got: \(line)")
        XCTAssertTrue(line.contains("min=1"))
        XCTAssertTrue(line.contains("max=8"))
        XCTAssertTrue(line.contains("batches=6"))
        XCTAssertTrue(line.contains("positions=22"))
        // Histogram must be ascending-sorted by key.
        XCTAssertTrue(line.contains("hist={1:2,4:3,8:1}"), "got: \(line)")
    }
}
