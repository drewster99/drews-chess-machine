//
//  ChartZoomTests.swift
//  DrewsChessMachineTests
//
//  Correctness tests for the chart zoom state machine's pure helpers.
//  The zoom stops list and the auto-index / max-zoom-out rules are
//  load-bearing — a bug here would let the user zoom to "30 days"
//  with five minutes of data (footgun the ticket explicitly rules
//  out) or drop the 15-minute tightest stop.
//

import XCTest
@testable import DrewsChessMachine

final class ChartZoomTests: XCTestCase {

    // MARK: - Stops list invariants

    func testStopsAndLabelsSameLength() {
        XCTAssertEqual(ChartZoom.stops.count, ChartZoom.labels.count)
    }

    func testStopsAreStrictlyAscending() {
        for i in 1..<ChartZoom.stops.count {
            XCTAssertGreaterThan(ChartZoom.stops[i], ChartZoom.stops[i - 1],
                "stops must be strictly ascending — zoom-in/out walks them by index")
        }
    }

    func testStopsTightestIs15Minutes() {
        XCTAssertEqual(ChartZoom.stops.first!, 15 * 60, accuracy: 1e-9)
        XCTAssertEqual(ChartZoom.labels.first!, "15m")
    }

    func testStopsWidestIs30Days() {
        XCTAssertEqual(ChartZoom.stops.last!, 30 * 86400, accuracy: 1e-9)
        XCTAssertEqual(ChartZoom.labels.last!, "30d")
    }

    func testExpectedHumanFriendlyStops() {
        // Ticket-specified stops: 15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h,
        // 2d, 3d, 4d, 5d, 6d, 7d, 10d, 20d, 30d (17 stops total).
        let expected: [Double] = [
            15 * 60, 30 * 60,
            3600, 2 * 3600, 4 * 3600, 8 * 3600, 12 * 3600, 24 * 3600,
            2 * 86400, 3 * 86400, 4 * 86400, 5 * 86400, 6 * 86400, 7 * 86400,
            10 * 86400, 20 * 86400, 30 * 86400
        ]
        XCTAssertEqual(ChartZoom.stops.count, expected.count)
        for (got, want) in zip(ChartZoom.stops, expected) {
            XCTAssertEqual(got, want, accuracy: 1e-9)
        }
    }

    func testLabelsMatchStops() {
        let expectedLabels = [
            "15m", "30m", "1h", "2h", "4h", "8h", "12h", "24h",
            "2d", "3d", "4d", "5d", "6d", "7d", "10d", "20d", "30d"
        ]
        XCTAssertEqual(ChartZoom.labels, expectedLabels)
    }

    func testDefaultIndexIs30Minutes() {
        // Starting stop matches the pre-zoom constant (1800s = 30m)
        // so sessions from before this feature don't visually jump
        // on first render.
        XCTAssertEqual(ChartZoom.stops[ChartZoom.defaultIndex], 1800, accuracy: 1e-9)
        XCTAssertEqual(ChartZoom.labels[ChartZoom.defaultIndex], "30m")
    }

    // MARK: - autoIndex(forDataSec:)

    func testAutoIndexEmptyData() {
        // Zero data → pick the tightest stop (15m) since "0 <= 15m".
        XCTAssertEqual(ChartZoom.autoIndex(forDataSec: 0), 0)
    }

    func testAutoIndexExactBoundaryFits() {
        // Exactly 15m of data fits in the 15m stop.
        XCTAssertEqual(ChartZoom.autoIndex(forDataSec: 15 * 60), 0)
    }

    func testAutoIndexJustOverBoundaryJumpsUp() {
        // 15m + 1s doesn't fit in 15m, jumps to 30m.
        XCTAssertEqual(ChartZoom.autoIndex(forDataSec: 15 * 60 + 1), 1)
    }

    func testAutoIndexUserSpecExampleTwentyNineMinutes() {
        // User-described example: "if we have 29 minutes of data,
        // 30 minutes would be the size that fits" (auto index) —
        // the actual picked index at Auto time.
        let idx = ChartZoom.autoIndex(forDataSec: 29 * 60)
        XCTAssertEqual(ChartZoom.labels[idx], "30m")
    }

    func testAutoIndexThreeHoursFortyFiveMinutes() {
        // User-described example: with 3h45m of data, the fitting
        // Auto stop is 4h.
        let idx = ChartZoom.autoIndex(forDataSec: 3 * 3600 + 45 * 60)
        XCTAssertEqual(ChartZoom.labels[idx], "4h")
    }

    func testAutoIndexOverflowPinsToLast() {
        // Data exceeding widest stop → last index (30d), not crash.
        let idx = ChartZoom.autoIndex(forDataSec: 60 * 86400)
        XCTAssertEqual(idx, ChartZoom.stops.count - 1)
        XCTAssertEqual(ChartZoom.labels[idx], "30d")
    }

    func testAutoIndexNegativeClampsToZero() {
        // Defensive — negative elapsed shouldn't happen but we
        // clamp so we never end up with a wild index.
        XCTAssertEqual(ChartZoom.autoIndex(forDataSec: -100), 0)
    }

    // MARK: - maxZoomOutIndex(forDataSec:) — the "auto + 3" rule

    func testMaxZoomOutIsAutoPlusThreeInMiddle() {
        // 3h45m data → auto is 4h (idx 4). Max zoom-out is idx 7
        // (24h). User: "if we have 3 hrs and 45 minutes of data,
        // we would allow zoom out to 24hr."
        let dataSec: Double = 3 * 3600 + 45 * 60
        let autoIdx = ChartZoom.autoIndex(forDataSec: dataSec)
        let maxIdx = ChartZoom.maxZoomOutIndex(forDataSec: dataSec)
        XCTAssertEqual(autoIdx, 4)
        XCTAssertEqual(maxIdx, 7)
        XCTAssertEqual(ChartZoom.labels[maxIdx], "24h")
    }

    func testMaxZoomOutNearEndClamps() {
        // Auto near the top (7d, idx 13) → max zoom-out would be
        // idx 16, which happens to equal stops.count - 1. Safe.
        let dataSec: Double = 7 * 86400
        let maxIdx = ChartZoom.maxZoomOutIndex(forDataSec: dataSec)
        XCTAssertEqual(maxIdx, ChartZoom.stops.count - 1)
    }

    func testMaxZoomOutPastEndClampsToLast() {
        // Auto picks the last index already (data > 30d). Max
        // zoom-out can't go any further. `+3` must clamp.
        let dataSec: Double = 100 * 86400
        let maxIdx = ChartZoom.maxZoomOutIndex(forDataSec: dataSec)
        XCTAssertEqual(maxIdx, ChartZoom.stops.count - 1)
    }

    func testMaxZoomOutEmptyData() {
        // Zero data → auto is idx 0, max zoom-out is idx 3 (2h).
        // Matches "only allow zooming OUT to 3 stops past the
        // current longest length of data".
        let maxIdx = ChartZoom.maxZoomOutIndex(forDataSec: 0)
        XCTAssertEqual(maxIdx, 3)
        XCTAssertEqual(ChartZoom.labels[maxIdx], "2h")
    }

    // MARK: - clamp(_:forDataSec:)

    func testClampWithinRangeIsIdentity() {
        let dataSec: Double = 2 * 3600  // auto = 2h (idx 3), max = 6 (12h)
        for idx in 0...6 {
            XCTAssertEqual(ChartZoom.clamp(idx, forDataSec: dataSec), idx)
        }
    }

    func testClampBelowZeroPinsToZero() {
        XCTAssertEqual(ChartZoom.clamp(-5, forDataSec: 0), 0)
    }

    func testClampAboveMaxPinsToMax() {
        let dataSec: Double = 2 * 3600  // max idx = 6
        XCTAssertEqual(ChartZoom.clamp(999, forDataSec: dataSec), 6)
    }

    func testClampRespectsDataShrinkage() {
        // User had 5 hours of data and manually zoomed out to
        // idx 8 (2d). If data is later cleared or shrinks, clamp
        // must pull the index back into legal range.
        let shrunkDataSec: Double = 10 * 60  // auto = 15m (idx 0), max = 3 (2h)
        XCTAssertEqual(ChartZoom.clamp(8, forDataSec: shrunkDataSec), 3)
    }

    // MARK: - Consistency between helpers

    func testAutoIndexNeverExceedsMaxZoomOutIndex() {
        // By definition max = auto + 3 (clamped). Sanity-check
        // across a wide range of data spans.
        let samples: [Double] = [
            0, 60, 300, 900, 1800, 3600, 5 * 3600,
            24 * 3600, 2 * 86400, 7 * 86400, 30 * 86400, 60 * 86400
        ]
        for d in samples {
            let auto = ChartZoom.autoIndex(forDataSec: d)
            let maxOut = ChartZoom.maxZoomOutIndex(forDataSec: d)
            XCTAssertLessThanOrEqual(auto, maxOut, "auto must fit inside max-zoom-out at data=\(d)")
        }
    }
}
