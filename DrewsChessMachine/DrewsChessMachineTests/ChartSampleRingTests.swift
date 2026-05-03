import XCTest
@testable import DrewsChessMachine

@MainActor
final class ChartSampleRingTests: XCTestCase {

    /// Lightweight stand-in for chart sample types so the tests
    /// exercise `ChartSampleRing` in isolation, without dragging the
    /// real `TrainingChartSample` schema along.
    private struct TestSample: Equatable {
        let id: Int
        let elapsedSec: Double
    }

    func testEmptyRingReportsCorrectState() {
        let ring = ChartSampleRing<TestSample>()
        XCTAssertEqual(ring.count, 0)
        XCTAssertTrue(ring.isEmpty)
        XCTAssertNil(ring.last)
    }

    func testAppendUpdatesCountLastAndSubscript() {
        let ring = ChartSampleRing<TestSample>()
        ring.append(TestSample(id: 0, elapsedSec: 0))
        ring.append(TestSample(id: 1, elapsedSec: 1))
        ring.append(TestSample(id: 2, elapsedSec: 2))

        XCTAssertEqual(ring.count, 3)
        XCTAssertFalse(ring.isEmpty)
        XCTAssertEqual(ring.last, TestSample(id: 2, elapsedSec: 2))
        XCTAssertEqual(ring[0], TestSample(id: 0, elapsedSec: 0))
        XCTAssertEqual(ring[1], TestSample(id: 1, elapsedSec: 1))
        XCTAssertEqual(ring[2], TestSample(id: 2, elapsedSec: 2))
    }

    /// Cross a block boundary in the middle of an append sequence
    /// and confirm every element survives. The block size is large
    /// (86 400) so we rely on `ChartSampleRing.blockSize` being
    /// available rather than re-defining the threshold here.
    func testAppendCrossesBlockBoundaryWithoutLoss() {
        let ring = ChartSampleRing<TestSample>()
        let totalCount = ChartSampleRing<TestSample>.blockSize + 5

        for i in 0..<totalCount {
            ring.append(TestSample(id: i, elapsedSec: Double(i)))
        }

        XCTAssertEqual(ring.count, totalCount)
        // Spot-check elements straddling the boundary: last of block 0,
        // first of block 1, and a few past the boundary.
        let boundary = ChartSampleRing<TestSample>.blockSize - 1
        XCTAssertEqual(ring[boundary], TestSample(id: boundary, elapsedSec: Double(boundary)))
        XCTAssertEqual(ring[boundary + 1], TestSample(id: boundary + 1, elapsedSec: Double(boundary + 1)))
        XCTAssertEqual(ring[totalCount - 1], TestSample(id: totalCount - 1, elapsedSec: Double(totalCount - 1)))
        XCTAssertEqual(ring.last, TestSample(id: totalCount - 1, elapsedSec: Double(totalCount - 1)))
    }

    func testResetClearsContentsButKeepsRingUsable() {
        let ring = ChartSampleRing<TestSample>()
        for i in 0..<10 {
            ring.append(TestSample(id: i, elapsedSec: Double(i)))
        }
        ring.reset()

        XCTAssertEqual(ring.count, 0)
        XCTAssertTrue(ring.isEmpty)
        XCTAssertNil(ring.last)

        // Ring is reusable after reset and indices restart at 0.
        ring.append(TestSample(id: 99, elapsedSec: 99))
        XCTAssertEqual(ring.count, 1)
        XCTAssertEqual(ring[0], TestSample(id: 99, elapsedSec: 99))
    }

    /// Reset after multi-block growth must release extra blocks but
    /// leave the ring in a working state (otherwise long sessions
    /// would leak block storage across resume cycles).
    func testResetAfterMultiBlockGrowthReleasesExtraBlocks() {
        let ring = ChartSampleRing<TestSample>()
        let total = ChartSampleRing<TestSample>.blockSize + 100
        for i in 0..<total {
            ring.append(TestSample(id: i, elapsedSec: Double(i)))
        }
        ring.reset()
        XCTAssertEqual(ring.count, 0)

        // After reset, appending fresh data still works and behaves
        // identically to a never-grown ring.
        for i in 0..<3 {
            ring.append(TestSample(id: i, elapsedSec: Double(i)))
        }
        XCTAssertEqual(ring.count, 3)
        XCTAssertEqual(ring[2], TestSample(id: 2, elapsedSec: 2))
    }

    func testFirstIndexBinarySearchOnEmptyRing() {
        let ring = ChartSampleRing<TestSample>()
        let idx = ring.firstIndex(elapsedSecAtLeast: 5) { $0.elapsedSec }
        XCTAssertEqual(idx, 0)
    }

    func testFirstIndexBinarySearchHits() {
        let ring = ChartSampleRing<TestSample>()
        for i in 0..<10 {
            ring.append(TestSample(id: i, elapsedSec: Double(i) * 2))
        }
        // Elapsed values are 0, 2, 4, 6, 8, 10, 12, 14, 16, 18.
        XCTAssertEqual(ring.firstIndex(elapsedSecAtLeast: 0) { $0.elapsedSec }, 0)
        XCTAssertEqual(ring.firstIndex(elapsedSecAtLeast: 1) { $0.elapsedSec }, 1)
        XCTAssertEqual(ring.firstIndex(elapsedSecAtLeast: 2) { $0.elapsedSec }, 1)
        XCTAssertEqual(ring.firstIndex(elapsedSecAtLeast: 3) { $0.elapsedSec }, 2)
        XCTAssertEqual(ring.firstIndex(elapsedSecAtLeast: 18) { $0.elapsedSec }, 9)
        XCTAssertEqual(ring.firstIndex(elapsedSecAtLeast: 19) { $0.elapsedSec }, 10)
        XCTAssertEqual(ring.firstIndex(elapsedSecAtLeast: 100) { $0.elapsedSec }, 10)
    }

    /// Binary search must remain correct after the ring has spilled
    /// across a block boundary, because the random-access subscript
    /// maps `(blockIdx, offset)`.
    func testFirstIndexBinarySearchAcrossBlockBoundary() {
        let ring = ChartSampleRing<TestSample>()
        let total = ChartSampleRing<TestSample>.blockSize + 10
        for i in 0..<total {
            ring.append(TestSample(id: i, elapsedSec: Double(i)))
        }
        let boundary = ChartSampleRing<TestSample>.blockSize
        XCTAssertEqual(
            ring.firstIndex(elapsedSecAtLeast: Double(boundary)) { $0.elapsedSec },
            boundary
        )
        XCTAssertEqual(
            ring.firstIndex(elapsedSecAtLeast: Double(boundary - 1)) { $0.elapsedSec },
            boundary - 1
        )
    }
}
