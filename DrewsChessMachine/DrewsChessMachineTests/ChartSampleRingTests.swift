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

    /// `bulkRestore` of a single-block-or-smaller payload must be
    /// readable end-to-end. Baseline guard for the multi-block case
    /// below.
    func testBulkRestoreSingleBlockIsFullyReadable() {
        let ring = ChartSampleRing<TestSample>()
        let total = 1_000
        let samples = (0..<total).map { TestSample(id: $0, elapsedSec: Double($0)) }
        ring.bulkRestore(samples)

        XCTAssertEqual(ring.count, total)
        for i in 0..<total {
            XCTAssertEqual(ring[i], TestSample(id: i, elapsedSec: Double(i)))
        }
    }

    /// Regression: a `bulkRestore` payload that spans more than one
    /// block must leave EVERY linear index backed by real storage.
    /// The earlier implementation pre-appended empty blocks before
    /// the append loop, but `append` always writes to the last
    /// block — so every sample landed in the final block while
    /// `blocks[0]` stayed empty, and `ring[0]` trapped with
    /// "Index out of range". This is the session-resume crash that
    /// fired once a saved chart trajectory exceeded `blockSize`
    /// samples.
    func testBulkRestoreMultiBlockIsFullyReadable() {
        let ring = ChartSampleRing<TestSample>()
        let total = ChartSampleRing<TestSample>.blockSize + 25
        let samples = (0..<total).map { TestSample(id: $0, elapsedSec: Double($0)) }
        ring.bulkRestore(samples)

        XCTAssertEqual(ring.count, total)
        // Read the first element, the elements straddling the block
        // boundary, and the last element. The first element is the
        // one that crashed before the fix.
        let boundary = ChartSampleRing<TestSample>.blockSize
        XCTAssertEqual(ring[0], TestSample(id: 0, elapsedSec: 0))
        XCTAssertEqual(ring[boundary - 1], TestSample(id: boundary - 1, elapsedSec: Double(boundary - 1)))
        XCTAssertEqual(ring[boundary], TestSample(id: boundary, elapsedSec: Double(boundary)))
        XCTAssertEqual(ring[total - 1], TestSample(id: total - 1, elapsedSec: Double(total - 1)))
        XCTAssertEqual(ring.last, TestSample(id: total - 1, elapsedSec: Double(total - 1)))

        // Exhaustive read so no interior index is left unbacked.
        for i in 0..<total {
            XCTAssertEqual(ring[i], TestSample(id: i, elapsedSec: Double(i)))
        }

        // Binary search (the actual crash site in the stack trace)
        // must also walk every index without trapping.
        XCTAssertEqual(
            ring.firstIndex(elapsedSecAtLeast: Double(boundary)) { $0.elapsedSec },
            boundary
        )
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
