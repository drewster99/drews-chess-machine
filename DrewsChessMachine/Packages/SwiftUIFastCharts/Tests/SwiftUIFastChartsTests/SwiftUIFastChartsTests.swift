import XCTest
import CoreGraphics
import SwiftUI
@testable import SwiftUIFastCharts

final class SwiftUIFastChartsTests: XCTestCase {

    // MARK: - Coordinate transform

    func testCoordTransformMapsDomainEndsToRectEdges() throws {
        let rect = CGRect(x: 10, y: 20, width: 100, height: 50)
        let t = ChartCoordTransform(
            xDomain: 0...10,
            yDomain: 0...5,
            rect: rect
        )
        let p0 = t.point(0, 0)
        let p1 = t.point(10, 5)
        // X=0 lands at rect.minX, X=10 at rect.maxX.
        XCTAssertEqual(p0.x, 10, accuracy: 0.001)
        XCTAssertEqual(p1.x, 110, accuracy: 0.001)
        // Y=0 lands at rect.maxY (flipped), Y=max at rect.minY.
        XCTAssertEqual(p0.y, 70, accuracy: 0.001)
        XCTAssertEqual(p1.y, 20, accuracy: 0.001)
    }

    func testCoordTransformDegenerateDomainDoesNotCrash() {
        let rect = CGRect(x: 0, y: 0, width: 100, height: 50)
        let t = ChartCoordTransform(xDomain: 5...5, yDomain: 1...1, rect: rect)
        // Should produce finite values for any input.
        let p = t.point(5, 1)
        XCTAssertTrue(p.x.isFinite)
        XCTAssertTrue(p.y.isFinite)
    }

    // MARK: - Visible range slicing

    func testVisibleRangePointsExcludesOutOfDomain() {
        let pts: [CGPoint] = (0..<10).map { CGPoint(x: Double($0), y: 0) }
        let r = ChartPathBuilder.visibleRange(in: pts, xDomain: 3.5...6.5)
        // First idx with x >= 3.5 is 4; first idx with x > 6.5 is 7.
        // Range pads by one point on each edge so the stroke crosses
        // the viewport — 3..<8.
        XCTAssertEqual(r.lowerBound, 3)
        XCTAssertEqual(r.upperBound, 8)
    }

    func testVisibleRangeBucketsEmpty() {
        let r = ChartPathBuilder.visibleRange(
            in: [FastChartBucket](),
            xDomain: 0...10
        )
        XCTAssertTrue(r.isEmpty)
    }

    func testVisibleRangeBucketsSorted() {
        let buckets = (0..<20).map {
            FastChartBucket(id: $0, x: Double($0), yMin: 0, yMax: 1)
        }
        let r = ChartPathBuilder.visibleRange(in: buckets, xDomain: 10...12)
        XCTAssertTrue(r.contains(10))
        XCTAssertTrue(r.contains(12))
        XCTAssertFalse(r.contains(15))
    }

    // MARK: - Path-builder linear vs step

    func testStrokePathLinearAndStepEndEmitDifferentSegmentCounts() {
        let buckets = (0..<3).map {
            FastChartBucket(id: $0, x: Double($0), yMin: 0, yMax: Double($0))
        }
        let rect = CGRect(x: 0, y: 0, width: 100, height: 100)
        let t = ChartCoordTransform(xDomain: 0...2, yDomain: 0...2, rect: rect)
        let linear = ChartPathBuilder.strokePath(
            buckets: buckets,
            visible: 0..<3,
            interpolation: .linear,
            transform: t
        )
        let step = ChartPathBuilder.strokePath(
            buckets: buckets,
            visible: 0..<3,
            interpolation: .stepEnd,
            transform: t
        )
        // Linear: 1 move + 2 lines = 3 elements.
        // StepEnd: 1 move + 4 lines (two L's per segment) = 5 elements.
        XCTAssertEqual(elementCount(of: linear), 3)
        XCTAssertEqual(elementCount(of: step), 5)
    }

    func testStrokePathBreaksOnNaN() {
        let buckets: [FastChartBucket] = [
            FastChartBucket(id: 0, x: 0, yMin: 0, yMax: 0),
            FastChartBucket(id: 1, x: 1, yMin: 0, yMax: .nan),
            FastChartBucket(id: 2, x: 2, yMin: 0, yMax: 1)
        ]
        let rect = CGRect(x: 0, y: 0, width: 100, height: 100)
        let t = ChartCoordTransform(xDomain: 0...2, yDomain: 0...2, rect: rect)
        let path = ChartPathBuilder.strokePath(
            buckets: buckets,
            visible: 0..<3,
            interpolation: .linear,
            transform: t
        )
        var moveCount = 0
        path.forEach { element in
            if case .move = element { moveCount += 1 }
        }
        // One move at the first valid point, a second after the NaN gap.
        XCTAssertEqual(moveCount, 2)
    }

    private func elementCount(of path: SwiftUI.Path) -> Int {
        var count = 0
        path.forEach { _ in count += 1 }
        return count
    }

    // MARK: - Decimator

    func testDecimatorEmptyInput() {
        let d = FastChartDecimator()
        let out = d.decimate(rawSeries: [], xDomain: 0...10, maxBucketCount: 10)
        XCTAssertTrue(out.isEmpty)
    }

    func testDecimatorRespectsBucketCount() {
        let pts = (0..<1_000).map { CGPoint(x: Double($0), y: Double($0)) }
        let raw = FastChartRawSeries(id: "x", color: .red, points: pts)
        let d = FastChartDecimator()
        let out = d.decimate(rawSeries: [raw], xDomain: 0...1_000, maxBucketCount: 100)
        guard case .buckets(let bs) = out[0].data else {
            return XCTFail("expected buckets output")
        }
        XCTAssertEqual(bs.count, 100)
        // Last bucket's max should hit ~999 (we filter strictly <
        // xDomain.upperBound, so X=1000 is excluded — fine).
        XCTAssertEqual(bs.last!.yMax, 999, accuracy: 1)
    }

    func testDecimatorSkipsOutOfDomainPoints() {
        let pts: [CGPoint] = [
            CGPoint(x: -1, y: 99),
            CGPoint(x: 5, y: 10),
            CGPoint(x: 1000, y: 99)
        ]
        let raw = FastChartRawSeries(id: "x", color: .red, points: pts)
        let d = FastChartDecimator()
        let out = d.decimate(rawSeries: [raw], xDomain: 0...20, maxBucketCount: 4)
        guard case .buckets(let bs) = out[0].data else {
            return XCTFail("expected buckets output")
        }
        XCTAssertEqual(bs.count, 1)
        XCTAssertEqual(bs[0].yMax, 10)
    }

    func testDecimatorMinMaxAggregation() {
        // 4 points in one bucket — verifies min/max accumulation.
        let pts: [CGPoint] = [
            CGPoint(x: 0.1, y: 1),
            CGPoint(x: 0.2, y: 5),
            CGPoint(x: 0.3, y: -2),
            CGPoint(x: 0.4, y: 3)
        ]
        let raw = FastChartRawSeries(id: "x", color: .red, points: pts)
        let d = FastChartDecimator()
        let out = d.decimate(rawSeries: [raw], xDomain: 0...1, maxBucketCount: 1)
        guard case .buckets(let bs) = out[0].data else {
            return XCTFail("expected buckets output")
        }
        XCTAssertEqual(bs.count, 1)
        XCTAssertEqual(bs[0].yMin, -2)
        XCTAssertEqual(bs[0].yMax, 5)
    }

    func testDecimatorSkipsNaN() {
        let pts: [CGPoint] = [
            CGPoint(x: 0.1, y: .nan),
            CGPoint(x: 0.2, y: 3),
            CGPoint(x: 0.3, y: .nan)
        ]
        let raw = FastChartRawSeries(id: "x", color: .red, points: pts)
        let d = FastChartDecimator()
        let out = d.decimate(rawSeries: [raw], xDomain: 0...1, maxBucketCount: 1)
        guard case .buckets(let bs) = out[0].data else {
            return XCTFail("expected buckets output")
        }
        XCTAssertEqual(bs.count, 1)
        XCTAssertEqual(bs[0].yMin, 3)
        XCTAssertEqual(bs[0].yMax, 3)
    }

    // MARK: - Axis layout

    func testEvenlySpacedTicks() {
        let ticks = ChartAxisLayout.evenlySpacedTicks(domain: 0...10, count: 5)
        XCTAssertEqual(ticks.count, 5)
        XCTAssertEqual(ticks.first, 0)
        XCTAssertEqual(ticks.last, 10)
        XCTAssertEqual(ticks[2], 5, accuracy: 0.001)
    }

    func testEvenlySpacedTicksDegenerate() {
        let ticks = ChartAxisLayout.evenlySpacedTicks(domain: 3...3, count: 5)
        XCTAssertEqual(ticks, [3])
    }

    // MARK: - Formatters smoke

    func testCompactFormatter() {
        XCTAssertEqual(FastChartFormatters.compact(0), "0")
        XCTAssertEqual(FastChartFormatters.compact(1234), "1.2K")
        XCTAssertEqual(FastChartFormatters.compact(1234567), "1.2M")
        XCTAssertEqual(FastChartFormatters.compact(0.5), "0.50")
    }

    func testElapsedTimeFormatter() {
        XCTAssertEqual(FastChartFormatters.elapsedTime(5), "0:05")
        XCTAssertEqual(FastChartFormatters.elapsedTime(125), "2:05")
        XCTAssertEqual(FastChartFormatters.elapsedTime(3725), "1:02:05")
    }
}
