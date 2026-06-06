import XCTest
@testable import DrewsChessMachine

final class TrainingStepBackfillTests: XCTestCase {
    private typealias Anchor = TrainingStepBackfill.Anchor

    // MARK: - normalize

    func testNormalizeSortsAndEnforcesMonotonicStep() {
        let raw = [
            Anchor(elapsedSec: 30, step: 300),
            Anchor(elapsedSec: 10, step: 100),
            Anchor(elapsedSec: 20, step: 90),   // out-of-order step → clamped up to 100
        ]
        let out = TrainingStepBackfill.normalize(raw)
        XCTAssertEqual(out.map(\.elapsedSec), [10, 20, 30])
        XCTAssertEqual(out.map(\.step), [100, 100, 300])
    }

    func testNormalizeDropsNonFiniteAndNegative() {
        let raw = [
            Anchor(elapsedSec: .nan, step: 5),
            Anchor(elapsedSec: -1, step: 5),
            Anchor(elapsedSec: 10, step: -3),
            Anchor(elapsedSec: 5, step: 50),
        ]
        let out = TrainingStepBackfill.normalize(raw)
        XCTAssertEqual(out, [Anchor(elapsedSec: 5, step: 50)])
    }

    func testNormalizeCollapsesDuplicateElapsedToMaxStep() {
        let raw = [
            Anchor(elapsedSec: 10, step: 100),
            Anchor(elapsedSec: 10, step: 140),
        ]
        let out = TrainingStepBackfill.normalize(raw)
        XCTAssertEqual(out, [Anchor(elapsedSec: 10, step: 140)])
    }

    // MARK: - interpolatedStep

    func testNoAnchorsReturnsNil() {
        XCTAssertNil(TrainingStepBackfill.interpolatedStep(elapsedSec: 5, anchors: []))
    }

    func testExactAtAnchorReturnsThatStep() {
        let a = TrainingStepBackfill.normalize([
            Anchor(elapsedSec: 100, step: 1000),
            Anchor(elapsedSec: 200, step: 2000),
        ])
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 100, anchors: a), 1000)
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 200, anchors: a), 2000)
    }

    func testBetweenAnchorsIsLinear() {
        let a = TrainingStepBackfill.normalize([
            Anchor(elapsedSec: 100, step: 1000),
            Anchor(elapsedSec: 200, step: 2000),
        ])
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 150, anchors: a), 1500)
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 125, anchors: a), 1250)
    }

    func testBeforeFirstAnchorLinearFromOrigin() {
        let a = TrainingStepBackfill.normalize([
            Anchor(elapsedSec: 100, step: 1000),
        ])
        // Origin (0,0) → (100, 1000): half-way time → half the steps.
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 50, anchors: a), 500)
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 0, anchors: a), 0)
    }

    func testAfterLastAnchorExtrapolatesAlongFinalSlope() {
        let a = TrainingStepBackfill.normalize([
            Anchor(elapsedSec: 100, step: 1000),
            Anchor(elapsedSec: 200, step: 2000),
        ])
        // Slope 10 steps/sec; 50s past last → +500.
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 250, anchors: a), 2500)
    }

    func testSingleAnchorAfterIsFlat() {
        let a = TrainingStepBackfill.normalize([
            Anchor(elapsedSec: 100, step: 1000),
        ])
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 999, anchors: a), 1000)
    }

    func testFlatSegmentAcrossPauseStaysFlat() {
        // Simulates an arena pause: time advances (100→200) while step
        // is frozen (1000→1000), then training resumes.
        let a = TrainingStepBackfill.normalize([
            Anchor(elapsedSec: 100, step: 1000),
            Anchor(elapsedSec: 200, step: 1000),
            Anchor(elapsedSec: 300, step: 2000),
        ])
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 150, anchors: a), 1000)
        XCTAssertEqual(TrainingStepBackfill.interpolatedStep(elapsedSec: 250, anchors: a), 1500)
    }

    func testResultIsNonDecreasingAcrossQueries() {
        let a = TrainingStepBackfill.normalize([
            Anchor(elapsedSec: 10, step: 50),
            Anchor(elapsedSec: 40, step: 50),    // pause
            Anchor(elapsedSec: 90, step: 900),
            Anchor(elapsedSec: 120, step: 1200),
        ])
        var prev = Int.min
        for t in stride(from: 0.0, through: 200.0, by: 1.0) {
            let s = TrainingStepBackfill.interpolatedStep(elapsedSec: t, anchors: a) ?? Int.min
            XCTAssertGreaterThanOrEqual(s, prev, "step went backwards at t=\(t)")
            prev = s
        }
    }
}
