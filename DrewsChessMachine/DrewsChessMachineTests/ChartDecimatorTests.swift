import XCTest
@testable import DrewsChessMachine

@MainActor
final class ChartDecimatorTests: XCTestCase {

    // MARK: - Sample factories

    private func trainingSample(
        id: Int,
        elapsedSec: Double,
        policyLoss: Double? = nil,
        policyEntropy: Double? = nil,
        gradNorm: Double? = nil,
        replayRatio: Double? = nil,
        cpuPercent: Double? = nil,
        appMemoryMB: Double? = nil,
        lowPowerMode: Bool? = nil,
        thermalState: ProcessInfo.ThermalState? = nil
    ) -> TrainingChartSample {
        TrainingChartSample(
            id: id,
            elapsedSec: elapsedSec,
            rollingPolicyLoss: policyLoss,
            rollingValueLoss: nil,
            rollingPolicyEntropy: policyEntropy,
            rollingPolicyNonNegCount: nil,
            rollingPolicyNonNegIllegalCount: nil,
            rollingGradNorm: gradNorm,
            rollingVelocityNorm: nil,
            rollingPolicyHeadWeightNorm: nil,
            replayRatio: replayRatio,
            rollingPolicyLossWin: nil,
            rollingPolicyLossLoss: nil,
            rollingLegalEntropy: nil,
            rollingLegalMass: nil,
            cpuPercent: cpuPercent,
            gpuBusyPercent: nil,
            gpuMemoryMB: nil,
            appMemoryMB: appMemoryMB,
            lowPowerMode: lowPowerMode,
            thermalState: thermalState
        )
    }

    private func progressSample(
        id: Int,
        elapsedSec: Double,
        selfPlayMovesPerHour: Double = 0,
        trainingMovesPerHour: Double = 0
    ) -> ProgressRateSample {
        ProgressRateSample(
            id: id,
            timestamp: Date(timeIntervalSince1970: elapsedSec),
            elapsedSec: elapsedSec,
            selfPlayCumulativeMoves: 0,
            trainingCumulativeMoves: 0,
            selfPlayMovesPerHour: selfPlayMovesPerHour,
            trainingMovesPerHour: trainingMovesPerHour
        )
    }

    // MARK: - Empty / out-of-range

    func testEmptyInputsProduceEmptyFrame() {
        let frame = ChartDecimator.decimate(
            trainingSamples: [],
            progressRateSamples: [],
            visibleStart: 0,
            visibleEnd: 60,
            lastT: nil,
            lastP: nil,
            trainingBucketBudget: 100,
            progressRateBucketBudget: 100
        )

        XCTAssertTrue(frame.trainingBuckets.isEmpty)
        XCTAssertTrue(frame.progressRateBuckets.isEmpty)
        XCTAssertNil(frame.lastTrainingElapsedSec)
        XCTAssertNil(frame.lastProgressRateElapsedSec)
    }

    func testVisibleWindowEntirelyOutsideDataReturnsEmpty() {
        // Samples run 0...9; caller passes only the visible-window slice,
        // which for window [100, 160] is empty.
        let buckets = ChartDecimator.decimateTraining(
            samples: [],
            visibleStart: 100,
            visibleEnd: 160,
            bucketBudget: 50
        )
        XCTAssertTrue(buckets.isEmpty)
    }

    func testVisibleWindowEntirelyBeforeDataReturnsEmpty() {
        // Samples run 50...59; the visible-window slice for [0, 30] is empty.
        let buckets = ChartDecimator.decimateTraining(
            samples: [],
            visibleStart: 0,
            visibleEnd: 30,
            bucketBudget: 50
        )
        XCTAssertTrue(buckets.isEmpty)
    }

    // MARK: - Bucket count + budget cap

    func testBucketCountNeverExceedsBudget() {
        // 1000 samples, all distinct elapsedSec values inside [0, 1000).
        let samples = (0..<1000).map { trainingSample(id: $0, elapsedSec: Double($0), policyLoss: Double($0)) }
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 1000,
            bucketBudget: 50
        )
        XCTAssertLessThanOrEqual(buckets.count, 50)
        // 1000 samples / 50 buckets = 20 samples per bucket; every
        // bucket should be non-empty and emitted.
        XCTAssertEqual(buckets.count, 50)
    }

    func testBucketBudgetClampedToMinimum() {
        let samples = (0..<100).map { trainingSample(id: $0, elapsedSec: Double($0)) }
        // Caller asks for 1 bucket; decimator clamps up to the floor.
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 100,
            bucketBudget: 1
        )
        XCTAssertGreaterThanOrEqual(buckets.count, 1)
        XCTAssertLessThanOrEqual(
            buckets.count,
            ChartDecimator.minimumBucketCount
        )
    }

    func testBucketBudgetClampedToMaximum() {
        let samples = (0..<5000).map { trainingSample(id: $0, elapsedSec: Double($0)) }
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 5000,
            bucketBudget: 999_999
        )
        XCTAssertLessThanOrEqual(
            buckets.count,
            ChartDecimator.maximumBucketCount
        )
    }

    // MARK: - Min/max envelope

    func testMinMaxEnvelopeIsExactWithinBucket() {
        // Two samples, both fall in the same bucket. One has a known
        // min, the other a known max. Decimator should report exactly
        // those values.
        //
        // Visible window [0, 1] with the bucket-budget floor (32)
        // gives bucket 0 = [0, 1/32). Cluster both samples inside it.
        let samples = [
            trainingSample(id: 0, elapsedSec: 0.005, policyLoss: -3.5),
            trainingSample(id: 1, elapsedSec: 0.020, policyLoss: 7.25)
        ]
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 1,
            // Force everything into one bucket via the floor; samples
            // are clustered inside bucket 0 above.
            bucketBudget: 1
        )
        XCTAssertEqual(buckets.count, 1)
        let range = buckets[0].policyLoss
        XCTAssertNotNil(range)
        XCTAssertEqual(range?.min, -3.5)
        XCTAssertEqual(range?.max, 7.25)
    }

    func testNilFieldDoesNotAffectEnvelope() {
        // Cluster all samples inside bucket 0 = [0, 1/32) under the
        // floor-clamped layout.
        let samples = [
            trainingSample(id: 0, elapsedSec: 0.005, policyLoss: nil),
            trainingSample(id: 1, elapsedSec: 0.015, policyLoss: 5.0),
            trainingSample(id: 2, elapsedSec: 0.025, policyLoss: nil)
        ]
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 1,
            bucketBudget: 1
        )
        XCTAssertEqual(buckets.count, 1)
        // Only the middle sample reported a value; envelope is min=max=5.0.
        let range = buckets[0].policyLoss
        XCTAssertEqual(range?.min, 5.0)
        XCTAssertEqual(range?.max, 5.0)
    }

    func testFieldWithNoSamplesReportsNilRange() {
        // Cluster inside bucket 0 = [0, 1/32) under the floor.
        let samples = [
            trainingSample(id: 0, elapsedSec: 0.005),
            trainingSample(id: 1, elapsedSec: 0.020)
        ]
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 1,
            bucketBudget: 1
        )
        XCTAssertEqual(buckets.count, 1)
        XCTAssertNil(buckets[0].policyLoss)
        XCTAssertNil(buckets[0].gradNorm)
        XCTAssertNil(buckets[0].cpuPercent)
    }

    // MARK: - Categorical fields

    func testCategoricalFieldUsesLastObservedValue() {
        // Three samples in one bucket; lowPowerMode flips false → true → false.
        // Last-observed semantics should report `false`.
        // Cluster inside bucket 0 = [0, 1/32) under the floor.
        let samples = [
            trainingSample(id: 0, elapsedSec: 0.005, lowPowerMode: false),
            trainingSample(id: 1, elapsedSec: 0.015, lowPowerMode: true),
            trainingSample(id: 2, elapsedSec: 0.025, lowPowerMode: false)
        ]
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 1,
            bucketBudget: 1
        )
        XCTAssertEqual(buckets.count, 1)
        XCTAssertEqual(buckets[0].lowPowerMode, false)
    }

    func testThermalStateLastObservedSurvivesNilSamples() {
        // Cluster inside bucket 0 = [0, 1/32) under the floor.
        let samples = [
            trainingSample(id: 0, elapsedSec: 0.005, thermalState: .nominal),
            trainingSample(id: 1, elapsedSec: 0.015, thermalState: .fair),
            trainingSample(id: 2, elapsedSec: 0.025, thermalState: nil)
        ]
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 1,
            bucketBudget: 1
        )
        XCTAssertEqual(buckets.count, 1)
        // Last NON-NIL value wins — a `nil` sample after `.fair`
        // should not reset the bucket's thermal state.
        XCTAssertEqual(buckets[0].thermalState, .fair)
    }

    // MARK: - Empty buckets are skipped

    func testEmptyBucketsAreOmittedFromOutput() {
        // Samples cluster at t=0.05 and t=0.95, leaving the middle
        // buckets entirely empty.
        let samples = [
            trainingSample(id: 0, elapsedSec: 0.05, policyLoss: 1.0),
            trainingSample(id: 1, elapsedSec: 0.95, policyLoss: 2.0)
        ]
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 1,
            bucketBudget: 100
        )
        // Only two buckets contain data; everything else is skipped.
        XCTAssertEqual(buckets.count, 2)
        XCTAssertEqual(buckets[0].policyLoss?.max, 1.0)
        XCTAssertEqual(buckets[1].policyLoss?.max, 2.0)
        // First bucket id ≈ 5/100 of the way; second ≈ 95/100.
        XCTAssertLessThan(buckets[0].id, buckets[1].id)
    }

    // MARK: - Boundary inclusivity

    /// Auto-follow positions `visibleEnd` exactly at the latest
    /// sample's `elapsedSec`. Decimator must include that sample;
    /// otherwise the chart reads one tick stale every 1 Hz append.
    func testSampleAtVisibleEndIsIncluded() {
        // Window [40, 99] — the sample at exactly 99 must appear. The
        // caller passes the in-window slice (samples 40...99).
        let samples = (40...99).map { trainingSample(id: $0, elapsedSec: Double($0), policyLoss: Double($0)) }
        let buckets = ChartDecimator.decimateTraining(
            samples: samples,
            visibleStart: 40,
            visibleEnd: 99,
            bucketBudget: 60
        )
        XCTAssertFalse(buckets.isEmpty)
        let lastBucket = buckets.last
        XCTAssertNotNil(lastBucket)
        // The last bucket should have absorbed the sample at t=99.
        XCTAssertEqual(lastBucket?.policyLoss?.max, 99)
    }

    // MARK: - Last-elapsed reporting

    func testLastElapsedReportsCallerSuppliedTailNotVisibleWindow() {
        // The decimator no longer knows about the underlying ring — the
        // caller passes `lastT` / `lastP` explicitly (it's the ring
        // tail, computed independently of the visible-window slice).
        // Visible window [0, 5) here, but the supplied tail is 9; the
        // frame must echo the supplied tail, not anything derived from
        // the window.
        let allTraining = (0..<10).map { trainingSample(id: $0, elapsedSec: Double($0)) }
        let allProgress = (0..<10).map { progressSample(id: $0, elapsedSec: Double($0)) }
        let inWindowTraining = allTraining.filter { (0.0...5.0).contains($0.elapsedSec) }
        let inWindowProgress = allProgress.filter { (0.0...5.0).contains($0.elapsedSec) }
        let frame = ChartDecimator.decimate(
            trainingSamples: inWindowTraining,
            progressRateSamples: inWindowProgress,
            visibleStart: 0,
            visibleEnd: 5,
            lastT: allTraining.last?.elapsedSec,
            lastP: allProgress.last?.elapsedSec,
            trainingBucketBudget: 50,
            progressRateBucketBudget: 50
        )
        XCTAssertEqual(frame.lastTrainingElapsedSec, 9)
        XCTAssertEqual(frame.lastProgressRateElapsedSec, 9)
    }

    // MARK: - Progress rate

    func testProgressRateDecimationIncludesDerivedCombinedRate() {
        let samples = [
            progressSample(
                id: 0, elapsedSec: 0.5,
                selfPlayMovesPerHour: 100,
                trainingMovesPerHour: 250
            )
        ]
        let buckets = ChartDecimator.decimateProgressRate(
            samples: samples,
            visibleStart: 0,
            visibleEnd: 1,
            bucketBudget: 1
        )
        XCTAssertEqual(buckets.count, 1)
        XCTAssertEqual(buckets[0].selfPlayMovesPerHour?.min, 100)
        XCTAssertEqual(buckets[0].trainingMovesPerHour?.min, 250)
        XCTAssertEqual(buckets[0].combinedMovesPerHour?.min, 350)
    }
}
