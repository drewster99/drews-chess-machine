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

    func testEmptyRingsProduceEmptyFrame() {
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        let progressRing = ChartSampleRing<ProgressRateSample>()

        let frame = ChartDecimator.decimate(
            trainingRing: trainingRing,
            progressRateRing: progressRing,
            visibleStart: 0,
            visibleLength: 60,
            trainingBucketBudget: 100,
            progressRateBucketBudget: 100
        )

        XCTAssertTrue(frame.trainingBuckets.isEmpty)
        XCTAssertTrue(frame.progressRateBuckets.isEmpty)
        XCTAssertNil(frame.lastTrainingElapsedSec)
        XCTAssertNil(frame.lastProgressRateElapsedSec)
    }

    func testVisibleWindowEntirelyOutsideDataReturnsEmpty() {
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        for i in 0..<10 {
            trainingRing.append(trainingSample(id: i, elapsedSec: Double(i)))
        }

        // Window [100, 160] is past every sample (which run 0...9).
        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
            visibleStart: 100,
            visibleEnd: 160,
            bucketBudget: 50
        )
        XCTAssertTrue(buckets.isEmpty)
    }

    func testVisibleWindowEntirelyBeforeDataReturnsEmpty() {
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        for i in 0..<10 {
            trainingRing.append(trainingSample(id: i, elapsedSec: Double(50 + i)))
        }
        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
            visibleStart: 0,
            visibleEnd: 30,
            bucketBudget: 50
        )
        XCTAssertTrue(buckets.isEmpty)
    }

    // MARK: - Bucket count + budget cap

    func testBucketCountNeverExceedsBudget() {
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        // 1000 samples, all distinct elapsedSec values inside [0, 1000).
        for i in 0..<1000 {
            trainingRing.append(trainingSample(
                id: i, elapsedSec: Double(i),
                policyLoss: Double(i)
            ))
        }
        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        for i in 0..<100 {
            trainingRing.append(trainingSample(id: i, elapsedSec: Double(i)))
        }
        // Caller asks for 1 bucket; decimator clamps up to the floor.
        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        for i in 0..<5000 {
            trainingRing.append(trainingSample(id: i, elapsedSec: Double(i)))
        }
        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        // Two samples, both fall in the same bucket. One has a known
        // min, the other a known max. Decimator should report exactly
        // those values.
        //
        // Visible window [0, 1] with the bucket-budget floor (32)
        // gives bucket 0 = [0, 1/32). Cluster both samples inside it.
        trainingRing.append(trainingSample(id: 0, elapsedSec: 0.005, policyLoss: -3.5))
        trainingRing.append(trainingSample(id: 1, elapsedSec: 0.020, policyLoss: 7.25))

        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        // Cluster all samples inside bucket 0 = [0, 1/32) under the
        // floor-clamped layout.
        trainingRing.append(trainingSample(id: 0, elapsedSec: 0.005, policyLoss: nil))
        trainingRing.append(trainingSample(id: 1, elapsedSec: 0.015, policyLoss: 5.0))
        trainingRing.append(trainingSample(id: 2, elapsedSec: 0.025, policyLoss: nil))

        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        // Cluster inside bucket 0 = [0, 1/32) under the floor.
        trainingRing.append(trainingSample(id: 0, elapsedSec: 0.005))
        trainingRing.append(trainingSample(id: 1, elapsedSec: 0.020))

        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        // Three samples in one bucket; lowPowerMode flips false → true → false.
        // Last-observed semantics should report `false`.
        // Cluster inside bucket 0 = [0, 1/32) under the floor.
        trainingRing.append(trainingSample(id: 0, elapsedSec: 0.005, lowPowerMode: false))
        trainingRing.append(trainingSample(id: 1, elapsedSec: 0.015, lowPowerMode: true))
        trainingRing.append(trainingSample(id: 2, elapsedSec: 0.025, lowPowerMode: false))

        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
            visibleStart: 0,
            visibleEnd: 1,
            bucketBudget: 1
        )
        XCTAssertEqual(buckets.count, 1)
        XCTAssertEqual(buckets[0].lowPowerMode, false)
    }

    func testThermalStateLastObservedSurvivesNilSamples() {
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        // Cluster inside bucket 0 = [0, 1/32) under the floor.
        trainingRing.append(trainingSample(id: 0, elapsedSec: 0.005, thermalState: .nominal))
        trainingRing.append(trainingSample(id: 1, elapsedSec: 0.015, thermalState: .fair))
        trainingRing.append(trainingSample(id: 2, elapsedSec: 0.025, thermalState: nil))

        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        // Samples cluster at t=0.05 and t=0.95, leaving the middle
        // buckets entirely empty.
        trainingRing.append(trainingSample(id: 0, elapsedSec: 0.05, policyLoss: 1.0))
        trainingRing.append(trainingSample(id: 1, elapsedSec: 0.95, policyLoss: 2.0))

        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        for i in 0..<100 {
            trainingRing.append(trainingSample(
                id: i, elapsedSec: Double(i),
                policyLoss: Double(i)
            ))
        }
        // Window [40, 99] — the sample at exactly 99 must appear.
        let buckets = ChartDecimator.decimateTraining(
            ring: trainingRing,
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

    func testLastElapsedReportsRingTailNotVisibleWindow() {
        let trainingRing = ChartSampleRing<TrainingChartSample>()
        let progressRing = ChartSampleRing<ProgressRateSample>()
        for i in 0..<10 {
            trainingRing.append(trainingSample(id: i, elapsedSec: Double(i)))
            progressRing.append(progressSample(id: i, elapsedSec: Double(i)))
        }
        let frame = ChartDecimator.decimate(
            trainingRing: trainingRing,
            progressRateRing: progressRing,
            // Visible window is BEFORE the data — but lastElapsed is
            // about the underlying ring, not the visible window.
            visibleStart: 0,
            visibleLength: 5,
            trainingBucketBudget: 50,
            progressRateBucketBudget: 50
        )
        XCTAssertEqual(frame.lastTrainingElapsedSec, 9)
        XCTAssertEqual(frame.lastProgressRateElapsedSec, 9)
    }

    // MARK: - Progress rate

    func testProgressRateDecimationIncludesDerivedCombinedRate() {
        let progressRing = ChartSampleRing<ProgressRateSample>()
        progressRing.append(progressSample(
            id: 0, elapsedSec: 0.5,
            selfPlayMovesPerHour: 100,
            trainingMovesPerHour: 250
        ))
        let buckets = ChartDecimator.decimateProgressRate(
            ring: progressRing,
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
