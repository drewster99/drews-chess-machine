import XCTest
@testable import DrewsChessMachine

/// Rule 14's pure parts (BN_RUNNING_VARIANCE_CHANGE_ALARM_PLAN, Part X1): the
/// one ratio definition, each threshold on both sides of its boundary, the
/// lookback bounds, the baseline as the lookback minimum, and the offline
/// (largest-channel-only) lower bound.
final class BatchNormRunningVarianceJumpTests: XCTestCase {

    typealias Jump = BatchNormRunningVarianceJump

    private func channels(
        _ ratios: [Double?],
        site: String = "blocks.2.bn1"
    ) -> LayerHealthDigest.RunningVarianceChannels {
        LayerHealthDigest.RunningVarianceChannels(profile: BatchNormRunningVarianceProfile(sites: [
            BatchNormRunningVarianceProfile.Site(site: site, channelCount: ratios.count, ratios: ratios),
        ]))
    }

    private func largest(
        _ ratio: Double,
        channel: Int,
        site: String = "blocks.2.bn1",
        outliers: Int? = nil
    ) -> LayerHealthDigest.RunningVarianceChannels {
        LayerHealthDigest.RunningVarianceChannels(
            coverage: .largestOnly(site: site, channel: channel, ratio: ratio), outlierCount: outliers)
    }

    private func entry(_ step: Int, _ channels: LayerHealthDigest.RunningVarianceChannels) -> Jump.HistoryEntry {
        Jump.HistoryEntry(trainerStep: step, channels: channels)
    }

    // MARK: Ratio

    func testRatioMedianOfOddAndEvenCounts() throws {
        let odd = LayerHealth.runningVarianceRatios([1, 2, 3, 4, 100])
        XCTAssertEqual(odd.median, 3)
        XCTAssertEqual(odd.ratios, [1.0 / 3, 2.0 / 3, 1, 4.0 / 3, 100.0 / 3])
        let even = LayerHealth.runningVarianceRatios([4, 1, 3, 2])
        XCTAssertEqual(even.median, 2.5)
        XCTAssertEqual(even.ratios, [1.6, 0.4, 1.2, 0.8])
    }

    func testNonFiniteVarianceHasNoRatioAndIsLeftOutOfTheMedian() {
        let ratios = LayerHealth.runningVarianceRatios([1, .nan, 3, .infinity, 5])
        XCTAssertEqual(ratios.median, 3)
        XCTAssertEqual(ratios.ratios, [1.0 / 3, nil, 1, nil, 5.0 / 3])
    }

    func testNonPositiveMedianGivesNoRatios() {
        let zero = LayerHealth.runningVarianceRatios([0, 0, 0, 5])
        XCTAssertEqual(zero.median, 0)
        XCTAssertNil(zero.ratios)
        let none = LayerHealth.runningVarianceRatios([.nan, .infinity])
        XCTAssertNil(none.median)
        XCTAssertNil(none.ratios)
        XCTAssertEqual(LayerHealth.runningVarianceOutlierCount(zero.ratios), 0)
    }

    /// The per-site max/median is the largest channel's ratio from the one
    /// definition, so it is unchanged from the value computed before it.
    func testMaxOverMedianIsTheLargestChannelsRatio() {
        let variance: [Float] = [0.5, 2, 1, 7, 1200, .infinity, 3]
        let health = LayerHealth.batchNormSiteHealth(
            site: LayerHealth.BatchNormSite(name: "s", channels: variance.count, activation: .relu),
            gamma: [Float](repeating: 1, count: variance.count), beta: [Float](repeating: 0, count: variance.count),
            runningVariance: variance)
        let ratios = LayerHealth.runningVarianceRatios(variance)
        XCTAssertEqual(health.runningVarianceMedian, ratios.median)
        XCTAssertEqual(health.runningVarianceMaxOverMedian, ratios.ratios?[4])
        // Finite values 0.5, 1, 2, 3, 7, 1200: median (2 + 3) / 2.
        XCTAssertEqual(health.runningVarianceMaxOverMedian, 1200.0 / 2.5)
        XCTAssertEqual(health.runningVarianceOutlierCount, 1)
    }

    // MARK: Thresholds (both sides of every boundary)

    func testJumpLevelAndRiseBoundaries() {
        // Level: ratio 9.99 against a baseline of 0.5 is not an outlier.
        XCTAssertFalse(Jump.channelJumped(ratio: 9.99, baseline: 0.5))
        XCTAssertTrue(Jump.channelJumped(ratio: 10, baseline: 0.5))
        // Rise: 9.99× its baseline does not jump, 10× does.
        XCTAssertFalse(Jump.channelJumped(ratio: 50, baseline: 50 / 9.99))
        XCTAssertTrue(Jump.channelJumped(ratio: 50, baseline: 5))
        // A zero baseline: any outlier jumped.
        XCTAssertTrue(Jump.channelJumped(ratio: 10, baseline: 0))
    }

    func testCriticalLevelBoundary() throws {
        func reading(_ ratio: Double) throws -> Jump.Reading {
            try XCTUnwrap(Jump.read(
                channels([ratio, 1, 1]), history: [entry(1000, channels([0.02, 1, 1]))], trainerStep: 1050))
        }
        let below = try reading(99.9)
        XCTAssertTrue(below.jumpHolds)
        XCTAssertFalse(below.jumpIsCritical)
        let at = try reading(100)
        XCTAssertTrue(at.jumpHolds)
        XCTAssertTrue(at.jumpIsCritical)
    }

    func testOutlierCountRiseBoundaries() {
        let cases: [(baseline: Int, count: Int, rises: Bool)] = [
            (1, 2, false), (2, 4, false), (3, 6, false), (4, 8, false), (4, 9, true), (5, 9, false),
            (5, 10, true), (0, 4, false), (0, 5, true), (11, 53, true),
        ]
        for example in cases {
            XCTAssertEqual(Jump.outlierCountRose(from: example.baseline, to: example.count), example.rises,
                           "\(example.baseline) → \(example.count)")
        }
    }

    // MARK: Lookback

    func testLookbackIncludesExactlyOneThousandStepsBack() throws {
        let current = channels([20, 1])
        let atBound = try XCTUnwrap(Jump.read(current, history: [entry(1000, channels([1, 1]))], trainerStep: 2000))
        XCTAssertEqual(atBound.jumped.map(\.channel), [0], "a read at s − 1000 counts")
        XCTAssertNil(Jump.read(current, history: [entry(999, channels([1, 1]))], trainerStep: 2000),
                     "a read at s − 1001 does not")
        XCTAssertNil(Jump.read(current, history: [entry(2000, channels([1, 1]))], trainerStep: 2000),
                     "the read at s itself is not its own baseline")
    }

    func testBaselineIsTheLookbackMinimumNotThePreviousRead() throws {
        let history = [
            entry(1100, channels([0.02, 1])),
            entry(1500, channels([50, 1])),
            entry(1950, channels([50, 1])),
        ]
        let reading = try XCTUnwrap(Jump.read(channels([154, 1]), history: history, trainerStep: 2000))
        let jumped = try XCTUnwrap(reading.jumped.first)
        XCTAssertEqual(jumped.baseline, 0.02)
        XCTAssertTrue(jumped.baselineIsExact)
        XCTAssertEqual(jumped.ratio, 154)
        XCTAssertTrue(reading.jumpIsCritical)
        XCTAssertEqual(reading.largestRise, jumped)
        XCTAssertEqual(jumped.riseFactor, 154 / 0.02, accuracy: 1e-6)
    }

    func testOutlierBaselineIsTheLowestCountOverTheLookback() throws {
        // Counts 5, 3, 7 in the lookback; now 8 ≥ max(2 × 3, 3 + 5) = 8.
        let history = [
            entry(1200, largest(20, channel: 1, outliers: 5)),
            entry(1500, largest(20, channel: 1, outliers: 3)),
            entry(1900, largest(20, channel: 1, outliers: 7)),
        ]
        let reading = try XCTUnwrap(Jump.read(largest(21, channel: 1, outliers: 8), history: history, trainerStep: 2000))
        XCTAssertEqual(reading.outlierBaseline, 3)
        XCTAssertEqual(reading.outlierCount, 8)
        XCTAssertTrue(reading.outlierCountRiseHolds)
        XCTAssertFalse(reading.jumpHolds)
    }

    func testLargestRiseCoversOutliersThatDidNotJump() throws {
        let reading = try XCTUnwrap(Jump.read(
            channels([60, 12]), history: [entry(1500, channels([50, 4]))], trainerStep: 2000))
        XCTAssertTrue(reading.jumped.isEmpty)
        XCTAssertEqual(reading.largestRise?.channel, 1)
        XCTAssertEqual(reading.largestRise?.riseFactor ?? 0, 3, accuracy: 1e-12)
    }

    // MARK: Offline coverage (largest channel only)

    func testOfflineJumpIsJudgedAgainstThePastLinesLargestRatios() throws {
        let history = [
            entry(1100, largest(61.2, channel: 31)),
            entry(1500, largest(61.3, channel: 31)),
        ]
        // Ten times the smallest past bound (61.2) is 612: below it no jump,
        // above it a jump against an upper bound.
        let below = try XCTUnwrap(Jump.read(largest(611, channel: 76), history: history, trainerStep: 2000))
        XCTAssertTrue(below.jumped.isEmpty)
        let at = try XCTUnwrap(Jump.read(largest(613, channel: 76), history: history, trainerStep: 2000))
        let jumped = try XCTUnwrap(at.jumped.first)
        XCTAssertEqual(jumped.baseline, 61.2)
        XCTAssertFalse(jumped.baselineIsExact, "other channels' largest ratios only bound channel 76")
        XCTAssertNil(at.outlierCount)
        XCTAssertFalse(at.outlierCountRiseHolds, "no rvOver10xMedian= on the lines: the count arm has no data")
    }

    func testOfflineBaselineIsExactWhenEveryPastLineNamedTheChannel() throws {
        let history = [entry(1500, largest(2, channel: 76)), entry(1900, largest(3, channel: 76))]
        let reading = try XCTUnwrap(Jump.read(largest(40, channel: 76), history: history, trainerStep: 2000))
        let jumped = try XCTUnwrap(reading.jumped.first)
        XCTAssertEqual(jumped.baseline, 2)
        XCTAssertTrue(jumped.baselineIsExact)
    }

    /// B-silu's live lines from `TrainingHealthIncident-Bsilu.log` (trainer
    /// steps 18,800–19,950, every line's largest channel): offline, the
    /// first jump is at 19,900 (851.4× against 61.3×, the smallest largest
    /// ratio of the lines in 18,900–19,850); 154.0× at 19,800 and 603.8× at
    /// 19,850 stay under ten times their lookback's smallest bound.
    func testBSiluLiveLinesJumpFirstAt19900Offline() throws {
        let lines: [(Int, Double, Int)] = [
            (18_800, 61.2, 31), (18_850, 61.4, 31), (18_900, 61.4, 31), (18_950, 61.7, 31), (19_000, 61.9, 31),
            (19_050, 62.1, 31), (19_100, 61.9, 31), (19_150, 62.0, 31), (19_200, 61.6, 31), (19_250, 61.3, 31),
            (19_300, 62.2, 31), (19_350, 62.6, 31), (19_400, 61.9, 31), (19_450, 61.8, 31), (19_500, 62.4, 31),
            (19_550, 63.0, 31), (19_600, 63.4, 31), (19_650, 63.4, 31), (19_700, 62.6, 31), (19_750, 61.7, 31),
            (19_800, 154.0, 76), (19_850, 603.8, 76), (19_900, 851.4, 76), (19_950, 956.7, 76),
        ]
        var history: [Jump.HistoryEntry] = []
        var firstJump: (step: Int, channel: Jump.JumpedChannel)?
        for (step, ratio, channel) in lines {
            let current = largest(ratio, channel: channel)
            if firstJump == nil, let reading = Jump.read(current, history: history, trainerStep: step),
               let jumped = reading.jumped.first {
                firstJump = (step: step, channel: jumped)
            }
            history.append(entry(step, current))
        }
        let jump = try XCTUnwrap(firstJump)
        XCTAssertEqual(jump.step, 19_900)
        XCTAssertEqual(jump.channel.site, "blocks.2.bn1")
        XCTAssertEqual(jump.channel.channel, 76)
        XCTAssertEqual(jump.channel.baseline, 61.3)
        XCTAssertGreaterThanOrEqual(jump.channel.ratio, TrainingHealthThresholds.batchNormRunningVarianceJumpCriticalRatio)
    }
}
