import XCTest
@testable import DrewsChessMachine

/// Rule 14 (`bn_running_variance_jump`) through the real evaluator
/// (BN_RUNNING_VARIANCE_CHANGE_ALARM_PLAN, Part X2): raise, escalate, worsen
/// and clear; the learning gate; trainer-clock rewinds; stale observations;
/// which tiers feed the rule; stop actions. The learning gate of
/// `TrainingHealthTestSupport.config()` is 2,000 (warmup 1,000 + grace 1,000).
final class TrainingHealthRunningVarianceJumpRuleTests: XCTestCase {

    typealias S = TrainingHealthTestSupport
    private let rule = TrainingHealthRule.batchNormRunningVarianceJump

    /// A live digest whose one BN site holds `ratios` (the outlier count is
    /// the profile's own).
    private func digest(_ ratios: [Double?]) -> LayerHealthDigest {
        let profile = BatchNormRunningVarianceProfile(sites: [
            BatchNormRunningVarianceProfile.Site(site: "blocks.2.bn1", channelCount: ratios.count, ratios: ratios),
        ])
        return LayerHealthDigest(
            tier: .live, deadChannels: nil, nonFiniteValueCount: 0, runningVariance: nil, valueFC1: nil,
            runningVarianceChannels: LayerHealthDigest.RunningVarianceChannels(profile: profile))
    }

    /// Eight channels: channel 0 at `first`, the rest at 1.
    private func digest(first: Double) -> LayerHealthDigest {
        digest([first] + [Double?](repeating: 1, count: 7))
    }

    @discardableResult
    private func evaluate(
        _ evaluator: inout TrainingHealthEvaluator,
        _ digest: LayerHealthDigest?,
        at step: Int,
        config: TrainingHealthConfig
    ) -> TrainingHealthEvaluation {
        evaluator.evaluate(S.liveObservation(step: step, digest: digest), config: config)
    }

    private func events(_ evaluation: TrainingHealthEvaluation) -> [TrainingHealthEvent] {
        evaluation.events.filter { $0.rule == rule && $0.kind != .active }
    }

    private func active(_ evaluator: TrainingHealthEvaluator) -> TrainingHealthActiveAlarm? {
        evaluator.activeAlarms.first { $0.rule == rule }
    }

    // MARK: Raise, escalate, clear

    func testRaisesImmediatelyThenEscalatesWhenTheJumpedChannelPasses100x() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        XCTAssertTrue(events(evaluate(&evaluator, digest(first: 0.02), at: 2000, config: config)).isEmpty)

        let raise = try XCTUnwrap(events(evaluate(&evaluator, digest(first: 50), at: 2050, config: config)).first)
        XCTAssertEqual(raise.kind, .raise)
        XCTAssertEqual(raise.severity, .warning)
        XCTAssertEqual(raise.value, "jumped=1 outliers=1/0")
        XCTAssertEqual(raise.threshold, "jump>=10xmin&ratio>=10")
        XCTAssertEqual(raise.detail, "channels=blocks.2.bn1[0]:0.02->50.0")

        let escalate = try XCTUnwrap(events(evaluate(&evaluator, digest(first: 150), at: 2100, config: config)).first)
        XCTAssertEqual(escalate.kind, .escalate)
        XCTAssertEqual(escalate.severity, .critical)
        XCTAssertEqual(escalate.since, 2050)
        XCTAssertEqual(escalate.threshold, "jump>=10xmin&ratio>=100")
        XCTAssertEqual(active(evaluator)?.severity, .critical)
    }

    /// The baseline is the lookback minimum, so an alarm raised at 2,050
    /// holds while the 2,000 read (0.02×) is in the lookback and clears on
    /// the second evaluation after it leaves (3,050, 3,100).
    func testClearsOneLookbackAfterTheJumpOnTwoEvaluations() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        evaluate(&evaluator, digest(first: 0.02), at: 2000, config: config)
        XCTAssertEqual(events(evaluate(&evaluator, digest(first: 50), at: 2050, config: config)).map(\.kind), [.raise])
        for step in stride(from: 2100, through: 3000, by: 50) {
            let evaluation = evaluate(&evaluator, digest(first: 50), at: step, config: config)
            XCTAssertTrue(events(evaluation).isEmpty, "step \(step)")
            XCTAssertNotNil(active(evaluator), "step \(step)")
        }
        XCTAssertTrue(events(evaluate(&evaluator, digest(first: 50), at: 3050, config: config)).isEmpty,
                      "the first evaluation without either arm only starts the clear")
        let clear = try XCTUnwrap(events(evaluate(&evaluator, digest(first: 50), at: 3100, config: config)).first)
        XCTAssertEqual(clear.kind, .clear)
        XCTAssertEqual(clear.since, 2050)
        XCTAssertNil(active(evaluator))
    }

    /// The outlier-count arm alone: six channels already at 12× and six at
    /// 2× (a 6× rise, not a jump) going to twelve at 12×.
    func testOutlierCountArmRaisesAWarningWithoutAJump() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        let base = [Double?](repeating: 12, count: 6) + [Double?](repeating: 2, count: 6) + [1, 1, 1, 1]
        let risen = [Double?](repeating: 12, count: 12) + [1, 1, 1, 1]
        evaluate(&evaluator, digest(base), at: 2000, config: config)
        let raise = try XCTUnwrap(events(evaluate(&evaluator, digest(risen), at: 2050, config: config)).first)
        XCTAssertEqual(raise.kind, .raise)
        XCTAssertEqual(raise.severity, .warning)
        XCTAssertEqual(raise.value, "jumped=0 outliers=12/6")
        XCTAssertEqual(raise.threshold, "outliers>=2xmin&+5")
        XCTAssertEqual(raise.detail, "")
    }

    /// `worsen` follows the outlier count, at most once per check interval.
    func testWorsensOnARisingOutlierCountOncePerCheckInterval() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        func outliers(_ count: Int) -> LayerHealthDigest {
            digest([Double?](repeating: 12, count: count) + [Double?](repeating: 1, count: 16 - count))
        }
        evaluate(&evaluator, outliers(0), at: 2000, config: config)
        let raise = try XCTUnwrap(events(evaluate(&evaluator, outliers(6), at: 2050, config: config)).first)
        XCTAssertEqual(raise.kind, .raise)
        XCTAssertEqual(raise.threshold, "jump>=10xmin&ratio>=10|outliers>=2xmin&+5")
        let worsen = try XCTUnwrap(events(evaluate(&evaluator, outliers(8), at: 2100, config: config)).first)
        XCTAssertEqual(worsen.kind, .worsen)
        XCTAssertTrue(worsen.detail.hasPrefix("was=6 "), worsen.detail)
        XCTAssertTrue(events(evaluate(&evaluator, outliers(10), at: 2150, config: config)).isEmpty,
                      "one worsen line per check interval")
        let next = try XCTUnwrap(events(evaluate(&evaluator, outliers(12), at: 3000, config: config)).first)
        XCTAssertEqual(next.kind, .worsen)
        XCTAssertTrue(next.detail.hasPrefix("was=10 "), next.detail)
    }

    // MARK: Gate

    func testReadsBeforeTheLearningGateAreNeitherJudgedNorStored() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        let early = evaluate(&evaluator, digest(first: 0.02), at: 1950, config: config)
        XCTAssertFalse(early.noDataRules.contains(rule))
        XCTAssertNil(early.runningVarianceJump?.reading)
        XCTAssertEqual(early.runningVarianceJump?.outlierCount, 0)
        XCTAssertTrue(evaluator.state.runningVarianceJumpHistory.isEmpty, "a pre-gate read is not stored")
        // The first read at the gate has no baseline (the 1,950 read was not
        // stored), so 50× after 0.02× does not raise.
        let atGate = evaluate(&evaluator, digest(first: 50), at: 2000, config: config)
        XCTAssertTrue(events(atGate).isEmpty)
        XCTAssertNotNil(atGate.runningVarianceJump)
        XCTAssertNil(atGate.runningVarianceJump?.reading, "baseline=none")
        XCTAssertTrue(events(evaluate(&evaluator, digest(first: 50), at: 2050, config: config)).isEmpty)
        XCTAssertEqual(evaluator.assessRunningVarianceJump(
            evaluator.runningVarianceJumpInput(S.liveObservation(step: 1900, digest: digest(first: 1)), config: config)).holdValue,
            "gate=2000")
    }

    // MARK: Settle window (owner decision 2026-10-07)

    /// Reads before the process has trained 100 steps (a launch, resume or
    /// segment start) only build the baseline: stored, held, never judged.
    func testReadsInTheFirst100StepsOfAProcessOnlyBuildTheBaseline() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        func observe(_ first: Double, at step: Int, trained: Int) -> TrainingHealthEvaluation {
            evaluator.evaluate(
                S.liveObservation(step: step, digest: digest(first: first), stepsTrained: trained), config: config)
        }
        XCTAssertEqual(TrainingHealthThresholds.batchNormRunningVarianceJumpSettleSteps, 100)
        XCTAssertTrue(events(observe(0.02, at: 18_050, trained: 50)).isEmpty)
        let settling = observe(50, at: 18_099, trained: 99)
        XCTAssertTrue(events(settling).isEmpty, "a jump within the first 100 steps is not judged")
        XCTAssertNil(settling.runningVarianceJump?.reading)
        XCTAssertEqual(evaluator.state.runningVarianceJumpHistory.map(\.trainerStep), [18_050, 18_099],
                       "settling reads are stored as the baseline")
        XCTAssertEqual(
            evaluator.assessRunningVarianceJump(evaluator.runningVarianceJumpInput(
                S.liveObservation(step: 18_099, digest: digest(first: 50), stepsTrained: 99), config: config)).holdValue,
            "settling=99/100")
        // From 100 steps trained on, judged against the settling reads.
        let judged = try XCTUnwrap(events(observe(50, at: 18_100, trained: 100)).first)
        XCTAssertEqual(judged.kind, .raise)
        XCTAssertEqual(judged.detail, "channels=blocks.2.bn1[0]:0.02->50.0")
    }

    // MARK: Rewind

    func testRewindClearsTheLookbackButKeepsTheActiveAlarm() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        evaluate(&evaluator, digest(first: 0.02), at: 2000, config: config)
        XCTAssertEqual(events(evaluate(&evaluator, digest(first: 50), at: 2050, config: config)).map(\.kind), [.raise])
        evaluator.resetForTrainerClockRewind()
        XCTAssertTrue(evaluator.state.runningVarianceJumpHistory.isEmpty)
        // 500× after a rewind has no baseline: held, and the alarm stays.
        let afterRewind = evaluate(&evaluator, digest(first: 500), at: 2100, config: config)
        XCTAssertTrue(events(afterRewind).isEmpty)
        XCTAssertNil(afterRewind.runningVarianceJump?.reading)
        XCTAssertNotNil(active(evaluator))
        // It clears against post-rewind reads only.
        XCTAssertTrue(events(evaluate(&evaluator, digest(first: 500), at: 2150, config: config)).isEmpty)
        XCTAssertEqual(events(evaluate(&evaluator, digest(first: 500), at: 2200, config: config)).map(\.kind), [.clear])
    }

    // MARK: Stale, tiers, no data

    func testAStaleObservationIsNeitherJudgedNorStored() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        evaluate(&evaluator, digest(first: 50), at: 2000, config: config)
        evaluate(&evaluator, digest(first: 50), at: 2100, config: config)
        let stale = evaluate(&evaluator, digest(first: 0.02), at: 2050, config: config)
        XCTAssertTrue(stale.staleRules.contains(rule))
        XCTAssertNil(stale.runningVarianceJump)
        XCTAssertEqual(evaluator.state.runningVarianceJumpHistory.map(\.trainerStep), [2000, 2100])
        XCTAssertTrue(events(evaluate(&evaluator, digest(first: 50), at: 2150, config: config)).isEmpty,
                      "the stale 0.02× read would have been a baseline for a jump")
    }

    func testCheckpointAndValueFC1DigestsNeverFeedTheRule() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        evaluate(&evaluator, digest(first: 50), at: 2000, config: config)
        let checkpointDigest = LayerHealthDigest(
            tier: .checkpoint, deadChannels: nil, nonFiniteValueCount: 0, runningVariance: nil, valueFC1: nil,
            runningVarianceChannels: digest(first: 0.02).runningVarianceChannels)
        let checkpoint = evaluator.evaluate(S.checkpointObservation(step: 2010, digest: checkpointDigest), config: config)
        XCTAssertFalse(checkpoint.noDataRules.contains(rule))
        XCTAssertNil(checkpoint.runningVarianceJump)
        let valueFC1 = evaluator.evaluate(S.checkpointObservation(step: 2020, digest: S.valueFC1Digest(zero: 0)), config: config)
        XCTAssertFalse(valueFC1.noDataRules.contains(rule))
        XCTAssertEqual(evaluator.state.runningVarianceJumpHistory.map(\.trainerStep), [2000])
        XCTAssertTrue(events(evaluate(&evaluator, digest(first: 50), at: 2050, config: config)).isEmpty)
        XCTAssertFalse(LayerHealthDigest.Tier.checkpoint.feeds(rule))
        XCTAssertFalse(LayerHealthDigest.Tier.valueFC1Read.feeds(rule))
        XCTAssertTrue(LayerHealthDigest.Tier.live.feeds(rule))
    }

    func testALiveReadWithoutPerChannelRatiosIsNoData() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        let withoutChannels = evaluate(&evaluator, S.deadDigest(dead: 0), at: 2000, config: config)
        XCTAssertTrue(withoutChannels.noDataRules.contains(rule))
        let failedRead = evaluate(&evaluator, nil, at: 2050, config: config)
        XCTAssertTrue(failedRead.noDataRules.contains(rule))
        XCTAssertTrue(evaluator.state.runningVarianceJumpHistory.isEmpty)
    }

    // MARK: Stop actions

    func testStopOnCriticalStopsOnTheCriticalJumpOnly() throws {
        var actions = TrainingHealthActions { _ in .log }
        actions[rule] = .stopOnCritical
        let config = try S.config(actions: actions)
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        evaluate(&evaluator, digest(first: 0.02), at: 2000, config: config)
        XCTAssertNil(evaluate(&evaluator, digest(first: 50), at: 2050, config: config).stopRequest, "warning")
        let stop = try XCTUnwrap(evaluate(&evaluator, digest(first: 150), at: 2100, config: config).stopRequest)
        XCTAssertEqual(stop.rule, rule)
        XCTAssertEqual(stop.severity, .critical)
    }

    func testStopOnAnyStopsOnEitherArm() throws {
        var actions = TrainingHealthActions { _ in .log }
        actions[rule] = .stopOnAny
        let config = try S.config(actions: actions)

        var jump = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        evaluate(&jump, digest(first: 0.02), at: 2000, config: config)
        XCTAssertEqual(evaluate(&jump, digest(first: 50), at: 2050, config: config).stopRequest?.rule, rule)

        var count = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        let base = [Double?](repeating: 12, count: 6) + [Double?](repeating: 2, count: 6) + [1, 1, 1, 1]
        evaluate(&count, digest(base), at: 2000, config: config)
        let stop = try XCTUnwrap(evaluate(
            &count, digest([Double?](repeating: 12, count: 12) + [1, 1, 1, 1]), at: 2050, config: config).stopRequest)
        XCTAssertEqual(stop.rule, rule)
        XCTAssertEqual(stop.severity, .warning)
    }
}

private extension TrainingHealthEvaluator.Assessment {
    /// The rendered value of a `hold`; nil for any other assessment.
    var holdValue: String? {
        if case .hold(let value, _) = self { return value }
        return nil
    }
}
