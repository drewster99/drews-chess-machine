import XCTest
@testable import DrewsChessMachine

/// The pure evaluator (`TrainingHealthEvaluator`) on synthetic observations:
/// every rule's raise / no-raise / clear, sustain, gates, reminders, worsen,
/// stop policy and determinism (the alarms plan, Part R and X1).
final class TrainingHealthEvaluatorTests: XCTestCase {

    typealias S = TrainingHealthTestSupport

    private func evaluator(_ applicability: TrainingHealthValueFC1Applicability = .applies) -> TrainingHealthEvaluator {
        TrainingHealthEvaluator(valueFC1Applicability: applicability)
    }

    private func events(_ result: TrainingHealthEvaluation, _ rule: TrainingHealthRule) -> [TrainingHealthEvent] {
        result.events.filter { $0.rule == rule }
    }

    private func kinds(_ result: TrainingHealthEvaluation, _ rule: TrainingHealthRule) -> [TrainingHealthEvent.Kind] {
        events(result, rule).map(\.kind)
    }

    /// A live observation whose single record carries `illegal`.
    private func illegalWindow(_ step: Int, _ illegal: Float) -> TrainingHealthObservation {
        S.liveObservation(step: step, records: [S.record(step, illegal: illegal)])
    }

    private func gradientWindow(_ step: Int, _ gradient: Float) -> TrainingHealthObservation {
        S.liveObservation(step: step, records: [S.record(step, gradient: gradient)])
    }

    private func offsetWindow(_ step: Int, _ offset: Float?) -> TrainingHealthObservation {
        S.liveObservation(step: step, records: [S.record(step, offset: offset)])
    }

    // MARK: Rule 1 — non_finite

    func testNonFiniteRaisesAtThreshold() throws {
        var e = evaluator()
        let digest = LayerHealthDigest(tier: .live, deadChannels: nil, nonFiniteValueCount: 1, runningVariance: nil, valueFC1: nil)
        let result = e.evaluate(S.liveObservation(step: 100, digest: digest), config: try S.config())
        XCTAssertEqual(events(result, .nonFinite).map(\.kind), [.raise])
        XCTAssertEqual(events(result, .nonFinite).first?.severity, .critical)
    }

    func testNonFiniteDoesNotRaiseJustBelow() throws {
        var e = evaluator()
        let digest = LayerHealthDigest(tier: .live, deadChannels: nil, nonFiniteValueCount: 0, runningVariance: nil, valueFC1: nil)
        let result = e.evaluate(S.liveObservation(step: 100, digest: digest), config: try S.config())
        XCTAssertTrue(events(result, .nonFinite).isEmpty)
    }

    func testNonFinitePolicyLogitMeanInWindowRaises() throws {
        var e = evaluator()
        let result = e.evaluate(offsetWindow(100, .nan), config: try S.config())
        XCTAssertEqual(kinds(result, .nonFinite), [.raise])
    }

    func testNonFiniteNeverClears() throws {
        var e = evaluator()
        let config = try S.config()
        let bad = LayerHealthDigest(tier: .live, deadChannels: nil, nonFiniteValueCount: 3, runningVariance: nil, valueFC1: nil)
        let good = LayerHealthDigest(tier: .live, deadChannels: nil, nonFiniteValueCount: 0, runningVariance: nil, valueFC1: nil)
        _ = e.evaluate(S.liveObservation(step: 100, digest: bad), config: config)
        for step in stride(from: 150, through: 600, by: 50) {
            let result = e.evaluate(S.liveObservation(step: step, digest: good), config: config)
            XCTAssertFalse(kinds(result, .nonFinite).contains(.clear))
        }
        XCTAssertEqual(e.activeAlarms.map(\.rule), [.nonFinite])
    }

    // MARK: Rule 2 — dead_channels

    func testDeadChannelsWarnsOnAnyDeadChannel() throws {
        var e = evaluator()
        let result = e.evaluate(
            S.liveObservation(step: 100, digest: S.deadDigest(dead: 1, sites: [("blocks.0.bn1", 1, 128)])),
            config: try S.config())
        let raise = try XCTUnwrap(events(result, .deadChannels).first)
        XCTAssertEqual(raise.kind, .raise)
        XCTAssertEqual(raise.severity, .warning)
        XCTAssertEqual(raise.threshold, "dead>0")
        XCTAssertEqual(raise.value, "dead=1/1040")
    }

    func testDeadChannelsDoesNotRaiseJustBelow() throws {
        var e = evaluator()
        let result = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 0)), config: try S.config())
        XCTAssertTrue(events(result, .deadChannels).isEmpty)
    }

    func testDeadChannelsCriticalOverallFivePercent() throws {
        let config = try S.config()
        var below = evaluator()
        let warning = below.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 49, channels: 1000)), config: config)
        XCTAssertEqual(events(warning, .deadChannels).first?.severity, .warning)
        var at = evaluator()
        let critical = at.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 50, channels: 1000)), config: config)
        XCTAssertEqual(events(critical, .deadChannels).first?.severity, .critical)
        XCTAssertEqual(events(critical, .deadChannels).first?.threshold, "overall>=0.05")
    }

    func testDeadChannelsCriticalSiteTwentyPercent() throws {
        let config = try S.config()
        var below = evaluator()
        let warning = below.evaluate(
            S.liveObservation(step: 100, digest: S.deadDigest(dead: 3, sites: [("value.bn", 3, 16)])), config: config)
        XCTAssertEqual(events(warning, .deadChannels).first?.severity, .warning, "3/16 = 18.75% is below 20%")
        var at = evaluator()
        let critical = at.evaluate(
            S.liveObservation(step: 100, digest: S.deadDigest(dead: 2, sites: [("policy.pre_bn", 2, 10)])), config: config)
        XCTAssertEqual(events(critical, .deadChannels).first?.severity, .critical)
        XCTAssertEqual(events(critical, .deadChannels).first?.threshold, "site>=0.2")
    }

    func testDeadChannelsNamesEveryAffectedSite() throws {
        var e = evaluator()
        let digest = S.deadDigest(dead: 9, sites: [
            ("blocks.0.bn1", 2, 128), ("value.bn", 5, 16), ("blocks.2.bn2", 0, 128), ("policy.pre_bn", 2, 128),
        ])
        let result = e.evaluate(S.liveObservation(step: 300, digest: digest), config: try S.config())
        let raise = try XCTUnwrap(events(result, .deadChannels).first)
        XCTAssertEqual(raise.detail, "sites=value.bn(5/16),blocks.0.bn1(2/128),policy.pre_bn(2/128)")
        XCTAssertEqual(e.activeAlarms.first?.detail, raise.detail)
    }

    func testDeadChannelsClearsOnlyAfterClearSustain() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 1)), config: config)
        let first = e.evaluate(S.liveObservation(step: 150, digest: S.deadDigest(dead: 0)), config: config)
        XCTAssertTrue(events(first, .deadChannels).isEmpty)
        let second = e.evaluate(S.liveObservation(step: 200, digest: S.deadDigest(dead: 0)), config: config)
        XCTAssertEqual(kinds(second, .deadChannels), [.clear])
    }

    func testDeadChannelsEscalatesWarningToCritical() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 1)), config: config)
        let result = e.evaluate(S.liveObservation(step: 150, digest: S.deadDigest(dead: 60)), config: config)
        XCTAssertEqual(kinds(result, .deadChannels), [.escalate])
        XCTAssertEqual(e.activeAlarms.first?.severity, .critical)
        XCTAssertEqual(e.activeAlarms.first?.since, 100)
    }

    func testDeadChannelsNeverDeescalates() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 60)), config: config)
        let result = e.evaluate(S.liveObservation(step: 150, digest: S.deadDigest(dead: 1)), config: config)
        XCTAssertTrue(events(result, .deadChannels).isEmpty)
        XCTAssertEqual(e.activeAlarms.first?.severity, .critical)
    }

    func testDeadChannelsCriticalNeverClearsWhileDeadRemains() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 60)), config: config)
        for step in stride(from: 150, through: 2000, by: 50) {
            let result = e.evaluate(S.liveObservation(step: step, digest: S.deadDigest(dead: 1)), config: config)
            XCTAssertFalse(kinds(result, .deadChannels).contains(.clear))
        }
        XCTAssertEqual(e.activeAlarms.first?.severity, .critical)
    }

    func testDeadChannelsWithoutAnyActivatedSiteDoesNotApply() throws {
        var e = evaluator()
        let digest = LayerHealthDigest(
            tier: .live,
            deadChannels: LayerHealthDigest.DeadChannels(
                modeledSiteCount: 0, modeledChannelCount: 0, parkedChannelCount: 0, sites: [],
                coversEveryActivation: true),
            nonFiniteValueCount: 0, runningVariance: nil, valueFC1: nil)
        let result = e.evaluate(S.liveObservation(step: 100, digest: digest), config: try S.config())
        XCTAssertEqual(result.newlyNotApplicable.map(\.rule), [.deadChannels])
        XCTAssertTrue(events(result, .deadChannels).isEmpty)
        let again = e.evaluate(S.liveObservation(step: 150, digest: digest), config: try S.config())
        XCTAssertTrue(again.newlyNotApplicable.isEmpty, "said once")
    }

    // MARK: Rule 3 — value_fc1_zero_velocity

    func testValueFC1RaisesAtThreshold() throws {
        var e = evaluator()
        let result = e.evaluate(S.checkpointObservation(step: 1000, digest: S.valueFC1Digest(zero: 7)), config: try S.config())
        XCTAssertEqual(kinds(result, .valueFC1ZeroVelocity), [.raise])
        XCTAssertEqual(events(result, .valueFC1ZeroVelocity).first?.severity, .warning)
    }

    func testValueFC1DoesNotRaiseJustBelow() throws {
        var e = evaluator()
        let result = e.evaluate(S.checkpointObservation(step: 1000, digest: S.valueFC1Digest(zero: 6)), config: try S.config())
        XCTAssertTrue(events(result, .valueFC1ZeroVelocity).isEmpty, "6/128 = 4.7% is below 5%")
    }

    func testValueFC1ClearsOnlyAfterClearSustain() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.checkpointObservation(step: 1000, digest: S.valueFC1Digest(zero: 7)), config: config)
        let hold = e.evaluate(S.checkpointObservation(step: 2000, digest: S.valueFC1Digest(zero: 4)), config: config)
        XCTAssertTrue(events(hold, .valueFC1ZeroVelocity).isEmpty, "4/128 = 3.1% is in the hysteresis band")
        let clear = e.evaluate(S.checkpointObservation(step: 3000, digest: S.valueFC1Digest(zero: 3)), config: config)
        XCTAssertEqual(kinds(clear, .valueFC1ZeroVelocity), [.clear])
    }

    func testValueFC1EscalatesWarningToCritical() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.checkpointObservation(step: 1000, digest: S.valueFC1Digest(zero: 10)), config: config)
        let result = e.evaluate(S.checkpointObservation(step: 2000, digest: S.valueFC1Digest(zero: 64)), config: config)
        XCTAssertEqual(kinds(result, .valueFC1ZeroVelocity), [.escalate])
    }

    func testValueFC1NeverDeescalates() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.checkpointObservation(step: 1000, digest: S.valueFC1Digest(zero: 128)), config: config)
        let result = e.evaluate(S.checkpointObservation(step: 2000, digest: S.valueFC1Digest(zero: 10)), config: config)
        XCTAssertTrue(events(result, .valueFC1ZeroVelocity).isEmpty)
        XCTAssertEqual(e.activeAlarms.first?.severity, .critical)
    }

    func testValueFC1RuleNeedsStepsTrainedByThisProcess() throws {
        var e = evaluator()
        let config = try S.config()
        let untrained = e.evaluate(
            S.checkpointObservation(step: 274, digest: S.valueFC1Digest(zero: 16, units: 16), stepsTrained: 0),
            config: config)
        XCTAssertTrue(events(untrained, .valueFC1ZeroVelocity).isEmpty)
        XCTAssertTrue(untrained.noDataRules.contains(.valueFC1ZeroVelocity))
        let trained = e.evaluate(
            S.checkpointObservation(step: 474, digest: S.valueFC1Digest(zero: 16, units: 16), stepsTrained: 200),
            config: config)
        XCTAssertEqual(kinds(trained, .valueFC1ZeroVelocity), [.raise])
        XCTAssertEqual(events(trained, .valueFC1ZeroVelocity).first?.severity, .critical)
    }

    func testValueFC1RuleNotApplicableForNonReluActivation() throws {
        var e = evaluator(.doesNotApply(activation: .leakyRelu))
        let result = e.evaluate(S.checkpointObservation(step: 1000, digest: S.valueFC1Digest(zero: 128)), config: try S.config())
        XCTAssertTrue(events(result, .valueFC1ZeroVelocity).isEmpty)
        XCTAssertEqual(result.newlyNotApplicable.map(\.rule), [.valueFC1ZeroVelocity])
        XCTAssertFalse(result.noDataRules.contains(.valueFC1ZeroVelocity))
    }

    // MARK: Rule 4 — illegal_mass

    func testIllegalMassRaisesAtThreshold() throws {
        // Arm (i): learned below 0.5, now at 0.8 for 2 evaluations spanning 50.
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(illegalWindow(100, 0.4), config: config)
        XCTAssertTrue(kinds(e.evaluate(illegalWindow(150, 0.8), config: config), .illegalMass).isEmpty)
        let result = e.evaluate(illegalWindow(200, 0.8), config: config)
        XCTAssertEqual(kinds(result, .illegalMass), [.raise])
        XCTAssertEqual(events(result, .illegalMass).first?.severity, .critical)
        XCTAssertEqual(events(result, .illegalMass).first?.threshold, "regression>=0.8")
    }

    func testIllegalMassDoesNotRaiseJustBelow() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(illegalWindow(100, 0.4), config: config)
        _ = e.evaluate(illegalWindow(150, 0.79), config: config)
        let result = e.evaluate(illegalWindow(200, 0.79), config: config)
        XCTAssertTrue(events(result, .illegalMass).isEmpty, "0.79 misses arm (i) and is below 10 x 0.4 for arm (ii)")
    }

    func testIllegalMassRelativeRegressionRaises() throws {
        // Arm (ii), the B-silu shape: learned to 0.0023, then ~0.79.
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(illegalWindow(19000, 0.0023), config: config)
        _ = e.evaluate(illegalWindow(20650, 0.7988), config: config)
        let result = e.evaluate(illegalWindow(20700, 0.7817), config: config)
        XCTAssertEqual(kinds(result, .illegalMass), [.raise])
        XCTAssertEqual(events(result, .illegalMass).first?.threshold, "regression>=0.3&>=10xmin")
    }

    func testIllegalMassRelativeRegressionNeedsBothFloorAndMultiple() throws {
        let config = try S.config()
        var belowFloor = evaluator()
        _ = belowFloor.evaluate(illegalWindow(100, 0.005), config: config)
        _ = belowFloor.evaluate(illegalWindow(150, 0.29), config: config)
        XCTAssertTrue(events(belowFloor.evaluate(illegalWindow(200, 0.29), config: config), .illegalMass).isEmpty)
        var belowMultiple = evaluator()
        _ = belowMultiple.evaluate(illegalWindow(100, 0.04), config: config)
        _ = belowMultiple.evaluate(illegalWindow(150, 0.35), config: config)
        XCTAssertTrue(events(belowMultiple.evaluate(illegalWindow(200, 0.35), config: config), .illegalMass).isEmpty)
    }

    func testIllegalMassClearsOnlyAfterClearSustain() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(illegalWindow(100, 0.004), config: config)
        _ = e.evaluate(illegalWindow(150, 0.8), config: config)
        _ = e.evaluate(illegalWindow(200, 0.8), config: config)
        XCTAssertEqual(e.activeAlarms.map(\.rule), [.illegalMass])
        XCTAssertTrue(events(e.evaluate(illegalWindow(250, 0.16), config: config), .illegalMass).isEmpty)
        XCTAssertTrue(events(e.evaluate(illegalWindow(300, 0.14), config: config), .illegalMass).isEmpty)
        XCTAssertTrue(events(e.evaluate(illegalWindow(320, 0.14), config: config), .illegalMass).isEmpty, "span 20 < 50")
        XCTAssertEqual(kinds(e.evaluate(illegalWindow(350, 0.14), config: config), .illegalMass), [.clear])
    }

    func testLearningGateUsesTrainerClockAndWarmup() throws {
        let config = try S.config(learningGraceSteps: 1000, lrWarmupSteps: 1000)
        var fresh = evaluator()
        _ = fresh.evaluate(illegalWindow(1950, 0.6), config: config)
        XCTAssertTrue(events(fresh.evaluate(illegalWindow(1999, 0.6), config: config), .illegalMass).isEmpty)
        _ = fresh.evaluate(illegalWindow(2000, 0.6), config: config)
        let raised = fresh.evaluate(illegalWindow(2050, 0.6), config: config)
        XCTAssertEqual(kinds(raised, .illegalMass), [.raise])
        XCTAssertEqual(events(raised, .illegalMass).first?.threshold, "notLearned>=0.5")

        var resumed = evaluator()
        _ = resumed.evaluate(illegalWindow(5000, 0.6), config: config)
        XCTAssertEqual(kinds(resumed.evaluate(illegalWindow(5050, 0.6), config: config), .illegalMass), [.raise])
    }

    func testRegressionFormIgnoresTheGate() throws {
        var e = evaluator()
        let config = try S.config(learningGraceSteps: 1000, lrWarmupSteps: 1000)
        _ = e.evaluate(illegalWindow(250, 0.236), config: config)
        _ = e.evaluate(illegalWindow(300, 0.9971), config: config)
        XCTAssertEqual(kinds(e.evaluate(illegalWindow(350, 0.9458), config: config), .illegalMass), [.raise])
    }

    // MARK: Rule 5 — gradient_collapse

    func testGradientCollapseRaisesAtThreshold() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(gradientWindow(350, 0.059), config: config)
        let result = e.evaluate(gradientWindow(400, 0.018), config: config)
        XCTAssertEqual(kinds(result, .gradientCollapse), [.raise])
        XCTAssertEqual(events(result, .gradientCollapse).first?.severity, .critical)
    }

    func testGradientCollapseDoesNotRaiseJustBelow() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(gradientWindow(350, 0.1), config: config)
        XCTAssertTrue(events(e.evaluate(gradientWindow(400, 0.1), config: config), .gradientCollapse).isEmpty)
    }

    func testGradientCollapseClearsOnlyAfterClearSustain() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(gradientWindow(350, 0.05), config: config)
        _ = e.evaluate(gradientWindow(400, 0.05), config: config)
        XCTAssertTrue(events(e.evaluate(gradientWindow(450, 0.2), config: config), .gradientCollapse).isEmpty)
        XCTAssertEqual(kinds(e.evaluate(gradientWindow(500, 0.2), config: config), .gradientCollapse), [.clear])
    }

    // MARK: Rule 6 — loss_spike

    private func lossWindow(_ step: Int, _ losses: [Float]) -> TrainingHealthObservation {
        let records = losses.enumerated().map { S.record(step - losses.count + 1 + $0.offset, loss: $0.element) }
        return S.liveObservation(step: step, records: records, history: S.history(before: step - losses.count + 1))
    }

    func testLossSpikeRaisesAtThreshold() throws {
        let config = try S.config()
        var median = evaluator()
        let medianResult = median.evaluate(lossWindow(1000, [6.0]), config: config)
        XCTAssertEqual(kinds(medianResult, .lossSpike), [.raise])
        XCTAssertEqual(events(medianResult, .lossSpike).first?.threshold, "median>=1.5xref")
        var maximum = evaluator()
        let maxResult = maximum.evaluate(lossWindow(1000, [4.0, 4.0, 12.0]), config: config)
        XCTAssertEqual(kinds(maxResult, .lossSpike), [.raise])
        XCTAssertEqual(events(maxResult, .lossSpike).first?.threshold, "max>=3xref")
    }

    func testLossSpikeDoesNotRaiseJustBelow() throws {
        var e = evaluator()
        let result = e.evaluate(lossWindow(1000, [4.0, 5.99, 11.99]), config: try S.config())
        XCTAssertTrue(events(result, .lossSpike).isEmpty)
    }

    func testLossSpikeClearsOnlyAfterClearSustain() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(lossWindow(1000, [6.0]), config: config)
        XCTAssertTrue(events(e.evaluate(lossWindow(1050, [4.9]), config: config), .lossSpike).isEmpty, "1.225 holds")
        XCTAssertEqual(kinds(e.evaluate(lossWindow(1100, [4.7]), config: config), .lossSpike), [.clear])
    }

    // MARK: Rule 7 — policy_offset_drift

    func testPolicyOffsetDriftRaisesAtThreshold() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(offsetWindow(850, -3.0), config: config)
        let result = e.evaluate(offsetWindow(900, -3.8), config: config)
        XCTAssertEqual(kinds(result, .policyOffsetDrift), [.raise])
        XCTAssertEqual(events(result, .policyOffsetDrift).first?.severity, .warning)
    }

    func testPolicyOffsetDriftDoesNotRaiseJustBelow() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(offsetWindow(850, 2.99), config: config)
        XCTAssertTrue(events(e.evaluate(offsetWindow(900, -2.99), config: config), .policyOffsetDrift).isEmpty)
    }

    func testPolicyOffsetDriftClearsOnlyAfterClearSustain() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(offsetWindow(150, -4.5), config: config)
        _ = e.evaluate(offsetWindow(200, -5.7), config: config)
        XCTAssertTrue(events(e.evaluate(offsetWindow(350, -0.81), config: config), .policyOffsetDrift).isEmpty)
        XCTAssertEqual(kinds(e.evaluate(offsetWindow(400, -0.76), config: config), .policyOffsetDrift), [.clear])
    }

    // MARK: Rule 8 — bn_running_variance_runaway

    func testRunningVarianceRunawayRaisesAtThreshold() throws {
        var e = evaluator()
        let result = e.evaluate(S.liveObservation(step: 200, digest: S.runningVarianceDigest(1000)), config: try S.config())
        XCTAssertEqual(kinds(result, .batchNormRunningVarianceRunaway), [.raise])
    }

    func testRunningVarianceRunawayDoesNotRaiseJustBelow() throws {
        var e = evaluator()
        let result = e.evaluate(S.liveObservation(step: 200, digest: S.runningVarianceDigest(999.9)), config: try S.config())
        XCTAssertTrue(events(result, .batchNormRunningVarianceRunaway).isEmpty)
    }

    func testRunningVarianceRunawayClearsOnlyAfterClearSustain() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 200, digest: S.runningVarianceDigest(1415.5)), config: config)
        XCTAssertTrue(events(e.evaluate(S.liveObservation(step: 250, digest: S.runningVarianceDigest(299)), config: config),
                             .batchNormRunningVarianceRunaway).isEmpty)
        XCTAssertEqual(kinds(e.evaluate(S.liveObservation(step: 300, digest: S.runningVarianceDigest(299)), config: config),
                             .batchNormRunningVarianceRunaway), [.clear])
    }

    // MARK: Clears need two observed states (review M4)

    /// At a save step the live evaluation and the checkpoint evaluation read
    /// the same weights at the same trainer step: two evaluations, one state.
    func testDeadChannelsClearNeedsTwoDistinctTrainerSteps() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 950, digest: S.deadDigest(dead: 60)), config: config)
        let live = e.evaluate(S.liveObservation(step: 1000, digest: S.deadDigest(dead: 0)), config: config)
        XCTAssertTrue(kinds(live, .deadChannels).filter { $0 == .clear }.isEmpty)
        let sameStep = e.evaluate(
            S.checkpointObservation(step: 1000, digest: S.deadDigest(dead: 0, tier: .checkpoint)), config: config)
        XCTAssertTrue(kinds(sameStep, .deadChannels).isEmpty, "a second read of the step-1000 weights is not a second state")
        XCTAssertEqual(e.activeAlarms.map(\.rule), [.deadChannels])
        let next = e.evaluate(S.liveObservation(step: 1050, digest: S.deadDigest(dead: 0)), config: config)
        XCTAssertEqual(kinds(next, .deadChannels), [.clear])
    }

    func testRunningVarianceClearNeedsTwoDistinctTrainerSteps() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 950, digest: S.runningVarianceDigest(1415.5)), config: config)
        _ = e.evaluate(S.liveObservation(step: 1000, digest: S.runningVarianceDigest(80)), config: config)
        let sameStep = e.evaluate(S.checkpointObservation(step: 1000, digest: S.runningVarianceDigest(80)), config: config)
        XCTAssertTrue(kinds(sameStep, .batchNormRunningVarianceRunaway).isEmpty)
        let next = e.evaluate(S.liveObservation(step: 1050, digest: S.runningVarianceDigest(80)), config: config)
        XCTAssertEqual(kinds(next, .batchNormRunningVarianceRunaway), [.clear])
    }

    // MARK: Rule 9 — gradient_spike

    private func spikeWindow(_ step: Int, _ gradient: Float) -> TrainingHealthObservation {
        S.liveObservation(step: step, records: [S.record(step, gradient: gradient)],
                          history: S.history(before: step, gradient: 0.375))
    }

    func testGradientSpikeRaisesAtThreshold() throws {
        var e = evaluator()
        let result = e.evaluate(spikeWindow(20600, 1.875), config: try S.config())
        XCTAssertEqual(kinds(result, .gradientSpike), [.raise])
        XCTAssertEqual(events(result, .gradientSpike).first?.severity, .warning)
        XCTAssertEqual(events(result, .gradientSpike).first?.threshold, "max>=5xref")
    }

    func testGradientSpikeDoesNotRaiseJustBelow() throws {
        var e = evaluator()
        XCTAssertTrue(events(e.evaluate(spikeWindow(20600, 1.874), config: try S.config()), .gradientSpike).isEmpty)
    }

    func testGradientSpikeClearsOnlyAfterClearSustain() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(spikeWindow(20600, 2.566), config: config)
        XCTAssertTrue(events(e.evaluate(spikeWindow(20650, 0.9375), config: config), .gradientSpike).isEmpty, "2.5x holds")
        XCTAssertEqual(kinds(e.evaluate(spikeWindow(20700, 0.937), config: config), .gradientSpike), [.clear])
    }

    func testGradientSpikeNeedsAReference() throws {
        var e = evaluator()
        let observation = S.liveObservation(step: 100, records: [S.record(100, gradient: 50)])
        let result = e.evaluate(observation, config: try S.config())
        XCTAssertTrue(events(result, .gradientSpike).isEmpty)
        XCTAssertTrue(result.noDataRules.contains(.gradientSpike))
    }

    // MARK: Shared mechanics

    func testNoDataHoldsStateAndCountsIt() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(offsetWindow(150, -4.5), config: config)
        _ = e.evaluate(offsetWindow(200, -5.7), config: config)
        XCTAssertEqual(e.activeAlarms.map(\.rule), [.policyOffsetDrift])
        // Clear progress, then windows with no diagnostic step: no clear, no progress.
        _ = e.evaluate(offsetWindow(250, -0.5), config: config)
        for step in stride(from: 300, through: 600, by: 50) {
            let result = e.evaluate(offsetWindow(step, nil), config: config)
            XCTAssertTrue(result.noDataRules.contains(.policyOffsetDrift))
            XCTAssertTrue(events(result, .policyOffsetDrift).isEmpty)
        }
        XCTAssertEqual(e.activeAlarms.map(\.rule), [.policyOffsetDrift])
        // The streak resumes: one more clear evaluation 50+ steps after the first completes it.
        XCTAssertEqual(kinds(e.evaluate(offsetWindow(650, -0.5), config: config), .policyOffsetDrift), [.clear])
    }

    func testSustainNeedsBothCountAndSpan() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(gradientWindow(100, 0.01), config: config)
        XCTAssertTrue(events(e.evaluate(gradientWindow(110, 0.01), config: config), .gradientCollapse).isEmpty)
        XCTAssertEqual(kinds(e.evaluate(gradientWindow(150, 0.01), config: config), .gradientCollapse), [.raise])
    }

    func testReminderEveryCheckInterval() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 300, digest: S.deadDigest(dead: 5)), config: config)
        let before = e.evaluate(S.liveObservation(step: 950, digest: S.deadDigest(dead: 5)), config: config)
        XCTAssertFalse(before.checkDue)
        XCTAssertTrue(kinds(before, .deadChannels).isEmpty)
        let at = e.evaluate(S.liveObservation(step: 1000, digest: S.deadDigest(dead: 5)), config: config)
        XCTAssertTrue(at.checkDue)
        XCTAssertEqual(kinds(at, .deadChannels), [.active])
        XCTAssertEqual(events(at, .deadChannels).first?.since, 300)
        let mid = e.evaluate(S.liveObservation(step: 1500, digest: S.deadDigest(dead: 5)), config: config)
        XCTAssertFalse(mid.checkDue)
        let next = e.evaluate(S.liveObservation(step: 2010, digest: S.deadDigest(dead: 5)), config: config)
        XCTAssertTrue(next.checkDue)
        XCTAssertEqual(kinds(next, .deadChannels), [.active])
    }

    func testCheckpointEvaluationsNeverMakeACheckDue() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 950, digest: S.deadDigest(dead: 0)), config: config)
        let checkpoint = e.evaluate(S.checkpointObservation(step: 1000, digest: S.deadDigest(dead: 0, tier: .checkpoint)), config: config)
        XCTAssertFalse(checkpoint.checkDue)
        XCTAssertTrue(e.evaluate(S.liveObservation(step: 1000, digest: S.deadDigest(dead: 0)), config: config).checkDue)
    }

    func testWorsenRateLimitedToOncePerInterval() throws {
        var e = evaluator()
        let config = try S.config()
        _ = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 5)), config: config)
        let first = e.evaluate(S.liveObservation(step: 150, digest: S.deadDigest(dead: 6, sites: [("blocks.0.bn1", 6, 128)])), config: config)
        XCTAssertEqual(kinds(first, .deadChannels), [.worsen])
        XCTAssertEqual(events(first, .deadChannels).first?.detail, "was=5 sites=blocks.0.bn1(6/128)")
        let limited = e.evaluate(S.liveObservation(step: 200, digest: S.deadDigest(dead: 7)), config: config)
        XCTAssertTrue(kinds(limited, .deadChannels).isEmpty)
        let nextInterval = e.evaluate(S.liveObservation(step: 1000, digest: S.deadDigest(dead: 9)), config: config)
        XCTAssertEqual(kinds(nextInterval, .deadChannels), [.worsen, .active])
        XCTAssertEqual(events(nextInterval, .deadChannels).first?.detail.hasPrefix("was=7"), true)
    }

    // MARK: Stop policy

    func testActionLogNeverRequestsStop() throws {
        var e = evaluator()
        let result = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 500)), config: try S.config())
        XCTAssertNil(result.stopRequest)
        XCTAssertFalse(e.stopRequested)
    }

    func testStopOnCriticalIgnoresWarnings() throws {
        var e = evaluator()
        let config = try S.config(actions: TrainingHealthActions { _ in .stopOnCritical })
        let warning = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 1)), config: config)
        XCTAssertNil(warning.stopRequest)
        let critical = e.evaluate(S.liveObservation(step: 150, digest: S.deadDigest(dead: 100)), config: config)
        XCTAssertEqual(critical.stopRequest?.rule, .deadChannels)
        XCTAssertEqual(critical.stopRequest?.kind, .stop)
        XCTAssertEqual(kinds(critical, .deadChannels), [.escalate, .stop])
    }

    func testStopOnCriticalNeverFiresForRulesWithoutCritical() throws {
        var e = evaluator()
        let config = try S.config(actions: TrainingHealthActions { _ in .stopOnCritical })
        _ = e.evaluate(offsetWindow(150, -4.5), config: config)
        let raised = e.evaluate(offsetWindow(200, -5.7), config: config)
        XCTAssertEqual(kinds(raised, .policyOffsetDrift), [.raise])
        XCTAssertNil(raised.stopRequest)
        for rule in TrainingHealthRule.allCases where !rule.hasCriticalLevel {
            XCTAssertFalse(TrainingHealthStopPolicy.qualifies(severity: .warning, action: .stopOnCritical), "\(rule)")
        }
    }

    func testStopRequestedOnce() throws {
        var e = evaluator()
        let config = try S.config(actions: TrainingHealthActions { _ in .stopOnAny })
        let first = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 1)), config: config)
        XCTAssertNotNil(first.stopRequest)
        let second = e.evaluate(S.liveObservation(step: 150, digest: S.deadDigest(dead: 100)), config: config)
        XCTAssertNil(second.stopRequest)
        XCTAssertFalse(second.events.contains { $0.kind == .stop })
        XCTAssertEqual(kinds(second, .deadChannels), [.escalate], "later events are still logged")
    }

    func testStopFollowsCurrentActionForAnAlreadyActiveAlarm() throws {
        var e = evaluator()
        _ = e.evaluate(S.liveObservation(step: 100, digest: S.deadDigest(dead: 100)), config: try S.config())
        var actions = TrainingHealthActions { _ in .log }
        actions[.deadChannels] = .stopOnCritical
        let result = e.evaluate(S.liveObservation(step: 150, digest: S.deadDigest(dead: 100)), config: try S.config(actions: actions))
        XCTAssertEqual(result.stopRequest?.rule, .deadChannels)
        XCTAssertEqual(result.stopRequest?.action, .stopOnCritical)
    }

    private func active(_ rule: TrainingHealthRule, _ severity: TrainingAlarm.Severity) -> TrainingHealthActiveAlarm {
        TrainingHealthActiveAlarm(rule: rule, severity: severity, since: 100, value: "", detail: "", action: .log)
    }

    func testFirstQualifyingFollowsRuleOrder() {
        let actions = TrainingHealthActions { _ in .stopOnAny }
        let chosen = TrainingHealthStopPolicy.firstQualifying(
            active: [active(.gradientSpike, .warning), active(.illegalMass, .critical), active(.deadChannels, .warning)],
            actions: actions)
        XCTAssertEqual(chosen?.rule, .deadChannels)
    }

    func testActionChangedToLogMeansNoStop() {
        // A detached checkpoint pass evaluated under stop_on_critical; the
        // owner set the rule to log before the result was delivered.
        let current = TrainingHealthActions { _ in .log }
        XCTAssertNil(TrainingHealthStopPolicy.firstQualifying(active: [active(.deadChannels, .critical)], actions: current))
    }

    func testActionChangedToStopOnAnyStopsAnAlreadyActiveWarning() {
        // Evaluated under log; delivered after the owner chose stop_on_any.
        var current = TrainingHealthActions { _ in .log }
        current[.policyOffsetDrift] = .stopOnAny
        let chosen = TrainingHealthStopPolicy.firstQualifying(
            active: [active(.policyOffsetDrift, .warning)], actions: current)
        XCTAssertEqual(chosen?.rule, .policyOffsetDrift)
    }

    // MARK: Disabled, determinism, config

    func testDisabledConfigEmitsNothing() throws {
        var e = evaluator()
        let config = try S.config(enabled: false, actions: TrainingHealthActions { _ in .stopOnAny })
        let result = e.evaluate(S.liveObservation(step: 1000, digest: S.deadDigest(dead: 500)), config: config)
        XCTAssertTrue(result.events.isEmpty)
        XCTAssertNil(result.stopRequest)
        XCTAssertFalse(result.checkDue)
        XCTAssertTrue(e.activeAlarms.isEmpty)
    }

    func testEventOrderIsDeterministic() throws {
        var e = evaluator()
        let digest = LayerHealthDigest(
            tier: .live,
            deadChannels: LayerHealthDigest.DeadChannels(
                modeledSiteCount: 9, modeledChannelCount: 1040, parkedChannelCount: 300, sites: [],
                coversEveryActivation: true),
            nonFiniteValueCount: 2,
            runningVariance: LayerHealthDigest.RunningVarianceRunaway(maxOverMedian: 5000, site: "value.bn"),
            valueFC1: nil)
        let config = try S.config(actions: TrainingHealthActions { _ in .stopOnAny })
        let result = e.evaluate(S.liveObservation(step: 1000, history: S.history(before: 1000), digest: digest), config: config)
        let order = result.events.map { "\($0.rule.rawValue):\($0.kind.rawValue)" }
        XCTAssertEqual(order, [
            "non_finite:raise", "non_finite:stop", "dead_channels:raise", "bn_running_variance_runaway:raise",
        ])
        var again = evaluator()
        let repeated = again.evaluate(S.liveObservation(step: 1000, history: S.history(before: 1000), digest: digest), config: config)
        XCTAssertEqual(repeated.events, result.events)
    }

    func testConfigRefusesAnUnusableInterval() {
        XCTAssertThrowsError(try S.config(checkIntervalSteps: 0))
        XCTAssertThrowsError(try S.config(learningGraceSteps: -1))
    }

    func testActionNamesRoundTripAndEncodeByName() throws {
        for action in TrainingHealthAction.allCases {
            XCTAssertEqual(try TrainingHealthAction(name: action.name), action)
        }
        XCTAssertThrowsError(try TrainingHealthAction(name: "stop"))
        let data = try JSONEncoder().encode(TrainingHealthAction.stopOnCritical)
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "\"stop_on_critical\"")
        XCTAssertEqual(try JSONDecoder().decode(TrainingHealthAction.self, from: data), .stopOnCritical)
    }

    func testSeverityEncodesByRawValue() throws {
        XCTAssertEqual(String(decoding: try JSONEncoder().encode(TrainingAlarm.Severity.warning), as: UTF8.self), "\"warning\"")
        XCTAssertEqual(String(decoding: try JSONEncoder().encode(TrainingAlarm.Severity.critical), as: UTF8.self), "\"critical\"")
    }

    func testConfigEncodesSnakeCaseWithActionsByName() throws {
        var actions = TrainingHealthActions { _ in .log }
        actions[.illegalMass] = .stopOnCritical
        let config = try S.config(actions: actions)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: try encoder.encode(config)) as? [String: Any])
        XCTAssertEqual(Set(object.keys), [
            "enabled", "check_interval_steps", "learning_grace_steps", "lr_warmup_steps", "momentum_coefficient", "actions",
        ])
        let encodedActions = try XCTUnwrap(object["actions"] as? [String: String])
        XCTAssertEqual(encodedActions.count, TrainingHealthRule.allCases.count)
        XCTAssertEqual(encodedActions["illegal_mass"], "stop_on_critical")
        XCTAssertEqual(encodedActions["gradient_spike"], "log")
    }

    func testEveryRuleHasExactlyOneAction() {
        var actions = TrainingHealthActions { _ in .log }
        for rule in TrainingHealthRule.allCases {
            actions[rule] = .stopOnAny
            XCTAssertEqual(actions[rule], .stopOnAny)
        }
        XCTAssertEqual(Set(TrainingHealthRule.allCases.map(\.ruleOrder)).count, TrainingHealthRule.allCases.count)
        XCTAssertEqual(TrainingHealthRule.allCases.map(\.ruleOrder), Array(0..<TrainingHealthRule.allCases.count))
    }

    func testLiveEvaluationCadenceIsEveryFiftyTrainerSteps() {
        XCTAssertFalse(TrainingHealthCadence.isLiveEvaluationStep(trainerStep: 0))
        XCTAssertFalse(TrainingHealthCadence.isLiveEvaluationStep(trainerStep: 49))
        XCTAssertTrue(TrainingHealthCadence.isLiveEvaluationStep(trainerStep: 50))
        XCTAssertFalse(TrainingHealthCadence.isLiveEvaluationStep(trainerStep: 513))
        XCTAssertTrue(TrainingHealthCadence.isLiveEvaluationStep(trainerStep: 550))
    }
}
