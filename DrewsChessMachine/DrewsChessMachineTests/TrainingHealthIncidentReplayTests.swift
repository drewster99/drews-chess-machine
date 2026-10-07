import XCTest
@testable import DrewsChessMachine

/// The plan's evidence as executable tests: verbatim excerpts of the
/// incident and healthy-baseline session logs (bundled resources, with their
/// source hashes in each file's header) replayed through the real monitor
/// and evaluator (`TrainingHealthLogReplay`). The expected trainer steps are
/// the prototype's (the alarms plan, Evidence and owner decisions of
/// 2026-10-06); a difference is reported to the owner, never quietly
/// absorbed by changing an expectation.
final class TrainingHealthIncidentReplayTests: XCTestCase {

    typealias S = TrainingHealthTestSupport

    private func replay(_ names: [String]) throws -> TrainingHealthLogReplay.Output {
        let sources = try names.map { name in
            TrainingHealthLogReplay.Source(
                name: name, text: try S.resourceText("TrainingHealthIncident-\(name)", extension: "log"))
        }
        // The runs' own lr_warmup_steps (experiments/20261005-lr-schedule-ab/parameters-*.json) and the defaults.
        let config = try S.config(checkIntervalSteps: 1000, learningGraceSteps: 1000, lrWarmupSteps: 1000)
        return try TrainingHealthLogReplay.run(
            sources, options: TrainingHealthLogReplay.Options(segmentStepAsTrainerStep: false, config: config))
    }

    private func steps(
        _ events: [TrainingHealthEvent],
        _ rule: TrainingHealthRule,
        _ kind: TrainingHealthEvent.Kind
    ) -> [Int] {
        events.filter { $0.rule == rule && $0.kind == kind }.map(\.trainerStep)
    }

    private func describe(_ events: [TrainingHealthEvent]) -> String {
        events.map { "\($0.trainerStep) \($0.rule.rawValue) \($0.kind.rawValue) \($0.severity.rawValue) \($0.value)" }
            .joined(separator: "\n")
    }

    // MARK: Arm C, one process (both segments passed together)

    private func armC() throws -> [TrainingHealthEvent] {
        try replay(["C-seg0", "C-seg1"]).events
    }

    func testArmCRaisesDeadChannelsCriticalAt50() throws {
        let events = try armC()
        let raise = try XCTUnwrap(events.first { $0.rule == .deadChannels && $0.kind == .raise }, describe(events))
        XCTAssertEqual(raise.trainerStep, 50)
        XCTAssertEqual(raise.severity, .critical)
        XCTAssertTrue(raise.detail.hasPrefix("sites=value.bn(12/16)"), raise.detail)
        XCTAssertEqual(steps(events, .deadChannels, .worsen).first, 100, describe(events))
    }

    func testArmCRaisesIllegalMassCriticalAt350() throws {
        let events = try armC()
        XCTAssertEqual(steps(events, .illegalMass, .raise), [350], describe(events))
        XCTAssertEqual(events.first { $0.rule == .illegalMass && $0.kind == .raise }?.severity, .critical)
    }

    func testArmCRaisesGradientCollapseAt400() throws {
        let events = try armC()
        XCTAssertEqual(steps(events, .gradientCollapse, .raise), [400, 1713], describe(events))
        XCTAssertEqual(steps(events, .gradientCollapse, .clear), [913], describe(events))
    }

    func testArmCRaisesValueFC1CriticalAt1513() throws {
        let events = try armC()
        let raise = try XCTUnwrap(events.first { $0.rule == .valueFC1ZeroVelocity && $0.kind == .raise }, describe(events))
        XCTAssertEqual(raise.trainerStep, 1513)
        XCTAssertEqual(raise.severity, .critical)
    }

    func testArmCRaisesLossSpikeAt300() throws {
        let events = try armC()
        XCTAssertEqual(steps(events, .lossSpike, .raise), [300], describe(events))
        XCTAssertEqual(steps(events, .lossSpike, .clear), [500], describe(events))
    }

    func testArmCRaisesGradientSpikeAt300() throws {
        let events = try armC()
        XCTAssertEqual(steps(events, .gradientSpike, .raise).first, 300, describe(events))
    }

    func testArmCPolicyOffsetDriftRaisedAt200ClearedAt400() throws {
        let events = try armC()
        XCTAssertEqual(steps(events, .policyOffsetDrift, .raise), [200], describe(events))
        XCTAssertEqual(steps(events, .policyOffsetDrift, .clear), [400], describe(events))
    }

    func testArmCRaisesRunningVarianceRunawayAt200() throws {
        let events = try armC()
        XCTAssertEqual(steps(events, .batchNormRunningVarianceRunaway, .raise), [200], describe(events))
    }

    func testArmCNoRaiseEscalateOrClearAt250() throws {
        let events = try armC()
        XCTAssertTrue(events.filter { $0.trainerStep == 250 && [.raise, .escalate, .clear].contains($0.kind) }.isEmpty,
                      describe(events))
    }

    func testArmCNeverRaisesNonFinite() throws {
        let events = try armC()
        XCTAssertTrue(events.filter { $0.rule == .nonFinite }.isEmpty, describe(events))
    }

    func testArmCSegmentOneAloneRaisesDeadChannelsAtFirstEvaluation() throws {
        let events = try replay(["C-seg1"]).events
        let dead = try XCTUnwrap(events.first { $0.rule == .deadChannels && $0.kind == .raise }, describe(events))
        XCTAssertEqual(dead.trainerStep, 514)
        XCTAssertEqual(dead.severity, .critical)
        XCTAssertEqual(steps(events, .batchNormRunningVarianceRunaway, .raise), [514], describe(events))
        XCTAssertEqual(steps(events, .valueFC1ZeroVelocity, .raise), [1513], describe(events))
        XCTAssertEqual(steps(events, .gradientCollapse, .raise).first, 1713, describe(events))
        XCTAssertEqual(steps(events, .illegalMass, .raise), [2063], describe(events))
    }

    // MARK: Arm B

    func testArmBRaisesValueBNDeadChannelCriticalAt300() throws {
        let events = try replay(["B"]).events
        let raise = try XCTUnwrap(events.first { $0.rule == .deadChannels && $0.kind == .raise }, describe(events))
        XCTAssertEqual(raise.trainerStep, 300)
        XCTAssertEqual(raise.severity, .critical)
        XCTAssertTrue(raise.detail.hasPrefix("sites=value.bn(5/16)"), raise.detail)
        XCTAssertEqual(steps(events, .deadChannels, .worsen).first, 1300, describe(events))
    }

    func testArmBPolicyOffsetDriftWarningAt900() throws {
        let events = try replay(["B"]).events
        XCTAssertEqual(steps(events, .policyOffsetDrift, .raise), [900], describe(events))
    }

    func testArmBValueFC1WarningAt1000() throws {
        let events = try replay(["B"]).events
        let raise = try XCTUnwrap(events.first { $0.rule == .valueFC1ZeroVelocity && $0.kind == .raise }, describe(events))
        XCTAssertEqual(raise.trainerStep, 1000)
        XCTAssertEqual(raise.severity, .warning)
    }

    func testArmBOnlyDeadChannelsIsCritical() throws {
        let events = try replay(["B"]).events
        let critical = Set(events.filter { $0.severity == .critical }.map(\.rule))
        XCTAssertEqual(critical, [.deadChannels], describe(events))
        XCTAssertTrue(events.filter { $0.rule == .gradientSpike }.isEmpty, describe(events))
    }

    // MARK: Healthy runs

    func testArmARaisesNothing() throws {
        let events = try replay(["A"]).events
        XCTAssertTrue(events.isEmpty, describe(events))
    }

    func testR7RaisesNothing() throws {
        let events = try replay(["R7"]).events
        XCTAssertTrue(events.isEmpty, describe(events))
    }

    func testR8RaisesNothing() throws {
        let events = try replay(["R8"]).events
        XCTAssertTrue(events.isEmpty, describe(events))
    }

    // MARK: B-silu (owner decisions 2026-10-06)

    func testBSiluIllegalMassRegressionRaisesAt20700() throws {
        let events = try replay(["Bsilu"]).events
        let raise = try XCTUnwrap(events.first { $0.rule == .illegalMass && $0.kind == .raise }, describe(events))
        XCTAssertEqual(raise.trainerStep, 20_700)
        XCTAssertEqual(raise.severity, .critical)
        XCTAssertEqual(raise.threshold, "regression>=0.3&>=10xmin")
    }

    func testBSiluGradientSpikeRaisesAt20600() throws {
        let events = try replay(["Bsilu"]).events
        XCTAssertEqual(steps(events, .gradientSpike, .raise).first, 20_600, describe(events))
    }

    /// B-silu's log predates the parked counts, so its live lines count only
    /// the two leaky_relu sites (144 channels). The overall arm's denominator
    /// is still every channel an activation consumes — 1,040, from the run's
    /// checkpoint table — so 20 parked is 1.9% overall; `policy.pre_bn`
    /// (19/128 = 14.8%) and `value.bn` (1/16 = 6%) stay under the site arm.
    /// The app, which sees every site, would raise a warning; dividing by the
    /// 144 relu-family channels alone made it a false critical.
    func testBSiluDeadChannelsIsAWarningNotACritical() throws {
        let events = try replay(["Bsilu"]).events
        let raise = try XCTUnwrap(events.first { $0.rule == .deadChannels && $0.kind == .raise }, describe(events))
        XCTAssertEqual(raise.trainerStep, 20_650)
        XCTAssertEqual(raise.severity, .warning)
        XCTAssertEqual(raise.threshold, "dead>0")
        XCTAssertEqual(raise.value, "dead=20/1040")
        XCTAssertTrue(events.filter { $0.rule == .deadChannels && $0.severity == .critical }.isEmpty, describe(events))
    }

    func testBSiluIllegalMassClearsAt21400() throws {
        let events = try replay(["Bsilu"]).events
        XCTAssertEqual(steps(events, .illegalMass, .clear), [21_400], describe(events))
    }

    // MARK: bn_running_variance_jump offline (BN_RUNNING_VARIANCE_CHANGE_ALARM_PLAN X5)

    /// Offline the jump arm sees only each live line's largest channel, a
    /// lower bound on the app's: B-silu raises critical at 19,900 (851.4×
    /// against 61.3×, the smallest of the lines' largest ratios in
    /// 18,900–19,850) — before the offline gradient_spike (20,600) and
    /// illegal_mass (20,700). The app, which sees every channel, raises at
    /// 19,800 (validation V-1).
    func testBSiluRunningVarianceJumpRaisesCriticalAt19900Offline() throws {
        let events = try replay(["Bsilu"]).events
        let raise = try XCTUnwrap(
            events.first { $0.rule == .batchNormRunningVarianceJump && $0.kind == .raise }, describe(events))
        XCTAssertEqual(raise.trainerStep, 19_900)
        XCTAssertEqual(raise.severity, .critical)
        XCTAssertEqual(raise.detail, "channels=blocks.2.bn1[76]:<=61.30->851.4")
        XCTAssertEqual(raise.value, "jumped=1 outliers=--/--", "the excerpt predates rvOver10xMedian=")
        let gradientSpike = try XCTUnwrap(steps(events, .gradientSpike, .raise).first)
        let illegalMass = try XCTUnwrap(steps(events, .illegalMass, .raise).first)
        XCTAssertLessThan(raise.trainerStep, gradientSpike)
        XCTAssertLessThan(raise.trainerStep, illegalMass)
    }

    func testRunningVarianceJumpIsQuietOnTheOtherIncidentAndHealthyLogs() throws {
        for names in [["A"], ["B"], ["C-seg0", "C-seg1"], ["R7"], ["R8"]] {
            let events = try replay(names).events.filter { $0.rule == .batchNormRunningVarianceJump }
            XCTAssertTrue(events.isEmpty, "\(names): \(describe(events))")
        }
    }
}
