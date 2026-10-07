import XCTest
@testable import DrewsChessMachine

/// Exact strings for every training-health line kind: the grep contract
/// (the alarms plan, D3).
final class TrainingHealthLogTests: XCTestCase {

    private func event(
        _ kind: TrainingHealthEvent.Kind,
        rule: TrainingHealthRule = .deadChannels,
        severity: TrainingAlarm.Severity = .critical,
        step: Int = 300,
        since: Int? = 300,
        value: String = "dead=5/1040",
        threshold: String = "site>=0.2",
        detail: String = "sites=value.bn(5/16)",
        action: TrainingHealthAction = .log,
        lr: Double? = 0.3,
        mom: Double? = 0.85
    ) -> TrainingHealthEvent {
        TrainingHealthEvent(
            kind: kind, rule: rule, severity: severity, trainerStep: step, since: since, value: value,
            threshold: threshold, detail: detail, action: action, learningRate: lr, momentum: mom)
    }

    func testRaiseLine() {
        XCTAssertEqual(
            TrainingHealthLog.eventLine(event(.raise)),
            "[ALARM] health raise rule=dead_channels severity=critical trainerStep=300 value=dead=5/1040 sites=value.bn(5/16) threshold=site>=0.2 action=log lr=0.3 mom=0.85")
    }

    func testRaiseLineWithoutDetailAndWithNotMeasuredRates() {
        XCTAssertEqual(
            TrainingHealthLog.eventLine(event(
                .raise, rule: .valueFC1ZeroVelocity, step: 1513, value: "zero=128/128", threshold: "zero>=0.5",
                detail: "", action: .stopOnCritical, lr: nil, mom: nil)),
            "[ALARM] health raise rule=value_fc1_zero_velocity severity=critical trainerStep=1513 value=zero=128/128 threshold=zero>=0.5 action=stop_on_critical lr=-- mom=--")
    }

    func testEscalateLine() {
        XCTAssertEqual(
            TrainingHealthLog.eventLine(event(.escalate, step: 1300)),
            "[ALARM] health escalate rule=dead_channels severity=critical trainerStep=1300 value=dead=5/1040 sites=value.bn(5/16) threshold=site>=0.2 action=log lr=0.3 mom=0.85")
    }

    func testWorsenLine() {
        XCTAssertEqual(
            TrainingHealthLog.eventLine(event(.worsen, step: 1300, value: "dead=6/1040", threshold: "", detail: "was=5 sites=value.bn(6/16)")),
            "[ALARM] health worsen rule=dead_channels severity=critical trainerStep=1300 value=dead=6/1040 was=5 sites=value.bn(6/16)")
    }

    func testActiveLine() {
        XCTAssertEqual(
            TrainingHealthLog.eventLine(event(.active, step: 2000, value: "dead=6/1040", threshold: "", detail: "sites=value.bn(6/16)")),
            "[ALARM] health active rule=dead_channels severity=critical since=300 trainerStep=2000 value=dead=6/1040 sites=value.bn(6/16)")
    }

    func testClearLine() {
        XCTAssertEqual(
            TrainingHealthLog.eventLine(event(
                .clear, rule: .lossSpike, severity: .warning, step: 500, value: "median/ref=1.04", threshold: "",
                detail: "max/ref=1.04 ref=7.5600")),
            "[ALARM] health clear rule=loss_spike severity=warning since=300 trainerStep=500 value=median/ref=1.04 max/ref=1.04 ref=7.5600")
    }

    func testStopLine() {
        XCTAssertEqual(
            TrainingHealthLog.eventLine(event(.stop, rule: .illegalMass, step: 350, action: .stopOnCritical)),
            "[ALARM] health stop rule=illegal_mass severity=critical trainerStep=350 action=stop_on_critical")
    }

    func testSignedAndSmallValues() {
        XCTAssertEqual(TrainingHealthLog.learningRate(0.000123456), "0.000123")
        XCTAssertEqual(TrainingHealthLog.learningRate(10), "10")
        XCTAssertEqual(TrainingHealthLog.learningRate(.nan), "--")
        XCTAssertEqual(TrainingHealthLog.momentum(0.9), "0.9")
        XCTAssertEqual(TrainingHealthEvaluator.fixed(-3.80321, 4), "-3.8032")
    }

    func testConfigLine() throws {
        var actions = TrainingHealthActions { _ in .log }
        actions[.illegalMass] = .stopOnCritical
        let config = try TrainingHealthConfig(
            enabled: true, checkIntervalSteps: 1000, learningGraceSteps: 1000, lrWarmupSteps: 1000,
            momentumCoefficient: 0.85, actions: actions)
        XCTAssertEqual(
            TrainingHealthLog.configLine(config: config, path: "replay", valueFC1Applicability: .applies),
            "[HEALTH] config enabled=true interval=1000 grace=1000 warmup=1000 momentum=0.85 path=replay actions=non_finite:log,dead_channels:log,value_fc1_zero_velocity:log,illegal_mass:stop_on_critical,gradient_collapse:log,loss_spike:log,policy_offset_drift:log,bn_running_variance_runaway:log,gradient_spike:log value_fc1_zero_velocity=applies")
        XCTAssertTrue(TrainingHealthLog.configLine(
            config: config, path: "gui", valueFC1Applicability: .doesNotApply(activation: .leakyRelu))
            .hasSuffix(" value_fc1_zero_velocity=not_applicable(activation=leaky_relu)"))
        XCTAssertEqual(TrainingHealthLog.disabledLine(path: "vsuci"), "[HEALTH] alarms disabled path=vsuci")
    }

    func testMonitorLines() {
        XCTAssertEqual(TrainingHealthLog.rewindLine(from: 5200, to: 4900, generation: 1),
                       "[HEALTH] trainer clock rewound 5200 -> 4900; generation 1; windows reset")
        XCTAssertEqual(TrainingHealthLog.rewindLine(from: nil, to: 0, generation: 1),
                       "[HEALTH] trainer clock rewound none -> 0; generation 1; windows reset")
        XCTAssertEqual(TrainingHealthLog.unannouncedRewindLine(from: 100, to: 50, generation: 2),
                       "[HEALTH] trainer clock rewind detected by recordStep (not announced): 100 -> 50; generation 2")
        XCTAssertEqual(
            TrainingHealthLog.staleObservationLine(
                tier: .checkpoint, trainerStep: 4800, generation: 0, currentGeneration: 1, newestApplied: 4900, reason: nil),
            "[HEALTH] stale checkpoint observation ignored: trainerStep=4800 generation=0 (current generation 1, newest applied 4900)")
        XCTAssertEqual(
            TrainingHealthLog.staleObservationLine(
                tier: .live, trainerStep: 300, generation: 1, currentGeneration: 2, newestApplied: nil,
                reason: "trainer clock rewound during the evaluation"),
            "[HEALTH] stale live observation ignored: trainerStep=300 generation=1 (current generation 2, newest applied none; trainer clock rewound during the evaluation)")
        XCTAssertEqual(TrainingHealthLog.liveReadFailedLine(boundary: 750),
                       "[HEALTH] live read failed at trainerStep=750; window boundary = last recorded step 750; layer-health rules no data")
        XCTAssertEqual(TrainingHealthLog.notApplicableLine(rule: .valueFC1ZeroVelocity, reason: "value FC1 activation is silu, not relu"),
                       "[HEALTH] rule value_fc1_zero_velocity does not apply: value FC1 activation is silu, not relu")
    }

    func testCheckLine() {
        let fields = TrainingHealthLog.CheckFields(
            trainerStep: 2000, generation: 0, live: 20, checkpoint: 1, stale: 0, truncated: 0, costMs: 3.14,
            trainMs: 651_400, liveReadFailed: 0, lossMaxRatio: 1.14, lossMedianRatio: 1.02, gradientMaxRatio: nil,
            noData: [.policyOffsetDrift: 3, .valueFC1ZeroVelocity: 20],
            active: [
                TrainingHealthActiveAlarm(rule: .deadChannels, severity: .critical, since: 300, value: "dead=6/1040",
                                          detail: "sites=value.bn(6/16),blocks.0.bn1(1/128)", action: .log),
                TrainingHealthActiveAlarm(rule: .policyOffsetDrift, severity: .warning, since: 900, value: "medianAbs=3.8",
                                          detail: "", action: .log),
            ],
            marker: nil)
        XCTAssertEqual(
            TrainingHealthLog.checkLine(fields),
            "[HEALTH] check trainerStep=2000 generation=0 evaluations=21 live=20 checkpoint=1 stale=0 truncated=0 cost_ms=3.1 train_ms=651400.0 liveReadFailed=0 lossMaxRatio=1.14 lossMedianRatio=1.02 gradMaxRatio=-- nodata=value_fc1_zero_velocity:20,policy_offset_drift:3 active=dead_channels:critical,policy_offset_drift:warning dead_channels_sites=value.bn(6/16),blocks.0.bn1(1/128)")
        let empty = TrainingHealthLog.CheckFields(
            trainerStep: 1250, generation: 2, live: 5, checkpoint: 0, stale: 1, truncated: 0, costMs: 0.04,
            trainMs: 250, liveReadFailed: 1, lossMaxRatio: nil, lossMedianRatio: nil, gradientMaxRatio: 1.25,
            noData: [:], active: [], marker: .final)
        XCTAssertEqual(
            TrainingHealthLog.checkLine(empty),
            "[HEALTH] check trainerStep=1250 generation=2 evaluations=5 live=5 checkpoint=0 stale=1 truncated=0 cost_ms=0.0 train_ms=250.0 liveReadFailed=1 lossMaxRatio=-- lossMedianRatio=-- gradMaxRatio=1.25 nodata=none active=none final=true")
    }

    /// Review minor 1: every field of the active `dead_channels` detail gets
    /// the rule's prefix, so the coverage note is not a bare `coverage=`.
    func testCheckLinePrefixesEveryDeadChannelsField() {
        let fields = TrainingHealthLog.CheckFields(
            trainerStep: 1000, generation: 0, live: 20, checkpoint: 0, stale: 0, truncated: 0, costMs: 1,
            trainMs: 1000, liveReadFailed: 0, lossMaxRatio: nil, lossMedianRatio: nil, gradientMaxRatio: nil,
            noData: [:],
            active: [
                TrainingHealthActiveAlarm(rule: .deadChannels, severity: .warning, since: 300, value: "dead=20/1040",
                                          detail: "sites=policy.pre_bn(19/128) coverage=relu_leaky_relu_only", action: .log),
            ],
            marker: nil)
        XCTAssertTrue(
            TrainingHealthLog.checkLine(fields).hasSuffix(
                " active=dead_channels:warning dead_channels_sites=policy.pre_bn(19/128) dead_channels_coverage=relu_leaky_relu_only"),
            TrainingHealthLog.checkLine(fields))
    }

    func testValueFC1Line() {
        XCTAssertEqual(
            TrainingHealthLog.valueFC1Line(
                trainerStep: 3000, stepsTrainedByThisProcess: 2500, zeroVelocityUnitCount: 2, unitCount: 128, lowVelocityUnitCount: 9, readMs: 1.234, summaryMs: 0.5),
            "[LAYER-HEALTH] value-fc1 trainerStep=3000 trained=2500 valueFC1ZeroVel=2/128 lowVel=9 readMs=1.23 summaryMs=0.50")
    }

    func testEveryAlarmLineIsSelectedByTheGrepAndNothingElse() {
        for kind in TrainingHealthEvent.Kind.allCases {
            XCTAssertTrue(TrainingHealthLog.eventLine(event(kind)).hasPrefix("[ALARM] health \(kind.rawValue) rule="))
        }
        XCTAssertFalse(TrainingHealthLog.rewindLine(from: 1, to: 0, generation: 1).hasPrefix("[ALARM] health"))
    }
}
