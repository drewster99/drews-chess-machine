import XCTest
@testable import DrewsChessMachine

/// P5 (owner decision OD-9): the GUI banner detectors' and legal-mass
/// probe's conditions as shared functions (`TrainingHealthDetectorConditions`)
/// and as the evaluator's rules 10–13, so the command-line paths judge them
/// too. Written before the rules existed: they fail without them.
final class TrainingHealthDetectorRuleTests: XCTestCase {

    typealias S = TrainingHealthTestSupport
    typealias C = TrainingHealthDetectorConditions

    // MARK: The shared conditions

    func testDivergenceLevels() {
        XCTAssertEqual(C.divergenceLevel(entropy: 0.4, gradientNorm: 1), .critical)
        XCTAssertEqual(C.divergenceLevel(entropy: 2.0, gradientNorm: 501), .critical)
        XCTAssertEqual(C.divergenceLevel(entropy: nil, gradientNorm: 501), .critical, "the gradient arm alone")
        XCTAssertEqual(C.divergenceLevel(entropy: 0.9, gradientNorm: 51), .warning)
        XCTAssertNil(C.divergenceLevel(entropy: 0.9, gradientNorm: 50), "warning needs both arms")
        XCTAssertNil(C.divergenceLevel(entropy: 1.0, gradientNorm: 400))
        XCTAssertNil(C.divergenceLevel(entropy: nil, gradientNorm: 400), "a missing input never satisfies an arm")
    }

    func testValueSaturationAndDrawLevels() {
        XCTAssertNil(C.valueSaturationLevel(valueAbsMean: 0.969))
        XCTAssertEqual(C.valueSaturationLevel(valueAbsMean: 0.97), .warning)
        XCTAssertEqual(C.valueSaturationLevel(valueAbsMean: 0.995), .critical)
        XCTAssertNil(C.valueDrawSaturationLevel(valueProbDraw: 0.75), "a fresh head's prior")
        XCTAssertEqual(C.valueDrawSaturationLevel(valueProbDraw: 0.92), .warning)
        XCTAssertEqual(C.valueDrawSaturationLevel(valueProbDraw: 0.97), .critical)
        XCTAssertNil(C.valueDrawSaturationLevel(valueProbDraw: nil))
    }

    func testLegalMassStalledMatchesTheProbe() {
        let threshold = 0.99
        // Eight readings, all with illegal mass above 0.99, no improvement.
        let flat = Array(repeating: 0.005, count: 8)
        XCTAssertTrue(C.legalMassStalled(legalMassRun: flat, illegalMassThreshold: threshold, evaluations: 8))
        XCTAssertFalse(C.legalMassStalled(legalMassRun: Array(flat.prefix(7)), illegalMassThreshold: threshold, evaluations: 8),
                       "the window must be full")
        var improving = flat
        improving[7] = 0.009
        XCTAssertFalse(C.legalMassStalled(legalMassRun: improving, illegalMassThreshold: threshold, evaluations: 8),
                       "legal mass rose from the oldest to the newest")
        var oneBelow = flat
        oneBelow[3] = 0.02
        XCTAssertFalse(C.legalMassStalled(legalMassRun: oneBelow, illegalMassThreshold: threshold, evaluations: 8),
                       "every reading must be above the threshold")
    }

    /// The banner reads its levels from the shared conditions: a critical
    /// divergence reading raises the banner's critical title after its
    /// streak, exactly as before the move.
    @MainActor
    func testBannerUsesTheSharedDivergenceLevel() {
        let controller = TrainingAlarmController()
        for _ in 0..<TrainingAlarmController.divergenceAlarmConsecutiveCriticalSamples {
            controller.evaluate(rollingPolicyEntropy: 2.0, rollingGradNorm: TrainingHealthThresholds.divergenceGradientNormCritical + 1)
        }
        XCTAssertEqual(controller.active?.title, TrainingAlarmController.divergenceCriticalAlarmTitle)
        controller.clear()
    }

    // MARK: The rules on the step window

    private func diagnostic(
        _ step: Int, entropy: Float? = 2.0, gradient: Float = 1.0, valueAbs: Float? = 0.3,
        draw: Float? = 0.5, illegal: Float = 0.01
    ) -> TrainingHealthStepRecord {
        TrainingHealthStepRecord(
            trainerStep: step, loss: 4, illegalMassPenalty: illegal, gradGlobalNorm: gradient, totalMs: 10,
            policyLogitMean: 0.1, policyEntropy: entropy, valueAbsMean: valueAbs, valueProbDraw: draw)
    }

    private func evaluate(
        _ evaluator: inout TrainingHealthEvaluator,
        _ records: [TrainingHealthStepRecord],
        at step: Int,
        config: TrainingHealthConfig
    ) -> TrainingHealthEvaluation {
        evaluator.evaluate(S.liveObservation(step: step, records: records), config: config)
    }

    private func events(_ evaluation: TrainingHealthEvaluation, _ rule: TrainingHealthRule) -> [TrainingHealthEvent] {
        evaluation.events.filter { $0.rule == rule }
    }

    func testDivergenceRaisesCriticalAfterItsSustainAndClears() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        let first = evaluate(&evaluator, [diagnostic(50, entropy: 0.3)], at: 50, config: config)
        XCTAssertTrue(events(first, .divergence).isEmpty, "one evaluation is not sustained")
        let second = evaluate(&evaluator, [diagnostic(100, entropy: 0.3)], at: 100, config: config)
        XCTAssertEqual(events(second, .divergence).map(\.kind), [.raise])
        XCTAssertEqual(events(second, .divergence).first?.severity, .critical)
        _ = evaluate(&evaluator, [diagnostic(150)], at: 150, config: config)
        let cleared = evaluate(&evaluator, [diagnostic(200)], at: 200, config: config)
        XCTAssertEqual(events(cleared, .divergence).map(\.kind), [.clear])
    }

    func testDivergenceWithoutEntropyHasNoDataUnlessTheGradientIsCritical() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        let lean = evaluate(&evaluator, [S.record(50)], at: 50, config: config)
        XCTAssertTrue(lean.noDataRules.contains(.divergence))
        _ = evaluate(&evaluator, [S.record(100, gradient: 600)], at: 100, config: config)
        let raised = evaluate(&evaluator, [S.record(150, gradient: 600)], at: 150, config: config)
        XCTAssertEqual(events(raised, .divergence).first?.severity, .critical)
    }

    func testValueSaturationWarnsThenEscalates() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        _ = evaluate(&evaluator, [diagnostic(50, valueAbs: 0.98)], at: 50, config: config)
        let warned = evaluate(&evaluator, [diagnostic(100, valueAbs: 0.98)], at: 100, config: config)
        XCTAssertEqual(events(warned, .valueSaturation).map(\.kind), [.raise])
        XCTAssertEqual(events(warned, .valueSaturation).first?.severity, .warning)
        let first = evaluate(&evaluator, [diagnostic(150, valueAbs: 0.999)], at: 150, config: config)
        let second = evaluate(&evaluator, [diagnostic(200, valueAbs: 0.999)], at: 200, config: config)
        let kinds = (events(first, .valueSaturation) + events(second, .valueSaturation)).map(\.kind)
        XCTAssertEqual(kinds, [.escalate], "one escalation to critical, never a de-escalation")
        XCTAssertEqual(evaluator.activeAlarms.first { $0.rule == .valueSaturation }?.severity, .critical)
    }

    func testValueDrawSaturationRaisesAndFreshPriorDoesNot() throws {
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        _ = evaluate(&evaluator, [diagnostic(50, draw: 0.75)], at: 50, config: config)
        let prior = evaluate(&evaluator, [diagnostic(100, draw: 0.75)], at: 100, config: config)
        XCTAssertTrue(events(prior, .valueDrawSaturation).isEmpty)
        _ = evaluate(&evaluator, [diagnostic(150, draw: 0.98)], at: 150, config: config)
        let raised = evaluate(&evaluator, [diagnostic(200, draw: 0.98)], at: 200, config: config)
        XCTAssertEqual(events(raised, .valueDrawSaturation).first?.severity, .critical)
    }

    func testLegalMassStallRaisesAfterTheProbeCountPastTheGate() throws {
        let config = try S.config(learningGraceSteps: 0, lrWarmupSteps: 100)
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        var raisedAt: Int?
        for index in 1...10 {
            let step = index * 50
            let evaluation = evaluate(&evaluator, [S.record(step, illegal: 0.995)], at: step, config: config)
            if let raise = events(evaluation, .legalMassStall).first(where: { $0.kind == .raise }) {
                XCTAssertEqual(raise.severity, .critical)
                raisedAt = raisedAt ?? raise.trainerStep
            }
        }
        XCTAssertEqual(raisedAt, 400, "eight consecutive stalled evaluations (50…400)")
    }

    func testLegalMassStallNeverRaisesBeforeTheGateOrWhileImproving() throws {
        let gated = try S.config(learningGraceSteps: 1000, lrWarmupSteps: 1000)
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        for index in 1...12 {
            let evaluation = evaluate(&evaluator, [S.record(index * 50, illegal: 0.995)], at: index * 50, config: gated)
            XCTAssertTrue(events(evaluation, .legalMassStall).isEmpty, "before the gate (2000)")
        }
        let open = try S.config(learningGraceSteps: 0, lrWarmupSteps: 0)
        var improving = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        for index in 1...12 {
            // Illegal mass falling every window: legal mass improves.
            let illegal = Float(0.999 - Double(index) * 0.0005)
            let evaluation = evaluate(&improving, [S.record(index * 50, illegal: illegal)], at: index * 50, config: open)
            XCTAssertTrue(events(evaluation, .legalMassStall).isEmpty, "improving at \(index * 50)")
        }
    }

    func testNewRulesCarryNoCheckpointData() throws {
        // A checkpoint pass never feeds the window rules, so it never counts
        // them as "no data".
        let config = try S.config()
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        let evaluation = evaluator.evaluate(
            S.checkpointObservation(step: 1000, digest: S.deadDigest(dead: 0, tier: .checkpoint)), config: config)
        for rule in [TrainingHealthRule.divergence, .valueSaturation, .valueDrawSaturation, .legalMassStall] {
            XCTAssertFalse(evaluation.noDataRules.contains(rule), rule.rawValue)
        }
    }
}
