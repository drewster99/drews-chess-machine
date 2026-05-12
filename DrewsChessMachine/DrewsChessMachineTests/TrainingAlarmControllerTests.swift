import XCTest
@testable import DrewsChessMachine

/// Exercises the divergence-streak detector and the raise / auto-clear / dismiss
/// lifecycle on `TrainingAlarmController` — the pure-logic core of the in-app
/// training-alarm banner, extracted out of `UpperContentView` in the
/// decomposition. Drives `evaluate(rollingPolicyEntropy:rollingGradNorm:)`
/// directly so no `TrainingChartSample` needs to be constructed.
@MainActor
final class TrainingAlarmControllerTests: XCTestCase {

    // A reading that is neither warning- nor critical-out-of-line.
    private let healthyEntropy: Double = 2.0
    private let healthyGradNorm: Double = 5.0
    // Warning-out-of-line (entropy < policyEntropyAlarmThreshold AND gradNorm
    // > divergenceAlarmGradNormWarningThreshold) but NOT critical.
    private let warningEntropy: Double = 0.7
    private let warningGradNorm: Double = 60.0
    // Critical-out-of-line (entropy < divergenceAlarmEntropyCriticalThreshold).
    private let criticalEntropy: Double = 0.3
    private let criticalGradNorm: Double = 5.0

    private func feed(_ c: TrainingAlarmController, entropy: Double, gradNorm: Double, times: Int) {
        for _ in 0..<times {
            c.evaluate(rollingPolicyEntropy: entropy, rollingGradNorm: gradNorm)
        }
    }

    func testCriticalAlarmRaisesAfterConsecutiveCriticalSamples() {
        let c = TrainingAlarmController()
        feed(c, entropy: criticalEntropy, gradNorm: criticalGradNorm,
             times: TrainingAlarmController.divergenceAlarmConsecutiveCriticalSamples - 1)
        XCTAssertNil(c.active, "should not raise before the consecutive-critical threshold")
        c.evaluate(rollingPolicyEntropy: criticalEntropy, rollingGradNorm: criticalGradNorm)
        XCTAssertEqual(c.active?.title, TrainingAlarmController.divergenceCriticalAlarmTitle)
        XCTAssertEqual(c.active?.severity, .critical)
    }

    func testWarningAlarmRaisesAfterConsecutiveWarningSamples() {
        let c = TrainingAlarmController()
        feed(c, entropy: warningEntropy, gradNorm: warningGradNorm,
             times: TrainingAlarmController.divergenceAlarmConsecutiveWarningSamples - 1)
        XCTAssertNil(c.active, "should not raise before the consecutive-warning threshold")
        c.evaluate(rollingPolicyEntropy: warningEntropy, rollingGradNorm: warningGradNorm)
        XCTAssertEqual(c.active?.title, TrainingAlarmController.divergenceWarningAlarmTitle)
        XCTAssertEqual(c.active?.severity, .warning)
    }

    func testHealthyStreakAutoClearsOurOwnAlarm() {
        let c = TrainingAlarmController()
        feed(c, entropy: criticalEntropy, gradNorm: criticalGradNorm,
             times: TrainingAlarmController.divergenceAlarmConsecutiveCriticalSamples)
        XCTAssertNotNil(c.active)
        feed(c, entropy: healthyEntropy, gradNorm: healthyGradNorm,
             times: TrainingAlarmController.divergenceAlarmRecoverySamples - 1)
        XCTAssertNotNil(c.active, "should not auto-clear before the recovery threshold")
        c.evaluate(rollingPolicyEntropy: healthyEntropy, rollingGradNorm: healthyGradNorm)
        XCTAssertNil(c.active, "healthy-reading streak should auto-clear our own alarm")
    }

    func testHealthyStreakDoesNotClearAnotherDetectorsAlarm() {
        let c = TrainingAlarmController()
        c.raise(severity: .critical, title: "Policy Collapse (legal mass)", detail: "x")
        XCTAssertNotNil(c.active)
        feed(c, entropy: healthyEntropy, gradNorm: healthyGradNorm,
             times: TrainingAlarmController.divergenceAlarmRecoverySamples + 5)
        XCTAssertEqual(c.active?.title, "Policy Collapse (legal mass)",
                       "the divergence auto-clear must not wipe an alarm it didn't raise")
    }

    func testDismissResetsStreaks() {
        let c = TrainingAlarmController()
        feed(c, entropy: criticalEntropy, gradNorm: criticalGradNorm,
             times: TrainingAlarmController.divergenceAlarmConsecutiveCriticalSamples)
        XCTAssertNotNil(c.active)
        c.dismiss()
        XCTAssertNil(c.active)
        // One more critical sample: with streaks reset, the critical streak is
        // now 1 (< threshold), so nothing should re-raise. If dismiss() had not
        // reset the streaks, the critical streak would still be at/above the
        // threshold and the alarm would immediately re-raise.
        c.evaluate(rollingPolicyEntropy: criticalEntropy, rollingGradNorm: criticalGradNorm)
        XCTAssertNil(c.active, "dismiss() should reset the divergence streak counters")
    }

    func testClearDoesNotResetStreaks() {
        let c = TrainingAlarmController()
        // Build the critical streak just shy of raising.
        feed(c, entropy: criticalEntropy, gradNorm: criticalGradNorm,
             times: TrainingAlarmController.divergenceAlarmConsecutiveCriticalSamples - 1)
        XCTAssertNil(c.active)
        c.clear()  // no-op on the banner here, but must NOT touch the streaks
        c.evaluate(rollingPolicyEntropy: criticalEntropy, rollingGradNorm: criticalGradNorm)
        XCTAssertEqual(c.active?.title, TrainingAlarmController.divergenceCriticalAlarmTitle,
                       "clear() must leave the streak counters alone")
    }

    // MARK: - Value-head saturation detector

    /// `vAbs` (= mean |p_win − p_loss| for the W/D/L head) well below
    /// the warning threshold — a fresh-init head sits near 0 (bias init
    /// [0, ln6, 0] ⇒ p_win = p_loss); anything < 0.97 is "healthy".
    private let healthyValueAbsMean: Double = 0.30
    /// `vAbs` in the warning band (head calling nearly every position a
    /// clean win/loss).
    private let warningValueAbsMean: Double = 0.98
    /// `vAbs` in the critical band.
    private let criticalValueAbsMean: Double = 0.999

    private func feedV(_ c: TrainingAlarmController, valueAbsMean: Double, times: Int) {
        // Drive the detector with healthy entropy/gradNorm so the divergence
        // detector never trips and contaminates the saturation tests.
        for _ in 0..<times {
            c.evaluate(rollingPolicyEntropy: healthyEntropy,
                       rollingGradNorm: healthyGradNorm,
                       rollingValueAbsMean: valueAbsMean)
        }
    }

    func testCriticalValueAbsMeanSaturationRaises() {
        let c = TrainingAlarmController()
        feedV(c, valueAbsMean: criticalValueAbsMean,
              times: TrainingAlarmController.valueAbsMeanSaturationConsecutiveCriticalSamples - 1)
        XCTAssertNil(c.active, "should not raise before the consecutive-critical threshold")
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueAbsMean: criticalValueAbsMean)
        XCTAssertEqual(c.active?.title, TrainingAlarmController.valueAbsMeanSaturationCriticalAlarmTitle)
        XCTAssertEqual(c.active?.severity, .critical)
    }

    func testWarningValueAbsMeanSaturationRaises() {
        let c = TrainingAlarmController()
        feedV(c, valueAbsMean: warningValueAbsMean,
              times: TrainingAlarmController.valueAbsMeanSaturationConsecutiveWarningSamples - 1)
        XCTAssertNil(c.active, "should not raise before the consecutive-warning threshold")
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueAbsMean: warningValueAbsMean)
        XCTAssertEqual(c.active?.title, TrainingAlarmController.valueAbsMeanSaturationWarningAlarmTitle)
        XCTAssertEqual(c.active?.severity, .warning)
    }

    func testValueAbsMeanRecoveryAutoClearsOurOwnAlarm() {
        let c = TrainingAlarmController()
        feedV(c, valueAbsMean: criticalValueAbsMean,
              times: TrainingAlarmController.valueAbsMeanSaturationConsecutiveCriticalSamples)
        XCTAssertNotNil(c.active)
        feedV(c, valueAbsMean: healthyValueAbsMean,
              times: TrainingAlarmController.valueAbsMeanSaturationRecoverySamples - 1)
        XCTAssertNotNil(c.active, "should not auto-clear before the recovery threshold")
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueAbsMean: healthyValueAbsMean)
        XCTAssertNil(c.active, "healthy vAbs streak should auto-clear our own alarm")
    }

    func testValueAbsMeanRecoveryDoesNotClearAnotherDetectorsAlarm() {
        let c = TrainingAlarmController()
        c.raise(severity: .critical, title: "Policy Collapse (legal mass)", detail: "x")
        XCTAssertNotNil(c.active)
        feedV(c, valueAbsMean: healthyValueAbsMean,
              times: TrainingAlarmController.valueAbsMeanSaturationRecoverySamples + 5)
        XCTAssertEqual(c.active?.title, "Policy Collapse (legal mass)",
                       "the saturation auto-clear must not wipe an alarm it didn't raise")
    }

    func testValueAbsMeanNilResetsSaturationStreaks() {
        let c = TrainingAlarmController()
        // Almost trigger the critical alarm.
        feedV(c, valueAbsMean: criticalValueAbsMean,
              times: TrainingAlarmController.valueAbsMeanSaturationConsecutiveCriticalSamples - 1)
        XCTAssertNil(c.active)
        // One sample with no vAbs data — should reset the saturation streak.
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueAbsMean: nil)
        // One more critical-vAbs reading — with the streak reset, the critical
        // streak is now 1, well below the threshold, so no alarm should raise.
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueAbsMean: criticalValueAbsMean)
        XCTAssertNil(c.active, "nil vAbs must reset the saturation streak counters")
    }

    func testDivergenceAndSaturationCoexistIndependently() {
        let c = TrainingAlarmController()
        // Build a divergence-critical streak with healthy vAbs.
        for _ in 0..<TrainingAlarmController.divergenceAlarmConsecutiveCriticalSamples {
            c.evaluate(rollingPolicyEntropy: criticalEntropy,
                       rollingGradNorm: criticalGradNorm,
                       rollingValueAbsMean: healthyValueAbsMean)
        }
        XCTAssertEqual(c.active?.title, TrainingAlarmController.divergenceCriticalAlarmTitle)
        // Now flip to healthy entropy/gradNorm and a critical vAbs streak.
        // Long enough for the divergence detector's recovery threshold
        // *and* the saturation detector's critical threshold.
        let recoveryAndSaturationStream = max(
            TrainingAlarmController.divergenceAlarmRecoverySamples,
            TrainingAlarmController.valueAbsMeanSaturationConsecutiveCriticalSamples
        ) + 1
        for _ in 0..<recoveryAndSaturationStream {
            c.evaluate(rollingPolicyEntropy: healthyEntropy,
                       rollingGradNorm: healthyGradNorm,
                       rollingValueAbsMean: criticalValueAbsMean)
        }
        // The divergence detector should have auto-cleared its banner (its
        // own recovery streak completed); the saturation detector should
        // have raised the saturation banner instead. The most-recent
        // raise wins.
        XCTAssertEqual(c.active?.title, TrainingAlarmController.valueAbsMeanSaturationCriticalAlarmTitle)
    }

    // MARK: - Value-head draw-collapse detector

    /// `pD` (= mean p_draw for the W/D/L head) — a fresh head sits at
    /// 0.75 (the [0, ln6, 0] bias init) and healthy training pulls it
    /// down; anything below the 0.92 warning threshold is "healthy".
    private let healthyValueProbDraw: Double = 0.55
    /// `pD` in the warning band (regressing toward all-draws).
    private let warningValueProbDraw: Double = 0.93
    /// `pD` in the critical band (essentially collapsed).
    private let criticalValueProbDraw: Double = 0.99

    private func feedPD(_ c: TrainingAlarmController, valueProbDraw: Double, times: Int) {
        // Healthy entropy/gradNorm and nil vAbs so neither the divergence
        // nor the saturation detector contaminates the draw-collapse tests.
        for _ in 0..<times {
            c.evaluate(rollingPolicyEntropy: healthyEntropy,
                       rollingGradNorm: healthyGradNorm,
                       rollingValueAbsMean: nil,
                       rollingValueProbDraw: valueProbDraw)
        }
    }

    func testCriticalValueDrawCollapseRaises() {
        let c = TrainingAlarmController()
        feedPD(c, valueProbDraw: criticalValueProbDraw,
               times: TrainingAlarmController.valueDrawCollapseConsecutiveCriticalSamples - 1)
        XCTAssertNil(c.active, "should not raise before the consecutive-critical threshold")
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueProbDraw: criticalValueProbDraw)
        XCTAssertEqual(c.active?.title, TrainingAlarmController.valueDrawCollapseCriticalAlarmTitle)
        XCTAssertEqual(c.active?.severity, .critical)
    }

    func testWarningValueDrawCollapseRaises() {
        let c = TrainingAlarmController()
        feedPD(c, valueProbDraw: warningValueProbDraw,
               times: TrainingAlarmController.valueDrawCollapseConsecutiveWarningSamples - 1)
        XCTAssertNil(c.active, "should not raise before the consecutive-warning threshold")
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueProbDraw: warningValueProbDraw)
        XCTAssertEqual(c.active?.title, TrainingAlarmController.valueDrawCollapseWarningAlarmTitle)
        XCTAssertEqual(c.active?.severity, .warning)
    }

    func testValueDrawCollapseRecoveryAutoClearsOurOwnAlarm() {
        let c = TrainingAlarmController()
        feedPD(c, valueProbDraw: criticalValueProbDraw,
               times: TrainingAlarmController.valueDrawCollapseConsecutiveCriticalSamples)
        XCTAssertNotNil(c.active)
        feedPD(c, valueProbDraw: healthyValueProbDraw,
               times: TrainingAlarmController.valueDrawCollapseRecoverySamples - 1)
        XCTAssertNotNil(c.active, "should not auto-clear before the recovery threshold")
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueProbDraw: healthyValueProbDraw)
        XCTAssertNil(c.active, "healthy pD streak should auto-clear our own alarm")
    }

    func testValueDrawCollapseRecoveryDoesNotClearAnotherDetectorsAlarm() {
        let c = TrainingAlarmController()
        c.raise(severity: .critical, title: "Policy Collapse (legal mass)", detail: "x")
        XCTAssertNotNil(c.active)
        feedPD(c, valueProbDraw: healthyValueProbDraw,
               times: TrainingAlarmController.valueDrawCollapseRecoverySamples + 5)
        XCTAssertEqual(c.active?.title, "Policy Collapse (legal mass)",
                       "the draw-collapse auto-clear must not wipe an alarm it didn't raise")
    }

    func testValueProbDrawNilResetsCollapseStreaks() {
        let c = TrainingAlarmController()
        feedPD(c, valueProbDraw: criticalValueProbDraw,
               times: TrainingAlarmController.valueDrawCollapseConsecutiveCriticalSamples - 1)
        XCTAssertNil(c.active)
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueProbDraw: nil)
        c.evaluate(rollingPolicyEntropy: healthyEntropy,
                   rollingGradNorm: healthyGradNorm,
                   rollingValueProbDraw: criticalValueProbDraw)
        XCTAssertNil(c.active, "nil pD must reset the draw-collapse streak counters")
    }

    /// The two value-head detectors key off mutually-exclusive extremes
    /// (`vAbs → 1` vs `pD → 1`), so a sample can't trip both — but their
    /// streak state and ownership-scoped auto-clear must stay independent.
    func testSaturationAndDrawCollapseAreIndependent() {
        let c = TrainingAlarmController()
        // Raise the draw-collapse banner.
        feedPD(c, valueProbDraw: criticalValueProbDraw,
               times: TrainingAlarmController.valueDrawCollapseConsecutiveCriticalSamples)
        XCTAssertEqual(c.active?.title, TrainingAlarmController.valueDrawCollapseCriticalAlarmTitle)
        // Now flip to a saturation-critical streak (high vAbs, low pD).
        // Long enough for the draw-collapse recovery AND the saturation
        // critical threshold.
        let stream = max(
            TrainingAlarmController.valueDrawCollapseRecoverySamples,
            TrainingAlarmController.valueAbsMeanSaturationConsecutiveCriticalSamples
        ) + 1
        for _ in 0..<stream {
            c.evaluate(rollingPolicyEntropy: healthyEntropy,
                       rollingGradNorm: healthyGradNorm,
                       rollingValueAbsMean: criticalValueAbsMean,
                       rollingValueProbDraw: healthyValueProbDraw)
        }
        XCTAssertEqual(c.active?.title, TrainingAlarmController.valueAbsMeanSaturationCriticalAlarmTitle)
    }
}
