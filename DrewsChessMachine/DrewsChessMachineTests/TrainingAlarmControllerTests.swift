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

    /// `vAbs` reading well below the warning threshold — a fresh-init value
    /// head produces ~0.30 empirically (see CHANGELOG 2026-05-11 finding).
    private let healthyValueAbsMean: Double = 0.30
    /// `vAbs` at warning band: `tanh` gradient ~17× weaker than vAbs=0.
    private let warningValueAbsMean: Double = 0.98
    /// `vAbs` at critical band: `tanh` gradient ~100× weaker.
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
}
