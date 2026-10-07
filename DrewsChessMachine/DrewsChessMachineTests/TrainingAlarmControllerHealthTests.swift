import XCTest
@testable import DrewsChessMachine

/// The training-health half of `TrainingAlarmController` (alarms plan T9):
/// the mirrored alarm list and the one beep loop both sources share.
/// `TrainingAlarmControllerTests` covers the banner detectors, unchanged.
@MainActor
final class TrainingAlarmControllerHealthTests: XCTestCase {

    private func alarm(_ rule: TrainingHealthRule, _ severity: TrainingAlarm.Severity) -> TrainingHealthActiveAlarm {
        TrainingHealthActiveAlarm(rule: rule, severity: severity, since: 300, value: "v", detail: "", action: .log)
    }

    func testShouldSoundForBannerOnly() {
        let controller = TrainingAlarmController()
        controller.raise(severity: .warning, title: "Banner", detail: "d")
        XCTAssertTrue(controller.shouldSound)
        controller.silence()
        XCTAssertFalse(controller.shouldSound)
        controller.clear()
    }

    func testShouldSoundForACriticalHealthAlarmAloneAndNotForWarnings() {
        let controller = TrainingAlarmController()
        controller.refreshHealth([alarm(.policyOffsetDrift, .warning)])
        XCTAssertFalse(controller.shouldSound, "warnings alone never beep (OD-8)")
        controller.refreshHealth([alarm(.policyOffsetDrift, .warning), alarm(.deadChannels, .critical)])
        XCTAssertNil(controller.active, "the banner shows nothing")
        XCTAssertTrue(controller.shouldSound, "a critical health alarm beeps although the banner shows nothing")
        controller.silence()
        XCTAssertFalse(controller.shouldSound)
        controller.refreshHealth([])
    }

    /// `clear()` at a promotion clears the banner only: the health alarms
    /// stay, the silence is reset, and the sound resumes for a remaining
    /// critical alarm.
    func testClearKeepsHealthAlarmsAndResumesSoundForACritical() {
        let controller = TrainingAlarmController()
        controller.raise(severity: .critical, title: "Banner", detail: "d")
        controller.refreshHealth([alarm(.illegalMass, .critical)])
        controller.silence()
        XCTAssertFalse(controller.shouldSound)
        controller.clear()
        XCTAssertNil(controller.active)
        XCTAssertEqual(controller.healthAlarms.map(\.rule), [.illegalMass])
        XCTAssertFalse(controller.silenced)
        XCTAssertTrue(controller.shouldSound)
        controller.refreshHealth([])
        XCTAssertFalse(controller.shouldSound)
    }

    func testDismissKeepsHealthAlarms() {
        let controller = TrainingAlarmController()
        controller.raise(severity: .warning, title: "Banner", detail: "d")
        controller.refreshHealth([alarm(.gradientSpike, .warning)])
        controller.dismiss()
        XCTAssertEqual(controller.healthAlarms.count, 1)
        XCTAssertFalse(controller.shouldSound)
        controller.refreshHealth([])
    }

    /// `refreshHealth(from:)` mirrors the monitor's current active set.
    func testRefreshHealthMirrorsTheMonitor() throws {
        let controller = TrainingAlarmController()
        let monitor = TrainingHealthMonitor(valueFC1Applicability: .applies, stopDecision: .byCaller)
        monitor.evaluateCheckpoint(
            stamp: monitor.observationStamp(),
            layerHealth: TrainingHealthTestSupport.deadDigest(
                dead: 8, channels: 1040, sites: [("value.bn", 8, 16)], tier: .checkpoint),
            digestTrainerStep: 300, config: try TrainingHealthTestSupport.config(), log: { _ in })
        controller.refreshHealth(from: monitor)
        XCTAssertEqual(controller.healthAlarms, monitor.activeAlarmsSnapshot())
        XCTAssertEqual(controller.healthAlarms.first?.rule, .deadChannels)
        XCTAssertEqual(controller.healthAlarms.first?.severity, .critical)
        controller.refreshHealth([])
    }

    func testHealthSuspensionHeaderState() {
        let controller = TrainingAlarmController()
        controller.setHealthSuspension(.deadChannels)
        XCTAssertEqual(controller.healthSuspendedRule, .deadChannels)
        controller.setHealthSuspension(nil)
        XCTAssertNil(controller.healthSuspendedRule)
    }
}
