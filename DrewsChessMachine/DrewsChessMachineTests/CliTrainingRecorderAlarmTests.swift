import XCTest
@testable import DrewsChessMachine

/// The training-health part of `results.json` (alarms plan D3): the
/// always-present `alarms` array, the `alarm_config` object, and the
/// `training_health_alarm` termination reason. Snake-case keys are the
/// contract with downstream tooling, so each is checked by name.
final class CliTrainingRecorderAlarmTests: XCTestCase {

    typealias S = TrainingHealthTestSupport

    private func json(_ recorder: CliTrainingRecorder) throws -> [String: Any] {
        let data = try recorder.encodedJSONData(totalTrainingSeconds: 1)
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testAlarmsArePresentAndEmptyByDefault() throws {
        let object = try json(CliTrainingRecorder())
        XCTAssertEqual((object["alarms"] as? [Any])?.count, 0)
        XCTAssertFalse(object.keys.contains("alarm_config"))
    }

    func testEventEncoding() throws {
        let recorder = CliTrainingRecorder()
        recorder.appendAlarmEvents([
            TrainingHealthEvent(
                kind: .raise, rule: .deadChannels, severity: .critical, trainerStep: 300, since: nil,
                value: "dead=5/1040", threshold: "site>=0.2", detail: "sites=value.bn(5/16)",
                action: .stopOnCritical, learningRate: 0.3, momentum: 0.85),
            TrainingHealthEvent(
                kind: .clear, rule: .lossSpike, severity: .warning, trainerStep: 500, since: 300,
                value: "median/ref=1.04", threshold: "", detail: "", action: .log,
                learningRate: nil, momentum: nil),
        ])
        recorder.appendAlarmEvents([])
        let alarms = try XCTUnwrap(try json(recorder)["alarms"] as? [[String: Any]])
        XCTAssertEqual(alarms.count, 2)
        let raise = alarms[0]
        XCTAssertEqual(raise["kind"] as? String, "raise")
        XCTAssertEqual(raise["rule"] as? String, "dead_channels")
        XCTAssertEqual(raise["severity"] as? String, "critical")
        XCTAssertEqual(raise["trainer_step"] as? Int, 300)
        XCTAssertEqual(raise["value"] as? String, "dead=5/1040")
        XCTAssertEqual(raise["threshold"] as? String, "site>=0.2")
        XCTAssertEqual(raise["detail"] as? String, "sites=value.bn(5/16)")
        XCTAssertEqual(raise["action"] as? String, "stop_on_critical")
        XCTAssertEqual(raise["learning_rate"] as? Double, 0.3)
        XCTAssertEqual(raise["momentum"] as? Double, 0.85)
        let clear = alarms[1]
        XCTAssertEqual(clear["since"] as? Int, 300)
        XCTAssertEqual(clear["severity"] as? String, "warning")
        XCTAssertEqual(clear["action"] as? String, "log")
    }

    func testAlarmConfigEncoding() throws {
        let recorder = CliTrainingRecorder()
        recorder.setAlarmConfig(try S.config(
            checkIntervalSteps: 500, learningGraceSteps: 200, lrWarmupSteps: 100,
            actions: TrainingHealthActions { $0 == .illegalMass ? .stopOnCritical : .log }))
        let config = try XCTUnwrap(try json(recorder)["alarm_config"] as? [String: Any])
        XCTAssertEqual(config["enabled"] as? Bool, true)
        XCTAssertEqual(config["check_interval_steps"] as? Int, 500)
        XCTAssertEqual(config["learning_grace_steps"] as? Int, 200)
        XCTAssertEqual(config["lr_warmup_steps"] as? Int, 100)
        XCTAssertEqual(config["momentum_coefficient"] as? Double, 0.85)
        let actions = try XCTUnwrap(config["actions"] as? [String: String])
        XCTAssertEqual(actions.count, TrainingHealthRule.allCases.count)
        XCTAssertEqual(actions["illegal_mass"], "stop_on_critical")
        XCTAssertEqual(actions["gradient_spike"], "log")
    }

    func testSeverityEncodesAsItsName() throws {
        let data = try JSONEncoder().encode([TrainingAlarm.Severity.warning, .critical])
        XCTAssertEqual(String(decoding: data, as: UTF8.self), #"["warning","critical"]"#)
    }

    func testTerminationReasonTrainingHealthAlarm() throws {
        XCTAssertEqual(CliTrainingRecorder.TerminationReason.trainingHealthAlarm.rawValue, "training_health_alarm")
        let recorder = CliTrainingRecorder()
        recorder.setTerminationReason(.trainingHealthAlarm)
        XCTAssertEqual(try json(recorder)["termination_reason"] as? String, "training_health_alarm")
    }
}
