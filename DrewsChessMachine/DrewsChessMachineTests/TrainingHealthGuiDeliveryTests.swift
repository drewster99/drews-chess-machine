import XCTest
@testable import DrewsChessMachine

/// The GUI side of a training-health evaluation's delivery
/// (`SessionController.deliverTrainingHealth(from:)`, alarms plan R2, R3,
/// T8): the identity check against the current start's monitor, the list
/// refresh, and the stop decision from the actions in force when the result
/// arrives. Settings are restored with persistence suppressed, so the
/// user's saved settings are never written.
@MainActor
final class TrainingHealthGuiDeliveryTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        TrainingParameters.suppressPersistence = true
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    /// A GUI monitor (stops decided by the caller) with `dead_channels`
    /// active at critical.
    private func monitorWithCriticalDeadChannels(
        actions: TrainingHealthActions = TrainingHealthActions { _ in .stopOnCritical }
    ) throws -> TrainingHealthMonitor {
        let monitor = TrainingHealthMonitor(valueFC1Applicability: .applies, stopDecision: .byCaller)
        let evaluation = monitor.evaluateCheckpoint(
            stamp: monitor.observationStamp(),
            layerHealth: TrainingHealthTestSupport.deadDigest(
                dead: 8, channels: 1040, sites: [("value.bn", 8, 16)], tier: .checkpoint),
            digestTrainerStep: 300,
            config: try TrainingHealthTestSupport.config(actions: actions),
            log: { _ in })
        let committed = try XCTUnwrap(evaluation)
        XCTAssertNil(committed.stopRequest, "a caller-decided monitor never requests a stop itself")
        XCTAssertFalse(committed.events.contains { $0.kind == .stop }, "nor writes a stop line")
        return monitor
    }

    func testStaleHopFromAnEarlierMonitorIsIgnored() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let alarms = TrainingAlarmController()
        harness.controller.trainingAlarm = alarms
        let earlier = try monitorWithCriticalDeadChannels()
        harness.controller.trainingHealthMonitor = TrainingHealthMonitor(
            valueFC1Applicability: .applies, stopDecision: .byCaller)
        TrainingParameters.shared.trainingHealthActionDeadChannels = .stopOnCritical

        harness.controller.deliverTrainingHealth(from: earlier)

        XCTAssertTrue(alarms.healthAlarms.isEmpty, "an earlier start's monitor must not reach the list")
        XCTAssertNil(harness.controller.trainingSuspension, "nor stop the current run")
        XCTAssertFalse(earlier.parkRequested)
    }

    func testStopActionSuspendsTrainingAndParksTheWorker() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let alarms = TrainingAlarmController()
        harness.controller.trainingAlarm = alarms
        // Evaluated under `log` (an older action), delivered after the
        // action changed to a stop: the action in force at delivery governs.
        let monitor = try monitorWithCriticalDeadChannels(actions: TrainingHealthActions { _ in .log })
        harness.controller.trainingHealthMonitor = monitor
        TrainingParameters.shared.trainingHealthActionDeadChannels = .stopOnCritical

        harness.controller.deliverTrainingHealth(from: monitor)

        XCTAssertEqual(alarms.healthAlarms.map(\.rule), [.deadChannels])
        guard case .healthAlarm(let rule, _)? = harness.controller.trainingSuspension else {
            return XCTFail("training must be suspended by the health alarm, got \(String(describing: harness.controller.trainingSuspension))")
        }
        XCTAssertEqual(rule, .deadChannels)
        XCTAssertTrue(monitor.parkRequested, "the worker is asked to park, not to return")
        XCTAssertEqual(alarms.healthSuspendedRule, .deadChannels)
        alarms.refreshHealth([])
    }

    func testLogActionAtDeliveryMeansNoStop() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let alarms = TrainingAlarmController()
        harness.controller.trainingAlarm = alarms
        // Evaluated under a stop action, delivered after it changed to log.
        let monitor = try monitorWithCriticalDeadChannels(actions: TrainingHealthActions { _ in .stopOnCritical })
        harness.controller.trainingHealthMonitor = monitor
        TrainingParameters.shared.trainingHealthActionDeadChannels = .log

        harness.controller.deliverTrainingHealth(from: monitor)

        XCTAssertEqual(alarms.healthAlarms.map(\.rule), [.deadChannels], "still listed")
        XCTAssertNil(harness.controller.trainingSuspension)
        XCTAssertFalse(monitor.parkRequested)
        alarms.refreshHealth([])
    }

    func testDisabledAlarmsNeverStop() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let monitor = try monitorWithCriticalDeadChannels()
        harness.controller.trainingHealthMonitor = monitor
        TrainingParameters.shared.trainingHealthActionDeadChannels = .stopOnAny
        TrainingParameters.shared.trainingHealthAlarmsEnabled = false

        harness.controller.deliverTrainingHealth(from: monitor)

        XCTAssertNil(harness.controller.trainingSuspension)
    }

    /// Stop after a health suspension (final review M1): the ended run's
    /// critical alarms stay listed for review but never sound again, a late
    /// delivery from that run's monitor does not restart the beep, and the
    /// "suspended" header is cleared.
    func testStopSilencesTheEndedRunsHealthAlarmsAndClearsTheSuspendedHeader() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let alarms = TrainingAlarmController()
        harness.controller.trainingAlarm = alarms
        let monitor = try monitorWithCriticalDeadChannels()
        harness.controller.trainingHealthMonitor = monitor
        TrainingParameters.shared.trainingHealthActionDeadChannels = .stopOnCritical
        harness.controller.deliverTrainingHealth(from: monitor)
        XCTAssertEqual(alarms.healthSuspendedRule, .deadChannels)
        alarms.silence()

        harness.controller.stopRealTraining()

        XCTAssertNil(harness.controller.trainingSuspension)
        XCTAssertNil(alarms.healthSuspendedRule, "the suspended header ends with the run")
        XCTAssertEqual(alarms.healthAlarms.map(\.rule), [.deadChannels], "kept for review")
        XCTAssertFalse(alarms.shouldSound, "Stop must not restart the beep for the ended run's alarms")
        harness.controller.deliverTrainingHealth(from: monitor)
        XCTAssertFalse(alarms.shouldSound, "nor may a late delivery from the ended run")
        alarms.refreshHealth([])
    }

    /// A save during a health suspension (final review m1) closes the
    /// training segment and leaves it closed: the parked idle is not
    /// training wall time.
    func testASaveDuringAHealthSuspensionDoesNotReopenTheTrainingSegment() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let checkpoint = CheckpointController()
        harness.controller.checkpoint = checkpoint
        checkpoint.beginActiveTrainingSegment()
        let monitor = try monitorWithCriticalDeadChannels()
        harness.controller.trainingHealthMonitor = monitor
        TrainingParameters.shared.trainingHealthActionDeadChannels = .stopOnCritical
        harness.controller.deliverTrainingHealth(from: monitor)
        XCTAssertNotNil(harness.controller.trainingSuspension)
        XCTAssertNil(checkpoint.activeSegmentStart, "the suspension closes the segment")

        _ = try harness.controller.buildCurrentSessionState(
            championID: "c", trainerID: "t", arenaClock: .live, includeReplayBuffer: false)

        XCTAssertNil(checkpoint.activeSegmentStart, "a save while parked must not reopen the segment")
        withExtendedLifetime(checkpoint) {}
    }

    /// The parked worker acknowledges a training pause, so a session save
    /// (which waits for that acknowledgement) completes during a health
    /// suspension; Stop (cancellation) ends the parked loop.
    func testParkedWorkerAcknowledgesPausesAndEndsOnCancel() async throws {
        let gate = WorkerPauseGate()
        let parked = Task { await GuiTrainingHealthWorker.park(at: gate) }
        let acquired = await gate.pauseAndWait(timeoutMs: 5_000)
        XCTAssertTrue(acquired, "the parked worker must acknowledge the pause")
        gate.resume()
        parked.cancel()
        await parked.value
    }
}
