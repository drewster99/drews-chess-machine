//
//  HealthAlarmsLineageTests.swift
//  DrewsChessMachineTests
//
//  `configuration.health_alarms` (hyperparameter recording plan P4, alarm
//  plan OD-10): a segment's training-health summary is every monitor it
//  ran merged. A GUI segment spans several monitors — each start makes one,
//  and a Continue keeps the segment — so the monitor a start replaces is
//  folded into the segment's stored summary once, at the replacement, and a
//  save records the stored summary merged with the live monitor's without
//  storing it.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class HealthAlarmsLineageTests: XCTestCase {

    typealias S = TrainingHealthTestSupport

    private static let ended = TrainingHealthSegmentSummary(evaluations: 3, raised: [
        TrainingHealthSegmentSummary.Raised(rule: .deadChannels, firstTrainerStep: 100, highestSeverity: .warning,
                                            raiseCount: 1),
    ])

    /// A monitor that has committed one live evaluation.
    private func evaluatedMonitor() throws -> TrainingHealthMonitor {
        let monitor = TrainingHealthMonitor(valueFC1Applicability: .applies)
        for step in 1...100 { monitor.recordStep(S.record(step)) }
        monitor.evaluateLive(stamp: monitor.observationStamp(),
                             layerHealth: .read(S.deadDigest(dead: 0), trainerStep: 100),
                             learningRate: 0.1, momentum: 0.85, config: try S.config(), log: S.LineSink().sink)
        XCTAssertEqual(monitor.segmentSummary().evaluations, 1)
        return monitor
    }

    func testTheTrackerMergesTheLiveSummaryWithoutStoringIt() throws {
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .gui, argv: ["dcm"],
                                         startedAt: Date(), segmentStartTrainerStep: 0)
        XCTAssertNil(tracker.healthAlarms(withLive: nil), "no monitor: nothing to state")
        XCTAssertEqual(tracker.healthAlarms(withLive: .empty), .empty)
        tracker.mergeEndedHealthMonitor(Self.ended)
        let live = TrainingHealthSegmentSummary(evaluations: 2, raised: [
            TrainingHealthSegmentSummary.Raised(rule: .deadChannels, firstTrainerStep: 40, highestSeverity: .critical,
                                                raiseCount: 2),
        ])
        let merged = try XCTUnwrap(tracker.healthAlarms(withLive: live))
        XCTAssertEqual(merged.evaluations, 5)
        XCTAssertEqual(merged.raised, [TrainingHealthSegmentSummary.Raised(
            rule: .deadChannels, firstTrainerStep: 40, highestSeverity: .critical, raiseCount: 3)])
        XCTAssertEqual(tracker.healthAlarms(withLive: .empty), Self.ended, "a cut never stores the live summary")
    }

    func testAGuiSaveRecordsTheStoredSummaryMergedWithTheLiveMonitor() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let controller = harness.controller
        let tracker = try XCTUnwrap(controller.lineageTracker)
        tracker.mergeEndedHealthMonitor(Self.ended)
        let live = try evaluatedMonitor()
        controller.trainingHealthMonitor = live

        let cut = try controller.takeConfigurationCut(trainer: harness.trainer)
        let expected = Self.ended.merging(live.segmentSummary())
        XCTAssertEqual(cut.saveInputs.healthAlarms, expected)
        XCTAssertEqual(expected.evaluations, 4)
        let record = try controller.lineageRecordForSave(
            at: Date(), cut: cut, trainerCompletedSteps: harness.trainer.completedTrainSteps,
            dropoutPhiloxState: nil, dropoutStreamState: nil)
        XCTAssertEqual(record.configuration.value?.healthAlarms, .recorded(expected))
        let text = try record.jsonText()
        XCTAssertTrue(text.contains("\"health_alarms\":{\"recorded\":true,\"value\":{\"evaluations\":4"), text)
    }

    /// A start replaces the monitor; the one it replaces is folded into the
    /// segment still held, exactly once.
    func testAStartFoldsTheMonitorItReplacesIntoTheSegment() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let controller = harness.controller
        let tracker = try XCTUnwrap(controller.lineageTracker)
        let first = try evaluatedMonitor()
        controller.trainingHealthMonitor = first

        let second = controller.beginTrainingHealthRun(arch: GuiSaveHarness.architecture, recorder: nil, autoTrainStop: nil)
        XCTAssertFalse(second === first)
        XCTAssertEqual(tracker.healthAlarms(withLive: .empty), first.segmentSummary())
        let cut = try controller.takeConfigurationCut(trainer: harness.trainer)
        XCTAssertEqual(cut.saveInputs.healthAlarms, first.segmentSummary().merging(second.segmentSummary()))

        _ = controller.beginTrainingHealthRun(arch: GuiSaveHarness.architecture, recorder: nil, autoTrainStop: nil)
        XCTAssertEqual(tracker.healthAlarms(withLive: .empty)?.evaluations, first.segmentSummary().evaluations,
                       "the second monitor evaluated nothing; the first is not folded in again")
    }
}
