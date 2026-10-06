//
//  RunStartParameterCaptureTests.swift
//  DrewsChessMachineTests
//
//  The batch size, pre-train fill and replay-buffer capacity a GUI run
//  captured at its start are what its saves record, whatever the settings
//  popover wrote to `TrainingParameters.shared` since.
//
//  Every test that assigns `TrainingParameters.shared` snapshots it in setUp,
//  suppresses persistence, and restores it in tearDown, so nothing leaks into
//  another test or into the user's saved settings.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class RunStartParameterCaptureTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    func testInForceReplacesOnlyTheCapturedKeys() throws {
        let snapshot = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(4096),
            ReplayBufferMinPositionsBeforeTraining.id: .int(250_000),
            ReplayBufferCapacity.id: .int(1_000_000),
        ])
        let capture = RunStartParameterCapture(
            trainingBatchSize: 1024, replayBufferMinPositionsBeforeTraining: 777, replayBufferCapacity: 123_456)
        let inForce = capture.inForce(over: snapshot)
        XCTAssertEqual(inForce.trainingBatchSize, 1024)
        XCTAssertEqual(inForce.replayBufferMinPositionsBeforeTraining, 777)
        XCTAssertEqual(inForce.replayBufferCapacity, 123_456)
        let before = snapshot.rawValueMap()
        let after = inForce.rawValueMap()
        XCTAssertEqual(Set(after.keys), Set(before.keys), "no key added or dropped")
        for key in TrainingParameters.allKeys where !RunStartParameterCapture.capturedKeyIDs.contains(key.id) {
            XCTAssertEqual(after[key.id], before[key.id], "\(key.id) is not captured and is unchanged")
        }
        XCTAssertEqual(RunStartParameterCapture.capturedKeyIDs.count, 3)
    }

    func testCapturedKeysAreNotLiveTunable() {
        for key in TrainingParameters.allKeys where RunStartParameterCapture.capturedKeyIDs.contains(key.id) {
            XCTAssertFalse(key.definition.liveTunable,
                           "\(key.id) is read once at the run's start, so the capture describes the whole run")
        }
        XCTAssertEqual(TrainingParameters.allKeys.filter { RunStartParameterCapture.capturedKeyIDs.contains($0.id) }.count, 3,
                       "every captured id is a declared parameter")
    }

    /// The batch size in a saved record's parameter snapshot.
    private func recordedValue(_ record: LineageRecord, _ id: String) throws -> Int {
        let parameters = try XCTUnwrap(record.parameters)
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(parameters.snapshotJSON.utf8)) as? [String: Any])
        return try XCTUnwrap(object[id] as? Int, "\(id) in \(parameters.snapshotJSON)")
    }

    func testAGuiSaveRecordsTheBatchSizeTheRunTrainsAt() throws {
        TrainingParameters.shared.trainingBatchSize = 64
        TrainingParameters.shared.replayBufferMinPositionsBeforeTraining = 300
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        // The popover writes the settings during the run.
        TrainingParameters.shared.trainingBatchSize = 128
        TrainingParameters.shared.replayBufferMinPositionsBeforeTraining = 900
        TrainingParameters.shared.replayBufferCapacity = harness.buffer.capacity * 2

        let record = try harness.controller.lineageRecordForSave(
            at: Date(), trainerCompletedSteps: harness.trainer.completedTrainSteps,
            dropoutPhiloxState: nil, dropoutStreamState: nil)
        XCTAssertEqual(try recordedValue(record, TrainingBatchSize.id), 64, "the batch size the run steps at")
        XCTAssertEqual(try recordedValue(record, ReplayBufferMinPositionsBeforeTraining.id), 300,
                       "the pre-train fill the run waited for")
        XCTAssertEqual(try recordedValue(record, ReplayBufferCapacity.id), harness.buffer.capacity,
                       "the capacity of the run's buffer")
    }

    func testSessionStateBatchSizeIsTheRunsBatchSize() throws {
        TrainingParameters.shared.trainingBatchSize = 64
        TrainingParameters.shared.replayBufferMinPositionsBeforeTraining = 300
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        harness.controller.trainingStats = TrainingRunStats()
        harness.controller.trainingStats?.steps = 10
        TrainingParameters.shared.trainingBatchSize = 128
        TrainingParameters.shared.replayBufferMinPositionsBeforeTraining = 900

        let state = try harness.controller.buildCurrentSessionState(
            championID: "c", trainerID: "t", arenaClock: .live, includeReplayBuffer: false)
        XCTAssertEqual(state.batchSize, 64)
        XCTAssertEqual(state.trainingPositionsSeen, 10 * 64)
    }
}
