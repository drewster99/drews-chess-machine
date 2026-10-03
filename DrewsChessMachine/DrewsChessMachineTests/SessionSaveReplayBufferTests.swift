//
//  SessionSaveReplayBufferTests.swift
//  DrewsChessMachineTests
//
//  Determinism plan D-8: a session save writes the replay buffer only when
//  asked — `session_save_include_replay_buffer` for automatic saves, the Save
//  Session sheet's checkbox for a manual one — and its session.json and log
//  line say which.
//

import XCTest
@testable import DrewsChessMachine
import TrainingParametersMacroSupport

@MainActor
final class SessionSaveReplayBufferTests: XCTestCase {

    func testTheParameterIsDeclaredOffByDefaultAndKeepsTheLiveSettingOnAnOldResume() {
        let definition = SessionSaveIncludeReplayBuffer.definition
        XCTAssertEqual(SessionSaveIncludeReplayBuffer.id, "session_save_include_replay_buffer")
        XCTAssertEqual(SessionSaveIncludeReplayBuffer.declaredDefault, false)
        XCTAssertEqual(definition.category, "Sessions")
        XCTAssertTrue(definition.liveTunable)
        XCTAssertEqual(SessionSaveIncludeReplayBuffer.absentValue, .currentSetting)
        XCTAssertEqual(TrainingParameters.allKeys.filter { $0.id == SessionSaveIncludeReplayBuffer.id }.count, 1)
    }

    private func controllerWithBuffer() -> (SessionController, ReplayBuffer) {
        let controller = SessionController()
        let buffer = ReplayBuffer(capacity: 64, inputEncoding: .basic30, sampler: DCMRandom(seed: 3))
        controller.replayBuffer = buffer
        return (controller, buffer)
    }

    /// The save's session.json describes the buffer only when the save
    /// writes it, and records the automatic-save setting either way.
    func testTheSessionStateDescribesTheBufferOnlyWhenTheSaveIncludesIt() throws {
        let (controller, buffer) = controllerWithBuffer()
        let omitted = controller.buildCurrentSessionState(championID: "c", trainerID: "t", arenaClock: .live,
                                                          includeReplayBuffer: false)
        XCTAssertEqual(omitted.hasReplayBuffer, false)
        XCTAssertNil(omitted.replayBufferCapacity)
        XCTAssertEqual(omitted.sessionSaveIncludeReplayBuffer, TrainingParameters.shared.sessionSaveIncludeReplayBuffer)

        let included = controller.buildCurrentSessionState(championID: "c", trainerID: "t", arenaClock: .live,
                                                           includeReplayBuffer: true)
        XCTAssertEqual(included.hasReplayBuffer, true)
        XCTAssertEqual(included.replayBufferCapacity, buffer.capacity)

        let decoded = try SessionCheckpointState.decode(
            try included.withLineage(LineageRecord.sessionTestFixture).encode())
        XCTAssertEqual(decoded.sessionSaveIncludeReplayBuffer, included.sessionSaveIncludeReplayBuffer)
    }

    func testTheSaveLineSaysWhetherTheBufferWasWritten() {
        let (_, buffer) = controllerWithBuffer()
        XCTAssertEqual(SessionController.savedReplayBufferLogFields(writtenBuffer: nil), " buffer=omitted")
        XCTAssertEqual(SessionController.savedReplayBufferLogFields(writtenBuffer: buffer),
                       " buffer=included replay=0/\(buffer.capacity)")
    }

    func testTheSheetStartsFromTheSettingAndShowsTheBuffersSizeInBase2Units() {
        let (controller, _) = controllerWithBuffer()
        let request = controller.saveSessionSheetRequest()
        XCTAssertEqual(request.initialIncludeReplayBuffer, TrainingParameters.shared.sessionSaveIncludeReplayBuffer)
        XCTAssertEqual(request.replayBufferSizeText, "≈ 0 B")
    }

    func testByteCountsUseBase2Units() {
        XCTAssertEqual(BinaryByteCount.text(512), "512 B")
        XCTAssertEqual(BinaryByteCount.text(1024), "1.0 KB")
        XCTAssertEqual(BinaryByteCount.text(3 * 1024 * 1024 / 2), "1.5 MB")
        XCTAssertEqual(BinaryByteCount.text(Int(7.3 * 1024 * 1024 * 1024)), "7.3 GB")
        XCTAssertEqual(BinaryByteCount.text(150 * 1024 * 1024 * 1024), "150 GB")
    }
}
