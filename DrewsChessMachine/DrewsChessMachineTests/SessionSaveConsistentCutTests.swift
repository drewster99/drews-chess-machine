//
//  SessionSaveConsistentCutTests.swift
//  DrewsChessMachineTests
//
//  A GUI session save records the trainer state together with the run's
//  stream positions — the replay buffer's sampler, the next self-play game
//  serial — and the games the segment fed. A resume reports those streams
//  as continued, so they must come from the same instant as the trainer
//  state: with self-play and training both held. The save used to release
//  self-play before it paused training and read the streams after both had
//  resumed, so self-play had already started new games (taking serials) and
//  the trainer had drawn more minibatches; a buffer-included save also
//  wrote games past that point. These tests drive the real save path with
//  fake workers that behave like the real ones at their pause gates.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class SessionSaveConsistentCutTests: XCTestCase {

    private var harness: GuiSaveHarness!

    override func setUp() async throws {
        try await super.setUp()
        harness = try GuiSaveHarness()
        harness.startWorkers()
        while harness.serials.nextSerial < 20 {
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    override func tearDown() async throws {
        harness.stopWorkers()
        try harness.removeSessionsDirectory()
        harness = nil
        try await super.tearDown()
    }

    private func onlyObservation(file: StaticString = #filePath, line: UInt = #line) throws -> GuiSaveHarness.TrainerPauseObservation {
        let observations = harness.trainerPauseObservations.value
        XCTAssertEqual(observations.count, 1, "the save pauses training once", file: file, line: line)
        return try XCTUnwrap(observations.first, file: file, line: line)
    }

    private func savedTrainerRecord() throws -> (LineageRecord, LoadedSession) {
        let sessions = try harness.savedSessions()
        XCTAssertEqual(sessions.count, 1)
        let loaded = try CheckpointManager.loadSession(at: try XCTUnwrap(sessions.first))
        return (try XCTUnwrap(loaded.trainerFile.safetensorsProvenance?.lineage.record), loaded)
    }

    func testASessionSaveKeepsSelfPlayPausedUntilTheTrainerIsPaused() async throws {
        let saved = await harness.save(trigger: .manual, includeReplayBuffer: false)
        XCTAssertTrue(saved)
        XCTAssertTrue(try onlyObservation().selfPlayHeld,
                      "self-play must still be held when training acknowledges the save's pause")
    }

    func testASessionSaveRecordsTheSamplerSerialAndFedCountsAtItsPause() async throws {
        let saved = await harness.save(trigger: .manual, includeReplayBuffer: false)
        XCTAssertTrue(saved)
        let cut = try onlyObservation()
        let (record, _) = try savedTrainerRecord()
        let streams = try XCTUnwrap(record.rng.streams)
        XCTAssertEqual(streams.samplerState, cut.samplerState, "sampler position at the cut")
        XCTAssertEqual(streams.nextGameSerial, cut.nextGameSerial, "next self-play game serial at the cut")
        XCTAssertEqual(record.fed.segmentGames, cut.emittedGames, "games fed by the cut")
        XCTAssertEqual(record.fed.segmentPositions, cut.emittedPositions, "positions fed by the cut")
    }

    func testABufferIncludedSaveWritesTheBufferAtTheCut() async throws {
        let saved = await harness.save(trigger: .manual, includeReplayBuffer: true)
        XCTAssertTrue(saved)
        let cut = try onlyObservation()
        let (_, loaded) = try savedTrainerRecord()
        XCTAssertEqual(loaded.state.replayBufferTotalPositionsAdded, cut.totalPositionsAdded,
                       "the written buffer holds exactly the positions added by the cut")
    }

    /// A save that fails after taking its cut still releases both workers.
    func testAFailedSaveReleasesBothWorkers() async throws {
        harness.controller.runBehaviorFingerprint = nil
        let saved = await harness.save(trigger: .manual, includeReplayBuffer: true)
        XCTAssertFalse(saved)
        XCTAssertFalse(harness.selfPlayGate.isRequestedToPause)
        XCTAssertFalse(harness.trainingGate.isRequestedToPause)
        let serialAfterFailure = harness.serials.nextSerial
        let deadline = Date().addingTimeInterval(10)
        while harness.serials.nextSerial == serialAfterFailure && Date() < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertGreaterThan(harness.serials.nextSerial, serialAfterFailure, "self-play runs again after the failed save")
    }
}
