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

    /// Every count session.json carries — the step count, the self-play and
    /// fed game counts — describes the save's cut, the instant the trainer
    /// file, the run's record and the buffer describe, and so does the
    /// trainer file's training step. The save used to read them before it
    /// paused anything, from the counters the heartbeat last published, so
    /// a resume restored the run's counters from an earlier instant than
    /// its trainer clock and fed totals.
    func testASessionSavesCountsDescribeItsCut() async throws {
        harness.publishCountersLikeTheHeartbeat()
        let publishedSerial = harness.serials.nextSerial
        let publishedSteps = harness.trainer.completedTrainSteps
        while harness.serials.nextSerial < publishedSerial + 10 || harness.trainer.completedTrainSteps < publishedSteps + 10 {
            try await Task.sleep(for: .milliseconds(10))
        }
        let saved = await harness.save(trigger: .manual, includeReplayBuffer: false)
        XCTAssertTrue(saved)
        let cut = try onlyObservation()
        let (record, loaded) = try savedTrainerRecord()
        let trainerMetadata = loaded.trainerFile.metadata

        XCTAssertEqual(trainerMetadata.trainerSchedule?.completedTrainSteps, cut.trainerCompletedSteps,
                       "the trainer file's clock at the cut")
        XCTAssertEqual(record.steps.cumTrainerStep, cut.trainerCompletedSteps, "the record's trainer step at the cut")
        XCTAssertEqual(cut.trainingSteps, cut.trainerCompletedSteps, "the run's step count and the trainer clock agree")
        XCTAssertEqual(loaded.state.trainingSteps, cut.trainingSteps, "session.json's step count at the cut")
        XCTAssertEqual(trainerMetadata.trainingStep, cut.trainingSteps, "the trainer file's training step at the cut")
        XCTAssertEqual(loaded.state.selfPlayGames, cut.selfPlayGames, "session.json's self-play games at the cut")
        XCTAssertEqual(loaded.state.emittedGames, cut.emittedGames, "session.json's fed games at the cut")
        XCTAssertEqual(loaded.state.emittedPositions, cut.emittedPositions, "session.json's fed positions at the cut")
        XCTAssertEqual(loaded.state.emittedGames, record.fed.segmentGames,
                       "a fresh run's fed games in session.json and in the record")
    }

    /// Promote Trainee Now records its history entry at the step of its own
    /// pause, and its autosave's counts describe that save's cut. The entry
    /// used to take its step from the counters the heartbeat last published.
    func testPromoteTraineeNowRecordsItsStepAtItsPause() async throws {
        harness.publishCountersLikeTheHeartbeat()
        let publishedSteps = harness.trainer.completedTrainSteps
        while harness.trainer.completedTrainSteps < publishedSteps + 10 {
            try await Task.sleep(for: .milliseconds(10))
        }
        harness.controller.promoteTrainerNow(sessionsDirectory: harness.sessionsDirectory)
        let savedURL = try await harness.waitForSavedSession(timeout: 30)
        let url = try XCTUnwrap(savedURL, "the promotion's autosave was not written")
        let observations = harness.trainerPauseObservations.value
        XCTAssertEqual(observations.count, 2, "the promotion and its autosave each pause training once")
        let promotionCut = try XCTUnwrap(observations.first)
        let saveCut = try XCTUnwrap(observations.last)
        let loaded = try CheckpointManager.loadSession(at: url)
        XCTAssertEqual(loaded.state.arenaHistory.last?.finishedAtStep, promotionCut.trainingSteps,
                       "the promotion's history entry at the promotion's pause")
        XCTAssertEqual(loaded.state.trainingSteps, saveCut.trainingSteps, "the autosave's step count at its cut")
        XCTAssertEqual(loaded.state.emittedGames, saveCut.emittedGames, "the autosave's fed games at its cut")
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
