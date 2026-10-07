//
//  TrainVsUciTrainedPositionsTests.swift
//  DrewsChessMachineTests
//
//  A train-vs-UCI session.json's `trainingPositionsSeen` counts the positions
//  trained over the trainer's clock at the batch each step trained at. Its
//  step count is the lifetime trainer clock, so the steps before this run —
//  which may have trained at another batch size, on another path — are
//  counted from what a record says about them, never restated at this run's
//  batch; where nothing records them the value is unrecorded.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class TrainVsUciTrainedPositionsTests: XCTestCase {

    private func state(trainerCompletedSteps: Int, trainedPositions: Int?) throws -> SessionCheckpointState {
        let parameters = try TrainingParametersSnapshot.declaredDefaults(overriding: [TrainingBatchSize.id: .int(64)])
        return TrainVsUciSession.sessionState(
            sessionID: "20261006-1-TPOS", savedAt: Date(timeIntervalSince1970: 1_800_000_100),
            runStart: Date(timeIntervalSince1970: 1_800_000_000), trainerCompletedSteps: trainerCompletedSteps,
            trainedPositions: trainedPositions,
            parameters: parameters, hyperparameters: TrainerHyperparameters(parameters),
            arch: ResumeEquivalenceTests.architecture, bufferSnapshot: nil, maxPliesPerGame: 400)
    }

    // MARK: - Regression

    func testTheRecordedCountIsWrittenNotTheClockTimesThisRunsBatch() throws {
        let recorded = try state(trainerCompletedSteps: 1000, trainedPositions: 12_345)
        XCTAssertEqual(recorded.trainingPositionsSeen, 12_345)
        let unrecorded = try state(trainerCompletedSteps: 1000, trainedPositions: nil)
        XCTAssertNil(unrecorded.trainingPositionsSeen, "steps no record describes are unrecorded, never 1000 × 64")
    }

    // MARK: - The run's count

    func testAFreshTrainerCountsFromZero() throws {
        let count = TrainVsUciSession.trainedPositionsCount(startTrainerSteps: 0, startSession: nil, batchSize: 64)
        XCTAssertEqual(try count.positions(atSteps: 10), 640)
    }

    func testASessionCoveringTheStartClockCarriesItsCount() throws {
        let resumed = try state(trainerCompletedSteps: 1000, trainedPositions: 32_000)
        let count = TrainVsUciSession.trainedPositionsCount(startTrainerSteps: 1000, startSession: resumed, batchSize: 64)
        XCTAssertEqual(try count.positions(atSteps: 1010), 32_000 + 10 * 64,
                       "the session's 1000 steps at the batch they trained at, then this run's at 64")
    }

    func testStepsNoRecordCoversAreUnrecorded() throws {
        let fromModelFile = TrainVsUciSession.trainedPositionsCount(startTrainerSteps: 1000, startSession: nil, batchSize: 64)
        XCTAssertNil(try fromModelFile.positions(atSteps: 1010), "a model file records no positions trained")
        let keptTrainerSession = try state(trainerCompletedSteps: 300, trainedPositions: 9_600)
        let count = TrainVsUciSession.trainedPositionsCount(
            startTrainerSteps: 1000, startSession: keptTrainerSession, batchSize: 64)
        XCTAssertNil(try count.positions(atSteps: 1010),
                     "a session counting fewer steps than the trainer's clock does not cover it")
        let unrecordedSession = try state(trainerCompletedSteps: 1000, trainedPositions: nil)
        XCTAssertNil(try TrainVsUciSession.trainedPositionsCount(
            startTrainerSteps: 1000, startSession: unrecordedSession, batchSize: 64).positions(atSteps: 1010))
    }

    func testAStepCountBeforeTheStartThrows() {
        let count = TrainVsUciSession.trainedPositionsCount(startTrainerSteps: 100, startSession: nil, batchSize: 64)
        XCTAssertThrowsError(try count.positions(atSteps: 99))
    }
}
