//
//  TrainVsUciSessionStateTests.swift
//  DrewsChessMachineTests
//
//  A train-vs-UCI session's `session.json` records the run's own game ply
//  cap (`--max-plies`), not the self-play setting this path never reads.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class TrainVsUciSessionStateTests: XCTestCase {

    func testSessionStateRecordsTheRunsPlyCap() throws {
        let parameters = try TrainingParametersSnapshot.declaredDefaults(overriding: [:])
        XCTAssertNotEqual(parameters.selfPlayMaxPliesPerGame, 123, "the run's cap must differ from the self-play setting")
        let state = TrainVsUciSession.sessionState(
            sessionID: "20261006-1-PLYC", savedAt: Date(timeIntervalSince1970: 1_800_000_100),
            runStart: Date(timeIntervalSince1970: 1_800_000_000), trainerCompletedSteps: 0, trainedPositions: 0,
            parameters: parameters, hyperparameters: TrainerHyperparameters(parameters),
            arch: ResumeEquivalenceTests.architecture, bufferSnapshot: nil, maxPliesPerGame: 123)
        XCTAssertEqual(state.maxPliesPerGame, 123, "session.json records the ply cap the run's games were played to")
    }
}
