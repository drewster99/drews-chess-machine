//
//  TrainVsUciRefusalTests.swift
//  DrewsChessMachineTests
//
//  A train-vs-UCI run refused at launch throws a `CLIRunRefusal` out of
//  `TrainVsUciRunner.runTraining` instead of ending the process; only
//  `runAndExit` turns it into stderr text and exit status 2, after draining
//  the session log. The refusal here comes before any network is built or
//  engine launched, so the opponent's executable is never started.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class TrainVsUciRefusalTests: XCTestCase {

    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-vsuci-refusal-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
    }

    override func tearDown() async throws {
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    func testUnknownPresetThrows() async throws {
        let sessions = tempDir.appendingPathComponent("Sessions", isDirectory: true)
        let neverStarted = tempDir.appendingPathComponent("never-started")
        try Data("not an engine".utf8).write(to: neverStarted)
        let config = TrainVsUciConfig(
            opponents: [TrainVsUciOpponentSpec(command: neverStarted.path,
                                               count: 1, goLimit: "nodes 1", options: [], kind: "never-started")],
            stepLimit: 1,
            timeLimitSec: nil,
            startModelPath: nil,
            resumeExact: false,
            acceptInexact: [],
            presetName: "no_such_preset",
            sessionDirectory: sessions,
            saveReplayBuffer: false,
            enumerateCheckpoints: false,
            checkpointStem: nil,
            maxPliesPerGame: 400,
            evalSyncEverySteps: 1,
            runModelID: "20261003-5-RFSV",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 0x5EF6, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
        let params = try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [:]))
        do {
            _ = try await TrainVsUciRunner.runTraining(
                config: config, params: params,
                executableDigests: try TrainVsUciOpponentExecutableDigests(hashingExecutablesOf: config),
                abort: TrainVsUciAbortFlag())
            XCTFail("a run with an unknown preset trained")
        } catch let refusal as CLIRunRefusal {
            XCTAssertTrue(refusal.message.contains("unknown --preset 'no_such_preset'"), refusal.message)
        } catch {
            XCTFail("the run failed instead of being refused: \(error.localizedDescription)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: sessions.path), "a refused run writes no session")
    }
}
