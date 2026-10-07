//
//  TrainingSideReviewRegressionTests.swift
//  DrewsChessMachineTests
//
//  Regression tests for findings of the final training-side review that
//  have no natural home in a larger suite: the journal's value kinds, a
//  pre-v11 unknown writer's segment step, the BN pass-through counts and
//  the GUI step line's clock.
//

import XCTest
@testable import DrewsChessMachine

final class TrainingSideReviewRegressionTests: XCTestCase {

    /// A journalled change of a `Double` key whose value is whole (180.0)
    /// is written by `JSONEncoder` as `180`, which reads back as an `.int`
    /// unless the known key's type is applied (review m4).
    func testAParameterChangeOfAWholeDoubleRoundTripsItsKind() throws {
        let change = LineageRecord.ParameterChange(
            committedAtTrainerStep: 7, recordedUnix: 1_790_000_000, id: StepLineIntervalSec.id,
            old: StepLineIntervalSec.encode(180.0), new: StepLineIntervalSec.encode(600.0), restampedFrom: nil)
        let data = try JSONEncoder().encode(change)
        let decoded = try JSONDecoder().decode(LineageRecord.ParameterChange.self, from: data)
        XCTAssertEqual(decoded, change)
        XCTAssertEqual(decoded.old, .double(180))
    }

    /// A key this build does not know keeps the value the file states.
    func testAParameterChangeOfAnUnknownKeyKeepsItsStatedValue() throws {
        let change = LineageRecord.ParameterChange(
            committedAtTrainerStep: 7, recordedUnix: 1_790_000_000, id: "no_such_parameter",
            old: .int(1), new: .int(2), restampedFrom: nil)
        let decoded = try JSONDecoder().decode(
            LineageRecord.ParameterChange.self, from: try JSONEncoder().encode(change))
        XCTAssertEqual(decoded, change)
    }

    /// A pre-v11 file by an unknown writer with no lineage record has no
    /// known segment step: nil, never its stated step (review m6).
    func testAPreV11UnknownWriterWithoutARecordHasNoSegmentStep() throws {
        let reading = try ModelFileStepReading.reading(
            formatVersion: 6, creator: "test", statedTrainingStep: 40, trainerCompletedSteps: nil,
            recordSteps: { nil }, source: "test")
        XCTAssertEqual(reading.basis, .legacyUnknownWriter)
        XCTAssertEqual(reading.trainerStep, 40)
        XCTAssertNil(reading.segmentStep)
    }

    /// γ and β of different lengths can only come from a bug; the
    /// pass-through counts refuse them rather than counting the shorter
    /// length (review m8).
    func testPassThroughHealthRefusesMismatchedGammaAndBeta() {
        XCTAssertThrowsError(try LayerHealth.passThroughHealth(activation: .relu, gamma: [1, 1, 1], beta: [0, 0]))
    }

    /// The GUI's step line is keyed on the trainer's clock, not on the
    /// session's stats-box count, which restarts at 0 after "New Session,
    /// keep trainer" (review m5): a trainer at 5000 has no dense lines.
    func testTheGuiStepLineFollowsTheTrainerClockNotTheSessionCount() {
        var schedule = TrainingStepLineSchedule()
        XCTAssertNil(schedule.guiPollLineDue(sessionSteps: 0, trainerStep: 5_000, elapsedSec: 0,
                                             carriesDiagnostics: false, intervalSec: 600),
                     "no line before the session's first step")
        XCTAssertEqual(schedule.guiPollLineDue(sessionSteps: 1, trainerStep: 5_001, elapsedSec: 1,
                                               carriesDiagnostics: false, intervalSec: 600), .segmentStart)
        XCTAssertNil(schedule.guiPollLineDue(sessionSteps: 51, trainerStep: 5_051, elapsedSec: 2,
                                             carriesDiagnostics: true, intervalSec: 600),
                     "session step 51 is not a dense line of a trainer past step 1000")
        XCTAssertEqual(schedule.guiPollLineDue(sessionSteps: 1_000, trainerStep: 6_000, elapsedSec: 3,
                                               carriesDiagnostics: true, intervalSec: 600), .fixedStep)
        XCTAssertEqual(schedule.lastObservedTrainerStep, 6_000)
    }
}

/// Train-vs-UCI hashes the opponents' executables in its synchronous
/// pre-flight, and the lineage reads those digests instead of the files
/// (review m7: reading and hashing engine binaries inside the async run
/// blocked a cooperative-pool thread).
@MainActor
final class TrainVsUciExecutableDigestTests: XCTestCase {

    private func config(commands: [String]) -> TrainVsUciConfig {
        TrainVsUciConfig(
            opponents: commands.map {
                TrainVsUciOpponentSpec(command: $0, count: 1, goLimit: "nodes 1", options: [], kind: "k")
            },
            stepLimit: 1, timeLimitSec: nil, startModelPath: nil, resumeExact: false, acceptInexact: [],
            presetName: nil, sessionDirectory: FileManager.default.temporaryDirectory,
            saveReplayBuffer: false, enumerateCheckpoints: false, checkpointStem: nil,
            maxPliesPerGame: 400, evalSyncEverySteps: 1, runModelID: "20261007-1-DGST", output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 7, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
    }

    func testThePreflightHashesEachExecutableAndTheLineageReadsNoFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-vsuci-digest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: false)
        defer { XCTAssertNoThrow(try FileManager.default.removeItem(at: dir)) }
        let engine = dir.appendingPathComponent("engine")
        try Data("abc".utf8).write(to: engine)
        let hashed = config(commands: [engine.path, engine.path])
        let digests = try TrainVsUciOpponentExecutableDigests(hashingExecutablesOf: hashed)
        let abcSHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        XCTAssertEqual(try digests.sha256(ofExecutableNamedBy: engine.path), abcSHA256)

        try FileManager.default.removeItem(at: engine)
        let generation = try TrainVsUciSession.lineageGeneration(
            config: hashed, executableDigests: digests, trainerMoveSelection: .uniform)
        XCTAssertEqual(generation.opponents.map(\.executableSHA256), [abcSHA256, abcSHA256],
                       "the lineage uses the pre-flight's digests; the file is gone")

        XCTAssertThrowsError(try TrainVsUciSession.lineageGeneration(
            config: config(commands: ["/not/hashed"]), executableDigests: digests, trainerMoveSelection: .uniform),
                             "a command the pre-flight did not hash is an error, never a default")
    }
}
