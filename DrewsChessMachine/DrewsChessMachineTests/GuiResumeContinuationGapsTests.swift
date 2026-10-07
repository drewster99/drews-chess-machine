//
//  GuiResumeContinuationGapsTests.swift
//  DrewsChessMachineTests
//
//  What a GUI resume reports as not restored must follow what the resume
//  actually restored, not what the session file contains. A session whose
//  trainer file records the run's streams still does not continue them when
//  `--seed` names another seed or the streams were named under another
//  derivation (the run then draws a seed and starts its serials at 0), and
//  a session whose replay buffer file fails to restore (corrupt, or paired
//  with another session's session.json) trains on an empty buffer. Both
//  used to report EXACT, and the false verdict was written into every
//  later save's `not_exact_items`.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class GuiResumeContinuationGapsTests: XCTestCase {

    private let arch = NetworkArchitecture.current
    // The architecture's own tail: from format v12 the record's configuration
    // is composed from it.
    private let savedPrecision = NetworkArchitecture.current.policyTailPrecision
    private static let fingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab")

    /// A trainer file whose record is a GUI save's, with the run's streams.
    private func trainerFileWithStreams() throws -> ModelCheckpointFile {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .gui, argv: ["DrewsChessMachine"],
                                         startedAt: start, segmentStartTrainerStep: 0)
        let seed = RunRandomSeed.resolve(mode: .seeded, configuredSeed: 7, commandLineSeed: nil, drawSeed: { 0 })
        try tracker.noteSegmentStartForTests(trainerStep: 0, policyTailPrecision: savedPrecision, seed: seed)
        let streams = seed.runStreams(samplerState: seed.streams.generator(.sampler),
                                      dropoutStreamState: seed.streams.generator(.dropout),
                                      nextGameSerial: 12, arenasStarted: 3, opponentGameIndices: nil)
        let record = try tracker.record(
            at: start.addingTimeInterval(60), trainerCompletedSteps: 4, segmentLocalStep: 4,
            segmentGames: 2, segmentPositions: 120, corpus: nil,
            parameters: try .forTests(adopting: TrainerScheduleState(completedTrainSteps: 4, lrWarmupSteps: 3, lrMomentumCycle: .disabled),
                                      overriding: ["learning_rate": .double(0.0005)]),
            rng: LineageRecord.RNG(dropoutPhiloxState: try DropoutPhiloxState(words: [1, 2, 3, 4, 5, 6, 7]),
                                   streams: streams, behaviorFingerprint: Self.fingerprint),
            inputs: tracker.testInputs)
        let weights = arch.weightTensorPlan().enumerated().map { i, spec in
            (0..<spec.elementCount).map { Float((i * 5 + $0) % 11) * 0.01 }
        } + arch.trainableTensorPlan().map { [Float](repeating: -0.25, count: $0.elementCount) }
        let data = try SafetensorsModelIO.encode(
            modelID: "20261003-1-GUIC", createdAtUnix: 1_790_000_060,
            metadata: ModelCheckpointMetadata.trainerFile(
                creator: "manual", trainingStep: 4, parentModelID: "", notes: "gui resume continuation test",
                schedule: TrainerScheduleState(completedTrainSteps: 4, lrWarmupSteps: 3, lrMomentumCycle: .disabled)),
            weights: weights, architecture: arch, includesVelocity: true, lineage: record)
        return try SafetensorsModelIO.decode(data).file
    }

    /// A session with a replay buffer file and an arena clock: nothing is
    /// missing from the file.
    private func completeSession(file: ModelCheckpointFile) -> LoadedSession {
        let state = SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "gui-continuation", savedAtUnix: 1_790_000_060, sessionStartUnix: 1_790_000_000,
            elapsedTrainingSec: 60, trainingSteps: 4, selfPlayGames: 2, selfPlayMoves: 120,
            trainingPositionsSeen: 4 * 4096, batchSize: 4096, learningRate: 5e-4,
            promoteThreshold: 0.55, arenaGames: 200,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 4, championID: "champ", trainerID: "train", arenaHistory: []
        ).withLineage(LineageRecord.sessionTestFixture).withArenaClock(secondsSinceLastArena: 30)
        let directory = URL(fileURLWithPath: "/nonexistent/gui-continuation.dcmsession")
        return LoadedSession(
            directoryURL: directory, state: state, championFile: file, trainerFile: file,
            replayBufferURL: directory.appendingPathComponent("replay_buffer.bin"), chartDataURLs: nil)
    }

    private func gaps(_ resumed: LoadedSession, continuedRunStreams: LineageRecord.RunStreams?,
                      replayBufferRestored: Bool) throws -> [String] {
        let gaps = SessionController.guiResumeGaps(
            resumed: resumed, continuedRunStreams: continuedRunStreams, replayBufferRestored: replayBufferRestored,
            runningBuild: try .current, runningDevice: .current,
            runningFingerprint: Self.fingerprint)
        return ResumeExactness.resume(of: resumed.trainerFile.lineageParent, gaps: gaps).tokens
    }

    func testStreamsContinuedAndBufferRestoredResumeExactly() throws {
        let resumed = completeSession(file: try trainerFileWithStreams())
        let streams = try XCTUnwrap(SessionController.resumableRunStreams(of: resumed))
        XCTAssertEqual(try gaps(resumed, continuedRunStreams: streams, replayBufferRestored: true), [])
    }

    func testStreamsInTheFileButNotContinuedAreNamed() throws {
        let resumed = completeSession(file: try trainerFileWithStreams())
        XCTAssertEqual(try gaps(resumed, continuedRunStreams: nil, replayBufferRestored: true), ["rng_sampler", "serials"])
    }

    func testFailedBufferRestoreIsNamed() throws {
        let resumed = completeSession(file: try trainerFileWithStreams())
        let streams = try XCTUnwrap(SessionController.resumableRunStreams(of: resumed))
        XCTAssertEqual(try gaps(resumed, continuedRunStreams: streams, replayBufferRestored: false), ["buffer"])
    }
}
