//
//  GuiResumeGapsTests.swift
//  DrewsChessMachineTests
//
//  Determinism plan P9 / D-1: what a GUI session resume reports as not
//  restored. A GUI resume never refuses, so this report is the only place a
//  missing piece shows — each gap must appear exactly when its piece is
//  missing.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class GuiResumeGapsTests: XCTestCase {

    private let arch = NetworkArchitecture.current
    // The architecture's own tail: from format v12 the record's configuration
    // is composed from it.
    private let savedPrecision = NetworkArchitecture.current.policyTailPrecision
    private static let fingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab")

    private func trainerWeights() -> [[Float]] {
        arch.weightTensorPlan().enumerated().map { i, spec in
            (0..<spec.elementCount).map { Float((i * 5 + $0) % 11) * 0.01 }
        } + arch.trainableTensorPlan().map { [Float](repeating: -0.25, count: $0.elementCount) }
    }

    /// A trainer file whose record is a GUI save's: parameters, the dropout
    /// Philox state and — when `withStreams` — the run's streams with the
    /// self-play serial and arena count.
    private func trainerFile(withStreams: Bool) throws -> ModelCheckpointFile {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests, naming: .unnamedWithoutPreset), pathKind: .gui, argv: ["DrewsChessMachine"],
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
                                   streams: withStreams ? streams : nil,
                                   behaviorFingerprint: Self.fingerprint),
            inputs: tracker.testInputs)
        let data = try SafetensorsModelIO.encode(
            modelID: "20261002-1-GUIR", createdAtUnix: 1_790_000_060,
            metadata: ModelCheckpointMetadata.trainerFile(
                creator: "manual", trainingStep: 4, parentModelID: "", notes: "gui resume gaps test",
                schedule: TrainerScheduleState(completedTrainSteps: 4, lrWarmupSteps: 3, lrMomentumCycle: .disabled)),
            weights: trainerWeights(), architecture: arch, includesVelocity: true, lineage: record)
        return try SafetensorsModelIO.decode(data).file
    }

    private func sessionState(arenaClock: Double?) -> SessionCheckpointState {
        let state = SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "gui-gaps", savedAtUnix: 1_790_000_060, sessionStartUnix: 1_790_000_000,
            elapsedTrainingSec: 60, trainingSteps: 4, selfPlayGames: 2, selfPlayMoves: 120,
            trainingPositionsSeen: 4 * 4096, batchSize: 4096, learningRate: 5e-4,
            promoteThreshold: 0.55, arenaGames: 200,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 4, championID: "champ", trainerID: "train", arenaHistory: []
        ).withLineage(LineageRecord.sessionTestFixture)
        guard let arenaClock else { return state }
        return state.withArenaClock(secondsSinceLastArena: arenaClock)
    }

    private func session(file: ModelCheckpointFile, buffer: Bool, arenaClock: Double?) -> LoadedSession {
        let directory = URL(fileURLWithPath: "/nonexistent/gui-gaps.dcmsession")
        return LoadedSession(
            directoryURL: directory, state: sessionState(arenaClock: arenaClock),
            championFile: file, trainerFile: file,
            replayBufferURL: buffer ? directory.appendingPathComponent("replay_buffer.bin") : nil,
            chartDataURLs: nil)
    }

    private func gaps(_ resumed: LoadedSession) throws -> [String] {
        let gaps = SessionController.guiResumeGaps(
            resumed: resumed, continuedRunStreams: SessionController.resumableRunStreams(of: resumed),
            replayBufferRestored: resumed.replayBufferURL != nil,
            runningBuild: try .current, runningDevice: .current, runningFingerprint: Self.fingerprint)
        return ResumeExactness.resume(of: resumed.trainerFile.lineageParent, gaps: gaps).tokens
    }

    func testAFullSessionFromThisBuildResumesExactly() throws {
        let file = try trainerFile(withStreams: true)
        XCTAssertEqual(try gaps(session(file: file, buffer: true, arenaClock: 30)), [])
    }

    func testEachMissingPieceIsNamed() throws {
        let file = try trainerFile(withStreams: true)
        XCTAssertEqual(try gaps(session(file: file, buffer: false, arenaClock: 30)), ["buffer"])
        XCTAssertEqual(try gaps(session(file: file, buffer: true, arenaClock: nil)), ["clocks"])
        // The policy tail is the architecture's (format v12), which the
        // resumed trainer is built from, so it is never a gap; the removed
        // `policy_tail` token is unknown.
        XCTAssertNil(ResumeGap(rawValue: "policy_tail"))
        let withoutStreams = try trainerFile(withStreams: false)
        XCTAssertEqual(try gaps(session(file: withoutStreams, buffer: false, arenaClock: nil)),
                       ["rng_sampler", "buffer", "serials", "clocks"])
    }
}
