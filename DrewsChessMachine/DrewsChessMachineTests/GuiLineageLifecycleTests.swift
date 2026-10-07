//
//  GuiLineageLifecycleTests.swift
//  DrewsChessMachineTests
//
//  The GUI lineage segment lives exactly as long as the trainer whose
//  training it records:
//  - rebuilding the champion for another architecture drops the trainer,
//    and its lineage segment must end with it — both rebuild paths, Build
//    Network and the auto-build before a load, the same way — so no later
//    save attributes another trainer's weights to the old segment;
//  - a Play-and-Train start whose lineage segment cannot begin leaves
//    things as a next start can use them: a loaded session stays pending
//    (it is redone in full next time), a continuing segment stays as it
//    was, and a segment whose trainer was just reset is not left behind
//    describing weights it no longer matches.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class GuiLineageLifecycleTests: XCTestCase {

    private static let architecture = ResumeEquivalenceTests.architecture

    private func resumedTracker(segmentStart: Int) throws -> LineageTracker {
        let parent = LineageTracker.ParentFile(
            modelID: "20261003-2-OLDT", contentSHA256: nil, trainerCompletedSteps: segmentStart,
            lineage: .recorded(try LineageRecord.forTests(trainerCompletedSteps: segmentStart, corpus: nil)),
            derivationHistory: [])
        return try LineageTracker(start: .resume(parent: parent, gaps: [], legacyTotals: nil), pathKind: .gui,
                                  argv: ["DrewsChessMachine"], startedAt: Date(), segmentStartTrainerStep: segmentStart)
    }

    private func trainer() throws -> ChessTrainer {
        try ChessTrainer(dropoutStream: DCMRandom(seed: 31),
                         hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
                         arch: Self.architecture, initialization: .seeded(initSeed: 31))
    }

    private struct FingerprintFailure: Error {}

    private static let fingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab")

    /// A loaded session as Load Session leaves it pending: a trainer file
    /// whose record is a GUI save's, and a session.json with a lineage.
    private func loadedSession() throws -> LoadedSession {
        let arch = Self.architecture
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .gui,
                                         argv: ["DrewsChessMachine"], startedAt: start, segmentStartTrainerStep: 0)
        try tracker.noteSegmentStartForTests(trainerStep: 0)
        let record = try tracker.record(
            at: start.addingTimeInterval(60), trainerCompletedSteps: 4, segmentLocalStep: 4,
            segmentGames: 2, segmentPositions: 120, corpus: nil,
            parameters: try .forTests(
                adopting: TrainerScheduleState(completedTrainSteps: 4, lrWarmupSteps: 3, lrMomentumCycle: .disabled),
                overriding: ["learning_rate": .double(0.0005)]),
            rng: LineageRecord.RNG(dropoutPhiloxState: try DropoutPhiloxState(words: [1, 2, 3, 4, 5, 6, 7]),
                                   streams: nil, behaviorFingerprint: Self.fingerprint),
            inputs: tracker.testInputs)
        let weights = arch.weightTensorPlan().enumerated().map { i, spec in
            (0..<spec.elementCount).map { Float((i * 5 + $0) % 11) * 0.01 }
        } + arch.trainableTensorPlan().map { [Float](repeating: -0.25, count: $0.elementCount) }
        let data = try SafetensorsModelIO.encode(
            modelID: "20261003-2-LDED", createdAtUnix: 1_790_000_060,
            metadata: ModelCheckpointMetadata.trainerFile(
                creator: "manual", trainingStep: 4, parentModelID: "", notes: "gui lineage lifecycle test",
                schedule: TrainerScheduleState(completedTrainSteps: 4, lrWarmupSteps: 3, lrMomentumCycle: .disabled)),
            weights: weights, architecture: arch, includesVelocity: true, lineage: record)
        let file = try SafetensorsModelIO.decode(data).file
        let state = SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "gui-lineage-lifecycle", savedAtUnix: 1_790_000_060, sessionStartUnix: 1_790_000_000,
            elapsedTrainingSec: 60, trainingSteps: 4, selfPlayGames: 2, selfPlayMoves: 120,
            trainingPositionsSeen: 4 * 4096, batchSize: 4096, learningRate: 5e-4,
            promoteThreshold: 0.55, arenaGames: 200,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 4, championID: "champ", trainerID: "train", arenaHistory: []
        ).withLineage(record).withArenaClock(secondsSinceLastArena: 30)
        return LoadedSession(
            directoryURL: URL(fileURLWithPath: "/nonexistent/gui-lineage-lifecycle.dcmsession"), state: state,
            championFile: file, trainerFile: file, replayBufferURL: nil, chartDataURLs: nil)
    }

    private func runSeed() -> RunRandomSeed {
        RunRandomSeed.resolve(mode: .seeded, configuredSeed: 41, commandLineSeed: nil, drawSeed: { 0 })
    }

    func testALineageStartFailureKeepsThePendingSessionForTheNextStart() throws {
        let controller = SessionController()
        let trainer = try trainer()
        controller.trainer = trainer
        controller.pendingLoadedSession = try loadedSession()
        controller.pendingLoadedSessionAcceptedReplacements = ["x"]

        let result = controller.beginRunLineage(
            mode: .freshOrFromLoadedSession, trainer: trainer, championIdentifier: ModelID(value: "20261003-1-CHMP"),
            continuedRunStreams: nil, replayBufferRestored: false, fingerprintResult: .failure(FingerprintFailure()))

        guard case .failure = result else { return XCTFail("a failed fingerprint fails the lineage start") }
        XCTAssertNotNil(controller.pendingLoadedSession, "the loaded session stays pending for the next start")
        XCTAssertEqual(controller.pendingLoadedSessionAcceptedReplacements, ["x"])
        XCTAssertNil(controller.lineageTracker)
    }

    func testAFailedStartAfterATrainerResetLeavesNoSegmentBehind() throws {
        let controller = SessionController()
        let trainer = try trainer()
        controller.trainer = trainer
        controller.lineageTracker = try resumedTracker(segmentStart: 0)
        controller.beginRunStartCapture(buffer: ReplayBuffer(capacity: 64, inputEncoding: Self.architecture.inputEncoding, sampler: DCMRandom(seed: 3)))
        controller.lineageFedCarry = SessionController.LineageFedCarry(games: 7, positions: 300,
                                                                       baselineGames: nil, baselinePositions: nil)
        controller.championOrigin = .built(initialization: .forTests)
        // No run seed: the segment begins, then its [RUN] record fails.
        controller.runRandomSeed = nil

        let result = controller.beginRunLineage(
            mode: .newSessionResetTrainerFromChampion, trainer: trainer,
            championIdentifier: ModelID(value: "20261003-1-CHMP"),
            continuedRunStreams: nil, replayBufferRestored: false, fingerprintResult: .success(Self.fingerprint))

        guard case .failure = result else { return XCTFail("a start with no run seed fails") }
        XCTAssertNil(controller.lineageTracker, "neither the old segment nor a half-begun one is left")
        XCTAssertEqual(controller.lineageFedCarry, SessionController.LineageFedCarry())
    }

    func testAFailedContinueLeavesTheRunningSegmentUnchanged() throws {
        let controller = SessionController()
        let trainer = try trainer()
        controller.trainer = trainer
        let running = try resumedTracker(segmentStart: 0)
        let carry = SessionController.LineageFedCarry(games: 7, positions: 300, baselineGames: nil, baselinePositions: nil)
        controller.lineageTracker = running
        controller.beginRunStartCapture(buffer: ReplayBuffer(capacity: 64, inputEncoding: Self.architecture.inputEncoding, sampler: DCMRandom(seed: 3)))
        controller.lineageFedCarry = carry
        controller.runRandomSeed = nil

        let result = controller.beginRunLineage(
            mode: .continueAfterStop, trainer: trainer, championIdentifier: ModelID(value: "20261003-1-CHMP"),
            continuedRunStreams: nil, replayBufferRestored: false, fingerprintResult: .success(Self.fingerprint))

        guard case .failure = result else { return XCTFail("a start with no run seed fails") }
        XCTAssertTrue(controller.lineageTracker === running)
        XCTAssertEqual(controller.lineageFedCarry, carry)
    }

    func testASuccessfulStartConsumesThePendingSession() throws {
        let controller = SessionController()
        let trainer = try trainer()
        controller.trainer = trainer
        let session = try loadedSession()
        controller.pendingLoadedSession = session
        controller.pendingLoadedSessionAcceptedReplacements = ["x"]
        controller.runRandomSeed = runSeed()
        // What Load Session and `startRealTraining` set before the lineage
        // begins: the loaded champion's origin, how the start resolved its
        // seed, and its replay-ratio start.
        controller.adoptLoadedChampionOrigin(session.championFile)
        controller.runSeedStartKind = .resolvedOrInherited
        controller.replayRatioStart = ReplayRatioInitialDelay.resolve(
            autoAdjust: false, savedAutoDelayMs: nil, trainingStepDelayMs: 0)
        controller.beginRunStartCapture(buffer: ReplayBuffer(capacity: 64, inputEncoding: Self.architecture.inputEncoding, sampler: DCMRandom(seed: 3)))

        let result = controller.beginRunLineage(
            mode: .freshOrFromLoadedSession, trainer: trainer, championIdentifier: ModelID(value: "20261003-1-CHMP"),
            continuedRunStreams: nil, replayBufferRestored: false, fingerprintResult: .success(Self.fingerprint))

        guard case .success(let tracker) = result else { return XCTFail("the start succeeds: \(result)") }
        XCTAssertTrue(controller.lineageTracker === tracker)
        XCTAssertNil(controller.pendingLoadedSession)
        XCTAssertTrue(controller.pendingLoadedSessionAcceptedReplacements.isEmpty)
        XCTAssertEqual(trainer.identifier?.description, "20261003-2-LDED")
    }

    func testRebuildingTheChampionForAnotherArchitectureEndsTheTrainersLineageSegment() async throws {
        let controller = SessionController()
        controller.trainer = try trainer()
        controller.lineageTracker = try resumedTracker(segmentStart: 50)
        controller.beginRunStartCapture(buffer: ReplayBuffer(capacity: 64, inputEncoding: Self.architecture.inputEncoding, sampler: DCMRandom(seed: 3)))
        controller.lineageFedCarry = SessionController.LineageFedCarry(games: 7, positions: 300,
                                                                       baselineGames: 2, baselinePositions: 80)

        let rebuilt = await controller.ensureChampionBuilt(arch: Self.architecture)
        guard case .success = rebuilt else { return XCTFail("the rebuild failed: \(rebuilt)") }

        XCTAssertNil(controller.trainer, "the rebuild drops the trainer")
        XCTAssertNil(controller.lineageTracker, "the dropped trainer's lineage segment ends with it")
        XCTAssertEqual(controller.lineageFedCarry, SessionController.LineageFedCarry())
    }
}
