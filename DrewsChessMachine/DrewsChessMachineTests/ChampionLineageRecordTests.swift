//
//  ChampionLineageRecordTests.swift
//  DrewsChessMachineTests
//
//  A session's champion file holds the champion's weights, which differ from
//  the trainer's whenever training has moved on since the last promotion (or
//  since the champion was built or loaded). Its lineage record must describe
//  those weights — where they came from — not the trainer's run at the save:
//  a built champion carries a mint record, a loaded or promoted one the
//  record of the weights it holds (without any trainer state, which a model
//  file does not carry), and a session save writes that record into the
//  champion file while the trainer file and session.json keep the run's.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ChampionLineageRecordTests: XCTestCase {

    private let saveDate = Date(timeIntervalSince1970: 1_790_000_500)

    /// A record at cum step 1000 with every piece of trainer state set.
    private func trainedRecordWithTrainerState() throws -> LineageRecord {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(
            start: .fresh(initialization: .forTests), pathKind: .gui, argv: ["DrewsChessMachine"],
            startedAt: start, segmentStartTrainerStep: 0)
        let seed = RunRandomSeed.resolve(mode: .seeded, configuredSeed: 11, commandLineSeed: nil,
                                         drawSeed: { 0 })
        let streams = seed.runStreams(samplerState: DCMRandom(seed: 5), dropoutStreamState: DCMRandom(seed: 6),
                                      nextGameSerial: 40, arenasStarted: 2, opponentGameIndices: nil)
        return try tracker.record(
            at: start.addingTimeInterval(300), trainerCompletedSteps: 1000, segmentLocalStep: 1000,
            segmentGames: 40, segmentPositions: 3000, corpus: nil, parameters: nil,
            rng: LineageRecord.RNG(dropoutPhiloxState: try DropoutPhiloxState(words: [1, 2, 3, 4, 5, 6, 7]),
                                   streams: streams,
                                   behaviorFingerprint: BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe,
                                                                                   sha256: "ab")))
    }

    private func assertNoTrainerState(_ record: LineageRecord, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertNil(record.rng.dropoutPhiloxState, file: file, line: line)
        XCTAssertNil(record.rng.streams, file: file, line: line)
        XCTAssertNil(record.rng.behaviorFingerprint, file: file, line: line)
    }

    func testPromotedChampionKeepsItsPromotionRecord() throws {
        let promotion = try trainedRecordWithTrainerState()
        let origin = SessionController.ChampionOrigin.file(LineageTracker.ParentFile(
            modelID: "20261003-1-PROM", contentSHA256: nil, trainerCompletedSteps: 1000,
            lineage: .recorded(promotion.withoutTrainerState()), derivationHistory: []))
        let record = try SessionController.championFileLineageRecord(origin: origin, at: saveDate)
        XCTAssertEqual(record.steps.cumTrainerStep, 1000)
        XCTAssertEqual(record.run.lineageRunID, promotion.run.lineageRunID)
        XCTAssertEqual(record.fed.cumGames, promotion.fed.cumGames)
        XCTAssertEqual(record.rng.initialization, promotion.rng.initialization)
        assertNoTrainerState(record)
    }

    func testALoadedFilesRecordIsCarriedWithoutItsTrainerState() throws {
        let fileRecord = try trainedRecordWithTrainerState()
        let origin = SessionController.ChampionOrigin.file(LineageTracker.ParentFile(
            modelID: "20261003-1-LOAD", contentSHA256: "00", trainerCompletedSteps: 1000,
            lineage: .recorded(fileRecord), derivationHistory: []))
        let record = try SessionController.championFileLineageRecord(origin: origin, at: saveDate)
        XCTAssertEqual(record, fileRecord.withoutTrainerState())
        assertNoTrainerState(record)
    }

    func testBuiltChampionGetsAMintRecord() throws {
        let record = try SessionController.championFileLineageRecord(origin: .built(initialization: .forTests), at: saveDate)
        XCTAssertEqual(record.run.start, .fresh)
        XCTAssertEqual(record.steps.cumTrainerStep, 0)
        XCTAssertNil(record.parent)
        XCTAssertEqual(record.rng.initialization, .forTests)
        XCTAssertEqual(try SessionController.championFileTrainingStep(origin: .built(initialization: .forTests)), 0)
    }

    func testAPreLineageSourceGetsAnUntrainedCopyRecord() throws {
        let origin = SessionController.ChampionOrigin.file(LineageTracker.ParentFile(
            modelID: "20260801-1-PREL", contentSHA256: nil, trainerCompletedSteps: 2500,
            lineage: .unrecorded(formatVersion: 6), derivationHistory: []))
        let record = try SessionController.championFileLineageRecord(origin: origin, at: saveDate)
        XCTAssertEqual(record.run.start, .derive)
        XCTAssertEqual(record.parent?.modelID, "20260801-1-PREL")
        XCTAssertNil(record.steps.cumTrainerStep)
        XCTAssertEqual(try SessionController.championFileTrainingStep(origin: origin), 2500)
    }

    func testAChampionWithNoOriginIsRefused() {
        XCTAssertThrowsError(try SessionController.championFileLineageRecord(origin: nil, at: saveDate))
        XCTAssertThrowsError(try SessionController.championFileTrainingStep(origin: nil))
    }

    func testAPromotionRecordsTheRecordWithoutTrainerStateAsTheChampionsOrigin() throws {
        let controller = SessionController()
        let promotion = try trainedRecordWithTrainerState()
        controller.recordPromotedChampionOrigin(
            championID: ModelID(value: "20261003-1-PROM"), trainerCompletedSteps: 1000, record: .success(promotion))
        guard case .file(let parent) = controller.championOrigin else {
            return XCTFail("a promotion's origin is the promoted weights' record")
        }
        XCTAssertEqual(parent.modelID, "20261003-1-PROM")
        XCTAssertEqual(parent.trainerCompletedSteps, 1000)
        XCTAssertEqual(parent.lineage.record, promotion.withoutTrainerState())
    }

    func testAPromotionWhoseRecordFailedClearsTheChampionsOrigin() throws {
        struct RecordFailure: Error {}
        let controller = SessionController()
        controller.championOrigin = .built(initialization: .forTests)
        controller.recordPromotedChampionOrigin(
            championID: ModelID(value: "20261003-1-PROM"), trainerCompletedSteps: 1000, record: .failure(RecordFailure()))
        XCTAssertNil(controller.championOrigin)
        controller.championOrigin = .built(initialization: .forTests)
        controller.recordPromotedChampionOrigin(
            championID: nil, trainerCompletedSteps: 1000, record: .success(try trainedRecordWithTrainerState()))
        XCTAssertNil(controller.championOrigin)
    }

    /// A session save writes the champion's record into the champion file
    /// and the run's record into the trainer file and session.json.
    func testASessionSaveWritesEachFileItsOwnRecord() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ChampionLineageRecordTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer {
            do { try FileManager.default.removeItem(at: root) }
            catch { XCTFail("could not remove \(root.path): \(error)") }
        }
        let arch = ResumeEquivalenceTests.architecture
        let base = arch.weightTensorPlan().enumerated().map { index, spec in
            (0..<spec.elementCount).map { Float((index + $0) % 9) * 0.03125 }
        }
        let velocity = arch.trainableTensorPlan().map { [Float](repeating: 0, count: $0.elementCount) }
        let schedule = TrainerScheduleState(completedTrainSteps: 1000, lrWarmupSteps: 10, lrMomentumCycle: .disabled)
        let runRecord = try trainedRecordWithTrainerState()
        let championRecord = try SessionController.championFileLineageRecord(
            origin: .built(initialization: .forTests), at: saveDate)
        let state = try SessionCheckpointState.decode(Data("""
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "20261003-1-SESS", "savedAtUnix": 1790000500,
          "sessionStartUnix": 1790000000, "elapsedTrainingSec": 500,
          "trainingSteps": 1000, "selfPlayGames": 40, "selfPlayMoves": 3000,
          "trainingPositionsSeen": 32000, "batchSize": 32, "learningRate": 0.001,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.02},
          "selfPlayWorkerCount": 4,
          "championID": "20261003-1-CHMP", "trainerID": "20261003-2-TRNR", "arenaHistory": []
        }
        """.utf8))
        let url = try await CheckpointManager.saveSession(
            championWeights: base,
            championID: "20261003-1-CHMP",
            championMetadata: ModelCheckpointMetadata(creator: "manual", trainingStep: 0, parentModelID: "", notes: "champion"),
            championCreatedAtUnix: 1_790_000_500,
            trainerWeights: base + velocity,
            trainerID: "20261003-2-TRNR",
            trainerMetadata: ModelCheckpointMetadata.trainerFile(
                creator: "manual", trainingStep: 1000, parentModelID: "20261003-1-CHMP", notes: "trainer",
                schedule: schedule, policyTailPrecision: .default),
            trainerCreatedAtUnix: 1_790_000_500,
            state: state,
            lineage: runRecord,
            championLineage: championRecord,
            architecture: arch,
            trigger: "unittest",
            at: saveDate,
            sessionsDirectory: root)
        let loaded = try CheckpointManager.loadSession(at: url)
        XCTAssertEqual(loaded.championFile.safetensorsProvenance?.lineage.record, championRecord)
        XCTAssertEqual(loaded.trainerFile.safetensorsProvenance?.lineage.record, runRecord)
        XCTAssertEqual(loaded.state.lineage, runRecord)
    }
}
