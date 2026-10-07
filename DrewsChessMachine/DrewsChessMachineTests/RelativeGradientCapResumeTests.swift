//
//  RelativeGradientCapResumeTests.swift
//  DrewsChessMachineTests
//
//  The gradient-norm history is trainer state (plan X4): it is saved in the
//  trainer-state file, restored by `restoreExactly` (so an exact resume needs
//  no second warm-up and decides the uninterrupted run's next cap), reported
//  as a `grad_norm_history` gap only when a history-less checkpoint is
//  resumed in mode `clip`, rewound with the clock by a promotion, and a clock
//  moved without the history stops the next step before it trains.
//

import XCTest
@testable import DrewsChessMachine

final class RelativeGradientCapResumeTests: XCTestCase {

    private let stepsBeforeSave = 4
    private let stepsAfterSave = 4
    private let hardMax: Float = 1.0e9

    /// Save `snapshot` as the CLI runners do and read it back as their
    /// `--resume-exact` does.
    private func saveAndReload(_ snapshot: TrainerResumeSnapshot, withHistory: Bool) throws -> TrainerResumeSnapshot {
        let metadata: ModelCheckpointMetadata
        if withHistory {
            metadata = ModelCheckpointMetadata.trainerFile(
                creator: ModelCheckpointMetadata.corpusReplayCreator,
                trainingStep: snapshot.schedule.completedTrainSteps, parentModelID: "", notes: "unit test",
                schedule: snapshot.schedule,
                gradNormHistory: snapshot.gradNormHistory.history)
        } else {
            // The form of every trainer file written before the relative cap.
            metadata = ModelCheckpointMetadata.trainerFile(
                creator: ModelCheckpointMetadata.corpusReplayCreator,
                trainingStep: snapshot.schedule.completedTrainSteps, parentModelID: "", notes: "unit test",
                schedule: snapshot.schedule)
        }
        let data = try SafetensorsModelIO.encode(
            modelID: "20261007-1-TEST", createdAtUnix: 1_790_000_000, metadata: metadata,
            weights: snapshot.trainerWeights, architecture: .current, includesVelocity: true,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: snapshot.schedule.completedTrainSteps, corpus: nil))
        let file = try CheckpointManager.decodeAnyModelFile(data)
        return try TrainerResumeSnapshot(checkpoint: file, fileName: "relcap-unit-test.safetensors")
    }

    func test_exactResume_restoresTheHistory_andDecidesTheUninterruptedCap() async throws {
        let config = try RelativeGradientCapFixture.configuration(.clip, k: 1, n: 4, w: 2)
        let uninterrupted = try RelativeGradientCapFixture.makeTrainer(hardMax: hardMax, configuration: config)
        let buffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        for _ in 0..<stepsBeforeSave {
            _ = try await RelativeGradientCapFixture.step(uninterrupted, buffer)
        }
        let saved = try await uninterrupted.exportResumeSnapshot()
        let savedHistory = try XCTUnwrap(saved.gradNormHistory.history)
        XCTAssertEqual(savedHistory.count, stepsBeforeSave)
        XCTAssertEqual(savedHistory.lastTrainerStep, stepsBeforeSave)

        let reloaded = try saveAndReload(saved, withHistory: true)
        XCTAssertEqual(reloaded.gradNormHistory, .restored(savedHistory), "bit-exact history round trip")
        XCTAssertEqual(ResumeGap.gradNormHistoryGaps(restoring: reloaded.gradNormHistory, runningMode: .clip), [])

        let resumed = try RelativeGradientCapFixture.makeTrainer(initSeed: 2, hardMax: hardMax, configuration: config)
        try await resumed.restoreExactly(from: reloaded)
        let restoredHistory = try await resumed.exportGradNormHistory()
        XCTAssertEqual(restoredHistory, savedHistory)

        // The first resumed step's cap is a function of the restored history
        // alone, so it equals the uninterrupted run's — no second warm-up.
        let u = try await RelativeGradientCapFixture.step(uninterrupted, buffer)
        let r = try await RelativeGradientCapFixture.step(resumed, RelativeGradientCapFixture.makeReplayBuffer(arch: .current))
        XCTAssertEqual(r.gradientCap, u.gradientCap)
        XCTAssertEqual(r.gradientCap.binding, .relative)

        // Every later step's cap is the rule applied to that trainer's own
        // history, and history and clock stay together.
        for _ in 1..<stepsAfterSave {
            let before = try await resumed.exportGradNormHistory()
            let expected = GradientCapPolicy.decide(configuration: config, hardMax: hardMax, history: before,
                                                    nextTrainerStep: resumed.completedTrainSteps + 1)
            let timing = try await RelativeGradientCapFixture.step(resumed, buffer)
            XCTAssertEqual(timing.gradientCap, expected)
        }
        let finalHistory = try await resumed.exportGradNormHistory()
        XCTAssertEqual(finalHistory.lastTrainerStep, stepsBeforeSave + stepsAfterSave)
        XCTAssertEqual(finalHistory.count, stepsBeforeSave + stepsAfterSave)
        XCTAssertEqual(Array(finalHistory.preClipNorms.prefix(stepsBeforeSave)), savedHistory.preClipNorms)
        XCTAssertEqual(resumed.completedTrainSteps, stepsBeforeSave + stepsAfterSave)
    }

    func test_historyLessCheckpoint_isAGapOnlyInClipMode_andWarmsUpAgain() async throws {
        let config = try RelativeGradientCapFixture.configuration(.clip, k: 1, n: 4, w: 2)
        let source = try RelativeGradientCapFixture.makeTrainer(hardMax: hardMax, configuration: config)
        let buffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        for _ in 0..<stepsBeforeSave {
            _ = try await RelativeGradientCapFixture.step(source, buffer)
        }
        let reloaded = try saveAndReload(try await source.exportResumeSnapshot(), withHistory: false)
        XCTAssertEqual(reloaded.gradNormHistory, .notInCheckpoint)
        XCTAssertEqual(ResumeGap.gradNormHistoryGaps(restoring: reloaded.gradNormHistory, runningMode: .clip), [.gradNormHistory])
        XCTAssertEqual(ResumeGap.gradNormHistoryGaps(restoring: reloaded.gradNormHistory, runningMode: .logOnly), [])
        XCTAssertEqual(ResumeGap.gradNormHistoryGaps(restoring: reloaded.gradNormHistory, runningMode: .off), [])
        XCTAssertEqual(ResumeGap(rawValue: "grad_norm_history"), .gradNormHistory)

        let resumed = try RelativeGradientCapFixture.makeTrainer(initSeed: 2, hardMax: hardMax, configuration: config)
        try await resumed.restoreExactly(from: reloaded)
        let restoredHistory = try await resumed.exportGradNormHistory()
        XCTAssertTrue(restoredHistory.isEmpty)
        // W = 2: the first two resumed steps feed the hard max, the third the
        // relative cap.
        let first = try await RelativeGradientCapFixture.step(resumed, buffer)
        let second = try await RelativeGradientCapFixture.step(resumed, buffer)
        let third = try await RelativeGradientCapFixture.step(resumed, buffer)
        XCTAssertEqual(first.gradientCap.fedCap, hardMax)
        XCTAssertEqual(first.gradientCap.binding, .hard)
        XCTAssertEqual(second.gradientCap.fedCap, hardMax)
        XCTAssertEqual(third.gradientCap.binding, .relative)
        let history = try await resumed.exportGradNormHistory()
        XCTAssertEqual(history.firstTrainerStep, stepsBeforeSave + 1, "the new history starts at the resumed clock + 1")
    }

    func test_writerRefusesAHistoryThatDoesNotEndAtTheClock() async throws {
        let trainer = try RelativeGradientCapFixture.makeTrainer(
            hardMax: hardMax, configuration: try RelativeGradientCapFixture.configuration(.clip))
        let weights = try await trainer.exportTrainerWeights()
        var history = GradientNormHistory()
        try history.append(trainerStep: 3, preClipNorm: 1, fedCap: 15)
        let schedule = TrainerScheduleState(completedTrainSteps: 4, lrWarmupSteps: 3, lrMomentumCycle: .disabled)
        let metadata = ModelCheckpointMetadata.trainerFile(
            creator: ModelCheckpointMetadata.corpusReplayCreator, trainingStep: 4, parentModelID: "", notes: "unit test",
            schedule: schedule, gradNormHistory: history)
        XCTAssertThrowsError(try SafetensorsModelIO.encode(
            modelID: "20261007-1-TEST", createdAtUnix: 1_790_000_000, metadata: metadata,
            weights: weights, architecture: .current, includesVelocity: true,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: 4, corpus: nil))) { error in
            XCTAssertEqual(error as? GradientNormHistoryError, .clockMismatch(historyLastStep: 3, trainerClock: 4))
        }
    }

    func test_promotionRewind_restoresTheHistoryWithTheClock() async throws {
        let config = try RelativeGradientCapFixture.configuration(.clip, k: 1, n: 4, w: 2)
        let trainer = try RelativeGradientCapFixture.makeTrainer(hardMax: hardMax, configuration: config)
        let buffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        for _ in 0..<3 {
            _ = try await RelativeGradientCapFixture.step(trainer, buffer)
        }
        // Arena start: weights, velocity, clock and history captured together
        // (`SessionController.runArenaParallel`).
        let weights = try await trainer.network.exportWeights()
        let velocity = try await trainer.exportVelocitySnapshot()
        let clock = trainer.completedTrainSteps
        let history = try await trainer.exportGradNormHistory()
        for _ in 0..<3 {
            _ = try await RelativeGradientCapFixture.step(trainer, buffer)
        }
        // Promotion: the rewind the arena performs.
        try await trainer.network.loadWeights(weights)
        try await trainer.syncMastersFromWorking()
        try await trainer.loadVelocitySnapshot(velocity)
        trainer.completedTrainSteps = clock
        try await trainer.restoreGradNormHistory(history)
        let rewoundHistory = try await trainer.exportGradNormHistory()
        XCTAssertEqual(rewoundHistory, history)
        let next = try await RelativeGradientCapFixture.step(trainer, buffer)
        XCTAssertEqual(next.gradientCap, GradientCapPolicy.decide(
            configuration: config, hardMax: hardMax, history: history, nextTrainerStep: clock + 1))
        let after = try await trainer.exportGradNormHistory()
        XCTAssertEqual(after.lastTrainerStep, clock + 1)
    }

    func test_clockMovedWithoutTheHistory_stopsTheNextStepBeforeItTrains() async throws {
        let config = try RelativeGradientCapFixture.configuration(.logOnly)
        let trainer = try RelativeGradientCapFixture.makeTrainer(hardMax: hardMax, configuration: config)
        let buffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        for _ in 0..<3 {
            _ = try await RelativeGradientCapFixture.step(trainer, buffer)
        }
        trainer.completedTrainSteps = 1
        let before = try await trainer.exportTrainerWeights()
        do {
            _ = try await trainer.trainStep(replayBuffer: buffer, batchSize: RelativeGradientCapFixture.batchSize)
            XCTFail("a step after the clock moved without the history must throw")
        } catch {
            XCTAssertEqual(error as? GradientNormHistoryError, .discontinuity(expected: 4, got: 2))
        }
        let after = try await trainer.exportTrainerWeights()
        XCTAssertEqual(after.map { $0.map(\.bitPattern) }, before.map { $0.map(\.bitPattern) },
                       "the refused step changed no weight")
        XCTAssertEqual(trainer.completedTrainSteps, 1)
        // Restoring a history that does not end at the clock is refused too.
        do {
            try await trainer.restoreGradNormHistory(try await trainer.exportGradNormHistory())
            XCTFail("a history ending at 3 cannot be restored onto clock 1")
        } catch {
            XCTAssertEqual(error as? GradientNormHistoryError, .clockMismatch(historyLastStep: 3, trainerClock: 1))
        }
    }
}
