//
//  RunStartReadingsTests.swift
//  DrewsChessMachineTests
//
//  While a Play-and-Train run is active, everything that states its batch
//  size, pre-train fill or buffer capacity — the status displays, the
//  learning-rate readouts, the Run All Analyses export, the Lichess probe
//  history, session.json — takes the run's start-time capture, never the
//  settings, which the popover may have changed for the next start. Positions
//  trained are counted at the batch each step actually trained at, so a
//  Continue at another batch size never restates the steps before it. A start
//  that fails puts back what it replaced: the capture, the positions count and
//  the replay buffer, together.
//
//  Every test that assigns `TrainingParameters.shared` snapshots it in setUp,
//  suppresses persistence, and restores it in tearDown.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class RunStartReadingsTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    /// A running harness session captured at batch 64 / pre-train fill 300,
    /// whose settings the popover has since changed to batch 128 / fill 900.
    private func harnessWithEditedSettings() throws -> GuiSaveHarness {
        TrainingParameters.shared.trainingBatchSize = 64
        TrainingParameters.shared.replayBufferMinPositionsBeforeTraining = 300
        let harness = try GuiSaveHarness()
        TrainingParameters.shared.trainingBatchSize = 128
        TrainingParameters.shared.replayBufferMinPositionsBeforeTraining = 900
        return harness
    }

    private struct NotAnInteger: Error { let id: String }

    /// The integer parameter `id` in a saved record's parameter snapshot.
    private func recordedValue(_ record: LineageRecord, _ id: String) throws -> Int {
        let parameters = try XCTUnwrap(record.parameters)
        let values = try JSONDecoder().decode([String: ParameterValue].self, from: Data(parameters.snapshotJSON.utf8))
        guard case .int(let value) = try XCTUnwrap(values[id], "\(id) in \(parameters.snapshotJSON)") else {
            throw NotAnInteger(id: id)
        }
        return value
    }

    /// Train `steps` more steps on the harness's trainer clock and stats box,
    /// as the real training worker counts them.
    private func trainSteps(_ steps: Int, on harness: GuiSaveHarness) {
        let target = harness.trainer.completedTrainSteps + steps
        harness.trainer.completedTrainSteps = target
        harness.trainingStatsBox.setStepCount(target)
        harness.publishCountersLikeTheHeartbeat()
    }

    /// 10 steps at the captured batch 64, then a Continue at batch 128 (the
    /// edited setting) and 5 more steps.
    private func continueAtADifferentBatch(_ harness: GuiSaveHarness) {
        let controller = harness.controller
        controller.anchorRunTrainedPositions(atTrainerStep: harness.trainer.completedTrainSteps)
        trainSteps(10, on: harness)
        controller.realTraining = false
        controller.beginRunStartCapture(buffer: harness.buffer)
        controller.realTraining = true
        controller.anchorRunTrainedPositions(atTrainerStep: harness.trainer.completedTrainSteps)
        trainSteps(5, on: harness)
    }

    // MARK: - Displays and exports (regression)

    func testTheDisplaysShowTheRunsCaptureDuringARunAndTheSettingOtherwise() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        let controller = harness.controller
        XCTAssertTrue(controller.realTraining)
        XCTAssertEqual(controller.trainingBatchSizeDisplay(), .activeRun(64), "the batch the run steps at")
        XCTAssertEqual(controller.replayBufferMinPositionsDisplay(), .activeRun(300), "the fill the run waits for")

        controller.realTraining = false
        XCTAssertEqual(controller.trainingBatchSizeDisplay(), .setting(128), "no run: the setting, labelled as one")
        XCTAssertEqual(controller.replayBufferMinPositionsDisplay(), .setting(900))
    }

    func testTheAnalysisExportStatesTheRunsBatchSize() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        let metadata = harness.controller.currentAnalysisExportMetadata()
        XCTAssertEqual(metadata.training?.batchSize, 64, "the batch size the run trains at, not the edited setting")
    }

    func testSessionStateRecordsTheRunsPreTrainFill() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        let state = try harness.controller.buildCurrentSessionState(
            championID: "c", trainerID: "t", arenaClock: .live, includeReplayBuffer: false)
        XCTAssertEqual(state.replayBufferMinPositionsBeforeTraining, 300, "the pre-train fill the run waited for")
    }

    // MARK: - Positions trained (regression)

    func testSessionStatePositionsDoNotRestateEarlierStepsAtTheCurrentBatch() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        continueAtADifferentBatch(harness)
        let state = try harness.controller.buildCurrentSessionState(
            championID: "c", trainerID: "t", arenaClock: .live, includeReplayBuffer: false)
        XCTAssertEqual(state.trainingSteps, 15)
        XCTAssertEqual(state.trainingPositionsSeen, 10 * 64 + 5 * 128,
                       "each step counted at the batch it trained at")
    }

    func testTheStatusBarPositionsDoNotRestateEarlierStepsAtTheCurrentBatch() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        continueAtADifferentBatch(harness)
        XCTAssertEqual(harness.controller.trainedPositionsForDisplay(atSessionSteps: 15), 10 * 64 + 5 * 128)
    }

    func testTheProbePositionsDoNotRestateEarlierStepsAtTheCurrentBatch() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        continueAtADifferentBatch(harness)
        XCTAssertEqual(harness.controller.trainedPositions(atTrainerStep: 15), 10 * 64 + 5 * 128)
    }

    // MARK: - A failed start (regression)

    func testAFailedStartPutsBackTheCaptureAndTheBufferTogether() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        let controller = harness.controller
        let captureA = try XCTUnwrap(controller.runStartCapture)
        TrainingParameters.shared.replayBufferCapacity = harness.buffer.capacity * 2

        // "New Session, keep trainer": a new buffer at the current capacity,
        // captured, then installed — and the start's setup fails.
        let newBuffer = ReplayBuffer(capacity: TrainingParameters.shared.replayBufferCapacity,
                                     inputEncoding: GuiSaveHarness.architecture.inputEncoding,
                                     sampler: DCMRandom(seed: 9))
        controller.beginRunStartCapture(buffer: newBuffer)
        controller.replayBuffer = newBuffer
        controller.restoreRunStartCaptureAfterFailedStart()

        XCTAssertEqual(controller.runStartCapture, captureA, "the capture the segment left in place was trained under")
        XCTAssertTrue(controller.replayBuffer === harness.buffer, "the buffer that capture describes")
        let record = try controller.lineageRecordForSave(
            at: Date(), cut: try controller.takeConfigurationCut(trainer: harness.trainer),
            trainerCompletedSteps: harness.trainer.completedTrainSteps,
            dropoutPhiloxState: nil, dropoutStreamState: nil)
        let state = try controller.buildCurrentSessionState(
            championID: "c", trainerID: "t", arenaClock: .live, includeReplayBuffer: true)
        XCTAssertEqual(state.replayBufferCapacity, try recordedValue(record, ReplayBufferCapacity.id),
                       "session.json's buffer and the record's capacity describe the same buffer")
        XCTAssertEqual(state.batchSize, try recordedValue(record, TrainingBatchSize.id))
    }

    // MARK: - Heartbeat (regression)

    func testTheTrainingChartIsSampledEvenWithoutACapture() async throws {
        let controller = SessionController()
        let coordinator = ChartCoordinator()
        coordinator.collectionEnabled = true
        controller.chartCoordinator = coordinator
        controller.realTraining = true
        controller.parallelStats = ParallelWorkerStatsBox(sessionStart: Date()).snapshot()
        XCTAssertNil(controller.runStartCapture)

        await controller.refreshProgressRateIfNeeded()

        XCTAssertEqual(coordinator.trainingChartNextId, 1, "the training chart (and its divergence alarm) still runs")
        XCTAssertNotNil(controller.trainingError, "a run with no capture is a bug, surfaced rather than skipped")
    }

    // MARK: - Labels with no run active

    func testTheAnalysisExportLabelsTheSettingWhenNoRunIsActive() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        harness.controller.realTraining = false
        let training = try XCTUnwrap(harness.controller.currentAnalysisExportMetadata().training)
        XCTAssertNil(training.batchSize, "no run is active: no run's batch size")
        XCTAssertEqual(training.batchSizeSetting, 128, "the setting, under the setting's own key")
        XCTAssertEqual(AnalysisExportMetadata.currentSchemaVersion, 4)
    }

    func testTheDisplayLabelSaysWhenAValueIsTheSetting() {
        XCTAssertEqual(SessionController.CapturedParameterDisplay.activeRun(64).label("Batch size"), "Batch size")
        XCTAssertEqual(SessionController.CapturedParameterDisplay.setting(128).label("Batch size"), "Batch size (setting)")
        XCTAssertEqual(SessionController.CapturedParameterDisplay.setting(128).value, 128)
    }

    // MARK: - Continue and resume

    func testAContinueRecapturesTheSettingsAndKeepsTheReusedBuffersCapacity() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        let controller = harness.controller
        let first = try XCTUnwrap(controller.runStartCapture)
        TrainingParameters.shared.replayBufferCapacity = harness.buffer.capacity * 2

        controller.realTraining = false
        let second = controller.beginRunStartCapture(buffer: harness.buffer)
        XCTAssertEqual(second, RunStartParameterCapture(
            trainingBatchSize: 128, replayBufferMinPositionsBeforeTraining: 900,
            replayBufferCapacity: harness.buffer.capacity),
                       "batch and fill from the settings now; capacity from the reused buffer, not the setting")
        XCTAssertEqual(controller.runStartCapture, second)
        XCTAssertEqual(controller.runStartStateReplacedByLatestStart?.capture, first,
                       "the replaced capture is kept for a failed Continue")
        controller.restoreRunStartCaptureAfterFailedStart()
        XCTAssertEqual(controller.runStartCapture, first)
        XCTAssertTrue(controller.replayBuffer === harness.buffer, "a Continue reuses, and so puts back, the same buffer")
    }

    func testAResumeCarriesThePositionsItsSessionRecorded() throws {
        let controller = SessionController()
        let box = TrainingLiveStatsBox(rollingWindow: SessionController.rollingLossWindow)
        var seeded = TrainingRunStats()
        seeded.steps = 4
        box.seed(seeded)
        controller.trainingBox = box
        TrainingParameters.shared.trainingBatchSize = 64
        let buffer = ReplayBuffer(capacity: 64, inputEncoding: .basic30, sampler: DCMRandom(seed: 3))

        controller.pendingLoadedSession = try Self.loadedSession(trainingSteps: 4, trainingPositionsSeen: 4 * 32)
        controller.beginRunStartCapture(buffer: buffer)
        let recorded = try XCTUnwrap(controller.runTrainedPositions).recorded
        XCTAssertEqual(recorded, TrainedPositionsCount(stepsAtStart: 4, positionsBeforeStart: 4 * 32, batchSize: 64),
                       "the steps before the resume at the batch they trained at, as the session recorded them")
        XCTAssertEqual(try recorded.positions(atSteps: 6), 4 * 32 + 2 * 64)

        controller.pendingLoadedSession = try Self.loadedSession(trainingSteps: 4, trainingPositionsSeen: nil)
        controller.runTrainedPositions = nil
        controller.beginRunStartCapture(buffer: buffer)
        XCTAssertNil(try XCTUnwrap(controller.runTrainedPositions).recorded.positions(atSteps: 6),
                     "a session that recorded none leaves them unrecorded, never modeled")
    }

    func testStepsNoRecordDescribesAreUnrecorded() throws {
        let controller = SessionController()
        let box = TrainingLiveStatsBox(rollingWindow: SessionController.rollingLossWindow)
        var seeded = TrainingRunStats()
        seeded.steps = 9
        box.seed(seeded)
        controller.trainingBox = box
        controller.beginRunStartCapture(buffer: ReplayBuffer(capacity: 64, inputEncoding: .basic30, sampler: DCMRandom(seed: 3)))
        XCTAssertNil(try XCTUnwrap(controller.runTrainedPositions).recorded.positionsBeforeStart)
        XCTAssertEqual(try XCTUnwrap(controller.runTrainedPositions).observedPositions(atSteps: 9), 0,
                       "the rate counter only needs differences, so it counts on from here")
    }

    func testTheProbePositionsAreUnrecordedWhenTheSessionDoesNotCoverTheTrainersClock() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        // "New Session, keep trainer": the session counts from 0 on a trainer at clock 500.
        harness.controller.anchorRunTrainedPositions(atTrainerStep: 500)
        XCTAssertNil(harness.controller.trainedPositions(atTrainerStep: 510))
    }

    func testTheProbePositionsAreUnrecordedOnceTheTrainerMovedOutsideTheRun() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        let controller = harness.controller
        controller.anchorRunTrainedPositions(atTrainerStep: 0)
        trainSteps(10, on: harness)
        XCTAssertEqual(controller.trainedPositions(atTrainerStep: 10), 10 * 64)
        controller.realTraining = false
        XCTAssertEqual(controller.trainedPositions(atTrainerStep: 10), 10 * 64, "stopped, nothing else moved the trainer")
        harness.trainer.completedTrainSteps = 11
        XCTAssertNil(controller.trainedPositions(atTrainerStep: 11), "a step outside the run (demo training) is not the run's")
    }

    // MARK: - Arena

    func testAnArenaWithoutACaptureStopsBeforePausingAnything() async throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        let controller = harness.controller
        controller.runStartCapture = nil
        let arenaFlag = ArenaActiveFlag()
        let tBox = TournamentLiveBox()

        await controller.runArenaParallel(
            trainer: harness.trainer, champion: harness.champion,
            candidateInference: harness.champion, arenaChampion: harness.champion,
            tBox: tBox, selfPlayGate: harness.selfPlayGate, trainingGate: harness.trainingGate,
            arenaFlag: arenaFlag, overrideBox: ArenaOverrideBox())

        XCTAssertFalse(arenaFlag.isActive, "the arena released its claim")
        XCTAssertFalse(controller.isArenaRunning)
        XCTAssertFalse(harness.selfPlayGate.isRequestedToPause, "self-play was never paused")
        XCTAssertFalse(harness.trainingGate.isRequestedToPause, "training was never paused")
        XCTAssertTrue(controller.tournamentHistory.isEmpty, "no arena was recorded")
        let statsSnapshot = await harness.trainingStatsBox.snapshot()
        let error = try XCTUnwrap(statsSnapshot.error)
        XCTAssertTrue(error.contains("Arena aborted"), error)
    }

    // MARK: - Settings popover

    func testTheDeferredEditLineNamesTheValueTheActiveRunKeeps() {
        let model = TrainingSettingsPopoverModel(selfPlayDelayMaxMs: 1000, stepDelayMaxMs: 1000, maxSelfPlayWorkers: 8)
        let capture = RunStartParameterCapture(
            trainingBatchSize: 4096, replayBufferMinPositionsBeforeTraining: 300, replayBufferCapacity: 1000)
        model.runStartCaptureProvider = { capture }
        XCTAssertEqual(
            model.capturedKeyEditLogLine(id: TrainingBatchSize.id, old: 4096, new: 1024, inForce: \.trainingBatchSize),
            "[PARAM] training_batch_size: 4096 -> 1024 (applies at the next Play-and-Train start; this run keeps 4096)")
        model.runStartCaptureProvider = { nil }
        XCTAssertEqual(
            model.capturedKeyEditLogLine(id: TrainingBatchSize.id, old: 4096, new: 1024, inForce: \.trainingBatchSize),
            "[PARAM] training_batch_size: 4096 -> 1024")
    }

    func testThePopoverHearsOfACaptureOnlyWhileARunIsActive() throws {
        let harness = try harnessWithEditedSettings()
        defer { XCTAssertNoThrow(try harness.removeSessionsDirectory()) }
        XCTAssertNotNil(harness.controller.activeRunStartCapture)
        harness.controller.realTraining = false
        XCTAssertNil(harness.controller.activeRunStartCapture,
                     "after Stop an edit applies at Continue too, so no run keeps the old value")
        XCTAssertNotNil(harness.controller.runStartCapture, "the capture itself is kept for the stopped run's saves")
    }

    // MARK: - Helpers

    /// A loaded session, as Load Session leaves it pending, whose session.json
    /// recorded `trainingSteps` and `trainingPositionsSeen`.
    private static func loadedSession(trainingSteps: Int, trainingPositionsSeen: Int?) throws -> LoadedSession {
        let arch = ResumeEquivalenceTests.architecture
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .gui,
                                         argv: ["DrewsChessMachine"], startedAt: start, segmentStartTrainerStep: 0)
        try tracker.noteSegmentStartForTests(trainerStep: 0)
        let record = try tracker.record(
            at: start.addingTimeInterval(60), trainerCompletedSteps: trainingSteps, segmentLocalStep: trainingSteps,
            segmentGames: 2, segmentPositions: 120, corpus: nil,
            parameters: try .forTests(
                adopting: TrainerScheduleState(completedTrainSteps: trainingSteps, lrWarmupSteps: 3, lrMomentumCycle: .disabled),
                overriding: ["learning_rate": .double(0.0005)]),
            rng: LineageRecord.RNG(dropoutPhiloxState: try DropoutPhiloxState(words: [1, 2, 3, 4, 5, 6, 7]),
                                   streams: nil,
                                   behaviorFingerprint: BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab")),
            inputs: tracker.testInputs)
        let weights = arch.weightTensorPlan().enumerated().map { i, spec in
            (0..<spec.elementCount).map { Float((i * 5 + $0) % 11) * 0.01 }
        } + arch.trainableTensorPlan().map { [Float](repeating: -0.25, count: $0.elementCount) }
        let data = try SafetensorsModelIO.encode(
            modelID: "20261006-2-RSPS", createdAtUnix: 1_790_000_060,
            metadata: ModelCheckpointMetadata.trainerFile(
                creator: "manual", trainingStep: trainingSteps, parentModelID: "", notes: "run start readings test",
                schedule: TrainerScheduleState(completedTrainSteps: trainingSteps, lrWarmupSteps: 3, lrMomentumCycle: .disabled),
                policyTailPrecision: .default),
            weights: weights, architecture: arch, includesVelocity: true, lineage: record)
        let file = try SafetensorsModelIO.decode(data).file
        let state = SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "run-start-readings", savedAtUnix: 1_790_000_060, sessionStartUnix: 1_790_000_000,
            elapsedTrainingSec: 60, trainingSteps: trainingSteps, selfPlayGames: 2, selfPlayMoves: 120,
            trainingPositionsSeen: trainingPositionsSeen, batchSize: 32, learningRate: 5e-4,
            promoteThreshold: 0.55, arenaGames: 200,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 4, championID: "champ", trainerID: "train", arenaHistory: []
        ).withLineage(record).withArenaClock(secondsSinceLastArena: 30)
        return LoadedSession(
            directoryURL: URL(fileURLWithPath: "/nonexistent/run-start-readings.dcmsession"), state: state,
            championFile: file, trainerFile: file, replayBufferURL: nil, chartDataURLs: nil)
    }

    func testTheOptimizerReadoutIsPublishedOnlyWhileARunIsActive() throws {
        let controller = SessionController()
        TrainingParameters.shared.trainingBatchSize = 64
        controller.beginRunStartCapture(buffer: ReplayBuffer(
            capacity: 64, inputEncoding: .basic30, sampler: DCMRandom(seed: 3)))
        controller.realTraining = false
        XCTAssertNil(controller.optimizerReadoutBatchSize(),
                     "a stopped run's capture describes no trainer stepping now (a sweep or demo training may be)")
        controller.realTraining = true
        XCTAssertEqual(controller.optimizerReadoutBatchSize(), 64)
    }
}
