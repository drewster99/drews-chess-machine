//
//  ExactResumeTests.swift
//  DrewsChessMachineTests
//
//  Resume means training continues exactly as if it had never stopped. The
//  bugs these pin: corpus replay and train-vs-UCI restarted the trainer's
//  completed-step clock at zero on every resume (`phaseOrigin=segment-step-0`),
//  so warmup re-ran and the LR/momentum cycle phase and the decay envelope
//  restarted; they saved no optimizer velocity (and, under mixed precision,
//  the bf16 working copy instead of the fp32 masters), so momentum restarted
//  cold; and a GUI session saved after "New Session, keep trainer" recorded
//  the stats box's step count, not the trainer's clock. Also pinned: a
//  resumed session restores its own saved parameter values even outside
//  today's declared range, and the τ parameters' declared minimum is 0.01.
//
//  The resume tests train a real trainer for N steps, save it through the
//  path under test, keep training it for k more steps (the uninterrupted
//  run), and compare against a second trainer — built from deliberately
//  different hyperparameters, as a resumed process with other settings would
//  be — restored from the save and trained the same k steps. The effective
//  learning rate and momentum the optimizer is fed must agree at every one of
//  the k steps. Both trainers take real-data steps from one fixture buffer of
//  real positions, but each batch is sampled at random (the sampler is not
//  seeded), so the weights themselves are not compared after training; the
//  restored state is compared bit-exactly before it.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ExactResumeTests: XCTestCase {

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

    // MARK: - Fixtures

    private let batchSize = 32
    /// Steps trained before the save. Past warmup, so a re-run warmup would
    /// visibly change the learning rate.
    private let stepsBeforeSave = 5
    /// Steps trained after the save by both the uninterrupted and the
    /// resumed trainer. Crosses an LR-cycle period boundary.
    private let stepsAfterSave = 6

    /// A schedule whose every piece moves on a scale of a few steps: warmup,
    /// a short LR cycle, momentum following it, and a decay envelope.
    private func originalRunHyperparameters() -> TrainerHyperparameters {
        var h = TrainerHyperparameters(TrainingParameters.shared.snapshot())
        h.learningRate = 0.001
        h.momentumCoeff = 0.5
        h.lrWarmupSteps = 3
        h.sqrtBatchScalingForLR = true
        h.lrMomentumCycle = LRMomentumCycle(
            lrEnabled: true, lrPeriodSteps: 4, lrCount: 0, lrMin: 1.0e-4, lrMax: 1.0e-2, lrInvert: false,
            momentumEnabled: true, momentumPeriodSteps: 4, momentumCount: 0,
            momentumMin: 0.8, momentumMax: 0.95, momentumInvert: true,
            envelope: LRMomentumCycleEnvelope(
                lrPeakEnd: 1.0e-3, lrTroughEnd: 1.0e-5, decayHorizonSteps: 10,
                momentumFollowsLRCycle: true,
                momentumFollowStartLow: 0.85, momentumFollowStartHigh: 0.95,
                momentumFollowEndLow: 0.9, momentumFollowEndHigh: 0.97
            )
        )
        return h
    }

    /// What a resumed process might be configured with: a different warmup
    /// and a different cycle. An exact resume must ignore both in favour of
    /// the checkpoint's.
    private func differentlyConfiguredHyperparameters() -> TrainerHyperparameters {
        var h = originalRunHyperparameters()
        h.lrWarmupSteps = 50
        h.lrMomentumCycle.lrPeriodSteps = 20
        h.lrMomentumCycle.lrMax = 3.0e-2
        h.lrMomentumCycle.envelope.decayHorizonSteps = 1_000
        return h
    }

    private struct ScheduleReading: Equatable {
        let completedSteps: Int
        let learningRate: Float
        let momentum: Float
    }

    private func reading(_ trainer: ChessTrainer) -> ScheduleReading {
        let steps = trainer.completedTrainSteps
        return ScheduleReading(
            completedSteps: steps,
            learningRate: trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: steps),
            momentum: trainer.effectiveMomentum(completedSteps: steps)
        )
    }

    /// Positions held by the fixture replay buffer — several batches' worth.
    private let replayPositions = 512

    /// A replay buffer of real positions: deterministic pseudo-random legal
    /// games from the starting position (a fixed LCG picks each move; a game
    /// restarts after 80 plies or when no legal move is left), each position
    /// encoded with the played move as its policy target. Real positions are
    /// required because the real-data step builds every sample's legal-move
    /// mask from the decoded board.
    private func makeReplayBuffer(arch: NetworkArchitecture) -> ReplayBuffer {
        let encoding = arch.inputEncoding
        let floatsPerBoard = BoardEncoder.tensorLength(for: encoding)
        var boards = [Float](repeating: 0, count: replayPositions * floatsPerBoard)
        var moves = [Int32](repeating: 0, count: replayPositions)
        var plies = [UInt16](repeating: 0, count: replayPositions)
        let taus = [Float](repeating: 1.0, count: replayPositions)
        var hashes = [UInt64](repeating: 0, count: replayPositions)
        let materials = [UInt8](repeating: 32, count: replayPositions)
        var outcomes = [Float](repeating: 0, count: replayPositions)
        var lcg: UInt64 = 0x0123_4567_89AB_CDEF
        func next() -> UInt64 {
            lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            return lcg >> 33
        }
        var state = GameState.starting
        var ply = 0
        var filled = 0
        while filled < replayPositions {
            let legal = MoveGenerator.legalMoves(for: state)
            if legal.isEmpty || ply >= 80 {
                state = GameState.starting
                ply = 0
                continue
            }
            let move = legal[Int(next() % UInt64(legal.count))]
            let encoded = BoardEncoder.encode(state, encoding: encoding)
            boards.replaceSubrange(filled * floatsPerBoard ..< (filled + 1) * floatsPerBoard, with: encoded)
            moves[filled] = Int32(PolicyEncoding.policyIndex(move, currentPlayer: state.currentPlayer))
            plies[filled] = UInt16(ply)
            hashes[filled] = next()
            outcomes[filled] = Float(Int(next() % 3)) - 1.0
            state = MoveGenerator.applyMove(move, to: state)
            ply += 1
            filled += 1
        }
        let buffer = ReplayBuffer(capacity: replayPositions, inputEncoding: encoding)
        boards.withUnsafeBufferPointer { b in
        moves.withUnsafeBufferPointer { m in
        plies.withUnsafeBufferPointer { p in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        materials.withUnsafeBufferPointer { mc in
        outcomes.withUnsafeBufferPointer { o in
            guard let bb = b.baseAddress, let mb = m.baseAddress, let pb = p.baseAddress,
                  let tb = t.baseAddress, let hb = h.baseAddress, let mcb = mc.baseAddress,
                  let ob = o.baseAddress else {
                preconditionFailure("fixture arrays are non-empty, so every base address exists")
            }
            buffer.append(
                boards: bb, policyIndices: mb, plyIndices: pb, samplingTaus: tb,
                stateHashes: hb, materialCounts: mcb,
                gameLength: UInt16(replayPositions),
                workerId: 0, intraWorkerGameIndex: 0,
                outcomes: ob, count: replayPositions)
        }}}}}}}
        return buffer
    }

    /// Train `trainer` for `steps` real-data steps, returning the schedule
    /// reading before each one — the LR and momentum that step is fed.
    /// Uses `trainStep(replayBuffer:batchSize:)`, the step every production
    /// runner takes: it is the only one that advances `completedTrainSteps`
    /// (the random-data `trainStep(batchSize:)` deliberately never does, so
    /// diagnostics can't consume warmup).
    private func train(_ trainer: ChessTrainer, steps: Int, buffer: ReplayBuffer) async throws -> [ScheduleReading] {
        var readings: [ScheduleReading] = []
        for _ in 0..<steps {
            readings.append(reading(trainer))
            let timing = try await trainer.trainStep(replayBuffer: buffer, batchSize: batchSize)
            XCTAssertNotNil(timing, "the fixture buffer holds more than one batch, so every step must train")
        }
        readings.append(reading(trainer))
        return readings
    }

    private func assertBitExact(_ a: [[Float]], _ b: [[Float]], _ what: String) {
        XCTAssertEqual(a.count, b.count, "\(what): tensor count")
        for (i, (x, y)) in zip(a, b).enumerated() {
            XCTAssertEqual(x.map(\.bitPattern), y.map(\.bitPattern), "\(what): tensor \(i)")
        }
    }

    /// Run the uninterrupted trainer, save through `saveAndReload`, restore a
    /// differently configured trainer from what comes back through `resume`,
    /// and compare the two over the next `stepsAfterSave` steps.
    private func assertExactResume(
        _ path: String,
        saveAndReload: (TrainerResumeSnapshot) async throws -> TrainerResumeSnapshot,
        resume: (ChessTrainer, TrainerResumeSnapshot) async throws -> Void,
        resumedTrainerHyperparameters: (TrainerResumeSnapshot) -> TrainerHyperparameters
    ) async throws {
        let original = originalRunHyperparameters()
        let uninterrupted = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), hyperparameters: original, arch: .current)
        let buffer = makeReplayBuffer(arch: .current)
        _ = try await train(uninterrupted, steps: stepsBeforeSave, buffer: buffer)
        XCTAssertGreaterThan(uninterrupted.completedTrainSteps, original.lrWarmupSteps, "\(path): save must land past warmup")

        let saved = try await uninterrupted.exportResumeSnapshot()
        XCTAssertEqual(saved.schedule.completedTrainSteps, stepsBeforeSave)
        let reloaded = try await saveAndReload(saved)
        XCTAssertEqual(reloaded.schedule, saved.schedule, "\(path): schedule round trip")
        assertBitExact(reloaded.trainerWeights, saved.trainerWeights, "\(path): trainer state round trip")

        let resumed = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), hyperparameters: resumedTrainerHyperparameters(reloaded), arch: .current)
        try await resume(resumed, reloaded)

        // Restored state, bit-exact, before any step.
        XCTAssertEqual(resumed.completedTrainSteps, stepsBeforeSave, "\(path): clock")
        XCTAssertEqual(resumed.lrWarmupSteps, original.lrWarmupSteps, "\(path): warmup length")
        XCTAssertEqual(resumed.lrMomentumCycle, original.lrMomentumCycle, "\(path): cycle")
        assertBitExact(try await resumed.exportTrainerWeights(), saved.trainerWeights, "\(path): masters + velocity")

        // Warmup does not re-run: the first resumed step is fed the full,
        // un-ramped LR the uninterrupted run is fed — and that differs from
        // what a trainer whose clock restarted would be fed.
        let restartedClock = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), hyperparameters: original, arch: .current)
        XCTAssertNotEqual(
            restartedClock.effectiveLearningRate(forBatchSize: batchSize),
            uninterrupted.effectiveLearningRate(forBatchSize: batchSize),
            "\(path): the fixture must be able to tell a restarted clock apart"
        )

        let uninterruptedReadings = try await train(uninterrupted, steps: stepsAfterSave, buffer: buffer)
        let resumedReadings = try await train(resumed, steps: stepsAfterSave, buffer: buffer)
        XCTAssertEqual(resumedReadings.count, stepsAfterSave + 1)
        for (k, (u, r)) in zip(uninterruptedReadings, resumedReadings).enumerated() {
            XCTAssertEqual(r, u, "\(path): schedule at N+\(k)")
        }
    }

    // MARK: - Corpus replay and train-vs-UCI (`--resume-exact`)

    /// The CLI runners' save → `--start-model … --resume-exact` path: the
    /// checkpoint is encoded exactly as their `saveTrainerModel` encodes it,
    /// decoded as `CheckpointManager.loadModelFile` decodes it, turned back
    /// into a snapshot by `TrainerResumeSnapshot(checkpoint:)`, the trainer
    /// is built from `--parameters` with the checkpoint's schedule adopted,
    /// and `restoreExactly` restores it.
    private func assertCLIExactResume(creator: String, corpus: LineageRecord.CorpusPosition?) async throws {
        try await assertExactResume(
            creator,
            saveAndReload: { snapshot in
                let data = try SafetensorsModelIO.encode(
                    modelID: "20260929-1-TEST",
                    createdAtUnix: 1_780_000_000,
                    metadata: ModelCheckpointMetadata(
                        creator: creator,
                        trainingStep: 17,
                        parentModelID: "",
                        notes: "unit test",
                        trainerSchedule: snapshot.schedule
                    ),
                    weights: snapshot.trainerWeights,
                    architecture: .current,
                    includesVelocity: true,
                    lineage: try LineageRecord.forTests(
                        trainerCompletedSteps: snapshot.schedule.completedTrainSteps, corpus: corpus)
                )
                let file = try CheckpointManager.decodeAnyModelFile(data)
                // `training_step` stays the segment-local value it was given.
                XCTAssertEqual(file.metadata.trainingStep, 17)
                return try TrainerResumeSnapshot(checkpoint: file, fileName: "unit-test.safetensors")
            },
            resume: { trainer, snapshot in
                try await trainer.restoreExactly(from: snapshot)
            },
            resumedTrainerHyperparameters: { snapshot in
                differentlyConfiguredHyperparameters().adoptingSchedule(snapshot.schedule)
            }
        )
    }

    func test_corpusReplay_resumeExact_continuesTheScheduleAsIfNeverStopped() async throws {
        try await assertCLIExactResume(creator: "replay", corpus: LineageRecord.CorpusPosition(
            corpusID: "unit-test-corpus", corpusPath: "/unit-test-corpus", epoch: 0,
            nextGameIndex: 12, shard: 0, populatedPlies: 0, bufferCapacity: 0))
    }

    func test_trainVsUci_resumeExact_continuesTheScheduleAsIfNeverStopped() async throws {
        try await assertCLIExactResume(creator: "train-vs-uci", corpus: nil)
    }

    // MARK: - GUI Play-and-Train session resume

    private func minimalSessionState(trainingSteps: Int) throws -> SessionCheckpointState {
        let json = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "20260929-1-TEST", "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400, "elapsedTrainingSec": 3600,
          "trainingSteps": \(trainingSteps), "selfPlayGames": 678, "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280, "batchSize": 32, "learningRate": 0.001,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.02},
          "selfPlayWorkerCount": 4,
          "championID": "20260929-1-TEST", "trainerID": "20260929-2-TEST", "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(json.utf8))
    }

    /// The GUI's save (`CheckpointManager.saveSession` with the trainer
    /// file's schedule metadata) → `loadSession` → the session-resume clock
    /// resolution → `restoreExactly`. The session's own `trainingSteps` is
    /// deliberately wrong (0, as after "New Session, keep trainer"): the
    /// trainer file's clock must win.
    func test_guiSessionResume_continuesTheScheduleAsIfNeverStopped() async throws {
        var sessionDirectories: [URL] = []
        defer {
            for directory in sessionDirectories {
                do { try FileManager.default.removeItem(at: directory) }
                catch { XCTFail("could not remove unit-test session \(directory.path): \(error)") }
            }
        }
        try await assertExactResume(
            "gui",
            saveAndReload: { snapshot in
                let championWeights = Array(snapshot.trainerWeights.prefix(NetworkArchitecture.current.weightTensorPlan().count))
                let directory = try await CheckpointManager.saveSession(
                    championWeights: championWeights,
                    championID: "20260929-1-TEST",
                    championMetadata: ModelCheckpointMetadata(creator: "manual", trainingStep: 0, parentModelID: "", notes: "champion"),
                    championCreatedAtUnix: 1_780_000_000,
                    trainerWeights: snapshot.trainerWeights,
                    trainerID: "20260929-2-TEST",
                    trainerMetadata: ModelCheckpointMetadata(
                        creator: "manual", trainingStep: 0, parentModelID: "20260929-1-TEST", notes: "trainer",
                        trainerSchedule: snapshot.schedule
                    ),
                    trainerCreatedAtUnix: 1_780_000_001,
                    state: try minimalSessionState(trainingSteps: 0), lineage: try LineageRecord.forTests(trainerCompletedSteps: snapshot.schedule.completedTrainSteps, corpus: nil),
                    trigger: "unittest"
                )
                sessionDirectories.append(directory)
                let loaded = try CheckpointManager.loadSession(at: directory)
                let resolved = TrainerScheduleState.forSessionResume(
                    trainerFileSchedule: loaded.trainerFile.metadata.trainerSchedule,
                    sessionTrainingSteps: loaded.state.trainingSteps,
                    sessionWarmupSteps: snapshot.schedule.lrWarmupSteps,
                    sessionCycle: snapshot.schedule.lrMomentumCycle
                )
                XCTAssertEqual(resolved.schedule.completedTrainSteps, snapshot.schedule.completedTrainSteps)
                XCTAssertNotEqual(resolved.schedule.completedTrainSteps, loaded.state.trainingSteps)
                return TrainerResumeSnapshot(
                    trainerWeights: loaded.trainerFile.weights, schedule: resolved.schedule, dropoutRNG: .notInCheckpoint
                )
            },
            resume: { trainer, snapshot in
                // The GUI puts the session's warmup and cycle on the trainer
                // in its `[RESUME-PARAM]` block, resets the network (which
                // zeroes the clock), then restores.
                trainer.lrWarmupSteps = snapshot.schedule.lrWarmupSteps
                trainer.lrMomentumCycle = snapshot.schedule.lrMomentumCycle
                try await trainer.resetNetwork()
                XCTAssertEqual(trainer.completedTrainSteps, 0)
                try await trainer.restoreExactly(from: snapshot)
            },
            resumedTrainerHyperparameters: { _ in differentlyConfiguredHyperparameters() }
        )
    }

    // MARK: - Checkpoint format

    func test_trainerScheduleState_metadataRoundTripsBitExactly() throws {
        let schedule = TrainerScheduleState(
            completedTrainSteps: 1_234_567,
            lrWarmupSteps: 1000,
            lrMomentumCycle: originalRunHyperparameters().lrMomentumCycle
        )
        let decoded = try TrainerScheduleState.decode(fromMetadata: try schedule.metadataEntries())
        XCTAssertEqual(decoded, schedule)
        XCTAssertNil(try TrainerScheduleState.decode(fromMetadata: ["model_id": "x"]))

        var partial = try schedule.metadataEntries()
        partial[TrainerScheduleState.MetadataKey.lrMomentumCycleEnvelope] = nil
        XCTAssertThrowsError(try TrainerScheduleState.decode(fromMetadata: partial))
    }

    func test_safetensors_refusesTrainerScheduleWithoutVelocity() throws {
        let plan = NetworkArchitecture.current.weightTensorPlan()
        let baseWeights = plan.map { [Float](repeating: 0.25, count: $0.elementCount) }
        let schedule = TrainerScheduleState(completedTrainSteps: 9, lrWarmupSteps: 3, lrMomentumCycle: .disabled)
        XCTAssertThrowsError(try SafetensorsModelIO.encode(
            modelID: "20260929-1-TEST",
            createdAtUnix: 1_780_000_000,
            metadata: ModelCheckpointMetadata(creator: "replay", trainingStep: 9, parentModelID: "", notes: "", trainerSchedule: schedule),
            weights: baseWeights,
            architecture: .current,
            includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: schedule.completedTrainSteps, corpus: nil)
        ))
    }

    /// A checkpoint written by a build before exact resume (base tensors
    /// only, no `trainer_*` keys) — like the corpus-replay checkpoints running
    /// on 2026-09-29 — is refused by `--resume-exact`, naming what it lacks,
    /// and still loads as a model (its network weights are the whole file).
    func test_preExactResumeCheckpoint_isRefusedNamingWhatItLacks() throws {
        let plan = NetworkArchitecture.current.weightTensorPlan()
        let baseWeights = plan.map { [Float](repeating: 0.25, count: $0.elementCount) }
        let data = try SafetensorsModelIO.encode(
            modelID: "20260929-1-TEST",
            createdAtUnix: 1_780_000_000,
            metadata: ModelCheckpointMetadata(creator: "replay", trainingStep: 9, parentModelID: "", notes: ""),
            weights: baseWeights,
            architecture: .current,
            includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: 9, corpus: LineageRecord.CorpusPosition(
                corpusID: "unit-test-corpus", corpusPath: "/unit-test-corpus", epoch: 0,
                nextGameIndex: 0, shard: 0, populatedPlies: 0, bufferCapacity: 0))
        )
        let file = try CheckpointManager.decodeAnyModelFile(data)
        XCTAssertFalse(file.includesOptimizerVelocity)
        XCTAssertNil(file.metadata.trainerSchedule)
        XCTAssertEqual(file.networkWeights.count, plan.count)
        XCTAssertThrowsError(try TrainerResumeSnapshot(checkpoint: file, fileName: "old.safetensors")) { error in
            let message = error.localizedDescription
            XCTAssertTrue(message.contains("optimizer velocity"), message)
            XCTAssertTrue(message.contains("trainer_completed_steps"), message)
        }
    }

    /// A trainer-state file still loads into an inference network: every
    /// model consumer takes `networkWeights`, never the velocity tail.
    func test_trainerStateFile_networkWeightsExcludeVelocity() throws {
        let plan = NetworkArchitecture.current.weightTensorPlan()
        let trainables = plan.filter { $0.kind != .bnRunningStat }
        let baseWeights = plan.map { [Float](repeating: 0.25, count: $0.elementCount) }
        let velocity = trainables.map { [Float](repeating: -1, count: $0.elementCount) }
        let schedule = TrainerScheduleState(completedTrainSteps: 9, lrWarmupSteps: 3, lrMomentumCycle: .disabled)
        let data = try SafetensorsModelIO.encode(
            modelID: "20260929-1-TEST",
            createdAtUnix: 1_780_000_000,
            metadata: ModelCheckpointMetadata(creator: "replay", trainingStep: 9, parentModelID: "", notes: "", trainerSchedule: schedule),
            weights: baseWeights + velocity,
            architecture: .current,
            includesVelocity: true,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: schedule.completedTrainSteps, corpus: nil)
        )
        let file = try CheckpointManager.decodeAnyModelFile(data)
        XCTAssertTrue(file.includesOptimizerVelocity)
        XCTAssertEqual(file.metadata.trainerSchedule, schedule)
        assertBitExact(file.networkWeights, baseWeights, "network weights")
    }

    // MARK: - Resumed parameter values

    /// A resumed session's own value is restored even outside today's
    /// declared range, held in memory only (never persisted to app
    /// settings), and the ordinary setter keeps rejecting out-of-range
    /// writes afterwards.
    func test_restoreFromSession_restoresAnOutOfRangeValueWithoutPersistingIt() {
        let p = TrainingParameters.shared
        p.arenaTargetTau = 0.2
        TrainingParameters.suppressPersistence = false
        defer { TrainingParameters.suppressPersistence = true }
        let persistedBefore = TrainingParameters.persistedValue(ArenaTargetTau.self)

        p.restoreFromSession(ArenaTargetTau.self, 0.005, into: \.arenaTargetTau)
        XCTAssertEqual(p.arenaTargetTau, 0.005)
        XCTAssertEqual(TrainingParameters.persistedValue(ArenaTargetTau.self), persistedBefore)

        TrainingParameters.suppressPersistence = true
        p.arenaTargetTau = 0.3
        XCTAssertEqual(p.arenaTargetTau, 0.3)
        p.arenaTargetTau = 0.004
        XCTAssertEqual(p.arenaTargetTau, 0.3, "the ordinary setter still rejects out-of-range values")

        p.restoreFromSession(LRWarmupSteps.self, 250, into: \.lrWarmupSteps)
        XCTAssertEqual(p.lrWarmupSteps, 250, "an in-range value restores normally")
    }

    /// The owner's case: a session saved with an arena τ floor of 0.02 (which
    /// the drifted popover accepted) resumes on 0.02 — legal now that the
    /// declared minimum is 0.01 — rather than on the current setting.
    func test_restoreFromSession_restoresTheSavedTauFloor() throws {
        let p = TrainingParameters.shared
        p.arenaTargetTau = 0.2
        let state = try minimalSessionState(trainingSteps: 0)
        guard let saved = Double(state.arenaTau.floorTau.description) else {
            XCTFail("saved τ did not parse")
            return
        }
        XCTAssertEqual(saved, 0.02)
        p.restoreFromSession(ArenaTargetTau.self, saved, into: \.arenaTargetTau)
        XCTAssertEqual(p.arenaTargetTau, 0.02)
    }

    // MARK: - τ floors

    func test_everyTemperatureParameter_hasA0p01Minimum() {
        let keys: [any TrainingParameterKey.Type] = [
            SelfPlayStartTau.self, SelfPlayTargetTau.self, ArenaStartTau.self, ArenaTargetTau.self,
        ]
        for key in keys {
            XCTAssertEqual(key.definition.doubleRange?.min, 0.01, key.id)
        }
        XCTAssertEqual(SelfPlayStartTau.declaredClosedRange.lowerBound, 0.01)
        XCTAssertTrue(ArenaTargetTau.isWithinDeclaration(0.01))
        XCTAssertTrue(ArenaTargetTau.isWithinDeclaration(0.02))
        XCTAssertFalse(ArenaTargetTau.isWithinDeclaration(0.0099))
        XCTAssertEqual(PlayController.humanPlayTauMin, 0.01)

        let p = TrainingParameters.shared
        p.selfPlayTargetTau = 0.5
        p.selfPlayTargetTau = 0.01
        XCTAssertEqual(p.selfPlayTargetTau, 0.01)
        p.arenaStartTau = 0.6
        p.arenaStartTau = 0.01
        XCTAssertEqual(p.arenaStartTau, 0.01)
    }
}
