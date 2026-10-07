//
//  RelativeGradientCapIntegrationTests.swift
//  DrewsChessMachineTests
//
//  The relative gradient cap through the production paths (review MINOR 6):
//  corpus replay saves the gradient-norm history in its trainer file and an
//  exact resume carries it on; an exact resume from a history-less trainer
//  file in mode `clip` is refused naming `grad_norm_history` and proceeds
//  when that gap is accepted; a promotion's rewind (the arena's own
//  `captureArenaStartState` / `rewindToArenaStart`) takes the history back
//  with the clock and the next step-line window stays consistent; and an
//  exact resume fed the same batches (matched samplers) decides the same cap
//  and computes the same pre-clip norm as the uninterrupted run, step after
//  step.
//
//  The runner tests use `ResumeEquivalenceTests`' synthetic sealed corpus
//  and start model.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class RelativeGradientCapIntegrationTests: XCTestCase {

    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-relcap-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
    }

    override func tearDown() async throws {
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    // MARK: - Corpus replay

    /// The synthetic corpus's parameters with the relative cap in `mode`
    /// (k = 1, N = 100, W = 10).
    private func params(mode: RelativeGradientCapMode) throws -> ReplayParams {
        try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            LRWarmupSteps.id: .int(5),
            KLProbeInterval.id: .int(0),
            BatchStatsInterval.id: .int(0),
            RelativeGradClipMode.id: .int(mode.rawValue),
            RelativeGradClipMultiple.id: .double(1),
            RelativeGradClipWindowSteps.id: .int(100),
            RelativeGradClipMinHistorySteps.id: .int(10),
        ]))
    }

    private func config(corpus: URL, stepLimit: Int, startModel: URL, resumeExact: Bool,
                        accepting accepted: Set<ResumeGap>, out: URL) -> CorpusReplayConfig {
        CorpusReplayConfig(
            corpusDirectories: [corpus],
            stepLimit: stepLimit,
            epochs: nil,
            startModelPath: startModel.path,
            presetName: nil,
            startShard: nil,
            startGameIndex: nil,
            resumeExact: resumeExact,
            acceptInexact: accepted,
            outModelPath: out.path,
            overwriteOutModel: false,
            runModelID: "20261007-4-RCAP",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 0x5EF5, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
    }

    private func history(in url: URL) throws -> GradientNormHistory? {
        try CheckpointManager.decodeAnyModelFile(try Data(contentsOf: url)).metadata.trainerGradNormHistory
    }

    /// The trainer file at `source`, rewritten without its history — the
    /// form of every trainer file written before the relative cap. Only the
    /// header changes; the data region (and its `content_sha256`) is kept.
    private func writeWithoutHistory(_ source: URL, to destination: URL) throws {
        let decoded = try SafetensorsFile.decode(try Data(contentsOf: source))
        var metadata = decoded.metadata
        XCTAssertNotNil(metadata.removeValue(forKey: GradientNormHistory.metadataKey), "the source carries a history")
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        try SafetensorsFile.encode(tensors: decoded.tensors, metadata: metadata).write(to: destination)
    }

    func test_corpusReplayExactResume_carriesTheHistory() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let first = tempDir.appendingPathComponent("first.safetensors")
        let firstResult = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: start, resumeExact: false, accepting: [], out: first),
            params: try params(mode: .clip), abort: ReplayAbortFlag())
        XCTAssertEqual(firstResult.steps, 3)
        let saved = try XCTUnwrap(try history(in: first), "a corpus-replay trainer file carries the history")
        XCTAssertEqual(saved.lastTrainerStep, 3)
        XCTAssertEqual(saved.count, 3)

        let second = tempDir.appendingPathComponent("second.safetensors")
        let secondResult = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: first, resumeExact: true, accepting: [], out: second),
            params: try params(mode: .clip), abort: ReplayAbortFlag())
        XCTAssertEqual(secondResult.steps, 3)
        let continued = try XCTUnwrap(try history(in: second))
        XCTAssertEqual(continued.lastTrainerStep, 6, "the resumed run appends to the restored history")
        XCTAssertEqual(continued.count, 6)
        XCTAssertEqual(Array(continued.preClipNorms.prefix(3)).map(\.bitPattern), saved.preClipNorms.map(\.bitPattern))
        XCTAssertEqual(Array(continued.fedCaps.prefix(3)).map(\.bitPattern), saved.fedCaps.map(\.bitPattern))
    }

    func test_historyLessResumeInClipMode_isRefusedUnlessTheGapIsAccepted() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let first = tempDir.appendingPathComponent("first.safetensors")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: start, resumeExact: false, accepting: [], out: first),
            params: try params(mode: .clip), abort: ReplayAbortFlag())
        let historyLess = tempDir.appendingPathComponent("history-less.safetensors")
        try writeWithoutHistory(first, to: historyLess)
        XCTAssertNil(try history(in: historyLess))

        let refused = tempDir.appendingPathComponent("refused.safetensors")
        do {
            _ = try await CorpusReplayRunner.runReplay(
                config: config(corpus: corpus, stepLimit: 3, startModel: historyLess, resumeExact: true, accepting: [], out: refused),
                params: try params(mode: .clip), abort: ReplayAbortFlag())
            XCTFail("a history-less exact resume in clip mode must be refused")
        } catch let refusal as CLIRunRefusal {
            XCTAssertTrue(refusal.message.contains(ResumeGap.gradNormHistory.token), refusal.message)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: refused.path), "a refused resume writes no model")

        let accepted = tempDir.appendingPathComponent("accepted.safetensors")
        let result = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: historyLess, resumeExact: true,
                           accepting: [.gradNormHistory], out: accepted),
            params: try params(mode: .clip), abort: ReplayAbortFlag())
        XCTAssertEqual(result.steps, 3)
        let restarted = try XCTUnwrap(try history(in: accepted))
        XCTAssertEqual(restarted.firstTrainerStep, 4, "the history restarts at the resumed clock")
        XCTAssertEqual(restarted.lastTrainerStep, 6)
    }

    // MARK: - Promotion

    func test_promotionRewind_takesTheHistoryBackWithTheClock_andTheWindowStaysConsistent() async throws {
        let config = try RelativeGradientCapFixture.configuration(.clip, k: 1, n: 4, w: 2)
        let trainer = try RelativeGradientCapFixture.makeTrainer(hardMax: 1.0e9, configuration: config)
        let buffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        for _ in 0..<3 {
            _ = try await RelativeGradientCapFixture.step(trainer, buffer)
        }
        // Arena start: the arena's own capture.
        let arenaStart = try await trainer.captureArenaStartState()
        XCTAssertEqual(arenaStart.completedSteps, 3)
        XCTAssertEqual(arenaStart.gradNormHistory.lastTrainerStep, 3)
        var window = GradientCapStepLineWindow(startTrainerStep: 0)
        var seenGeneration = trainer.gradNormHistoryRestorePoint.generation
        for _ in 0..<3 {
            _ = try await RelativeGradientCapFixture.step(trainer, buffer)
        }
        let beforePromotion = try await trainer.exportGradNormHistory()
        _ = window.take(history: beforePromotion, throughTrainerStep: 6, fedCap: beforePromotion.lastFedCap)

        // Promotion: the arena's own rewind.
        try await trainer.rewindToArenaStart(arenaStart, promotedWeights: arenaStart.weights)
        XCTAssertEqual(trainer.completedTrainSteps, 3)
        let rewound = try await trainer.exportGradNormHistory()
        XCTAssertEqual(rewound, arenaStart.gradNormHistory)
        let restorePoint = trainer.gradNormHistoryRestorePoint
        XCTAssertNotEqual(restorePoint.generation, seenGeneration, "the rewind is a new restore generation")
        XCTAssertEqual(restorePoint.trainerStep, 3)

        // The next step appends at the rewound clock + 1, no discontinuity,
        // and decides its cap from the rewound history.
        let next = try await RelativeGradientCapFixture.step(trainer, buffer)
        XCTAssertEqual(next.gradientCap, GradientCapPolicy.decide(
            configuration: config, hardMax: 1.0e9, history: arenaStart.gradNormHistory, nextTrainerStep: 4))
        let after = try await trainer.exportGradNormHistory()
        XCTAssertEqual(after.lastTrainerStep, 4)

        // The step-line window, as the GUI `[STATS]` task drives it: a new
        // generation restarts it at the restored clock, so the line after the
        // promotion covers exactly the one step trained since.
        if restorePoint.generation != seenGeneration {
            window.rewind(toTrainerStep: restorePoint.trainerStep)
            seenGeneration = restorePoint.generation
        }
        let reading = window.take(history: after, throughTrainerStep: 4, fedCap: after.lastFedCap)
        XCTAssertEqual(reading.steps, 1)
        XCTAssertEqual(reading.maxPreClipNorm, next.gradGlobalNorm)
        XCTAssertEqual(reading.fedCap, next.gradientCap.fedCap)
    }

    // MARK: - Exact resume with matched samplers

    func test_exactResumeWithMatchedSamplers_decidesTheSameCapsAsTheUninterruptedRun() async throws {
        let config = try RelativeGradientCapFixture.configuration(.clip, k: 1, n: 4, w: 2)
        let hardMax: Float = 1.0e9
        let uninterrupted = try RelativeGradientCapFixture.makeTrainer(hardMax: hardMax, configuration: config)
        let uninterruptedBuffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        for _ in 0..<4 {
            _ = try await RelativeGradientCapFixture.step(uninterrupted, uninterruptedBuffer)
        }
        let saved = try await uninterrupted.exportResumeSnapshot()
        let samplerAtSave = uninterruptedBuffer.samplerState()

        // The CLI save → load path.
        let data = try SafetensorsModelIO.encode(
            modelID: "20261007-1-TEST", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata.trainerFile(
                creator: ModelCheckpointMetadata.corpusReplayCreator,
                trainingStep: saved.schedule.completedTrainSteps, parentModelID: "", notes: "unit test",
                schedule: saved.schedule,
                gradNormHistory: saved.gradNormHistory.history),
            weights: saved.trainerWeights, architecture: .current, includesVelocity: true,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: saved.schedule.completedTrainSteps, corpus: nil))
        let reloaded = try TrainerResumeSnapshot(checkpoint: try CheckpointManager.decodeAnyModelFile(data),
                                                 fileName: "relcap-matched.safetensors")
        let resumed = try RelativeGradientCapFixture.makeTrainer(initSeed: 2, hardMax: hardMax, configuration: config)
        try await resumed.restoreExactly(from: reloaded)
        let resumedBuffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        resumedBuffer.restoreSamplerState(samplerAtSave)

        var relativeSteps = 0
        for k in 1...4 {
            let u = try await RelativeGradientCapFixture.step(uninterrupted, uninterruptedBuffer)
            let r = try await RelativeGradientCapFixture.step(resumed, resumedBuffer)
            XCTAssertEqual(r.gradientCap, u.gradientCap, "fed cap at resumed step \(k)")
            XCTAssertEqual(r.gradGlobalNorm.bitPattern, u.gradGlobalNorm.bitPattern, "pre-clip norm at resumed step \(k)")
            if u.gradientCap.binding == .relative { relativeSteps += 1 }
        }
        XCTAssertEqual(relativeSteps, 4, "every compared step is decided by the restored relative cap")
        let uninterruptedHistory = try await uninterrupted.exportGradNormHistory()
        let resumedHistory = try await resumed.exportGradNormHistory()
        XCTAssertEqual(resumedHistory, uninterruptedHistory)
        let wu = try await uninterrupted.exportTrainerWeights()
        let wr = try await resumed.exportTrainerWeights()
        XCTAssertEqual(wr.map { $0.map(\.bitPattern) }, wu.map { $0.map(\.bitPattern) })
    }
}
