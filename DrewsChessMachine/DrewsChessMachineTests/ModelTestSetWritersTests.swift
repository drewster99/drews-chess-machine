import os
import XCTest
@testable import DrewsChessMachine

/// Every model-file writer records the test-set results of the weights it
/// writes (test-set results plan D2/D3, validation 3–6): both session files,
/// File ▸ Save Champion's `saveModel`, derive, graft and corpus replay; a
/// failed evaluation still writes the file; and an evaluation is a pure
/// read of the weights.
final class ModelTestSetWritersTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-test-set-writers-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
    }

    /// A small evaluator (a slice of each bundled set) so the GPU work stays
    /// short; the sets keep their identity.
    private static let smallEvaluator = ModelTestSetEvaluator(testSets: [
        subset(LichessProbeData.set200, count: 24),
        subset(LichessProbeData.wide, count: 40),
    ])

    private static func subset(_ set: ProbeTestSet, count: Int) -> ProbeTestSet {
        ProbeTestSet(id: set.id, title: set.title, description: set.description,
                     fingerprintSHA256: set.fingerprintSHA256, probes: Array(set.probes.prefix(count)))
    }

    private static func reading(_ url: URL) throws -> ModelTestSetResultsInFile {
        let (_, metadata) = try SafetensorsFile.decode(try Data(contentsOf: url))
        return ModelTestSetResultsField.reading(fromMetadata: metadata)
    }

    private static func reading(_ data: Data) throws -> ModelTestSetResultsInFile {
        let (_, metadata) = try SafetensorsFile.decode(data)
        return ModelTestSetResultsField.reading(fromMetadata: metadata)
    }

    private static func evaluatedSets(_ reading: ModelTestSetResultsInFile, file: StaticString = #filePath, line: UInt = #line) -> [ModelTestSetResults.SetResult]? {
        guard case .recorded(.evaluated(let results)) = reading else {
            XCTFail("expected evaluated results, got \(reading)", file: file, line: line)
            return nil
        }
        return results.sets
    }

    private static let evaluatedFixture = ModelTestSetResultsField.evaluated(ModelTestSetResults(
        evaluatedAtUnix: 1_791_414_697, build: 1, policyTailPrecision: .mixedFinalProjection,
        sets: [ModelTestSetResults.SetResult(
            id: "fixture", title: "Fixture", description: "d", fingerprintSHA256: String(repeating: "b", count: 64),
            positions: 10, top1Correct: 4, top5Correct: 8, avgCorrectProbability: 0.25, avgCorrectRank: 2.5,
            nll: 1.5, pElo: .estimate(1200), themes: [])]))

    private func minimalState(sessionID: String, championID: String, trainerID: String) throws -> SessionCheckpointState {
        let json = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "\(sessionID)", "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400, "elapsedTrainingSec": 3600,
          "trainingSteps": 12345, "selfPlayGames": 678, "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280, "batchSize": 1024, "learningRate": 5.0e-5,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "championID": "\(championID)", "trainerID": "\(trainerID)", "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(json.utf8))
    }

    private func saveSession(champion: [[Float]], trainer: [[Float]], evaluator: any ModelTestSetEvaluating) async throws -> URL {
        let meta = ModelCheckpointMetadata(creator: "manual", trainingStep: 1, parentModelID: "", notes: "test-set writers")
        return try await CheckpointManager.saveSession(
            championWeights: champion, championID: "20261007-1-CHMP",
            championMetadata: meta, championCreatedAtUnix: 1_790_000_000,
            trainerWeights: trainer, trainerID: "20261007-2-TRNR",
            trainerMetadata: meta, trainerCreatedAtUnix: 1_790_000_001,
            state: try minimalState(sessionID: "20261007-1-CHMP", championID: "20261007-1-CHMP", trainerID: "20261007-2-TRNR"),
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
            championLineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
            testSetEvaluator: evaluator, trigger: "unittest", sessionsDirectory: tempDir)
    }

    private static func velocity(for base: [[Float]]) -> [[Float]] {
        NetworkArchitecture.current.weightTensorPlan().filter { $0.kind != .bnRunningStat }.enumerated().map { j, spec in
            (0..<spec.elementCount).map { Float(j * 3 + $0 % 11) * 0.001 }
        }
    }

    // MARK: - Session save

    /// Both files carry results, each of its own weights: they equal a fresh
    /// evaluation of the weights read back from that file, and the champion's
    /// and trainer's differ.
    func testASessionSaveEvaluatesBothFilesOnTheirOwnWeights() async throws {
        let champion = try await ChessMPSNetwork(.randomWeights(initSeed: 2)).exportWeights()
        let trainerBase = try await ChessMPSNetwork(.randomWeights(initSeed: 9)).exportWeights()
        let dir = try await saveSession(champion: champion, trainer: trainerBase + Self.velocity(for: trainerBase),
                                        evaluator: Self.smallEvaluator)

        let loaded = try CheckpointManager.loadSession(at: dir)
        let championSets = Self.evaluatedSets(try Self.reading(SessionCheckpointLayout.championURL(in: dir)))
        let trainerSets = Self.evaluatedSets(try Self.reading(SessionCheckpointLayout.trainerURL(in: dir)))
        guard let championSets, let trainerSets else { return }

        let championAgain = await Self.smallEvaluator.evaluate(weights: loaded.championFile.weights, architecture: .current)
        let trainerAgain = await Self.smallEvaluator.evaluate(weights: loaded.trainerFile.weights, architecture: .current)
        XCTAssertEqual(championSets, Self.evaluatedSets(.recorded(championAgain)))
        XCTAssertEqual(trainerSets, Self.evaluatedSets(.recorded(trainerAgain)))
        XCTAssertNotEqual(championSets, trainerSets, "two different networks")
        XCTAssertEqual(championSets.map(\.id), ["lichess-200", "lichess-wide"])
    }

    /// The evaluator is asked for exactly the champion's weights and then
    /// the trainer's, velocity included (it reads only the base tensors),
    /// and each file records what it returned.
    func testASessionSaveAsksForEachFilesWeights() async throws {
        let champion = try await ChessMPSNetwork(.randomWeights(initSeed: 4)).exportWeights()
        let trainer = champion + Self.velocity(for: champion)
        let evaluator = FixtureTestSetEvaluator(Self.evaluatedFixture)
        let dir = try await saveSession(champion: champion, trainer: trainer, evaluator: evaluator)

        XCTAssertEqual(evaluator.evaluatedWeights.count, 2)
        XCTAssertEqual(evaluator.evaluatedWeights.first, champion)
        XCTAssertEqual(evaluator.evaluatedWeights.last, trainer)
        XCTAssertEqual(try Self.reading(SessionCheckpointLayout.championURL(in: dir)), .recorded(Self.evaluatedFixture))
        XCTAssertEqual(try Self.reading(SessionCheckpointLayout.trainerURL(in: dir)), .recorded(Self.evaluatedFixture))
    }

    /// A failed evaluation never fails the save: the files are written and
    /// say why there are no results.
    func testAFailedEvaluationStillSavesAndRecordsTheFailure() async throws {
        let weights = try await ChessMPSNetwork(.randomWeights(initSeed: 6)).exportWeights()
        let failed = ModelTestSetResultsField.failed(reason: "the forward pass failed for 3 of 200 positions in lichess-200")
        let dir = try await saveSession(champion: weights, trainer: weights + Self.velocity(for: weights),
                                        evaluator: FixtureTestSetEvaluator(failed))
        XCTAssertEqual(try Self.reading(SessionCheckpointLayout.championURL(in: dir)), .recorded(failed))
        XCTAssertEqual(try Self.reading(SessionCheckpointLayout.trainerURL(in: dir)), .recorded(failed))
        XCTAssertNoThrow(try CheckpointManager.loadSession(at: dir), "the session still loads")

        let url = try await CheckpointManager.saveModel(
            weights: weights, modelID: "20261007-3-MODL", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "manual", trainingStep: nil, parentModelID: "", notes: "save model"),
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
            testSetEvaluator: FixtureTestSetEvaluator(failed), trigger: "unittest", modelsDirectory: tempDir)
        XCTAssertEqual(try Self.reading(url), .recorded(failed))
    }

    /// Probe isolation (plan validation 6): evaluating a trainer's exported
    /// state (velocity included, which the evaluator ignores) leaves the
    /// trainer's weights, velocity, schedule, dropout state and gradient-norm
    /// history as they were; the same weights evaluate the same twice.
    func testAnEvaluationLeavesTheTrainerUntouchedAndRepeats() async throws {
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 41), arch: .current, initialization: .seeded(initSeed: 8))
        _ = try await trainer.trainStep(batchSize: 8)
        let before = try await trainer.exportResumeSnapshot()
        let first = await Self.smallEvaluator.evaluate(weights: before.trainerWeights, architecture: .current)
        let second = await Self.smallEvaluator.evaluate(weights: before.trainerWeights, architecture: .current)
        let after = try await trainer.exportResumeSnapshot()
        XCTAssertNotNil(Self.evaluatedSets(.recorded(first)))
        XCTAssertEqual(Self.evaluatedSets(.recorded(first)), Self.evaluatedSets(.recorded(second)))
        XCTAssertEqual(after.trainerWeights, before.trainerWeights)
        XCTAssertEqual(after.schedule, before.schedule)
        XCTAssertEqual(after.dropoutRNG, before.dropoutRNG)
        XCTAssertEqual(after.gradNormHistory, before.gradNormHistory)
    }

    /// A trainer file that can't be encoded fails the save before the replay
    /// buffer is written and before any evaluation: every encoding check runs
    /// first (`SafetensorsModelIO.prepare`).
    func testAnUnencodableTrainerFileFailsBeforeTheReplayBufferIsWritten() async throws {
        let base = NetworkArchitecture.current.weightTensorPlan().map { [Float](repeating: 0.01, count: $0.elementCount) }
        let trainer = Array((base + Self.velocity(for: base)).dropLast())
        var state = try minimalState(sessionID: "20261007-1-CHMP", championID: "20261007-1-CHMP", trainerID: "20261007-2-TRNR")
        state.hasReplayBuffer = true
        let buffer = ReplayBuffer(capacity: 8, inputEncoding: NetworkArchitecture.current.inputEncoding, sampler: DCMRandom(seed: 1))
        let evaluator = FixtureTestSetEvaluator()
        let bufferWrites = OSAllocatedUnfairLock(initialState: 0)
        let meta = ModelCheckpointMetadata(creator: "manual", trainingStep: 1, parentModelID: "", notes: "unencodable")
        do {
            _ = try await CheckpointManager.saveSession(
                championWeights: base, championID: "20261007-1-CHMP", championMetadata: meta, championCreatedAtUnix: 1_790_000_000,
                trainerWeights: trainer, trainerID: "20261007-2-TRNR", trainerMetadata: meta, trainerCreatedAtUnix: 1_790_000_001,
                state: state, lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
                championLineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
                testSetEvaluator: evaluator, replayBuffer: buffer, trigger: "unittest", sessionsDirectory: tempDir,
                onReplayBufferWritten: { bufferWrites.withLock { $0 += 1 } })
            XCTFail("a trainer file one velocity tensor short must not save")
        } catch SafetensorsModelIO.IOError.tensorCountMismatch {
            // Expected.
        }
        XCTAssertEqual(bufferWrites.withLock { $0 }, 0, "the replay buffer must not be written")
        XCTAssertTrue(evaluator.evaluatedWeights.isEmpty, "no evaluation of a file that can't be written")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: tempDir.path), [])
    }

    /// A tensor of the wrong size is refused with its name, instead of a
    /// short linear tensor trapping in the torch-layout transpose or a long
    /// one being truncated, and a velocity tensor is checked too.
    func testAWrongSizedTensorIsRefusedByName() throws {
        let arch = NetworkArchitecture.current
        let plan = arch.weightTensorPlan()
        let base = plan.map { [Float](repeating: 0.01, count: $0.elementCount) }
        let lineage = try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil)
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "sizes")
        let linear = try XCTUnwrap(plan.firstIndex { $0.kind == .linear })
        for longer in [false, true] {
            var weights = base
            if longer {
                weights[linear].append(0)
            } else {
                weights[linear].removeLast()
            }
            XCTAssertThrowsError(try SafetensorsModelIO.prepare(
                modelID: "20261007-1-SIZE", createdAtUnix: 1, metadata: meta, weights: weights,
                architecture: arch, includesVelocity: false, lineage: lineage)) { error in
                guard case SafetensorsModelIO.IOError.tensorShapeMismatch(let name, _, _) = error else {
                    return XCTFail("\(error)")
                }
                XCTAssertEqual(name, plan[linear].name)
            }
        }
        var withVelocity = base + Self.velocity(for: base)
        withVelocity[withVelocity.count - 1].append(0)
        XCTAssertThrowsError(try SafetensorsModelIO.prepare(
            modelID: "20261007-1-SIZE", createdAtUnix: 1, metadata: meta, weights: withVelocity,
            architecture: arch, includesVelocity: true, lineage: lineage)) { error in
            guard case SafetensorsModelIO.IOError.tensorShapeMismatch(let name, _, _) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertTrue(name.hasPrefix("opt."), name)
        }
    }

    // MARK: - Derive and graft

    private static func encodedSource(_ arch: NetworkArchitecture, results: ModelTestSetResultsField) throws -> Data {
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.0625 + 0.03125 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261007-4-SRCE", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil), testSetResults: results)
    }

    /// A derived file records its own weights' results, never the source's.
    func testADeriveRecordsItsOwnResultsNotTheSources() throws {
        var source = NetworkArchitecture.current
        try source.setMainActivationEverywhere(.relu)
        let sourceData = try Self.encodedSource(source, results: Self.evaluatedFixture)
        let derivedResults = ModelTestSetResultsField.failed(reason: "derived evaluation")
        var asked: [(weights: Int, architecture: NetworkArchitecture)] = []
        let result = try ModelDerivation.derive(
            sourceData: sourceData, sourceName: "source.safetensors",
            operations: [SetActivationDeriveOperation(value: .leakyRelu)],
            newModelID: "20261007-5-DRV1", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["test"],
            renamedTo: nil,
            testSetEvaluation: { weights, architecture in
                asked.append((weights.count, architecture))
                return derivedResults
            })
        XCTAssertEqual(try Self.reading(result.data), .recorded(derivedResults))
        XCTAssertEqual(asked.count, 1)
        XCTAssertEqual(asked.first?.weights, result.targetArchitecture.weightTensorPlan().count)
        XCTAssertEqual(asked.first?.architecture, result.targetArchitecture)
    }

    func testAGraftRecordsItsOwnResults() throws {
        let source = NetworkArchitecture.preset(.nt8y_3x3stem)
        var target = source
        target.blockGroups[0].count += 1
        let sourceData = try Self.encodedSource(source, results: Self.evaluatedFixture)
        let fresh = try GraftFreshTarget.build(architecture: target, initSeed: 1234)
        let graftResults = ModelTestSetResultsField.failed(reason: "graft evaluation")
        var askedArchitecture: NetworkArchitecture?
        let result = try ModelDerivation.graft(
            sourceData: sourceData, sourceName: "source.safetensors", fresh: fresh, targetLabel: "test target",
            targetPreset: nil, map: .empty, initSeedOrigin: "entered", newModelID: "20261007-6-GRFT",
            createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["test"], renamedTo: nil,
            testSetEvaluation: { _, architecture in
                askedArchitecture = architecture
                return graftResults
            })
        XCTAssertEqual(try Self.reading(result.data), .recorded(graftResults))
        XCTAssertEqual(askedArchitecture, target)
    }

    // MARK: - Corpus replay

    /// A corpus-replay save carries the real evaluation of the file's
    /// weights, for every bundled set.
    @MainActor
    func testACorpusReplaySaveCarriesTheEvaluation() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let p = try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            LRWarmupSteps.id: .int(0),
            KLProbeInterval.id: .int(0),
        ]))
        let outModel = tempDir.appendingPathComponent("out.safetensors")
        let config = CorpusReplayConfig(
            corpusDirectories: [corpus], stepLimit: 5, epochs: nil, startModelPath: start.path, presetName: nil,
            startShard: nil, startGameIndex: nil, resumeExact: false, acceptInexact: [],
            outModelPath: outModel.path, overwriteOutModel: false, runModelID: "20261007-7-REPL",
            output: try CliResultsOutput.preflight(url: tempDir.appendingPathComponent("results.json"), overwriteAuthorized: false),
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 0x7E57, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
        _ = try await CorpusReplayRunner.runReplay(config: config, params: p, abort: ReplayAbortFlag())

        guard let sets = Self.evaluatedSets(try Self.reading(outModel)) else { return }
        XCTAssertEqual(sets.map(\.id), ["lichess-200", "lichess-wide"])
        XCTAssertEqual(sets.map(\.positions), [200, 4435])
        let file = try CheckpointManager.loadModelFile(at: outModel)
        let again = await ModelTestSetEvaluator.modelFiles.evaluate(weights: file.weights, architecture: file.architecture)
        XCTAssertEqual(sets, Self.evaluatedSets(.recorded(again)))
    }
}
