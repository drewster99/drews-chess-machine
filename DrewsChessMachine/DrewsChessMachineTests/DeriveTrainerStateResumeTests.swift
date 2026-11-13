//
//  DeriveTrainerStateResumeTests.swift
//  DrewsChessMachineTests
//
//  A precision-only derive of a trainer-state file continued through the real
//  paths (GPU; owner decisions 2026-11-12,
//  `documentation/plans-active/DERIVE_TRAINER_STATE_DTYPE_PLAN.md`):
//
//  - the trainer state crosses a compute-dtype change losslessly: a bf16
//    trainer's fp32 masters load as an fp32 trainer's working weights and
//    back, bit for bit, velocity included — the premise that makes carrying
//    the state worth anything;
//  - corpus replay `--resume-exact` of a bf16 save derived to fp32 is refused
//    naming `compute_data_type` and `policy_tail_precision` until they are
//    accepted, then continues as the next segment of the same run (OD-B),
//    from the saved clock and feed position, in fp32. It runs on the same
//    build, OS and Mac as the save, which pins the reason for the gaps:
//    without them such a resume would have logged EXACT.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class DeriveTrainerStateResumeTests: XCTestCase {

    private var tempDir: URL!
    private var corpusDir: URL!

    private static let runSeed: UInt64 = 0xDE51_7E57

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-derive-trainer-state-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
        corpusDir = try ResumeEquivalenceTests.writeCorpus(in: dir)
    }

    override func tearDown() async throws {
        if let tempDir {
            do { try FileManager.default.removeItem(at: tempDir) } catch { /* best-effort cleanup */ }
        }
        try await super.tearDown()
    }

    private func bf16() throws -> NetworkArchitecture {
        try DeriveTrainerStateTests.bf16Architecture()
    }

    /// The replay tests' parameters: declared defaults with a small batch and
    /// buffer, dropout on, a short warmup, diagnostics off.
    private func params() throws -> ReplayParams {
        try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            ReplayRatioTarget.id: .double(0.48),
            DropoutRate.id: .double(0.1),
            LRWarmupSteps.id: .int(5),
            KLProbeInterval.id: .int(0),
            BatchStatsInterval.id: .int(0),
        ]))
    }

    private func config(stepLimit: Int, startModel: URL, resumeExact: Bool, accepting: Set<ResumeGap>,
                        out: String) -> CorpusReplayConfig {
        CorpusReplayConfig(
            corpusDirectories: [corpusDir],
            stepLimit: stepLimit,
            epochs: nil,
            startModelPath: startModel.path,
            presetName: nil,
            startShard: nil,
            startGameIndex: nil,
            resumeExact: resumeExact,
            acceptInexact: accepting,
            outModelPath: tempDir.appendingPathComponent(out).path,
            overwriteOutModel: false,
            runModelID: "20261112-2-DTSR",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: Self.runSeed, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
    }

    /// A bf16 start model with seeded weights and no trainer state.
    private func writeStartModel() async throws -> URL {
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 11), arch: try bf16())
        let weights = try await net.network.exportWeights()
        let data = try SafetensorsModelIO.encode(
            modelID: "20261112-1-BFST", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "",
                                              notes: "derive-trainer-state start model"),
            weights: weights, architecture: try bf16(), includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let url = tempDir.appendingPathComponent("start.safetensors")
        try data.write(to: url, options: .withoutOverwriting)
        return url
    }

    // MARK: - Tests

    func testTrainerStateCrossesAComputeDtypeChangeLosslessly() async throws {
        let p = try params()
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 11), arch: try bf16())
        let start = try await net.network.exportWeights()
        let bf16Trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 3), hyperparameters: p.trainer,
                                           arch: try bf16(), initialization: .overwrittenByLoad)
        try await bf16Trainer.loadBaseWeightsResetVelocity(start)
        let buffer = try ResumeEquivalenceTests.fixtureBuffer(sampler: DCMRandom(seed: 5))
        for _ in 0..<3 {
            let timing = try await bf16Trainer.trainStep(replayBuffer: buffer, batchSize: p.trainingBatchSize)
            XCTAssertNotNil(timing, "the fixture buffer holds more than a batch")
        }
        let bf16State = try await bf16Trainer.exportTrainerWeights()

        let fp32Trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 3), hyperparameters: p.trainer,
                                           arch: try bf16().withComputeDataType(.float32, tail: nil),
                                           initialization: .overwrittenByLoad)
        try await fp32Trainer.loadTrainerWeights(bf16State)
        let fp32State = try await fp32Trainer.exportTrainerWeights()
        XCTAssertEqual(fp32State.count, bf16State.count)
        for (index, (a, b)) in zip(fp32State, bf16State).enumerated() {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern),
                           "tensor \(index): bf16 masters → fp32 working weights (and velocity) bit for bit")
        }

        let backTrainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 3), hyperparameters: p.trainer,
                                           arch: try bf16(), initialization: .overwrittenByLoad)
        try await backTrainer.loadTrainerWeights(fp32State)
        let backState = try await backTrainer.exportTrainerWeights()
        for (index, (a, b)) in zip(backState, bf16State).enumerated() {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern),
                           "tensor \(index): fp32 working weights → bf16 masters bit for bit")
        }
    }

    func testCorpusReplayResumesADerivedSaveAsTheNextSegmentWithPrecisionGaps() async throws {
        let n = 12, m = 8
        let startURL = try await writeStartModel()
        let first = try await CorpusReplayRunner.runReplay(
            config: config(stepLimit: n, startModel: startURL, resumeExact: false, accepting: [], out: "bf16.safetensors"),
            params: try params(), abort: ReplayAbortFlag())
        XCTAssertEqual(first.steps, n)
        let saveURL = tempDir.appendingPathComponent("bf16.safetensors")
        let saveData = try Data(contentsOf: saveURL)
        let save = try SafetensorsModelIO.decode(saveData).file
        let saveRecord = try XCTUnwrap(save.lineageParent.lineage.record)
        XCTAssertEqual(save.architecture.computeDataType, .bFloat16)
        XCTAssertNotNil(saveRecord.fed.corpus)
        XCTAssertNotNil(saveRecord.rng.streams)
        XCTAssertNotNil(saveRecord.rng.behaviorFingerprint)

        let derived = try ModelDerivation.derive(
            sourceData: saveData, sourceName: "bf16.safetensors",
            operations: [SetComputeDataTypeDeriveOperation(value: .float32)], newModelID: "20261112-3-FP32",
            createdAtUnix: 1_790_002_000, build: "test", invocationArguments: ["DrewsChessMachine", "--derive-model"],
            renamedTo: nil)
        XCTAssertEqual(derived.lineage.run, saveRecord.run)
        XCTAssertEqual(derived.lineage.fed, saveRecord.fed)
        XCTAssertEqual(derived.lineage.rng, saveRecord.rng)
        let derivedURL = tempDir.appendingPathComponent("fp32-start.safetensors")
        try derived.data.write(to: derivedURL, options: .withoutOverwriting)

        // Same build, OS and Mac: only the precision gaps, and they refuse.
        do {
            _ = try await CorpusReplayRunner.runReplay(
                config: config(stepLimit: m, startModel: derivedURL, resumeExact: true, accepting: [],
                               out: "refused.safetensors"),
                params: try params(), abort: ReplayAbortFlag())
            XCTFail("a resume of a derived precision change must be refused until its gaps are accepted")
        } catch let refusal as CLIRunRefusal {
            XCTAssertTrue(refusal.message.contains("lacks compute_data_type,policy_tail_precision."), refusal.message)
        }

        let resumed = try await CorpusReplayRunner.runReplay(
            config: config(stepLimit: m, startModel: derivedURL, resumeExact: true,
                           accepting: [.computeDataType, .policyTailPrecision], out: "fp32.safetensors"),
            params: try params(), abort: ReplayAbortFlag())
        XCTAssertEqual(resumed.steps, m)
        let end = try SafetensorsModelIO.decode(Data(contentsOf: tempDir.appendingPathComponent("fp32.safetensors"))).file
        let endRecord = try XCTUnwrap(end.lineageParent.lineage.record)
        XCTAssertEqual(end.architecture.computeDataType, .float32)
        XCTAssertEqual(end.architecture.policyTailPrecision, .doesNotApply)
        XCTAssertEqual(end.metadata.trainerSchedule?.completedTrainSteps, n + m, "the clock continues from the save")
        // The same run's next segment, recorded as not exact with the two gaps.
        XCTAssertEqual(endRecord.run.lineageRunID, saveRecord.run.lineageRunID)
        XCTAssertEqual(endRecord.run.segmentIndex, saveRecord.run.segmentIndex + 1)
        XCTAssertEqual(endRecord.run.start, .resume)
        XCTAssertFalse(endRecord.run.exactResume)
        XCTAssertEqual(endRecord.run.notExactItems, ["compute_data_type", "policy_tail_precision"])
        XCTAssertEqual(endRecord.parent?.modelID, "20261112-3-FP32")
        XCTAssertEqual(endRecord.segments.last?.segmentID, saveRecord.run.segmentID)
        XCTAssertEqual(endRecord.steps.cumTrainerStep, n + m)
        XCTAssertEqual(endRecord.configuration.value?.policyTailPrecision, PolicyTailPrecisionSetting.doesNotApply.rawValue)
        // The feed continued from the saved position, not from the corpus start.
        let savedCorpus = try XCTUnwrap(saveRecord.fed.corpus)
        let endCorpus = try XCTUnwrap(endRecord.fed.corpus)
        XCTAssertEqual(endCorpus.segmentStart.value?.nextGameIndex, savedCorpus.nextGameIndex)
        XCTAssertEqual(endCorpus.segmentStart.value?.epoch, savedCorpus.epoch)
        // The derive stays in the history; the resumed segment's save is a
        // training save again, captured under its own (fp32) architecture.
        XCTAssertEqual(endRecord.derivationHistory.last?.modelID, "20261112-3-FP32")
        let capture = try TrainerStateCapture(fileArchitecture: end.architecture, record: endRecord)
        XCTAssertEqual(capture.architecture, end.architecture)
        XCTAssertEqual(capture.derivations, [])
    }
}
