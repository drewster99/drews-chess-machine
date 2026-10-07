//
//  CorpusReplayRefusalTests.swift
//  DrewsChessMachineTests
//
//  A corpus replay run refused at launch — conflicting start flags, a
//  `--resume-exact` that cannot continue its checkpoint exactly — throws a
//  `CLIRunRefusal` out of `CorpusReplayRunner.runReplay`; only `runAndExit`
//  turns it into stderr text and exit status 2, after draining the session
//  log. Refusing by ending the process where the problem was found lost the
//  log's last lines (the `[RESUME]` verdict among them), and ended any
//  in-process caller with it — this test runner included.
//
//  The runs here use `ResumeEquivalenceTests`' synthetic sealed corpus and
//  start model, the same fixtures the resume-equivalence gate trains on.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class CorpusReplayRefusalTests: XCTestCase {

    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-replay-refusal-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
    }

    override func tearDown() async throws {
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    /// Declared-default parameters with the small batch and buffer the
    /// synthetic corpus is sized for, diagnostics off.
    private func params(batchSize: Int) throws -> ReplayParams {
        try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(batchSize),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            LRWarmupSteps.id: .int(5),
            KLProbeInterval.id: .int(0),
            BatchStatsInterval.id: .int(0),
        ]))
    }

    private func config(corpus: URL, stepLimit: Int, startModel: URL?, startShard: Int?, startGameIndex: Int?,
                        resumeExact: Bool, out: URL) -> CorpusReplayConfig {
        CorpusReplayConfig(
            corpusDirectories: [corpus],
            stepLimit: stepLimit,
            epochs: nil,
            startModelPath: startModel?.path,
            presetName: nil,
            startShard: startShard,
            startGameIndex: startGameIndex,
            resumeExact: resumeExact,
            acceptInexact: [],
            outModelPath: out.path,
            overwriteOutModel: false,
            runModelID: "20261003-4-RFSL",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 0x5EF5, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
    }

    /// The refusal `runReplay` throws, or a failure naming what it did
    /// instead.
    private func refusal(_ cfg: CorpusReplayConfig, params p: ReplayParams) async -> CLIRunRefusal? {
        do {
            _ = try await CorpusReplayRunner.runReplay(config: cfg, params: p, abort: ReplayAbortFlag())
            XCTFail("the run trained instead of being refused")
            return nil
        } catch let refusal as CLIRunRefusal {
            return refusal
        } catch {
            XCTFail("the run failed instead of being refused: \(error.localizedDescription)")
            return nil
        }
    }

    func testConflictingStartFlagsThrowInsteadOfExiting() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let out = tempDir.appendingPathComponent("conflict-replay-latest.safetensors")
        let cfg = config(corpus: corpus, stepLimit: 3, startModel: nil, startShard: 0, startGameIndex: 0,
                         resumeExact: false, out: out)
        guard let refusal = await refusal(cfg, params: try params(batchSize: 32)) else { return }
        XCTAssertTrue(refusal.message.contains("--start-shard and --start-game-index are mutually exclusive"),
                      refusal.message)
        XCTAssertFalse(FileManager.default.fileExists(atPath: out.path), "a refused run writes no model")
    }

    /// The first run trains at one batch size; its exact resume at another
    /// feeds a different number of positions per step, so the feed phase
    /// cannot continue and the parameters differ — gaps the resume does not
    /// accept.
    func testAnExactResumeRefusedForAGapThrowsNamingTheGaps() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let first = tempDir.appendingPathComponent("first.safetensors")
        let firstResult = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: start, startShard: nil, startGameIndex: nil,
                           resumeExact: false, out: first),
            params: try params(batchSize: 32), abort: ReplayAbortFlag())
        XCTAssertEqual(firstResult.steps, 3)
        let second = tempDir.appendingPathComponent("second.safetensors")
        let cfg = config(corpus: corpus, stepLimit: 3, startModel: first, startShard: nil, startGameIndex: nil,
                         resumeExact: true, out: second)
        guard let refusal = await refusal(cfg, params: try params(batchSize: 64)) else { return }
        XCTAssertTrue(refusal.message.contains("--resume-exact cannot continue this checkpoint exactly"), refusal.message)
        XCTAssertTrue(refusal.message.contains(ResumeGap.feedCarry.token), refusal.message)
        XCTAssertTrue(refusal.message.contains(ResumeGap.params.token), refusal.message)
        XCTAssertFalse(FileManager.default.fileExists(atPath: second.path), "a refused resume writes no model")
    }
}
