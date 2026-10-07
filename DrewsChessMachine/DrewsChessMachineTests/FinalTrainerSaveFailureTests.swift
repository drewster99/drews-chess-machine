//
//  FinalTrainerSaveFailureTests.swift
//  DrewsChessMachineTests
//
//  A CLI training run's last save — the final save at its step or epoch
//  limit, or the abort save on Ctrl-C — is the only record of the state it
//  ends in. A failure of an earlier save is a warning the first time
//  (`TrainerSaveFailureStreak`), because the next save can still succeed;
//  a failure of the last one has no next save, so the run must fail rather
//  than exit as if its end state had been saved. Corpus replay and
//  train-vs-UCI apply the same rule through
//  `TrainerSaveFailureStreak.requireLastSaveSucceeded`.
//
//  The corpus-replay cases run the real replay loop
//  (`CorpusReplayRunner.runReplay`) over `ResumeEquivalenceTests`' synthetic
//  corpus with the rolling output in a folder the run cannot write to, so the
//  first save it attempts — its last one — fails with a permission error (not
//  disk full, which already halts a run on any save).
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class FinalTrainerSaveFailureTests: XCTestCase {

    private var tempDir: URL!
    private var readOnlyDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-final-save-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
        let readOnly = dir.appendingPathComponent("read-only", isDirectory: true)
        try FileManager.default.createDirectory(at: readOnly, withIntermediateDirectories: false)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: readOnly.path)
        readOnlyDir = readOnly
    }

    override func tearDown() async throws {
        if let readOnlyDir {
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: readOnlyDir.path)
        }
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    // MARK: - Corpus replay

    func testCorpusReplayFailsWhenItsFinalSaveFails() async throws {
        guard let error = try await replayError(stepLimit: 3, abortBeforeStart: false) else {
            return XCTFail("the run ended without an error, but its final save failed")
        }
        XCTAssertTrue(error.localizedDescription.contains("final"),
                      "the error names the failed save: \(error.localizedDescription)")
    }

    func testCorpusReplayFailsWhenItsAbortSaveFails() async throws {
        guard let error = try await replayError(stepLimit: 3, abortBeforeStart: true) else {
            return XCTFail("the run ended without an error, but its abort save failed")
        }
        XCTAssertTrue(error.localizedDescription.contains("abort"),
                      "the error names the failed save: \(error.localizedDescription)")
    }

    /// Run the real replay loop with its rolling output in the read-only
    /// folder; the error it ends with, or nil when it ends without one.
    private func replayError(stepLimit: Int, abortBeforeStart: Bool) async throws -> Error? {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let p = try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            LRWarmupSteps.id: .int(5),
            KLProbeInterval.id: .int(0),
            BatchStatsInterval.id: .int(0),
        ]))
        let outModel = readOnlyDir.appendingPathComponent("out-replay-latest.safetensors")
        let config = CorpusReplayConfig(
            corpusDirectories: [corpus],
            stepLimit: stepLimit,
            epochs: nil,
            startModelPath: start.path,
            presetName: nil,
            startShard: nil,
            startGameIndex: nil,
            resumeExact: false,
            acceptInexact: [],
            outModelPath: outModel.path,
            overwriteOutModel: false,
            runModelID: "20261003-3-FSVF",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 0xF1A1, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
        let abort = ReplayAbortFlag()
        if abortBeforeStart { abort.request() }
        do {
            _ = try await CorpusReplayRunner.runReplay(config: config, params: p, abort: abort)
            return nil
        } catch {
            XCTAssertFalse(FileManager.default.fileExists(atPath: outModel.path),
                           "the read-only folder holds no model, so the save really failed")
            return error
        }
    }
}
