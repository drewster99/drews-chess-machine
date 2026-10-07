//
//  ReplayResumeDiffLogTests.swift
//  DrewsChessMachineTests
//
//  A corpus-replay `--resume-exact` that trains under a changed parameter
//  writes one `[RESUME-DIFF]` line for it to the session log — the runner
//  itself, end to end, not only the comparison it calls. The run is the real
//  replay loop on `ResumeEquivalenceTests`' synthetic corpus and start model;
//  the line is read back from the log file this test process writes
//  (`SessionLogger.shared`, started by the test host app).
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ReplayResumeDiffLogTests: XCTestCase {

    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-replay-resume-diff-log-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
    }

    override func tearDown() async throws {
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    /// Declared-default parameters sized for the synthetic corpus, with the
    /// given weight decay.
    private func params(weightDecay: Double) throws -> ReplayParams {
        try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            KLProbeInterval.id: .int(0),
            BatchStatsInterval.id: .int(0),
            WeightDecay.id: .double(weightDecay),
        ]))
    }

    private func config(corpus: URL, startModel: URL, resumeExact: Bool, out: URL) -> CorpusReplayConfig {
        CorpusReplayConfig(
            corpusDirectories: [corpus],
            stepLimit: 2,
            epochs: nil,
            startModelPath: startModel.path,
            presetName: nil,
            startShard: nil,
            startGameIndex: nil,
            resumeExact: resumeExact,
            acceptInexact: [],
            outModelPath: out.path,
            overwriteOutModel: false,
            runModelID: "20261006-5-RDIF",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 0x5D1F, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
    }

    func testAnExactResumeLogsAResumeDiffLineForAChangedParameter() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let first = tempDir.appendingPathComponent("first.safetensors")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, startModel: start, resumeExact: false, out: first),
            params: try params(weightDecay: 0.000271), abort: ReplayAbortFlag())
        let second = tempDir.appendingPathComponent("second.safetensors")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, startModel: first, resumeExact: true, out: second),
            params: try params(weightDecay: 0.000314), abort: ReplayAbortFlag())

        // `activeLogPath` waits behind every line already queued.
        let path = try XCTUnwrap(SessionLogger.shared.activeLogPath, "the test host's session log is open")
        let log = try String(contentsOfFile: path, encoding: .utf8)
        let line = "[RESUME-DIFF] \(WeightDecay.id): parent=0.000271 this_run=0.000314"
        XCTAssertTrue(log.contains(line), "the resume logs the changed parameter: \(line)")
        XCTAssertFalse(log.contains("[RESUME-DIFF] \(WeightDecay.id): parent=0.000271 this_run=0.000271"))
    }
}
