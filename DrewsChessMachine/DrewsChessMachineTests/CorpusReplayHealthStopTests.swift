import XCTest
@testable import DrewsChessMachine

/// The training-health alarms on the real corpus-replay loop
/// (`CorpusReplayRunner.runReplay` over `ResumeEquivalenceTests`' synthetic
/// corpus), in-process — the alarms plan's V-3 / V-4 on a run small enough
/// for the suite.
///
/// The trigger is deterministic: with warmup and the learning grace both 0
/// the learning gate is trainer step 0, and with a learning rate of 1e-7 (the declared minimum)
/// the start model's illegal-move mass (≈ 0.99 at random init) cannot move,
/// so `illegal_mass`'s not-learned form holds on every window: raised
/// critical at the second live evaluation (trainer step 100: sustain is two
/// evaluations spanning ≥ 50 steps, live evaluations every 50).
@MainActor
final class CorpusReplayHealthStopTests: XCTestCase {

    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-health-stop-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
    }

    override func tearDown() async throws {
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    private func run(
        stepLimit: Int,
        illegalMassAction: TrainingHealthAction
    ) async throws -> (result: CorpusReplayRunner.Result, results: [String: Any]) {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let p = try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            LRWarmupSteps.id: .int(0),
            LearningRate.id: .double(1e-7),
            KLProbeInterval.id: .int(0),
            BatchStatsInterval.id: .int(10),
            TrainingHealthLearningGraceSteps.id: .int(0),
            TrainingHealthActionIllegalMass.id: .int(illegalMassAction.rawValue),
        ]))
        let resultsURL = tempDir.appendingPathComponent("results-\(UUID().uuidString).json")
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
            outModelPath: tempDir.appendingPathComponent("out-\(UUID().uuidString).safetensors").path,
            overwriteOutModel: false,
            runModelID: "20261006-1-HSTP",
            output: try CliResultsOutput.preflight(url: resultsURL, overwriteAuthorized: false),
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 0x4EA1, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
        let result = try await CorpusReplayRunner.runReplay(config: config, params: p, abort: ReplayAbortFlag())
        let data = try Data(contentsOf: resultsURL)
        let results = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return (result, results)
    }

    /// Log-only (the default action): the alarm is raised and recorded, the
    /// run continues to its step limit, and no stop is requested.
    func testLogOnlyAlarmIsRecordedAndTheRunContinues() async throws {
        let (result, results) = try await run(stepLimit: 150, illegalMassAction: .log)
        XCTAssertEqual(result.steps, 150)
        XCTAssertNil(result.healthStop)
        XCTAssertEqual(results["termination_reason"] as? String, "step_limit_reached")
        let alarms = try XCTUnwrap(results["alarms"] as? [[String: Any]])
        let raise = try XCTUnwrap(alarms.first { $0["rule"] as? String == "illegal_mass" && $0["kind"] as? String == "raise" })
        XCTAssertEqual(raise["severity"] as? String, "critical")
        XCTAssertEqual(raise["trainer_step"] as? Int, 100)
        XCTAssertFalse(alarms.contains { $0["kind"] as? String == "stop" })
        let config = try XCTUnwrap(results["alarm_config"] as? [String: Any])
        XCTAssertEqual((config["actions"] as? [String: String])?["illegal_mass"], "log")
    }

    /// `stop_on_critical`: the run stops before the step after the
    /// evaluation that raised it, saves with reason `health-stop`, and
    /// reports `training_health_alarm`.
    func testStopOnCriticalStopsAtTheRaisingEvaluation() async throws {
        let (result, results) = try await run(stepLimit: 400, illegalMassAction: .stopOnCritical)
        let stop = try XCTUnwrap(result.healthStop)
        XCTAssertEqual(stop.rule, .illegalMass)
        XCTAssertEqual(stop.kind, .stop)
        XCTAssertEqual(stop.trainerStep, 100)
        XCTAssertEqual(result.steps, 100, "no step runs after the evaluation that requested the stop")
        XCTAssertEqual(results["termination_reason"] as? String, "training_health_alarm")
        let alarms = try XCTUnwrap(results["alarms"] as? [[String: Any]])
        let kinds = alarms.filter { $0["rule"] as? String == "illegal_mass" }.compactMap { $0["kind"] as? String }
        XCTAssertEqual(Array(kinds.prefix(2)), ["raise", "stop"])
        XCTAssertEqual(CorpusReplayRunner.trainingHealthStopExitStatus, 35)
    }
}
