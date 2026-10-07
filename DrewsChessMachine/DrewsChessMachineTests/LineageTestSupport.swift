import Foundation
@testable import DrewsChessMachine

extension ModelInitRecord {
    /// The init seed and scheme of a fresh run a test starts.
    static let forTests = ModelInitRecord(initSeed: 1, scheme: WeightInitScheme.current)
}

extension LineageRecord {
    /// A lineage record for a file a test writes: a fresh run's record with
    /// `trainerCompletedSteps` as its step total (a trainer-state file's
    /// writer requires it to equal the file's trainer clock).
    static func forTests(trainerCompletedSteps: Int?, corpus: CorpusPosition?) throws -> LineageRecord {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(
            start: .fresh(initialization: .forTests), pathKind: .replay, argv: ["DrewsChessMachine", "--test"],
            startedAt: start, segmentStartTrainerStep: 0)
        return try tracker.record(
            at: start.addingTimeInterval(60),
            trainerCompletedSteps: trainerCompletedSteps,
            segmentLocalStep: trainerCompletedSteps ?? 0,
            segmentGames: 0,
            segmentPositions: 0,
            corpus: corpus,
            parameters: nil,
            rng: .withoutRunStreams(dropoutPhiloxState: nil),
            inputs: tracker.testInputs)
    }

    /// `sessionTestFixture` as the JSON value a session.json fixture at the
    /// current format version embeds under `"lineage"`.
    static let sessionTestFixtureJSON: String = {
        do {
            return try sessionTestFixture.jsonText()
        } catch {
            preconditionFailure("the lineage test fixture does not encode: \(error)")
        }
    }()

    /// A fixed record for state a test builds without throwing (a
    /// session.json fixture at the current format version).
    static let sessionTestFixture: LineageRecord = {
        do {
            return try sessionTestFixtureRecord()
        } catch {
            preconditionFailure("the lineage test fixture does not build: \(error)")
        }
    }()

    private static func sessionTestFixtureRecord() throws -> LineageRecord {
        try LineageRecord(
        run: Run(lineageRunID: "00000000-0000-0000-0000-000000000001", segmentIndex: 0,
                 segmentID: "00000000-0000-0000-0000-000000000002", segmentStartedUnix: 1_700_000_000,
                 start: .fresh, exactResume: false, notExactItems: [], continuesUnrecordedHistory: false,
                 recordedUnix: 1_700_000_100),
        parent: nil,
        steps: Steps(cumTrainerStep: 1234, segmentStartTrainerStep: 0, segmentLocalStep: 1234),
        fed: Fed(cumGames: 10, cumPositions: 600, segmentGames: 10, segmentPositions: 600, corpus: nil),
        time: Time(cumTrainStepSec: 900, cumWallSec: 1000, segmentTrainStepSec: 900, segmentWallSec: 1000),
        parameters: nil,
        configuration: .notTrained,
        runSeeds: .notTrained,
        build: try .forTests,
        invocation: Invocation(argv: ["DrewsChessMachine"], pathKind: .gui),
        device: Device(hardwareModel: "Mac16,8", cpu: "Apple M4 Pro", isVirtualMachine: false,
                       osVersion: "Version 27.2", gpu: "Apple M4 Pro"),
        rng: .withoutRunStreams(dropoutPhiloxState: nil),
        segments: [],
        ancestry: .fresh,
        derivationHistory: [])
    }
}

extension LineageRecord.Build {
    /// A clean build a fixture records.
    static var forTests: LineageRecord.Build {
        get throws {
            try LineageRecord.Build(buildNumber: 1, gitHash: "0000000", gitBranch: "main", gitDirty: false,
                                    gitDiffSHA256: .recorded(nil), xcodeBuild: .recorded("17A5241e"),
                                    sdkBuild: .recorded("26A5300a"), configuration: .recorded("Debug"))
        }
    }
}

extension LineageRecord.Parameters {
    /// A full composed snapshot (every parameter but the seed settings) at
    /// the declared defaults with `overrides`, adopting `schedule`: what a
    /// trainer-state file's record must hold, since the writer refuses a
    /// record whose schedule keys are not the file's schedule.
    static func forTests(adopting schedule: TrainerScheduleState,
                         overriding overrides: [String: ParameterValue] = [:]) throws -> LineageRecord.Parameters {
        try LineageRecord.Parameters(values: try TrainingParametersSnapshot.declaredDefaults(overriding: overrides)
            .adoptingSchedule(schedule).lineageValues())
    }
}

extension LineageTracker {
    /// The save inputs of a test record: the journals as they stand, with
    /// no derived values.
    var testInputs: SaveInputs {
        saveInputs(scheduleAtSave: nil, replayRatioAtSave: nil, healthAlarms: nil)
    }

    /// What a production start notes on its segment before a record with
    /// training behind it: the segment's configuration and the run's seed,
    /// and on a gui segment its `segment_start` champion.
    func noteSegmentStartForTests(trainerStep: Int,
                                  policyTailPrecision: PolicyTailPrecisionSetting = .mixedFinalProjection,
                                  seed: RunRandomSeed = RunRandomSeed.resolve(mode: .seeded, configuredSeed: 7,
                                                                              commandLineSeed: nil, drawSeed: { 0 })) throws {
        try configureSegment(SegmentConfiguration(
            policyTailPrecision: policyTailPrecision, budget: .none,
            vsuci: pathKind == .vsuci
                ? LineageRecord.VsUciGeneration(maxPliesPerGame: 400, evalSyncEverySteps: 10,
                                                trainerMoveSelection: LineageRecord.MoveSelection(.argmax), opponents: [])
                : nil,
            selfPlayDirichlet: pathKind == .gui ? LineageRecord.Dirichlet(.alphaZero) : nil,
            startValueHeadRecentered: .recorded(false)))
        noteRunSeed(seed, atTrainerStep: trainerStep)
        if pathKind == .gui {
            noteChampionChange(LineageRecord.ChampionChange(
                trainerStep: trainerStep, recordedUnix: 1_790_000_000, championModelID: "20261003-1-CHMP",
                championContentSHA256: nil, trigger: .segmentStart))
        }
    }
}
