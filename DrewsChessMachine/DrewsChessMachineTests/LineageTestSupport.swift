import Foundation
@testable import DrewsChessMachine

extension LineageRecord {
    /// A lineage record for a file a test writes: a fresh run's record with
    /// `trainerCompletedSteps` as its step total (a trainer-state file's
    /// writer requires it to equal the file's trainer clock).
    static func forTests(trainerCompletedSteps: Int?, corpus: CorpusPosition?) throws -> LineageRecord {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(
            start: .fresh, pathKind: .replay, argv: ["DrewsChessMachine", "--test"],
            startedAt: start, segmentStartTrainerStep: 0)
        return try tracker.record(
            at: start.addingTimeInterval(60),
            trainerCompletedSteps: trainerCompletedSteps,
            segmentLocalStep: trainerCompletedSteps ?? 0,
            segmentGames: 0,
            segmentPositions: 0,
            corpus: corpus,
            parameters: nil,
            dropoutPhiloxState: nil)
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
    static let sessionTestFixture = LineageRecord(
        schema: LineageRecord.currentSchema,
        run: Run(lineageRunID: "00000000-0000-0000-0000-000000000001", segmentIndex: 0,
                 segmentID: "00000000-0000-0000-0000-000000000002", segmentStartedUnix: 1_700_000_000,
                 start: .fresh, exactResume: false, notExactItems: [], continuesUnrecordedHistory: false,
                 recordedUnix: 1_700_000_100),
        parent: nil,
        steps: Steps(cumTrainerStep: 1234, segmentStartTrainerStep: 0, segmentLocalStep: 1234),
        fed: Fed(cumGames: 10, cumPositions: 600, segmentGames: 10, segmentPositions: 600, corpus: nil),
        time: Time(cumTrainStepSec: 900, cumWallSec: 1000, segmentTrainStepSec: 900, segmentWallSec: 1000),
        parameters: nil,
        build: Build(buildNumber: 1, gitHash: "0000000", gitBranch: "main", gitDirty: false),
        invocation: Invocation(argv: ["DrewsChessMachine"], pathKind: .gui),
        device: Device(hardwareModel: "Mac16,8", cpu: "Apple M4 Pro", isVirtualMachine: false,
                       osVersion: "Version 27.2", gpu: "Apple M4 Pro"),
        rng: .unseeded(dropoutPhiloxState: nil),
        segments: [],
        derivationHistory: [])
}
