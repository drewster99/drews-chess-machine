import XCTest
@testable import DrewsChessMachine

/// One segment of a training run, its records made by `LineageTracker` the
/// way the CLI paths make them (fresh, exact resume, branch), so every chain
/// and handoff in a test is the real thing, never hand-built.
struct LineageTestSegment {
    let tracker: LineageTracker
    let modelID: String

    /// The record of a save at `localStep`, recorded at `unix`.
    func record(localStep: Int, at unix: Int64) throws -> LineageRecord {
        try tracker.record(
            at: Date(timeIntervalSince1970: TimeInterval(unix)),
            trainerCompletedSteps: (tracker.segmentStartTrainerStep ?? 0) + localStep,
            segmentLocalStep: localStep,
            segmentGames: 0,
            segmentPositions: 0,
            corpus: nil,
            parameters: nil,
            rng: .withoutRunStreams(dropoutPhiloxState: nil),
            inputs: tracker.testInputs)
    }

    var segmentStartTrainerStep: Int? { tracker.segmentStartTrainerStep }
}

enum LineageTestRuns {
    /// A new run's first segment.
    static func fresh(modelID: String, startedUnix: Int64, pathKind: LineageRecord.PathKind = .replay) throws -> LineageTestSegment {
        let tracker = try LineageTracker(
            start: .fresh(initialization: .forTests), pathKind: pathKind, argv: ["DrewsChessMachine", "--test"],
            startedAt: Date(timeIntervalSince1970: TimeInterval(startedUnix)), segmentStartTrainerStep: 0)
        return LineageTestSegment(tracker: tracker, modelID: modelID)
    }

    /// An exact resume of `parent` (a file of `parentModelID` with hash
    /// `parentSHA256`): the same run's next segment.
    static func resume(from parent: LineageRecord, parentModelID: String, parentSHA256: String, modelID: String, startedUnix: Int64, pathKind: LineageRecord.PathKind = .replay) throws -> LineageTestSegment {
        let file = LineageTracker.ParentFile(
            modelID: parentModelID, contentSHA256: parentSHA256,
            trainerCompletedSteps: parent.steps.cumTrainerStep,
            lineage: .recorded(parent), derivationHistory: [])
        let tracker = try LineageTracker(
            start: .resume(parent: file, gaps: [], legacyTotals: nil), pathKind: pathKind, argv: ["DrewsChessMachine", "--test", "--resume-exact"],
            startedAt: Date(timeIntervalSince1970: TimeInterval(startedUnix)), segmentStartTrainerStep: parent.steps.cumTrainerStep)
        return LineageTestSegment(tracker: tracker, modelID: modelID)
    }

    /// A branch from `parent`: a new run.
    static func branch(from parent: LineageRecord, parentModelID: String, parentSHA256: String, modelID: String, startedUnix: Int64) throws -> LineageTestSegment {
        let file = LineageTracker.ParentFile(
            modelID: parentModelID, contentSHA256: parentSHA256,
            trainerCompletedSteps: parent.steps.cumTrainerStep,
            lineage: .recorded(parent), derivationHistory: [])
        let tracker = try LineageTracker(
            start: .branch(parent: file), pathKind: .replay, argv: ["DrewsChessMachine", "--test"],
            startedAt: Date(timeIntervalSince1970: TimeInterval(startedUnix)), segmentStartTrainerStep: 0)
        return LineageTestSegment(tracker: tracker, modelID: modelID)
    }
}
