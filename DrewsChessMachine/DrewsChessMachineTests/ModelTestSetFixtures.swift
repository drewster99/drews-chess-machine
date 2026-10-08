import Foundation
import os
@testable import DrewsChessMachine

/// Test support for the model-file test-set results (test-set results plan
/// D3). Production writers must pass an evaluation; the many tests that
/// write model files for other reasons call the overloads below, which pass
/// `notEvaluated` explicitly instead of spending a GPU evaluation per file.
/// Tests of the results themselves (`ModelTestSetResultsTests`,
/// `ModelTestSetWritersTests`) pass real or chosen evaluations.
enum ModelTestSetFixtures {
    /// What a fixture file records: a failure naming the fixture, so it can
    /// never read as a real evaluation.
    static let notEvaluated = ModelTestSetResultsField.failed(reason: "test fixture: not evaluated")

    static let notEvaluatedEvaluation: ModelTestSetEvaluation = { _, _ in notEvaluated }
}

/// An evaluator that returns a chosen result and records what it was asked
/// to evaluate.
final class FixtureTestSetEvaluator: ModelTestSetEvaluating, Sendable {
    private let field: ModelTestSetResultsField
    private let evaluated = OSAllocatedUnfairLock<[[[Float]]]>(initialState: [])

    init(_ field: ModelTestSetResultsField = ModelTestSetFixtures.notEvaluated) {
        self.field = field
    }

    /// The weights of every evaluation, in order.
    var evaluatedWeights: [[[Float]]] {
        evaluated.withLock { $0 }
    }

    func evaluate(weights: [[Float]], architecture: NetworkArchitecture) async -> ModelTestSetResultsField {
        evaluated.withLock { $0.append(weights) }
        return field
    }
}

extension SafetensorsModelIO {
    /// `encode`, for a test file whose test-set results don't matter.
    static func encode(
        modelID: String,
        createdAtUnix: Int64,
        metadata: ModelCheckpointMetadata,
        weights: [[Float]],
        architecture: NetworkArchitecture,
        includesVelocity: Bool,
        lineage: LineageRecord
    ) throws -> Data {
        try encode(modelID: modelID, createdAtUnix: createdAtUnix, metadata: metadata, weights: weights,
                   architecture: architecture, includesVelocity: includesVelocity, lineage: lineage,
                   testSetResults: ModelTestSetFixtures.notEvaluated)
    }
}

extension CheckpointManager {
    /// `saveModel`, for a test file whose test-set results don't matter.
    static func saveModel(
        weights: [[Float]],
        modelID: String,
        createdAtUnix: Int64,
        metadata: ModelCheckpointMetadata,
        architecture: NetworkArchitecture = .current,
        lineage: LineageRecord,
        trigger: String,
        at date: Date = Date(),
        modelsDirectory: URL = CheckpointPaths.modelsDir
    ) async throws -> URL {
        try await saveModel(weights: weights, modelID: modelID, createdAtUnix: createdAtUnix, metadata: metadata,
                            architecture: architecture, lineage: lineage,
                            testSetEvaluator: FixtureTestSetEvaluator(),
                            trigger: trigger, at: date, modelsDirectory: modelsDirectory)
    }

    /// `saveSession`, for a test session whose test-set results don't matter.
    static func saveSession(
        championWeights: [[Float]],
        championID: String,
        championMetadata: ModelCheckpointMetadata,
        championCreatedAtUnix: Int64,
        trainerWeights: [[Float]],
        trainerID: String,
        trainerMetadata: ModelCheckpointMetadata,
        trainerCreatedAtUnix: Int64,
        state: SessionCheckpointState,
        lineage: LineageRecord,
        championLineage: LineageRecord,
        architecture: NetworkArchitecture = .current,
        replayBuffer: ReplayBuffer? = nil,
        chartSnapshot: ChartCoordinatorSnapshot? = nil,
        trigger: String,
        at date: Date = Date(),
        sessionsDirectory: URL = CheckpointPaths.sessionsDir,
        onReplayBufferWritten: (@Sendable () -> Void)? = nil
    ) async throws -> URL {
        try await saveSession(
            championWeights: championWeights, championID: championID, championMetadata: championMetadata,
            championCreatedAtUnix: championCreatedAtUnix, trainerWeights: trainerWeights, trainerID: trainerID,
            trainerMetadata: trainerMetadata, trainerCreatedAtUnix: trainerCreatedAtUnix, state: state,
            lineage: lineage, championLineage: championLineage, architecture: architecture,
            testSetEvaluator: FixtureTestSetEvaluator(), replayBuffer: replayBuffer, chartSnapshot: chartSnapshot,
            trigger: trigger, at: date, sessionsDirectory: sessionsDirectory, onReplayBufferWritten: onReplayBufferWritten)
    }
}

extension ModelDerivation {
    /// `derive`, for a test file whose test-set results don't matter.
    static func derive(
        sourceData: Data,
        sourceName: String,
        operations: [any DeriveOperation],
        newModelID: String,
        createdAtUnix: Int64,
        build: String,
        invocationArguments: [String],
        renamedTo newName: String?
    ) throws -> Result {
        try derive(sourceData: sourceData, sourceName: sourceName, operations: operations, newModelID: newModelID,
                   createdAtUnix: createdAtUnix, build: build, invocationArguments: invocationArguments,
                   renamedTo: newName, testSetEvaluation: ModelTestSetFixtures.notEvaluatedEvaluation)
    }

    /// `graft`, for a test file whose test-set results don't matter.
    static func graft(
        sourceData: Data,
        sourceName: String,
        fresh: GraftFreshTarget,
        targetLabel: String,
        targetPreset: String?,
        map: GraftMap,
        initSeedOrigin: String,
        newModelID: String,
        createdAtUnix: Int64,
        build: String,
        invocationArguments: [String],
        renamedTo newName: String?
    ) throws -> GraftResult {
        try graft(sourceData: sourceData, sourceName: sourceName, fresh: fresh, targetLabel: targetLabel,
                  targetPreset: targetPreset, map: map, initSeedOrigin: initSeedOrigin, newModelID: newModelID,
                  createdAtUnix: createdAtUnix, build: build, invocationArguments: invocationArguments,
                  renamedTo: newName, testSetEvaluation: ModelTestSetFixtures.notEvaluatedEvaluation)
    }
}
