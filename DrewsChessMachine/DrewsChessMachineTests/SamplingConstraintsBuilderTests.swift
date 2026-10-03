//
//  SamplingConstraintsBuilderTests.swift
//  DrewsChessMachineTests
//
//  Every training path builds its replay buffer's sampling constraints with
//  one rule: the GUI from the live parameters, corpus replay and
//  train-vs-UCI (through `ReplayParams`) from a snapshot. These tests pin
//  that the two entry points agree, what the rule produces, and how the
//  constraints reach the hyperparameter log line and `results.json`.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class SamplingConstraintsBuilderTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    private func setSampling(maxPerGame: Int, maxDrawPercent: Int, targetLength: Int, stratify: Bool) {
        let p = TrainingParameters.shared
        p.maxPliesFromAnyOneGame = maxPerGame
        p.maxDrawPercentPerBatch = maxDrawPercent
        p.targetSampledGameLengthPlies = targetLength
        p.replayBufferStratifyByMaterial = stratify
    }

    /// The snapshot path (corpus replay, train-vs-UCI) and the live path (the
    /// GUI) build the same constraints from the same parameter values.
    func testTheReplayPathAndTheGUIPathBuildTheSameConstraints() throws {
        for (cap, draw, target, stratify) in [(10, 100, 999, false), (1, 40, 30, false), (25, 100, 0, true)] {
            setSampling(maxPerGame: cap, maxDrawPercent: draw, targetLength: target, stratify: stratify)
            let live = ReplayBuffer.SamplingConstraints.fromCurrentParameters()
            let replay = try ReplayParams(TrainingParameters.shared.snapshot()).samplingConstraints
            XCTAssertEqual(replay, live, "cap \(cap) draw \(draw) target \(target) stratify \(stratify)")
            XCTAssertEqual(replay.maxPerGame, cap)
            XCTAssertEqual(replay.maxDrawPercent, draw)
            XCTAssertEqual(replay.targetMeanGameLengthPlies, target)
        }
    }

    /// Stratification weights the active material buckets equally and leaves
    /// the unreachable last bucket at zero; off means no bucket weights.
    func testStratificationWeightsTheActiveBucketsEqually() throws {
        setSampling(maxPerGame: 10, maxDrawPercent: 100, targetLength: 999, stratify: true)
        let weights = try XCTUnwrap(ReplayBuffer.SamplingConstraints(TrainingParameters.shared.snapshot())
            .materialBucketWeights)
        let bucketCount = ReplayBufferAnalyzer.materialBuckets.count
        XCTAssertEqual(weights.count, bucketCount)
        XCTAssertEqual(weights.last, 0)
        for w in weights.dropLast() { XCTAssertEqual(w, 1 / Float(bucketCount - 1)) }
        setSampling(maxPerGame: 10, maxDrawPercent: 100, targetLength: 999, stratify: false)
        XCTAssertNil(ReplayBuffer.SamplingConstraints(TrainingParameters.shared.snapshot()).materialBucketWeights)
    }

    /// The declared defaults are not the uniform sampler: the per-game cap
    /// binds below a typical batch size and the length target is set, so a
    /// default run takes the constrained path.
    func testTheDeclaredDefaultsTakeTheConstrainedPath() {
        let defaults = ReplayBuffer.SamplingConstraints.fromParameters(
            maxPliesFromAnyOneGame: MaxPliesFromAnyOneGame.declaredDefault,
            maxDrawPercentPerBatch: MaxDrawPercentPerBatch.declaredDefault,
            targetSampledGameLengthPlies: TargetSampledGameLengthPlies.declaredDefault,
            stratifyByMaterial: ReplayBufferStratifyByMaterial.declaredDefault)
        XCTAssertFalse(defaults.isNoOp(forBatchSize: TrainingBatchSize.declaredDefault))
        XCTAssertTrue(ReplayBuffer.SamplingConstraints.unconstrained.isNoOp(forBatchSize: TrainingBatchSize.declaredDefault))
    }

    /// The hyperparameter log line names every constraint and whether a batch
    /// of the run's size takes the constrained path.
    func testTheHyperparameterLogFieldsNameEveryConstraint() {
        let constrained = ReplayBuffer.SamplingConstraints(
            maxPerGame: 10, maxDrawPercent: 60, targetMeanGameLengthPlies: 999, materialBucketWeights: nil)
        XCTAssertEqual(constrained.logFields(batchSize: 4096),
                       " sampling=(maxPerGame=10 maxDrawPct=60 targetLen=999 stratify=off applied=on)")
        let inactive = ReplayBuffer.SamplingConstraints(
            maxPerGame: 400, maxDrawPercent: 100, targetMeanGameLengthPlies: 0, materialBucketWeights: nil)
        XCTAssertEqual(inactive.logFields(batchSize: 32),
                       " sampling=(maxPerGame=400 maxDrawPct=100 targetLen=0 stratify=off applied=off)")
    }

    /// A declared-defaults snapshot ignores the live settings, applies its
    /// overrides, and refuses an override outside its declared range or for
    /// an unknown parameter.
    func testADeclaredDefaultsSnapshotIgnoresTheLiveSettings() throws {
        setSampling(maxPerGame: 3, maxDrawPercent: 40, targetLength: 50, stratify: true)
        let pinned = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            MaxDrawPercentPerBatch.id: MaxDrawPercentPerBatch.encode(70),
        ])
        XCTAssertEqual(pinned.maxPliesFromAnyOneGame, MaxPliesFromAnyOneGame.declaredDefault)
        XCTAssertEqual(pinned.targetSampledGameLengthPlies, TargetSampledGameLengthPlies.declaredDefault)
        XCTAssertEqual(pinned.replayBufferStratifyByMaterial, ReplayBufferStratifyByMaterial.declaredDefault)
        XCTAssertEqual(pinned.maxDrawPercentPerBatch, 70)
        XCTAssertThrowsError(try TrainingParametersSnapshot.declaredDefaults(overriding: [
            MaxDrawPercentPerBatch.id: MaxDrawPercentPerBatch.encode(101),
        ]))
        XCTAssertThrowsError(try TrainingParametersSnapshot.declaredDefaults(overriding: ["no_such_parameter": .int(1)]))
    }

    /// `results.json` carries the run's constraints, and leaves the key out
    /// for a run that set none.
    func testResultsRecordTheRunsSamplingConstraints() throws {
        let recorder = CliTrainingRecorder()
        let without = try JSONSerialization.jsonObject(with: recorder.encodedJSONData(totalTrainingSeconds: 1))
        XCTAssertNil((without as? [String: Any])?["sampling_constraints"])

        recorder.setSamplingConstraints(
            ReplayBuffer.SamplingConstraints(maxPerGame: 10, maxDrawPercent: 100, targetMeanGameLengthPlies: 999,
                                             materialBucketWeights: [0.25, 0.25, 0.25, 0.25, 0]),
            batchSize: 4096)
        let root = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: recorder.encodedJSONData(totalTrainingSeconds: 1)) as? [String: Any])
        let recorded = try XCTUnwrap(root["sampling_constraints"] as? [String: Any])
        XCTAssertEqual(recorded["applied"] as? Bool, true)
        XCTAssertEqual(recorded["max_per_game"] as? Int, 10)
        XCTAssertEqual(recorded["max_draw_pct"] as? Int, 100)
        XCTAssertEqual(recorded["target_length"] as? Int, 999)
        XCTAssertEqual(recorded["stratify_by_material"] as? Bool, true)
    }
}
