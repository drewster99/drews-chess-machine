//
//  DeriveTrainedSourceTests.swift
//  DrewsChessMachineTests
//
//  `--derive-model` exists to make bit-exact variants of a fresh net for an
//  A/B. An operation that rewrites tensors (a ReZero α init, an SE β init)
//  resets learned weights when the source is trained, while the source's
//  `training_step` and lineage are copied unchanged — a derived file that
//  claims a training step its weights no longer have. These tests pin that a
//  tensor rewrite on a source whose recorded `training_step` is above zero is
//  refused, that a malformed recorded step is an error rather than "absent",
//  and that architecture-only operations (no tensor rewritten) stay allowed
//  on a trained source.
//

import XCTest
@testable import DrewsChessMachine

final class DeriveTrainedSourceTests: XCTestCase {

    /// A ReZero + SE scale-and-bias tower, so both tensor-rewriting
    /// operations have something to rewrite.
    private static let architecture = NetworkArchitecture.preset(.nt8y_3x3stem)

    private func encodedModel(trainingStep: Int?) throws -> Data {
        let arch = Self.architecture
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 5 + $0 % 11) * 0.125 + 0.0625 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: trainingStep, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261002-1-TRND", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: meta.trainerSchedule.map(\.completedTrainSteps), corpus: nil))
    }

    /// `data` with its raw `training_step` metadata value replaced.
    private func withRawTrainingStep(_ data: Data, _ value: String) throws -> Data {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(data)
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata[SafetensorsModelIO.Key.trainingStep] = value
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    private func derive(_ data: Data, _ operations: [any DeriveOperation]) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: data, sourceName: "source.safetensors", operations: operations,
            newModelID: "20261002-2-DRVD", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["test"])
    }

    private func assertRefusedAsTrained(_ data: Data, _ operation: any DeriveOperation,
                                        file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try derive(data, [operation]), file: file, line: line) { error in
            XCTAssertTrue(String(describing: error).contains("training_step"),
                          "refusal should name training_step: \(error)", file: file, line: line)
        }
    }

    func testDeriveRefusesTensorRewritesOnATrainedSource() throws {
        let trained = try encodedModel(trainingStep: 1200)
        assertRefusedAsTrained(trained, SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: nil))
        assertRefusedAsTrained(trained, SetSEBetaInitDeriveOperation(value: .zero, groupIndices: nil))
    }

    func testDeriveRefusesAMalformedRecordedTrainingStep() throws {
        let fresh = try encodedModel(trainingStep: nil)
        for raw in ["twelve", "-5", "1.5"] {
            let malformed = try withRawTrainingStep(fresh, raw)
            assertRefusedAsTrained(malformed, SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: nil))
        }
    }

    func testDeriveAllowsTensorRewritesOnAnUntrainedSource() throws {
        for step in [nil, 0] as [Int?] {
            let source = try encodedModel(trainingStep: step)
            let result = try derive(source, [SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: nil)])
            XCTAssertTrue(result.targetArchitecture.blockGroups.allSatisfy { $0.rezeroAlphaInit == 0 })
        }
    }

    func testArchitectureOnlyOperationsStayAllowedOnATrainedSource() throws {
        let trained = try encodedModel(trainingStep: 1200)
        let cap = try derive(trained, [SetRezeroAlphaCapDeriveOperation(value: 1, groupIndices: nil)])
        XCTAssertTrue(cap.targetArchitecture.blockGroups.allSatisfy { $0.rezeroAlphaCap == 1 })
        let activation = try derive(trained, [SetActivationDeriveOperation(value: .leakyRelu)])
        for site in ArchitectureActivationSite.allCases {
            XCTAssertEqual(activation.targetArchitecture.activation(at: site), activation.targetArchitecture.hasActivationSite(site) ? .leakyRelu : .doesNotApply, "\(site)")
        }
    }
}
