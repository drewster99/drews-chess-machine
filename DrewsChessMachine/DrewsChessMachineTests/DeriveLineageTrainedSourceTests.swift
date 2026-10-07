//
//  DeriveLineageTrainedSourceTests.swift
//  DrewsChessMachineTests
//
//  A tensor-rewriting derive refuses a trained source. The raw
//  `training_step` key is not enough evidence on its own: a graft writes no
//  `training_step` (its record carries the source's step instead), and a
//  champion copy saved without a trainer clock carries none either, yet both
//  hold trained weights and continue the trained line's totals. These tests
//  pin that the refusal also reads the file's lineage — its step total, the
//  step its parent stated, and every earlier graft's recorded source step —
//  and that a graft of a fresh mint, whose lineage says it was never trained,
//  stays allowed.
//

import XCTest
@testable import DrewsChessMachine

final class DeriveLineageTrainedSourceTests: XCTestCase {

    /// A ReZero tower, so `set-rezero-alpha-init` has tensors to rewrite.
    private static let architecture = NetworkArchitecture.preset(.nt8y_3x3stem)

    private static var graftTarget: NetworkArchitecture {
        var arch = architecture
        arch.blockGroups[0].count += 1
        return arch
    }

    private func weights(for arch: NetworkArchitecture) -> [[Float]] {
        arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 3 + $0 % 7) * 0.125 + 0.0625 }
        }
    }

    private func encodedModel(trainingStep: Int?, lineage: LineageRecord,
                              arch: NetworkArchitecture = DeriveLineageTrainedSourceTests.architecture) throws -> Data {
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: trainingStep, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261003-1-LSRC", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights(for: arch),
            architecture: arch, includesVelocity: false, lineage: lineage)
    }

    private func graft(_ data: Data) throws -> Data {
        let fresh = try GraftFreshTarget.build(architecture: Self.graftTarget, initSeed: 77)
        return try ModelDerivation.graft(
            sourceData: data, sourceName: "source.safetensors", fresh: fresh, targetLabel: "test target", targetPreset: nil,
            map: .empty, initSeedOrigin: "entered", newModelID: "20261003-2-LGRF", createdAtUnix: 1_790_000_100,
            build: "test", invocationArguments: ["test"], renamedTo: nil).data
    }

    private func derive(_ data: Data, _ operation: any DeriveOperation) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: data, sourceName: "derive-source.safetensors", operations: [operation],
            newModelID: "20261003-3-LDRV", createdAtUnix: 1_790_000_200, build: "test", invocationArguments: ["test"], renamedTo: nil)
    }

    private var rewrite: any DeriveOperation { SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: nil) }

    private func assertRefusedAsTrained(_ data: Data, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try derive(data, rewrite), file: file, line: line) { error in
            XCTAssertTrue(String(describing: error).contains("training_step"),
                          "refusal should name training_step: \(error)", file: file, line: line)
        }
    }

    func testRewriteOnAGraftOfATrainedSourceIsRefused() throws {
        let source = try encodedModel(trainingStep: 4200,
                                      lineage: try LineageRecord.forTests(trainerCompletedSteps: 4200, corpus: nil))
        assertRefusedAsTrained(try graft(source))
    }

    func testRewriteOnAGraftOfAPreLineageTrainedSourceIsRefused() throws {
        let source = try encodedModel(trainingStep: 4200,
                                      lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        assertRefusedAsTrained(try graft(source))
    }

    func testRewriteOnAChampionCopyOfATrainedFileIsRefused() throws {
        let trainedParent = LineageTracker.ParentFile(
            modelID: "20261003-4-TRND", contentSHA256: nil, trainerCompletedSteps: 5000,
            lineage: .recorded(try LineageRecord.forTests(trainerCompletedSteps: 5000, corpus: nil)),
            derivationHistory: [])
        let copyRecord = try LineageTracker.untrainedCopyRecord(
            source: trainedParent, derivation: nil, sourceArchitecture: nil, naming: .unrecorded, pathKind: .gui, argv: ["test"],
            at: Date(timeIntervalSince1970: 1_790_000_050))
        assertRefusedAsTrained(try encodedModel(trainingStep: nil, lineage: copyRecord))
    }

    func testRewriteAfterAnArchitectureOnlyDeriveOfAGraftOfATrainedSourceIsRefused() throws {
        let source = try encodedModel(trainingStep: 4200,
                                      lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let grafted = try graft(source)
        let capped = try derive(grafted, SetRezeroAlphaCapDeriveOperation(value: 1, groupIndices: nil))
        assertRefusedAsTrained(capped.data)
    }

    func testRewriteOnAGraftOfAFreshMintStaysAllowed() throws {
        let mint = try LineageTracker.mintRecord(
            pathKind: .newModel, argv: ["test"], initialization: .forTests, naming: .unnamedWithoutPreset,
            at: Date(timeIntervalSince1970: 1_790_000_000))
        let grafted = try graft(try encodedModel(trainingStep: 0, lineage: mint))
        let result = try derive(grafted, rewrite)
        XCTAssertTrue(result.targetArchitecture.blockGroups.allSatisfy { $0.rezeroAlphaInit == 0 })
    }
}
