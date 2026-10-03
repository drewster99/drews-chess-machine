//
//  GraftDeriveTests.swift
//  DrewsChessMachineTests
//
//  `--derive-model --graft-to` (determinism plan B3 / P8): a graft copies
//  every target tensor whose name (after `--graft-map`) and shape match a
//  source tensor bit-exact, gives every other target tensor the value a
//  fresh mint of the target has under the graft's init seed, lists dropped
//  source tensors, and records all of it in `derivation_history`. These
//  tests pin the copy, the seeded fill and its reproducibility, the
//  inserted-block map, every refusal, the trained-source rule, and that the
//  record stays readable by — and leaves unchanged — the records of the
//  same-layout operations.
//

import XCTest
@testable import DrewsChessMachine

final class GraftDeriveTests: XCTestCase {

    private static let source = NetworkArchitecture.preset(.nt8y_3x3stem)

    private static func withBlocks(_ count: Int) -> NetworkArchitecture {
        var arch = source
        arch.blockGroups[0].count = count
        return arch
    }

    private func encodedSource(trainingStep: Int? = nil, includesVelocity: Bool = false) throws -> Data {
        let arch = Self.source
        let plan = arch.weightTensorPlan()
        var weights = plan.enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.0625 + 0.03125 }
        }
        if includesVelocity {
            weights += plan.filter { $0.kind != .bnRunningStat }.map { [Float](repeating: 0.5, count: $0.elementCount) }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: trainingStep, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261003-1-SRCE", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: includesVelocity,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
    }

    private func graft(_ data: Data, onto fresh: GraftFreshTarget, map: GraftMap = .empty,
                       id: String = "20261003-2-GRFT") throws -> ModelDerivation.GraftResult {
        try ModelDerivation.graft(
            sourceData: data, sourceName: "source.safetensors", fresh: fresh, targetLabel: "test target",
            map: map, initSeedOrigin: "entered", newModelID: id, createdAtUnix: 1_790_000_100,
            build: "test", invocationArguments: ["test"])
    }

    private func tensorsByName(_ data: Data) throws -> [String: SafetensorsTensor] {
        let (tensors, _) = try SafetensorsFile.decode(data)
        var byName: [String: SafetensorsTensor] = [:]
        for tensor in tensors { byName[tensor.name] = tensor }
        return byName
    }

    private func assertBitEqual(_ a: SafetensorsTensor?, _ b: SafetensorsTensor?, _ message: String,
                                file: StaticString = #filePath, line: UInt = #line) {
        guard let a, let b else { return XCTFail("missing tensor: \(message)", file: file, line: line) }
        XCTAssertEqual(a.shape, b.shape, message, file: file, line: line)
        XCTAssertTrue(a.data.count == b.data.count && zip(a.data, b.data).allSatisfy { $0.bitPattern == $1.bitPattern },
                      "values differ: \(message)", file: file, line: line)
    }

    func testGraftCopiesSharedTensorsAndDrawsTheNewBlockFromTheSeed() throws {
        let sourceData = try encodedSource()
        let fresh = try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 1234)
        let result = try graft(sourceData, onto: fresh)

        let source = try tensorsByName(sourceData)
        let output = try tensorsByName(result.data)
        let freshByName = Dictionary(uniqueKeysWithValues: fresh.tensors.map { ($0.name, $0) })

        XCTAssertEqual(Set(result.copied), Set(source.keys), "every source tensor has a same-name, same-shape target")
        for name in result.copied { assertBitEqual(output[name], source[name], "copied \(name)") }
        XCTAssertFalse(result.initialized.isEmpty)
        XCTAssertTrue(result.initialized.allSatisfy { $0.hasPrefix("blocks.3.") }, "\(result.initialized)")
        for name in result.initialized { assertBitEqual(output[name], freshByName[name], "initialized \(name)") }
        XCTAssertTrue(result.dropped.isEmpty)

        // New BN layers keep the builder's identity running statistics.
        for (name, tensor) in output where name.hasPrefix("blocks.3.") && name.hasSuffix("running_mean") {
            XCTAssertTrue(tensor.data.allSatisfy { $0 == 0 }, name)
        }
        for (name, tensor) in output where name.hasPrefix("blocks.3.") && name.hasSuffix("running_var") {
            XCTAssertTrue(tensor.data.allSatisfy { $0 == 1 }, name)
        }

        let decoded = try SafetensorsModelIO.decode(result.data, valueHead: .asStored, source: "graft")
        XCTAssertEqual(decoded.architecture, Self.withBlocks(4))
    }

    func testTheSameSeedGivesTheSameGraftAndTheInitializedValuesMatchAFreshMint() throws {
        let sourceData = try encodedSource()
        let first = try graft(sourceData, onto: try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 77))
        let second = try graft(sourceData, onto: try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 77))
        let other = try graft(sourceData, onto: try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 78))
        let a = try tensorsByName(first.data)
        let b = try tensorsByName(second.data)
        let c = try tensorsByName(other.data)
        for name in first.initialized { assertBitEqual(a[name], b[name], "same seed \(name)") }
        let drawn = first.initialized.first { $0.hasSuffix("conv1.weight") }
        let drawnName = try XCTUnwrap(drawn, "a drawn conv in the new block")
        XCTAssertFalse(zip(try XCTUnwrap(a[drawnName]).data, try XCTUnwrap(c[drawnName]).data)
            .allSatisfy { $0.bitPattern == $1.bitPattern }, "a different seed draws different values")

        // The drawn tensor is the one a seeded mint of the target holds.
        let mint = try ChessNetwork(arch: Self.withBlocks(4), bnMode: .inference, initialization: .seeded(initSeed: 77))
        let mintWeights = try mint.exportWeightsBlocking()
        let plan = Self.withBlocks(4).weightTensorPlan()
        let index = try XCTUnwrap(plan.firstIndex { $0.name == drawnName })
        let onDisk = SafetensorsModelIO.toTorchLayout(kind: plan[index].kind, nativeShape: plan[index].shape,
                                                      data: mintWeights[index])
        assertBitEqual(a[drawnName], SafetensorsTensor(name: drawnName, shape: onDisk.shape, data: onDisk.data), "vs mint")
    }

    func testAGraftMapMovesBlocksAroundAnInsertedBlock() throws {
        let sourceData = try encodedSource()
        let fresh = try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 5)
        let map = try GraftMap.parse("blocks.2.=blocks.3.,blocks.1.=blocks.2.")
        let result = try graft(sourceData, onto: fresh, map: map)
        let source = try tensorsByName(sourceData)
        let output = try tensorsByName(result.data)

        for (name, tensor) in source where name.hasPrefix("blocks.1.") {
            assertBitEqual(output["blocks.2." + name.dropFirst("blocks.1.".count)], tensor, "moved \(name)")
        }
        for (name, tensor) in source where name.hasPrefix("blocks.2.") {
            assertBitEqual(output["blocks.3." + name.dropFirst("blocks.2.".count)], tensor, "moved \(name)")
        }
        for (name, tensor) in source where name.hasPrefix("blocks.0.") {
            assertBitEqual(output[name], tensor, "kept \(name)")
        }
        XCTAssertTrue(result.initialized.allSatisfy { $0.hasPrefix("blocks.1.") }, "\(result.initialized)")
        let operation = try XCTUnwrap(result.record.operations.first)
        XCTAssertEqual(operation.arguments["graft_map"], "blocks.2.=blocks.3.,blocks.1.=blocks.2.")
        XCTAssertTrue(try XCTUnwrap(operation.copiedTensors).contains { $0.hasPrefix("blocks.1.") && $0.contains(" -> blocks.2.") })
    }

    func testAGraftRefusesASourceCarryingOptimizerState() throws {
        let trainerFile = try encodedSource(trainingStep: 10, includesVelocity: true)
        let fresh = try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 1)
        XCTAssertThrowsError(try graft(trainerFile, onto: fresh)) { error in
            guard case ModelDerivation.DeriveError.sourceHasOptimizerState = error else {
                return XCTFail("expected sourceHasOptimizerState, got \(error)")
            }
        }
    }

    func testASameNamedTensorOfADifferentShapeIsRefusedUnlessDroppedExplicitly() throws {
        let sourceData = try encodedSource()
        var wider = Self.source
        wider.policyPreConvChannels = 256
        let fresh = try GraftFreshTarget.build(architecture: wider, initSeed: 9)
        XCTAssertThrowsError(try graft(sourceData, onto: fresh)) { error in
            guard case ModelDerivation.GraftError.sameNameShapeMismatch = error else {
                return XCTFail("expected sameNameShapeMismatch, got \(error)")
            }
            XCTAssertTrue(String(describing: error).contains("--graft-map"), "\(error)")
        }

        let source = try tensorsByName(sourceData)
        let freshByName = Dictionary(uniqueKeysWithValues: fresh.tensors.map { ($0.name, $0) })
        let mismatched = source.keys.filter { name in
            guard let target = freshByName[name] else { return false }
            return target.shape != source[name]?.shape
        }.sorted()
        XCTAssertFalse(mismatched.isEmpty)
        let map = try GraftMap.parse(mismatched.map { "\($0)=" }.joined(separator: ","))
        let result = try graft(sourceData, onto: fresh, map: map)
        XCTAssertEqual(Set(result.dropped), Set(mismatched))
        for name in mismatched { XCTAssertTrue(result.initialized.contains(name), name) }
    }

    func testGraftMapEntriesMustNameRealTensorsOfTheSameShape() throws {
        let sourceData = try encodedSource()
        let fresh = try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 3)
        XCTAssertThrowsError(try graft(sourceData, onto: fresh, map: try GraftMap.parse("no.such.tensor=blocks.3.conv1.weight"))) {
            guard case ModelDerivation.GraftError.mapNamesUnknownSourceTensor = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertThrowsError(try graft(sourceData, onto: fresh, map: try GraftMap.parse("blocks.0.=blocks.9."))) {
            guard case ModelDerivation.GraftError.mapNamesUnknownTargetTensor = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertThrowsError(try graft(sourceData, onto: fresh,
                                       map: try GraftMap.parse("blocks.0.rezero_alpha=blocks.3.conv1.weight"))) {
            guard case ModelDerivation.GraftError.mapShapeMismatch = $0 else { return XCTFail("\($0)") }
        }
        // Two source tensors routed to one target slot.
        XCTAssertThrowsError(try graft(sourceData, onto: fresh, map: try GraftMap.parse("blocks.1.=blocks.0."))) {
            guard case ModelDerivation.GraftError.twoSourcesForOneTarget = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertThrowsError(try GraftMap.parse("blocks.1.=blocks.2"))
        XCTAssertThrowsError(try GraftMap.parse("=x"))
        XCTAssertThrowsError(try GraftMap.parse("a=b,a=c"))
    }

    func testATrainedSourceIsGraftedAndItsStepIsRecordedNotCarried() throws {
        let trained = try encodedSource(trainingStep: 4200)
        let fresh = try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 11)
        let result = try graft(trained, onto: fresh)
        let (_, metadata) = try SafetensorsFile.decode(result.data)
        XCTAssertNil(metadata[SafetensorsModelIO.Key.trainingStep], "a graft never claims a training step")
        let operation = try XCTUnwrap(result.record.operations.first)
        XCTAssertEqual(operation.arguments["source_training_step"], "4200")
    }

    func testTheGraftRecordIsCompleteAndRoundTripsThroughTheLineage() throws {
        let fresh = try GraftFreshTarget.build(architecture: Self.withBlocks(4), initSeed: 1234)
        let result = try graft(try encodedSource(), onto: fresh)
        let operation = try XCTUnwrap(result.record.operations.first)
        XCTAssertEqual(operation.operation, ModelDerivation.graftOperationName)
        XCTAssertEqual(operation.changedArchitectureFields, ["*"])
        XCTAssertEqual(operation.rewrittenTensors, result.initialized)
        XCTAssertEqual(operation.initSeed, "1234")
        XCTAssertEqual(operation.initRuleVersion, WeightInitScheme.current)
        XCTAssertEqual(try XCTUnwrap(operation.droppedTensors), [])
        let perTensor = try XCTUnwrap(operation.perTensorInit)
        XCTAssertEqual(Set(perTensor.keys), Set(result.initialized))
        XCTAssertTrue(perTensor.values.contains(RandomTensorRole.heNormal.rawValue))
        XCTAssertTrue(perTensor.values.contains(ModelDerivation.builderConstantInit))
        XCTAssertEqual(result.lineage.derivationHistory.last, result.record)

        let decoded = try SafetensorsModelIO.decode(result.data, valueHead: .asStored, source: "graft")
        guard case .recorded(let record) = try XCTUnwrap(decoded.file.safetensorsProvenance).lineage else {
            return XCTFail("the grafted file must carry a lineage record")
        }
        XCTAssertEqual(record.derivationHistory.last, result.record)
    }

    func testSameLayoutOperationRecordsEncodeWithoutTheGraftFields() throws {
        let record = ModelDerivation.OperationRecord(
            operation: "set-activation", arguments: ["value": "leaky_relu"],
            changedArchitectureFields: ["activation_function"], rewrittenTensors: [])
        let json = String(decoding: try JSONEncoder().encode(record), as: UTF8.self)
        for key in ["copied_tensors", "dropped_tensors", "init_seed", "init_rule_version", "per_tensor_init"] {
            XCTAssertFalse(json.contains(key), "\(key) in \(json)")
        }
        let legacy = #"{"operation":"set-se-beta-init","arguments":{"value":"zero"},"changed_architecture_fields":["x"],"rewritten_tensors":["t"]}"#
        let decoded = try JSONDecoder().decode(ModelDerivation.OperationRecord.self, from: Data(legacy.utf8))
        XCTAssertNil(decoded.copiedTensors)
        XCTAssertEqual(decoded.rewrittenTensors, ["t"])
    }

    func testTheDeriveHelpDescribesTheGraft() {
        XCTAssertTrue(DeriveModelCLI.helpText.contains(DeriveModelCLI.graftToFlag))
        XCTAssertTrue(DeriveModelCLI.helpText.contains(DeriveModelCLI.graftMapFlag))
    }
}
