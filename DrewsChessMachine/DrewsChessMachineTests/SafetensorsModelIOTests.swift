//
//  SafetensorsModelIOTests.swift
//  DrewsChessMachineTests
//
//  Round-trips ModelCheckpointFile <-> safetensors via SafetensorsModelIO,
//  using the real current-architecture weight plan so the name<->weight
//  mapping and metadata are exercised end to end.
//

import XCTest
@testable import DrewsChessMachine

final class SafetensorsModelIOTests: XCTestCase {

    private func syntheticWeights(_ names: [String], arch: NetworkArchitecture, velocity: Bool) -> [[Float]] {
        let plan = arch.weightTensorPlan()
        let trainables = plan.filter { $0.kind != .bnRunningStat }
        var out: [[Float]] = []
        for (i, spec) in plan.enumerated() {
            out.append((0..<spec.elementCount).map { Float(i * 1000 + $0) })
        }
        if velocity {
            for (j, spec) in trainables.enumerated() {
                out.append((0..<spec.elementCount).map { Float(900000 + j * 1000 + $0) })
            }
        }
        _ = names
        return out
    }

    func testBaseFileRoundTrip() throws {
        let arch = NetworkArchitecture.current
        let names = SafetensorsModelIO.tensorNames(for: arch, includesVelocity: false)
        XCTAssertEqual(names.count, arch.weightTensorPlan().count) // 100 for 5-block
        let weights = syntheticWeights(names, arch: arch, velocity: false)

        let meta = ModelCheckpointMetadata(creator: "manual", trainingStep: 265_804,
                                            parentModelID: "20260601-11-bzw3-26", notes: "rt test")
        let data = try SafetensorsModelIO.encode(
            modelID: "20260601-11-bzw3-25", createdAtUnix: 1_780_000_000,
            metadata: meta, weights: weights, architecture: arch, includesVelocity: false
        )
        let decoded = try SafetensorsModelIO.decode(data)

        XCTAssertEqual(decoded.file.modelID, "20260601-11-bzw3-25")
        XCTAssertEqual(decoded.file.createdAtUnix, 1_780_000_000)
        XCTAssertEqual(decoded.file.metadata, meta)
        XCTAssertEqual(decoded.architecture, arch)
        XCTAssertFalse(decoded.hasVelocity)
        XCTAssertEqual(decoded.file.weights.count, weights.count)
        for (a, b) in zip(weights, decoded.file.weights) {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern))
        }
    }

    func testTrainerFileRoundTripWithVelocity() throws {
        let arch = NetworkArchitecture.current
        let names = SafetensorsModelIO.tensorNames(for: arch, includesVelocity: true)
        let plan = arch.weightTensorPlan()
        let trainableCount = plan.filter { $0.kind != .bnRunningStat }.count
        XCTAssertEqual(names.count, plan.count + trainableCount)

        let weights = syntheticWeights(names, arch: arch, velocity: true)
        let meta = ModelCheckpointMetadata(creator: "promote", trainingStep: nil,
                                           parentModelID: "", notes: "")
        let data = try SafetensorsModelIO.encode(
            modelID: "20260601-11-bzw3-27", createdAtUnix: 1_780_000_001,
            metadata: meta, weights: weights, architecture: arch, includesVelocity: true
        )
        let decoded = try SafetensorsModelIO.decode(data)
        XCTAssertTrue(decoded.hasVelocity)
        XCTAssertNil(decoded.file.metadata.trainingStep)
        XCTAssertEqual(decoded.file.weights.count, weights.count)
        for (a, b) in zip(weights, decoded.file.weights) {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern))
        }
    }

    func testOnDiskLayoutIsPyTorchOriented() throws {
        let arch = NetworkArchitecture.current
        let plan = arch.weightTensorPlan()
        // value[k] = k per tensor, so a transpose is detectable by value.
        let weights = plan.map { spec in (0..<spec.elementCount).map { Float($0) } }
        let data = try SafetensorsModelIO.encode(
            modelID: "x", createdAtUnix: 0,
            metadata: ModelCheckpointMetadata(creator: "manual", trainingStep: nil, parentModelID: "", notes: ""),
            weights: weights, architecture: arch, includesVelocity: false
        )
        let (tensors, _) = try SafetensorsFile.decode(data)
        var shapeByName: [String: [Int]] = [:]
        for t in tensors { shapeByName[t.name] = t.shape }

        let flatten = arch.boardSize * arch.boardSize * arch.valueHeadConvChannels
        let seReduced = arch.channels / arch.seReductionRatio

        // FC weights transposed to torch [out, in]:
        XCTAssertEqual(shapeByName["value.fc1.weight"], [arch.valueHeadHiddenUnits, flatten])
        XCTAssertEqual(shapeByName["value.wdl_fc2.weight"], [arch.valueHeadClasses, arch.valueHeadHiddenUnits])
        XCTAssertEqual(shapeByName["blocks.0.se_scalebias.fc1.weight"], [seReduced, arch.channels])
        XCTAssertEqual(shapeByName["blocks.0.se_scalebias.fc2.weight"], [2 * arch.channels, seReduced])
        // Biases 1-D:
        XCTAssertEqual(shapeByName["value.wdl_fc2.bias"], [arch.valueHeadClasses])
        XCTAssertEqual(shapeByName["value.fc1.bias"], [arch.valueHeadHiddenUnits])
        XCTAssertEqual(shapeByName["policy.conv.bias"], [arch.policyChannels])
        // Conv OIHW unchanged; BN [C]; scalar [1]:
        XCTAssertEqual(shapeByName["policy.conv.weight"], [arch.policyChannels, arch.channels, 1, 1])
        XCTAssertEqual(shapeByName["stem.conv.weight"], [arch.channels, arch.inputPlanes, arch.towerConvKernelSize, arch.towerConvKernelSize])
        XCTAssertEqual(shapeByName["stem.bn.weight"], [arch.channels])
        XCTAssertEqual(shapeByName["blocks.0.rezero_alpha"], [1])

        // Verify the actual data transpose (not just the shape label).
        // value.wdl_fc2.weight native [in=hidden=128, out=classes=3], value[k]=k.
        // torch [3,128]: torch[c*128 + r] == native[r*3 + c].
        let fc2idx = try XCTUnwrap(plan.firstIndex { $0.name == "value.wdl_fc2.weight" })
        let nativeFC2 = weights[fc2idx]
        let torchFC2 = try XCTUnwrap(tensors.first { $0.name == "value.wdl_fc2.weight" }).data
        let H = arch.valueHeadHiddenUnits
        let C = arch.valueHeadClasses
        XCTAssertEqual(torchFC2[1 * H + 5], nativeFC2[5 * C + 1])
        XCTAssertEqual(torchFC2[2 * H + 100], nativeFC2[100 * C + 2])

        // And it still round-trips back to native bit-exact.
        let decoded = try SafetensorsModelIO.decode(data)
        XCTAssertEqual(decoded.file.weights.count, weights.count)
        for (a, b) in zip(weights, decoded.file.weights) {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern))
        }
    }

    func testDecodeRejectsArchHashInconsistentWithEmbeddedConfig() throws {
        // A file whose stored arch_hash disagrees with its embedded architecture
        // (hand-edited / tampered) must be rejected up front, not surface later
        // as a weight-shape error. The guard fires before weight reconstruction,
        // so empty tensors suffice.
        let archJSON = String(decoding: try JSONEncoder().encode(NetworkArchitecture.current), as: UTF8.self)
        let bytes = try SafetensorsFile.encode(
            tensors: [],
            metadata: ["architecture": archJSON, "arch_hash": "0xdeadbeef", "dcm_format_version": "3"]
        )
        XCTAssertThrowsError(try SafetensorsModelIO.decode(bytes)) { error in
            guard case SafetensorsModelIO.IOError.archHashMismatch = error else {
                return XCTFail("expected .archHashMismatch, got \(error)")
            }
        }
    }

    func testWrongWeightCountThrows() {
        let arch = NetworkArchitecture.current
        XCTAssertThrowsError(try SafetensorsModelIO.encode(
            modelID: "x", createdAtUnix: 0,
            metadata: ModelCheckpointMetadata(creator: "manual", trainingStep: nil, parentModelID: "", notes: ""),
            weights: [[1, 2, 3]], architecture: arch, includesVelocity: false
        ))
    }
}
