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

    func testWrongWeightCountThrows() {
        let arch = NetworkArchitecture.current
        XCTAssertThrowsError(try SafetensorsModelIO.encode(
            modelID: "x", createdAtUnix: 0,
            metadata: ModelCheckpointMetadata(creator: "manual", trainingStep: nil, parentModelID: "", notes: ""),
            weights: [[1, 2, 3]], architecture: arch, includesVelocity: false
        ))
    }
}
