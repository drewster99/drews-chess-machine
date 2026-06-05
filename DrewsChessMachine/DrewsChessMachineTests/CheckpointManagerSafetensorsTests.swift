//
//  CheckpointManagerSafetensorsTests.swift
//  DrewsChessMachineTests
//
//  Exercises the REAL save/load disk path now that model files are
//  safetensors-native: a live network's exported weights → saveModel (which
//  runs the bit-exact forward-pass verifyModelFile gate internally) →
//  loadModelFile → bit-compare. If the safetensors encode/decode or the
//  name<->weight mapping were wrong, saveModel would throw at the verify gate.
//  Metal-backed (builds ChessMPSNetwork), so it lives apart from the pure-logic
//  codec tests.
//

import XCTest
@testable import DrewsChessMachine

final class CheckpointManagerSafetensorsTests: XCTestCase {

    func testSaveModelWritesSafetensorsAndRoundTrips() async throws {
        let net = try ChessMPSNetwork(.randomWeights)
        let weights = try await net.network.exportWeights()
        XCTAssertEqual(weights.count, NetworkArchitecture.current.weightTensorPlan().count)

        let meta = ModelCheckpointMetadata(
            creator: "manual", trainingStep: 42, parentModelID: "", notes: "safetensors unit"
        )
        let url = try await CheckpointManager.saveModel(
            weights: weights,
            modelID: "unittest-st-roundtrip",
            createdAtUnix: 1_780_000_000,
            metadata: meta,
            trigger: "unittest"
        )
        defer {
            do { try FileManager.default.removeItem(at: url) }
            catch { /* best-effort cleanup of the unit-test artifact */ }
        }

        // New saves are .safetensors and load with HuggingFace tooling.
        XCTAssertEqual(url.pathExtension, "safetensors")

        let loaded = try CheckpointManager.loadModelFile(at: url)
        XCTAssertEqual(loaded.modelID, "unittest-st-roundtrip")
        XCTAssertEqual(loaded.createdAtUnix, 1_780_000_000)
        XCTAssertEqual(loaded.metadata, meta)
        XCTAssertEqual(loaded.weights.count, weights.count)
        for (a, b) in zip(weights, loaded.weights) {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern))
        }
    }
}
