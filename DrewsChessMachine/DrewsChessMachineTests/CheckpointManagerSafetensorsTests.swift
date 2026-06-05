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

    private func minimalState(sessionID: String, championID: String, trainerID: String) throws -> SessionCheckpointState {
        let json = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "sessionID": "\(sessionID)", "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400, "elapsedTrainingSec": 3600,
          "trainingSteps": 12345, "selfPlayGames": 678, "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280, "batchSize": 1024, "learningRate": 5.0e-5,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "championID": "\(championID)", "trainerID": "\(trainerID)", "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(json.utf8))
    }

    /// Full session round-trip through the real saveSession/loadSession path,
    /// exercising the trainer file's optimizer-velocity tensors end to end.
    func testSaveSessionTrainerVelocityRoundTrips() async throws {
        let net = try ChessMPSNetwork(.randomWeights)
        let base = try await net.network.exportWeights()

        // Synthesize velocity: one tensor per trainable (trainable order),
        // sized to each trainable's element count. verifyModelFile only
        // forward-checks the base prefix, so synthetic velocity is fine here.
        let trainables = NetworkArchitecture.current.weightTensorPlan().filter { $0.kind != .bnRunningStat }
        let velocity: [[Float]] = trainables.enumerated().map { (j, spec) in
            (0..<spec.elementCount).map { Float(700000 + j * 131 + $0) }
        }
        let trainerWeights = base + velocity

        let cMeta = ModelCheckpointMetadata(creator: "manual", trainingStep: 12345, parentModelID: "", notes: "champ")
        let tMeta = ModelCheckpointMetadata(creator: "manual", trainingStep: 12345, parentModelID: "20260420-1-abcd", notes: "trainer+vel")
        let state = try minimalState(sessionID: "20260420-1-abcd", championID: "20260420-1-abcd", trainerID: "20260420-2-efgh")

        let dir = try await CheckpointManager.saveSession(
            championWeights: base, championID: "20260420-1-abcd",
            championMetadata: cMeta, championCreatedAtUnix: 1_700_000_000,
            trainerWeights: trainerWeights, trainerID: "20260420-2-efgh",
            trainerMetadata: tMeta, trainerCreatedAtUnix: 1_700_000_001,
            state: state, trigger: "unittest"
        )
        defer {
            do { try FileManager.default.removeItem(at: dir) }
            catch { /* best-effort cleanup */ }
        }

        // Inner files are .safetensors now.
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: SessionCheckpointLayout.championURL(in: dir).path))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: SessionCheckpointLayout.trainerURL(in: dir).path))

        let loaded = try CheckpointManager.loadSession(at: dir)
        XCTAssertEqual(loaded.championFile.weights.count, base.count)
        XCTAssertEqual(loaded.trainerFile.weights.count, trainerWeights.count) // base + velocity
        XCTAssertEqual(loaded.trainerFile.modelID, "20260420-2-efgh")
        for (a, b) in zip(base, loaded.championFile.weights) {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern))
        }
        for (a, b) in zip(trainerWeights, loaded.trainerFile.weights) {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern))
        }
    }

    /// Builds a NON-default architecture (half-width, 64 channels) and runs a
    /// forward pass. This is the oracle for the static->instance refactor: any
    /// leftover hardcoded channel literal (128) would mis-shape a layer and
    /// crash the build or the forward pass on a 64-channel net.
    func testNonDefaultArchitectureBuildsAndEvaluates() async throws {
        var arch = NetworkArchitecture.current
        arch.channels = 64
        try arch.validate()

        let net = try ChessMPSNetwork(.randomWeights, arch: arch)
        XCTAssertEqual(net.network.arch.channels, 64)

        let weights = try await net.network.exportWeights()
        XCTAssertEqual(weights.count, arch.weightTensorPlan().count)

        let board = BoardEncoder.encode(.starting)
        try await net.evaluate(board: board) { policyBuf, value in
            XCTAssertEqual(policyBuf.count, arch.policySize)
            XCTAssertTrue(value.isFinite)
        }
    }
}
