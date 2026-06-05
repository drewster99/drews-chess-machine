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

    /// The trainer-arch wiring: a non-default (8-block) champion's weights must
    /// load into a trainer built to the SAME arch (the fork), and the non-default
    /// model must save (verifyModelFile uses an arch-matched scratch) and reload
    /// bit-exact with the correct embedded architecture. A default-arch trainer
    /// or verify-scratch would throw a shape mismatch here.
    func testNonDefaultArchTrainerForkAndSave() async throws {
        let arch = NetworkArchitecture.preset(.v4_8block_3x3) // 8 blocks 3x3 — not the default
        try arch.validate()

        let champion = try ChessMPSNetwork(.randomWeights, arch: arch)
        let championWeights = try await champion.network.exportWeights()
        XCTAssertEqual(championWeights.count, arch.weightTensorPlan().count) // 145 for 8-block

        // Trainer built to the champion's arch must accept its weights (fork).
        let trainer = try ChessTrainer(arch: arch)
        XCTAssertEqual(trainer.arch, arch)
        try await trainer.network.loadWeights(championWeights) // would throw if arch mismatched

        // Save the non-default champion (verifyModelFile builds an arch-matched
        // scratch internally) and reload bit-exact.
        let url = try await CheckpointManager.saveModel(
            weights: championWeights, modelID: "unittest-nondefault",
            createdAtUnix: 1_780_000_000,
            metadata: ModelCheckpointMetadata(creator: "manual", trainingStep: nil, parentModelID: "", notes: "nd"),
            architecture: arch, trigger: "unittest"
        )
        defer { do { try FileManager.default.removeItem(at: url) } catch {} }

        let loaded = try CheckpointManager.loadModelFile(at: url)
        assertBitEqual(loaded.weights, championWeights, "non-default save/reload")

        // The embedded architecture is the actual (non-default) one, not the default.
        let decoded = try SafetensorsModelIO.decode(try Data(contentsOf: url))
        XCTAssertEqual(decoded.architecture, arch)
        XCTAssertNotEqual(decoded.architecture, .current)
    }

    /// Non-default architecture through the full session save/load path
    /// (champion + trainer-with-velocity), bit-exact, with the correct embedded
    /// arch — the saveSession analog of the trainer-fork test.
    func testNonDefaultArchSessionRoundTrips() async throws {
        let arch = NetworkArchitecture.preset(.v4_8block_3x3)
        try arch.validate()
        let champion = try ChessMPSNetwork(.randomWeights, arch: arch)
        let base = try await champion.network.exportWeights()
        let trainables = arch.weightTensorPlan().filter { $0.kind != .bnRunningStat }
        let velocity: [[Float]] = trainables.enumerated().map { (j, spec) in
            (0..<spec.elementCount).map { Float(640000 + j * 7 + $0) }
        }
        let trainerWeights = base + velocity

        let meta = ModelCheckpointMetadata(creator: "manual", trainingStep: 1, parentModelID: "", notes: "nd-session")
        let state = try minimalState(sessionID: "20260420-9-nd99", championID: "20260420-9-nd99", trainerID: "20260420-9-ndtt")

        let dir = try await CheckpointManager.saveSession(
            championWeights: base, championID: "20260420-9-nd99",
            championMetadata: meta, championCreatedAtUnix: 1_780_000_000,
            trainerWeights: trainerWeights, trainerID: "20260420-9-ndtt",
            trainerMetadata: meta, trainerCreatedAtUnix: 1_780_000_001,
            state: state, architecture: arch, trigger: "unittest-nd"
        )
        defer { do { try FileManager.default.removeItem(at: dir) } catch {} }

        let loaded = try CheckpointManager.loadSession(at: dir)
        assertBitEqual(loaded.championFile.weights, base, "nd session champion")
        assertBitEqual(loaded.trainerFile.weights, trainerWeights, "nd session trainer+velocity")
    }

    private func assertBitEqual(_ a: [[Float]], _ b: [[Float]], _ label: String) {
        XCTAssertEqual(a.count, b.count, "\(label): tensor count")
        for (i, pair) in zip(a, b).enumerated() {
            XCTAssertEqual(pair.0.map(\.bitPattern), pair.1.map(\.bitPattern), "\(label): tensor \(i)")
        }
    }

    /// Cross-format conversion both directions, bit-exact:
    ///   legacy .dcmmodel -> safetensors -> reload, and the reverse.
    /// Validates the "lazy convert on re-save" promise — a model can move
    /// between the old and new containers without losing a bit.
    func testCrossFormatRoundTripBothDirections() async throws {
        let net = try ChessMPSNetwork(.randomWeights)
        let w0 = try await net.network.exportWeights()
        let meta = ModelCheckpointMetadata(creator: "manual", trainingStep: 7,
                                           parentModelID: "", notes: "xfmt")
        let id = "unittest-xfmt"
        let created: Int64 = 1_780_000_000

        // Direction 1: legacy .dcmmodel -> decode -> re-encode safetensors -> decode.
        let legacyData = try ModelCheckpointFile(
            modelID: id, createdAtUnix: created, metadata: meta, weights: w0
        ).encode()
        let fromLegacy = try CheckpointManager.decodeAnyModelFile(legacyData)
        assertBitEqual(fromLegacy.weights, w0, "legacy decode")

        let stData = try SafetensorsModelIO.encode(
            modelID: fromLegacy.modelID, createdAtUnix: fromLegacy.createdAtUnix,
            metadata: fromLegacy.metadata, weights: fromLegacy.weights,
            architecture: .current, includesVelocity: false
        )
        let viaSafetensors = try CheckpointManager.decodeAnyModelFile(stData)
        assertBitEqual(viaSafetensors.weights, w0, "legacy->safetensors->reload")
        XCTAssertEqual(viaSafetensors.modelID, id)
        XCTAssertEqual(viaSafetensors.metadata, meta)
        XCTAssertEqual(viaSafetensors.createdAtUnix, created)

        // Direction 2: safetensors -> decode -> re-encode legacy .dcmmodel -> decode.
        let stData2 = try SafetensorsModelIO.encode(
            modelID: id, createdAtUnix: created, metadata: meta, weights: w0,
            architecture: .current, includesVelocity: false
        )
        let fromST = try CheckpointManager.decodeAnyModelFile(stData2)
        let legacyData2 = try ModelCheckpointFile(
            modelID: fromST.modelID, createdAtUnix: fromST.createdAtUnix,
            metadata: fromST.metadata, weights: fromST.weights
        ).encode()
        let viaLegacy = try CheckpointManager.decodeAnyModelFile(legacyData2)
        assertBitEqual(viaLegacy.weights, w0, "safetensors->legacy->reload")
        XCTAssertEqual(viaLegacy.modelID, id)
        XCTAssertEqual(viaLegacy.metadata, meta)
    }

    /// Session-level cross-format migration: stage a LEGACY .dcmmodel session
    /// (champion + trainer-with-velocity) on disk, load it, re-save via the
    /// real saveSession (which now writes safetensors), reload, and confirm the
    /// champion AND trainer (incl. optimizer velocity) survive bit-exact.
    func testLegacySessionLoadsAndReSavesAsSafetensorsBitExact() async throws {
        let net = try ChessMPSNetwork(.randomWeights)
        let base = try await net.network.exportWeights()
        let trainables = NetworkArchitecture.current.weightTensorPlan().filter { $0.kind != .bnRunningStat }
        let velocity: [[Float]] = trainables.enumerated().map { (j, spec) in
            (0..<spec.elementCount).map { Float(820000 + j * 17 + $0) }
        }
        let trainerWeights = base + velocity

        let cMeta = ModelCheckpointMetadata(creator: "manual", trainingStep: 99, parentModelID: "", notes: "champ")
        let tMeta = ModelCheckpointMetadata(creator: "manual", trainingStep: 99, parentModelID: "20260420-1-abcd", notes: "trainer")
        let state = try minimalState(sessionID: "20260420-1-abcd", championID: "20260420-1-abcd", trainerID: "20260420-2-efgh")

        // --- Stage a LEGACY .dcmmodel session in a temp dir ---
        let legacyDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("legacy_session_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: legacyDir, withIntermediateDirectories: true)
        defer { do { try FileManager.default.removeItem(at: legacyDir) } catch {} }

        try ModelCheckpointFile(modelID: "20260420-1-abcd", createdAtUnix: 1_700_000_000,
                                metadata: cMeta, weights: base).encode()
            .write(to: legacyDir.appendingPathComponent(SessionCheckpointLayout.legacyChampionFilename))
        try ModelCheckpointFile(modelID: "20260420-2-efgh", createdAtUnix: 1_700_000_001,
                                metadata: tMeta, weights: trainerWeights).encode()
            .write(to: legacyDir.appendingPathComponent(SessionCheckpointLayout.legacyTrainerFilename))
        try state.encode().write(to: SessionCheckpointLayout.stateURL(in: legacyDir))

        // --- Load the legacy session ---
        let legacyLoaded = try CheckpointManager.loadSession(at: legacyDir)
        assertBitEqual(legacyLoaded.championFile.weights, base, "legacy session champion")
        assertBitEqual(legacyLoaded.trainerFile.weights, trainerWeights, "legacy session trainer+velocity")

        // --- Re-save via saveSession (writes safetensors) and reload ---
        let newDir = try await CheckpointManager.saveSession(
            championWeights: legacyLoaded.championFile.weights,
            championID: legacyLoaded.championFile.modelID,
            championMetadata: legacyLoaded.championFile.metadata,
            championCreatedAtUnix: legacyLoaded.championFile.createdAtUnix,
            trainerWeights: legacyLoaded.trainerFile.weights,
            trainerID: legacyLoaded.trainerFile.modelID,
            trainerMetadata: legacyLoaded.trainerFile.metadata,
            trainerCreatedAtUnix: legacyLoaded.trainerFile.createdAtUnix,
            state: legacyLoaded.state, trigger: "unittest-migrate"
        )
        defer { do { try FileManager.default.removeItem(at: newDir) } catch {} }

        // Inner files are now safetensors.
        XCTAssertTrue(FileManager.default.fileExists(atPath: SessionCheckpointLayout.championURL(in: newDir).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: SessionCheckpointLayout.trainerURL(in: newDir).path))

        let migrated = try CheckpointManager.loadSession(at: newDir)
        assertBitEqual(migrated.championFile.weights, base, "migrated champion")
        assertBitEqual(migrated.trainerFile.weights, trainerWeights, "migrated trainer+velocity")
    }
}
