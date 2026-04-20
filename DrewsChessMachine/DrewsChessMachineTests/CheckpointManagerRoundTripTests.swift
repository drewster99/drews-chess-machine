//
//  CheckpointManagerRoundTripTests.swift
//  DrewsChessMachineTests
//
//  File-format round-trip tests for the `.dcmsession` read path.
//
//  These tests stay at the serialization layer — they build
//  `ModelCheckpointFile` and `SessionCheckpointState` structs
//  directly, encode them, write them to a tmp directory using
//  `SessionCheckpointLayout`, and then call
//  `CheckpointManager.loadSession(at:)` to read them back. This
//  deliberately avoids `CheckpointManager.saveSession` because
//  that path runs a forward-pass verification through MPSGraph
//  (for bit-exact weight round-tripping) and would pollute the
//  real `~/Library/Application Support/DrewsChessMachine/Sessions`
//  folder with test artifacts.
//
//  The auto-resume flow's critical property — "a pointer whose
//  target folder still parses as a `.dcmsession` produces a
//  `LoadedSession` identical to what went in" — is what we want
//  the test to guarantee.
//

import XCTest
@testable import DrewsChessMachine

final class CheckpointManagerRoundTripTests: XCTestCase {

    private var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-checkpoint-test-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tmpDir, FileManager.default.fileExists(atPath: tmpDir.path) {
            try FileManager.default.removeItem(at: tmpDir)
        }
    }

    // MARK: - Fixtures

    /// A minimal-but-valid `SessionCheckpointState`, built by
    /// decoding a JSON string that supplies only the non-Optional
    /// fields. The synthesized memberwise init demands all ~45
    /// fields (Optional `var`s don't get `= nil` defaults in
    /// Swift), so threading everything manually would make every
    /// test a 45-line literal and break whenever the struct grows
    /// a new field. JSON decoding sidesteps both problems: the
    /// decoder fills Optional fields with nil when the key is
    /// absent, and the test stays insensitive to unrelated field
    /// additions.
    private func makeState(
        sessionID: String,
        championID: String,
        trainerID: String
    ) throws -> SessionCheckpointState {
        let formatVersion = SessionCheckpointState.currentFormatVersion
        let jsonText = """
        {
          "formatVersion": \(formatVersion),
          "sessionID": "\(sessionID)",
          "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400,
          "elapsedTrainingSec": 3600,
          "trainingSteps": 12345,
          "selfPlayGames": 678,
          "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280,
          "batchSize": 1024,
          "learningRate": 5.0e-5,
          "promoteThreshold": 0.55,
          "arenaGames": 200,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.05, "floorTau": 0.4},
          "arenaTau": {"startTau": 1.0, "decayPerPly": 0.05, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "championID": "\(championID)",
          "trainerID": "\(trainerID)",
          "arenaHistory": []
        }
        """
        let data = Data(jsonText.utf8)
        return try SessionCheckpointState.decode(data)
    }

    /// Build a minimal `.dcmmodel` blob. `ChessNetwork.loadWeights`
    /// would reject these shapes, but `CheckpointManager.loadSession`
    /// stops at decoding the file — it doesn't load into a live
    /// network — so a small synthetic weight set is sufficient for
    /// testing the file I/O path.
    private func makeModel(
        id: String,
        creator: String,
        weights: [[Float]] = [[0.5, -0.25, 0.125], [1.0]]
    ) -> ModelCheckpointFile {
        let metadata = ModelCheckpointMetadata(
            creator: creator,
            trainingStep: 100,
            parentModelID: "",
            notes: "unit-test fixture"
        )
        return ModelCheckpointFile(
            modelID: id,
            createdAtUnix: 1_700_000_000,
            metadata: metadata,
            weights: weights
        )
    }

    /// Stage a complete `.dcmsession` directory into `tmpDir/<name>`
    /// by encoding the three pieces and writing them with the
    /// canonical filenames. Returns the directory URL.
    private func stageSession(
        name: String,
        state: SessionCheckpointState,
        champion: ModelCheckpointFile,
        trainer: ModelCheckpointFile
    ) throws -> URL {
        let sessionDir = tmpDir.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true)
        let stateURL = SessionCheckpointLayout.stateURL(in: sessionDir)
        let championURL = SessionCheckpointLayout.championURL(in: sessionDir)
        let trainerURL = SessionCheckpointLayout.trainerURL(in: sessionDir)
        try state.encode().write(to: stateURL)
        try champion.encode().write(to: championURL)
        try trainer.encode().write(to: trainerURL)
        return sessionDir
    }

    // MARK: - Round-trip

    func testLoadSessionReproducesState() throws {
        let state = try makeState(
            sessionID: "20260420-1-abcd",
            championID: "20260420-1-abcd",
            trainerID: "20260420-2-efgh"
        )
        let champion = makeModel(id: "20260420-1-abcd", creator: "manual")
        let trainer = makeModel(id: "20260420-2-efgh", creator: "manual")

        let sessionDir = try stageSession(
            name: "20260420-103214-20260420-1-abcd-manual.dcmsession",
            state: state,
            champion: champion,
            trainer: trainer
        )

        let loaded = try CheckpointManager.loadSession(at: sessionDir)

        XCTAssertEqual(loaded.state, state)
        XCTAssertEqual(loaded.championFile.modelID, champion.modelID)
        XCTAssertEqual(loaded.championFile.createdAtUnix, champion.createdAtUnix)
        XCTAssertEqual(loaded.championFile.metadata, champion.metadata)
        XCTAssertEqual(loaded.championFile.weights, champion.weights)
        XCTAssertEqual(loaded.trainerFile.modelID, trainer.modelID)
        XCTAssertEqual(loaded.trainerFile.createdAtUnix, trainer.createdAtUnix)
        XCTAssertEqual(loaded.trainerFile.metadata, trainer.metadata)
        XCTAssertEqual(loaded.trainerFile.weights, trainer.weights)
        XCTAssertNil(loaded.replayBufferURL)
    }

    func testLoadSessionWeightsAreBitExact() throws {
        // The binary model format stores weights as little-endian
        // Float32 blobs with a trailing SHA-256. Bit-exact here is
        // the real invariant — a rounding bug in encode/decode
        // would surface as `!=` on the `[[Float]]` comparison since
        // `Float ==` is bitwise on finite values.
        let weights: [[Float]] = [
            [0.0, -0.0, 1.0, -1.0, .leastNormalMagnitude, .greatestFiniteMagnitude],
            [.pi, -.pi, .ulpOfOne]
        ]
        let state = try makeState(
            sessionID: "bits", championID: "bits", trainerID: "bits-T"
        )
        let champion = makeModel(id: "bits", creator: "manual", weights: weights)
        let trainer = makeModel(id: "bits-T", creator: "manual", weights: weights)
        let dir = try stageSession(
            name: "bits.dcmsession",
            state: state, champion: champion, trainer: trainer
        )
        let loaded = try CheckpointManager.loadSession(at: dir)
        XCTAssertEqual(loaded.championFile.weights.count, weights.count)
        for (i, (want, got)) in zip(weights, loaded.championFile.weights).enumerated() {
            XCTAssertEqual(want.count, got.count, "tensor \(i) element count")
            for j in 0..<want.count {
                XCTAssertEqual(
                    want[j].bitPattern, got[j].bitPattern,
                    "tensor \(i) element \(j) bitPattern"
                )
            }
        }
    }

    func testLoadSessionMissingStateFileThrows() throws {
        let state = try makeState(sessionID: "X", championID: "X", trainerID: "X-T")
        let champion = makeModel(id: "X", creator: "manual")
        let trainer = makeModel(id: "X-T", creator: "manual")
        let dir = try stageSession(
            name: "missing-state.dcmsession",
            state: state, champion: champion, trainer: trainer
        )
        // Delete the session.json to simulate corruption.
        try FileManager.default.removeItem(at: SessionCheckpointLayout.stateURL(in: dir))

        XCTAssertThrowsError(try CheckpointManager.loadSession(at: dir)) { error in
            guard let err = error as? SessionCheckpointError else {
                return XCTFail("Expected SessionCheckpointError, got \(error)")
            }
            switch err {
            case .missingSessionJSON:
                break
            default:
                XCTFail("Expected .missingSessionJSON, got \(err)")
            }
        }
    }

    func testLoadSessionMissingChampionFileThrows() throws {
        let state = try makeState(sessionID: "X", championID: "X", trainerID: "X-T")
        let champion = makeModel(id: "X", creator: "manual")
        let trainer = makeModel(id: "X-T", creator: "manual")
        let dir = try stageSession(
            name: "missing-champion.dcmsession",
            state: state, champion: champion, trainer: trainer
        )
        try FileManager.default.removeItem(at: SessionCheckpointLayout.championURL(in: dir))

        XCTAssertThrowsError(try CheckpointManager.loadSession(at: dir)) { error in
            guard let err = error as? SessionCheckpointError else {
                return XCTFail("Expected SessionCheckpointError, got \(error)")
            }
            switch err {
            case .missingChampionFile:
                break
            default:
                XCTFail("Expected .missingChampionFile, got \(err)")
            }
        }
    }

    func testLoadSessionMissingTrainerFileThrows() throws {
        let state = try makeState(sessionID: "X", championID: "X", trainerID: "X-T")
        let champion = makeModel(id: "X", creator: "manual")
        let trainer = makeModel(id: "X-T", creator: "manual")
        let dir = try stageSession(
            name: "missing-trainer.dcmsession",
            state: state, champion: champion, trainer: trainer
        )
        try FileManager.default.removeItem(at: SessionCheckpointLayout.trainerURL(in: dir))

        XCTAssertThrowsError(try CheckpointManager.loadSession(at: dir)) { error in
            guard let err = error as? SessionCheckpointError else {
                return XCTFail("Expected SessionCheckpointError, got \(error)")
            }
            switch err {
            case .missingTrainerFile:
                break
            default:
                XCTFail("Expected .missingTrainerFile, got \(err)")
            }
        }
    }

    func testLoadSessionReplayBufferFlagIgnoredWhenFileAbsent() throws {
        // `state.hasReplayBuffer == true` but the file is not on
        // disk: the loader treats the buffer as missing (nil URL)
        // rather than failing, so a manually-copied session that
        // dropped the 5 GB binary still loads.
        var state = try makeState(sessionID: "no-buf", championID: "no-buf", trainerID: "no-buf-T")
        state.hasReplayBuffer = true
        state.replayBufferStoredCount = 42
        state.replayBufferCapacity = 1000
        state.replayBufferTotalPositionsAdded = 42
        let champion = makeModel(id: "no-buf", creator: "manual")
        let trainer = makeModel(id: "no-buf-T", creator: "manual")
        let dir = try stageSession(
            name: "no-buf.dcmsession",
            state: state, champion: champion, trainer: trainer
        )
        let loaded = try CheckpointManager.loadSession(at: dir)
        XCTAssertNil(loaded.replayBufferURL)
    }

    // MARK: - File sanity

    func testSessionJsonIsValidJSON() throws {
        let state = try makeState(sessionID: "json", championID: "C", trainerID: "T")
        let champion = makeModel(id: "C", creator: "manual")
        let trainer = makeModel(id: "T", creator: "manual")
        let dir = try stageSession(
            name: "json.dcmsession",
            state: state, champion: champion, trainer: trainer
        )
        let data = try Data(contentsOf: SessionCheckpointLayout.stateURL(in: dir))
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: data))
    }
}
