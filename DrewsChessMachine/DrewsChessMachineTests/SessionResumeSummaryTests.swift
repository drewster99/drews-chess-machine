//
//  SessionResumeSummaryTests.swift
//  DrewsChessMachineTests
//
//  Tests the lightweight `SessionResumeSummary` projection of
//  `SessionCheckpointState` and the `CheckpointManager.peekSessionMetadata`
//  fast-path loader the auto-resume sheet uses to render rich
//  context (training counters, arena/promotion totals, build
//  version) without paying the full model-load cost.
//

import XCTest
@testable import DrewsChessMachine

final class SessionResumeSummaryTests: XCTestCase {

    private var tmpDir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "SessionResumeSummaryTests-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let dir = tmpDir, FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
        try super.tearDownWithError()
    }

    // MARK: - State derivation

    func testSummaryPullsAllFieldsFromState() throws {
        let state = try makeState(
            arenaHistoryJSON: """
            [
              \(arenaEntryJSON(promoted: true)),
              \(arenaEntryJSON(promoted: false)),
              \(arenaEntryJSON(promoted: true))
            ]
            """,
            extraFields: """
            ,"replayBufferTotalPositionsAdded": 9999000,
            "buildNumber": 420,
            "buildGitHash": "abc1234",
            "buildGitDirty": false,
            "buildTimestamp": "2026-04-30T17:36:30-0500"
            """
        )
        let summary = SessionResumeSummary(state: state)
        XCTAssertEqual(summary.sessionID, "TEST-SESSION")
        XCTAssertEqual(summary.sessionStartUnix, 1699996400)
        XCTAssertEqual(summary.savedAtUnix, 1700000000)
        XCTAssertEqual(summary.elapsedTrainingSec, 3600)
        XCTAssertEqual(summary.trainingSteps, 12345)
        XCTAssertEqual(summary.trainingPositionsSeen, 12641280)
        XCTAssertEqual(summary.selfPlayGames, 678)
        XCTAssertEqual(summary.selfPlayMoves, 45678)
        XCTAssertEqual(summary.replayBufferTotalPositionsAdded, 9999000)
        XCTAssertEqual(summary.arenaCount, 3)
        XCTAssertEqual(summary.promotionCount, 2,
            "two of three arena entries have promoted=true")
        XCTAssertEqual(summary.buildNumber, 420)
        XCTAssertEqual(summary.buildGitHash, "abc1234")
        XCTAssertEqual(summary.buildGitDirty, false)
        XCTAssertEqual(summary.buildTimestamp, "2026-04-30T17:36:30-0500")
    }

    func testSummaryHandlesEmptyArenaHistory() throws {
        let state = try makeState(arenaHistoryJSON: "[]", extraFields: "")
        let summary = SessionResumeSummary(state: state)
        XCTAssertEqual(summary.arenaCount, 0)
        XCTAssertEqual(summary.promotionCount, 0)
    }

    func testSummaryHandlesMissingBuildInfoOnLegacySessions() throws {
        // Older session.json files predate the build-metadata fields;
        // the summary must surface those as nil rather than crashing.
        let state = try makeState(arenaHistoryJSON: "[]", extraFields: "")
        let summary = SessionResumeSummary(state: state)
        XCTAssertNil(summary.buildNumber)
        XCTAssertNil(summary.buildGitHash)
        XCTAssertNil(summary.buildGitDirty)
        XCTAssertNil(summary.buildTimestamp)
        XCTAssertNil(summary.replayBufferTotalPositionsAdded)
    }

    // MARK: - Disk peek

    func testPeekReadsSessionJsonFromDirectory() throws {
        let state = try makeState(
            arenaHistoryJSON: "[\(arenaEntryJSON(promoted: true))]",
            extraFields: """
            ,"buildNumber": 421,
            "buildGitHash": "deadbee"
            """
        )
        let sessionDir = tmpDir.appendingPathComponent("test.dcmsession", isDirectory: true)
        try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true)
        try state.encode().write(to: SessionCheckpointLayout.stateURL(in: sessionDir))

        let summary = try CheckpointManager.peekSessionMetadata(at: sessionDir)
        XCTAssertEqual(summary.sessionID, "TEST-SESSION")
        XCTAssertEqual(summary.arenaCount, 1)
        XCTAssertEqual(summary.promotionCount, 1)
        XCTAssertEqual(summary.buildNumber, 421)
        XCTAssertEqual(summary.buildGitHash, "deadbee")
    }

    func testPeekThrowsMissingSessionJSONForEmptyDirectory() throws {
        let sessionDir = tmpDir.appendingPathComponent("empty.dcmsession", isDirectory: true)
        try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true)
        XCTAssertThrowsError(try CheckpointManager.peekSessionMetadata(at: sessionDir)) { error in
            guard case SessionCheckpointError.missingSessionJSON = error else {
                XCTFail("expected .missingSessionJSON, got \(error)")
                return
            }
        }
    }

    func testPeekThrowsInvalidJSONForGarbageContents() throws {
        let sessionDir = tmpDir.appendingPathComponent("bad.dcmsession", isDirectory: true)
        try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true)
        try Data("{ this is not json".utf8).write(
            to: SessionCheckpointLayout.stateURL(in: sessionDir)
        )
        XCTAssertThrowsError(try CheckpointManager.peekSessionMetadata(at: sessionDir)) { error in
            guard case SessionCheckpointError.invalidJSON = error else {
                XCTFail("expected .invalidJSON, got \(error)")
                return
            }
        }
    }

    // MARK: - Fixtures

    /// Build a `SessionCheckpointState` via JSON-decode (same idiom
    /// as `CheckpointManagerRoundTripTests.makeState`) so the
    /// fixture stays insensitive to unrelated schema growth — the
    /// decoder fills any new Optional field with nil.
    private func makeState(
        arenaHistoryJSON: String,
        extraFields: String
    ) throws -> SessionCheckpointState {
        let formatVersion = SessionCheckpointState.currentFormatVersion
        let jsonText = """
        {
          "formatVersion": \(formatVersion),
          "sessionID": "TEST-SESSION",
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
          "championID": "CHAMP",
          "trainerID": "TRAIN",
          "arenaHistory": \(arenaHistoryJSON)
          \(extraFields)
        }
        """
        let data = Data(jsonText.utf8)
        return try SessionCheckpointState.decode(data)
    }

    private func arenaEntryJSON(promoted: Bool) -> String {
        return """
        {
          "finishedAtStep": 1000,
          "candidateWins": 60,
          "championWins": 40,
          "draws": 100,
          "score": 0.55,
          "promoted": \(promoted),
          "durationSec": 90.0
        }
        """
    }
}
