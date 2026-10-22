//
//  SessionListingTests.swift
//  DrewsChessMachineTests
//
//  `--list-sessions` (`ListSessionsCLI`) and the folder-name reading it shares
//  with the Load Session picker (`SessionManifest.parseFolderName`,
//  `identity`, `resolve`), on session folders the app saves.
//

import XCTest
@testable import DrewsChessMachine

final class SessionListingTests: XCTestCase {

    private var sessionsDir: URL!

    override func setUpWithError() throws {
        sessionsDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("SessionListingTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: sessionsDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let sessionsDir, FileManager.default.fileExists(atPath: sessionsDir.path) {
            try FileManager.default.removeItem(at: sessionsDir)
        }
    }

    // MARK: - Fixtures

    private func weights(_ arch: NetworkArchitecture) -> [[Float]] {
        arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
    }

    private func state(sessionID: String, savedAtUnix: Int, trainingSteps: Int) throws -> SessionCheckpointState {
        try SessionCheckpointState.decode(Data("""
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "\(sessionID)", "savedAtUnix": \(savedAtUnix),
          "sessionStartUnix": 1699996400, "elapsedTrainingSec": 7200,
          "trainingSteps": \(trainingSteps), "selfPlayGames": 678, "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280, "batchSize": 1024, "learningRate": 5.0e-5,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "championID": "\(sessionID)", "trainerID": "\(sessionID)-1",
          "arenaHistory": []
        }
        """.utf8))
    }

    /// The run record each saved session's trainer file was given, by folder.
    private var savedRecords: [String: LineageRecord] = [:]

    /// A session the app saves into `sessionsDir`, named for `savedAtUnix`.
    private func saveSession(_ sessionID: String, trigger: String, savedAtUnix: Int,
                             trainingSteps: Int = 12345) async throws -> URL {
        let record = try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil)
        let arch = NetworkArchitecture.current
        let champion = weights(arch)
        let velocity = arch.weightTensorPlan().filter { $0.kind != .bnRunningStat }.map {
            [Float](repeating: 0.001, count: $0.elementCount)
        }
        let meta = ModelCheckpointMetadata(creator: "manual", trainingStep: 1, parentModelID: "", notes: "listing")
        let folder = try await CheckpointManager.saveSession(
            championWeights: champion, championID: sessionID, championMetadata: meta,
            championCreatedAtUnix: 1_790_000_000,
            trainerWeights: champion + velocity, trainerID: "\(sessionID)-1", trainerMetadata: meta,
            trainerCreatedAtUnix: 1_790_000_000,
            state: try state(sessionID: sessionID, savedAtUnix: savedAtUnix, trainingSteps: trainingSteps),
            lineage: record,
            championLineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
            trigger: trigger, at: Date(timeIntervalSince1970: TimeInterval(savedAtUnix)),
            sessionsDirectory: sessionsDir)
        savedRecords[folder.lastPathComponent] = record
        return folder
    }

    // MARK: - Folder names

    /// Protects: a save tag holding a hyphen is read whole.
    /// Fails if: the trigger is the last hyphen component again — a
    /// train-vs-UCI save would show trigger `periodic` and lineage `vsuci`.
    func testAHyphenatedTriggerIsReadWhole() {
        let parsed = SessionManifest.parseFolderName("20261003-020000-20261003-9-TeSt-vsuci-periodic.dcmsession")
        XCTAssertEqual(parsed.trigger, "vsuci-periodic")
        XCTAssertEqual(parsed.lineageTag, "TeSt")
        XCTAssertNotNil(parsed.savedAt)
    }

    /// Protects: a hand-renamed folder states no trigger and no lineage tag.
    /// Fails if: words of the new name are read as a trigger and tag.
    func testARenamedFolderStatesNoTriggerOrTag() {
        let parsed = SessionManifest.parseFolderName("old-Ko63-try-to-resume-from-here.dcmsession")
        XCTAssertNil(parsed.trigger)
        XCTAssertNil(parsed.lineageTag)
        XCTAssertNil(parsed.savedAt)
    }

    /// Protects: a folder named for another session ID is a renamed one; it
    /// groups under the session it holds.
    /// Fails if: the name's trigger is trusted without the recorded ID.
    func testANameOfAnotherSessionIsReadAsRenamed() {
        let identity = SessionManifest.identity(
            folderName: "20261003-010000-20261003-1-AAAA-manual.dcmsession", recordedSessionID: "20261003-2-BBBB")
        XCTAssertNil(identity.trigger)
        XCTAssertEqual(identity.lineageTag, "BBBB")
        let own = SessionManifest.identity(
            folderName: "20261003-010000-20261003-1-AAAA-manual.dcmsession", recordedSessionID: "20261003-1-AAAA")
        XCTAssertEqual(own.trigger, "manual")
        XCTAssertEqual(own.lineageTag, "AAAA")
    }

    /// Protects: a saved session's stored `manifest.json` is labeled by the
    /// folder's current name, so the picker opens the folder that is there.
    /// Fails if: `resolve` returns the stored manifest as written — its
    /// folder name (the old one) and trigger would describe a folder that no
    /// longer exists.
    func testAStoredManifestIsRelabeledForARenamedFolder() async throws {
        let saved = try await saveSession("20261021-1-RNMD", trigger: "manual", savedAtUnix: 1_790_000_000)
        XCTAssertEqual(SessionManifest.resolve(sessionFolder: saved).trigger, "manual")
        let renamed = sessionsDir.appendingPathComponent("keep-this-one.dcmsession", isDirectory: true)
        try FileManager.default.moveItem(at: saved, to: renamed)
        let manifest = SessionManifest.resolve(sessionFolder: renamed)
        XCTAssertEqual(manifest.folderName, "keep-this-one.dcmsession")
        XCTAssertNil(manifest.trigger)
        XCTAssertEqual(manifest.lineageTag, "RNMD")
        XCTAssertEqual(manifest.sessionID, "20261021-1-RNMD")
    }

    /// Protects: a manifest stored by a build of another manifest schema is
    /// not trusted; the session is read again from session.json.
    /// Fails if: `resolve` accepts a stored manifest without checking its
    /// schema — a field added since (or a reading changed) would show stale.
    func testAStoredManifestOfAnotherSchemaIsReadAgain() async throws {
        let saved = try await saveSession("20261021-9-SCHM", trigger: "manual", savedAtUnix: 1_790_000_000,
                                          trainingSteps: 4321)
        let manifestURL = saved.appendingPathComponent("manifest.json")
        var stored = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
        XCTAssertEqual(stored["schema"] as? Int, SessionManifest.currentSchema)
        stored["schema"] = nil
        stored["trainingSteps"] = 1
        try JSONSerialization.data(withJSONObject: stored).write(to: manifestURL)
        XCTAssertEqual(SessionManifest.resolve(sessionFolder: saved).trainingSteps, 4321)
    }

    // MARK: - Rows

    /// Protects: the listing is every session folder, oldest save first, each
    /// with the trainer file's own lineage record, and a folder that can't be
    /// read listed with why (never skipped).
    /// Fails if: rows are ordered by name (a renamed folder sorts after the
    /// newest saves), a non-session folder is listed, the lineage comes from
    /// anywhere but the trainer file, or an unreadable session is dropped.
    func testRowsAreEverySessionOldestFirstWithTheTrainerFilesLineage() async throws {
        let newer = try await saveSession("20261021-2-NEWR", trigger: "periodic", savedAtUnix: 1_790_100_000)
        let older = try await saveSession("20261021-3-OLDR", trigger: "vsuci-final", savedAtUnix: 1_790_000_000)
        let renamed = sessionsDir.appendingPathComponent("zz-renamed.dcmsession", isDirectory: true)
        try FileManager.default.moveItem(at: older, to: renamed)
        // No session.json and no timestamp in its name: no known save time.
        let broken = sessionsDir.appendingPathComponent("broken.dcmsession", isDirectory: true)
        try FileManager.default.createDirectory(at: broken, withIntermediateDirectories: false)
        try FileManager.default.createDirectory(at: sessionsDir.appendingPathComponent("notes", isDirectory: true),
                                                withIntermediateDirectories: false)

        let rows = try ListSessionsCLI.rows(in: sessionsDir)
        XCTAssertEqual(rows.map(\.folder.lastPathComponent),
                       [broken.lastPathComponent, "zz-renamed.dcmsession", newer.lastPathComponent],
                       "unknown save time first, then by save time")
        XCTAssertFalse(rows[0].errors.isEmpty, "the folder without session.json is listed with its error")
        XCTAssertTrue(rows[1].errors.isEmpty, "\(rows[1].errors)")
        XCTAssertNil(rows[1].manifest.trigger)
        XCTAssertEqual(rows[2].manifest.trigger, "periodic")

        let expected = try XCTUnwrap(savedRecords[newer.lastPathComponent])
        guard case .recorded(let record, _) = rows[2].lineage else { return XCTFail("\(rows[2].lineage)") }
        XCTAssertEqual(record.run.lineageRunID, expected.run.lineageRunID)
        XCTAssertEqual(record.steps.cumTrainerStep, expected.steps.cumTrainerStep)

        XCTAssertEqual(ListSessionsCLI.listedRows(rows, tail: 2).map(\.folder), Array(rows.suffix(2)).map(\.folder))
        XCTAssertEqual(ListSessionsCLI.listedRows(rows, tail: nil).count, 3)
    }

    /// Protects: the table lines up and an unreadable row shows its error.
    /// Fails if: a column's width ignores a row, or an error row drops its
    /// reason.
    func testTheTableAlignsAndShowsErrors() async throws {
        _ = try await saveSession("20261021-5-TBLE", trigger: "manual", savedAtUnix: 1_790_000_000)
        try FileManager.default.createDirectory(at: sessionsDir.appendingPathComponent("broken.dcmsession"),
                                                withIntermediateDirectories: false)
        let rows = try ListSessionsCLI.rows(in: sessionsDir)
        let lines = ListSessionsCLI.render(rows)
        XCTAssertEqual(lines.count, 3)
        let header = lines[0], broken = lines[1], saved = lines[2]
        XCTAssertTrue(broken.contains("ERROR: session.json:"), broken)
        let triggerColumn = try XCTUnwrap(header.range(of: "trigger")).lowerBound.utf16Offset(in: header)
        let manualColumn = try XCTUnwrap(saved.range(of: "manual", range: saved.index(
            saved.startIndex, offsetBy: triggerColumn)..<saved.endIndex)).lowerBound.utf16Offset(in: saved)
        XCTAssertEqual(triggerColumn, manualColumn)
        let lineageColumn = try XCTUnwrap(header.range(of: "lineage")).lowerBound.utf16Offset(in: header)
        let lineage = ListSessionsCLI.lineageText(rows[1].lineage)
        XCTAssertTrue(saved.hasSuffix(lineage), saved)
        XCTAssertEqual(saved.utf16.count - lineage.utf16.count, lineageColumn)
    }

    /// Protects: the buffer column reports what session.json declares against
    /// what the folder holds.
    /// Fails if: a declared-but-absent buffer reads `no`, or an undeclared
    /// one in the folder reads `yes` — the case a missing 10 GB file hides in.
    func testTheBufferColumnComparesTheDeclarationWithTheFolder() throws {
        func label(declared: Bool?, present: Bool) -> String {
            var dict: [String: Any] = ["sessionID": "20261021-7-BUFF"]
            if let declared { dict["hasReplayBuffer"] = declared }
            let manifest = SessionManifest.extract(jsonDict: dict, folderName: "x.dcmsession", disk: nil,
                                                   srcBytes: nil, srcMTime: nil)
            return ListSessionsCLI.Row(folder: URL(fileURLWithPath: "/x.dcmsession"), manifest: manifest,
                                       lineage: .legacyTrainerFile, replayBufferFilePresent: present).bufferLabel
        }
        XCTAssertEqual(label(declared: true, present: true), "yes")
        XCTAssertEqual(label(declared: false, present: false), "no")
        XCTAssertEqual(label(declared: true, present: false), "MISSING")
        XCTAssertEqual(label(declared: false, present: true), "UNLISTED")
        XCTAssertEqual(label(declared: nil, present: true), "yes")
        XCTAssertEqual(label(declared: nil, present: false), "no")
    }

    /// Protects: a session whose arena history holds every arena counts its
    /// promotions from that history, which records each one as it happens.
    /// Fails if: the probe history's counter, updated only at probe ticks,
    /// wins again — a save made right after a promotion would show one fewer.
    func testACompleteArenaHistoryCountsPromotions() {
        let promoted: [[String: Any]] = [["promoted": true], ["promoted": false], ["promoted": true]]
        let manifest = SessionManifest.extract(
            jsonDict: ["sessionID": "20261021-8-PRMO", "selfPlayGames": 99, "arenaHistory": promoted,
                       "lichessProbeHistory": ["latestPromotionCount": 1, "latestArenaCount": 2]],
            folderName: "x.dcmsession", disk: nil, srcBytes: nil, srcMTime: nil)
        XCTAssertEqual(manifest.promotionCount, 2)
        XCTAssertEqual(manifest.selfPlayGames, 99)
    }

    // MARK: - Options

    /// Protects: the command's options, and its refusals.
    /// Fails if: an unknown or repeated option, a bad `--tail`, or
    /// `--tail` with `--all` is accepted.
    func testOptions() throws {
        let defaults = try ListSessionsCLI.parse(["app", ListSessionsCLI.flag])
        XCTAssertEqual(defaults.tail, ListSessionsCLI.defaultTail)
        XCTAssertFalse(defaults.json)
        XCTAssertEqual(defaults.sessionsDirectory, CheckpointPaths.sessionsDir)
        let all = try ListSessionsCLI.parse([ListSessionsCLI.flag, "--all", "--json", "--sessions-dir", "/tmp/s"])
        XCTAssertNil(all.tail)
        XCTAssertTrue(all.json)
        XCTAssertEqual(all.sessionsDirectory.path, "/tmp/s")
        XCTAssertEqual(try ListSessionsCLI.parse([ListSessionsCLI.flag, "--tail", "5"]).tail, 5)
        for bad in [["--tail", "0"], ["--tail", "x"], ["--tail"], ["--bogus"], ["--json", "--json"],
                    ["--tail", "5", "--all"]] {
            XCTAssertThrowsError(try ListSessionsCLI.parse([ListSessionsCLI.flag] + bad), "\(bad)")
        }
    }
}
