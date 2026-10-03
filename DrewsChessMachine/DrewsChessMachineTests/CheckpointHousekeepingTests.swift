//
//  CheckpointHousekeepingTests.swift
//  DrewsChessMachineTests
//
//  Pins the rule that checkpoint housekeeping only deletes what it can
//  prove is its own.
//
//  - Automatic-save retention keeps one global pool of `-periodic` and
//    `-promote` session folders, from every session, capped by
//    `max_periodic_autosaves_kept` (owner decision 2026-10-01). These pin
//    that the pool spans sessions and is ranked by timestamp alone, that
//    manual and SIGUSR2 saves are never pool members, that the just-written
//    save and the resume-pointer target always survive, and that nothing is
//    deleted unless it is a real directory whose name and `session.json`
//    agree on a minted session ID — so a symbolic link, a stray file, a
//    folder without `session.json`, another session's `session.json`, or a
//    placeholder session ID is kept and not counted. They also pin which
//    save triggers start the sweep.
//  - The launch-time orphan sweep once deleted every `*.tmp` in `Sessions/`
//    and `Models/` by suffix alone, so a second app instance launching
//    mid-save deleted the first instance's live staging folder. These pin
//    that a fresh staging item, or one of the wrong filesystem type,
//    survives the sweep; that the removal is identity-checked, so an item
//    that took a judged entry's name is not removed; and that FileSafety's
//    own hidden `.<name>.<UUID>.tmp` staging files — which nothing used to
//    clean up — are removed once aged, while anything not exactly that
//    shape, or not a regular file, is kept.
//  - A failed save once deleted `<final>.tmp` even when that path existed
//    before the save began. These pin that a save refuses a pre-existing
//    staging path and leaves it untouched, while still removing the
//    staging item it created itself.
//
//  Everything runs in a temporary folder with an ephemeral UserDefaults
//  suite; nothing touches the real `Application Support` folders or the
//  real resume pointer.
//

import XCTest
@testable import DrewsChessMachine

final class CheckpointHousekeepingTests: XCTestCase {

    private var root: URL!
    private var sessionsDir: URL!
    private var modelsDir: URL!
    private var defaults: UserDefaults!
    private var defaultsSuiteName: String!

    private let sessionA = "20261001-1-AbCd"
    private let sessionB = "20261001-2-WxYz"

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("CheckpointHousekeepingTests-\(UUID().uuidString)", isDirectory: true)
        sessionsDir = root.appendingPathComponent("Sessions", isDirectory: true)
        modelsDir = root.appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: sessionsDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        defaultsSuiteName = "dcm-housekeeping-tests-\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: defaultsSuiteName))
        defaults.removePersistentDomain(forName: defaultsSuiteName)
    }

    override func tearDownWithError() throws {
        if let defaults, let defaultsSuiteName {
            defaults.removePersistentDomain(forName: defaultsSuiteName)
        }
        if let root, FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.removeItem(at: root)
        }
    }

    // MARK: - Fixtures

    /// A valid `SessionCheckpointState` for `sessionID`, decoded from JSON
    /// holding only the non-Optional fields (the same approach
    /// `CheckpointManagerRoundTripTests` uses, so the fixture does not break
    /// when Optional fields are added).
    private func makeState(sessionID: String) throws -> SessionCheckpointState {
        let jsonText = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
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
          "championID": "\(sessionID)",
          "trainerID": "\(sessionID)-1",
          "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(jsonText.utf8))
    }

    /// The folder name the app gives a save of `sessionID` with `trigger`,
    /// `hour` hours after a fixed instant — later hours sort newer.
    private func folderName(_ sessionID: String, _ trigger: String, hour: Int) -> String {
        CheckpointPaths.makeSessionDirectoryName(
            sessionID: sessionID,
            trigger: trigger,
            at: Date(timeIntervalSince1970: 1_790_000_000 + TimeInterval(hour) * 3600)
        )
    }

    /// Stage a session folder whose `session.json` names `jsonSessionID`.
    @discardableResult
    private func makeSessionFolder(named name: String, jsonSessionID: String) throws -> URL {
        let folder = sessionsDir.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        try makeState(sessionID: jsonSessionID).encode()
            .write(to: SessionCheckpointLayout.stateURL(in: folder))
        return folder
    }

    private func exists(_ name: String, in directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
    }

    private func sweep(keep: Int, justWritten: String) {
        CheckpointPaths.pruneAutomaticSaves(
            keeping: keep,
            protecting: sessionsDir.appendingPathComponent(justWritten, isDirectory: true),
            in: sessionsDir,
            lastSessionPointerDefaults: defaults
        )
    }

    /// A planner candidate for `name` that verified, with a distinct
    /// made-up identity per `inode`.
    private func verifiedCandidate(_ name: String, inode: Int) throws -> CheckpointPaths.AutomaticSaveCandidate {
        let parsed = try XCTUnwrap(CheckpointPaths.parseAutomaticSaveFolderName(name), "\(name) must parse")
        return CheckpointPaths.AutomaticSaveCandidate(
            folder: parsed,
            verification: .verified(identity: FileSafety.FileIdentity(device: 1, inode: ino_t(inode)))
        )
    }

    private func verifiedCandidates(_ names: [String]) throws -> [CheckpointPaths.AutomaticSaveCandidate] {
        try names.enumerated().map { try verifiedCandidate($0.element, inode: $0.offset + 1) }
    }

    private func pointResumeAt(_ name: String, sessionID: String) {
        LastSessionPointer(
            sessionID: sessionID,
            directoryPath: sessionsDir.appendingPathComponent(name, isDirectory: true).path,
            savedAtUnix: 1_790_000_000,
            trigger: "periodic"
        ).write(to: defaults)
    }

    private func setModificationDate(_ date: Date, of url: URL) throws {
        try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: url.path)
    }

    // MARK: - Automatic-save retention: sweep over a real folder

    func testSweepPoolSpansSessions() throws {
        // Session B's saves are the oldest; session A's the newest. One
        // global pool with a cap of 2 keeps A's newest two and deletes
        // everything older, whichever session wrote it.
        let bNames = (0..<3).map { folderName(sessionB, "periodic", hour: $0) }
        let aNames = (10..<13).map { folderName(sessionA, "periodic", hour: $0) }
        for name in bNames { try makeSessionFolder(named: name, jsonSessionID: sessionB) }
        for name in aNames { try makeSessionFolder(named: name, jsonSessionID: sessionA) }

        sweep(keep: 2, justWritten: aNames[2])

        for name in bNames {
            XCTAssertFalse(exists(name, in: sessionsDir), "another session's old autosave \(name) is in the same pool")
        }
        XCTAssertFalse(exists(aNames[0], in: sessionsDir))
        XCTAssertTrue(exists(aNames[1], in: sessionsDir))
        XCTAssertTrue(exists(aNames[2], in: sessionsDir))
    }

    func testSweepRanksPeriodicAndPromoteTogetherByTimestamp() throws {
        // Interleaved triggers across two sessions; the survivors are the
        // newest three by timestamp, whatever their trigger or session.
        let ordered = [
            folderName(sessionA, "promote", hour: 0),
            folderName(sessionB, "periodic", hour: 1),
            folderName(sessionA, "periodic", hour: 2),
            folderName(sessionB, "promote", hour: 3),
            folderName(sessionA, "periodic", hour: 4),
            folderName(sessionA, "promote", hour: 5),
        ]
        for name in ordered {
            let parsed = try XCTUnwrap(CheckpointPaths.parseAutomaticSaveFolderName(name))
            try makeSessionFolder(named: name, jsonSessionID: parsed.sessionID)
        }

        sweep(keep: 3, justWritten: ordered[5])

        for name in ordered[0..<3] { XCTAssertFalse(exists(name, in: sessionsDir), "\(name) is older than the newest three") }
        for name in ordered[3..<6] { XCTAssertTrue(exists(name, in: sessionsDir), "\(name) is among the newest three") }
    }

    func testSweepNeverTouchesManualOrSignalSavesAndDoesNotCountThem() throws {
        // Manual and SIGUSR2 saves, old and new, are never pool members:
        // they survive, and the newer ones do not push an automatic save
        // out of the cap.
        let deliberate = [
            folderName(sessionA, "manual", hour: 0),
            folderName(sessionB, "sigusr2", hour: 1),
            folderName(sessionA, "manual", hour: 20),
            folderName(sessionA, "sigusr2", hour: 21),
        ]
        for name in deliberate { try makeSessionFolder(named: name, jsonSessionID: name.contains(sessionB) ? sessionB : sessionA) }
        let automatic = [
            folderName(sessionA, "periodic", hour: 10),
            folderName(sessionA, "promote", hour: 11),
        ]
        for name in automatic { try makeSessionFolder(named: name, jsonSessionID: sessionA) }

        sweep(keep: 1, justWritten: automatic[1])

        for name in deliberate { XCTAssertTrue(exists(name, in: sessionsDir), "\(name) is a deliberate save and must never be pruned") }
        XCTAssertFalse(exists(automatic[0], in: sessionsDir))
        XCTAssertTrue(exists(automatic[1], in: sessionsDir))
    }

    func testSweepKeepsResumePointerTargetFromAnySession() throws {
        let names = (0..<4).map { folderName(sessionA, "periodic", hour: $0) }
        for name in names { try makeSessionFolder(named: name, jsonSessionID: sessionA) }
        let otherSessionsPromote = folderName(sessionB, "promote", hour: 1)
        try makeSessionFolder(named: otherSessionsPromote, jsonSessionID: sessionB)
        pointResumeAt(otherSessionsPromote, sessionID: sessionB)

        sweep(keep: 1, justWritten: names[3])

        XCTAssertTrue(exists(otherSessionsPromote, in: sessionsDir), "the resume pointer's target must never be pruned")
        XCTAssertFalse(exists(names[0], in: sessionsDir))
        XCTAssertFalse(exists(names[1], in: sessionsDir))
        XCTAssertFalse(exists(names[2], in: sessionsDir))
        XCTAssertTrue(exists(names[3], in: sessionsDir))
    }

    func testSweepKeepsJustWrittenSaveEvenWhenOlderThanTheCap() throws {
        let names = (0..<3).map { folderName(sessionA, "promote", hour: $0) }
        for name in names { try makeSessionFolder(named: name, jsonSessionID: sessionA) }

        sweep(keep: 1, justWritten: names[0])

        XCTAssertTrue(exists(names[0], in: sessionsDir))
        XCTAssertFalse(exists(names[1], in: sessionsDir), "a protected folder must not pull an older one back under the cap")
        XCTAssertTrue(exists(names[2], in: sessionsDir))
    }

    func testSweepKeepsAndDoesNotCountFoldersItCannotVerify() throws {
        let fm = FileManager.default
        let oldest = (0..<7).map { folderName(sessionA, $0.isMultiple(of: 2) ? "periodic" : "promote", hour: $0) }
        let verifiedNewest = (10..<12).map { folderName(sessionA, "periodic", hour: $0) }

        // A regular file with an automatic save's name.
        try Data("not a folder".utf8).write(to: sessionsDir.appendingPathComponent(oldest[0]))
        // A folder with no session.json.
        try fm.createDirectory(at: sessionsDir.appendingPathComponent(oldest[1], isDirectory: true), withIntermediateDirectories: false)
        // A folder whose session.json names a different session.
        try makeSessionFolder(named: oldest[2], jsonSessionID: sessionB)
        // A folder whose session.json is not JSON.
        let garbled = sessionsDir.appendingPathComponent(oldest[3], isDirectory: true)
        try fm.createDirectory(at: garbled, withIntermediateDirectories: false)
        try Data("{ not json".utf8).write(to: SessionCheckpointLayout.stateURL(in: garbled))
        // A symbolic link to a real, verifiable session folder.
        let linkTarget = root.appendingPathComponent("elsewhere", isDirectory: true)
        try fm.createDirectory(at: linkTarget, withIntermediateDirectories: false)
        try makeState(sessionID: sessionA).encode().write(to: SessionCheckpointLayout.stateURL(in: linkTarget))
        try fm.createSymbolicLink(at: sessionsDir.appendingPathComponent(oldest[4]), withDestinationURL: linkTarget)
        // A folder whose session.json is itself a directory.
        let stateIsDirectory = sessionsDir.appendingPathComponent(oldest[5], isDirectory: true)
        try fm.createDirectory(at: stateIsDirectory, withIntermediateDirectories: false)
        try fm.createDirectory(at: SessionCheckpointLayout.stateURL(in: stateIsDirectory), withIntermediateDirectories: false)
        // A placeholder session ID, in the name and in session.json alike.
        let placeholderID = "unknown-session"
        let placeholder = folderName(placeholderID, "periodic", hour: 7)
        try makeSessionFolder(named: placeholder, jsonSessionID: placeholderID)
        // One genuine old save, which is the only thing eligible for deletion.
        try makeSessionFolder(named: oldest[6], jsonSessionID: sessionA)
        for name in verifiedNewest { try makeSessionFolder(named: name, jsonSessionID: sessionA) }

        sweep(keep: 2, justWritten: verifiedNewest[1])

        for name in oldest[0..<6] + [placeholder] {
            XCTAssertTrue(exists(name, in: sessionsDir), "\(name) cannot be verified and must survive")
        }
        XCTAssertTrue(fm.fileExists(atPath: SessionCheckpointLayout.stateURL(in: linkTarget).path),
                      "a symbolic link's target must never be reached through it")
        XCTAssertFalse(exists(oldest[6], in: sessionsDir), "the genuine old save beyond the cap is pruned")
        XCTAssertTrue(exists(verifiedNewest[0], in: sessionsDir), "unverified folders must not count toward the cap")
        XCTAssertTrue(exists(verifiedNewest[1], in: sessionsDir))
    }

    func testSweepWithZeroCapDeletesNothing() throws {
        let names = (0..<3).map { folderName(sessionA, $0 == 1 ? "promote" : "periodic", hour: $0) }
        for name in names { try makeSessionFolder(named: name, jsonSessionID: sessionA) }

        sweep(keep: 0, justWritten: names[2])

        for name in names { XCTAssertTrue(exists(name, in: sessionsDir)) }
    }

    func testSweepWithExactlyCapFoldersDeletesNothing() throws {
        let names = [
            folderName(sessionA, "periodic", hour: 0),
            folderName(sessionB, "promote", hour: 1),
            folderName(sessionA, "promote", hour: 2),
        ]
        for name in names {
            let parsed = try XCTUnwrap(CheckpointPaths.parseAutomaticSaveFolderName(name))
            try makeSessionFolder(named: name, jsonSessionID: parsed.sessionID)
        }

        sweep(keep: 3, justWritten: names[2])

        for name in names { XCTAssertTrue(exists(name, in: sessionsDir)) }
    }

    func testInspectedIdentityRefusesAFolderSwappedInBeforeDeletion() throws {
        // The sweep deletes with the identity recorded at inspection, so a
        // folder put in place of the inspected one under the same name is
        // refused, not removed.
        let name = folderName(sessionA, "periodic", hour: 0)
        let url = try makeSessionFolder(named: name, jsonSessionID: sessionA)
        let parsed = try XCTUnwrap(CheckpointPaths.parseAutomaticSaveFolderName(name))
        guard case .verified(let inspectedIdentity) = CheckpointPaths.inspectAutomaticSaveFolder(at: url, parsed: parsed) else {
            return XCTFail("a genuine save must verify")
        }

        let parked = root.appendingPathComponent("parked", isDirectory: true)
        try FileManager.default.moveItem(at: url, to: parked)
        let replacement = try makeSessionFolder(named: name, jsonSessionID: sessionA)

        XCTAssertThrowsError(try FileSafety.removeOwnedItem(at: replacement, identity: inspectedIdentity)) { error in
            guard case FileSafetyError.fileChangedSinceWritten = error else {
                return XCTFail("expected fileChangedSinceWritten, got \(error)")
            }
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: SessionCheckpointLayout.stateURL(in: replacement).path),
                      "the swapped-in folder must be left alone")
    }

    // MARK: - Automatic-save retention: names, IDs, triggers

    func testAutomaticSaveFolderNameParsingIsExact() throws {
        let periodic = try XCTUnwrap(CheckpointPaths.parseAutomaticSaveFolderName(folderName(sessionA, "periodic", hour: 0)))
        XCTAssertEqual(periodic.sessionID, sessionA)
        XCTAssertEqual(periodic.kind, .periodic)
        let promote = try XCTUnwrap(CheckpointPaths.parseAutomaticSaveFolderName(folderName(sessionB, "promote", hour: 0)))
        XCTAssertEqual(promote.sessionID, sessionB)
        XCTAssertEqual(promote.kind, .promotion)
        // A placeholder ID still parses — the sweep reports it as kept.
        let placeholder = try XCTUnwrap(CheckpointPaths.parseAutomaticSaveFolderName(folderName("unknown-session", "periodic", hour: 0)))
        XCTAssertEqual(placeholder.sessionID, "unknown-session")

        let own = folderName(sessionA, "periodic", hour: 0)
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName(folderName(sessionA, "manual", hour: 0)))
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName(folderName(sessionA, "sigusr2", hour: 0)))
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName(folderName(sessionA, "post-promotion", hour: 0)))
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName(own + ".tmp"), "a staging folder is never a pool member")
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName("copy-of-" + own))
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName("20261001-1200-\(sessionA)-periodic.dcmsession"))
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName("20261001-120000-periodic.dcmsession"))
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName("20261001-120000--periodic.dcmsession"))
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName("20261001_120000-\(sessionA)-promote.dcmsession"))
    }

    func testMintedSessionIDShape() {
        XCTAssertTrue(CheckpointPaths.isMintedSessionID(sessionA))
        XCTAssertTrue(CheckpointPaths.isMintedSessionID("20261001-12-0aZ9"))
        XCTAssertFalse(CheckpointPaths.isMintedSessionID("unknown"))
        XCTAssertFalse(CheckpointPaths.isMintedSessionID("unknown-session"))
        XCTAssertFalse(CheckpointPaths.isMintedSessionID(""))
        XCTAssertFalse(CheckpointPaths.isMintedSessionID("20261001-0-AbCd"))
        XCTAssertFalse(CheckpointPaths.isMintedSessionID("2026101-1-AbCd"))
        XCTAssertFalse(CheckpointPaths.isMintedSessionID("20261001-1-"))
        XCTAssertFalse(CheckpointPaths.isMintedSessionID("\(sessionA)-1"), "a trainer-generation ID is not a session ID")
    }

    func testRetentionSweepRunsAfterPeriodicAndPromotionSavesOnly() {
        // The sweep scheduler decides from the save's disk tag; walk every
        // trigger so a new one has to state its retention behavior here.
        for trigger in SessionSaveTrigger.allCases {
            let kind = CheckpointPaths.AutomaticSaveKind(diskTag: trigger.diskTag)
            switch trigger {
            case .periodic:
                XCTAssertEqual(kind, .periodic)
            case .manualPromote:
                XCTAssertEqual(kind, .promotion, "Promote Trainee Now is treated exactly like an arena promotion")
            case .manual, .signalSave:
                XCTAssertNil(kind, "\(trigger) saves are deliberate and never start a sweep")
            }
        }
        // The arena's inline post-promotion save writes this tag directly.
        XCTAssertEqual(CheckpointPaths.AutomaticSaveKind(diskTag: SessionSaveTrigger.promotionDiskTag), .promotion)
        XCTAssertEqual(SessionSaveTrigger.manualPromote.diskTag, SessionSaveTrigger.promotionDiskTag)
    }

    // MARK: - Automatic-save retention: pure planner

    func testPlanKeepsProtectedFolderWithoutShiftingTheCap() throws {
        let names = [
            folderName(sessionB, "promote", hour: 0),
            folderName(sessionA, "periodic", hour: 1),
            folderName(sessionB, "periodic", hour: 2),
            folderName(sessionA, "promote", hour: 3),
            folderName(sessionA, "periodic", hour: 4),
        ]
        let unverified = CheckpointPaths.AutomaticSaveCandidate(
            folder: try XCTUnwrap(CheckpointPaths.parseAutomaticSaveFolderName(folderName(sessionA, "periodic", hour: 9))),
            verification: .missingSessionJSON
        )

        let plan = CheckpointPaths.planAutomaticSavePrune(
            candidates: try verifiedCandidates(names) + [unverified],
            keep: 2,
            protectedFolderNames: [names[0]]
        )

        XCTAssertEqual(plan.keptWithinCap.map(\.folderName), [names[4], names[3]])
        XCTAssertEqual(plan.keptProtected.map(\.folderName), [names[0]])
        XCTAssertEqual(plan.delete.map(\.folderName), [names[2], names[1]])
        XCTAssertEqual(plan.keptUnverified.map(\.folderName), [unverified.folderName],
                       "an unverified folder newer than every verified one still does not take a slot")
        XCTAssertEqual(plan.verifiedCount(of: .periodic), 3)
        XCTAssertEqual(plan.verifiedCount(of: .promotion), 2)
    }

    func testPlanOrdersByTimestampNotTriggerOrSession() throws {
        // Session IDs and tags chosen so that sorting by either would give
        // a different order than the timestamps do.
        let laterSortingSession = "20261001-9-zzzz"
        let earlierSortingSession = "20261001-1-AAAA"
        let names = [
            folderName(laterSortingSession, "promote", hour: 0),
            folderName(earlierSortingSession, "periodic", hour: 1),
            folderName(laterSortingSession, "periodic", hour: 2),
            folderName(earlierSortingSession, "promote", hour: 3),
        ]

        let plan = CheckpointPaths.planAutomaticSavePrune(
            candidates: try verifiedCandidates(names.reversed()),
            keep: 2,
            protectedFolderNames: []
        )

        XCTAssertEqual(plan.keptWithinCap.map(\.folderName), [names[3], names[2]])
        XCTAssertEqual(plan.delete.map(\.folderName), [names[1], names[0]])
    }

    func testPlanWithExactlyCapFoldersDeletesNothing() throws {
        let names = (0..<3).map { folderName(sessionA, $0 == 1 ? "promote" : "periodic", hour: $0) }
        let plan = CheckpointPaths.planAutomaticSavePrune(
            candidates: try verifiedCandidates(names),
            keep: 3,
            protectedFolderNames: []
        )
        XCTAssertEqual(plan.delete, [])
        XCTAssertEqual(plan.keptWithinCap.count, 3)
    }

    func testPlanWithZeroCapDeletesNothing() throws {
        let names = (0..<5).map { folderName(sessionA, $0.isMultiple(of: 2) ? "promote" : "periodic", hour: $0) }
        let plan = CheckpointPaths.planAutomaticSavePrune(
            candidates: try verifiedCandidates(names),
            keep: 0,
            protectedFolderNames: []
        )
        XCTAssertEqual(plan.delete, [])
        XCTAssertEqual(plan.keptProtected, [])
        XCTAssertEqual(plan.keptWithinCap.count, 5)
    }

    // MARK: - Launch-time orphan sweep

    func testCleanupLeavesFreshSessionStagingDirectoryAlone() throws {
        let staging = sessionsDir.appendingPathComponent(folderName(sessionA, "manual", hour: 0) + ".tmp", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        try Data("in progress".utf8).write(to: staging.appendingPathComponent("champion.safetensors"))

        CheckpointPaths.cleanupOrphans(sessionsDirectory: sessionsDir, modelsDirectory: modelsDir)

        XCTAssertTrue(FileManager.default.fileExists(atPath: staging.appendingPathComponent("champion.safetensors").path),
                      "a fresh staging folder may be another instance's save in progress")
    }

    func testCleanupKeepsAgedStagingDirectoryWithAFreshChild() throws {
        // The folder's own date stops moving while a save writes one large
        // file into it; the child's date is what shows the save is alive.
        let staging = sessionsDir.appendingPathComponent(folderName(sessionA, "manual", hour: 0) + ".tmp", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        let child = staging.appendingPathComponent("replay_buffer.bin")
        try Data("streaming".utf8).write(to: child)
        try setModificationDate(Date(), of: child)
        try setModificationDate(Date().addingTimeInterval(-3 * CheckpointPaths.orphanStagingMinimumAge), of: staging)

        CheckpointPaths.cleanupOrphans(sessionsDirectory: sessionsDir, modelsDirectory: modelsDir)

        XCTAssertTrue(FileManager.default.fileExists(atPath: child.path))
    }

    func testCleanupRemovesAgedStagingDebrisOfTheExpectedKind() throws {
        let fm = FileManager.default
        let longAgo = Date().addingTimeInterval(-3 * CheckpointPaths.orphanStagingMinimumAge)

        let sessionStaging = sessionsDir.appendingPathComponent(folderName(sessionA, "manual", hour: 0) + ".tmp", isDirectory: true)
        try fm.createDirectory(at: sessionStaging, withIntermediateDirectories: false)
        let sessionChild = sessionStaging.appendingPathComponent("champion.safetensors")
        try Data("dead".utf8).write(to: sessionChild)
        try setModificationDate(longAgo, of: sessionChild)
        try setModificationDate(longAgo, of: sessionStaging)

        let modelStaging = modelsDir.appendingPathComponent("20261001-120000-\(sessionA)-manual.safetensors.tmp")
        try Data("dead".utf8).write(to: modelStaging)
        try setModificationDate(longAgo, of: modelStaging)

        CheckpointPaths.cleanupOrphans(sessionsDirectory: sessionsDir, modelsDirectory: modelsDir)

        XCTAssertFalse(fm.fileExists(atPath: sessionStaging.path))
        XCTAssertFalse(fm.fileExists(atPath: modelStaging.path))
    }

    func testCleanupKeepsAgedEntriesOfTheWrongKindOrName() throws {
        let fm = FileManager.default
        let longAgo = Date().addingTimeInterval(-3 * CheckpointPaths.orphanStagingMinimumAge)

        // Session staging name, but a regular file.
        let fileWithSessionStagingName = sessionsDir.appendingPathComponent(folderName(sessionA, "manual", hour: 0) + ".tmp")
        try Data("x".utf8).write(to: fileWithSessionStagingName)
        // Model staging name, but a directory.
        let directoryWithModelStagingName = modelsDir.appendingPathComponent("20261001-120000-\(sessionA)-manual.safetensors.tmp", isDirectory: true)
        try fm.createDirectory(at: directoryWithModelStagingName, withIntermediateDirectories: false)
        // A `.tmp` in Sessions/ that is not a session staging name.
        let unrelatedTmp = sessionsDir.appendingPathComponent("notes.tmp", isDirectory: true)
        try fm.createDirectory(at: unrelatedTmp, withIntermediateDirectories: false)
        // A finished session.
        let finished = try makeSessionFolder(named: folderName(sessionA, "manual", hour: 1), jsonSessionID: sessionA)
        // A symbolic link with a model staging name.
        let linkTarget = root.appendingPathComponent("target.bin")
        try Data("x".utf8).write(to: linkTarget)
        let link = modelsDir.appendingPathComponent("20261001-130000-\(sessionA)-manual.safetensors.tmp")
        try fm.createSymbolicLink(at: link, withDestinationURL: linkTarget)

        for url in [fileWithSessionStagingName, directoryWithModelStagingName, unrelatedTmp, finished, linkTarget] {
            try setModificationDate(longAgo, of: url)
        }

        CheckpointPaths.cleanupOrphans(sessionsDirectory: sessionsDir, modelsDirectory: modelsDir)

        XCTAssertTrue(fm.fileExists(atPath: fileWithSessionStagingName.path))
        XCTAssertTrue(fm.fileExists(atPath: directoryWithModelStagingName.path))
        XCTAssertTrue(fm.fileExists(atPath: unrelatedTmp.path))
        XCTAssertTrue(fm.fileExists(atPath: SessionCheckpointLayout.stateURL(in: finished).path))
        XCTAssertEqual(try fm.destinationOfSymbolicLink(atPath: link.path), linkTarget.path, "the symbolic link itself must survive")
        XCTAssertTrue(fm.fileExists(atPath: linkTarget.path))
    }

    func testOrphanVerdictAtTheAgeBoundary() {
        let now = Date(timeIntervalSince1970: 1_790_000_000)
        let minimumAge = CheckpointPaths.orphanStagingMinimumAge
        func candidate(lastActivity: Date) -> CheckpointPaths.OrphanStagingCandidate {
            .init(name: "x.dcmsession.tmp", isDirectory: true, isRegularFile: false, isSymbolicLink: false, lastActivity: lastActivity)
        }
        XCTAssertEqual(
            CheckpointPaths.orphanStagingVerdict(for: candidate(lastActivity: now.addingTimeInterval(-minimumAge)), kind: .sessionDirectory, now: now, minimumAge: minimumAge),
            .remove
        )
        guard case .keep = CheckpointPaths.orphanStagingVerdict(for: candidate(lastActivity: now.addingTimeInterval(-minimumAge + 1)), kind: .sessionDirectory, now: now, minimumAge: minimumAge) else {
            return XCTFail("an item just younger than the bound must be kept")
        }
        guard case .keep = CheckpointPaths.orphanStagingVerdict(for: candidate(lastActivity: now.addingTimeInterval(60)), kind: .sessionDirectory, now: now, minimumAge: minimumAge) else {
            return XCTFail("an item dated in the future must be kept")
        }
    }

    /// The sweep once removed what it judged with `removeItem(at:)` — by
    /// name. A staging folder replaced between inspection and removal
    /// (another instance's live save taking the name) would have been
    /// deleted on the strength of the old folder's age. The removal now
    /// insists on the identity read at inspection.
    func testSweepRemovalRefusesAnItemSwappedInAfterInspection() throws {
        let fm = FileManager.default
        let longAgo = Date().addingTimeInterval(-3 * CheckpointPaths.orphanStagingMinimumAge)
        let staging = sessionsDir.appendingPathComponent(folderName(sessionA, "manual", hour: 0) + ".tmp", isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        let child = staging.appendingPathComponent("champion.safetensors")
        try Data("dead".utf8).write(to: child)
        try setModificationDate(longAgo, of: child)
        try setModificationDate(longAgo, of: staging)

        let inspected = try CheckpointPaths.inspectOrphanStagingCandidate(at: staging)
        XCTAssertEqual(inspected.identity, try FileSafety.existingItem(at: staging)?.identity)
        XCTAssertEqual(
            CheckpointPaths.orphanStagingVerdict(for: inspected.candidate, kind: .sessionDirectory, now: Date(),
                                                 minimumAge: CheckpointPaths.orphanStagingMinimumAge),
            .remove)

        // Another instance's live staging folder takes the name. The judged
        // folder is parked rather than deleted, so the newcomer cannot reuse
        // its inode number.
        let replacement = root.appendingPathComponent("replacement", isDirectory: true)
        try fm.createDirectory(at: replacement, withIntermediateDirectories: false)
        try Data("live".utf8).write(to: replacement.appendingPathComponent("live.bin"))
        try fm.moveItem(at: staging, to: root.appendingPathComponent("parked", isDirectory: true))
        try fm.moveItem(at: replacement, to: staging)

        CheckpointPaths.removeInspectedOrphan(inspected, at: staging)

        XCTAssertEqual(try Data(contentsOf: staging.appendingPathComponent("live.bin")), Data("live".utf8),
                       "a folder that took the judged folder's place must survive")
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: sessionsDir.path), [staging.lastPathComponent],
                       "nothing else, hidden or not, may be left in Sessions/")
    }

    // MARK: - Launch-time orphan sweep: FileSafety staging files

    /// A process killed between staging a file and renaming it into place
    /// leaves FileSafety's hidden `.<name>.<UUID>.tmp` behind — e.g. a
    /// corpus replay's rolling `--out-model` in `Models/`. Nothing ever
    /// removed those.
    func testCleanupRemovesAgedFileSafetyStagingFilesInBothFolders() throws {
        let fm = FileManager.default
        let longAgo = Date().addingTimeInterval(-3 * CheckpointPaths.orphanStagingMinimumAge)
        let inModels = FileSafety.temporarySibling(of: modelsDir.appendingPathComponent("corpus-replay-latest.safetensors"))
        let inSessions = FileSafety.temporarySibling(of: sessionsDir.appendingPathComponent("notes.json"))
        for url in [inModels, inSessions] {
            try Data("dead".utf8).write(to: url)
            try setModificationDate(longAgo, of: url)
        }

        CheckpointPaths.cleanupOrphans(sessionsDirectory: sessionsDir, modelsDirectory: modelsDir)

        XCTAssertFalse(fm.fileExists(atPath: inModels.path))
        XCTAssertFalse(fm.fileExists(atPath: inSessions.path))
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: modelsDir.path), [])
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: sessionsDir.path), [])
    }

    func testCleanupKeepsFreshMisshapenOrNonFileFileSafetyStagingNames() throws {
        let fm = FileManager.default
        let longAgo = Date().addingTimeInterval(-3 * CheckpointPaths.orphanStagingMinimumAge)

        // Exactly the shape, but fresh: may be a write in flight.
        let fresh = FileSafety.temporarySibling(of: modelsDir.appendingPathComponent("live-replay-latest.safetensors"))
        try Data("in flight".utf8).write(to: fresh)

        // Aged, but not exactly the shape FileSafety builds.
        let misshapen = [
            ".notes.tmp",
            ".x.not-a-uuid.tmp",
            ".x.0f8e5d6c-1a2b-4c3d-9e8f-abcdefabcdef.tmp",   // lower-case UUID
            "..\(UUID().uuidString).tmp",                    // empty destination name
            "x.\(UUID().uuidString).tmp",                    // not hidden
            ".x.\(UUID().uuidString).temp",
        ].map { modelsDir.appendingPathComponent($0) }
        for url in misshapen {
            try Data("someone's".utf8).write(to: url)
            try setModificationDate(longAgo, of: url)
        }

        // Exactly the shape, aged, but a folder (an interrupted removal) —
        // FileSafety never stages one.
        let folder = FileSafety.temporarySibling(of: sessionsDir.appendingPathComponent("old.dcmsession"))
        try fm.createDirectory(at: folder, withIntermediateDirectories: false)
        let folderChild = folder.appendingPathComponent("replay_buffer.bin")
        try Data("x".utf8).write(to: folderChild)
        try setModificationDate(longAgo, of: folderChild)
        try setModificationDate(longAgo, of: folder)

        // Exactly the shape, but a symbolic link.
        let linkTarget = root.appendingPathComponent("target.bin")
        try Data("target".utf8).write(to: linkTarget)
        try setModificationDate(longAgo, of: linkTarget)
        let link = FileSafety.temporarySibling(of: modelsDir.appendingPathComponent("linked.safetensors"))
        try fm.createSymbolicLink(at: link, withDestinationURL: linkTarget)

        CheckpointPaths.cleanupOrphans(sessionsDirectory: sessionsDir, modelsDirectory: modelsDir)

        XCTAssertTrue(fm.fileExists(atPath: fresh.path), "a fresh staging file may be a write in flight")
        for url in misshapen {
            XCTAssertTrue(fm.fileExists(atPath: url.path), "\(url.lastPathComponent) is not FileSafety's to remove")
        }
        XCTAssertTrue(fm.fileExists(atPath: folderChild.path))
        XCTAssertEqual(try fm.destinationOfSymbolicLink(atPath: link.path), linkTarget.path)
        XCTAssertEqual(try Data(contentsOf: linkTarget), Data("target".utf8))
    }

    // MARK: - Save staging ownership

    private let fixedSaveDate = Date(timeIntervalSince1970: 1_790_000_000)

    private var testMetadata: ModelCheckpointMetadata {
        ModelCheckpointMetadata(creator: "unit-test", trainingStep: 1, parentModelID: "", notes: "staging test")
    }

    func testSaveModelRefusesPreExistingStagingFileAndLeavesItUntouched() async throws {
        let filename = CheckpointPaths.makeFilename(modelID: sessionA, trigger: "manual", ext: "safetensors", at: fixedSaveDate)
        let staging = modelsDir.appendingPathComponent(filename).appendingPathExtension(CheckpointPaths.stagingPathExtension)
        let sentinel = Data("someone else's staging".utf8)
        try sentinel.write(to: staging)

        do {
            _ = try await CheckpointManager.saveModel(
                weights: [],
                modelID: sessionA,
                createdAtUnix: 1_790_000_000,
                metadata: testMetadata,
                lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil), trigger: "manual",
                at: fixedSaveDate,
                modelsDirectory: modelsDir
            )
            XCTFail("a save must refuse a staging path that already exists")
        } catch CheckpointManagerError.stagingPathAlreadyExists(let url) {
            XCTAssertEqual(url.lastPathComponent, staging.lastPathComponent)
        }
        XCTAssertEqual(try Data(contentsOf: staging), sentinel, "the pre-existing staging file must be left exactly as it was")
    }

    func testSaveModelFailureRemovesTheStagingFileItCreated() async throws {
        let filename = CheckpointPaths.makeFilename(modelID: sessionA, trigger: "manual", ext: "safetensors", at: fixedSaveDate)
        let staging = modelsDir.appendingPathComponent(filename).appendingPathExtension(CheckpointPaths.stagingPathExtension)

        do {
            _ = try await CheckpointManager.saveModel(
                weights: [],
                modelID: sessionA,
                createdAtUnix: 1_790_000_000,
                metadata: testMetadata,
                lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil), trigger: "manual",
                at: fixedSaveDate,
                modelsDirectory: modelsDir
            )
            XCTFail("an empty weight list must fail to encode")
        } catch {
            guard let ioError = error as? SafetensorsModelIO.IOError, case .tensorCountMismatch = ioError else {
                return XCTFail("expected the encode failure, got \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path), "the save's own staging file must be cleaned up")
        XCTAssertFalse(FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent(filename).path))
    }

    func testSaveSessionRefusesPreExistingStagingDirectoryAndLeavesItUntouched() async throws {
        let state = try makeState(sessionID: sessionA)
        let dirName = CheckpointPaths.makeSessionDirectoryName(sessionID: sessionA, trigger: "manual", at: fixedSaveDate)
        let staging = sessionsDir.appendingPathComponent(dirName + "." + CheckpointPaths.stagingPathExtension, isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        let sentinelURL = staging.appendingPathComponent("champion.safetensors")
        let sentinel = Data("another instance's live staging".utf8)
        try sentinel.write(to: sentinelURL)

        do {
            _ = try await CheckpointManager.saveSession(
                championWeights: [],
                championID: sessionA,
                championMetadata: testMetadata,
                championCreatedAtUnix: 1_790_000_000,
                trainerWeights: [[0]],
                trainerID: "\(sessionA)-1",
                trainerMetadata: testMetadata,
                trainerCreatedAtUnix: 1_790_000_000,
                state: state, lineage: try LineageRecord.forTests(trainerCompletedSteps: testMetadata.trainerSchedule.map(\.completedTrainSteps), corpus: nil), championLineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
                trigger: "manual",
                at: fixedSaveDate,
                sessionsDirectory: sessionsDir
            )
            XCTFail("a save must refuse a staging path that already exists")
        } catch CheckpointManagerError.stagingPathAlreadyExists(let url) {
            XCTAssertEqual(url.lastPathComponent, staging.lastPathComponent)
        }
        XCTAssertEqual(try Data(contentsOf: sentinelURL), sentinel, "the pre-existing staging folder must be left exactly as it was")
    }

    func testSaveSessionFailureRemovesTheStagingDirectoryItCreated() async throws {
        let state = try makeState(sessionID: sessionA)
        let dirName = CheckpointPaths.makeSessionDirectoryName(sessionID: sessionA, trigger: "manual", at: fixedSaveDate)
        let staging = sessionsDir.appendingPathComponent(dirName + "." + CheckpointPaths.stagingPathExtension, isDirectory: true)

        do {
            _ = try await CheckpointManager.saveSession(
                championWeights: [],
                championID: sessionA,
                championMetadata: testMetadata,
                championCreatedAtUnix: 1_790_000_000,
                trainerWeights: [[0]],
                trainerID: "\(sessionA)-1",
                trainerMetadata: testMetadata,
                trainerCreatedAtUnix: 1_790_000_000,
                state: state, lineage: try LineageRecord.forTests(trainerCompletedSteps: testMetadata.trainerSchedule.map(\.completedTrainSteps), corpus: nil), championLineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
                trigger: "manual",
                at: fixedSaveDate,
                sessionsDirectory: sessionsDir
            )
            XCTFail("an empty champion weight list must fail to encode")
        } catch {
            guard let ioError = error as? SafetensorsModelIO.IOError, case .tensorCountMismatch = ioError else {
                return XCTFail("expected the encode failure, got \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path), "the save's own staging folder must be cleaned up")
        XCTAssertFalse(FileManager.default.fileExists(atPath: sessionsDir.appendingPathComponent(dirName).path))
    }

    func testExclusiveStagingCreationRefusesAnExistingPath() throws {
        let file = modelsDir.appendingPathComponent("x.safetensors.tmp")
        try Data("keep".utf8).write(to: file)
        XCTAssertThrowsError(try CheckpointPaths.createStagingFileExclusively(at: file)) { error in
            guard case CheckpointManagerError.stagingPathAlreadyExists = error else {
                return XCTFail("expected stagingPathAlreadyExists, got \(error)")
            }
        }
        XCTAssertEqual(try Data(contentsOf: file), Data("keep".utf8))

        let directory = sessionsDir.appendingPathComponent("x.dcmsession.tmp", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        XCTAssertThrowsError(try CheckpointPaths.createStagingDirectoryExclusively(at: directory)) { error in
            guard case CheckpointManagerError.stagingPathAlreadyExists = error else {
                return XCTFail("expected stagingPathAlreadyExists, got \(error)")
            }
        }
    }
}
