import XCTest
@testable import DrewsChessMachine

/// The automatic-save sweep protects the folder the resume pointer names. A
/// pointer that is stored but cannot be read must stop the sweep: treating it
/// as "no pointer" would leave the resume target unprotected and deletable.
final class PruneUnreadableResumePointerTests: XCTestCase {

    private var sessionsDir: URL!
    private var defaults: UserDefaults!
    private var defaultsSuiteName: String!
    private let sessionID = "20261001-1-AbCd"

    override func setUpWithError() throws {
        sessionsDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("PruneUnreadableResumePointerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: sessionsDir, withIntermediateDirectories: true)
        defaultsSuiteName = "dcm-prune-pointer-tests-\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: defaultsSuiteName))
        defaults.removePersistentDomain(forName: defaultsSuiteName)
    }

    override func tearDownWithError() throws {
        if let defaults, let defaultsSuiteName {
            defaults.removePersistentDomain(forName: defaultsSuiteName)
        }
        if let sessionsDir, FileManager.default.fileExists(atPath: sessionsDir.path) {
            try FileManager.default.removeItem(at: sessionsDir)
        }
    }

    private func makeState() throws -> SessionCheckpointState {
        let jsonText = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
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

    /// Four verified periodic autosaves, oldest first.
    private func makeFourAutosaves() throws -> [String] {
        let names = (0..<4).map { hour in
            CheckpointPaths.makeSessionDirectoryName(
                sessionID: sessionID, trigger: "periodic",
                at: Date(timeIntervalSince1970: 1_790_000_000 + TimeInterval(hour) * 3600))
        }
        let state = try makeState().encode()
        for name in names {
            let folder = sessionsDir.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            try state.write(to: SessionCheckpointLayout.stateURL(in: folder))
        }
        return names
    }

    private func sweepKeepingOne(justWritten: String) {
        CheckpointPaths.pruneAutomaticSaves(
            keeping: 1,
            protecting: sessionsDir.appendingPathComponent(justWritten, isDirectory: true),
            in: sessionsDir,
            lastSessionPointerDefaults: defaults)
    }

    private func assertAllExist(_ names: [String], file: StaticString = #filePath, line: UInt = #line) {
        for name in names {
            XCTAssertTrue(FileManager.default.fileExists(atPath: sessionsDir.appendingPathComponent(name).path),
                          "\(name) must survive a sweep whose resume pointer could not be read", file: file, line: line)
        }
    }

    func testUndecodablePointerStopsTheSweep() throws {
        let names = try makeFourAutosaves()
        defaults.set(Data("not a pointer".utf8), forKey: LastSessionPointer.userDefaultsKey)
        sweepKeepingOne(justWritten: names[3])
        assertAllExist(names)
    }

    func testPointerOfTheWrongTypeStopsTheSweep() throws {
        let names = try makeFourAutosaves()
        defaults.set("a string, not encoded pointer data", forKey: LastSessionPointer.userDefaultsKey)
        sweepKeepingOne(justWritten: names[3])
        assertAllExist(names)
    }
}
