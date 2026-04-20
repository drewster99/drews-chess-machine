//
//  LastSessionPointerTests.swift
//  DrewsChessMachineTests
//
//  Unit tests for the persisted "most-recently-saved session"
//  pointer used by the app-launch auto-resume flow. Each test
//  uses a fresh in-memory `UserDefaults` suite so the global
//  `.standard` store is never touched.
//

import XCTest
@testable import DrewsChessMachine

final class LastSessionPointerTests: XCTestCase {

    /// Fresh UserDefaults backed by an in-memory suite unique to
    /// this test invocation. Using a suiteName per invocation —
    /// rather than UserDefaults() — guarantees isolation even if a
    /// crash leaves stale on-disk data from a previous run.
    private func makeEphemeralDefaults() -> UserDefaults {
        let suiteName = "dcm-tests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        // Wipe the suite's on-disk plist before the test starts so
        // we always observe a clean slate. `removePersistentDomain`
        // is the canonical clear for a suite.
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }

    // MARK: - Basic round-trip

    func testReadOnEmptyReturnsNil() {
        let defaults = makeEphemeralDefaults()
        XCTAssertNil(LastSessionPointer.read(from: defaults))
    }

    func testWriteThenReadRoundTrip() {
        let defaults = makeEphemeralDefaults()
        let pointer = LastSessionPointer(
            sessionID: "20260420-1-abcd",
            directoryPath: "/tmp/fake/20260420-103214-20260420-1-abcd-manual.dcmsession",
            savedAtUnix: 1_700_000_000,
            trigger: "manual"
        )
        pointer.write(to: defaults)
        let round = LastSessionPointer.read(from: defaults)
        XCTAssertEqual(round, pointer)
    }

    func testWriteOverwritesPriorValue() {
        let defaults = makeEphemeralDefaults()
        let first = LastSessionPointer(
            sessionID: "A", directoryPath: "/tmp/A",
            savedAtUnix: 100, trigger: "manual"
        )
        let second = LastSessionPointer(
            sessionID: "B", directoryPath: "/tmp/B",
            savedAtUnix: 200, trigger: "periodic"
        )
        first.write(to: defaults)
        second.write(to: defaults)
        XCTAssertEqual(LastSessionPointer.read(from: defaults), second)
    }

    func testClearRemovesValue() {
        let defaults = makeEphemeralDefaults()
        let pointer = LastSessionPointer(
            sessionID: "A", directoryPath: "/tmp/A",
            savedAtUnix: 100, trigger: "manual"
        )
        pointer.write(to: defaults)
        XCTAssertNotNil(LastSessionPointer.read(from: defaults))
        LastSessionPointer.clear(in: defaults)
        XCTAssertNil(LastSessionPointer.read(from: defaults))
    }

    // MARK: - Corrupt data tolerance

    func testCorruptDataReturnsNilWithoutCrash() {
        let defaults = makeEphemeralDefaults()
        // Write garbage bytes under the key. The decoder should
        // fail cleanly and `read` should return nil rather than
        // propagate the error.
        let garbage = "not-json-at-all".data(using: .utf8)!
        defaults.set(garbage, forKey: LastSessionPointer.userDefaultsKey)
        XCTAssertNil(LastSessionPointer.read(from: defaults))
    }

    func testCorruptDataLeftAlone() {
        // Deliberate behavior: a failed decode does not wipe the
        // key. A future build with a different schema might still
        // be able to interpret those bytes. Verify the raw Data
        // survives the read call.
        let defaults = makeEphemeralDefaults()
        let garbage = "still-not-json".data(using: .utf8)!
        defaults.set(garbage, forKey: LastSessionPointer.userDefaultsKey)
        _ = LastSessionPointer.read(from: defaults)
        let stored = defaults.data(forKey: LastSessionPointer.userDefaultsKey)
        XCTAssertEqual(stored, garbage)
    }

    // MARK: - directoryExists

    func testDirectoryExistsTrueWhenFolderPresent() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-pointer-test-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let pointer = LastSessionPointer(
            sessionID: "present",
            directoryPath: tmp.path,
            savedAtUnix: 0,
            trigger: "manual"
        )
        XCTAssertTrue(pointer.directoryExists)
    }

    func testDirectoryExistsFalseWhenMissing() {
        let pointer = LastSessionPointer(
            sessionID: "missing",
            directoryPath: "/tmp/definitely-does-not-exist-\(UUID().uuidString)",
            savedAtUnix: 0,
            trigger: "manual"
        )
        XCTAssertFalse(pointer.directoryExists)
    }

    func testDirectoryExistsFalseWhenPathIsARegularFile() throws {
        // Pointer must name a *directory*. If the path happens to
        // name a regular file, directoryExists should return false
        // so the launch prompt doesn't try to load a bogus target.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-pointer-file-\(UUID().uuidString).txt")
        try "not a directory".write(to: tmp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let pointer = LastSessionPointer(
            sessionID: "is-file",
            directoryPath: tmp.path,
            savedAtUnix: 0,
            trigger: "manual"
        )
        XCTAssertFalse(pointer.directoryExists)
    }

    // MARK: - directoryURL helper

    func testDirectoryURLReconstructsPath() {
        let pointer = LastSessionPointer(
            sessionID: "X",
            directoryPath: "/a/b/c.dcmsession",
            savedAtUnix: 0,
            trigger: "periodic"
        )
        XCTAssertEqual(pointer.directoryURL.path, "/a/b/c.dcmsession")
    }
}
