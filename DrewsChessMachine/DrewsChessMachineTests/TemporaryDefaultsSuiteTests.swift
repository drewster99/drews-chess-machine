//
//  TemporaryDefaultsSuiteTests.swift
//  DrewsChessMachineTests
//
//  A test's private defaults suite must leave nothing behind. The suites
//  used to be named suites (`UserDefaults(suiteName: "<Test>-<UUID>")`)
//  cleared with `removePersistentDomain(forName:)`, which empties the domain
//  but leaves its plist file in the user's `~/Library/Preferences`; every
//  test run added another file per test there, and thousands accumulated.
//  Deleting that file afterwards is not enough either, because the
//  preferences daemon writes the emptied domain back shortly afterwards —
//  so the check below keeps watching for a while after the suite is gone.
//

import CoreFoundation
import XCTest
@testable import DrewsChessMachine

final class TemporaryDefaultsSuiteTests: XCTestCase {

    /// How long `tearDown` keeps checking that nothing reappears after the
    /// suite's teardown: long enough for the preferences daemon's deferred
    /// write of an emptied domain, which is what put a deleted plist back.
    private static let reappearanceWatchDuration: Duration = .seconds(8)
    private static let reappearanceWatchInterval: Duration = .milliseconds(250)

    /// The suite a test made through `makeTemporaryDefaultsSuite()`, checked
    /// in `tearDown` — which XCTest runs after the test's teardown blocks, so
    /// after the helper's own removal of the suite.
    private var suiteCheckedAfterTeardown: TemporaryDefaultsSuite?

    override func tearDown() async throws {
        if let suite = suiteCheckedAfterTeardown {
            let preferencesFolder = try Self.userPreferencesFolder()
            let clock = ContinuousClock()
            let watchEnd = clock.now.advanced(by: Self.reappearanceWatchDuration)
            while true {
                XCTAssertFalse(FileManager.default.fileExists(atPath: suite.plistURL.path),
                               "the suite's plist is still on disk after the test: \(suite.plistURL.path)")
                let leftovers = try FileManager.default.contentsOfDirectory(atPath: preferencesFolder.path)
                    .filter { $0.contains(suite.identifier) }
                XCTAssertEqual(leftovers, [], "the suite left files in \(preferencesFolder.path)")
                if !leftovers.isEmpty || FileManager.default.fileExists(atPath: suite.plistURL.path) { break }
                if clock.now >= watchEnd { break }
                try await Task.sleep(for: Self.reappearanceWatchInterval)
            }
        }
        try await super.tearDown()
    }

    private static func userPreferencesFolder() throws -> URL {
        let library = try XCTUnwrap(FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first)
        return library.appendingPathComponent("Preferences", isDirectory: true)
    }

    /// The suite is stored where it says, holds what is written to it, and
    /// is gone — from there and from the user's Preferences folder — once the
    /// test's teardown has run, and stays gone.
    func testSuiteLeavesNoFilesBehindAfterTeardown() throws {
        let suite = try makeTemporaryDefaultsSuite()
        suiteCheckedAfterTeardown = suite
        suite.defaults.set(42, forKey: "probe")
        XCTAssertTrue(CFPreferencesAppSynchronize(suite.suiteName as CFString), "the suite must write to disk")
        XCTAssertTrue(FileManager.default.fileExists(atPath: suite.plistURL.path),
                      "the suite must be stored at \(suite.plistURL.path)")
        XCTAssertEqual(suite.defaults.integer(forKey: "probe"), 42)
    }

    /// Opening the suite again by name — what a test does to simulate a
    /// later launch — sees what the first instance wrote.
    func testSuiteOpenedAgainByNameSeesTheSameValues() throws {
        let suite = try makeTemporaryDefaultsSuite()
        suiteCheckedAfterTeardown = suite
        suite.defaults.set("remembered", forKey: "probe")
        let reopened = try XCTUnwrap(UserDefaults(suiteName: suite.suiteName))
        XCTAssertEqual(reopened.string(forKey: "probe"), "remembered")
    }

    /// Two suites never share values.
    func testSuitesAreIndependent() throws {
        let first = try makeTemporaryDefaults()
        let second = try makeTemporaryDefaults()
        first.set("first", forKey: "probe")
        XCTAssertNil(second.string(forKey: "probe"))
    }
}
