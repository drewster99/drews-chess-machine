//
//  TemporaryDefaultsSuite.swift
//  DrewsChessMachineTests
//
//  The one way a test gets a private `UserDefaults` suite.
//

import Foundation
import XCTest
@testable import DrewsChessMachine

/// A private `UserDefaults` suite for one test, stored in a temporary folder
/// of its own and removed with that folder when the test ends.
///
/// Why not a named suite (`UserDefaults(suiteName: "<Test>-<UUID>")`) cleared
/// with `removePersistentDomain(forName:)`, which is what the tests used to
/// do: a named suite is a plist in the user's `~/Library/Preferences`, and
/// removing its persistent domain only empties it — the preferences daemon
/// keeps the empty plist file. Deleting that file as well does not help,
/// because the daemon writes the emptied domain back shortly afterwards.
/// Every test run therefore left one more file per test in the user's
/// Preferences folder, and thousands accumulated.
///
/// A suite name that is an absolute path names the plist file itself
/// (CFPreferences treats a domain that begins with `/` as a file path), so
/// this suite lives in a folder the helper created and owns. Removing the
/// folder removes the suite, and a late write from the daemon finds no
/// folder to write into. Nothing is ever written to the Preferences folder.
///
/// Opening the suite again with `UserDefaults(suiteName: suiteName)` gives
/// another instance over the same values — how a test simulates a later
/// launch of the app reading what an earlier one saved.
struct TemporaryDefaultsSuite {
    /// The suite's defaults.
    let defaults: UserDefaults
    /// The name the suite was opened with: the absolute path of its plist.
    let suiteName: String
    /// `<test class>-<UUID>`: the name of the suite's folder, unique to it.
    let identifier: String
    /// The folder holding the suite's plist, created for this suite alone.
    let directory: URL
    /// Where the suite's values are stored on disk, inside `directory`.
    let plistURL: URL
}

/// Why a temporary defaults suite could not be made or removed.
enum TemporaryDefaultsSuiteError: Error, CustomStringConvertible {
    case suiteNotOpened(suiteName: String)
    case folderRemovedBeforeTeardown(path: String)

    var description: String {
        switch self {
        case .suiteNotOpened(let suiteName):
            return "UserDefaults(suiteName:) refused the temporary suite \(suiteName)"
        case .folderRemovedBeforeTeardown(let path):
            return "the temporary defaults folder \(path) was gone before the test's teardown removed it"
        }
    }
}

extension XCTestCase {

    /// A new private defaults suite, removed after this test by a teardown
    /// block registered here. Teardown blocks run last-registered first, so
    /// one registered later (a controller's shutdown, say) still finds the
    /// suite in place.
    func makeTemporaryDefaultsSuite() throws -> TemporaryDefaultsSuite {
        let identifier = "\(String(describing: type(of: self)))-\(UUID().uuidString)"
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(identifier, isDirectory: true)
        let directoryIdentity = try FileSafety.createNewDirectory(at: directory)
        addTeardownBlock {
            switch try FileSafety.removeOwnedItem(at: directory, identity: directoryIdentity) {
            case .removed:
                break
            case .alreadyGone:
                throw TemporaryDefaultsSuiteError.folderRemovedBeforeTeardown(path: directory.path)
            }
        }
        let plistURL = directory.appendingPathComponent("defaults.plist", isDirectory: false)
        let suiteName = plistURL.path
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            throw TemporaryDefaultsSuiteError.suiteNotOpened(suiteName: suiteName)
        }
        return TemporaryDefaultsSuite(
            defaults: defaults,
            suiteName: suiteName,
            identifier: identifier,
            directory: directory,
            plistURL: plistURL
        )
    }

    /// The defaults of a new private suite (`makeTemporaryDefaultsSuite()`),
    /// for a test that never opens the suite a second time.
    func makeTemporaryDefaults() throws -> UserDefaults {
        try makeTemporaryDefaultsSuite().defaults
    }
}
