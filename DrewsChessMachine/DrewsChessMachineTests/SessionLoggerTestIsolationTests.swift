import XCTest
@testable import DrewsChessMachine

/// Pins that a test process never writes into the user's real session-log
/// folder, `~/Library/Logs/DrewsChessMachine/`.
///
/// The test bundle is hosted by the app, so every test run launches the app's
/// `init`, which starts `SessionLogger.shared`; and any code under test that
/// logs goes through the same shared logger. Before the fix both landed in the
/// real folder, which the experiment tooling, the dashboards and people read,
/// leaving `dcm_log_*.txt` files full of fake Lichess opponents and synthetic
/// training lines among the real sessions. See
/// `documentation/plans-active/TEST_SESSION_LOG_ISOLATION_PLAN.md`.
final class SessionLoggerTestIsolationTests: XCTestCase {

    /// Path of `url` with symbolic links resolved and a trailing slash, so a
    /// prefix test cannot match a sibling folder (`DrewsChessMachineX`) or miss
    /// through `/var` vs `/private/var`.
    private func comparablePath(of url: URL) -> String {
        let resolved = url.resolvingSymlinksInPath().standardizedFileURL.path
        return resolved.hasSuffix("/") ? resolved : resolved + "/"
    }

    func testSharedLoggerUnderXCTestNeverWritesIntoTheUserLogsFolder() throws {
        // The app host has normally started it already; `start()` is a no-op
        // then. Logging a line makes sure the file exists and the path is set.
        SessionLogger.shared.start()
        SessionLogger.shared.log("[TEST] SessionLoggerTestIsolationTests probe line")
        let activePath = try XCTUnwrap(SessionLogger.shared.activeLogPath,
                                       "the shared logger opened no file")
        let activeLogFolder = comparablePath(of: URL(fileURLWithPath: activePath).deletingLastPathComponent())

        let libraryURL = try FileManager.default.url(
            for: .libraryDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        )
        let userLogsFolder = comparablePath(
            of: libraryURL
                .appendingPathComponent("Logs", isDirectory: true)
                .appendingPathComponent("DrewsChessMachine", isDirectory: true)
        )
        let temporaryFolder = comparablePath(of: FileManager.default.temporaryDirectory)

        XCTAssertFalse(activeLogFolder.hasPrefix(userLogsFolder),
                       "a test process wrote its session log into the user's real log folder: \(activePath)")
        XCTAssertTrue(activeLogFolder.hasPrefix(temporaryFolder),
                      "the test process's session log is not in the temporary folder \(temporaryFolder): \(activePath)")
    }

    func testTheTestProcessIsDetectedAsAnXCTestHost() {
        XCTAssertTrue(XCTestHostDetection.isRunningUnderXCTest)
    }

    func testOnlyAnXCTestHostLeavesTheUserLogsFolder() {
        // The app and every CLI run without the XCTest signal and must keep
        // logging to the real folder exactly as before.
        XCTAssertEqual(SessionLogger.Location.forProcess(runningUnderXCTest: false), .userLibraryLogs)
        XCTAssertEqual(SessionLogger.Location.forProcess(runningUnderXCTest: true), .xcTestRunTemporaryDirectory)
    }

    func testTheSharedLoggerWritesIntoThisRunsOwnTemporaryFolder() throws {
        SessionLogger.shared.start()
        SessionLogger.shared.log("[TEST] SessionLoggerTestIsolationTests run-folder probe line")
        let activePath = try XCTUnwrap(SessionLogger.shared.activeLogPath,
                                       "the shared logger opened no file")
        XCTAssertEqual(comparablePath(of: URL(fileURLWithPath: activePath).deletingLastPathComponent()),
                       comparablePath(of: SessionLogger.xcTestRunLogsDirectory))
    }
}
