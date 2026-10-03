import XCTest
@testable import DrewsChessMachine

/// Light coverage for `AutoResumeController` — the launch-time auto-resume
/// flow extracted out of `UpperContentView`. The sheet presentation / countdown
/// are UI flow (and `maybePresentSheet` deliberately short-circuits under
/// XCTest), so this just pins the invariants that are testable in isolation.
@MainActor
final class AutoResumeControllerTests: XCTestCase {

    func testFreshStateIsInert() {
        let c = AutoResumeController()
        XCTAssertFalse(c.sheetShowing)
        XCTAssertNil(c.pointer)
        XCTAssertNil(c.summary)
        XCTAssertFalse(c.inFlight)
        XCTAssertEqual(c.stateVersion, 0)
        XCTAssertEqual(c.countdownRemaining, 0)
    }

    func testStateVersionTracksSheetShowing() {
        let c = AutoResumeController()
        XCTAssertEqual(c.stateVersion, 0)
        c.sheetShowing = true
        XCTAssertEqual(c.stateVersion & 2, 2)
    }

    func testMaybePresentSheetIsNoOpUnderXCTest() {
        // The XCTestConfigurationFilePath env var is set by the test runner, so
        // maybePresentSheet should bail before touching any state.
        let c = AutoResumeController()
        var resumeCalled = false
        c.onResume = { _ in resumeCalled = true }
        c.maybePresentSheet(isTrainingActive: false)
        XCTAssertFalse(c.sheetShowing)
        XCTAssertNil(c.pointer)
        XCTAssertFalse(resumeCalled)
    }

    func testPerformResumeWithNoPointerDismissesWithoutResuming() {
        let c = AutoResumeController()
        var resumeCalled = false
        c.onResume = { _ in resumeCalled = true }
        c.sheetShowing = true
        c.performResume()  // no pointer set → should dismiss(), not resume
        XCTAssertFalse(c.sheetShowing)
        XCTAssertNil(c.summary)
        XCTAssertFalse(resumeCalled)
        XCTAssertFalse(c.inFlight)
    }

    func testMarkResumeFinishedClearsInFlight() {
        let c = AutoResumeController()
        // inFlight starts false; markResumeFinished must keep/leave it false
        // and never crash even when called redundantly (the load path may call
        // it on both the success and failure legs).
        c.markResumeFinished()
        XCTAssertFalse(c.inFlight)
    }

    // MARK: - The countdown runs only while its sheet is on screen
    //
    // At launch the invalid-settings sheet and the auto-resume sheet can both
    // be asked for. Only one sheet shows at a time, and the countdown used to
    // start when the auto-resume sheet was requested, so it could reach zero
    // and resume the session behind the other sheet without the prompt ever
    // having been seen.

    /// A pointer naming a folder that does not exist: nothing here reads it.
    private func dummyPointer() -> LastSessionPointer {
        LastSessionPointer(
            sessionID: "test-session",
            directoryPath: NSTemporaryDirectory() + "AutoResumeControllerTests-\(UUID().uuidString).dcmsession",
            savedAtUnix: Int64(Date().timeIntervalSince1970),
            trigger: "manual"
        )
    }

    func testPresentingDoesNotStartTheCountdown() async throws {
        let c = AutoResumeController()
        var resumeCalled = false
        c.onResume = { _ in resumeCalled = true }
        c.present(pointer: dummyPointer(), summary: nil)
        XCTAssertTrue(c.sheetShowing)
        try await Task.sleep(for: .milliseconds(1500))
        XCTAssertEqual(c.countdownRemaining, AutoResumeController.countdownStartSec,
                       "the countdown must wait for the sheet to appear")
        XCTAssertFalse(resumeCalled)
        c.dismiss()
    }

    func testSheetAppearanceStartsTheCountdown() async throws {
        let c = AutoResumeController()
        c.present(pointer: dummyPointer(), summary: nil)
        c.sheetDidAppear()
        c.sheetDidAppear()  // a repeated `.onAppear` must not start a second countdown
        try await Task.sleep(for: .milliseconds(2500))
        let remaining = c.countdownRemaining
        XCTAssertLessThan(remaining, AutoResumeController.countdownStartSec)
        XCTAssertGreaterThanOrEqual(remaining, AutoResumeController.countdownStartSec - 3,
                                    "one countdown ticks once a second")
        c.dismiss()
    }

    /// SwiftUI can close the sheet itself (clearing `sheetShowing`), which
    /// ends the countdown loop without `dismiss()`. Presenting again must
    /// still get a countdown once the sheet reappears.
    func testRepresentingAfterAnExternalCloseRestartsTheCountdown() async throws {
        let c = AutoResumeController()
        c.present(pointer: dummyPointer(), summary: nil)
        c.sheetDidAppear()
        try await Task.sleep(for: .milliseconds(1200))
        c.sheetShowing = false
        try await Task.sleep(for: .milliseconds(1200))
        c.present(pointer: dummyPointer(), summary: nil)
        XCTAssertEqual(c.countdownRemaining, AutoResumeController.countdownStartSec)
        c.sheetDidAppear()
        try await Task.sleep(for: .milliseconds(1500))
        XCTAssertLessThan(c.countdownRemaining, AutoResumeController.countdownStartSec)
        c.dismiss()
    }
}
