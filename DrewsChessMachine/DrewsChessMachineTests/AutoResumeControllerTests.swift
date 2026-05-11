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
}
