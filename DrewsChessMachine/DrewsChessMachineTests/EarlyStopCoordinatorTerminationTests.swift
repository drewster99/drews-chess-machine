//
//  EarlyStopCoordinatorTerminationTests.swift
//  DrewsChessMachineTests
//
//  SIGTERM quits a GUI process the way ⌘Q does (owner request 2026-10-10:
//  shut the bot down nicely from outside, finishing its running games).
//  These pin the handler: the first SIGTERM asks AppKit to quit once, a
//  repeat while that quit is finishing games does nothing more, and a quit
//  the operator cancelled can be asked for again. The signal itself isn't
//  sent: in the test host that would quit the test run.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class EarlyStopCoordinatorTerminationTests: XCTestCase {

    func testTheFirstSigtermQuitsAndARepeatDoesNotAskAgain() {
        var quits = 0
        let coordinator = EarlyStopCoordinator(terminateApp: { quits += 1 })
        coordinator.handleTerminationSignal()
        XCTAssertEqual(quits, 1)
        coordinator.handleTerminationSignal()
        XCTAssertEqual(quits, 1, "a second SIGTERM while the quit finishes games doesn't ask again")
    }

    func testACancelledQuitCanBeAskedForAgain() {
        var quits = 0
        let coordinator = EarlyStopCoordinator(terminateApp: { quits += 1 })
        coordinator.handleTerminationSignal()
        coordinator.quitWasCancelled()
        coordinator.handleTerminationSignal()
        XCTAssertEqual(quits, 2)
    }

    func testACancelledQuitWithoutASigtermChangesNothing() {
        var quits = 0
        let coordinator = EarlyStopCoordinator(terminateApp: { quits += 1 })
        coordinator.quitWasCancelled()
        coordinator.handleTerminationSignal()
        XCTAssertEqual(quits, 1)
    }
}
