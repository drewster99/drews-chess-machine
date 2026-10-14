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

    /// The quit SIGTERM asks for runs AppKit's `applicationShouldTerminate`
    /// inside `terminateApp`; it must not already see a quit in progress,
    /// or it answers `.terminateLater` without asking the bot and nothing
    /// ever answers (the app hangs half-quit).
    func testTheSigtermQuitIsNotPendingWhileAppKitAsks() {
        var pendingWhileAsked: [Bool] = []
        var coordinator: EarlyStopCoordinator?
        coordinator = EarlyStopCoordinator(terminateApp: {
            pendingWhileAsked.append(coordinator?.isQuitInProgress ?? true)
        })
        coordinator?.handleTerminationSignal()
        XCTAssertEqual(pendingWhileAsked, [false])
        XCTAssertEqual(coordinator?.isQuitInProgress, false, "only the delegate marks a deferred quit pending (quitDeferred)")
    }

    /// A quit AppKit deferred, whatever started it (⌘Q), makes a SIGTERM
    /// only log.
    func testAQuitDeferredByTheAppMakesASigtermOnlyLog() {
        var quits = 0
        let coordinator = EarlyStopCoordinator(terminateApp: { quits += 1 })
        coordinator.quitDeferred()
        coordinator.handleTerminationSignal()
        XCTAssertEqual(quits, 0)
    }

    /// The quit SIGTERM asks for runs later, from the run loop, never inside
    /// the handler's main-actor task: AppKit's quit loop run inside that task
    /// kept the main actor busy, so the bot's reply to `.terminateLater`
    /// never ran and the process hung half-quit (2026-10-13).
    func testTheSigtermQuitRunsFromTheRunLoopNotInsideTheHandler() {
        var ran = false
        let deferred = EarlyStopCoordinator.deferredToTheRunLoop { ran = true }
        deferred()
        XCTAssertFalse(ran, "not run inside the caller")
        RunLoop.main.run(until: Date().addingTimeInterval(0.2))
        XCTAssertTrue(ran, "run from the main run loop")
    }
}
