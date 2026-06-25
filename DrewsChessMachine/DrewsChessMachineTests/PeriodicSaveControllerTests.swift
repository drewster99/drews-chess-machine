//
//  PeriodicSaveControllerTests.swift
//  DrewsChessMachineTests
//
//  Unit tests for the 4-hour periodic session-autosave scheduler.
//  The controller is pure logic (no Timer, no Task) with an
//  injectable Date at every entry point, so these tests drive its
//  state deterministically without touching the main actor's
//  runloop at runtime.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class PeriodicSaveControllerTests: XCTestCase {

    /// Shortcut: construct a controller with a small interval so
    /// test arithmetic is easy to read. The controller does not
    /// know about "4 hours" internally; all durations are just
    /// TimeInterval values.
    private func makeController(intervalSec: TimeInterval = 100)
    -> PeriodicSaveController {
        return PeriodicSaveController(interval: intervalSec)
    }

    private let t0 = Date(timeIntervalSince1970: 1_000_000)

    // MARK: - Arm / disarm

    func testUnarmedControllerIsIdle() {
        let c = makeController()
        XCTAssertFalse(c.armed)
        XCTAssertNil(c.nextFireAt)
        XCTAssertEqual(c.decide(now: t0), .idle)
    }

    func testArmSetsDeadlineOneIntervalOut() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        XCTAssertTrue(c.armed)
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(100))
    }

    func testDisarmClearsDeadlineAndPending() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        // Cross the deadline while an arena is running → pending.
        c.noteArenaBegan()
        _ = c.decide(now: t0.addingTimeInterval(200))
        XCTAssertTrue(c.pendingFire)
        c.disarm()
        XCTAssertFalse(c.armed)
        XCTAssertNil(c.nextFireAt)
        XCTAssertFalse(c.pendingFire)
    }

    func testArmAgainResetsDeadline() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.arm(now: t0.addingTimeInterval(50))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(150))
    }

    // MARK: - Basic fire cadence

    func testIdleBeforeDeadline() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(50)), .idle)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(99)), .idle)
    }

    func testFireAtExactDeadline() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(100)), .fire)
    }

    func testFireAfterDeadline() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(150)), .fire)
    }

    func testFireRepeatsIfNoSuccessNotification() {
        // Deliberate spec: on a failed save (caller omits
        // noteSuccessfulSave) the deadline stays in the past and
        // the next decide() will re-fire. This keeps a flaky save
        // from silently skipping an interval.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(110)), .fire)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(120)), .fire)
    }

    func testSuccessfulSaveSlidesDeadlineForward() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(100)), .fire)
        c.noteSuccessfulSave(at: t0.addingTimeInterval(102))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(202))
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(150)), .idle)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(202)), .fire)
    }

    func testManualSaveBetweenTicksSlidesDeadline() {
        // A manual save at t=50 (before the scheduled deadline at
        // t=100) should push the next scheduled save to t=150.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteSuccessfulSave(at: t0.addingTimeInterval(50))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(150))
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(100)), .idle)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(150)), .fire)
    }

    // MARK: - Arena deferral

    func testDeadlineDuringArenaDefersFire() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteArenaBegan()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(150)), .idle)
        XCTAssertTrue(c.pendingFire)
    }

    func testPendingFireDispatchesOnArenaEnd() {
        // Deadline crosses during arena. Arena ends without a
        // promotion save. Next decide() should fire.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteArenaBegan()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(150)), .idle)
        c.noteArenaEnded()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(160)), .fire)
    }

    func testPostPromotionSaveSwallowsPendingFire() {
        // Deadline crosses during arena. Arena's promotion path
        // saves mid-arena; controller sees a successful save
        // before we ever reach arena-ended. Pending fire is
        // cleared, and decide() stays idle until the fresh
        // 4-hour deadline.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteArenaBegan()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(150)), .idle)
        XCTAssertTrue(c.pendingFire)
        c.noteSuccessfulSave(at: t0.addingTimeInterval(155))
        XCTAssertFalse(c.pendingFire)
        c.noteArenaEnded()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(160)), .idle)
        // Next deadline is 100s after the save, i.e. t=255.
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(255))
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(254)), .idle)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(255)), .fire)
    }

    func testArenaStartAfterDeadlineStillDefers() {
        // Timing quirk: decide() is called before the arena-begin
        // notification, so the crossing returns .fire. The next
        // call after arena-began, though, should NOT fire again
        // (we expect the caller to have taken the first .fire
        // and dispatched a save — they might still be mid-save
        // when the arena fires up).
        //
        // But actually, what's being tested: if we consume a
        // .fire, then the arena starts, and we haven't yet
        // notified noteSuccessfulSave (save is in flight), the
        // controller's nextFireAt is still stuck in the past —
        // decide() would re-fire. The arena running suppresses
        // that re-fire into a pending flag, which is correct.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(100)), .fire)
        // Caller takes the fire, save starts. Arena begins before
        // the save finishes.
        c.noteArenaBegan()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(105)), .idle)
        XCTAssertTrue(c.pendingFire)
        // Save completes successfully.
        c.noteSuccessfulSave(at: t0.addingTimeInterval(110))
        XCTAssertFalse(c.pendingFire)
    }

    func testPendingFlagClearedByDecideWhenArenaRunning() {
        // If pendingFire is somehow left set and the arena is still
        // running, decide() must not return .fire.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteArenaBegan()
        _ = c.decide(now: t0.addingTimeInterval(200))
        XCTAssertTrue(c.pendingFire)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(210)), .idle)
    }

    func testPendingFlagFiresOnceThenCleared() {
        // After an arena-ended pending fire is dispatched, the
        // controller should not immediately re-fire on the next
        // tick. In the current spec, clearing pendingFire on .fire
        // return and relying on the caller to invoke
        // noteSuccessfulSave slides the deadline forward, ending
        // the immediate re-fire risk.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteArenaBegan()
        _ = c.decide(now: t0.addingTimeInterval(200))
        c.noteArenaEnded()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(205)), .fire)
        XCTAssertFalse(c.pendingFire)
        // Caller dispatches the save; while it's in flight the
        // caller is expected to guard itself (ContentView has
        // `periodicSaveInFlight`) against calling decide() again.
        // If it does call decide() anyway, we will see a second
        // fire because the deadline is still in the past — that's
        // the documented "failed save → re-fire" behavior.
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(210)), .fire)
        // Once the save succeeds, the deadline rolls forward and
        // everything is quiet.
        c.noteSuccessfulSave(at: t0.addingTimeInterval(211))
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(220)), .idle)
    }

    // MARK: - Live interval changes

    func testUpdateIntervalReanchorsPendingDeadline() {
        // Armed at t0 with a 100s interval → deadline t=100. Lengthening
        // the interval to 150s should re-anchor relative to the SAME
        // anchor (the arm time t0), pushing the deadline to t=150 — not
        // 150s from "now".
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(100))
        c.updateInterval(150, now: t0.addingTimeInterval(40))
        XCTAssertEqual(c.interval, 150)
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(150))
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(149)), .idle)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(150)), .fire)
    }

    func testUpdateIntervalShorteningCanMoveDeadlineIntoPast() {
        // Shrinking the interval below the elapsed time should bring the
        // deadline into the past, so the next decide() fires immediately
        // — the intended "save sooner" behavior.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.updateInterval(30, now: t0.addingTimeInterval(40))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(30))
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(40)), .fire)
    }

    func testUpdateIntervalReanchorsFromLastSave() {
        // After a save at t=50 the deadline is t=150 (interval 100).
        // Changing the interval to 80 should re-anchor off the save
        // time: t=50 + 80 = t=130.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteSuccessfulSave(at: t0.addingTimeInterval(50))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(150))
        c.updateInterval(80, now: t0.addingTimeInterval(60))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(130))
    }

    func testUpdateIntervalNoOpWhenUnchanged() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.updateInterval(100, now: t0.addingTimeInterval(40))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(100))
        XCTAssertEqual(c.interval, 100)
    }

    func testUpdateIntervalWhileDisarmedJustStoresInterval() {
        // Disarmed: no deadline to re-anchor, but a subsequent arm()
        // must use the new interval.
        let c = makeController(intervalSec: 100)
        c.updateInterval(250, now: t0)
        XCTAssertEqual(c.interval, 250)
        XCTAssertNil(c.nextFireAt)
        c.arm(now: t0)
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(250))
    }

    func testUpdateIntervalLengtheningClearsStalePendingFire() {
        // Armed at t0 (interval 100 → deadline t=100). An arena is
        // running and the deadline crosses at t=120, queuing a pending
        // fire. The user then LENGTHENS the interval to 300; the deadline
        // re-anchors to t=300, which is in the future relative to now.
        // The stale pending fire must be dropped so that when the arena
        // ends the next decide() does NOT fire a save the live cadence
        // says isn't due until t=300.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteArenaBegan()
        _ = c.decide(now: t0.addingTimeInterval(120))
        XCTAssertTrue(c.pendingFire)
        c.updateInterval(300, now: t0.addingTimeInterval(130))
        XCTAssertFalse(c.pendingFire)
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(300))
        c.noteArenaEnded()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(150)), .idle)
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(300)), .fire)
    }

    func testUpdateIntervalKeepsPendingFireWhenNewDeadlineStillPast() {
        // Same setup, but the deadline crosses later (t=250) and the user
        // SHORTENS the interval to 200 → re-anchored deadline t=200, which
        // is still in the past at now=260. The pending fire is genuinely
        // due, so it must be preserved and fire once the arena ends.
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        c.noteArenaBegan()
        _ = c.decide(now: t0.addingTimeInterval(250))
        XCTAssertTrue(c.pendingFire)
        c.updateInterval(200, now: t0.addingTimeInterval(260))
        XCTAssertTrue(c.pendingFire)
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(200))
        c.noteArenaEnded()
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(270)), .fire)
    }

    // MARK: - Disarmed edge cases

    func testSuccessfulSaveWhileDisarmedDoesNotArm() {
        // If a promotion save completes just after Stop (the user
        // hit Stop during the save), the controller is disarmed
        // but still gets the callback. It should record the save
        // (for diagnostics) but NOT re-arm the schedule.
        let c = makeController(intervalSec: 100)
        c.noteSuccessfulSave(at: t0)
        XCTAssertFalse(c.armed)
        XCTAssertNil(c.nextFireAt)
        XCTAssertEqual(c.lastSuccessfulSaveAt, t0)
    }

    func testArenaNotifsWhileDisarmedAreSafe() {
        // Defensive — arena can't run while disarmed in practice,
        // but we don't want arena notifications mid-shutdown to
        // crash.
        let c = makeController(intervalSec: 100)
        c.noteArenaBegan()
        c.noteArenaEnded()
        XCTAssertEqual(c.decide(now: t0), .idle)
    }

    // MARK: - Multiple interval rollovers

    func testMultipleSequentialFiresOverLongRun() {
        let c = makeController(intervalSec: 100)
        c.arm(now: t0)
        // t=100: first deadline.
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(100)), .fire)
        c.noteSuccessfulSave(at: t0.addingTimeInterval(101))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(201))
        // t=201: second deadline.
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(201)), .fire)
        c.noteSuccessfulSave(at: t0.addingTimeInterval(202))
        XCTAssertEqual(c.nextFireAt, t0.addingTimeInterval(302))
        // t=302: third deadline.
        XCTAssertEqual(c.decide(now: t0.addingTimeInterval(302)), .fire)
    }
}
