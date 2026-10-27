import XCTest
@testable import DrewsChessMachine

/// The one record of what stops new games (`LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`).
final class LichessBotNewGameHoldsTests: XCTestCase {
    private let start = Date(timeIntervalSince1970: 1_800_000_000)

    /// Protects: holding a reason again keeps the later end, so a lone
    /// 429's short hold never cuts a breaker's longer one short.
    /// Fails if: a new hold replaces the end unconditionally (the old
    /// `startRateLimitHold` did).
    func testTheLaterEndWins() {
        var holds = LichessBotNewGameHolds()
        holds.hold(.rateLimit, until: .at(start.addingTimeInterval(7200)))
        holds.hold(.rateLimit, until: .at(start.addingTimeInterval(900)))
        XCTAssertEqual(holds.ends[.rateLimit], .at(start.addingTimeInterval(7200)))
        holds.hold(.rateLimit, until: .at(start.addingTimeInterval(9000)))
        XCTAssertEqual(holds.ends[.rateLimit], .at(start.addingTimeInterval(9000)))
    }

    /// Protects: a hold only the operator ends outlasts any date for its
    /// reason.
    /// Fails if: a dated hold replaces an operator hold.
    func testAnOperatorHoldOutlastsDates() {
        var holds = LichessBotNewGameHolds()
        holds.hold(.lostOnTime, until: .whenOperatorGoesOnline)
        holds.hold(.lostOnTime, until: .at(start.addingTimeInterval(60)))
        XCTAssertEqual(holds.ends[.lostOnTime], .whenOperatorGoesOnline)
        XCTAssertTrue(holds.requiresOperator)
        holds.hold(.movePostFailures, until: .at(start))
        holds.hold(.movePostFailures, until: .whenOperatorGoesOnline)
        XCTAssertEqual(holds.ends[.movePostFailures], .whenOperatorGoesOnline, "an operator hold replaces a dated one")
    }

    /// Protects: expiry ends exactly the dated holds that are over, and
    /// never an operator hold.
    /// Fails if: expiry ends holds early, keeps finished ones, or ends an
    /// operator hold.
    func testExpireEndsOnlyFinishedDatedHolds() {
        var holds = LichessBotNewGameHolds()
        holds.hold(.rateLimit, until: .at(start.addingTimeInterval(60)))
        holds.hold(.reconnectStorm, until: .at(start.addingTimeInterval(1800)))
        holds.hold(.gameStoppedMoving, until: .whenOperatorGoesOnline)
        XCTAssertEqual(holds.expire(now: start.addingTimeInterval(59)), [])
        XCTAssertEqual(holds.expire(now: start.addingTimeInterval(60)), [.rateLimit], "a hold ends at its time")
        XCTAssertEqual(holds.reasons, [.reconnectStorm, .gameStoppedMoving])
        XCTAssertEqual(holds.expire(now: start.addingTimeInterval(1_000_000)), [.reconnectStorm])
        XCTAssertEqual(holds.reasons, [.gameStoppedMoving])
        XCTAssertTrue(holds.isHolding)
    }

    /// Protects: going online ends the operator holds and keeps the dated
    /// holds still running (a 429's outlives going offline and back).
    /// Fails if: going online clears every hold, or none.
    func testGoingOnlineEndsOnlyOperatorHolds() {
        var holds = LichessBotNewGameHolds()
        holds.hold(.rateLimit, until: .at(start.addingTimeInterval(600)))
        holds.hold(.lostOnTime, until: .whenOperatorGoesOnline)
        holds.hold(.gameStoppedMoving, until: .whenOperatorGoesOnline)
        XCTAssertEqual(holds.operatorWentOnline(), [.lostOnTime, .gameStoppedMoving])
        XCTAssertEqual(holds.reasons, [.rateLimit])
        XCTAssertFalse(holds.requiresOperator)
        XCTAssertEqual(holds.operatorWentOnline(), [])
    }

    /// Protects: the summary names every hold with its end, and is nil with
    /// none (the Overview line and the outgoing-send reason read it).
    /// Fails if: a hold is left out or nothing-held reads as a hold.
    func testSummaryNamesEveryHold() {
        var holds = LichessBotNewGameHolds()
        XCTAssertNil(holds.summary { _ in "t" })
        holds.hold(.rateLimitBreaker, until: .at(start))
        holds.hold(.lostOnTime, until: .whenOperatorGoesOnline)
        XCTAssertEqual(holds.summary { _ in "12:00" }, "rate-limit breaker until 12:00; lost-on-time breaker until you go online")
    }
}
