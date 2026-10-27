import XCTest
@testable import DrewsChessMachine

/// The circuit breakers' counts (`LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`).
final class LichessBotBreakersTests: XCTestCase {
    private let start = Date(timeIntervalSince1970: 1_800_000_000)

    /// Protects: an event-rate breaker trips at its count of events inside
    /// the window, and events older than the window don't count.
    /// Fails if: it trips early, never, or counts events across the window.
    func testEventRateBreakerTripsAtTheCountInsideTheWindow() {
        var breaker = LichessBotEventRateBreaker(count: 3, windowMinutes: 10)
        XCTAssertFalse(breaker.note(at: start))
        XCTAssertFalse(breaker.note(at: start.addingTimeInterval(60)))
        XCTAssertTrue(breaker.note(at: start.addingTimeInterval(120)))

        var spread = LichessBotEventRateBreaker(count: 3, windowMinutes: 10)
        XCTAssertFalse(spread.note(at: start))
        XCTAssertFalse(spread.note(at: start.addingTimeInterval(400)))
        XCTAssertFalse(spread.note(at: start.addingTimeInterval(601)), "the first event fell out of the window")
        XCTAssertTrue(spread.note(at: start.addingTimeInterval(700)))
    }

    /// Protects: after a trip the count starts over, so the next trip takes
    /// a fresh run of events.
    /// Fails if: every event after the trip trips again.
    func testEventRateBreakerStartsOverAfterATrip() {
        var breaker = LichessBotEventRateBreaker(count: 2, windowMinutes: 10)
        XCTAssertFalse(breaker.note(at: start))
        XCTAssertTrue(breaker.note(at: start.addingTimeInterval(1)))
        XCTAssertFalse(breaker.note(at: start.addingTimeInterval(2)))
        XCTAssertTrue(breaker.note(at: start.addingTimeInterval(3)))
    }

    /// Protects: a count of one trips at the first event.
    /// Fails if: the count is compared off by one.
    func testEventRateBreakerWithACountOfOne() {
        var breaker = LichessBotEventRateBreaker(count: 1, windowMinutes: 1)
        XCTAssertTrue(breaker.note(at: start))
    }

    /// Protects: the lost-on-time breaker trips at its count of time losses
    /// among the last N games (owner: 3 of the last 30), only on a time loss.
    /// Fails if: it counts across more games than the window, or trips on a
    /// game that wasn't lost on time.
    func testLostOnTimeTripsAtTheCountAmongTheLastGames() {
        var breaker = LichessBotLostOnTimeBreaker(count: 3, windowGames: 5)
        XCTAssertFalse(breaker.noteFinished(gameID: "a", lostOnTime: true))
        XCTAssertFalse(breaker.noteFinished(gameID: "b", lostOnTime: true))
        XCTAssertFalse(breaker.noteFinished(gameID: "c", lostOnTime: false))
        XCTAssertFalse(breaker.noteFinished(gameID: "d", lostOnTime: false))
        XCTAssertFalse(breaker.noteFinished(gameID: "e", lostOnTime: false))
        XCTAssertEqual(breaker.lossesInWindow, 2)
        XCTAssertFalse(breaker.noteFinished(gameID: "f", lostOnTime: true), "game a fell out of the last five")
        XCTAssertEqual(breaker.lossesInWindow, 2)
        XCTAssertFalse(breaker.noteFinished(gameID: "g", lostOnTime: true), "game b fell out too")
        XCTAssertEqual(breaker.lossesInWindow, 2)
        XCTAssertTrue(breaker.noteFinished(gameID: "h", lostOnTime: true), "f, g and h: three in the last five")
        XCTAssertEqual(breaker.lossesInWindow, 0, "the count starts over after a trip")
    }

    /// Protects: a game is counted once, however many times its end is
    /// reported (a resumed game, a repeated final state).
    /// Fails if: one game's repeated end trips the breaker.
    func testLostOnTimeCountsAGameOnce() {
        var breaker = LichessBotLostOnTimeBreaker(count: 2, windowGames: 30)
        XCTAssertFalse(breaker.noteFinished(gameID: "a", lostOnTime: true))
        XCTAssertFalse(breaker.noteFinished(gameID: "a", lostOnTime: true))
        XCTAssertEqual(breaker.lossesInWindow, 1)
    }

    /// Protects: lost on time means our flag fell and we scored nothing; a
    /// flag with no winner (opponent can't mate) or the opponent's flag is
    /// not.
    /// Fails if: the rule counts draws or wins on time.
    func testTimeForfeitLossRule() {
        XCTAssertTrue(LichessBotGameEnding.isTimeForfeitLoss(status: "outoftime", ourScore: 0))
        XCTAssertFalse(LichessBotGameEnding.isTimeForfeitLoss(status: "outoftime", ourScore: 0.5))
        XCTAssertFalse(LichessBotGameEnding.isTimeForfeitLoss(status: "outoftime", ourScore: 1))
        XCTAssertFalse(LichessBotGameEnding.isTimeForfeitLoss(status: "mate", ourScore: 0))
        XCTAssertFalse(LichessBotGameEnding.isTimeForfeitLoss(status: "outoftime", ourScore: nil))
    }
}
