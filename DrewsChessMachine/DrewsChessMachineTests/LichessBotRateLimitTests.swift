import XCTest
@testable import DrewsChessMachine

/// Rate-limit rules and backoff (Lichess bot plan §5.4, §6, E27).
final class LichessBotRateLimitTests: XCTestCase {

    func testMinimumCooldownIsAFullMinute() {
        XCTAssertEqual(LichessBotRateLimit.minimumCooldown, .seconds(60))
    }

    func testCooldownNeverGoesBelowTheFloor() {
        XCTAssertEqual(LichessBotRateLimit.cooldown(retryAfter: nil), .seconds(60))
        XCTAssertEqual(LichessBotRateLimit.cooldown(retryAfter: .zero), .seconds(60))
        XCTAssertEqual(LichessBotRateLimit.cooldown(retryAfter: .seconds(5)), .seconds(60), "a shorter Retry-After still waits the full floor")
        XCTAssertEqual(LichessBotRateLimit.cooldown(retryAfter: .seconds(60)), .seconds(60))
    }

    func testLongerRetryAfterIsHonored() {
        XCTAssertEqual(LichessBotRateLimit.cooldown(retryAfter: .seconds(300)), .seconds(300))
    }

    func testParsesDeltaSecondsRetryAfter() {
        let now = Date()
        XCTAssertEqual(LichessBotRateLimit.parseRetryAfter("120", now: now), .seconds(120))
        XCTAssertEqual(LichessBotRateLimit.parseRetryAfter(" 7 ", now: now), .seconds(7))
        XCTAssertNil(LichessBotRateLimit.parseRetryAfter("-3", now: now))
    }

    func testParsesHTTPDateRetryAfter() throws {
        let now = try XCTUnwrap(ISO8601DateFormatter().date(from: "2015-10-21T07:26:00Z"))
        let parsed = try XCTUnwrap(LichessBotRateLimit.parseRetryAfter("Wed, 21 Oct 2015 07:28:00 GMT", now: now))
        XCTAssertEqual(parsed, .seconds(120))
        XCTAssertEqual(LichessBotRateLimit.parseRetryAfter("Wed, 21 Oct 2015 07:20:00 GMT", now: now), .zero, "a date in the past means no extra wait")
    }

    func testUnparseableRetryAfterIsNil() {
        let now = Date()
        XCTAssertNil(LichessBotRateLimit.parseRetryAfter(nil, now: now))
        XCTAssertNil(LichessBotRateLimit.parseRetryAfter("", now: now))
        XCTAssertNil(LichessBotRateLimit.parseRetryAfter("soon", now: now))
    }

    // MARK: - Backoff

    private let reconnect = LichessBotBackoff(initial: .seconds(2), multiplier: 2, cap: .seconds(60))

    func testBaseDelayGrowsAndIsCapped() {
        XCTAssertEqual(reconnect.baseDelay(attempt: 0), .seconds(2))
        XCTAssertEqual(reconnect.baseDelay(attempt: 1), .seconds(4))
        XCTAssertEqual(reconnect.baseDelay(attempt: 4), .seconds(32))
        XCTAssertEqual(reconnect.baseDelay(attempt: 5), .seconds(60))
        XCTAssertEqual(reconnect.baseDelay(attempt: 40), .seconds(60))
    }

    /// Equal jitter: never less than half the base, so a retry is never
    /// immediate.
    func testJitteredDelayStaysWithinHalfToFullBase() {
        for attempt in 0..<8 {
            let base = LichessBotBackoff.seconds(reconnect.baseDelay(attempt: attempt))
            XCTAssertEqual(LichessBotBackoff.seconds(reconnect.delay(attempt: attempt, unitRandom: 0)), base / 2, accuracy: 1e-9)
            XCTAssertEqual(LichessBotBackoff.seconds(reconnect.delay(attempt: attempt, unitRandom: 1)), base, accuracy: 1e-9)
            let middle = LichessBotBackoff.seconds(reconnect.delay(attempt: attempt, unitRandom: 0.5))
            XCTAssertGreaterThan(middle, base / 2)
            XCTAssertLessThan(middle, base)
        }
    }
}
