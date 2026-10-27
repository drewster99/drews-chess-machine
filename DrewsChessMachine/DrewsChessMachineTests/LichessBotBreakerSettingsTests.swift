import XCTest
@testable import DrewsChessMachine

/// The circuit-breaker settings (`LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`):
/// the owner's defaults, validation, and a saved breaker window keeping its
/// meaning across the move into `breakers`.
final class LichessBotBreakerSettingsTests: XCTestCase {

    /// Protects: the owner's decided defaults (2026-10-26).
    /// Fails if: a default drifts from the decision.
    func testDefaultsAreTheOwnersDecisions() {
        let b = LichessBotBreakerSettings()
        XCTAssertEqual(b.rateLimit, LichessBotTimedBreakerSettings(count: 2, windowMinutes: 60, resumesAutomatically: true, resumeAfterMinutes: 120))
        XCTAssertEqual(b.lostOnTime, LichessBotLostOnTimeBreakerSettings(count: 3, windowGames: 30, resumesAutomatically: false, resumeAfterMinutes: 120))
        XCTAssertEqual(b.movePostFailures, LichessBotTimedBreakerSettings(count: 5, windowMinutes: 10, resumesAutomatically: true, resumeAfterMinutes: 30))
        XCTAssertEqual(b.reconnectStorm, LichessBotTimedBreakerSettings(count: 10, windowMinutes: 10, resumesAutomatically: true, resumeAfterMinutes: 30))
        XCTAssertEqual(LichessBotSettings().validationProblems(), [])
    }

    /// Protects: each breaker's limits are validated.
    /// Fails if: a zero count, a lost-on-time window smaller than its count,
    /// a zero resume time or an out-of-range margin is accepted.
    func testInvalidBreakerSettingsAreRefused() {
        func problems(_ edit: (inout LichessBotSettings) -> Void) -> Int {
            var settings = LichessBotSettings()
            edit(&settings)
            return settings.validationProblems().count
        }
        XCTAssertEqual(problems { $0.breakers.movePostFailures.count = 0 }, 1)
        XCTAssertEqual(problems { $0.breakers.reconnectStorm.windowMinutes = 0 }, 1)
        XCTAssertEqual(problems { $0.breakers.rateLimit.resumeAfterMinutes = 0 }, 1)
        XCTAssertEqual(problems { $0.breakers.lostOnTime.windowGames = 2 }, 1)
        XCTAssertEqual(problems { $0.breakers.lostOnTime.count = 0 }, 1)
        XCTAssertEqual(problems { $0.connection.rateLimitCooldownMarginSeconds = -1 }, 1)
        XCTAssertEqual(problems { $0.connection.rateLimitCooldownMarginSeconds = 601 }, 1)
        XCTAssertEqual(problems { $0.breakers.rateLimit.count = 1; $0.breakers.rateLimit.windowMinutes = 1 }, 0, "one 429 trips it: the window doesn't matter")
    }

    /// Protects: a save's `connection.rateLimitBreakerWindowMinutes` keeps
    /// its meaning as `breakers.rateLimit.windowMinutes` (owner rule: saved
    /// settings survive changes).
    /// Fails if: the saved window falls back to the default, or the old key
    /// is reported as unknown instead of carried forward.
    func testASavedBreakerWindowIsCarriedForward() throws {
        let defaults = try makeTemporaryDefaults()
        let data = try JSONEncoder().encode(LichessBotSettings())
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["breakers"] = nil
        var connection = try XCTUnwrap(object["connection"] as? [String: Any])
        connection["rateLimitBreakerWindowMinutes"] = 90
        object["connection"] = connection
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: LichessBotSettingsStore.defaultsKey)

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings.breakers.rateLimit.windowMinutes, 90)
        XCTAssertEqual(result.settings.breakers.rateLimit.count, 2, "the rest of the breakers come from the defaults")
        XCTAssertEqual(result.ignoredSavedKeys, [])
        XCTAssertEqual(result.carriedForward.count, 1)
    }
}
