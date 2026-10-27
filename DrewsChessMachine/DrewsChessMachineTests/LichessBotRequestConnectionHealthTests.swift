import XCTest
@testable import DrewsChessMachine

/// The rule for replacing the bot's request connections
/// (`LICHESS_BOT_REQUEST_FAILURE_RECOVERY_PLAN.md`).
final class LichessBotRequestConnectionHealthTests: XCTestCase {

    private let timedOut = URLError(.timedOut)

    /// Protects: two requests in a row that got no answer replace the
    /// request session.
    /// Fails if: connection failures are never acted on (2026-10-26).
    func testTwoConsecutiveConnectionFailuresReset() {
        var health = LichessBotRequestConnectionHealth()
        XCTAssertNil(health.noteFailure(timedOut, generation: 0, now: .seconds(0)))
        let reset = health.noteFailure(timedOut, generation: 0, now: .seconds(1))
        XCTAssertEqual(reset?.failures, 2)
        XCTAssertEqual(reset?.generation, 1)
        XCTAssertEqual(reset?.lastFailure, timedOut.localizedDescription)
        XCTAssertEqual(health.generation, 1)
        XCTAssertEqual(health.consecutiveFailures, 0)
    }

    /// Protects: an answer between two failures proves the connection
    /// works.
    /// Fails if: failures are counted in total rather than in a row.
    func testAnAnswerBetweenFailuresClearsTheCount() {
        var health = LichessBotRequestConnectionHealth()
        XCTAssertNil(health.noteFailure(timedOut, generation: 0, now: .seconds(0)))
        health.noteAnswered(generation: 0)
        XCTAssertNil(health.noteFailure(timedOut, generation: 0, now: .seconds(1)))
        XCTAssertEqual(health.generation, 0)
    }

    /// Protects: only "no answer" counts; Lichess answering with an error,
    /// DCM cancelling, or any other error leaves the count alone.
    /// Fails if: a non-connection error counts toward a reset.
    func testOnlyConnectionFailuresCount() {
        let ignored: [(name: String, error: any Error)] = [
            ("cancelled", URLError(.cancelled)),
            ("bad server response", URLError(.badServerResponse)),
            ("HTTP status", LichessBotAPIError.http(status: 503, message: nil)),
            ("cancellation", CancellationError()),
            ("stream headers", LichessBotTransportError.streamHeadersTimedOut(.seconds(15))),
        ]
        for (name, error) in ignored {
            var health = LichessBotRequestConnectionHealth()
            XCTAssertNil(health.noteFailure(error, generation: 0, now: .seconds(0)), name)
            XCTAssertNil(health.noteFailure(error, generation: 0, now: .seconds(1)), name)
            XCTAssertEqual(health.consecutiveFailures, 0, name)
            XCTAssertNil(health.noteFailure(timedOut, generation: 0, now: .seconds(2)), "\(name): one counted failure is not enough")
        }
    }

    /// Protects: every code that means no answer arrived counts, including
    /// as an `NSError` in `NSURLErrorDomain` (how URLSession reports it).
    /// Fails if: a connection-failure code is missed.
    func testEveryConnectionFailureCodeCounts() {
        let codes: [URLError.Code] = [.timedOut, .cannotConnectToHost, .networkConnectionLost, .cannotFindHost, .dnsLookupFailed, .notConnectedToInternet, .secureConnectionFailed]
        for code in codes {
            XCTAssertTrue(LichessBotRequestConnectionHealth.isConnectionFailure(URLError(code)), "\(code)")
            let bridged = NSError(domain: NSURLErrorDomain, code: code.rawValue)
            XCTAssertTrue(LichessBotRequestConnectionHealth.isConnectionFailure(bridged), "NSError \(code)")
        }
        XCTAssertEqual(LichessBotRequestConnectionHealth.connectionFailureCodes, Set(codes))
    }

    /// Protects: a network that stays down doesn't replace the session more
    /// than once per spacing, and the first failure after it does.
    /// Fails if: resets churn, or failures inside the spacing are dropped.
    func testResetsAreSpacedAndFailuresInsideTheSpacingKeepCounting() {
        var health = LichessBotRequestConnectionHealth()
        let spacing = LichessBotRequestConnectionHealth.minimumResetSpacing
        _ = health.noteFailure(timedOut, generation: 0, now: .seconds(0))
        XCTAssertNotNil(health.noteFailure(timedOut, generation: 0, now: .seconds(1)))
        XCTAssertNil(health.noteFailure(timedOut, generation: 1, now: .seconds(2)))
        XCTAssertNil(health.noteFailure(timedOut, generation: 1, now: .seconds(3)), "inside the spacing")
        XCTAssertNil(health.noteFailure(timedOut, generation: 1, now: .seconds(1) + spacing - .milliseconds(1)), "still inside")
        let reset = health.noteFailure(timedOut, generation: 1, now: .seconds(1) + spacing)
        XCTAssertEqual(reset?.failures, 4)
        XCTAssertEqual(reset?.generation, 2)
    }

    /// Protects: results from a replaced session say nothing about the new
    /// one.
    /// Fails if: a late failure (or answer) on the old session counts
    /// toward (or clears) the new session's run.
    func testAReplacedSessionsResultsAreIgnored() {
        var health = LichessBotRequestConnectionHealth()
        _ = health.noteFailure(timedOut, generation: 0, now: .seconds(0))
        XCTAssertNotNil(health.noteFailure(timedOut, generation: 0, now: .seconds(1)))
        XCTAssertNil(health.noteFailure(timedOut, generation: 0, now: .seconds(2)))
        XCTAssertEqual(health.consecutiveFailures, 0, "an old session's failure doesn't count")
        _ = health.noteFailure(timedOut, generation: 1, now: .seconds(3))
        health.noteAnswered(generation: 0)
        XCTAssertEqual(health.consecutiveFailures, 1, "an old session's answer doesn't clear the new one's run")
    }
}
