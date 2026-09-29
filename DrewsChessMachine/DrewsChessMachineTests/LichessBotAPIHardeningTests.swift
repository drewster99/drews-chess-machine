import XCTest
@testable import DrewsChessMachine

/// Hardening of the Lichess API layer: per-priority request timeouts, the
/// transport's bounded line buffer, how a 429 is recorded, the gate staying
/// closed and honoring a running cooldown across close and reopen, a failed
/// cooldown timer, the stream reader's watchdog ending with its stream, and
/// the settings and error checks around them.
final class LichessBotAPIHardeningTests: XCTestCase {

    /// A time source whose sleep always fails, as a broken timer would.
    private struct BrokenSleepTime: LichessBotTimeSource {
        func now() -> Duration { .zero }
        func sleep(for duration: Duration) async throws {
            try Task.checkCancellation()
            throw URLError(.unknown)
        }
    }

    /// Real time that counts sleeps, to see whether a watchdog is still
    /// running.
    private final class SleepCountingTime: LichessBotTimeSource, @unchecked Sendable {
        private let base = LichessBotSystemTimeSource()
        let sleeps = SyncBox<Int>(0)
        func now() -> Duration { base.now() }
        func sleep(for duration: Duration) async throws {
            sleeps.modify { $0 += 1 }
            try await base.sleep(for: duration)
        }
    }

    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<10_000 {
            if await condition() { return }
            await Task.yield()
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func makeClient(
        gate: LichessBotRequestGate,
        records: SyncBox<[LichessBotRequestRecord]>,
        transport: LichessBotScriptedTransport
    ) throws -> LichessBotAPIClient {
        LichessBotAPIClient(
            baseURL: try LichessBotAPIClient.lichessBaseURL(),
            token: "lip_x",
            transport: transport,
            gate: gate,
            onRequest: { record in records.modify { $0.append(record) } }
        )
    }

    // MARK: - Transport and timeouts

    func testTransportPartialDeliveryIsBelowTheSplitterCap() {
        XCTAssertLessThan(LichessBotURLSessionTransport.partialLineDeliveryBytes, LichessBotNDJSONSplitter.defaultMaximumLineLength)
    }

    /// Checks the value each request carries. Which of a request's and the
    /// session's idle limits URLSession honors can't be observed here; the
    /// session's is the longest, so either way the shorter one binds.
    func testRequestsCarryTheirPriorityIdleTimeoutAndStreamsKeepTheDefault() async throws {
        let transport = LichessBotScriptedTransport { _ in (200, Data(#"{"ok":true}"#.utf8), [:]) }
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        let client = try makeClient(gate: gate, records: SyncBox([]), transport: transport)
        try await client.makeMove(gameID: "g1", uci: "e2e4", offeringDraw: false)
        try await client.chat(gameID: "g1", room: .player, text: "hi")
        try await client.upgradeToBot()
        _ = try await client.openEventStream()
        let sent = transport.requests.value
        XCTAssertEqual(sent.count, 4)
        XCTAssertEqual(Array(sent.prefix(3)).map(\.timeoutInterval), [
            LichessBotRequestTimeouts.urgentIdle,
            LichessBotRequestTimeouts.deferrableIdle,
            LichessBotRequestTimeouts.deferrableIdle,
        ])
        XCTAssertEqual(sent[3].timeoutInterval, URLRequest(url: try LichessBotAPIClient.lichessBaseURL()).timeoutInterval, "a stream open keeps the stream session's long idle limit")
        XCTAssertLessThan(LichessBotRequestTimeouts.deferrableIdle, LichessBotRequestTimeouts.urgentIdle)
        XCTAssertEqual(LichessBotRequestTimeouts.longestIdle, LichessBotRequestTimeouts.urgentIdle)
        XCTAssertLessThanOrEqual(LichessBotRequestTimeouts.longestIdle, LichessBotRequestTimeouts.resource)
    }

    // MARK: - Recording a 429

    func testRateLimitedRequestIsRecordedWithStatusAndBody() async throws {
        let records = SyncBox<[LichessBotRequestRecord]>([])
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        let transport = LichessBotScriptedTransport { _ in (429, Data(#"{"error":"Too many requests"}"#.utf8), [:]) }
        let client = try makeClient(gate: gate, records: records, transport: transport)
        do {
            try await client.upgradeToBot()
            XCTFail("a 429 must throw")
        } catch LichessBotGateError.rateLimited {
        }
        do {
            _ = try await client.openEventStream()
            XCTFail("a 429 must throw")
        } catch LichessBotGateError.rateLimited {
        }
        let recorded = records.value
        XCTAssertEqual(recorded.map(\.status), [429, 429])
        XCTAssertEqual(recorded.map(\.errorMessage), ["Too many requests", "Too many requests"])
        XCTAssertEqual(recorded.map(\.failure), [nil, nil])
    }

    // MARK: - Gate: close, cooldown, reopen

    func testALateRateLimitDoesNotReopenAClosedGate() async throws {
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        let latch = LichessBotTestLatch()
        let limited = try lichessBotTestResponse(status: 429)
        let ok = try lichessBotTestResponse(status: 200)
        let inFlight = Task {
            try await gate.perform(priority: .move, label: "in flight") {
                await latch.wait()
                return (value: 0, response: limited)
            }
        }
        try await waitUntil("the request owns the gate") { await gate.snapshot().busy }
        await gate.close(reason: "closed mid-request")
        latch.open()
        do {
            _ = try await inFlight.value
            XCTFail("a 429 must throw")
        } catch LichessBotGateError.rateLimited {
        }
        let phase = await gate.snapshot().phase
        XCTAssertEqual(phase, .closed(reason: "closed mid-request"))
        do {
            _ = try await gate.perform(priority: .move, label: "late") { (value: 0, response: ok) }
            XCTFail("the gate must stay closed")
        } catch let error as LichessBotGateError {
            XCTAssertEqual(error, .closed(reason: "closed mid-request"))
        }
    }

    /// Reopening after the breaker trips resumes the cooldown still running
    /// from the second 429, instead of sending straight away.
    func testReopenHonorsACooldownStillRunning() async throws {
        let time = LichessBotVirtualTime()
        let gate = LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in }
        let limited = try lichessBotTestResponse(status: 429)
        let ok = try lichessBotTestResponse(status: 200)
        for attempt in 0..<2 {
            do {
                _ = try await gate.perform(priority: .move, label: "move \(attempt)") { (value: 0, response: limited) }
                XCTFail("a 429 must throw")
            } catch LichessBotGateError.rateLimited {
            }
        }
        let trippedAt = time.now()
        await gate.reopen()
        let startedAt = SyncBox<Duration?>(nil)
        _ = try await gate.perform(priority: .move, label: "after reopen") {
            startedAt.value = time.now()
            return (value: 0, response: ok)
        }
        let started = try XCTUnwrap(startedAt.value)
        XCTAssertGreaterThanOrEqual(started, trippedAt + LichessBotRateLimit.minimumCooldown)
    }

    func testAFailedCooldownTimerClosesTheGate() async throws {
        let gate = LichessBotRequestGate(time: BrokenSleepTime(), breakerWindow: .seconds(3600)) { _ in }
        let limited = try lichessBotTestResponse(status: 429)
        let ok = try lichessBotTestResponse(status: 200)
        do {
            _ = try await gate.perform(priority: .move, label: "move") { (value: 0, response: limited) }
            XCTFail("a 429 must throw")
        } catch LichessBotGateError.rateLimited {
        }
        try await waitUntil("the gate closes") {
            if case .closed = await gate.snapshot().phase { return true }
            return false
        }
        do {
            _ = try await gate.perform(priority: .move, label: "after") { (value: 0, response: ok) }
            XCTFail("the gate must be closed")
        } catch LichessBotGateError.closed {
        }
    }

    // MARK: - Stream reader

    /// An invariant guard: once the stream has ended on its own, its
    /// watchdog stops too.
    func testTheWatchdogStopsWhenTheStreamEndsOnItsOwn() async throws {
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        continuation.yield(Data("{\"a\":1}\n".utf8))
        continuation.finish()
        let time = SleepCountingTime()
        let interval = Duration.milliseconds(10)
        for try await _ in LichessBotStreamReader.items(from: chunks, time: time, stallTimeout: { nil }, checkInterval: interval) {}
        try await Task.sleep(for: interval * 5)
        let settled = time.sleeps.value
        try await Task.sleep(for: interval * 10)
        XCTAssertEqual(time.sleeps.value, settled, "the watchdog must stop once the stream has ended")
    }

    // MARK: - Settings and errors

    func testBreakerWindowMustOutlastTheMinimumCooldown() {
        var settings = LichessBotSettings.testBaseline()
        settings.connection.rateLimitBreakerWindowMinutes = Int(LichessBotRateLimit.minimumCooldown.components.seconds / 60)
        XCTAssertEqual(settings.validationProblems().count, 1)
        settings.connection.rateLimitBreakerWindowMinutes += 1
        XCTAssertEqual(settings.validationProblems(), [])
    }

    func testOnlineBotsReportsAnOversizeLine() async throws {
        let huge = String(repeating: "x", count: LichessBotNDJSONSplitter.defaultMaximumLineLength + 1)
        let body = #"{"id":"bota","username":"BotA","title":"BOT"}"# + "\n" + huge + "\n"
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        let client = try makeClient(gate: gate, records: SyncBox([]), transport: LichessBotScriptedTransport { _ in (200, Data(body.utf8), [:]) })
        do {
            _ = try await client.onlineBots(count: 50)
            XCTFail("an oversize line must be reported")
        } catch LichessBotAPIError.undecodableResponse(let endpoint, _) {
            XCTAssertEqual(endpoint, "/api/bot/online")
        }
    }

    func testUsersStatusRejectsTooManyIDs() async throws {
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        let transport = LichessBotScriptedTransport { _ in (200, Data("[]".utf8), [:]) }
        let client = try makeClient(gate: gate, records: SyncBox([]), transport: transport)
        let ids = (0...LichessBotLimits.userStatusMaximumIDs).map { "u\($0)" }
        do {
            _ = try await client.usersStatus(ids: ids)
            XCTFail("too many ids must throw")
        } catch let error as LichessBotAPIError {
            XCTAssertEqual(error, .tooManyIDs(endpoint: "/api/users/status", count: ids.count, maximum: LichessBotLimits.userStatusMaximumIDs))
        }
        XCTAssertTrue(transport.requests.value.isEmpty, "nothing is sent")
    }

    func testSpeedOrderIsFastestFirstAndMatchesCaseOrder() {
        XCTAssertEqual(LichessBotSpeed.allCases.sorted(), LichessBotSpeed.allCases)
        XCTAssertTrue(LichessBotSpeed.bullet < .blitz)
        XCTAssertFalse(LichessBotSpeed.blitz < .blitz)
    }
}
