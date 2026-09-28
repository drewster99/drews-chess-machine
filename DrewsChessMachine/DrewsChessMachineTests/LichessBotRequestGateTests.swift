import XCTest
@testable import DrewsChessMachine

/// Time that moves only when something sleeps: `sleep(for:)` advances the
/// clock by the requested amount at once. Lets tests check cooldowns
/// measured in minutes without waiting for them.
final class LichessBotVirtualTime: LichessBotTimeSource, @unchecked Sendable {
    private let current = SyncBox<Duration>(.zero)

    func now() -> Duration {
        current.value
    }

    func sleep(for duration: Duration) async throws {
        try Task.checkCancellation()
        current.modify { $0 += duration }
        await Task.yield()
    }
}

/// A one-shot signal a request body can wait on, so a test can hold the gate
/// busy while it queues other requests behind it.
final class LichessBotTestLatch: @unchecked Sendable {
    private let stream: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation

    init() {
        (stream, continuation) = AsyncStream<Void>.makeStream()
    }

    func wait() async {
        for await _ in stream {
            return
        }
    }

    func open() {
        continuation.yield()
        continuation.finish()
    }
}

func lichessBotTestResponse(status: Int, headers: [String: String] = [:]) throws -> HTTPURLResponse {
    let url = try XCTUnwrap(URL(string: "https://lichess.org/test"))
    return try XCTUnwrap(HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers))
}

/// `LichessBotRequestGate` — single flight, priority order, eligibility,
/// and the 429 cooldown and breaker (Lichess bot plan §5, E18).
final class LichessBotRequestGateTests: XCTestCase {

    private final class EventLog: @unchecked Sendable {
        let events = SyncBox<[LichessBotGateEvent]>([])
        func append(_ event: LichessBotGateEvent) {
            events.modify { $0.append(event) }
        }
    }

    private func makeGate(time: any LichessBotTimeSource = LichessBotVirtualTime(), breakerWindow: Duration = .seconds(3600), log: EventLog = EventLog()) -> LichessBotRequestGate {
        LichessBotRequestGate(time: time, breakerWindow: breakerWindow) { event in
            log.append(event)
        }
    }

    private func ok() throws -> HTTPURLResponse {
        try lichessBotTestResponse(status: 200)
    }

    /// Spin until `condition` holds, failing after a bounded number of yields.
    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<10_000 {
            if await condition() { return }
            await Task.yield()
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    func testOnlyOneRequestIsEverInFlight() async throws {
        let gate = makeGate()
        let inFlight = SyncBox<Int>(0)
        let maximum = SyncBox<Int>(0)
        let response = try ok()

        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<24 {
                let priority = LichessBotRequestPriority.allCases[index % LichessBotRequestPriority.allCases.count]
                group.addTask {
                    _ = try await gate.perform(priority: priority, label: "r\(index)") {
                        let now = inFlight.mutate { value -> Int in
                            value += 1
                            return value
                        }
                        maximum.modify { $0 = max($0, now) }
                        for _ in 0..<5 { await Task.yield() }
                        inFlight.modify { $0 -= 1 }
                        return (value: index, response: response)
                    }
                }
            }
            try await group.waitForAll()
        }
        XCTAssertEqual(maximum.value, 1)
    }

    func testWaitersAreServedInPriorityOrder() async throws {
        let gate = makeGate()
        let latch = LichessBotTestLatch()
        let order = SyncBox<[LichessBotRequestPriority]>([])
        let response = try ok()

        let holder = Task {
            try await gate.perform(priority: .housekeeping, label: "holder") {
                await latch.wait()
                return (value: 0, response: response)
            }
        }
        try await waitUntil("the holder owns the gate") { await gate.snapshot().busy }

        let queued: [LichessBotRequestPriority] = [.housekeeping, .chat, .streamOpen, .move, .challengeResponse, .gameCritical]
        var tasks: [Task<(value: Int, response: HTTPURLResponse), Error>] = []
        for priority in queued {
            tasks.append(Task {
                try await gate.perform(priority: priority, label: priority.label) {
                    order.modify { $0.append(priority) }
                    return (value: 0, response: response)
                }
            })
            // Enqueue one at a time so arrival order is known.
            let expected = tasks.count
            try await waitUntil("\(expected) waiting") { await gate.snapshot().waiting == expected }
        }

        latch.open()
        _ = try await holder.value
        for task in tasks { _ = try await task.value }
        XCTAssertEqual(order.value, [.move, .gameCritical, .challengeResponse, .streamOpen, .chat, .housekeeping])
    }

    func testHousekeepingWaitsWhileAGameAwaitsOurMove() async throws {
        let gate = makeGate()
        let ran = SyncBox<Bool>(false)
        let response = try ok()
        await gate.setGamesAwaitingOurMove(1)

        let housekeeping = Task {
            try await gate.perform(priority: .housekeeping, label: "export") {
                ran.value = true
                return (value: 0, response: response)
            }
        }
        try await waitUntil("housekeeping is queued") { await gate.snapshot().waiting == 1 }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertFalse(ran.value, "housekeeping must not run while a game awaits our move")
        let busy = await gate.snapshot().busy
        XCTAssertFalse(busy)

        // A move still goes straight through.
        _ = try await gate.perform(priority: .move, label: "move") { (value: 0, response: response) }

        await gate.setGamesAwaitingOurMove(0)
        _ = try await housekeeping.value
        XCTAssertTrue(ran.value)
    }

    /// E18: while any game's clock is low, only moves, game-critical actions
    /// and stream opens go out.
    func testLowClockAdmitsOnlyUrgentTraffic() async throws {
        let gate = makeGate()
        let response = try ok()
        await gate.setLowClockUrgency(true)

        let chatRan = SyncBox<Bool>(false)
        let chat = Task {
            try await gate.perform(priority: .chat, label: "chat") {
                chatRan.value = true
                return (value: 0, response: response)
            }
        }
        try await waitUntil("chat is queued") { await gate.snapshot().waiting == 1 }

        for priority in [LichessBotRequestPriority.move, .gameCritical, .streamOpen] {
            _ = try await gate.perform(priority: priority, label: priority.label) { (value: 0, response: response) }
        }
        XCTAssertFalse(chatRan.value)

        await gate.setLowClockUrgency(false)
        _ = try await chat.value
        XCTAssertTrue(chatRan.value)
    }

    func testRateLimitStopsEverythingForAtLeastAMinute() async throws {
        let time = LichessBotVirtualTime()
        let log = EventLog()
        let gate = makeGate(time: time, log: log)
        let limited = try lichessBotTestResponse(status: 429)
        let response = try ok()

        do {
            _ = try await gate.perform(priority: .chat, label: "chat") { (value: 0, response: limited) }
            XCTFail("a 429 must throw")
        } catch let error as LichessBotGateError {
            XCTAssertEqual(error, .rateLimited(cooldown: .seconds(60)))
        }
        let limitedAt = time.now()

        let startedAt = SyncBox<Duration?>(nil)
        _ = try await gate.perform(priority: .move, label: "move") {
            startedAt.value = time.now()
            return (value: 0, response: response)
        }
        let start = try XCTUnwrap(startedAt.value)
        XCTAssertGreaterThanOrEqual(start - limitedAt, .seconds(60), "nothing — not even a move — goes out during the cooldown")

        let events = log.events.value
        XCTAssertTrue(events.contains { if case .rateLimited = $0 { return true } else { return false } })
        XCTAssertTrue(events.contains { if case .cooldownEnded = $0 { return true } else { return false } })
    }

    func testLongerRetryAfterIsHonored() async throws {
        let time = LichessBotVirtualTime()
        let gate = makeGate(time: time)
        let limited = try lichessBotTestResponse(status: 429, headers: ["Retry-After": "300"])
        let response = try ok()

        do {
            _ = try await gate.perform(priority: .move, label: "move") { (value: 0, response: limited) }
            XCTFail("a 429 must throw")
        } catch let error as LichessBotGateError {
            XCTAssertEqual(error, .rateLimited(cooldown: .seconds(300)))
        }
        let limitedAt = time.now()
        let startedAt = SyncBox<Duration?>(nil)
        _ = try await gate.perform(priority: .move, label: "move") {
            startedAt.value = time.now()
            return (value: 0, response: response)
        }
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(startedAt.value) - limitedAt, .seconds(300))
    }

    func testSecondRateLimitInsideTheWindowTripsTheBreaker() async throws {
        let log = EventLog()
        let gate = makeGate(breakerWindow: .seconds(3600), log: log)
        let limited = try lichessBotTestResponse(status: 429)
        let response = try ok()

        for attempt in 0..<2 {
            do {
                _ = try await gate.perform(priority: .move, label: "move \(attempt)") { (value: 0, response: limited) }
                XCTFail("a 429 must throw")
            } catch LichessBotGateError.rateLimited {
                // expected
            }
        }
        XCTAssertTrue(log.events.value.contains { if case .breakerTripped = $0 { return true } else { return false } })

        do {
            _ = try await gate.perform(priority: .move, label: "after breaker") { (value: 0, response: response) }
            XCTFail("the gate must stay closed after the breaker trips")
        } catch LichessBotGateError.closed {
            // expected
        }

        await gate.reopen()
        _ = try await gate.perform(priority: .move, label: "after reopen") { (value: 0, response: response) }
    }

    func testRateLimitsOutsideTheWindowDoNotTripTheBreaker() async throws {
        let log = EventLog()
        let gate = makeGate(breakerWindow: .seconds(30), log: log)
        let limited = try lichessBotTestResponse(status: 429)
        let response = try ok()

        for attempt in 0..<2 {
            do {
                _ = try await gate.perform(priority: .move, label: "move \(attempt)") { (value: 0, response: limited) }
                XCTFail("a 429 must throw")
            } catch LichessBotGateError.rateLimited {
                // expected: the second 429 comes at least a full cooldown
                // after the first, outside the short window.
            }
        }
        XCTAssertFalse(log.events.value.contains { if case .breakerTripped = $0 { return true } else { return false } })
        _ = try await gate.perform(priority: .move, label: "still open") { (value: 0, response: response) }
    }

    func testCloseFailsWaitersAndLaterRequests() async throws {
        let gate = makeGate()
        let latch = LichessBotTestLatch()
        let response = try ok()

        let holder = Task {
            try await gate.perform(priority: .move, label: "holder") {
                await latch.wait()
                return (value: 0, response: response)
            }
        }
        try await waitUntil("the holder owns the gate") { await gate.snapshot().busy }
        let waiter = Task {
            try await gate.perform(priority: .chat, label: "waiter") { (value: 0, response: response) }
        }
        try await waitUntil("one waiter") { await gate.snapshot().waiting == 1 }

        await gate.close(reason: "going offline")
        do {
            _ = try await waiter.value
            XCTFail("a waiter must fail when the gate closes")
        } catch let error as LichessBotGateError {
            XCTAssertEqual(error, .closed(reason: "going offline"))
        }

        latch.open()
        _ = try await holder.value

        do {
            _ = try await gate.perform(priority: .move, label: "late") { (value: 0, response: response) }
            XCTFail("requests must fail while closed")
        } catch let error as LichessBotGateError {
            XCTAssertEqual(error, .closed(reason: "going offline"))
        }
    }

    func testCancellingAWaiterRemovesIt() async throws {
        let gate = makeGate()
        let latch = LichessBotTestLatch()
        let response = try ok()

        let holder = Task {
            try await gate.perform(priority: .move, label: "holder") {
                await latch.wait()
                return (value: 0, response: response)
            }
        }
        try await waitUntil("the holder owns the gate") { await gate.snapshot().busy }
        let waiter = Task {
            try await gate.perform(priority: .chat, label: "waiter") { (value: 0, response: response) }
        }
        try await waitUntil("one waiter") { await gate.snapshot().waiting == 1 }

        waiter.cancel()
        do {
            _ = try await waiter.value
            XCTFail("a cancelled waiter must throw")
        } catch is CancellationError {
            // expected
        }
        try await waitUntil("the cancelled waiter is gone") { await gate.snapshot().waiting == 0 }

        latch.open()
        _ = try await holder.value
    }

    func testBodyErrorsAreRethrownAndReleaseTheGate() async throws {
        let gate = makeGate()
        let response = try ok()
        do {
            _ = try await gate.perform(priority: .move, label: "failing") { () async throws -> (value: Int, response: HTTPURLResponse) in
                throw URLError(.networkConnectionLost)
            }
            XCTFail("the body's error must be rethrown")
        } catch let error as URLError {
            XCTAssertEqual(error.code, .networkConnectionLost)
        }
        _ = try await gate.perform(priority: .move, label: "next") { (value: 0, response: response) }
    }
}
