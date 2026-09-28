import Foundation

/// What a request is for. The gate serves waiting requests in this order.
enum LichessBotRequestPriority: Int, Sendable, Hashable, Comparable, CaseIterable, Codable {
    /// Posting our move.
    case move = 0
    /// Claim victory / draw, resign, abort, draw and takeback responses.
    case gameCritical = 1
    /// Accepting or declining a challenge.
    case challengeResponse = 2
    /// Opening (not holding) an event or game stream.
    case streamOpen = 3
    case chat = 4
    /// Account and token checks, rating history, game exports.
    case housekeeping = 5

    static func < (lhs: LichessBotRequestPriority, rhs: LichessBotRequestPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    var label: String {
        switch self {
        case .move: return "move"
        case .gameCritical: return "gameCritical"
        case .challengeResponse: return "challengeResponse"
        case .streamOpen: return "streamOpen"
        case .chat: return "chat"
        case .housekeeping: return "housekeeping"
        }
    }
}

/// Things the gate reports, for the protocol log, the UI and the controller.
enum LichessBotGateEvent: Sendable {
    case requestStarted(id: UInt64, priority: LichessBotRequestPriority, label: String)
    case requestFinished(id: UInt64, priority: LichessBotRequestPriority, label: String, status: Int, latency: Duration, retryAfterHeader: String?)
    case requestFailed(id: UInt64, priority: LichessBotRequestPriority, label: String, error: String, latency: Duration)
    /// A 429 closed the gate for `cooldown`. `recentRequestCounts` is how many
    /// requests of each priority started in the preceding minute — the
    /// diagnosis of what spent the budget.
    case rateLimited(cooldown: Duration, triggerLabel: String, triggerPriority: LichessBotRequestPriority, recentRequestCounts: [LichessBotRequestPriority: Int])
    case cooldownEnded
    /// A second 429 inside the breaker window: the gate is closed until
    /// `reopen()` — an explicit operator action.
    case breakerTripped(rateLimitsInWindow: Int, window: Duration)
    case closed(reason: String)
    case reopened
}

enum LichessBotGateError: LocalizedError, Equatable {
    /// The gate is closed (breaker tripped, or the bot went offline).
    case closed(reason: String)
    /// The request got a 429. The gate is now cooling down; nothing is
    /// retried automatically (plan §5.4).
    case rateLimited(cooldown: Duration)

    var errorDescription: String? {
        switch self {
        case .closed(let reason):
            return "Lichess request gate is closed: \(reason)"
        case .rateLimited(let cooldown):
            return "Lichess rate limit (HTTP 429); requests paused for \(cooldown)"
        }
    }
}

/// The account-wide single-flight gate every non-stream Lichess request goes
/// through (plan §5.1–§5.4).
///
/// **One request at a time.** Lichess asks clients to "only make one request
/// at a time". Long-lived NDJSON streams are the API's intended mechanism
/// and are not held by the gate, but everything else is: moves, challenge
/// responses, chat, game actions, account and token checks, exports, and the
/// *opening* of each stream (until its response headers arrive). So at most
/// one of those is ever in flight across all games.
///
/// **Strict priority.** When the gate frees, the highest-priority eligible
/// waiter goes next (ties by arrival order). Housekeeping is not eligible
/// while any game awaits our move, and when a game's clock is low only
/// moves, game-critical actions and stream opens are eligible (plan E18).
///
/// **429 is a full stop.** Any 429 closes the gate for
/// `max(Retry-After, LichessBotRateLimit.minimumCooldown)`. Nothing is sent
/// during the cooldown — moves included, since Lichess would refuse them
/// anyway. The failed request is not retried; the caller re-derives what is
/// still needed from fresh state once the gate reopens. A second 429 inside
/// `breakerWindow` trips the breaker: the gate stays closed until `reopen()`,
/// because repeated 429s risk longer lockouts or action on the account.
///
/// **Reentrancy.** The gate is an actor, and while a request's body is
/// awaited other callers run: they join the queue, since `busy` stays set
/// until the body returns and `release()` runs.
actor LichessBotRequestGate {

    private enum State: Equatable {
        case open
        case cooldown(until: Duration)
        case closed(reason: String)
    }

    private struct Waiter {
        let id: UInt64
        let priority: LichessBotRequestPriority
        let continuation: CheckedContinuation<Void, Error>
    }

    private let time: any LichessBotTimeSource
    private let wallClock: @Sendable () -> Date
    private let onEvent: @Sendable (LichessBotGateEvent) -> Void

    private var state: State = .open
    private var busy = false
    private var waiters: [Waiter] = []
    private var nextID: UInt64 = 0
    private var wakeTask: Task<Void, Never>?

    /// Times of 429s still inside the breaker window.
    private var rateLimitTimes: [Duration] = []
    private var breakerWindow: Duration

    /// Start times and priorities of requests in the last minute, for
    /// telemetry and 429 diagnosis.
    private var recentStarts: [(time: Duration, priority: LichessBotRequestPriority)] = []
    private static let recentWindow: Duration = .seconds(60)

    /// Games currently waiting on our move; housekeeping waits while > 0.
    private var gamesAwaitingOurMove = 0
    /// Some game's clock is below the low-clock threshold; only urgent
    /// traffic is eligible while true.
    private var lowClockUrgency = false

    init(
        time: any LichessBotTimeSource,
        breakerWindow: Duration,
        wallClock: @escaping @Sendable () -> Date = { Date() },
        onEvent: @escaping @Sendable (LichessBotGateEvent) -> Void
    ) {
        self.time = time
        self.breakerWindow = breakerWindow
        self.wallClock = wallClock
        self.onEvent = onEvent
    }

    // MARK: - Performing requests

    /// Wait for the gate, run `body`, and release the gate. Throws
    /// `LichessBotGateError.rateLimited` if the response is a 429 (after
    /// starting the cooldown), and `.closed` if the gate is or becomes
    /// closed while waiting. Any error `body` throws is rethrown.
    func perform<Value: Sendable>(
        priority: LichessBotRequestPriority,
        label: String,
        _ body: @Sendable () async throws -> (value: Value, response: HTTPURLResponse)
    ) async throws -> (value: Value, response: HTTPURLResponse) {
        try await acquire(priority: priority)
        defer { release() }

        nextID += 1
        let id = nextID
        let start = time.now()
        recordStart(priority: priority, at: start)
        onEvent(.requestStarted(id: id, priority: priority, label: label))

        let result: (value: Value, response: HTTPURLResponse)
        do {
            result = try await body()
        } catch {
            onEvent(.requestFailed(id: id, priority: priority, label: label, error: String(describing: error), latency: time.now() - start))
            throw error
        }

        let status = result.response.statusCode
        let retryAfterHeader = result.response.value(forHTTPHeaderField: "Retry-After")
        onEvent(.requestFinished(id: id, priority: priority, label: label, status: status, latency: time.now() - start, retryAfterHeader: retryAfterHeader))

        if status == 429 {
            let cooldown = beginCooldown(retryAfterHeader: retryAfterHeader, triggerLabel: label, triggerPriority: priority)
            throw LichessBotGateError.rateLimited(cooldown: cooldown)
        }
        return result
    }

    // MARK: - Eligibility inputs

    func setGamesAwaitingOurMove(_ count: Int) {
        precondition(count >= 0, "games awaiting our move cannot be negative")
        gamesAwaitingOurMove = count
        dispatchNext()
    }

    func setLowClockUrgency(_ urgent: Bool) {
        lowClockUrgency = urgent
        dispatchNext()
    }

    func setBreakerWindow(_ window: Duration) {
        breakerWindow = window
    }

    // MARK: - Opening and closing

    /// Close the gate: every waiter and every later request fails with
    /// `.closed(reason)`. Used for Go Offline and by the breaker.
    func close(reason: String) {
        if case .closed = state { return }
        state = .closed(reason: reason)
        wakeTask?.cancel()
        wakeTask = nil
        failAllWaiters(reason: reason)
        onEvent(.closed(reason: reason))
    }

    /// Reopen after `close` or a breaker trip. Clears the 429 history, since
    /// reopening is an explicit operator decision.
    func reopen() {
        guard case .closed = state else { return }
        state = .open
        rateLimitTimes.removeAll()
        onEvent(.reopened)
        dispatchNext()
    }

    // MARK: - Inspection

    struct Snapshot: Sendable, Equatable {
        enum Phase: Sendable, Equatable {
            case open
            case coolingDown(remaining: Duration)
            case closed(reason: String)
        }
        let phase: Phase
        let busy: Bool
        let waiting: Int
        /// Requests started in the last minute, per priority.
        let recentRequestCounts: [LichessBotRequestPriority: Int]
        /// Games waiting on our move (housekeeping is deferred while any are).
        let gamesAwaitingOurMove: Int
        /// Some game waiting on our move is short of time (only urgent
        /// traffic is admitted).
        let lowClockUrgency: Bool
    }

    func snapshot() -> Snapshot {
        let now = time.now()
        let phase: Snapshot.Phase
        switch state {
        case .open:
            phase = .open
        case .cooldown(let until):
            phase = until > now ? .coolingDown(remaining: until - now) : .open
        case .closed(let reason):
            phase = .closed(reason: reason)
        }
        return Snapshot(phase: phase, busy: busy, waiting: waiters.count, recentRequestCounts: recentRequestCounts(now: now), gamesAwaitingOurMove: gamesAwaitingOurMove, lowClockUrgency: lowClockUrgency)
    }

    // MARK: - Queueing

    private func acquire(priority: LichessBotRequestPriority) async throws {
        if case .closed(let reason) = state {
            throw LichessBotGateError.closed(reason: reason)
        }
        nextID += 1
        let waiterID = nextID
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                waiters.append(Waiter(id: waiterID, priority: priority, continuation: continuation))
                dispatchNext()
            }
        } onCancel: {
            Task { await self.cancelWaiter(id: waiterID) }
        }
    }

    private func cancelWaiter(id: UInt64) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        busy = false
        dispatchNext()
    }

    private func dispatchNext() {
        guard !busy else { return }
        switch state {
        case .closed(let reason):
            failAllWaiters(reason: reason)
            return
        case .cooldown(let until):
            let now = time.now()
            if now < until {
                scheduleWake(after: until - now)
                return
            }
            state = .open
            onEvent(.cooldownEnded)
        case .open:
            break
        }
        guard let index = nextEligibleWaiterIndex() else { return }
        let waiter = waiters.remove(at: index)
        busy = true
        waiter.continuation.resume()
    }

    private func isEligible(_ priority: LichessBotRequestPriority) -> Bool {
        if lowClockUrgency {
            switch priority {
            case .move, .gameCritical, .streamOpen:
                break
            case .challengeResponse, .chat, .housekeeping:
                return false
            }
        }
        if priority == .housekeeping && gamesAwaitingOurMove > 0 {
            return false
        }
        return true
    }

    private func nextEligibleWaiterIndex() -> Int? {
        var best: Int?
        for (index, waiter) in waiters.enumerated() where isEligible(waiter.priority) {
            guard let current = best else {
                best = index
                continue
            }
            let incumbent = waiters[current]
            if waiter.priority < incumbent.priority
                || (waiter.priority == incumbent.priority && waiter.id < incumbent.id) {
                best = index
            }
        }
        return best
    }

    private func failAllWaiters(reason: String) {
        let failing = waiters
        waiters.removeAll()
        for waiter in failing {
            waiter.continuation.resume(throwing: LichessBotGateError.closed(reason: reason))
        }
    }

    // MARK: - Rate limiting

    private func beginCooldown(retryAfterHeader: String?, triggerLabel: String, triggerPriority: LichessBotRequestPriority) -> Duration {
        let now = time.now()
        let retryAfter = LichessBotRateLimit.parseRetryAfter(retryAfterHeader, now: wallClock())
        let cooldown = LichessBotRateLimit.cooldown(retryAfter: retryAfter)

        rateLimitTimes = rateLimitTimes.filter { now - $0 <= breakerWindow }
        rateLimitTimes.append(now)

        onEvent(.rateLimited(cooldown: cooldown, triggerLabel: triggerLabel, triggerPriority: triggerPriority, recentRequestCounts: recentRequestCounts(now: now)))

        if rateLimitTimes.count >= 2 {
            onEvent(.breakerTripped(rateLimitsInWindow: rateLimitTimes.count, window: breakerWindow))
            close(reason: "rate-limit breaker: \(rateLimitTimes.count) HTTP 429 responses within \(breakerWindow)")
        } else {
            state = .cooldown(until: now + cooldown)
        }
        return cooldown
    }

    private func scheduleWake(after delay: Duration) {
        guard wakeTask == nil else { return }
        let time = self.time
        wakeTask = Task { [weak self] in
            do {
                try await time.sleep(for: delay)
            } catch is CancellationError {
                return
            } catch {
                self?.reportWakeFailure(error)
                return
            }
            await self?.wakeFromCooldown()
        }
    }

    private nonisolated func reportWakeFailure(_ error: Error) {
        onEvent(.requestFailed(id: 0, priority: .housekeeping, label: "gate cooldown timer", error: String(describing: error), latency: .zero))
    }

    private func wakeFromCooldown() {
        wakeTask = nil
        dispatchNext()
    }

    // MARK: - Telemetry

    private func recordStart(priority: LichessBotRequestPriority, at now: Duration) {
        recentStarts.append((time: now, priority: priority))
        recentStarts.removeAll { now - $0.time > Self.recentWindow }
    }

    private func recentRequestCounts(now: Duration) -> [LichessBotRequestPriority: Int] {
        var counts: [LichessBotRequestPriority: Int] = [:]
        for start in recentStarts where now - start.time <= Self.recentWindow {
            counts[start.priority, default: 0] += 1
        }
        return counts
    }
}
