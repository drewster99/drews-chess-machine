import Foundation

/// A breaker that trips at `count` events inside a time window: move POSTs
/// that got no answer, event-stream reconnects (`LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`).
///
/// After a trip the count starts over, so tripping again takes a fresh run
/// of events; each trip extends the breaker's hold.
struct LichessBotEventRateBreaker: Sendable, Equatable {
    private(set) var count: Int
    private(set) var window: TimeInterval
    private var times: [Date] = []

    init(count: Int, windowMinutes: Int) {
        self.count = count
        self.window = TimeInterval(windowMinutes * 60)
    }

    /// Records an event at `at`; returns whether it trips the breaker.
    mutating func note(at now: Date) -> Bool {
        times = times.filter { now.timeIntervalSince($0) <= window }
        times.append(now)
        guard times.count >= count else { return false }
        times.removeAll()
        return true
    }

    /// New thresholds from saved settings; the events seen so far stay.
    mutating func update(count: Int, windowMinutes: Int) {
        self.count = count
        self.window = TimeInterval(windowMinutes * 60)
    }
}

/// The lost-on-time breaker: trips at `count` time-forfeit losses among the
/// last `windowGames` games finished since the bot went online (owner
/// decision 2026-10-26: 3 of the last 30).
///
/// Counted since going online, not from the filed history: going online is
/// the operator's resume after this breaker, and a count seeded from history
/// would trip again at the first time loss after it.
struct LichessBotLostOnTimeBreaker: Sendable, Equatable {
    private(set) var count: Int
    private(set) var windowGames: Int
    /// Newest last: whether each finished game was lost on time.
    private var results: [Bool] = []
    private var counted: Set<String> = []

    init(count: Int, windowGames: Int) {
        self.count = count
        self.windowGames = windowGames
    }

    /// The time-forfeit losses among the last `windowGames` games.
    var lossesInWindow: Int {
        results.suffix(windowGames).filter { $0 }.count
    }

    /// Records a finished game, once per game; returns whether it trips the
    /// breaker. Only a time-forfeit loss can trip it.
    mutating func noteFinished(gameID: String, lostOnTime: Bool) -> Bool {
        guard counted.insert(gameID).inserted else { return false }
        results.append(lostOnTime)
        if results.count > windowGames {
            results.removeFirst(results.count - windowGames)
        }
        guard lostOnTime, lossesInWindow >= count else { return false }
        results.removeAll()
        return true
    }

    /// New thresholds from saved settings; the games seen so far stay.
    mutating func update(count: Int, windowGames: Int) {
        self.count = count
        self.windowGames = windowGames
    }
}
