import Foundation

/// Monotonic time plus sleeping, injectable so rate-limit and backoff logic
/// can be tested without waiting in real time.
///
/// `now()` is a duration since an arbitrary fixed origin. It is monotonic,
/// so wall-clock jumps (NTP, time-zone and DST changes) never distort a
/// cooldown or a timeout (plan E19). Wall-clock timestamps for records come
/// from `Date` separately.
protocol LichessBotTimeSource: Sendable {
    func now() -> Duration
    func sleep(for duration: Duration) async throws
}

/// The real time source: `ContinuousClock`, which keeps counting while the
/// Mac sleeps, so a cooldown that spans a sleep is still honored.
struct LichessBotSystemTimeSource: LichessBotTimeSource {
    private let origin = ContinuousClock.now

    func now() -> Duration {
        ContinuousClock.now - origin
    }

    func sleep(for duration: Duration) async throws {
        try await Task.sleep(for: duration, clock: .continuous)
    }
}

/// Exponential backoff with "equal jitter": the delay for attempt `n` is
/// uniform in `[base/2, base]`, where `base = min(cap, initial · multiplierⁿ)`.
///
/// Equal jitter rather than full jitter: full jitter can produce a delay of
/// zero, and a zero-delay retry is exactly the burst Lichess's rate limiter
/// punishes. Keeping at least half the base delay spreads retries out
/// without ever retrying immediately.
struct LichessBotBackoff: Sendable, Equatable {
    let initial: Duration
    let multiplier: Double
    let cap: Duration

    /// The base (pre-jitter) delay for the zero-based `attempt`.
    func baseDelay(attempt: Int) -> Duration {
        precondition(attempt >= 0, "backoff attempt must be non-negative")
        let initialSeconds = Self.seconds(initial)
        let capSeconds = Self.seconds(cap)
        let grown = initialSeconds * pow(multiplier, Double(attempt))
        return .seconds(min(capSeconds, grown))
    }

    /// The delay for `attempt`, given `unitRandom` in `[0, 1]` (injected so
    /// the schedule is testable; production passes `Double.random(in: 0...1)`).
    func delay(attempt: Int, unitRandom: Double) -> Duration {
        precondition((0...1).contains(unitRandom), "unitRandom must be in [0, 1]")
        let base = Self.seconds(baseDelay(attempt: attempt))
        return .seconds(base / 2 + (base / 2) * unitRandom)
    }

    static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
