import Foundation

/// Which of matchmaking's send limits a send obeys
/// (`LichessBotMatchmakingRateLimiter.decision(now:rate:minimumSpacing:limits:)`).
enum LichessBotMatchmakingSendLimits: Sendable, Equatable {
    /// The hourly cap in force at this time of day
    /// (`LichessBotMatchmakingSchedule`) and the spacing between sends:
    /// every automatic pass, the casual resend, and Fill Open Slots by
    /// default.
    case scheduled
    /// The spacing between sends only: Fill Open Slots with "Override time
    /// limits". Lichess's own limits, the 429 holds and the circuit breakers
    /// still apply; they live outside the rate limiter.
    case spacingOnly
}

/// What the operator asked Fill Open Slots for in its sheet.
struct LichessBotFillOpenSlotsRequest: Sendable, Equatable {
    /// How many challenges to send: one per game it tries to start. A
    /// declined challenge frees its slot but isn't replaced.
    let challengeCount: Int
    let overridesTimeLimits: Bool

    var sendLimits: LichessBotMatchmakingSendLimits {
        overridesTimeLimits ? .spacingOnly : .scheduled
    }

    /// The counts the sheet offers: one up to the slots open now. Nil when
    /// no slot is open, so there is nothing to fill.
    static func challengeCountRange(openSlots: Int) -> ClosedRange<Int>? {
        openSlots >= 1 ? 1...openSlots : nil
    }
}

/// Fill Open Slots' progress, for the Overview: idle until the first fill,
/// running while it sends, then how it ended until the next fill starts or
/// the bot goes offline.
enum LichessBotFillOpenSlotsRun: Sendable, Equatable {
    case idle
    /// `waitingUntil` is the spacing's next allowed send while the fill
    /// waits for it; nil while it is deciding or sending.
    case running(LichessBotFillOpenSlotsRequest, sent: Int, startedAt: Date, waitingUntil: Date?)
    case finished(LichessBotFillOpenSlotsRequest, sent: Int, endedAt: Date, reason: String)

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }
}
