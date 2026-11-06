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

/// What a fill's count counts, and so when it is done.
enum LichessBotFillOpenSlotsTarget: Sendable, Equatable {
    /// Send this many challenges, into the slots open now. A declined
    /// challenge frees its slot but isn't replaced; the fill ends at the
    /// first outcome that isn't a send or a spacing wait.
    case challengesSent
    /// Keep sending until this many of the fill's challenges are accepted:
    /// a declined, canceled or refused one is replaced, a full house or a
    /// time limit is waited out. The count may exceed the slots open now.
    case gamesStarted
}

/// What the operator asked Fill Open Slots for in its sheet.
struct LichessBotFillOpenSlotsRequest: Sendable, Equatable {
    /// Challenges to send, or games to start, by `target`.
    let count: Int
    let target: LichessBotFillOpenSlotsTarget
    let overridesTimeLimits: Bool

    var sendLimits: LichessBotMatchmakingSendLimits {
        overridesTimeLimits ? .spacingOnly : .scheduled
    }

    /// The most games one fill may be asked to start. Lichess counts every
    /// bot game against DCM's daily bot-game limit (`LichessBotBotGameBudget`),
    /// and a fill that runs that far is better split into fills the operator
    /// sees end.
    static let maximumGamesToStart = 100

    /// The counts the sheet offers for `target`. Sending challenges: one up
    /// to the slots open now, nil when none is open. Starting games: one up
    /// to `maximumGamesToStart`, whatever is open now, since the fill waits
    /// for slots.
    static func countRange(target: LichessBotFillOpenSlotsTarget, openSlots: Int) -> ClosedRange<Int>? {
        switch target {
        case .challengesSent:
            return challengeCountRange(openSlots: openSlots)
        case .gamesStarted:
            return 1...maximumGamesToStart
        }
    }

    /// The challenge counts the sheet offers: one up to the slots open now.
    /// Nil when no slot is open, so there is nothing to fill.
    static func challengeCountRange(openSlots: Int) -> ClosedRange<Int>? {
        openSlots >= 1 ? 1...openSlots : nil
    }
}

/// A fill's challenges, followed from the send to Lichess's answer. Fed by
/// the controller's one send path and its one outcome path, so a challenge
/// answered before the fill looks again is still counted.
struct LichessBotFillOpenSlotsProgress: Sendable, Equatable {
    /// Challenges the fill created on Lichess.
    private(set) var sent = 0
    /// Of those, the ones accepted: each is a game starting.
    private(set) var accepted = 0
    /// Of those, the ones declined, canceled, expired or refused.
    private(set) var unanswered = 0
    /// The fill's challenges Lichess hasn't answered yet.
    private(set) var pendingChallengeIDs: Set<String> = []

    mutating func noteSent(challengeID: String) {
        guard pendingChallengeIDs.insert(challengeID).inserted else { return }
        sent += 1
    }

    /// Lichess's answer to one of the fill's challenges. Returns whether the
    /// challenge was the fill's; any other is left alone.
    @discardableResult
    mutating func noteAnswered(challengeID: String, accepted wasAccepted: Bool) -> Bool {
        guard pendingChallengeIDs.remove(challengeID) != nil else { return false }
        if wasAccepted {
            accepted += 1
        } else {
            unanswered += 1
        }
        return true
    }

    /// The fill has what it asked for.
    func isComplete(_ request: LichessBotFillOpenSlotsRequest) -> Bool {
        switch request.target {
        case .challengesSent:
            return sent >= request.count
        case .gamesStarted:
            return accepted >= request.count
        }
    }

    /// Another send could still be needed: starting games, the challenges
    /// pending now could all be accepted, so it sends only while accepted
    /// plus pending is short of the count.
    func needsAnotherSend(_ request: LichessBotFillOpenSlotsRequest) -> Bool {
        switch request.target {
        case .challengesSent:
            return sent < request.count
        case .gamesStarted:
            return accepted + pendingChallengeIDs.count < request.count
        }
    }
}

/// What a running fill is waiting for, for the Overview.
enum LichessBotFillOpenSlotsWait: Sendable, Equatable {
    /// The spacing between sends, the hourly cap or the off-peak schedule,
    /// until then.
    case sendLimit(until: Date)
    /// Every slot is taken (starting games).
    case freeSlot
    /// Its pending challenges could still bring it to the count (starting
    /// games).
    case pendingAnswers
    /// A pass found no bot to challenge or its send failed; it tries again
    /// then (starting games).
    case retry(until: Date, after: String)
}

/// Fill Open Slots' progress, for the Overview: idle until the first fill,
/// running while it sends, then how it ended until the next fill starts or
/// the bot goes offline.
enum LichessBotFillOpenSlotsRun: Sendable, Equatable {
    case idle
    /// `waiting` is nil while it is deciding or sending.
    case running(LichessBotFillOpenSlotsRequest, progress: LichessBotFillOpenSlotsProgress, startedAt: Date, waiting: LichessBotFillOpenSlotsWait?)
    case finished(LichessBotFillOpenSlotsRequest, progress: LichessBotFillOpenSlotsProgress, endedAt: Date, reason: String)

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }
}
