import Foundation

/// A snapshot of one position transition, holding exactly the facts the
/// event triggers read. Built by the game session from the board and the
/// value-head readings it already has; kept separate so trigger detection is
/// pure and testable.
///
/// "Before" values describe the last position this client acted on; "now"
/// values describe the position just reached. Win percentages come from the
/// value head and only change on our own moves, so they may repeat across
/// opponent moves — that is intentional (the plan measures swings "between
/// our moves").
struct LichessBotChatEventContext: Sendable, Equatable {
    /// Game-total plies played so far (the position just reached).
    let ply: Int
    /// The full-move number the game has reached.
    let moveNumber: Int

    let ourQueensBefore: Int
    let ourQueensNow: Int
    let theirQueensBefore: Int
    let theirQueensNow: Int

    /// A side is in check in the position just reached.
    let inCheckNow: Bool
    /// Any newly applied move was a promotion.
    let promotionJustPlayed: Bool

    /// Our win chance as a percentage (0…100) from our latest value reading,
    /// and from the reading before it; nil when we have no reading yet.
    let ourWinPercent: Int?
    let previousOurWinPercent: Int?
    /// A new value reading (one of our moves) was added since the previous
    /// evaluation. Win-chance triggers fire only then, so they don't repeat
    /// across the opponent's moves while the reading is unchanged.
    let newReadingThisStep: Bool

    let opponentName: String?
}

/// Decides which event triggers a context satisfies, purely from the board
/// and win-chance facts — independent of whether a rule is enabled, its
/// odds, or the spam guards (those are applied by the caller). Also builds
/// the placeholder values a fired message uses.
enum LichessBotChatEventEvaluator {

    /// The triggers whose condition this context meets, honoring each rule's
    /// numeric threshold. Order is by trigger priority, highest first, so the
    /// caller can take the most salient that passes its guards.
    static func matchingTriggers(_ context: LichessBotChatEventContext, rules: [LichessBotChatEventRule]) -> [LichessBotChatEventTrigger] {
        let thresholds = Dictionary(rules.map { ($0.trigger, $0.threshold) }, uniquingKeysWith: { first, _ in first })
        var matched: [LichessBotChatEventTrigger] = []
        for trigger in LichessBotChatEventTrigger.allCases {
            let threshold = thresholds[trigger] ?? trigger.defaultThreshold
            if condition(trigger, threshold: threshold, context: context) {
                matched.append(trigger)
            }
        }
        return matched.sorted { $0.priority > $1.priority }
    }

    private static func condition(_ trigger: LichessBotChatEventTrigger, threshold: Int, context c: LichessBotChatEventContext) -> Bool {
        switch trigger {
        case .weLostQueen:
            return c.ourQueensNow < c.ourQueensBefore
        case .weWonQueen:
            return c.theirQueensNow < c.theirQueensBefore
        case .earlyCheck:
            return c.inCheckNow && c.ply <= threshold
        case .winningBig:
            guard c.newReadingThisStep, let now = c.ourWinPercent else { return false }
            // Fire only on the move that crosses the line, not every move
            // after it, so a winning position doesn't repeat the message.
            let was = c.previousOurWinPercent ?? -1
            return now >= threshold && was < threshold
        case .losingBig:
            guard c.newReadingThisStep, let now = c.ourWinPercent else { return false }
            let was = c.previousOurWinPercent ?? 101
            return now <= threshold && was > threshold
        case .bigSwing:
            guard c.newReadingThisStep, let now = c.ourWinPercent, let was = c.previousOurWinPercent else { return false }
            return abs(now - was) >= threshold
        case .promotion:
            return c.promotionJustPlayed
        case .passedMove:
            // The move the game just reached is past the number: fire on the
            // crossing move only.
            return c.moveNumber >= threshold && (c.moveNumber - 1) < threshold
        }
    }

    /// Placeholder values for an event message. `{winPct}` is our current win
    /// chance; `{move}` the full-move number; `{ply}` the ply; `{opponent}`
    /// the opponent's display name when known.
    static func templateValues(_ context: LichessBotChatEventContext) -> [String: String] {
        var values = [
            "move": "\(context.moveNumber)",
            "ply": "\(context.ply)",
        ]
        if let pct = context.ourWinPercent {
            values["winPct"] = "\(pct)"
        }
        if let opponent = context.opponentName {
            values["opponent"] = opponent
        }
        return values
    }
}

/// The per-game spam-guard state shared by all event rules: how many
/// messages have gone out, and when (ply and wall time) the last one did,
/// plus which triggers have already fired. Pure: the session owns one and
/// asks it before sending, then records a send.
///
/// This does not hold the clock floor or the per-rule odds — the session
/// checks the clock and rolls the chance — only the counting and spacing
/// limits, so those are tested here in isolation.
struct LichessBotChatEventGuard: Sendable, Equatable {
    private(set) var sent = 0
    private var lastSentPly: Int?
    private var lastSentAt: Duration?
    private var firedTriggers: Set<LichessBotChatEventTrigger> = []

    init() {}

    /// Whether a message for `trigger` may be sent at `ply` and `now` (the
    /// session's monotonic clock) under `settings`: under the per-game
    /// maximum, not a repeat of this trigger, and far enough from the last
    /// message in both plies and seconds.
    func admits(trigger: LichessBotChatEventTrigger, ply: Int, now: Duration, settings: LichessBotChatEventSettings) -> Bool {
        if sent >= settings.maxMessagesPerGame { return false }
        if firedTriggers.contains(trigger) { return false }
        if let lastPly = lastSentPly, ply - lastPly < settings.minPliesBetweenMessages { return false }
        if let lastAt = lastSentAt, (now - lastAt) < .seconds(settings.minSecondsBetweenMessages) { return false }
        return true
    }

    /// Record that a message for `trigger` was sent.
    mutating func recordSend(trigger: LichessBotChatEventTrigger, ply: Int, now: Duration) {
        sent += 1
        lastSentPly = ply
        lastSentAt = now
        firedTriggers.insert(trigger)
    }
}
