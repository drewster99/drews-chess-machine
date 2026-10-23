import Foundation

/// Lichess refusing a request because DCM itself has played its daily
/// allowance of games against other bots.
///
/// lila (`modules/bot/src/main/BotLimit.scala`) answers a challenge DCM
/// sends to a bot, and DCM's acceptance of a bot's challenge, with
/// `Limited("bot.vsBot.day", "You played 100 games against other bots
/// today, please wait before challenging another bot.", until)` while DCM's
/// own count is at the limit. The challenge controller writes that as
/// **HTTP 429** (`JsonLimited`) with the body
/// `{"error": <message>, "ratelimit": {"key": "bot.vsBot.day", "seconds": <until − now>}}`
/// (`CtrlErrors.scala`), and no `Retry-After` header.
///
/// It is a 429, but not a request-rate limit: sending less often changes
/// nothing, and nothing but bot-vs-bot games is refused. Treated as one
/// (the 2026-10-14 incident), two of them within an hour tripped the request
/// gate's breaker and took the whole bot offline, human games included, for
/// nine hours. So the gate tells it apart (`LichessBotTooManyRequestsCause`)
/// and it stops only bot games, until Lichess's time.
///
/// The challenged bot being at *its* limit is a different answer: HTTP 400
/// with the same `ratelimit.key` and a message naming that bot
/// (`LichessBotBotLimitRefusal`).
struct LichessBotOwnBotGameLimitRefusal: Sendable, Equatable {
    /// lila's `BotLimit.limitKey`.
    static let rateLimitKey = "bot.vsBot.day"

    /// Lichess's `error` text, when the body had one.
    let message: String?
    /// `ratelimit.seconds`: whole seconds from Lichess's clock at the
    /// response to the end of the limit. Nil when the body carried no
    /// `ratelimit` object with lila's key.
    let secondsUntilClear: Int?
    /// When the response arrived, by DCM's clock; `secondsUntilClear` counts
    /// from here.
    let respondedAt: Date

    /// The refusal `body` represents, or nil when the response is anything
    /// else (another status, a request-rate 429, a body that isn't
    /// Lichess's JSON).
    static func parse(status: Int, body: Data, respondedAt: Date) -> LichessBotOwnBotGameLimitRefusal? {
        guard status == 429, !body.isEmpty else { return nil }
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: body)
        } catch {
            // Not JSON (a proxy's page): not lila's answer, so a plain 429.
            return nil
        }
        guard let fields = object as? [String: Any] else { return nil }
        let message = fields["error"] as? String
        let rateLimit = fields["ratelimit"] as? [String: Any]
        let keyMatches = (rateLimit?["key"] as? String) == rateLimitKey
        guard keyMatches || message.map(isOwnLimitMessage) == true else { return nil }
        // Seconds belong to this limit only when the key says so.
        let seconds = keyMatches ? (rateLimit?["seconds"] as? NSNumber).map { max(0, $0.intValue) } : nil
        return LichessBotOwnBotGameLimitRefusal(message: message, secondsUntilClear: seconds, respondedAt: respondedAt)
    }

    /// lila's wording for the challenger (or accepter) being at its own
    /// limit. Anchored at the start: the opponent's refusal names the
    /// opponent there instead of "You".
    static func isOwnLimitMessage(_ message: String) -> Bool {
        let pattern = #/^You played \d+ games against other bots today/#
        return message.trimmingCharacters(in: .whitespacesAndNewlines).prefixMatch(of: pattern) != nil
    }
}

/// What a 429 response means to the request gate.
enum LichessBotTooManyRequestsCause: Sendable, Equatable {
    /// Lichess's request-rate limiting: the gate cools down, and a second
    /// one inside the breaker window closes it.
    case requestRate
    /// DCM at its own daily bot-game limit: an ordinary refusal for the
    /// gate, which neither cools down nor counts it.
    case ownBotGameLimit(LichessBotOwnBotGameLimitRefusal)
}

/// When DCM's own daily bot-game limit ends, as recorded from a refusal.
struct LichessBotOwnBotGameLimit: Sendable, Equatable {
    enum EndSource: Sendable, Equatable {
        /// From Lichess's `ratelimit.seconds`.
        case lichess
        /// Lichess gave no time; DCM asks again (with the next bot
        /// challenge) after `recheckInterval`.
        case recheck
    }

    let until: Date
    let endSource: EndSource

    /// lila floors both its `until` and its `now` to whole seconds before
    /// subtracting, so the true end can be up to a second after
    /// `respondedAt + seconds`. Adding the second keeps DCM from asking
    /// a moment early.
    static let roundingMargin: TimeInterval = 1

    /// How long DCM waits before asking Lichess again when a refusal gives
    /// no end time. DCM keeps no copy of Lichess's count to guess the end
    /// from (`LichessBotBotGameBudget`). Asking again costs one refused
    /// challenge, which no longer affects the request gate; an hour bounds
    /// both that cost and how long bot games wait past a limit that has
    /// already lifted.
    static let recheckInterval: TimeInterval = 3600

    init(until: Date, endSource: EndSource) {
        self.until = until
        self.endSource = endSource
    }

    init(_ refusal: LichessBotOwnBotGameLimitRefusal) {
        if let seconds = refusal.secondsUntilClear {
            self.init(until: refusal.respondedAt.addingTimeInterval(TimeInterval(seconds) + Self.roundingMargin), endSource: .lichess)
        } else {
            self.init(until: refusal.respondedAt.addingTimeInterval(Self.recheckInterval), endSource: .recheck)
        }
    }

    /// The Overview's and the pass status's wording.
    static func pausedText(until: Date, now: Date, calendar: Calendar = .current) -> String {
        "Bot games paused: Lichess daily limit until \(LichessBotBotGameBudget.clearTimeText(until, now: now, calendar: calendar))"
    }
}
