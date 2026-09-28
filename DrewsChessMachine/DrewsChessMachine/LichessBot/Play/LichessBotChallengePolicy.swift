import Foundation

/// What to do with an incoming challenge.
enum LichessBotChallengeDecision: Equatable, Sendable {
    case accept
    /// Decline with the most specific reason Lichess accepts. `rule` names
    /// the rule that fired, for the protocol log.
    case decline(LichessBotDeclineReason, rule: String)
    /// Send nothing at all: the challenge is ours, or answering would spend
    /// a request the budget doesn't allow. It expires on Lichess's side.
    case ignore(rule: String)
}

/// The live state a challenge decision depends on, captured by the caller.
struct LichessBotChallengeContext: Sendable, Equatable {
    /// The bot is Online: not Draining, not cooling down from a 429.
    var acceptingNewGames: Bool
    /// A model generation is ready to play (plan E15).
    var modelReady: Bool
    var activeGames: Int
    /// Active games per opponent id.
    var activeGamesByOpponent: [String: Int]
    /// Games started today (local day), in total and per opponent id.
    var gamesToday: Int
    var gamesTodayByOpponent: [String: Int]
    /// Challenge responses (accept or decline) sent in the last minute.
    var challengeResponsesInLastMinute: Int
}

/// The challenge policy (plan §7): a pure function of the challenge, the
/// settings and the live context. Every rule maps to the most specific
/// decline reason Lichess accepts, and every decision names its rule.
enum LichessBotChallengePolicy {

    static func decide(
        _ challenge: LichessBotChallenge,
        compat: LichessBotCompat?,
        settings: LichessBotChallengeSettings,
        context: LichessBotChallengeContext
    ) -> LichessBotChallengeDecision {
        // Not ours to answer.
        if challenge.direction?.known == .outgoing {
            return .ignore(rule: "own outgoing challenge")
        }
        if context.challengeResponsesInLastMinute >= settings.challengeResponseBudgetPerMinute {
            return .ignore(rule: "challenge-response budget of \(settings.challengeResponseBudgetPerMinute)/min reached")
        }

        // Bot state.
        if !context.acceptingNewGames {
            return .decline(.later, rule: "not accepting new games")
        }
        if !context.modelReady {
            return .decline(.later, rule: "model not ready")
        }

        // What DCM cannot play at all.
        if challenge.variant.key.known != .standard {
            return .decline(.standard, rule: "variant \(challenge.variant.key)")
        }
        if let fen = challenge.initialFen, !LichessBotPositionTracker.isStandardStart(fen) {
            return .decline(.standard, rule: "from-position")
        }
        if compat?.bot == false {
            return .decline(.timeControl, rule: "not playable through the Bot API")
        }
        guard challenge.timeControl.type.known == .clock,
              let limit = challenge.timeControl.limit,
              let increment = challenge.timeControl.increment else {
            return .decline(.timeControl, rule: "time control \(challenge.timeControl.type) is not a clock")
        }

        // Speed and clock.
        guard let speed = challenge.speed.known else {
            return .decline(.timeControl, rule: "unknown speed \(challenge.speed)")
        }
        if !settings.allowedSpeeds.contains(speed) {
            let fasterThanAllAllowed = settings.allowedSpeeds.allSatisfy { speed < $0 }
            return .decline(fasterThanAllAllowed ? .tooFast : .tooSlow, rule: "speed \(speed.rawValue) not allowed")
        }
        if limit.value < settings.minimumInitialSeconds {
            return .decline(.tooFast, rule: "clock \(limit.value)s below minimum")
        }
        if limit.value > settings.maximumInitialSeconds {
            return .decline(.tooSlow, rule: "clock \(limit.value)s above maximum")
        }
        if increment.value < settings.minimumIncrementSeconds || increment.value > settings.maximumIncrementSeconds {
            return .decline(.timeControl, rule: "increment \(increment.value)s out of range")
        }

        // Rated or casual. Lichess's `casual` reason means "I only accept
        // casual games", and `rated` the reverse.
        if challenge.rated && !settings.acceptRated {
            return .decline(.casual, rule: "rated not accepted")
        }
        if !challenge.rated && !settings.acceptCasual {
            return .decline(.rated, rule: "casual not accepted")
        }

        // Opponent.
        let challenger = challenge.challenger
        let isBot = challenger.title == "BOT"
        if isBot && !settings.acceptBots {
            return .decline(.noBot, rule: "bots not accepted")
        }
        if !isBot && !settings.acceptHumans {
            return .decline(.onlyBot, rule: "humans not accepted")
        }
        if settings.blockedUserIDs.contains(challenger.id) {
            return .decline(.generic, rule: "blocked opponent")
        }
        if let rating = challenger.rating,
           rating < settings.minimumOpponentRating || rating > settings.maximumOpponentRating {
            return .decline(.generic, rule: "opponent rating \(rating) out of range")
        }
        if challenger.provisional == true && !settings.acceptProvisionalOpponents {
            return .decline(.generic, rule: "provisional opponent")
        }
        if challenge.rematchOf != nil && !settings.acceptRematches {
            return .decline(.generic, rule: "rematches not accepted")
        }

        // Capacity.
        if context.activeGames >= settings.maxConcurrentGames {
            return .decline(.later, rule: "at the concurrent-game limit")
        }
        if isBot && settings.maxConcurrentGames - context.activeGames <= settings.gamesReservedForHumans {
            return .decline(.later, rule: "remaining slots are reserved for humans")
        }
        if (context.activeGamesByOpponent[challenger.id] ?? 0) >= settings.maxSimultaneousGamesPerOpponent {
            return .decline(.later, rule: "already playing this opponent")
        }
        if context.gamesToday >= settings.maxGamesPerDay {
            return .decline(.later, rule: "daily game limit reached")
        }
        let todayAgainst = context.gamesTodayByOpponent[challenger.id] ?? 0
        if todayAgainst >= settings.maxGamesPerOpponentPerDay {
            return .decline(.later, rule: "daily limit against this opponent reached")
        }
        // Lichess itself limits each bot to a fixed number of games against
        // other bots per rolling day, and refuses challenges beyond it
        // (observed live); DCM keeps no copy of that rule.

        return .accept
    }
}
