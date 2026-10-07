import Foundation

/// Matchmaking's decisions (plan §7.3 B): when a pass may run, how many
/// slots it may fill, which rating window applies, and which online bot to
/// challenge. Pure functions of what the controller passes in — the clock
/// and the random source included — so every rule is unit-tested; the
/// controller does the requests and the logging.
enum LichessBotMatchmaking {

    // MARK: - Cadence

    /// While matchmaking is on, the online-bots list is refetched once it is
    /// this old (the Challenge sheet's own refresh runs only while the sheet
    /// is open). Lichess rebuilds the list every few seconds, but a bot that
    /// left since is caught by the online check before each send, so a few
    /// minutes is fresh enough and keeps the unauthenticated fetch rare.
    static let onlineBotsRefreshInterval: TimeInterval = 180
    /// A failed online-bots fetch is not retried sooner than this.
    static let onlineBotsRetryInterval: TimeInterval = 60
    /// The least time between two matchmaking sends, so a burst of free
    /// slots is filled gradually rather than all at once.
    static let minimumSendSpacing: TimeInterval = 20
    /// After an automatic pass that found nobody to challenge (or failed),
    /// the next waits this long: the online list changes slowly, and a pass
    /// every poll would only repeat the same log lines.
    static let retryAfterUnproductivePass: TimeInterval = 60

    // MARK: - When a pass may run

    /// What decides whether any matchmaking send may happen now.
    struct PassConditions: Sendable, Equatable {
        /// Online, and not Draining (which a 429 hold also shows as).
        var isOnline: Bool
        /// A post-429 hold is in force.
        var rateLimitHoldActive: Bool
        /// The request gate is open (not cooling down after a 429, not
        /// closed by the breaker).
        var gateOpen: Bool
        /// The token carries `challenge:write`.
        var hasChallengeScope: Bool
        /// "Play one game" is in force: that run takes a single game, so
        /// matchmaking must not fill other slots.
        var playOneGameActive: Bool
        /// The operator's challenge queue still has entries to send; its
        /// picks go first.
        var queueHasEntriesToSend: Bool
        /// Lichess' bot-game window as DCM's records show it; nil until
        /// those records have loaded.
        var botGameWindow: LichessBotBotGameWindow?
        /// The reserves matchmaking leaves short of Lichess' bot-game limit.
        var challengeSettings: LichessBotChallengeSettings
    }

    /// Why no pass may run now, or nil if one may.
    static func passBlockedReason(_ conditions: PassConditions) -> String? {
        if !conditions.isOnline {
            return "the bot is not Online"
        }
        if conditions.rateLimitHoldActive {
            return "rate-limit hold after a 429"
        }
        if !conditions.gateOpen {
            return "the request gate is not open"
        }
        if !conditions.hasChallengeScope {
            return "the token lacks challenge:write"
        }
        if conditions.playOneGameActive {
            return "Play One Game is in force"
        }
        if conditions.queueHasEntriesToSend {
            return "the challenge queue goes first"
        }
        guard let window = conditions.botGameWindow else {
            return "DCM's game records have not loaded"
        }
        return LichessBotBotGameBudget.blockedReason(window: window, sender: .matchmaking, settings: conditions.challengeSettings)
    }

    // MARK: - Slots

    /// How many challenges matchmaking may add now. `committed` counts
    /// everything that holds a slot: games in progress, accepted challenges
    /// awaiting their game, our challenges waiting for an answer, and sends
    /// under way. The slots reserved for humans are never used, so a human
    /// can always challenge DCM.
    static func openSlots(maxConcurrentGames: Int, gamesReservedForHumans: Int, fillMode: LichessBotMatchmakingSettings.FillMode, committed: Int) -> Int {
        let capacity = maxConcurrentGames - gamesReservedForHumans
        switch fillMode {
        case .everyFreeSlot:
            return max(0, capacity - committed)
        case .onlyWhenIdle:
            return committed == 0 && capacity > 0 ? 1 : 0
        }
    }

    // MARK: - Rating window

    enum RatingBasis: Sendable, Equatable {
        /// Offsets around DCM's own rating at the speed.
        case relative(ourRating: Int)
        /// DCM has no established rating at the speed: the absolute bounds.
        case absolute
    }

    struct RatingBounds: Sendable, Equatable {
        let minimum: Int
        let maximum: Int
        let basis: RatingBasis

        func contains(_ rating: Int) -> Bool {
            rating >= minimum && rating <= maximum
        }

        /// For logs: the bounds and where they came from.
        func description(speed: LichessBotSpeed) -> String {
            switch basis {
            case .relative(let ourRating):
                return "\(minimum)–\(maximum) (DCM's \(speed.rawValue) \(ourRating) + offsets)"
            case .absolute:
                return "\(minimum)–\(maximum) (absolute: DCM has no established \(speed.rawValue) rating)"
            }
        }
    }

    /// The opponent rating window at `speed`. DCM's rating counts only when
    /// it is established: Lichess starts every account at a provisional
    /// default it hasn't earned, and a provisional rating still swings by
    /// hundreds of points, so a window around it would be arbitrary.
    static func ratingBounds(settings: LichessBotMatchmakingSettings, ourPerfs: [String: LichessBotPerfRating]?, speed: LichessBotSpeed) -> RatingBounds {
        if let perf = ourPerfs?[speed.rawValue], let rating = perf.rating, perf.prov != true {
            return RatingBounds(minimum: rating + settings.minimumRatingOffset, maximum: rating + settings.maximumRatingOffset, basis: .relative(ourRating: rating))
        }
        return RatingBounds(minimum: settings.minimumRatingWithoutOwnRating, maximum: settings.maximumRatingWithoutOwnRating, basis: .absolute)
    }

    // MARK: - Candidates

    /// Why a bot is not a candidate, one per rule (plan §7.3 B, "Picking").
    enum Exclusion: String, Sendable, Hashable, CaseIterable, Comparable {
        case ourselves = "DCM itself"
        case noRating = "no rating at the speed"
        case provisional = "provisional at the speed"
        case outsideRatingWindow = "outside the rating window"
        case blocked = "blocked"
        case alreadyEngaged = "already playing, challenged or queued"
        case atBotLimit = "at its bot-game limit"
        case declineCooldown = "declined DCM recently"
        case refusesBots = "refuses bot games"
        case declinedThisKind = "declined this kind of challenge"
        case dailyOpponentLimit = "played DCM enough today"

        /// Rule order, the order `exclusion(of:)` checks them in.
        static func < (lhs: Exclusion, rhs: Exclusion) -> Bool {
            lhs.ruleOrder < rhs.ruleOrder
        }

        private var ruleOrder: Int {
            switch self {
            case .ourselves: return 0
            case .noRating: return 1
            case .provisional: return 2
            case .outsideRatingWindow: return 3
            case .blocked: return 4
            case .alreadyEngaged: return 5
            case .atBotLimit: return 6
            case .declineCooldown: return 7
            case .refusesBots: return 8
            case .declinedThisKind: return 9
            case .dailyOpponentLimit: return 10
            }
        }
    }

    /// What the candidate rules consult besides the bot itself.
    struct CandidateContext: Sendable, Equatable {
        /// Lowercased.
        var ourAccountID: String
        /// Lowercased ids.
        var blockedUserIDs: Set<String>
        /// Lowercased ids with a game in progress, a challenge waiting for
        /// an answer, or an entry in the challenge queue.
        var engagedUserIDs: Set<String>
        var notes: LichessBotPlayerNotes?
        /// Games started today, by lowercased opponent id.
        var gamesTodayByOpponent: [String: Int]
        var maxGamesPerOpponentPerDay: Int
        /// Declines still in force and each bot's latest contact.
        var opponentHistory: LichessBotOpponentHistory
        var now: Date
    }

    /// The first rule that excludes a challenge to `bot` on `terms`, or nil
    /// if it is a candidate.
    static func exclusion(of bot: LichessBotUserSummary, terms: LichessBotChallengeTerms, bounds: RatingBounds, context: CandidateContext) -> Exclusion? {
        let speed = terms.speed
        let id = bot.id.lowercased()
        if id == context.ourAccountID {
            return .ourselves
        }
        guard let perf = bot.rating(speed.rawValue), let rating = perf.rating else {
            return .noRating
        }
        if perf.prov == true {
            return .provisional
        }
        if !bounds.contains(rating) {
            return .outsideRatingWindow
        }
        if context.blockedUserIDs.contains(id) {
            return .blocked
        }
        if context.engagedUserIDs.contains(id) {
            return .alreadyEngaged
        }
        if context.notes?.limitUntil(id, now: context.now) != nil {
            return .atBotLimit
        }
        if context.notes?.declineCooldownEnds(id, now: context.now) != nil {
            return .declineCooldown
        }
        if let block = context.opponentHistory.blockingDecline(of: id, terms: terms) {
            return block.scope == .everyChallenge ? .refusesBots : .declinedThisKind
        }
        if context.gamesTodayByOpponent[id, default: 0] >= context.maxGamesPerOpponentPerDay {
            return .dailyOpponentLimit
        }
        return nil
    }

    struct Pick: Sendable, Equatable {
        let bot: LichessBotUserSummary
        /// The bot's rating at the clock's speed.
        let rating: Int
        let clock: LichessBotClockChoice
        let bounds: RatingBounds
        /// Chosen among favorites ("prefer favorites" and one fitted).
        let fromFavorites: Bool
        /// Bots that fitted every rule.
        let candidateCount: Int
        /// How "prefer bots not contacted recently" chose, or nil when it
        /// is off.
        let recency: RecencyChoice?
        let exclusions: [Exclusion: Int]
    }

    enum PickResult: Sendable, Equatable {
        case picked(Pick)
        case noCandidate(clock: LichessBotClockChoice, bounds: RatingBounds, listed: Int, exclusions: [Exclusion: Int])
        /// The settings name no time control (validation prevents this).
        case noTimeControl
    }

    /// How the recency preference narrowed the pool.
    enum RecencyChoice: Sendable, Equatable {
        /// Among `count` bots with no contact in the window.
        case notContactedRecently(count: Int)
        /// Every candidate was contacted in the window: the one contacted
        /// longest ago, at `lastContact`.
        case contactedLongestAgo(lastContact: Date)
    }

    /// The bots `pool` narrows to under "prefer bots not contacted
    /// recently": those with no challenge either way and no game within
    /// `recentContactHours` (never contacted counts as not recent); with
    /// none, those whose latest contact is the oldest.
    static func preferNotRecentlyContacted<Candidate>(_ pool: [Candidate], id: (Candidate) -> String,
                                                     settings: LichessBotMatchmakingSettings, context: CandidateContext) -> (pool: [Candidate], choice: RecencyChoice?) {
        guard settings.preferNotRecentlyContacted, !pool.isEmpty else { return (pool, nil) }
        let cutoff = context.now.addingTimeInterval(-TimeInterval(settings.recentContactHours) * 3600)
        let latest = pool.map { context.opponentHistory.contact(id($0))?.latest }
        let fresh = zip(pool, latest).filter { $0.1.map { $0 < cutoff } ?? true }.map(\.0)
        if !fresh.isEmpty {
            return (fresh, .notContactedRecently(count: fresh.count))
        }
        // None is fresh, so every one has a contact inside the window and
        // `pool` is not empty: there is an oldest.
        guard let oldest = latest.compactMap({ $0 }).min() else {
            preconditionFailure("a non-empty pool with no fresh candidate has a contact for each")
        }
        return (zip(pool, latest).filter { $0.1 == oldest }.map(\.0), .contactedLongestAgo(lastContact: oldest))
    }

    /// Choose a time control uniformly from the configured ones, then a bot
    /// uniformly among those that fit every rule — among fitting favorites
    /// first when "prefer favorites" is on, then among those not contacted
    /// recently when that preference is on.
    static func pick<Generator: RandomNumberGenerator>(
        from bots: [LichessBotUserSummary],
        settings: LichessBotMatchmakingSettings,
        ourPerfs: [String: LichessBotPerfRating]?,
        context: CandidateContext,
        using generator: inout Generator
    ) -> PickResult {
        // A fixed order, so a seeded generator picks the same clock every run.
        let clocks = LichessBotClockChoice.allCases.filter { settings.timeControls.contains($0) }
        guard let clock = clocks.randomElement(using: &generator) else { return .noTimeControl }
        let speed = clock.speed
        let terms = LichessBotChallengeTerms(limitSeconds: clock.seconds.limit, incrementSeconds: clock.seconds.increment, rated: settings.rated)
        let bounds = ratingBounds(settings: settings, ourPerfs: ourPerfs, speed: speed)
        var seen: Set<String> = []
        var candidates: [(bot: LichessBotUserSummary, rating: Int)] = []
        var exclusions: [Exclusion: Int] = [:]
        for bot in bots where seen.insert(bot.id.lowercased()).inserted {
            if let reason = exclusion(of: bot, terms: terms, bounds: bounds, context: context) {
                exclusions[reason, default: 0] += 1
            } else if let rating = bot.rating(speed.rawValue)?.rating {
                candidates.append((bot, rating))
            } else {
                // `exclusion(of:)` excludes a bot without a rating first.
                exclusions[.noRating, default: 0] += 1
            }
        }
        let favorites = settings.preferFavorites
            ? candidates.filter { context.notes?.isFavorite($0.bot.id) == true }
            : []
        let (pool, recency) = preferNotRecentlyContacted(favorites.isEmpty ? candidates : favorites, id: { $0.bot.id }, settings: settings, context: context)
        guard let chosen = pool.randomElement(using: &generator) else {
            return .noCandidate(clock: clock, bounds: bounds, listed: seen.count, exclusions: exclusions)
        }
        return .picked(Pick(bot: chosen.bot, rating: chosen.rating, clock: clock, bounds: bounds, fromFavorites: !favorites.isEmpty, candidateCount: candidates.count, recency: recency, exclusions: exclusions))
    }

    /// Exclusion counts for a log line, in rule order.
    static func describe(_ exclusions: [Exclusion: Int]) -> String {
        guard !exclusions.isEmpty else { return "none excluded" }
        return exclusions.sorted { $0.key < $1.key }.map { "\($0.key.rawValue) \($0.value)" }.joined(separator: ", ")
    }
}

/// Matchmaking's send rate (plan §7.3 B, "Rate"): a minimum spacing between
/// attempts and a cap on challenges per rolling hour. Pure; the caller
/// passes the time.
struct LichessBotMatchmakingRateLimiter: Sendable, Equatable {

    enum Decision: Sendable, Equatable {
        case allowed
        /// Too soon after the previous attempt: not before `until`.
        case spacing(until: Date)
        /// `count` challenges in the last hour reach the cap: not before
        /// `until`, when the oldest of them leaves the hour.
        case hourlyCap(until: Date, count: Int)
    }

    static let window: TimeInterval = 3600

    /// The latest attempt, whether or not it reached Lichess: spacing
    /// applies to every attempt, so a run of failures can't loop quickly.
    private(set) var lastAttemptAt: Date?
    /// Challenges that reached Lichess (created or refused) within the last
    /// hour, oldest first: what the per-hour cap counts.
    private(set) var challengeTimes: [Date] = []

    func challengesInLastHour(now: Date) -> Int {
        challengeTimes.filter { now.timeIntervalSince($0) < Self.window }.count
    }

    func decision(now: Date, perHourCap: Int, minimumSpacing: TimeInterval) -> Decision {
        let recent = challengeTimes.filter { now.timeIntervalSince($0) < Self.window }
        if recent.count >= perHourCap, let oldest = recent.first {
            return .hourlyCap(until: oldest.addingTimeInterval(Self.window), count: recent.count)
        }
        if let lastAttemptAt, now.timeIntervalSince(lastAttemptAt) < minimumSpacing {
            return .spacing(until: lastAttemptAt.addingTimeInterval(minimumSpacing))
        }
        return .allowed
    }

    /// Note an attempt at `now`; `reachedLichess` when the challenge itself
    /// was posted (created or refused), which is what the cap counts.
    mutating func recordAttempt(at now: Date, reachedLichess: Bool) {
        lastAttemptAt = now
        challengeTimes.removeAll { now.timeIntervalSince($0) >= Self.window }
        if reachedLichess {
            challengeTimes.append(now)
        }
    }
}
