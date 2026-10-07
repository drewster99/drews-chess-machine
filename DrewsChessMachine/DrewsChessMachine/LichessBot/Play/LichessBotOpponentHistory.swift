import Foundation

/// The terms of a challenge DCM may send: its clock and whether it is rated.
struct LichessBotChallengeTerms: Sendable, Equatable {
    let limitSeconds: Int
    let incrementSeconds: Int
    let rated: Bool

    var speed: LichessBotSpeed {
        LichessBotSpeed.forClock(limitSeconds: limitSeconds, incrementSeconds: incrementSeconds)
    }

    /// Lichess' estimate of a game's length at this clock, `limit + 40 ×
    /// increment` seconds, the number its speeds are cut on. "Too fast" and
    /// "too slow" compare clocks by it.
    var estimatedSeconds: Int {
        Self.estimatedSeconds(limitSeconds: limitSeconds, incrementSeconds: incrementSeconds)
    }

    static func estimatedSeconds(limitSeconds: Int, incrementSeconds: Int) -> Int {
        limitSeconds + 40 * incrementSeconds
    }
}

/// What a bot's decline of one of DCM's challenges rules out, and until
/// when. Only reasons that say what the bot would not play make one; a
/// generic or "later" decline leaves only matchmaking's short decline
/// cool-down (`LichessBotMatchmakingSettings.declineCooldownHours`).
struct LichessBotDeclineBlock: Sendable, Equatable {
    enum Scope: Sendable, Equatable {
        /// `noBot`: the bot plays no bots at all.
        case everyChallenge
        /// `tooFast`: this clock and every faster one, by estimated length.
        case clocksAtMostEstimatedSeconds(Int)
        /// `tooSlow`: this clock and every slower one.
        case clocksAtLeastEstimatedSeconds(Int)
        /// `timeControl`: this exact clock.
        case clock(limitSeconds: Int, incrementSeconds: Int)
        /// `casual` ("casual games only"): rated challenges.
        case ratedChallenges
        /// `rated` ("rated games only"): casual challenges.
        case casualChallenges

        func covers(_ terms: LichessBotChallengeTerms) -> Bool {
            switch self {
            case .everyChallenge:
                return true
            case .clocksAtMostEstimatedSeconds(let estimate):
                return terms.estimatedSeconds <= estimate
            case .clocksAtLeastEstimatedSeconds(let estimate):
                return terms.estimatedSeconds >= estimate
            case .clock(let limit, let increment):
                return terms.limitSeconds == limit && terms.incrementSeconds == increment
            case .ratedChallenges:
                return terms.rated
            case .casualChallenges:
                return !terms.rated
            }
        }
    }

    let scope: Scope
    let reason: LichessBotDeclineReason
    /// When the declined challenge was sent: the challenge-log row's first
    /// fact. The decline itself follows within minutes, which a window of
    /// days does not notice, and the rebuilt history records only the send.
    let challengedAt: Date
    let until: Date

    /// The block a decline for `reason` of a challenge with these clock and
    /// rated terms makes, or nil when the reason rules out nothing specific
    /// or the term it would need was not recorded (nil).
    static func make(reason: LichessBotDeclineReason, limitSeconds: Int?, incrementSeconds: Int?, rated: Bool?, challengedAt: Date,
                     settings: LichessBotMatchmakingSettings) -> LichessBotDeclineBlock? {
        let clockBlockUntil = challengedAt.addingTimeInterval(TimeInterval(settings.specificDeclineBlockDays) * 86_400)
        let scope: Scope
        switch reason {
        case .noBot:
            return LichessBotDeclineBlock(scope: .everyChallenge, reason: reason, challengedAt: challengedAt,
                                          until: challengedAt.addingTimeInterval(TimeInterval(settings.noBotDeclineBlockDays) * 86_400))
        case .tooFast, .tooSlow, .timeControl:
            guard let limitSeconds, let incrementSeconds else { return nil }
            let estimate = LichessBotChallengeTerms.estimatedSeconds(limitSeconds: limitSeconds, incrementSeconds: incrementSeconds)
            switch reason {
            case .tooFast: scope = .clocksAtMostEstimatedSeconds(estimate)
            case .tooSlow: scope = .clocksAtLeastEstimatedSeconds(estimate)
            default: scope = .clock(limitSeconds: limitSeconds, incrementSeconds: incrementSeconds)
            }
        case .casual:
            // Only a rated challenge can be declined for wanting casual.
            guard rated == true else { return nil }
            return LichessBotDeclineBlock(scope: .ratedChallenges, reason: reason, challengedAt: challengedAt,
                                          until: challengedAt.addingTimeInterval(TimeInterval(settings.ratedCasualDeclineBlockDays) * 86_400))
        case .rated:
            guard rated == false else { return nil }
            return LichessBotDeclineBlock(scope: .casualChallenges, reason: reason, challengedAt: challengedAt,
                                          until: challengedAt.addingTimeInterval(TimeInterval(settings.ratedCasualDeclineBlockDays) * 86_400))
        case .generic, .later, .standard, .variant, .onlyBot:
            // `standard` / `variant`: DCM sends only standard chess, so these
            // say nothing about a next challenge; `onlyBot` cannot come
            // from a bot asked by a bot.
            return nil
        }
        return LichessBotDeclineBlock(scope: scope, reason: reason, challengedAt: challengedAt, until: clockBlockUntil)
    }
}

/// What DCM's records say about each opponent, for matchmaking
/// (`LichessBotMatchmaking.pick`): the declines still in force, and when DCM
/// last challenged them, was challenged by them, or played them. Built from
/// the challenge log's rows (`LichessBotChallengeLogRow.rows`, the one merge
/// of the live log and the history rebuilt from the protocol log) and the
/// games — never a second store that could drift from them.
struct LichessBotOpponentHistory: Sendable, Equatable {
    struct Contact: Sendable, Equatable {
        var lastChallengeSent: Date?
        var lastChallengeReceived: Date?
        var lastGame: Date?

        /// The latest of the three; nil when there is none.
        var latest: Date? {
            [lastChallengeSent, lastChallengeReceived, lastGame].compactMap { $0 }.max()
        }
    }

    /// One game's opponent and start, from the games index or a live game.
    struct GameStart: Sendable, Equatable {
        /// Any case; lowercased when the history is built.
        let opponentID: String
        let at: Date
    }

    /// By lowercased opponent id.
    private(set) var contacts: [String: Contact] = [:]
    /// Declines still in force at `now`, by lowercased opponent id, newest
    /// first.
    private(set) var blocks: [String: [LichessBotDeclineBlock]] = [:]

    static let empty = LichessBotOpponentHistory()

    private init() {}

    init(challengeRows: [LichessBotChallengeLogRow], gameStarts: [GameStart], now: Date, settings: LichessBotMatchmakingSettings) {
        for row in challengeRows {
            guard case .player(let player) = row.opponent else { continue }
            let id = player.id.lowercased()
            switch row.direction {
            case .outgoing?:
                contacts[id, default: Contact()].lastChallengeSent = Self.later(contacts[id]?.lastChallengeSent, row.at)
                if case .challenge(.declined(.known(let reason)), _) = row.state,
                   let block = LichessBotDeclineBlock.make(reason: reason, limitSeconds: row.terms?.limitSeconds, incrementSeconds: row.terms?.incrementSeconds,
                                                           rated: row.terms?.rated, challengedAt: row.at, settings: settings),
                   block.until > now {
                    blocks[id, default: []].append(block)
                }
            case .incoming?:
                contacts[id, default: Contact()].lastChallengeReceived = Self.later(contacts[id]?.lastChallengeReceived, row.at)
            case nil:
                break
            }
        }
        for game in gameStarts {
            let id = game.opponentID.lowercased()
            contacts[id, default: Contact()].lastGame = Self.later(contacts[id]?.lastGame, game.at)
        }
        blocks = blocks.mapValues { $0.sorted { $0.challengedAt > $1.challengedAt } }
    }

    /// The newest decline in force that rules out a challenge to `userID`
    /// on `terms`, or nil.
    func blockingDecline(of userID: String, terms: LichessBotChallengeTerms) -> LichessBotDeclineBlock? {
        blocks[userID.lowercased()]?.first { $0.scope.covers(terms) }
    }

    func contact(_ userID: String) -> Contact? {
        contacts[userID.lowercased()]
    }

    private static func later(_ current: Date?, _ candidate: Date) -> Date {
        guard let current else { return candidate }
        return max(current, candidate)
    }
}
