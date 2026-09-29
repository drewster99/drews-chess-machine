import Foundation

/// Which challenges the bot accepts (plan §7). The defaults are deliberately
/// conservative; plan §7 explains each one.
struct LichessBotChallengeSettings: Sendable, Equatable, Codable {
    /// Checked speeds no clock inside the clock and increment bounds can
    /// have, fastest first: every challenge at such a speed is declined, so
    /// the checkbox does nothing. Correspondence is always listed when
    /// checked, since it has no clock. Speed rises with the clock and the
    /// increment, so the fastest and slowest speeds the bounds allow are
    /// those of the bounds' corners.
    var speedsRuledOutByClockBounds: [LichessBotSpeed] {
        let fastest = LichessBotSpeed.forClock(limitSeconds: minimumInitialSeconds, incrementSeconds: minimumIncrementSeconds)
        let slowest = LichessBotSpeed.forClock(limitSeconds: maximumInitialSeconds, incrementSeconds: maximumIncrementSeconds)
        return allowedSpeeds.sorted().filter { speed in
            speed == .correspondence || speed < fastest || slowest < speed
        }
    }

    var acceptRated = false
    var acceptCasual = true
    var allowedSpeeds: Set<LichessBotSpeed> = [.blitz, .rapid]
    var minimumInitialSeconds = 180
    var maximumInitialSeconds = 1800
    var minimumIncrementSeconds = 0
    var maximumIncrementSeconds = 30
    var acceptBots = true
    var acceptHumans = true
    var minimumOpponentRating = 0
    var maximumOpponentRating = 4000
    var acceptProvisionalOpponents = true
    /// Lichess user ids (lowercase).
    var blockedUserIDs: [String] = []
    var acceptRematches = true
    var maxConcurrentGames = 2
    /// Slots at the top of `maxConcurrentGames` that only humans may take.
    var gamesReservedForHumans = 0
    var maxSimultaneousGamesPerOpponent = 1
    var maxGamesPerDay = 200
    var maxGamesPerOpponentPerDay = 20
    /// Withdraw an outgoing challenge nobody has answered after this long.
    /// Zero waits indefinitely. Busy bots often leave challenges unanswered
    /// rather than declining them.
    var outgoingChallengeTimeoutSeconds = 180
    /// Past this many challenge responses in a minute, further challenges
    /// are left unanswered (they expire on Lichess's side) rather than
    /// spending requests (plan §5.3).
    var challengeResponseBudgetPerMinute = 10
}

/// How the bot plays (plan §12.4).
struct LichessBotPlaySettings: Sendable, Equatable, Codable {
    var temperatureStart: Float = 0.5
    var temperatureDecayPerPly: Float = 0.05
    var temperatureFloor: Float = 0.01
    var minimumThinkMilliseconds = 0

    var resignEnabled = false
    var resignLossProbability: Float = 0.95
    var resignConsecutiveMoves = 6
    var resignMinimumPly = 40

    var offerDrawEnabled = false
    var offerDrawProbability: Float = 0.85
    var offerDrawConsecutiveMoves = 8
    var offerDrawMinimumPly = 60

    var acceptDrawEnabled = false
    /// Accept a draw offer when our expected score `p_win + ½·p_draw` is at
    /// or below this.
    var acceptDrawExpectedScore: Float = 0.45

    var claimWhenOpponentGone = true
    var maxTakebacksAcceptedPerGame = 0

    /// The sampling schedule these settings describe. No Dirichlet noise,
    /// ever: that is a self-play exploration device, not play.
    var samplingSchedule: SamplingSchedule {
        SamplingSchedule(startTau: temperatureStart, decayPerPly: temperatureDecayPerPly, floorTau: temperatureFloor)
    }
}

/// Greeting and goodbye messages (plan §12.5).
struct LichessBotChatSettings: Sendable, Equatable, Codable {
    var greetingEnabled = true
    var greetingTemplate = "DrewsChessMachine: a from-scratch neural net, no search. Model {modelID}. Type !help for commands."
    var goodbyeEnabled = false
    var goodbyeTemplate = "Thanks for the game, {opponent}!"
    var room: LichessBotChatRoom = .player
}

extension LichessBotChatRoom: Codable {}

/// Where the bot's model comes from (plan §9).
enum LichessBotModelSourceKind: String, Sendable, Equatable, Codable, CaseIterable {
    /// The live champion, re-snapshotted when a promotion changes its
    /// ModelID.
    case champion
    /// The trainer, snapshotted when selected or on request.
    case trainerSnapshot
    /// The trainer, re-snapshotted on a cadence.
    case liveTrainer
    /// A model file.
    case file
}

struct LichessBotModelSettings: Sendable, Equatable, Codable {
    var source: LichessBotModelSourceKind = .champion
    /// The model file, when `source == .file`.
    var filePath: String?
    var liveTrainerRefreshIntervalSeconds = 120
    /// Live trainer only: whether games already in progress switch to each
    /// new snapshot, or keep the one they started with.
    var midGameRefresh = false
}

/// Connection, pacing and safety (plan §5, §6, §12.6, §14.4).
struct LichessBotConnectionSettings: Sendable, Equatable, Codable {
    /// The Lichess account the token must belong to.
    var expectedAccountID = "drewschessmachine"
    var autoConnectOnLaunch = false
    var preventSleepWhileOnline = true
    var reconnectInitialSeconds = 2
    var reconnectCapSeconds = 60
    var eventStreamStallTimeoutSeconds = 25
    /// Game-stream silence before a cheap resync, used until keep-alives
    /// have been seen on that game's stream.
    var gameStreamResyncSeconds = 60
    var rateLimitBreakerWindowMinutes = 60
    var postRateLimitDrainMinutes = 15
    var accountRefreshMinimumIntervalSeconds = 300
    var exportMinimumSpacingSeconds = 10
    var lowClockThresholdMilliseconds = 10_000
    var lostOnTimeBreakerCount = 3
    var lostOnTimeBreakerWindowGames = 10
    var movePostFailureBreakerCount = 5
    var reconnectStormBreakerCount = 10
    var breakerWindowMinutes = 10
}

/// Automatic challenges to online bots (plan §7.3). Off by default. Read
/// live: every matchmaking pass uses the settings in force at that moment.
struct LichessBotMatchmakingSettings: Sendable, Equatable, Codable {
    /// Which free slots matchmaking fills.
    enum FillMode: String, Sendable, Equatable, Codable, CaseIterable {
        /// Every slot outside those reserved for humans.
        case everyFreeSlot
        /// Only when DCM has nothing in play (no game, accepted challenge,
        /// or challenge waiting), one challenge at a time.
        case onlyWhenIdle
    }

    var enabled = false
    var fillMode: FillMode = .everyFreeSlot
    /// Each send uses one of these, chosen uniformly.
    var timeControls: Set<LichessBotClockChoice> = [.blitz3plus2, .blitz5plus3]
    var rated = false
    /// The opponent's rating at the chosen speed must lie within DCM's own
    /// rating at that speed plus these offsets.
    var minimumRatingOffset = -300
    var maximumRatingOffset = 300
    /// The window used instead when DCM has no established rating at the
    /// chosen speed.
    var minimumRatingWithoutOwnRating = 1000
    var maximumRatingWithoutOwnRating = 2200
    /// Pick among fitting favorites before any other bot.
    var preferFavorites = true
    /// Challenges matchmaking sends in any rolling hour.
    var maxChallengesPerHour = 20
    /// After a bot declines one of DCM's challenges, matchmaking leaves it
    /// alone this long. Zero records no cool-down.
    var declineCooldownHours = 6
}

/// How the bot's window presents games (plan §14.3a).
struct LichessBotDisplaySettings: Sendable, Equatable, Codable {
    /// How long a finished game stays in the live grid before it is removed
    /// (it can also be dismissed by hand). Zero removes it at once.
    var finishedGameRetentionMinutes = 10
}

/// All Lichess bot settings, persisted as one JSON blob (plan §12.1). Not
/// `TrainingParameters`: none of this is a training knob, and none of it
/// belongs in a `.dcmsession`. The API token is not here — it lives only in
/// the Keychain (plan §12.2).
struct LichessBotSettings: Sendable, Equatable, Codable {
    var challenge = LichessBotChallengeSettings()
    var play = LichessBotPlaySettings()
    var chat = LichessBotChatSettings()
    var model = LichessBotModelSettings()
    var connection = LichessBotConnectionSettings()
    var display = LichessBotDisplaySettings()
    var matchmaking = LichessBotMatchmakingSettings()

    /// Every problem with these settings. Empty means valid. Invalid
    /// settings are rejected as a whole, never partly applied.
    func validationProblems() -> [String] {
        var problems: [String] = []
        func require(_ condition: Bool, _ message: String) {
            if !condition { problems.append(message) }
        }

        let c = challenge
        require(c.acceptRated || c.acceptCasual, "Accept at least one of rated or casual games")
        require(!c.allowedSpeeds.isEmpty, "Allow at least one speed")
        require(c.acceptBots || c.acceptHumans, "Accept at least one of bots or humans")
        require(c.minimumInitialSeconds >= 0 && c.minimumInitialSeconds <= c.maximumInitialSeconds, "Clock minimum must be between zero and the maximum")
        require(c.minimumIncrementSeconds >= 0 && c.minimumIncrementSeconds <= c.maximumIncrementSeconds, "Increment minimum must be between zero and the maximum")
        require(c.minimumOpponentRating >= 0 && c.minimumOpponentRating <= c.maximumOpponentRating, "Rating minimum must be between zero and the maximum")
        require(c.maxConcurrentGames >= 1, "Allow at least one concurrent game")
        require(c.gamesReservedForHumans >= 0 && c.gamesReservedForHumans <= c.maxConcurrentGames, "Reserved human slots must be between zero and the concurrent-game limit")
        require(c.maxSimultaneousGamesPerOpponent >= 1, "Allow at least one game per opponent")
        require(c.maxGamesPerDay >= 1 && c.maxGamesPerOpponentPerDay >= 1, "Daily limits must be at least one")
        require(c.outgoingChallengeTimeoutSeconds >= 0, "The unanswered-challenge timeout cannot be negative")
        require(c.challengeResponseBudgetPerMinute >= 1, "The challenge-response budget must be at least one per minute")

        let p = play
        require(p.temperatureFloor >= LichessBotLimits.minimumTemperature, "Temperature floor must be at least \(LichessBotLimits.minimumTemperature)")
        require(p.temperatureStart >= p.temperatureFloor, "Starting temperature must be at least the floor")
        require(p.temperatureDecayPerPly >= 0, "Temperature decay cannot be negative")
        require(p.minimumThinkMilliseconds >= 0, "Minimum think time cannot be negative")
        for (value, name) in [(p.resignLossProbability, "Resign"), (p.offerDrawProbability, "Offer-draw"), (p.acceptDrawExpectedScore, "Accept-draw")] {
            require(value >= 0 && value <= 1, "\(name) threshold must be between 0 and 1")
        }
        require(p.resignConsecutiveMoves >= 1 && p.offerDrawConsecutiveMoves >= 1, "Consecutive-move counts must be at least one")
        require(p.resignMinimumPly >= 0 && p.offerDrawMinimumPly >= 0, "Minimum plies cannot be negative")
        require(p.maxTakebacksAcceptedPerGame >= 0, "Takeback limit cannot be negative")

        for (enabled, template, name) in [(chat.greetingEnabled, chat.greetingTemplate, "Greeting"), (chat.goodbyeEnabled, chat.goodbyeTemplate, "Goodbye")] where enabled {
            if let problem = LichessBotChat.templateProblem(template) {
                problems.append("\(name): \(problem)")
            }
        }

        require(model.liveTrainerRefreshIntervalSeconds >= LichessBotLimits.minimumLiveTrainerRefreshSeconds, "Live-trainer refresh must be at least \(LichessBotLimits.minimumLiveTrainerRefreshSeconds) seconds")
        if model.source == .file {
            require(!(model.filePath ?? "").isEmpty, "Choose a model file")
        }

        let n = connection
        require(!n.expectedAccountID.isEmpty, "Set the Lichess account the token must belong to")
        require(n.reconnectInitialSeconds >= 1 && n.reconnectInitialSeconds <= n.reconnectCapSeconds, "Reconnect delay must be at least one second and at most the cap")
        require(n.eventStreamStallTimeoutSeconds > LichessBotLimits.eventStreamKeepAliveSeconds, "Event-stream stall timeout must exceed Lichess's keep-alive interval of \(LichessBotLimits.eventStreamKeepAliveSeconds) seconds")
        require(n.gameStreamResyncSeconds >= 10, "Game-stream resync must be at least ten seconds")
        // A second 429 cannot come before the first one's cooldown ends, so a
        // breaker window no longer than the cooldown could never trip.
        let minimumCooldownMinutes = Int(LichessBotRateLimit.minimumCooldown.components.seconds / 60)
        require(n.rateLimitBreakerWindowMinutes > minimumCooldownMinutes, "The rate-limit breaker window must be longer than the cooldown after a 429, or a second 429 could never fall inside it")
        require(n.postRateLimitDrainMinutes >= 0, "The post-rate-limit hold cannot be negative")
        require(n.accountRefreshMinimumIntervalSeconds >= 60 && n.exportMinimumSpacingSeconds >= 1, "Housekeeping spacing is too aggressive")
        require(n.lowClockThresholdMilliseconds >= 0, "Low-clock threshold cannot be negative")
        require(n.lostOnTimeBreakerCount >= 1 && n.lostOnTimeBreakerWindowGames >= n.lostOnTimeBreakerCount, "Lost-on-time breaker must trip within its window")
        require(n.movePostFailureBreakerCount >= 1 && n.reconnectStormBreakerCount >= 1 && n.breakerWindowMinutes >= 1, "Breaker thresholds must be at least one")
        require(display.finishedGameRetentionMinutes >= 0, "Finished-game retention cannot be negative")

        let m = matchmaking
        require(!m.timeControls.isEmpty, "Matchmaking: choose at least one time control")
        require(m.minimumRatingOffset <= m.maximumRatingOffset, "Matchmaking: the rating window's lower offset must not exceed its upper offset")
        require(m.minimumRatingWithoutOwnRating >= 0 && m.minimumRatingWithoutOwnRating <= m.maximumRatingWithoutOwnRating, "Matchmaking: the rating bounds used without DCM's own rating must run from zero or more up to the maximum")
        require(m.maxChallengesPerHour >= 1, "Matchmaking: allow at least one challenge per hour")
        require(m.declineCooldownHours >= 0, "Matchmaking: the decline cool-down cannot be negative")
        return problems
    }
}

/// Fixed limits that are facts about Lichess or about DCM, not preferences.
enum LichessBotLimits {
    /// Lichess's documented event-stream keep-alive interval.
    static let eventStreamKeepAliveSeconds = 7
    /// `SamplingSchedule` needs a strictly positive temperature; the argmax
    /// stand-in is the practical floor.
    static let minimumTemperature: Float = 0.01
    /// A live-trainer snapshot briefly holds the lock SGD needs, so it may
    /// not run too often.
    static let minimumLiveTrainerRefreshSeconds = 30
    /// The most bots `GET /api/bot/online` returns in one request.
    static let onlineBotsMaximum = 512
    /// The most players `GET /player/online` returns (lila caps `nb`).
    static let onlinePlayersMaximum = 50
    /// The most players `GET /api/player/top/{nb}/{perfType}` returns.
    static let leaderboardMaximum = 100
    /// `GET /api/player/autocomplete` needs at least this many characters.
    static let autocompleteMinimumCharacters = 3
    /// The most ids `GET /api/users/status` accepts in one request.
    static let userStatusMaximumIDs = 100
    /// Lichess's limit on a BOT account's games against other bots in a
    /// rolling day (observed in its refusal text, 2026-09-28).
    static let botGamesPerDay = 100
}
