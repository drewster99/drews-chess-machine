import Foundation

/// Which challenges the bot accepts (plan §7). The defaults are the
/// owner's own running configuration; plan §7 explains each field.
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
    var allowedSpeeds: Set<LichessBotSpeed> = [.ultraBullet, .bullet, .blitz, .rapid, .classical]
    var minimumInitialSeconds = 15
    var maximumInitialSeconds = 1800
    var minimumIncrementSeconds = 0
    var maximumIncrementSeconds = 30
    var acceptBots = true
    var acceptHumans = true
    var minimumOpponentRating = 0
    var maximumOpponentRating = 2500
    var acceptProvisionalOpponents = true
    /// Lichess user ids (lowercase).
    var blockedUserIDs: [String] = []
    var acceptRematches = true
    var maxConcurrentGames = 12
    /// Slots at the top of `maxConcurrentGames` that only humans may take.
    var gamesReservedForHumans = 2
    var maxSimultaneousGamesPerOpponent = 1
    var maxGamesPerDay = 2000
    var maxGamesPerOpponentPerDay = 5
    /// Of Lichess' daily bot-vs-bot games (`LichessBotLimits.botGamesPerDay`),
    /// how many DCM's own challenges leave for bots that challenge DCM:
    /// matchmaking and the challenge queue stop this many short of the
    /// limit (`LichessBotBotGameBudget`). Games against humans never count
    /// toward that limit, so none are reserved for them.
    var botGamesReservedForIncoming = 0
    /// How many more matchmaking leaves for the operator's challenge queue.
    var botGamesReservedForChallengeQueue = 0
    /// Withdraw an outgoing challenge nobody has answered after this long.
    /// Zero waits indefinitely. Busy bots often leave challenges unanswered
    /// rather than declining them.
    var outgoingChallengeTimeoutSeconds = 999
    /// Past this many challenge responses in a minute, further challenges
    /// are left unanswered (they expire on Lichess's side) rather than
    /// spending requests (plan §5.3).
    var challengeResponseBudgetPerMinute = 10
}

/// How the bot plays (plan §12.4).
struct LichessBotPlaySettings: Sendable, Equatable, Codable {
    var temperatureStart: Float = 0.11
    var temperatureDecayPerPly: Float = 0.02
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
    var greetingTemplate = "Hi, I'm DrewsChessMachine ({modelID}), a from-scratch neural net, no search. Type !help for commands."
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
    /// A model lineage on disk: the newest file of one training run, from a
    /// chosen segment on, re-checked on a cadence (follow-lineage plan).
    case followLineage

    /// The operator-facing name, as the source picker shows it.
    var displayName: String {
        switch self {
        case .champion: return "Champion"
        case .trainerSnapshot: return "Trainer snapshot"
        case .liveTrainer: return "Live trainer"
        case .file: return "Model file"
        case .followLineage: return "Follow lineage"
        }
    }
}

/// The lineage the follow-lineage source plays: one training run, from one
/// of its segments on (follow-lineage plan §3.1). Identified by the
/// `dcm_lineage` record every current model file carries, never by a
/// filename.
struct LichessBotFollowedLineage: Sendable, Equatable, Hashable, Codable {
    /// `LineageRecord.Run.lineageRunID` of the run.
    let lineageRunID: String
    /// `LineageRecord.Run.segmentID` of the segment the operator chose.
    /// Files of this segment and of every later segment descending from it
    /// (each exact resume of the run) are candidates.
    let anchorSegmentID: String
}

/// What selects a model generation's weights: the source kind, plus the
/// file for the file source and the lineage for the follow-lineage source.
/// Settings with equal generation sources play the same weights; the other
/// model fields (intervals, mid-game toggle) and a file path or lineage left
/// over while another source is chosen don't change which weights are
/// played (follow-lineage plan §3.4, §3.10).
struct LichessBotGenerationSource: Sendable, Equatable {
    let kind: LichessBotModelSourceKind
    /// The model file's path; nil unless `kind` is `.file`.
    let filePath: String?
    /// The followed lineage; nil unless `kind` is `.followLineage`.
    let followedLineage: LichessBotFollowedLineage?
}

struct LichessBotModelSettings: Sendable, Equatable, Codable {
    var source: LichessBotModelSourceKind = .file
    /// The model file, when `source == .file`.
    var filePath: String? = "/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20260713-v5cont-resume-replay-step270000.safetensors"
    var liveTrainerRefreshIntervalSeconds = 120
    /// Live trainer and followed lineage: whether games already in progress
    /// switch to each new generation of their source, or keep the one they
    /// started with. (The key predates the followed lineage; it keeps its
    /// name so saved settings load unchanged.)
    var midGameRefresh = false
    /// The lineage the follow-lineage source plays. Optional, nil by
    /// default: settings saved before it existed load as nil.
    var followedLineage: LichessBotFollowedLineage? = nil
    /// How often the follow-lineage source looks for a newer file of its
    /// lineage. A check lists the models folder and reads only new or
    /// changed headers.
    var lineageCheckIntervalSeconds = 60

    /// The weights these settings select.
    var generationSource: LichessBotGenerationSource {
        LichessBotGenerationSource(
            kind: source,
            filePath: source == .file ? filePath : nil,
            followedLineage: source == .followLineage ? followedLineage : nil)
    }
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

/// Automatic challenges to online bots (plan §7.3). Read
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

    var enabled = true
    var fillMode: FillMode = .everyFreeSlot
    /// Each send uses one of these, chosen uniformly.
    var timeControls: Set<LichessBotClockChoice> = [.bullet1plus0, .bullet2plus1, .ultraBulletQuarterPlus0, .blitz3plus0, .blitz3plus2, .bullet1plus1, .blitz5plus3, .rapid10plus0, .blitz5plus0]
    var rated = true
    /// When a bot declines one of matchmaking's rated challenges with
    /// Lichess's `casual` reason ("please send me a casual challenge
    /// instead"), send it the same challenge once more, unrated, through
    /// matchmaking's own send path, and leave the decline cool-down to that
    /// resend's answer. Off, such a decline is handled like any other: the
    /// cool-down, plus the manual resend offer. Challenges the operator
    /// sends are never resent automatically.
    ///
    /// Settings saved before this field existed load it as `false`, today's
    /// default, which is the behavior they were saved under (the store
    /// fills an absent non-optional field from the defaults).
    var fallBackToCasual = false
    /// The opponent's rating at the chosen speed must lie within DCM's own
    /// rating at that speed plus these offsets.
    var minimumRatingOffset = -300
    var maximumRatingOffset = 300
    /// The window used instead when DCM has no established rating at the
    /// chosen speed.
    var minimumRatingWithoutOwnRating = 0
    var maximumRatingWithoutOwnRating = 2200
    /// Pick among fitting favorites before any other bot.
    var preferFavorites = false
    /// Challenges matchmaking sends in any rolling hour.
    var maxChallengesPerHour = 10
    /// After a bot declines one of DCM's challenges, matchmaking leaves it
    /// alone this long. Zero records no cool-down.
    var declineCooldownHours = 2
    /// After a bot declines with `noBot` (it plays no bots), matchmaking
    /// leaves it alone this many days (`LichessBotDeclineBlock`). Its own
    /// challenges to DCM are still answered by the acceptance settings.
    var noBotDeclineBlockDays = 30
    /// After a bot declines for a clock reason — too fast, too slow, that
    /// time control — matchmaking sends it no challenge with such a clock
    /// for this many days.
    var specificDeclineBlockDays = 14
    /// After a bot declines asking for casual (or for rated), matchmaking
    /// sends it no rated (or casual) challenge for this many days. Longer
    /// than the clock block by owner decision (2026-10-07): a casual-only
    /// bot was the second most common refusal (32 in the rebuilt history,
    /// after 39 noBot).
    var ratedCasualDeclineBlockDays = 30
    /// Pick among bots DCM has had no contact with (no challenge either way,
    /// no game) for `recentContactHours` first; with none, the one contacted
    /// longest ago.
    var preferNotRecentlyContacted = true
    var recentContactHours = 24
}

/// How the bot's window presents games (plan §14.3a).
struct LichessBotDisplaySettings: Sendable, Equatable, Codable {
    /// How long a finished game stays in the live grid before it is removed
    /// (it can also be dismissed by hand). Zero removes it at once.
    var finishedGameRetentionMinutes = 10
}

/// Tones played when a challenge arrives, so the owner notices activity
/// without watching the window. Read live on each arrival.
struct LichessBotAlertSettings: Sendable, Equatable, Codable {
    /// System sound name for challenges from BOT accounts; nil plays nothing.
    var botChallengeSoundName: String?
    /// System sound name for challenges from humans; nil plays nothing.
    var humanChallengeSoundName: String?
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
    var alerts = LichessBotAlertSettings()

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
        require(c.botGamesReservedForIncoming >= 0 && c.botGamesReservedForChallengeQueue >= 0, "Reserved bot games cannot be negative")
        require(matchmaking.noBotDeclineBlockDays >= 0 && matchmaking.specificDeclineBlockDays >= 0 && matchmaking.ratedCasualDeclineBlockDays >= 0, "Decline block days cannot be negative")
        require(matchmaking.recentContactHours >= 0, "Recent-contact hours cannot be negative")
        require(c.botGamesReservedForIncoming + c.botGamesReservedForChallengeQueue <= LichessBotLimits.botGamesPerDay, "Reserved bot games cannot exceed Lichess' \(LichessBotLimits.botGamesPerDay) per day")
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
        if model.source == .followLineage {
            if let followed = model.followedLineage {
                require(!followed.lineageRunID.isEmpty && !followed.anchorSegmentID.isEmpty, "The followed lineage names no run or segment; choose it again")
            } else {
                problems.append("Choose a lineage to follow")
            }
        }
        require(model.lineageCheckIntervalSeconds >= LichessBotLimits.minimumLineageCheckSeconds, "Lineage checks must be at least \(LichessBotLimits.minimumLineageCheckSeconds) seconds apart")

        let n = connection
        require(!n.expectedAccountID.isEmpty, "Set the Lichess account the token must belong to")
        require(n.reconnectInitialSeconds >= 1 && n.reconnectInitialSeconds <= n.reconnectCapSeconds, "Reconnect delay must be at least one second and at most the cap")
        require(n.eventStreamStallTimeoutSeconds > LichessBotLimits.eventStreamKeepAliveSeconds, "Event-stream stall timeout must exceed Lichess' keep-alive interval of \(LichessBotLimits.eventStreamKeepAliveSeconds) seconds")
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
    /// How often the online bot's poll loop asks its model source whether a
    /// newer generation is due. The one source of that cadence.
    static let modelRefreshPollSeconds = 15
    /// A lineage check runs from that poll, so it can't run more often.
    static let minimumLineageCheckSeconds = modelRefreshPollSeconds
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
    /// Lichess' limit on a BOT account's games against other bots in one
    /// 24-hour window (lila `BotLimit`; `LichessBotBotGameWindow` models
    /// the window).
    static let botGamesPerDay = 100
}
