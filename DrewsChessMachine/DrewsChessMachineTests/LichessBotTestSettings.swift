@testable import DrewsChessMachine

/// Explicit Lichess bot settings for tests.
///
/// The shipped `LichessBotSettings()` defaults are the owner's own running
/// configuration and change whenever that configuration does. Tests that
/// depend on particular values (which speeds are allowed, the clock bounds,
/// the concurrency limit, the model source the fake provider serves, whether
/// matchmaking runs) build from these baselines instead, so a change to the
/// shipped defaults cannot silently change what a test exercises. Only the
/// tests that pin the shipped defaults themselves (validity, round-trip,
/// store load/reset) use `LichessBotSettings()`.
extension LichessBotChallengeSettings {
    /// Blitz and rapid only, three to thirty minutes, casual only, two
    /// concurrent games and none reserved for humans.
    static func testBaseline() -> LichessBotChallengeSettings {
        var settings = LichessBotChallengeSettings()
        settings.acceptRated = false
        settings.acceptCasual = true
        settings.allowedSpeeds = [.blitz, .rapid]
        settings.minimumInitialSeconds = 180
        settings.maximumInitialSeconds = 1800
        settings.minimumIncrementSeconds = 0
        settings.maximumIncrementSeconds = 30
        settings.acceptBots = true
        settings.acceptHumans = true
        settings.minimumOpponentRating = 0
        settings.maximumOpponentRating = 4000
        settings.acceptProvisionalOpponents = true
        settings.blockedUserIDs = []
        settings.acceptRematches = true
        settings.maxConcurrentGames = 2
        settings.gamesReservedForHumans = 0
        settings.maxSimultaneousGamesPerOpponent = 1
        settings.maxGamesPerDay = 200
        settings.maxGamesPerOpponentPerDay = 20
        settings.outgoingChallengeTimeoutSeconds = 180
        return settings
    }
}

extension LichessBotPlaySettings {
    static func testBaseline() -> LichessBotPlaySettings {
        var settings = LichessBotPlaySettings()
        settings.temperatureStart = 0.5
        settings.temperatureDecayPerPly = 0.05
        settings.temperatureFloor = 0.01
        settings.minimumThinkMilliseconds = 0
        return settings
    }
}

extension LichessBotChatSettings {
    static func testBaseline() -> LichessBotChatSettings {
        var settings = LichessBotChatSettings()
        settings.greetingEnabled = true
        settings.greetingTemplate = "DrewsChessMachine: a from-scratch neural net, no search. Model {modelID}. Type !help for commands."
        settings.goodbyeEnabled = false
        settings.goodbyeTemplate = "Thanks for the game, {opponent}!"
        return settings
    }
}

extension LichessBotModelSettings {
    /// The champion source, which is what the tests' fake model providers
    /// serve; no file.
    static func testBaseline() -> LichessBotModelSettings {
        var settings = LichessBotModelSettings()
        settings.source = .champion
        settings.filePath = nil
        return settings
    }
}

extension LichessBotMatchmakingSettings {
    /// Matchmaking off, casual, two blitz controls, favorites preferred.
    static func testBaseline() -> LichessBotMatchmakingSettings {
        var settings = LichessBotMatchmakingSettings()
        settings.enabled = false
        settings.fillMode = .everyFreeSlot
        settings.timeControls = [.blitz3plus2, .blitz5plus3]
        settings.rated = false
        settings.minimumRatingOffset = -300
        settings.maximumRatingOffset = 300
        settings.minimumRatingWithoutOwnRating = 1000
        settings.maximumRatingWithoutOwnRating = 2200
        settings.preferFavorites = true
        settings.maxChallengesPerHour = 20
        settings.declineCooldownHours = 6
        return settings
    }
}

extension LichessBotSettings {
    /// Every section from its test baseline; alerts stay silent.
    static func testBaseline() -> LichessBotSettings {
        var settings = LichessBotSettings()
        settings.challenge = .testBaseline()
        settings.play = .testBaseline()
        settings.chat = .testBaseline()
        settings.model = .testBaseline()
        settings.matchmaking = .testBaseline()
        settings.alerts = LichessBotAlertSettings(botChallengeSoundName: nil, humanChallengeSoundName: nil)
        return settings
    }
}
