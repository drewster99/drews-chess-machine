import Foundation

/// One game DCM played or is playing, reduced to its start and what the
/// per-day counts read. Filed games come from the games index; live games
/// not yet filed come from the controller's list. This is the one place the
/// two are merged, so the account grid's Today / Last 24 h columns, Lichess'
/// bot-game window and matchmaking's latest-contact memory all see the same
/// games.
struct LichessBotGameStart: Sendable, Equatable {
    let startedAt: Date
    /// The game's speed ("blitz", …); nil for a live game whose `gameFull`
    /// has not arrived yet.
    let speed: String?
    /// The opponent's Lichess id; nil for Lichess' AI, and for a live game
    /// whose players are not known yet.
    let opponentID: String?
    /// Whether the opponent is a BOT account; nil while a live game's
    /// players are not known yet.
    let opponentIsBot: Bool?
}

extension LichessBotGameStart {
    init(filed row: LichessBotGameSummary) {
        self.init(startedAt: row.createdAt, speed: row.speed, opponentID: row.opponentID, opponentIsBot: row.opponentKind == .bot)
    }

    @MainActor
    init(live game: LichessBotLiveGame) {
        self.init(startedAt: game.startedAt, speed: game.speed, opponentID: game.opponent?.id, opponentIsBot: game.opponent.map { $0.title == "BOT" })
    }

    /// Every filed game, then every live game not filed yet. A live game
    /// stays in the controller's list after it is filed, so live games are
    /// matched to the filed ones by id rather than counted twice.
    @MainActor
    static func all(filedRows: [LichessBotGameSummary], liveGames: [LichessBotLiveGame]) -> [LichessBotGameStart] {
        let filedIDs = Set(filedRows.map(\.gameID))
        let filed = filedRows.map { LichessBotGameStart(filed: $0) }
        let live = liveGames.filter { !filedIDs.contains($0.id) }.map { LichessBotGameStart(live: $0) }
        return filed + live
    }
}
