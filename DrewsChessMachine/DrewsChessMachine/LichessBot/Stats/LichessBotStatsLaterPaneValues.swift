import Foundation

// The later panes' values (`LICHESS_BOT_RECORD_STATS_PLAN.md` §11, L1–L6):
// move choice, clock, game length, openings, opponents and bot health.
// Computed in the same pass as the first-pass panes
// (`LichessBotRecordStatistics.compute`).

/// L1: how DCM's moves related to its own policy, over some games.
struct LichessBotMoveChoiceLine: Sendable, Equatable {
    var decisions = 0
    var decisionsWithTopMoves = 0
    var topMoveChosen = 0
    var sumChosenProbability = 0.0
    var randomish = 0

    /// The share of decisions (that recorded the policy's top moves) that
    /// played the top move; nil with none.
    var topMoveShare: Double? {
        decisionsWithTopMoves > 0 ? Double(topMoveChosen) / Double(decisionsWithTopMoves) : nil
    }

    /// Mean sampling probability of the chosen move; nil with no decision.
    var meanChosenProbability: Double? {
        decisions > 0 ? sumChosenProbability / Double(decisions) : nil
    }

    mutating func add(_ choice: LichessBotMoveChoiceFacts, decisions count: Int) {
        decisions += count
        decisionsWithTopMoves += choice.decisionsWithTopMoves
        topMoveChosen += choice.topMoveChosen
        sumChosenProbability += choice.sumChosenProbability
        randomish += choice.randomish
    }
}

/// L1 for one model ID: the decisions of the games attributed to it (a
/// game played by two models counts wholly for the one that chose most of
/// its moves, as in the Models pane).
struct LichessBotModelMoveChoice: Sendable, Equatable, Identifiable {
    let modelID: String
    let line: LichessBotMoveChoiceLine

    var id: String { modelID }
}

/// L1 over one period.
struct LichessBotMoveChoiceStatistics: Sendable, Equatable {
    let overall: LichessBotMoveChoiceLine
    /// Most decisions first.
    let byModel: [LichessBotModelMoveChoice]
}

/// L2: DCM's clock in one time control over one period.
struct LichessBotClockRow: Sendable, Equatable, Identifiable {
    let speed: String
    /// Scored games with a clock.
    var games = 0
    var thinkTimeMoves = 0
    var sumThinkMilliseconds = 0.0
    var gamesWithFinalClock = 0
    var sumFinalClockMilliseconds = 0.0
    /// Games DCM lost on time (`outoftime`, the opponent won).
    var flagged = 0

    var id: String { speed }

    init(speed: String) {
        self.speed = speed
    }

    var meanThinkSeconds: Double? {
        thinkTimeMoves > 0 ? sumThinkMilliseconds / Double(thinkTimeMoves) / 1000 : nil
    }

    var meanFinalClockSeconds: Double? {
        gamesWithFinalClock > 0 ? sumFinalClockMilliseconds / Double(gamesWithFinalClock) / 1000 : nil
    }
}

/// L3: the length of games with one result.
struct LichessBotGameLengthRow: Sendable, Equatable, Identifiable {
    /// 1, ½ or 0: DCM's score in these games.
    let ourScore: Double
    let games: Int
    let meanPlies: Double?
    let medianPlies: Double?

    var id: Double { ourScore }

    var label: String {
        switch ourScore {
        case 1: return "Won"
        case 0: return "Lost"
        default: return "Drawn"
        }
    }
}

/// A short game (L3's short losses).
struct LichessBotShortGame: Sendable, Equatable, Identifiable {
    let gameID: String
    let createdAt: Date
    /// Nil for Lichess's AI, which has no account.
    let opponentName: String?
    let plies: Int

    var id: String { gameID }
}

/// L3 over one period.
struct LichessBotGameLengthStatistics: Sendable, Equatable {
    /// Won, drawn, lost, in that order (all three, even when empty).
    let rows: [LichessBotGameLengthRow]
    /// Lost games shorter than `shortLossPlies`, most recent first, at most
    /// `shortLossLimit`.
    let shortLosses: [LichessBotShortGame]
    /// All lost games shorter than `shortLossPlies`.
    let shortLossCount: Int

    static let shortLossPlies = 30
    static let shortLossLimit = 20
}

/// L4: DCM's results in one opening family, as White and as Black.
struct LichessBotOpeningRow: Sendable, Equatable, Identifiable {
    /// The opening name before its first colon ("Sicilian Defense" from
    /// "Sicilian Defense: Najdorf Variation").
    let family: String
    let lowestECO: String
    let highestECO: String
    let asWhite: LichessBotResultTally
    let asBlack: LichessBotResultTally

    var id: String { family }

    /// "B20", or "B20–B99" for a family spanning codes.
    var ecoRange: String {
        lowestECO == highestECO ? lowestECO : "\(lowestECO)–\(highestECO)"
    }

    /// The family of a Lichess opening name.
    static func family(of name: String) -> String {
        String(name.prefix { $0 != ":" }).trimmingCharacters(in: .whitespaces)
    }
}

/// L4 over one period.
struct LichessBotOpeningStatistics: Sendable, Equatable {
    /// Most games first, then by family.
    let rows: [LichessBotOpeningRow]
    /// Scored games without an opening (no export, or no facts).
    let gamesWithoutOpening: Int
}

/// L5: one opponent's record.
struct LichessBotOpponentRecordRow: Sendable, Equatable, Identifiable {
    /// Lowercased Lichess id.
    let id: String
    let name: String
    let kind: LichessBotOpponentKind
    let tally: LichessBotResultTally
    let lastPlayedAt: Date
}

/// L5's best win: the highest-rated opponent DCM beat.
struct LichessBotNotableWin: Sendable, Equatable {
    let gameID: String
    let createdAt: Date
    /// The name, else the user id; nil only if a rated opponent had neither.
    let opponentName: String?
    let rating: Int
}

/// A run of one result in consecutive scored games.
struct LichessBotStreak: Sendable, Equatable {
    /// 1, ½ or 0.
    let ourScore: Double
    let length: Int
}

/// L5 over one period.
struct LichessBotOpponentsStatistics: Sendable, Equatable {
    /// The most played opponents, most games first, at most `mostPlayedLimit`.
    let mostPlayed: [LichessBotOpponentRecordRow]
    /// Distinct opponents with an account.
    let opponentCount: Int
    let highestRatedWin: LichessBotNotableWin?
    /// The run the most recent scored games form; nil with none.
    let currentStreak: LichessBotStreak?
    /// The longest runs of wins and of losses; length 0 with none.
    let longestWinStreak: Int
    let longestLossStreak: Int

    static let mostPlayedLimit = 10
}

/// L6: bot health over one period, over every game (scored or not).
struct LichessBotHealthStatistics: Sendable, Equatable {
    var games = 0
    var anomalies = 0
    var gamesWithAnomalies = 0
    var rejectedMoves = 0
    var streamReconnects = 0
    /// Games the export corrected (the journal and the export disagreed).
    var reconciliationCorrected = 0
    /// Games filed without an export.
    var exportUnavailable = 0
    /// Rows without facts (rejected moves and reconnects unknown there).
    var rowsWithoutFacts = 0
}

/// D1: DCM's results in games of one origin (how the game began).
struct LichessBotOriginRow: Sendable, Equatable, Identifiable {
    let category: LichessBotGameOriginCategory
    let tally: LichessBotResultTally
    let performance: LichessBotRatingEstimate
    let ratedOpponentGames: Int

    var id: LichessBotGameOriginCategory { category }
}

/// D1 over one period: results by origin category, from the controller's
/// one origin resolver (challenge-log plan §3.6), which also covers games
/// played before origins were recorded. "Unknown" is its own row, never
/// folded into a known origin.
struct LichessBotOriginStatistics: Sendable, Equatable {
    /// Categories with scored games, in the category's own order.
    let rows: [LichessBotOriginRow]
    /// Scored games the resolver had no entry for.
    let gamesUnresolved: Int
}

/// Every later pane's values for one period and filter.
struct LichessBotLaterBreakdowns: Sendable, Equatable {
    let moveChoice: LichessBotMoveChoiceStatistics
    /// Lichess's speed order, then other speeds.
    let clock: [LichessBotClockRow]
    let gameLength: LichessBotGameLengthStatistics
    let openings: LichessBotOpeningStatistics
    let opponents: LichessBotOpponentsStatistics
    let health: LichessBotHealthStatistics
    /// Nil when the statistics were computed without origins.
    let origins: LichessBotOriginStatistics?
}
