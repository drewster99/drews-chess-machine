import Foundation

/// DCM's own games per speed, for the account grid's columns Lichess
/// doesn't report per speed: unrated games in DCM's game records, and games
/// started today (the Record card's Today, `LichessBotStatsPeriods`) and in
/// the 24 hours before now, live games included. Also DCM's own count of
/// today's games against bots, for the Challenge sheet.
struct LichessBotAccountGameCounts: Equatable, Sendable {
    let unrated: [String: Int]
    let today: [String: Int]
    let lastDay: [String: Int]
    /// Games against BOT accounts started today (the same Today), whatever
    /// their speed. DCM's own count, nothing more: Lichess counts its daily
    /// bot-game limit its own way (`LichessBotBotGameBudget`), and nothing
    /// is blocked on this number. A live game is counted once its opponent
    /// is known.
    let botGamesToday: Int

    static let lastDaySeconds: TimeInterval = 24 * 3600

    /// Unrated from `filedRows`; Today, Last 24 h and `botGamesToday` from `gameStarts`
    /// (`LichessBotGameStart.all`). A live game whose speed isn't known yet
    /// fits no row; it is counted once its `gameFull` arrives, moments after
    /// it starts.
    ///
    /// - Throws: `LichessBotStatsPeriods.CalendarError` when the calendar
    ///   gives no interval for `now`.
    init(filedRows: [LichessBotGameSummary], gameStarts: [LichessBotGameStart], now: Date, calendar: Calendar) throws {
        let periods = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        let dayAgo = now.addingTimeInterval(-Self.lastDaySeconds)
        var unrated: [String: Int] = [:]
        for row in filedRows where !row.rated {
            unrated[row.speed, default: 0] += 1
        }
        var today: [String: Int] = [:]
        var lastDay: [String: Int] = [:]
        var botGamesToday = 0
        for start in gameStarts {
            if start.opponentIsBot == true, periods.contains(start.startedAt, in: .today) { botGamesToday += 1 }
            guard let speed = start.speed else { continue }
            if periods.contains(start.startedAt, in: .today) { today[speed, default: 0] += 1 }
            if start.startedAt >= dayAgo { lastDay[speed, default: 0] += 1 }
        }
        self.unrated = unrated
        self.today = today
        self.lastDay = lastDay
        self.botGamesToday = botGamesToday
    }

    /// What the grid shows for the counts at one moment.
    enum Reading: Equatable, Sendable {
        /// The games index hasn't loaded.
        case loading
        case counted(LichessBotAccountGameCounts)
        /// The calendar gave no interval for now; the grid shows why.
        case failed(String)

        /// One cell: the speed's count in `column` (0 when it has none),
        /// "…" while the index loads, "!" when the counts failed.
        func text(_ column: KeyPath<LichessBotAccountGameCounts, [String: Int]>, speed: String) -> String {
            switch self {
            case .loading: return "…"
            case .counted(let counts): return "\(counts[keyPath: column][speed, default: 0])"
            case .failed: return "!"
            }
        }

        var isFailed: Bool {
            if case .failed = self { return true }
            return false
        }

        /// The line under the grid when the counts failed; empty otherwise,
        /// when the line is hidden.
        var failureText: String {
            if case .failed(let message) = self { return "Game counts unavailable: \(message)" }
            return ""
        }
    }
}
