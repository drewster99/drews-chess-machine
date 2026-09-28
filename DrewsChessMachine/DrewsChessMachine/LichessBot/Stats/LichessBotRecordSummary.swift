import Foundation

/// Wins, draws and losses for DCM over some set of games.
struct LichessBotResultTally: Sendable, Equatable {
    var wins = 0
    var draws = 0
    var losses = 0
    /// Games with no result for DCM (aborted, never started).
    var unscored = 0

    var scored: Int { wins + draws + losses }
    var games: Int { scored + unscored }

    /// DCM's score as a fraction of scored games, or nil with none scored.
    var score: Double? {
        scored > 0 ? (Double(wins) + 0.5 * Double(draws)) / Double(scored) : nil
    }

    mutating func add(ourScore: Double?) {
        switch ourScore {
        case .some(1): wins += 1
        case .some(0.5): draws += 1
        case .some(0): losses += 1
        default: unscored += 1
        }
    }
}

/// DCM's record over one period, broken down by opponent kind and color
/// (the Overview's Record card).
struct LichessBotPeriodRecord: Sendable, Equatable {
    var all = LichessBotResultTally()
    var versusBots = LichessBotResultTally()
    /// Humans; Lichess's own AI counts separately.
    var versusHumans = LichessBotResultTally()
    var versusLichessAI = LichessBotResultTally()
    var asWhite = LichessBotResultTally()
    var asBlack = LichessBotResultTally()
}

/// Today / this week / all time records from the games index. Pure, so the
/// period boundaries and tallies are testable.
enum LichessBotRecordSummary {
    enum Period: String, CaseIterable, Sendable {
        case today = "Today"
        case thisWeek = "This week"
        case allTime = "All time"
    }

    enum ComputeError: LocalizedError {
        case noWeekInterval(Date)

        var errorDescription: String? {
            switch self {
            case .noWeekInterval(let date):
                return "The calendar gives no week containing \(date)"
            }
        }
    }

    /// One record per period.
    struct Records: Sendable, Equatable {
        var today = LichessBotPeriodRecord()
        var thisWeek = LichessBotPeriodRecord()
        var allTime = LichessBotPeriodRecord()

        subscript(period: Period) -> LichessBotPeriodRecord {
            switch period {
            case .today: return today
            case .thisWeek: return thisWeek
            case .allTime: return allTime
            }
        }
    }

    /// Games against bots started at or after `since`.
    static func botGames(rows: [LichessBotGameSummary], since: Date) -> Int {
        rows.filter { $0.opponentKind == .bot && $0.createdAt >= since }.count
    }

    static func compute(rows: [LichessBotGameSummary], now: Date, calendar: Calendar) throws -> Records {
        let startOfToday = calendar.startOfDay(for: now)
        guard let startOfWeek = calendar.dateInterval(of: .weekOfYear, for: now)?.start else {
            throw ComputeError.noWeekInterval(now)
        }
        var records = Records()
        for row in rows {
            records.allTime.add(row)
            if row.createdAt >= startOfWeek { records.thisWeek.add(row) }
            if row.createdAt >= startOfToday { records.today.add(row) }
        }
        return records
    }
}

extension LichessBotPeriodRecord {
    mutating func add(_ row: LichessBotGameSummary) {
        all.add(ourScore: row.ourScore)
        switch row.opponentKind {
        case .bot: versusBots.add(ourScore: row.ourScore)
        case .human: versusHumans.add(ourScore: row.ourScore)
        case .lichessAI: versusLichessAI.add(ourScore: row.ourScore)
        }
        switch row.ourColor {
        case .white: asWhite.add(ourScore: row.ourScore)
        case .black: asBlack.add(ourScore: row.ourScore)
        }
    }
}
