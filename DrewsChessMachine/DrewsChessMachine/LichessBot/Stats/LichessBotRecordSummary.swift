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

/// Someone DCM has played, from its own records.
struct LichessBotPastOpponent: Sendable, Equatable, Identifiable {
    /// Lowercased Lichess user id.
    let id: String
    let name: String
    let title: String?
    let kind: LichessBotOpponentKind
    let lastPlayedAt: Date
    let record: LichessBotResultTally

    var gamesSortKey: Int { record.games }
    var nameSortKey: String { name.lowercased() }
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

/// DCM's record per period (last hour … all time) from the games index,
/// and the per-opponent views the Challenge sheet reads. Pure, so the
/// period boundaries and tallies are testable.
enum LichessBotRecordSummary {
    /// The one period type (`LichessBotStatsPeriod`), under the name the
    /// Record card first used.
    typealias Period = LichessBotStatsPeriod

    /// A calendar without an interval for a period's unit.
    typealias ComputeError = LichessBotStatsPeriods.CalendarError

    /// One record per period.
    struct Records: Sendable, Equatable {
        var lastHour = LichessBotPeriodRecord()
        var today = LichessBotPeriodRecord()
        var yesterday = LichessBotPeriodRecord()
        var thisWeek = LichessBotPeriodRecord()
        var lastWeek = LichessBotPeriodRecord()
        var thisMonth = LichessBotPeriodRecord()
        var lastMonth = LichessBotPeriodRecord()
        var thisYear = LichessBotPeriodRecord()
        var lastYear = LichessBotPeriodRecord()
        var allTime = LichessBotPeriodRecord()

        subscript(period: Period) -> LichessBotPeriodRecord {
            switch period {
            case .lastHour: return lastHour
            case .today: return today
            case .yesterday: return yesterday
            case .thisWeek: return thisWeek
            case .lastWeek: return lastWeek
            case .thisMonth: return thisMonth
            case .lastMonth: return lastMonth
            case .thisYear: return thisYear
            case .lastYear: return lastYear
            case .allTime: return allTime
            }
        }

        fileprivate mutating func add(_ row: LichessBotGameSummary, starts: LichessBotStatsPeriodStarts) {
            allTime.add(row)
            if starts.contains(row.createdAt, in: .thisYear) { thisYear.add(row) }
            if starts.contains(row.createdAt, in: .lastYear) { lastYear.add(row) }
            if starts.contains(row.createdAt, in: .thisMonth) { thisMonth.add(row) }
            if starts.contains(row.createdAt, in: .lastMonth) { lastMonth.add(row) }
            if starts.contains(row.createdAt, in: .thisWeek) { thisWeek.add(row) }
            if starts.contains(row.createdAt, in: .lastWeek) { lastWeek.add(row) }
            if starts.contains(row.createdAt, in: .today) { today.add(row) }
            if starts.contains(row.createdAt, in: .yesterday) { yesterday.add(row) }
            if starts.contains(row.createdAt, in: .lastHour) { lastHour.add(row) }
        }
    }

    /// Everyone DCM has played, most recent game first.
    static func pastOpponents(rows: [LichessBotGameSummary]) -> [LichessBotPastOpponent] {
        var byID: [String: (name: String, title: String?, kind: LichessBotOpponentKind, last: Date, record: LichessBotResultTally)] = [:]
        for row in rows {
            guard let opponentID = row.opponentID else { continue }
            let id = opponentID.lowercased()
            var entry = byID[id] ?? (row.opponentName ?? opponentID, row.opponentTitle, row.opponentKind, row.createdAt, LichessBotResultTally())
            entry.record.add(ourScore: row.ourScore)
            if row.createdAt >= entry.last {
                entry.last = row.createdAt
                entry.name = row.opponentName ?? entry.name
                entry.title = row.opponentTitle ?? entry.title
            }
            byID[id] = entry
        }
        return byID.map { id, entry in
            LichessBotPastOpponent(id: id, name: entry.name, title: entry.title, kind: entry.kind, lastPlayedAt: entry.last, record: entry.record)
        }
        .sorted { $0.lastPlayedAt > $1.lastPlayedAt }
    }

    /// DCM's results against each opponent, by lowercased user id.
    static func byOpponent(rows: [LichessBotGameSummary]) -> [String: LichessBotResultTally] {
        var tallies: [String: LichessBotResultTally] = [:]
        for row in rows {
            guard let opponentID = row.opponentID else { continue }
            tallies[opponentID.lowercased(), default: LichessBotResultTally()].add(ourScore: row.ourScore)
        }
        return tallies
    }

    /// Games against bots started at or after `since`.
    static func botGames(rows: [LichessBotGameSummary], since: Date) -> Int {
        rows.filter { $0.opponentKind == .bot && $0.createdAt >= since }.count
    }

    /// Each period's record. The boundaries come from
    /// `LichessBotStatsPeriods`, their one definition.
    static func compute(rows: [LichessBotGameSummary], now: Date, calendar: Calendar) throws -> Records {
        let starts = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        var records = Records()
        for row in rows {
            records.add(row, starts: starts)
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
