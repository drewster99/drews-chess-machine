import Foundation

/// The periods the Record card counts games over
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.2). The one period type: the
/// Record summary's `Period` is an alias of it.
///
/// Raw values are stable identifiers, because the panel's selected period
/// is remembered in the defaults; what the operator reads is `label`.
enum LichessBotStatsPeriod: String, CaseIterable, Sendable, Codable, Hashable {
    case lastHour
    case today
    case yesterday
    case thisWeek
    case lastWeek
    case thisMonth
    case lastMonth
    case thisYear
    case lastYear
    case allTime

    var label: String {
        switch self {
        case .lastHour: return "Last hour"
        case .today: return "Today"
        case .yesterday: return "Yesterday"
        case .thisWeek: return "This week"
        case .lastWeek: return "Last week"
        case .thisMonth: return "This month"
        case .lastMonth: return "Last month"
        case .thisYear: return "This year"
        case .lastYear: return "Last year"
        case .allTime: return "All time"
        }
    }

    /// The period table row's tooltip.
    var help: String {
        switch self {
        case .lastHour:
            return "The 60 minutes before now (not the clock hour), so just after midnight it can hold games Today does not"
        case .yesterday:
            return Self.previousPeriodHelp(unit: "day")
        case .lastWeek:
            return Self.previousPeriodHelp(unit: "week")
        case .lastMonth:
            return Self.previousPeriodHelp(unit: "month")
        case .lastYear:
            return Self.previousPeriodHelp(unit: "year")
        case .today, .thisWeek, .thisMonth, .thisYear, .allTime:
            return "Games started since the start of this period, in this Mac's calendar and time zone"
        }
    }

    private static func previousPeriodHelp(unit: String) -> String {
        "Games started in the whole previous \(unit), in this Mac's calendar and time zone"
    }
}

/// Where each period begins, for one `now` and one calendar. Every period
/// is anchored on a game's start (`createdAt`). A current period (today,
/// this week, …) has no upper bound: a `createdAt` slightly in the future
/// (clock skew between Lichess and this Mac) counts in every current period
/// rather than in none. A previous period (yesterday, last week, …) ends
/// where the current one of the same unit begins.
struct LichessBotStatsPeriodStarts: Sendable, Equatable {
    /// `now − 3,600 s`: a rolling hour, not the clock hour, so just after
    /// midnight it can hold games Today does not.
    let lastHour: Date
    let today: Date
    let yesterday: Date
    let thisWeek: Date
    let lastWeek: Date
    let thisMonth: Date
    let lastMonth: Date
    let thisYear: Date
    let lastYear: Date

    /// Whether a game started at `createdAt` counts in `period`.
    func contains(_ createdAt: Date, in period: LichessBotStatsPeriod) -> Bool {
        switch period {
        case .lastHour: return createdAt >= lastHour
        case .today: return createdAt >= today
        case .yesterday: return createdAt >= yesterday && createdAt < today
        case .thisWeek: return createdAt >= thisWeek
        case .lastWeek: return createdAt >= lastWeek && createdAt < thisWeek
        case .thisMonth: return createdAt >= thisMonth
        case .lastMonth: return createdAt >= lastMonth && createdAt < thisMonth
        case .thisYear: return createdAt >= thisYear
        case .lastYear: return createdAt >= lastYear && createdAt < thisYear
        case .allTime: return true
        }
    }
}

/// The period boundaries and when they next move. Pure: `now` and the
/// calendar (with its time zone and first weekday) are inputs, so every
/// boundary — DST days, week starts, a time zone far from UTC — is
/// testable.
enum LichessBotStatsPeriods {

    static let lastHourSeconds: TimeInterval = 3600

    /// The calendar gave no interval for a unit. Foundation's Gregorian
    /// calendars always do; a failure is reported (and shown on the card),
    /// never replaced by a guessed boundary.
    enum CalendarError: LocalizedError, Equatable {
        case noDayInterval(Date)
        case noWeekInterval(Date)
        case noMonthInterval(Date)
        case noYearInterval(Date)

        var errorDescription: String? {
            switch self {
            case .noDayInterval(let date):
                return "The calendar gives no day containing \(date)"
            case .noWeekInterval(let date):
                return "The calendar gives no week containing \(date)"
            case .noMonthInterval(let date):
                return "The calendar gives no month containing \(date)"
            case .noYearInterval(let date):
                return "The calendar gives no year containing \(date)"
            }
        }
    }

    /// Each period's start at `now`. The week starts on the calendar's
    /// `firstWeekday` (the system locale's, as the Record card always did).
    /// Each previous period is the calendar's interval of its unit holding
    /// the instant just before the current one begins, so a DST day, a short
    /// month or a leap year has its true length.
    static func starts(now: Date, calendar: Calendar) throws -> LichessBotStatsPeriodStarts {
        let today = try interval(.day, now: now, calendar: calendar).start
        let thisWeek = try interval(.week, now: now, calendar: calendar).start
        let thisMonth = try interval(.month, now: now, calendar: calendar).start
        let thisYear = try interval(.year, now: now, calendar: calendar).start
        return LichessBotStatsPeriodStarts(
            lastHour: now.addingTimeInterval(-lastHourSeconds),
            today: today,
            yesterday: try interval(.day, now: today.addingTimeInterval(-1), calendar: calendar).start,
            thisWeek: thisWeek,
            lastWeek: try interval(.week, now: thisWeek.addingTimeInterval(-1), calendar: calendar).start,
            thisMonth: thisMonth,
            lastMonth: try interval(.month, now: thisMonth.addingTimeInterval(-1), calendar: calendar).start,
            thisYear: thisYear,
            lastYear: try interval(.year, now: thisYear.addingTimeInterval(-1), calendar: calendar).start
        )
    }

    /// The earliest moment after `now` at which some period's numbers change
    /// without a new game: the oldest last-hour game leaving the rolling
    /// hour, or the next start of a day, week, month or year (which also
    /// moves yesterday, last week, last month or last year). The Record
    /// card recomputes when it passes (plan §4.3).
    static func nextChange(after now: Date, rows: [LichessBotGameSummary], calendar: Calendar) throws -> Date {
        var earliest = min(
            try interval(.day, now: now, calendar: calendar).end,
            try interval(.week, now: now, calendar: calendar).end,
            try interval(.month, now: now, calendar: calendar).end,
            try interval(.year, now: now, calendar: calendar).end
        )
        // A game counts in the rolling hour through `createdAt + 3,600 s`
        // inclusive and leaves it just after. Games already out of it
        // change nothing later. A game exactly at the edge gives
        // `now` itself, so the next check (a minute later) drops it.
        let leavesLastHour = rows
            .map { $0.createdAt.addingTimeInterval(lastHourSeconds) }
            .filter { $0 >= now }
            .min()
        if let leavesLastHour, leavesLastHour < earliest {
            earliest = leavesLastHour
        }
        return earliest
    }

    /// A calendar unit the periods are built from, with the error for each,
    /// so no unit's failure is reported as another's.
    private enum Unit {
        case day, week, month, year

        var component: Calendar.Component {
            switch self {
            case .day: return .day
            case .week: return .weekOfYear
            case .month: return .month
            case .year: return .year
            }
        }

        func missingInterval(at date: Date) -> CalendarError {
            switch self {
            case .day: return .noDayInterval(date)
            case .week: return .noWeekInterval(date)
            case .month: return .noMonthInterval(date)
            case .year: return .noYearInterval(date)
            }
        }
    }

    private static func interval(_ unit: Unit, now: Date, calendar: Calendar) throws -> DateInterval {
        guard let interval = calendar.dateInterval(of: unit.component, for: now) else {
            throw unit.missingInterval(at: now)
        }
        return interval
    }
}
