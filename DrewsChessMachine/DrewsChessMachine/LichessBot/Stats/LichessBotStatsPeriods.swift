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
    case thisWeek
    case thisMonth
    case thisYear
    case allTime

    var label: String {
        switch self {
        case .lastHour: return "Last hour"
        case .today: return "Today"
        case .thisWeek: return "This week"
        case .thisMonth: return "This month"
        case .thisYear: return "This year"
        case .allTime: return "All time"
        }
    }
}

/// Where each period begins, for one `now` and one calendar. Every period
/// is anchored on a game's start (`createdAt`) and has no upper bound: a
/// `createdAt` slightly in the future (clock skew between Lichess and this
/// Mac) counts in every current period rather than in none.
struct LichessBotStatsPeriodStarts: Sendable, Equatable {
    /// `now − 3,600 s`: a rolling hour, not the clock hour, so just after
    /// midnight it can hold games Today does not.
    let lastHour: Date
    let today: Date
    let thisWeek: Date
    let thisMonth: Date
    let thisYear: Date

    /// Whether a game started at `createdAt` counts in `period`.
    func contains(_ createdAt: Date, in period: LichessBotStatsPeriod) -> Bool {
        switch period {
        case .lastHour: return createdAt >= lastHour
        case .today: return createdAt >= today
        case .thisWeek: return createdAt >= thisWeek
        case .thisMonth: return createdAt >= thisMonth
        case .thisYear: return createdAt >= thisYear
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
    static func starts(now: Date, calendar: Calendar) throws -> LichessBotStatsPeriodStarts {
        LichessBotStatsPeriodStarts(
            lastHour: now.addingTimeInterval(-lastHourSeconds),
            today: calendar.startOfDay(for: now),
            thisWeek: try interval(.weekOfYear, now: now, calendar: calendar).start,
            thisMonth: try interval(.month, now: now, calendar: calendar).start,
            thisYear: try interval(.year, now: now, calendar: calendar).start
        )
    }

    /// The earliest moment after `now` at which some period's numbers change
    /// without a new game: the oldest last-hour game leaving the rolling
    /// hour, or the next start of a day, week, month or year. The Record
    /// card recomputes when it passes (plan §4.3).
    static func nextChange(after now: Date, rows: [LichessBotGameSummary], calendar: Calendar) throws -> Date {
        var earliest = min(
            try interval(.day, now: now, calendar: calendar).end,
            try interval(.weekOfYear, now: now, calendar: calendar).end,
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

    private static func interval(_ component: Calendar.Component, now: Date, calendar: Calendar) throws -> DateInterval {
        guard let interval = calendar.dateInterval(of: component, for: now) else {
            switch component {
            case .day: throw CalendarError.noDayInterval(now)
            case .weekOfYear: throw CalendarError.noWeekInterval(now)
            case .month: throw CalendarError.noMonthInterval(now)
            default: throw CalendarError.noYearInterval(now)
            }
        }
        return interval
    }
}
