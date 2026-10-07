import Foundation

/// Which games the Record card counts: every scored game, or only rated or
/// only casual ones (OD-3). Raw values are stable identifiers — the
/// selection is remembered in the defaults.
enum LichessBotStatsFilter: String, CaseIterable, Sendable, Codable, Hashable {
    case all
    case rated
    case casual

    var label: String {
        switch self {
        case .all: return "All games"
        case .rated: return "Rated"
        case .casual: return "Casual"
        }
    }

    func includes(_ row: LichessBotGameSummary) -> Bool {
        switch self {
        case .all: return true
        case .rated: return row.rated
        case .casual: return !row.rated
        }
    }
}

/// One value per period. A struct with a stored property per case rather
/// than a dictionary, so a lookup can't come back empty and no view needs a
/// default for a missing period.
struct LichessBotPeriodValues<Value> {
    var lastHour: Value
    var today: Value
    var thisWeek: Value
    var thisMonth: Value
    var thisYear: Value
    var allTime: Value

    init(_ make: (LichessBotStatsPeriod) throws -> Value) rethrows {
        lastHour = try make(.lastHour)
        today = try make(.today)
        thisWeek = try make(.thisWeek)
        thisMonth = try make(.thisMonth)
        thisYear = try make(.thisYear)
        allTime = try make(.allTime)
    }

    subscript(period: LichessBotStatsPeriod) -> Value {
        switch period {
        case .lastHour: return lastHour
        case .today: return today
        case .thisWeek: return thisWeek
        case .thisMonth: return thisMonth
        case .thisYear: return thisYear
        case .allTime: return allTime
        }
    }

    /// Mutate one period's value in place. Not a subscript setter: a
    /// get-and-set copies the value out and back, and when the value holds
    /// arrays (the statistics accumulators) every mutation would copy them
    /// — quadratic over the games. `inout` on the stored property mutates
    /// it where it lies.
    mutating func update(_ period: LichessBotStatsPeriod, _ body: (inout Value) -> Void) {
        switch period {
        case .lastHour: body(&lastHour)
        case .today: body(&today)
        case .thisWeek: body(&thisWeek)
        case .thisMonth: body(&thisMonth)
        case .thisYear: body(&thisYear)
        case .allTime: body(&allTime)
        }
    }

    func map<Other>(_ transform: (Value) throws -> Other) rethrows -> LichessBotPeriodValues<Other> {
        try LichessBotPeriodValues<Other> { try transform(self[$0]) }
    }
}

extension LichessBotPeriodValues: Sendable where Value: Sendable {}
extension LichessBotPeriodValues: Equatable where Value: Equatable {}

/// One value per filter, for the same reason as `LichessBotPeriodValues`.
struct LichessBotFilterValues<Value> {
    var all: Value
    var rated: Value
    var casual: Value

    init(_ make: (LichessBotStatsFilter) throws -> Value) rethrows {
        all = try make(.all)
        rated = try make(.rated)
        casual = try make(.casual)
    }

    subscript(filter: LichessBotStatsFilter) -> Value {
        switch filter {
        case .all: return all
        case .rated: return rated
        case .casual: return casual
        }
    }
}

extension LichessBotFilterValues: Sendable where Value: Sendable {}
extension LichessBotFilterValues: Equatable where Value: Equatable {}
