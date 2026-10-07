import Foundation

/// DCM's own games per speed, from its game records: unrated games all
/// time, and every game started today (this Mac's calendar) and in the 24
/// hours before now. Nil until the games index has loaded.
struct LichessBotAccountGameCounts: Equatable {
    let unrated: [String: Int]
    let today: [String: Int]
    let lastDay: [String: Int]

    init?(rows: [LichessBotGameSummary]?, now: Date, calendar: Calendar) {
        guard let rows else { return nil }
        let startOfToday = calendar.startOfDay(for: now)
        let dayAgo = now.addingTimeInterval(-24 * 3600)
        var unrated: [String: Int] = [:]
        var today: [String: Int] = [:]
        var lastDay: [String: Int] = [:]
        for row in rows {
            if !row.rated { unrated[row.speed, default: 0] += 1 }
            if row.createdAt >= startOfToday { today[row.speed, default: 0] += 1 }
            if row.createdAt >= dayAgo { lastDay[row.speed, default: 0] += 1 }
        }
        self.unrated = unrated
        self.today = today
        self.lastDay = lastDay
    }
}
