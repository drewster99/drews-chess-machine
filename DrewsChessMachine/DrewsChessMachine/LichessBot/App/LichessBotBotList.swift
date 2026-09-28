import Foundation

/// One row of the Challenge sheet's bot tables (plan §7.2): an online bot
/// from `GET /api/bot/online`, or a favorite that may be offline.
struct LichessBotBotRow: Identifiable, Sendable, Equatable {
    /// Lowercased Lichess user id.
    let id: String
    let username: String
    let isFavorite: Bool
    /// True when the bot is in the online list or its status says online;
    /// false when its status says offline; nil when unknown.
    let isOnline: Bool?
    /// Ratings, profile and so on; nil for a favorite not in the online
    /// list (only its status is known).
    let summary: LichessBotUserSummary?
    /// When Lichess said this bot can be challenged again, if still ahead.
    let limitUntil: Date?

    // Sort keys for `Table`; unrated or unknown sorts below every value.
    var favoriteSortKey: Int { isFavorite ? 0 : 1 }
    var usernameSortKey: String { username.lowercased() }
    var blitzSortKey: Int { summary?.blitzSortKey ?? Int.min }
    var rapidSortKey: Int { summary?.rapidSortKey ?? Int.min }
    var gamesSortKey: Int { summary?.blitzAndRapidGames ?? Int.min }
}

/// The Challenge sheet's search and rating filter (plan §7.2).
struct LichessBotBotListFilter: Sendable, Equatable {
    /// Case-insensitive substring over username, real name and bio.
    var search = ""
    /// Bounds on the rating for `speed`; nil means unbounded.
    var minimumRating: Int?
    var maximumRating: Int?
    /// The selected time control's speed, whose rating the bounds apply to.
    var speed: LichessBotSpeed = .blitz
    var hidesProvisional = false

    var isActive: Bool {
        !search.trimmingCharacters(in: .whitespaces).isEmpty || minimumRating != nil || maximumRating != nil || hidesProvisional
    }

    func matches(_ row: LichessBotBotRow) -> Bool {
        let query = search.trimmingCharacters(in: .whitespaces).lowercased()
        if !query.isEmpty {
            let fields = [row.username, row.summary?.profile?.realName, row.summary?.profile?.bio]
            guard fields.contains(where: { $0?.lowercased().contains(query) == true }) else { return false }
        }
        guard minimumRating != nil || maximumRating != nil || hidesProvisional else { return true }
        // A rating filter needs a rating: a row without one for this speed
        // (unrated, or an offline favorite with no summary) doesn't match.
        guard let perf = row.summary?.rating(speed.rawValue), let rating = perf.rating else { return false }
        if hidesProvisional && perf.prov == true { return false }
        if let minimumRating, rating < minimumRating { return false }
        if let maximumRating, rating > maximumRating { return false }
        return true
    }
}

enum LichessBotBotList {

    /// Rows for the Online Bots tab: every online bot, marked with favorites
    /// and limit times.
    static func onlineRows(bots: [LichessBotUserSummary], notes: LichessBotPlayerNotes?, now: Date) -> [LichessBotBotRow] {
        bots.map { bot in
            LichessBotBotRow(
                id: bot.id.lowercased(),
                username: bot.username,
                isFavorite: notes?.isFavorite(bot.id) == true,
                isOnline: true,
                summary: bot,
                limitUntil: notes?.limitUntil(bot.id, now: now)
            )
        }
    }

    /// Rows for the Favorites tab, in starring order: online favorites from
    /// the online list, the rest from their status (nil when unknown).
    static func favoriteRows(notes: LichessBotPlayerNotes?, bots: [LichessBotUserSummary], statuses: [String: LichessBotUserStatus], now: Date) -> [LichessBotBotRow] {
        guard let notes else { return [] }
        let online = Dictionary(bots.map { ($0.id.lowercased(), $0) }, uniquingKeysWith: { first, _ in first })
        return notes.favoriteIDs.map { id in
            if let bot = online[id] {
                return LichessBotBotRow(id: id, username: bot.username, isFavorite: true, isOnline: true, summary: bot, limitUntil: notes.limitUntil(id, now: now))
            }
            let status = statuses[id]
            return LichessBotBotRow(
                id: id,
                username: status?.name ?? id,
                isFavorite: true,
                isOnline: status.map { $0.online == true },
                summary: nil,
                limitUntil: notes.limitUntil(id, now: now)
            )
        }
    }

    /// Filtered, then favorites first, then the table's sort order.
    static func ordered(_ rows: [LichessBotBotRow], filter: LichessBotBotListFilter, sortOrder: [KeyPathComparator<LichessBotBotRow>]) -> [LichessBotBotRow] {
        rows.filter(filter.matches)
            .sorted(using: [KeyPathComparator(\LichessBotBotRow.favoriteSortKey)] + sortOrder)
    }
}
