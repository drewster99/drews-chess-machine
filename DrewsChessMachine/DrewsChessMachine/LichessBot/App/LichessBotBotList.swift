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
    /// DCM's results against this player from its own records; nil when
    /// they have never played.
    var record: LichessBotResultTally? = nil
    /// Lichess title ("BOT", "GM", …), when known.
    var title: String? = nil
    /// Playing a game right now, from a status fetch; nil when unknown.
    var isPlaying: Bool? = nil

    // Sort keys for `Table`; unrated or unknown sorts below every value.
    var favoriteSortKey: Int { isFavorite ? 0 : 1 }
    var usernameSortKey: String { username.lowercased() }
    var bulletSortKey: Int { summary?.bulletSortKey ?? Int.min }
    var blitzSortKey: Int { summary?.blitzSortKey ?? Int.min }
    var rapidSortKey: Int { summary?.rapidSortKey ?? Int.min }
    var gamesSortKey: Int { summary?.ratedSpeedGames ?? Int.min }
    var bestSortKey: Int { summary?.bestSpeed?.rating ?? Int.min }
    /// Games against DCM, then DCM's score, for sorting the "vs DCM" column.
    var recordSortKey: Double {
        guard let record, record.games > 0 else { return -1 }
        return Double(record.games) + (record.score ?? 0) / 2
    }
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

    /// A typed rating bound: a whole number, or nil for empty or anything
    /// that isn't one (the field shows the latter as invalid).
    static func bound(from text: String) -> Int? {
        Int(text.trimmingCharacters(in: .whitespaces))
    }

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

    /// Rows for the Online Bots and Online Players tabs: every listed
    /// player, marked with favorites, limit times, and whether they are
    /// playing (from `statuses`; the online lists don't say).
    static func onlineRows(bots: [LichessBotUserSummary], notes: LichessBotPlayerNotes?, now: Date, records: [String: LichessBotResultTally] = [:], statuses: [String: LichessBotUserStatus] = [:]) -> [LichessBotBotRow] {
        bots.map { bot in
            let id = bot.id.lowercased()
            return LichessBotBotRow(
                id: id,
                username: bot.username,
                isFavorite: notes?.isFavorite(bot.id) == true,
                // Listed means online; a status fetched since outranks the list.
                isOnline: statuses[id].map { $0.online == true } ?? true,
                summary: bot,
                limitUntil: notes?.limitUntil(bot.id, now: now),
                record: records[id],
                title: bot.title,
                isPlaying: statuses[id].map { $0.playing == true }
            )
        }
    }

    /// Rows for the Favorites tab, in starring order: online favorites from
    /// the online list, the rest from their status (nil when unknown).
    static func favoriteRows(notes: LichessBotPlayerNotes?, bots: [LichessBotUserSummary], statuses: [String: LichessBotUserStatus], now: Date, records: [String: LichessBotResultTally] = [:]) -> [LichessBotBotRow] {
        guard let notes else { return [] }
        let online = Dictionary(bots.map { ($0.id.lowercased(), $0) }, uniquingKeysWith: { first, _ in first })
        return notes.favoriteIDs.map { id in
            let status = statuses[id]
            let isPlaying = status.map { $0.playing == true }
            if let bot = online[id] {
                return LichessBotBotRow(id: id, username: bot.username, isFavorite: true, isOnline: status.map { $0.online == true } ?? true, summary: bot, limitUntil: notes.limitUntil(id, now: now), record: records[id], title: bot.title, isPlaying: isPlaying)
            }
            // Not in the online list: its status, if fetched, is all we know.
            // Without one the name falls back to the stored lowercased id.
            return LichessBotBotRow(
                id: id,
                username: status?.name ?? id,
                isFavorite: true,
                isOnline: status.map { $0.online == true },
                summary: nil,
                limitUntil: notes.limitUntil(id, now: now),
                record: records[id],
                title: status?.title,
                isPlaying: isPlaying
            )
        }
    }

    /// Filtered, then favorites first, then the table's sort order.
    static func ordered(_ rows: [LichessBotBotRow], filter: LichessBotBotListFilter, sortOrder: [KeyPathComparator<LichessBotBotRow>]) -> [LichessBotBotRow] {
        rows.filter(filter.matches)
            .sorted(using: [KeyPathComparator(\LichessBotBotRow.favoriteSortKey)] + sortOrder)
    }
}
