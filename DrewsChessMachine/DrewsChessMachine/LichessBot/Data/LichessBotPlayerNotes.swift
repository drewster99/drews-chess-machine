import Foundation

/// What the operator and Lichess have told us about other players (plan
/// §7.2, §7.3): favorite bots, when a bot is available again after its
/// bot-vs-bot daily limit, and matchmaking's decline cool-downs. Persisted to `player-notes.json`, apart from
/// Settings. Ids are lowercased Lichess user ids.
struct LichessBotPlayerNotes: Sendable, Codable, Equatable {
    /// In the order they were starred.
    var favoriteIDs: [String] = []
    /// When each bot can be challenged again, from Lichess's own refusal.
    var botLimitUntil: [String: Date] = [:]
    /// When each player's decline cool-down ends: matchmaking leaves a
    /// player who declined DCM alone until then (plan §7.3). Optional
    /// because notes files written before cool-downs existed have no such
    /// key, and a favorites file must never become unreadable; absent means
    /// no cool-down has been recorded.
    var declineCooldownUntil: [String: Date]?

    func isFavorite(_ userID: String) -> Bool {
        favoriteIDs.contains(userID.lowercased())
    }

    mutating func toggleFavorite(_ userID: String) {
        let id = userID.lowercased()
        if let index = favoriteIDs.firstIndex(of: id) {
            favoriteIDs.remove(at: index)
        } else {
            favoriteIDs.append(id)
        }
    }

    /// The bot's limit time if it is still in the future at `now`.
    func limitUntil(_ userID: String, now: Date) -> Date? {
        guard let until = botLimitUntil[userID.lowercased()], until > now else { return nil }
        return until
    }

    /// The player's decline cool-down end if it is still in the future at
    /// `now`.
    func declineCooldownEnds(_ userID: String, now: Date) -> Date? {
        guard let until = declineCooldownUntil?[userID.lowercased()], until > now else { return nil }
        return until
    }

    /// Start (or restart) a player's decline cool-down.
    mutating func recordDeclineCooldown(_ userID: String, until: Date) {
        var cooldowns = declineCooldownUntil ?? [:]
        cooldowns[userID.lowercased()] = until
        declineCooldownUntil = cooldowns
    }

    /// Drop limit times and decline cool-downs that have passed.
    mutating func pruneExpiredLimits(now: Date) {
        botLimitUntil = botLimitUntil.filter { $0.value > now }
        if let cooldowns = declineCooldownUntil {
            declineCooldownUntil = cooldowns.filter { $0.value > now }
        }
    }

    static func load(from url: URL) throws -> LichessBotPlayerNotes {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return LichessBotPlayerNotes()
        }
        return try decoder.decode(LichessBotPlayerNotes.self, from: Data(contentsOf: url))
    }

    func save(to url: URL) throws {
        try LichessBotAtomicWrite.write(Self.encoder.encode(self), to: url)
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}

/// Lichess's refusal of a challenge to a bot that has hit its bot-vs-bot
/// daily limit: "<user> played <count> games against other bots today,
/// please wait until <ISO-8601 time> to challenge them." (the exact text is
/// in the tests). No endpoint reports the limit, so this text is the only
/// source of the exact time.
enum LichessBotBotLimitRefusal {
    struct Parsed: Equatable {
        let userID: String
        let gamesPlayed: Int
        let until: Date
    }

    static func parse(_ message: String) -> Parsed? {
        let pattern = #/^(\S+) played (\d+) games against other bots today, please wait until (\S+) to challenge them\.?$/#
        guard let match = message.trimmingCharacters(in: .whitespacesAndNewlines).wholeMatch(of: pattern),
              let games = Int(match.2),
              let until = parseDate(String(match.3)) else {
            return nil
        }
        return Parsed(userID: String(match.1).lowercased(), gamesPlayed: games, until: until)
    }

    private static func parseDate(_ text: String) -> Date? {
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: text) {
            return date
        }
        let whole = ISO8601DateFormatter()
        whole.formatOptions = [.withInternetDateTime]
        return whole.date(from: text)
    }
}
