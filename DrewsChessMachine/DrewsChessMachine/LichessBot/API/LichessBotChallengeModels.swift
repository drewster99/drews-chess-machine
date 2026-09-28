import Foundation

/// A challenge DCM sends (plan §7.1).
struct LichessBotOutgoingChallenge: Sendable, Equatable, Codable {
    var rated: Bool
    var clockLimitSeconds: Int
    var clockIncrementSeconds: Int
    var color: LichessBotChallengeColorName

    /// The created challenge from `POST /api/challenge/{username}`. Lichess
    /// has answered both with the challenge object itself and wrapped as
    /// `{"challenge": …}`; either is accepted, and anything else is an error.
    /// Which one the live API sends is a Phase 5 live-verification item.
    static func decodeCreated(_ body: Data) throws -> LichessBotChallenge {
        let decoder = JSONDecoder()
        let wrappedError: Error
        do {
            return try decoder.decode(Wrapped.self, from: body).challenge
        } catch {
            wrappedError = error
        }
        do {
            return try decoder.decode(LichessBotChallenge.self, from: body)
        } catch {
            throw LichessBotAPIError.undecodableResponse(
                endpoint: "/api/challenge",
                detail: "neither {\"challenge\": …} (\(wrappedError)) nor a bare challenge (\(error))"
            )
        }
    }

    private struct Wrapped: Decodable {
        let challenge: LichessBotChallenge
    }
}

extension LichessBotChallengeColorName: Codable {}

/// A player's public summary: `GET /api/user/{username}` and each line of
/// `GET /api/bot/online`.
struct LichessBotUserSummary: Sendable, Hashable, Codable, Identifiable {
    let id: String
    let username: String
    let title: String?
    /// Keyed by perf name (`blitz`, `rapid`, …).
    let perfs: [String: LichessBotPerfRating]?
    let online: Bool?
    let disabled: Bool?
    let tosViolation: Bool?
    /// Milliseconds since 1970.
    let createdAt: Int64?
    let seenAt: Int64?
    let profile: LichessBotUserProfile?
    // `GET /api/user/{username}` only (the online-bots list omits them).
    var count: LichessBotAccountCount? = nil
    var verified: Bool? = nil
    var patronColor: Int? = nil
    var flair: String? = nil
    var playTime: LichessBotPlayTime? = nil

    var isBot: Bool {
        title == "BOT"
    }

    /// The bio's first non-empty line, for one-line display.
    var bioFirstLine: String? {
        profile?.bio?
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty }
    }

    func rating(_ perf: String) -> LichessBotPerfRating? {
        perfs?[perf]
    }

    /// Display text for a perf's rating: four digits then "?" when
    /// provisional or a space (so a column of them aligns), or "–" when the
    /// player has none.
    func ratingText(_ perf: String) -> String {
        guard let rating = rating(perf), let value = rating.rating else { return "–" }
        return String(format: "%4d", value) + (rating.prov == true ? "?" : " ")
    }

    /// Sort keys for tables. A player with no rating in a perf sorts below
    /// every rated one.
    var blitzSortKey: Int { rating("blitz")?.rating ?? Int.min }
    var rapidSortKey: Int { rating("rapid")?.rating ?? Int.min }
    var usernameSortKey: String { username.lowercased() }
    var blitzAndRapidGames: Int { (rating("blitz")?.games ?? 0) + (rating("rapid")?.games ?? 0) }
}

/// Seconds played in total, and against humans.
struct LichessBotPlayTime: Sendable, Hashable, Codable {
    let total: Int
    let human: Int?
}

/// `GET /api/crosstable/{user1}/{user2}`: each user's total score (a win
/// counts 1, a draw ½) across all their games against each other.
struct LichessBotCrosstable: Sendable, Hashable, Codable {
    let users: [String: Double]
    let nbGames: Int
}

/// One match of `GET /api/player/autocomplete?object=true`.
struct LichessBotLightUser: Sendable, Hashable, Codable, Identifiable {
    let id: String
    let name: String
    let title: String?
    /// Present when Lichess knows the player is connected.
    let online: Bool?
}

/// One entry of `GET /api/player/top/{nb}/{perfType}`.
struct LichessBotLeaderboardUser: Sendable, Hashable, Codable, Identifiable {
    struct Perf: Sendable, Hashable, Codable {
        let rating: Int
        let progress: Int
    }

    let id: String
    let username: String
    let title: String?
    let online: Bool?
    let perfs: [String: Perf]?

    func perf(_ speed: String) -> Perf? {
        perfs?[speed]
    }
}

/// One line of `GET /api/bot/game/{id}/chat`: the player room's messages
/// in order, without times.
struct LichessBotFetchedChatLine: Sendable, Hashable, Codable {
    let text: String
    let user: String
}

/// The public profile a player wrote. Bots often describe their engine in
/// `bio`. Shown as plain text only; `links` is never followed.
struct LichessBotUserProfile: Sendable, Hashable, Codable {
    let bio: String?
    let realName: String?
    let flag: String?
    let location: String?
    let fideRating: Int?
    let uscfRating: Int?
    let ecfRating: Int?
}

/// One entry of `GET /api/users/status`. `online` is absent rather than
/// false for a player who isn't connected.
struct LichessBotUserStatus: Sendable, Hashable, Codable {
    let id: String
    let name: String
    let title: String?
    let online: Bool?
    let playing: Bool?
}
