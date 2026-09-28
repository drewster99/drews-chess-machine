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

    var isBot: Bool {
        title == "BOT"
    }

    func rating(_ perf: String) -> LichessBotPerfRating? {
        perfs?[perf]
    }
}
