import Foundation

/// A game from `GET /game/export/{id}` as JSON, requested with `clocks`,
/// `opening` and `pgnInJson` (plan §10.2). Lichess's authoritative account
/// of the game, used to reconcile the journal. Decoding is tolerant: every
/// field that some game shapes omit is optional, and unknown enum values
/// are kept raw.
struct LichessBotGameExport: Sendable, Codable, Equatable {
    struct User: Sendable, Codable, Equatable {
        let id: String
        let name: String?
        let title: String?
    }

    struct Player: Sendable, Codable, Equatable {
        /// Absent for Lichess's own AI.
        let user: User?
        let rating: Int?
        let ratingDiff: Int?
        let provisional: Bool?
        let aiLevel: Int?
    }

    struct Players: Sendable, Codable, Equatable {
        let white: Player
        let black: Player
    }

    struct Opening: Sendable, Codable, Equatable {
        let eco: String?
        let name: String?
        let ply: Int?
    }

    struct Clock: Sendable, Codable, Equatable {
        /// Seconds.
        let initial: Int?
        /// Seconds.
        let increment: Int?
    }

    let id: String
    let rated: Bool?
    let variant: String?
    let speed: String?
    let perf: String?
    /// Milliseconds since the epoch.
    let createdAt: Int64?
    let lastMoveAt: Int64?
    let status: LichessBotOpenValue<LichessBotGameStatusName>
    let winner: LichessBotOpenValue<LichessBotColorName>?
    let players: Players
    let opening: Opening?
    /// Space-separated SAN.
    let moves: String?
    /// Remaining clock after each move, in centiseconds.
    let clocks: [Int]?
    let clock: Clock?
    let pgn: String?

    var sanMoves: [String] {
        (moves ?? "").split(separator: " ", omittingEmptySubsequences: true).map(String.init)
    }

    func player(_ color: LichessBotColorName) -> Player {
        switch color {
        case .white: return players.white
        case .black: return players.black
        }
    }

    static func decode(_ data: Data) throws -> LichessBotGameExport {
        try JSONDecoder().decode(LichessBotGameExport.self, from: data)
    }
}
