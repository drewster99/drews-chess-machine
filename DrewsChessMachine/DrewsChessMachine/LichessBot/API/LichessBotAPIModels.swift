import Foundation

// Decodable models for the Lichess Bot API payloads DCM consumes. Field
// names and shapes follow the Lichess OpenAPI spec
// (github.com/lichess-org/api, doc/specs/schemas/*.yaml).
//
// Decoding is deliberately tolerant. Lichess adds enum values and fields
// over time; a strict decoder would fail an entire game-stream line over one
// unfamiliar value and drop the game. So every enum-like field is a
// `LichessBotOpenValue`, which keeps the raw string and exposes the known
// case when there is one; fields the spec marks optional are optional; and
// unknown line `type`s decode to an `.unknown` case carrying the raw bytes
// (plan E28).

// MARK: - Open enum wrapper

/// A string value from a Lichess enum that decodes whatever the server sent.
/// `known` is the matching case of `Known`, or nil for a value this build
/// does not know — which callers must treat as "unknown", never as a
/// particular known case.
struct LichessBotOpenValue<Known: RawRepresentable & Sendable & Hashable>: Sendable, Hashable, Codable, CustomStringConvertible
where Known.RawValue == String {
    let raw: String

    init(raw: String) {
        self.raw = raw
    }

    init(_ known: Known) {
        self.raw = known.rawValue
    }

    init(from decoder: Decoder) throws {
        raw = try decoder.singleValueContainer().decode(String.self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(raw)
    }

    var known: Known? {
        Known(rawValue: raw)
    }

    var description: String {
        raw
    }
}

// MARK: - Enumerations

enum LichessBotGameStatusName: String, Sendable, Hashable, CaseIterable {
    case created
    case started
    case aborted
    case mate
    case resign
    case stalemate
    case timeout
    case draw
    case outOfTime = "outoftime"
    case cheat
    case noStart
    case unknownFinish
    case insufficientMaterialClaim
    case variantEnd
}

extension LichessBotOpenValue where Known == LichessBotGameStatusName {
    /// Whether the game is still being played: true for `created`/`started`,
    /// false for every known finishing status, and nil for a status this
    /// build does not recognise. Callers resolve nil through the export API
    /// rather than guessing (plan §6, E28).
    var isLive: Bool? {
        guard let known else { return nil }
        switch known {
        case .created, .started:
            return true
        case .aborted, .mate, .resign, .stalemate, .timeout, .draw, .outOfTime, .cheat,
             .noStart, .unknownFinish, .insufficientMaterialClaim, .variantEnd:
            return false
        }
    }
}

enum LichessBotColorName: String, Sendable, Hashable, CaseIterable {
    case white
    case black
}

enum LichessBotChallengeColorName: String, Sendable, Hashable, CaseIterable {
    case white
    case black
    case random
}

enum LichessBotVariantKey: String, Sendable, Hashable, CaseIterable {
    case standard
    case chess960
    case crazyhouse
    case antichess
    case atomic
    case horde
    case kingOfTheHill
    case racingKings
    case threeCheck
    case fromPosition
}

enum LichessBotSpeed: String, Sendable, Hashable, CaseIterable, Codable, Comparable {
    case ultraBullet
    case bullet
    case blitz
    case rapid
    case classical
    case correspondence

    /// Fastest first, so "faster than every allowed speed" is a comparison.
    static func < (lhs: LichessBotSpeed, rhs: LichessBotSpeed) -> Bool {
        guard let left = allCases.firstIndex(of: lhs), let right = allCases.firstIndex(of: rhs) else {
            return false
        }
        return left < right
    }
}

enum LichessBotTimeControlType: String, Sendable, Hashable, CaseIterable {
    case clock
    case correspondence
    case unlimited
}

enum LichessBotChallengeStatus: String, Sendable, Hashable, CaseIterable {
    case created
    case offline
    case canceled
    case declined
    case accepted
}

enum LichessBotChallengeDirection: String, Sendable, Hashable, CaseIterable {
    case incoming = "in"
    case outgoing = "out"
}

enum LichessBotChatRoom: String, Sendable, Hashable, CaseIterable {
    case player
    case spectator
}

/// Reasons Lichess accepts on `POST /api/challenge/{id}/decline`. Any other
/// value is treated by Lichess as `generic`.
enum LichessBotDeclineReason: String, Sendable, Hashable, CaseIterable, Codable {
    case generic
    case later
    case tooFast
    case tooSlow
    case timeControl
    case rated
    case casual
    case standard
    case variant
    case noBot
    case onlyBot
}

// MARK: - Shared pieces

struct LichessBotVariant: Sendable, Hashable, Codable {
    let key: LichessBotOpenValue<LichessBotVariantKey>
    let name: String?
    let short: String?
}

struct LichessBotCompat: Sendable, Hashable, Codable {
    /// Whether the challenge or game can be played through the Bot API.
    let bot: Bool?
    let board: Bool?
}

// MARK: - Game stream (`GET /api/bot/game/stream/{gameId}`)

/// A player in `gameFull`.
struct LichessBotGamePlayer: Sendable, Hashable, Codable {
    let id: String?
    let name: String?
    let title: String?
    let rating: Int?
    let provisional: Bool?
    /// Set when the player is Lichess's built-in AI.
    let aiLevel: Int?
}

struct LichessBotGameClock: Sendable, Hashable, Codable {
    let initial: LichessBotMilliseconds
    let increment: LichessBotMilliseconds
}

struct LichessBotGameExpiration: Sendable, Hashable, Codable {
    let idleMillis: LichessBotMilliseconds
    let millisToMove: LichessBotMilliseconds
}

/// `gameState`: the mutable part of a game.
struct LichessBotGameState: Sendable, Hashable, Codable {
    /// Space-separated UCI moves from the initial position. Castling may be
    /// in the king-to-rook form (plan E1). Empty at the start (plan E2).
    let moves: String
    let wtime: LichessBotMilliseconds
    let btime: LichessBotMilliseconds
    let winc: LichessBotMilliseconds
    let binc: LichessBotMilliseconds
    let status: LichessBotOpenValue<LichessBotGameStatusName>
    let winner: LichessBotOpenValue<LichessBotColorName>?
    let wdraw: Bool?
    let bdraw: Bool?
    let wtakeback: Bool?
    let btakeback: Bool?
    let expiration: LichessBotGameExpiration?

    /// The move list as tokens. An empty or all-whitespace string is zero
    /// moves, never one empty token (plan E2).
    var moveTokens: [String] {
        moves.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
    }

    /// Lichess omits the draw and takeback flags when false (plan E29).
    func isOfferingDraw(_ color: LichessBotColorName) -> Bool {
        switch color {
        case .white: return wdraw ?? false
        case .black: return bdraw ?? false
        }
    }

    func isProposingTakeback(_ color: LichessBotColorName) -> Bool {
        switch color {
        case .white: return wtakeback ?? false
        case .black: return btakeback ?? false
        }
    }

    func remaining(for color: LichessBotColorName) -> LichessBotMilliseconds {
        switch color {
        case .white: return wtime
        case .black: return btime
        }
    }
}

struct LichessBotGamePerf: Sendable, Hashable, Codable {
    let name: String?
}

/// `gameFull`: the complete game description, always the first line of a
/// game stream (and so of every reconnect).
struct LichessBotGameFull: Sendable, Hashable, Codable {
    let id: String
    let variant: LichessBotVariant
    let clock: LichessBotGameClock?
    let speed: LichessBotOpenValue<LichessBotSpeed>
    let perf: LichessBotGamePerf?
    let rated: Bool
    let createdAt: Int64
    let white: LichessBotGamePlayer
    let black: LichessBotGamePlayer
    /// `"startpos"` or a FEN. The spec gives `"startpos"` as the default when
    /// the field is absent.
    let initialFen: String
    let state: LichessBotGameState
    let daysPerTurn: Int?
    let tournamentId: String?

    private enum CodingKeys: String, CodingKey {
        case id, variant, clock, speed, perf, rated, createdAt, white, black, initialFen, state, daysPerTurn, tournamentId
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        variant = try container.decode(LichessBotVariant.self, forKey: .variant)
        clock = try container.decodeIfPresent(LichessBotGameClock.self, forKey: .clock)
        speed = try container.decode(LichessBotOpenValue<LichessBotSpeed>.self, forKey: .speed)
        perf = try container.decodeIfPresent(LichessBotGamePerf.self, forKey: .perf)
        rated = try container.decode(Bool.self, forKey: .rated)
        createdAt = try container.decode(Int64.self, forKey: .createdAt)
        white = try container.decode(LichessBotGamePlayer.self, forKey: .white)
        black = try container.decode(LichessBotGamePlayer.self, forKey: .black)
        // Spec: `initialFen` defaults to "startpos".
        initialFen = try container.decodeIfPresent(String.self, forKey: .initialFen) ?? "startpos"
        state = try container.decode(LichessBotGameState.self, forKey: .state)
        daysPerTurn = try container.decodeIfPresent(Int.self, forKey: .daysPerTurn)
        tournamentId = try container.decodeIfPresent(String.self, forKey: .tournamentId)
    }
}

struct LichessBotChatLine: Sendable, Hashable, Codable {
    let room: LichessBotOpenValue<LichessBotChatRoom>
    let username: String
    let text: String
}

struct LichessBotOpponentGone: Sendable, Hashable, Codable {
    let gone: Bool
    let claimWinInSeconds: Int?
}

/// One decoded line of a game stream.
enum LichessBotGameStreamLine: Sendable {
    case gameFull(LichessBotGameFull)
    case gameState(LichessBotGameState)
    case chatLine(LichessBotChatLine)
    case opponentGone(LichessBotOpponentGone)
    /// A line whose `type` this build does not know. Logged, never fatal.
    case unknown(type: String)

    static func decode(_ data: Data) throws -> LichessBotGameStreamLine {
        let decoder = JSONDecoder()
        let tag = try decoder.decode(LichessBotTypeTag.self, from: data)
        switch tag.type {
        case "gameFull": return .gameFull(try decoder.decode(LichessBotGameFull.self, from: data))
        case "gameState": return .gameState(try decoder.decode(LichessBotGameState.self, from: data))
        case "chatLine": return .chatLine(try decoder.decode(LichessBotChatLine.self, from: data))
        case "opponentGone": return .opponentGone(try decoder.decode(LichessBotOpponentGone.self, from: data))
        default: return .unknown(type: tag.type)
        }
    }
}

// MARK: - Event stream (`GET /api/stream/event`)

struct LichessBotGameEventStatus: Sendable, Hashable, Codable {
    let id: Int?
    let name: LichessBotOpenValue<LichessBotGameStatusName>
}

/// The opponent in a `gameStart`/`gameFinish` event: a player, or Lichess's
/// AI (`id` null, `ai` set).
struct LichessBotGameEventOpponent: Sendable, Hashable, Codable {
    let id: String?
    let username: String?
    let rating: Int?
    let ratingDiff: Int?
    let ai: Int?
}

/// The `game` payload of `gameStart` / `gameFinish`.
///
/// `fullId` is deliberately not decoded: it embeds a player-specific secret
/// suffix, and a field that is never decoded can never reach a record or a
/// log (plan E26).
struct LichessBotGameEventInfo: Sendable, Hashable, Codable {
    let gameId: String
    let fen: String?
    let color: LichessBotOpenValue<LichessBotColorName>?
    let lastMove: String?
    let source: String?
    let status: LichessBotGameEventStatus?
    let variant: LichessBotVariant?
    let speed: LichessBotOpenValue<LichessBotSpeed>?
    let perf: String?
    let rating: Int?
    let rated: Bool?
    let hasMoved: Bool?
    let opponent: LichessBotGameEventOpponent?
    let isMyTurn: Bool?
    let secondsLeft: Int?
    let winner: LichessBotOpenValue<LichessBotColorName>?
    let ratingDiff: Int?
    let compat: LichessBotCompat?
    let tournamentId: String?
}

struct LichessBotChallengeUser: Sendable, Hashable, Codable {
    let id: String
    let name: String
    let rating: Int?
    let title: String?
    let provisional: Bool?
    let online: Bool?
    let lag: Int?
}

/// A challenge's time control. `limit`/`increment` are seconds (plan E17).
struct LichessBotTimeControl: Sendable, Hashable, Codable {
    let type: LichessBotOpenValue<LichessBotTimeControlType>
    let limit: LichessBotSeconds?
    let increment: LichessBotSeconds?
    let daysPerTurn: Int?
    let show: String?
}

struct LichessBotChallengePerf: Sendable, Hashable, Codable {
    let icon: String?
    let name: String?
}

struct LichessBotChallenge: Sendable, Hashable, Codable {
    let id: String
    let url: String?
    let status: LichessBotOpenValue<LichessBotChallengeStatus>
    let challenger: LichessBotChallengeUser
    let destUser: LichessBotChallengeUser?
    let variant: LichessBotVariant
    let rated: Bool
    let speed: LichessBotOpenValue<LichessBotSpeed>
    let timeControl: LichessBotTimeControl
    let color: LichessBotOpenValue<LichessBotChallengeColorName>
    let finalColor: LichessBotOpenValue<LichessBotColorName>?
    let perf: LichessBotChallengePerf?
    let direction: LichessBotOpenValue<LichessBotChallengeDirection>?
    let initialFen: String?
    let rematchOf: String?
}

/// The minimal part of a canceled/declined challenge payload DCM uses.
struct LichessBotChallengeReference: Sendable, Hashable, Codable {
    let id: String
}

/// One decoded line of the event stream.
enum LichessBotEvent: Sendable {
    case gameStart(LichessBotGameEventInfo)
    case gameFinish(LichessBotGameEventInfo)
    case challenge(LichessBotChallenge, compat: LichessBotCompat?)
    case challengeCanceled(LichessBotChallengeReference)
    case challengeDeclined(LichessBotChallengeReference)
    /// An event `type` this build does not know. Logged, never fatal.
    case unknown(type: String)

    static func decode(_ data: Data) throws -> LichessBotEvent {
        let decoder = JSONDecoder()
        let tag = try decoder.decode(LichessBotTypeTag.self, from: data)
        switch tag.type {
        case "gameStart":
            return .gameStart(try decoder.decode(GamePayload.self, from: data).game)
        case "gameFinish":
            return .gameFinish(try decoder.decode(GamePayload.self, from: data).game)
        case "challenge":
            let payload = try decoder.decode(ChallengePayload.self, from: data)
            return .challenge(payload.challenge, compat: payload.compat)
        case "challengeCanceled":
            return .challengeCanceled(try decoder.decode(ChallengeReferencePayload.self, from: data).challenge)
        case "challengeDeclined":
            return .challengeDeclined(try decoder.decode(ChallengeReferencePayload.self, from: data).challenge)
        default:
            return .unknown(type: tag.type)
        }
    }

    private struct GamePayload: Decodable {
        let game: LichessBotGameEventInfo
    }

    private struct ChallengePayload: Decodable {
        let challenge: LichessBotChallenge
        let compat: LichessBotCompat?
    }

    private struct ChallengeReferencePayload: Decodable {
        let challenge: LichessBotChallengeReference
    }
}

/// Every NDJSON line carries a `type` discriminator.
private struct LichessBotTypeTag: Decodable {
    let type: String
}

// MARK: - Account and token

struct LichessBotPerfRating: Sendable, Hashable, Codable {
    let games: Int?
    let rating: Int?
    let rd: Int?
    let prog: Int?
    let prov: Bool?
}

struct LichessBotAccountCount: Sendable, Hashable, Codable {
    let all: Int
    let rated: Int?
    let win: Int?
    let loss: Int?
    let draw: Int?
    let playing: Int?
}

/// `GET /api/account`. `title == "BOT"` means the account has been upgraded.
struct LichessBotAccount: Sendable, Hashable, Codable {
    let id: String
    let username: String
    let title: String?
    let count: LichessBotAccountCount?
    /// Keyed by perf name (`blitz`, `rapid`, …). Puzzle modes decode here too,
    /// with only the fields they share.
    let perfs: [String: LichessBotPerfRating]?
    let disabled: Bool?
    let tosViolation: Bool?
    let createdAt: Int64?
    let seenAt: Int64?

    var isBot: Bool {
        title == "BOT"
    }
}

/// One token's entry in the `POST /api/token/test` response.
struct LichessBotTokenInfo: Sendable, Hashable, Codable {
    let userId: String
    /// Comma-separated scopes; empty if the token has none.
    let scopes: String
    /// Expiry as Unix milliseconds, or nil if the token never expires.
    let expires: Int64?

    var scopeList: [String] {
        scopes.split(separator: ",", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }
}

/// Lichess's `{"error": "..."}` body on 4xx responses.
struct LichessBotErrorBody: Sendable, Hashable, Codable {
    let error: String
}
