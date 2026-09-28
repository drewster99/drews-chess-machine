import Foundation

/// The Lichess calls a game session makes. `LichessBotAPIClient` is the real
/// implementation; tests script a fake.
protocol LichessBotGameAPI: Sendable {
    func openGameStream(gameID: String) async throws -> LichessBotChunkStream
    func makeMove(gameID: String, uci: String, offeringDraw: Bool) async throws
    func respondToDraw(gameID: String, accept: Bool) async throws
    func respondToTakeback(gameID: String, accept: Bool) async throws
    func resign(gameID: String) async throws
    func abort(gameID: String) async throws
    func claimVictory(gameID: String) async throws
    func claimDraw(gameID: String) async throws
    func chat(gameID: String, room: LichessBotChatRoom, text: String) async throws
}

extension LichessBotAPIClient: LichessBotGameAPI {}

/// Identity of the weights a move was played with (plan §9). Recorded with
/// every move, so every statistic is attributable to exact weights.
struct LichessBotGenerationInfo: Sendable, Equatable, Codable {
    /// Distinct per snapshot within one run of the app.
    let generationID: Int
    let sourceKind: LichessBotModelSourceKind
    let modelID: String
    /// The trainer's completed step count when snapshotted; nil for sources
    /// without one (a champion or a file).
    let trainingStep: Int?
    let snapshotAt: Date
    let architectureSummary: String
    let filePath: String?
    let fileSHA256: String?
}

/// Something that chooses moves: a model generation in production, a
/// scripted fake in tests.
protocol LichessBotMoveSource: Sendable {
    var info: LichessBotGenerationInfo { get }
    func decide(_ request: LichessBotMoveRequest, schedule: SamplingSchedule) async throws -> LichessBotMoveDecision
}

/// Whether a game is waiting on our move, and how much time we have. The
/// session manager aggregates these into the request gate's eligibility
/// inputs (plan §5.2, E18).
struct LichessBotTurnStatus: Sendable, Equatable {
    var awaitingOurMove: Bool
    var ourClock: LichessBotMilliseconds?
}

/// Everything a game session reports: the raw record for the journal
/// (plan §10.2), and the live view for the UI.
enum LichessBotGameEvent: Sendable {
    case streamOpened(attempt: Int)
    /// A raw NDJSON line exactly as received, and when.
    case streamLine(Data, receivedAt: Date)
    /// A keep-alive (empty line) on the game stream, and when.
    case keepAlive(receivedAt: Date)
    case streamEnded(reason: String)
    case gameInfo(LichessBotGameFull, ourColor: LichessBotColorName)
    case positionSynced(LichessBotPositionSync, ply: Int)
    case moveDecided(ply: Int, decision: LichessBotMoveDecision, generation: LichessBotGenerationInfo)
    case movePosted(ply: Int, uci: String, offeringDraw: Bool, milliseconds: Double)
    case moveRejected(ply: Int, uci: String, error: String)
    case action(String)
    /// DCM decided `uci` at `ply` and is holding it for the operator's
    /// Play move, or waiting out a per-game delay (plan §14.3c).
    case moveHeld(ply: Int, uci: String, san: String)
    /// A held or delayed move was released for posting, and why.
    case moveReleased(ply: Int, reason: String)
    case chat(LichessBotChatLine)
    /// DCM posted a chat message (Lichess answered 200). Lichess echoes it
    /// on the game stream only while the stream is open, so a message sent
    /// after the game ends (the goodbye) is known only from this event.
    case chatSent(room: LichessBotChatRoom, text: String, origin: LichessBotChatOrigin)
    /// A player-room message found by fetching the chat after the game (the
    /// stream had closed); only lines not already known are reported.
    case chatFetched(username: String, text: String)
    case anomaly(String)
    /// The session stopped moving in this game: its moves keep being
    /// rejected, or a move on our side appeared that this client did not
    /// send — another client may be playing the account (plan §6.1 B).
    case stoppedMoving(reason: String)
    /// Lichess rejected the token (401/403) on this game's requests.
    case tokenRejected(String)
    /// The game ended. `localDrawCondition` is the draw rule DCM's own
    /// engine sees in the final position, if any, recorded beside the
    /// server's status so the two rule sets can be compared (plan E10).
    case finished(status: LichessBotOpenValue<LichessBotGameStatusName>, winner: LichessBotOpenValue<LichessBotColorName>?, localDrawCondition: ChessDrawCondition?)
}

/// Receives a game session's events. The game journal and the UI model
/// implement it. Called in order from the session's actor.
protocol LichessBotGameObserver: Sendable {
    func gameEvent(gameID: String, _ event: LichessBotGameEvent) async
}

/// Who had DCM's account say something in chat.
enum LichessBotChatOrigin: String, Sendable, Codable, Equatable {
    case greeting
    case goodbye
    case commandReply
    case `operator`
}

/// The operator's per-game move pacing as the session reads it (plan
/// §14.3c). The default is no pacing: DCM moves as soon as it decides.
struct LichessBotMovePacingSnapshot: Sendable, Equatable {
    /// Seconds to wait before posting each move after DCM's first.
    var delaySeconds = 0
    /// Hold each move after DCM's first until the operator releases it.
    var holds = false
    /// The operator clicked Play move for the held move.
    var releaseRequested = false

    var isActive: Bool {
        delaySeconds > 0 || holds
    }

    /// Our clock at or below this forces a held or delayed move out: a
    /// margin plus a multiple of the last move's round trip, so the post lands in
    /// time (plan §14.3c).
    static func clockFloorMilliseconds(lastMoveRoundTripMilliseconds: Double?) -> Int {
        30_000 + Int(2 * (lastMoveRoundTripMilliseconds ?? 0))
    }
}
