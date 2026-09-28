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
    case chat(LichessBotChatLine)
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
