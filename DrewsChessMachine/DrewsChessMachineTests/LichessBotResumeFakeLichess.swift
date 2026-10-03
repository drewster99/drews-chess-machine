import XCTest
@testable import DrewsChessMachine

/// A Lichess for resume tests: the same endpoints as `LichessBotFakeLichess`,
/// plus a record of every POST path and a per-game move list the next
/// `gameFull` reports, so a game can be "continued" on a second controller
/// the way Lichess replays a live game after a relaunch.
final class LichessBotResumeFakeLichess: LichessBotTransport, @unchecked Sendable {
    static let token = "lip_RESUMETEST"
    static let botID = "drewschessmachine"

    /// Every POST path, in order.
    let postedPaths = SyncBox<[String]>([])
    /// The move list (space-separated UCI) each game's next `gameFull`
    /// carries, by game id; absent means no moves yet.
    let movesByGame = SyncBox<[String: String]>([:])
    /// Takeback proposals the next `gameFull` carries, by game id: the
    /// opponent (white) proposing one.
    let opponentProposesTakeback = SyncBox<Set<String>>([])
    /// Successive answers to `GET /api/bot/game/<id>/chat`, by game id, each
    /// a JSON array; the last one repeats.
    let chatFetchResponses = SyncBox<[String: [String]]>([:])
    /// Games whose export says DCM (black) won by resignation with no moves;
    /// any other export is a 404.
    let resignedExports = SyncBox<Set<String>>([])
    private let eventContinuation = SyncBox<LichessBotChunkStream.Continuation?>(nil)
    private let gameContinuations = SyncBox<[String: LichessBotChunkStream.Continuation]>([:])

    /// Event streams opened so far: a new launch (or a new go-online) opens
    /// a new one, while an earlier launch's may still look open.
    let eventStreamsOpened = SyncBox<Int>(0)

    var eventStreamIsOpen: Bool {
        eventContinuation.value != nil
    }

    func gameStreamIsOpen(_ gameID: String) -> Bool {
        gameContinuations.value[gameID] != nil
    }

    func sendEvent(_ line: String) {
        eventContinuation.value?.yield(Data((line + "\n").utf8))
    }

    func sendGameLine(_ gameID: String, _ line: String) {
        gameContinuations.value[gameID]?.yield(Data((line + "\n").utf8))
    }

    /// The game against `opponent` (id `c<opponent>`) starts, or is replayed
    /// on a new event stream.
    func startGame(against opponent: String) {
        sendEvent(#"{"type":"gameStart","game":{"gameId":"c\#(opponent)","opponent":{"id":"\#(opponent)"}}}"#)
    }

    func postCount(containing fragment: String) -> Int {
        postedPaths.value.filter { $0.contains(fragment) }.count
    }

    static func stateJSON(moves: String, whiteProposesTakeback: Bool = false) -> String {
        let takeback = whiteProposesTakeback ? #","wtakeback":true"# : ""
        return #"{"type":"gameState","moves":"\#(moves)","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"\#(takeback)}"#
    }

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        let (status, body) = answer(request)
        return LichessBotTransportResponse(body: Data(body.utf8), response: try lichessBotTestResponse(status: status), networkProtocolName: "h2")
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        let path = request.url?.path ?? ""
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        if path == "/api/stream/event" {
            eventContinuation.value = continuation
            eventStreamsOpened.modify { $0 += 1 }
            return (chunks, try lichessBotTestResponse(status: 200))
        }
        let gamePrefix = "/api/bot/game/stream/"
        if path.hasPrefix(gamePrefix) {
            let gameID = String(path.dropFirst(gamePrefix.count))
            gameContinuations.modify { $0[gameID] = continuation }
            continuation.yield(Data((gameFullJSON(gameID: gameID) + "\n").utf8))
            return (chunks, try lichessBotTestResponse(status: 200))
        }
        continuation.yield(Data(#"{"error":"Not found"}"#.utf8))
        continuation.finish()
        return (chunks, try lichessBotTestResponse(status: 404))
    }

    /// DCM plays black against `c<opponent>`'s opponent.
    private func gameFullJSON(gameID: String) -> String {
        let opponent = gameID.hasPrefix("c") ? String(gameID.dropFirst()) : gameID
        let white = #"{"id":"\#(opponent)","name":"\#(opponent)","title":"BOT","rating":1500}"#
        let black = #"{"id":"\#(Self.botID)","name":"DrewsChessMachine","title":"BOT","rating":1500}"#
        let state = Self.stateJSON(moves: movesByGame.value[gameID] ?? "", whiteProposesTakeback: opponentProposesTakeback.value.contains(gameID))
        return #"{"type":"gameFull","id":"\#(gameID)","variant":{"key":"standard","name":"Standard","short":"Std"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1700000000000,"white":\#(white),"black":\#(black),"initialFen":"startpos","state":\#(state)}"#
    }

    private func answer(_ request: URLRequest) -> (Int, String) {
        let method = request.httpMethod ?? "GET"
        let path = request.url.flatMap { URLComponents(url: $0, resolvingAgainstBaseURL: true) }?.path ?? ""
        if method == "POST" {
            postedPaths.modify { $0.append(path) }
        }
        switch (method, path) {
        case ("POST", "/api/token/test"):
            return (200, #"{"\#(Self.token)":{"userId":"\#(Self.botID)","scopes":"bot:play,challenge:write","expires":null}}"#)
        case ("GET", "/api/account"):
            return (200, #"{"id":"\#(Self.botID)","username":"DrewsChessMachine","title":"BOT","perfs":{}}"#)
        default:
            break
        }
        if method == "POST" {
            return (200, #"{"ok":true}"#)
        }
        let segments = path.split(separator: "/").map(String.init)
        if segments.count == 5, segments[0] == "api", segments[1] == "bot", segments[2] == "game", segments[4] == "chat" {
            let gameID = segments[3]
            let answer = chatFetchResponses.mutate { responses -> String in
                guard var queue = responses[gameID], let first = queue.first else { return "[]" }
                if queue.count > 1 {
                    queue.removeFirst()
                    responses[gameID] = queue
                }
                return first
            }
            return (200, answer)
        }
        if segments.count == 3, segments[0] == "game", segments[1] == "export", resignedExports.value.contains(segments[2]) {
            let gameID = segments[2]
            let opponent = gameID.hasPrefix("c") ? String(gameID.dropFirst()) : gameID
            return (200, #"{"id":"\#(gameID)","rated":false,"variant":"standard","speed":"blitz","perf":"blitz","createdAt":1700000000000,"lastMoveAt":1700000060000,"status":"resign","winner":"black","players":{"white":{"user":{"name":"\#(opponent)","title":"BOT","id":"\#(opponent)"},"rating":1500},"black":{"user":{"name":"DrewsChessMachine","title":"BOT","id":"\#(Self.botID)"},"rating":1500}},"moves":"","clock":{"initial":180,"increment":2}}"#)
        }
        return (404, #"{"error":"Not found"}"#)
    }

    /// White (the opponent) resigns `c<opponent>` before any move.
    func opponentResigns(_ opponent: String) {
        sendGameLine("c\(opponent)", #"{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"resign","winner":"black"}"#)
    }
}
