import XCTest
@testable import DrewsChessMachine

/// `LichessBotResumeFakeLichess`, forwarded, with two things it can't do on
/// its own:
///
/// - **A game Lichess no longer has.** `endGameStream(_:)` closes the game's
///   open stream the way Lichess closes it when a game is deleted, and from
///   then on every reopen of that game's stream is a 404 — so the session
///   ends without ever seeing a finish.
/// - **A held account request.** With `holdsAccount`, `GET /api/account`
///   waits for `accountRelease`, so a test can keep a go-online in its
///   `.connecting` phase while it checks something else.
///
/// Everything else goes to `base` unchanged.
final class LichessBotForwardingFakeLichess: LichessBotTransport, @unchecked Sendable {
    let base: LichessBotResumeFakeLichess
    let holdsAccount: Bool
    let accountRelease = LichessBotTestLatch()
    /// Account requests that have reached this transport.
    let accountRequestsReceived = SyncBox<Int>(0)
    /// Games whose stream reopens are answered 404.
    private let goneGames = SyncBox<Set<String>>([])
    /// The forwarded stream of each game, by id, so the test can close it.
    private let gameStreamContinuations = SyncBox<[String: LichessBotChunkStream.Continuation]>([:])
    /// 404 answers given to game-stream reopens, by game id.
    let goneGameStreamRefusals = SyncBox<[String: Int]>([:])

    init(base: LichessBotResumeFakeLichess = LichessBotResumeFakeLichess(), holdsAccount: Bool = false) {
        self.base = base
        self.holdsAccount = holdsAccount
    }

    /// Lichess no longer has `gameID` as a playable game: its stream closes
    /// now and every reopen is refused with a 404.
    func endGameStream(_ gameID: String) {
        goneGames.modify { $0.insert(gameID) }
        let continuation = gameStreamContinuations.mutate { $0.removeValue(forKey: gameID) }
        continuation?.finish()
    }

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        if request.httpMethod == "GET", request.url?.path == "/api/account" {
            accountRequestsReceived.modify { $0 += 1 }
            if holdsAccount {
                await accountRelease.wait()
                try Task.checkCancellation()
            }
        }
        return try await base.data(for: request)
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        let path = request.url?.path ?? ""
        let gamePrefix = "/api/bot/game/stream/"
        guard path.hasPrefix(gamePrefix) else {
            return try await base.stream(for: request)
        }
        let gameID = String(path.dropFirst(gamePrefix.count))
        if goneGames.value.contains(gameID) {
            goneGameStreamRefusals.modify { $0[gameID, default: 0] += 1 }
            let (chunks, continuation) = LichessBotChunkStream.makeStream()
            continuation.yield(Data(#"{"error":"Not found"}"#.utf8))
            continuation.finish()
            return (chunks, try lichessBotTestResponse(status: 404))
        }
        let opened = try await base.stream(for: request)
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        let forwarding = Task {
            do {
                for try await chunk in opened.chunks {
                    continuation.yield(chunk)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in
            forwarding.cancel()
        }
        gameStreamContinuations.modify { $0[gameID] = continuation }
        // `base` reports the stream open before it is registered here: a
        // game ended in between must not keep this stream.
        if goneGames.value.contains(gameID) {
            endGameStream(gameID)
        }
        return (chunks, opened.response)
    }
}
