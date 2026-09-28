import XCTest
@testable import DrewsChessMachine

/// A transport that records every request and answers from a script.
final class LichessBotScriptedTransport: LichessBotTransport, @unchecked Sendable {
    typealias Handler = @Sendable (URLRequest) throws -> (status: Int, body: Data, headers: [String: String])

    let requests = SyncBox<[URLRequest]>([])
    private let handler: Handler

    init(handler: @escaping Handler) {
        self.handler = handler
    }

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        requests.modify { $0.append(request) }
        let answer = try handler(request)
        return LichessBotTransportResponse(
            body: answer.body,
            response: try lichessBotTestResponse(status: answer.status, headers: answer.headers),
            networkProtocolName: "h2"
        )
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        requests.modify { $0.append(request) }
        let answer = try handler(request)
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        continuation.yield(answer.body)
        continuation.finish()
        return (chunks, try lichessBotTestResponse(status: answer.status, headers: answer.headers))
    }
}

/// `LichessBotAPIClient` — request construction, status mapping, and that
/// every call goes through the gate (Lichess bot plan §5, E27).
final class LichessBotAPIClientTests: XCTestCase {

    private let token = "lip_TESTTOKEN123"

    private func makeClient(_ handler: @escaping LichessBotScriptedTransport.Handler) throws -> (LichessBotAPIClient, LichessBotScriptedTransport) {
        let transport = LichessBotScriptedTransport(handler: handler)
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        let client = LichessBotAPIClient(baseURL: try LichessBotAPIClient.lichessBaseURL(), token: token, transport: transport, gate: gate)
        return (client, transport)
    }

    private func onlyRequest(_ transport: LichessBotScriptedTransport) throws -> URLRequest {
        let requests = transport.requests.value
        XCTAssertEqual(requests.count, 1)
        return try XCTUnwrap(requests.first)
    }

    private static let okBody = Data(#"{"ok":true}"#.utf8)

    func testMoveRequestShape() async throws {
        let (client, transport) = try makeClient { _ in (200, Self.okBody, [:]) }
        try await client.makeMove(gameID: "pG3WSP96", uci: "e2e4", offeringDraw: false)
        let request = try onlyRequest(transport)
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.url?.absoluteString, "https://lichess.org/api/bot/game/pG3WSP96/move/e2e4")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer \(token)")
    }

    func testMoveWithDrawOffer() async throws {
        let (client, transport) = try makeClient { _ in (200, Self.okBody, [:]) }
        try await client.makeMove(gameID: "g1", uci: "e7e8q", offeringDraw: true)
        XCTAssertEqual(try onlyRequest(transport).url?.absoluteString, "https://lichess.org/api/bot/game/g1/move/e7e8q?offeringDraw=true")
    }

    func testTheTokenNeverAppearsInAURL() async throws {
        let (client, transport) = try makeClient { request in
            if request.url?.path(percentEncoded: true) == "/api/token/test" {
                return (200, Data(#"{"lip_TESTTOKEN123":{"userId":"drewschessmachine","scopes":"bot:play","expires":null}}"#.utf8), [:])
            }
            return (200, Self.okBody, [:])
        }
        _ = try await client.testToken()
        try await client.makeMove(gameID: "g1", uci: "e2e4", offeringDraw: false)
        try await client.declineChallenge(id: "c1", reason: .tooFast)
        XCTAssertEqual(transport.requests.value.count, 3)
        for request in transport.requests.value {
            XCTAssertFalse(request.url?.absoluteString.contains(token) ?? true, "token leaked into \(String(describing: request.url))")
        }
    }

    func testTokenTestSendsTheTokenInTheBodyWithoutAuthorization() async throws {
        let (client, transport) = try makeClient { _ in
            (200, Data(#"{"lip_TESTTOKEN123":{"userId":"drewschessmachine","scopes":"bot:play","expires":1790000000000}}"#.utf8), [:])
        }
        let maybeInfo = try await client.testToken()
        let info = try XCTUnwrap(maybeInfo)
        XCTAssertEqual(info.userId, "drewschessmachine")
        XCTAssertEqual(info.scopeList, ["bot:play"])
        XCTAssertEqual(info.expires, 1_790_000_000_000)
        let request = try onlyRequest(transport)
        XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
        XCTAssertEqual(request.httpBody, Data(token.utf8))
    }

    func testInvalidTokenIsNil() async throws {
        let (client, _) = try makeClient { _ in (200, Data(#"{"lip_TESTTOKEN123":null}"#.utf8), [:]) }
        let info = try await client.testToken()
        XCTAssertNil(info)
    }

    func testDeclineCarriesItsReason() async throws {
        let (client, transport) = try makeClient { _ in (200, Self.okBody, [:]) }
        try await client.declineChallenge(id: "iOskobMC", reason: .tooFast)
        let request = try onlyRequest(transport)
        XCTAssertEqual(request.url?.absoluteString, "https://lichess.org/api/challenge/iOskobMC/decline")
        XCTAssertEqual(request.httpBody, Data("reason=tooFast".utf8))
    }

    func testChatIsFormEncoded() async throws {
        let (client, transport) = try makeClient { _ in (200, Self.okBody, [:]) }
        try await client.chat(gameID: "g1", room: .player, text: "Good luck & have fun")
        let body = String(decoding: try XCTUnwrap(try onlyRequest(transport).httpBody), as: UTF8.self)
        XCTAssertEqual(body, "room=player&text=Good%20luck%20%26%20have%20fun")
    }

    func testUnauthorizedMapsToUnauthorized() async throws {
        let (client, _) = try makeClient { _ in (401, Data(#"{"error":"No such token"}"#.utf8), [:]) }
        do {
            _ = try await client.account()
            XCTFail("401 must throw")
        } catch let error as LichessBotAPIError {
            XCTAssertEqual(error, .unauthorized(status: 401, message: "No such token"))
        }
    }

    /// E27: a proxy error page is reported as a short excerpt, never decoded
    /// as JSON and never quoted whole.
    func testNonJSONErrorBodyIsExcerpted() async throws {
        let html = "<html>" + String(repeating: "x", count: 5000) + "</html>"
        let (client, _) = try makeClient { _ in (502, Data(html.utf8), [:]) }
        do {
            _ = try await client.account()
            XCTFail("502 must throw")
        } catch LichessBotAPIError.http(let status, let message) {
            XCTAssertEqual(status, 502)
            XCTAssertLessThanOrEqual(message?.count ?? 0, 200)
            XCTAssertTrue(message?.hasPrefix("<html>") ?? false)
        }
    }

    func testRateLimitSurfacesAsAGateError() async throws {
        let (client, _) = try makeClient { _ in (429, Data(), [:]) }
        do {
            try await client.makeMove(gameID: "g1", uci: "e2e4", offeringDraw: false)
            XCTFail("429 must throw")
        } catch let error as LichessBotGateError {
            XCTAssertEqual(error, .rateLimited(cooldown: .seconds(60)))
        }
    }

    func testMalformedIdentifiersCannotAlterThePath() async throws {
        let (client, transport) = try makeClient { _ in (200, Self.okBody, [:]) }
        try await client.resign(gameID: "../account")
        XCTAssertEqual(try onlyRequest(transport).url?.path(percentEncoded: true), "/api/bot/game/%2E%2E%2Faccount/resign")
    }

    func testNonASCIIIdentifiersAreEncoded() async throws {
        let (client, transport) = try makeClient { _ in (200, Self.okBody, [:]) }
        try await client.resign(gameID: "é1")
        XCTAssertEqual(try onlyRequest(transport).url?.path(percentEncoded: true), "/api/bot/game/%C3%A91/resign")
    }

    func testRefusedStreamReportsLichessMessage() async throws {
        let (client, _) = try makeClient { _ in (404, Data(#"{"error":"No such game"}"#.utf8), [:]) }
        do {
            _ = try await client.openGameStream(gameID: "missing1")
            XCTFail("404 must throw")
        } catch let error as LichessBotAPIError {
            XCTAssertEqual(error, .http(status: 404, message: "No such game"))
        }
    }
}
