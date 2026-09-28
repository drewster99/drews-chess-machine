import Foundation

enum LichessBotAPIError: LocalizedError, Equatable {
    /// 401 or 403: the token is invalid, revoked, expired, or lacks a scope.
    /// The bot goes offline on this (plan §5.5).
    case unauthorized(status: Int, message: String?)
    /// Any other non-2xx status. `message` is Lichess's `error` field when
    /// the body is its JSON error shape, otherwise a truncated excerpt of the
    /// body (a proxy may return HTML; plan E27).
    case http(status: Int, message: String?)
    /// A 2xx response whose body could not be decoded as expected.
    case undecodableResponse(endpoint: String, detail: String)
    case invalidURL(String)

    var errorDescription: String? {
        switch self {
        case .unauthorized(let status, let message):
            return "Lichess rejected the token (HTTP \(status))\(message.map { ": \($0)" } ?? "")"
        case .http(let status, let message):
            return "Lichess returned HTTP \(status)\(message.map { ": \($0)" } ?? "")"
        case .undecodableResponse(let endpoint, let detail):
            return "Unexpected response from \(endpoint): \(detail)"
        case .invalidURL(let text):
            return "Could not build a Lichess URL from \(text)"
        }
    }
}

/// One Lichess request as DCM made it, for the protocol transcript and log
/// (plan §14.3a). Never contains the token: the `Authorization` header is not
/// recorded, and the token-test body (which *is* the token) is omitted.
struct LichessBotRequestRecord: Sendable, Codable, Equatable {
    let startedAt: Date
    /// The game the request belongs to, if any.
    let gameID: String?
    let label: String
    let method: String
    /// Path and query, without the host.
    let path: String
    /// Decoded form fields of a form-encoded body.
    let formFields: [String: String]
    /// Nil when no HTTP response arrived (a gate refusal or transport error).
    let status: Int?
    /// Time spent waiting in the request gate before the request went out.
    let queuedMilliseconds: Double
    /// Time from sending the request to receiving the response headers.
    let roundTripMilliseconds: Double?
    let networkProtocol: String?
    /// Lichess's error message (or a body excerpt) for a non-2xx response.
    let errorMessage: String?
    /// Why no response arrived.
    let failure: String?
}

/// Every Lichess Bot API call DCM makes. Each non-stream call, and the
/// opening of each stream, goes through the account-wide
/// `LichessBotRequestGate` at the call's priority (plan §5).
///
/// Immutable: the token and endpoints are fixed for the client's lifetime.
/// A token change makes a new client.
final class LichessBotAPIClient: Sendable {
    let baseURL: URL
    private let token: String
    private let transport: any LichessBotTransport
    private let gate: LichessBotRequestGate
    /// Receives a record of every request, when something is recording them.
    private let onRequest: (@Sendable (LichessBotRequestRecord) -> Void)?

    init(
        baseURL: URL,
        token: String,
        transport: any LichessBotTransport,
        gate: LichessBotRequestGate,
        onRequest: (@Sendable (LichessBotRequestRecord) -> Void)?
    ) {
        self.baseURL = baseURL
        self.token = token
        self.transport = transport
        self.gate = gate
        self.onRequest = onRequest
    }

    /// A client whose requests nobody records.
    convenience init(baseURL: URL, token: String, transport: any LichessBotTransport, gate: LichessBotRequestGate) {
        self.init(baseURL: baseURL, token: token, transport: transport, gate: gate, onRequest: nil)
    }

    /// `https://lichess.org`.
    static func lichessBaseURL() throws -> URL {
        let text = "https://lichess.org"
        guard let url = URL(string: text) else {
            throw LichessBotAPIError.invalidURL(text)
        }
        return url
    }

    // MARK: - Account and token

    /// `POST /api/token/test`. Returns the token's user, scopes and expiry,
    /// or nil if Lichess reports the token as invalid. The token is sent in
    /// the body (the endpoint takes a list of tokens and needs no auth).
    func testToken() async throws -> LichessBotTokenInfo? {
        var request = try makeRequest(path: "/api/token/test", method: "POST", authorized: false)
        request.setValue("text/plain", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(token.utf8)
        let body = try await send(request, priority: .housekeeping, label: "token test", gameID: nil, recordBody: false)
        let entries: [String: LichessBotTokenInfo?]
        do {
            entries = try JSONDecoder().decode([String: LichessBotTokenInfo?].self, from: body)
        } catch {
            // The response is keyed by the token itself, so the decoding
            // error's coding path would contain it: report the shape only.
            throw LichessBotAPIError.undecodableResponse(endpoint: "/api/token/test", detail: "the response is not the expected token-test shape")
        }
        guard let entry = entries[token] else {
            throw LichessBotAPIError.undecodableResponse(endpoint: "/api/token/test", detail: "response has no entry for the submitted token")
        }
        return entry
    }

    /// `GET /api/account`.
    func account() async throws -> LichessBotAccount {
        let request = try makeRequest(path: "/api/account", method: "GET")
        return try decode(LichessBotAccount.self, from: try await send(request, priority: .housekeeping, label: "account", gameID: nil), endpoint: "/api/account")
    }

    /// `POST /api/bot/account/upgrade`. **Irreversible.** Only the guarded
    /// Settings flow calls this (plan §12.2).
    func upgradeToBot() async throws {
        let request = try makeRequest(path: "/api/bot/account/upgrade", method: "POST")
        _ = try await send(request, priority: .housekeeping, label: "upgrade to BOT", gameID: nil)
    }

    // MARK: - Challenges

    func acceptChallenge(id: String) async throws {
        let request = try makeRequest(path: "/api/challenge/\(try Self.pathComponent(id))/accept", method: "POST")
        _ = try await send(request, priority: .challengeResponse, label: "accept challenge", gameID: nil)
    }

    func declineChallenge(id: String, reason: LichessBotDeclineReason) async throws {
        var request = try makeRequest(path: "/api/challenge/\(try Self.pathComponent(id))/decline", method: "POST")
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("reason=\(reason.rawValue)".utf8)
        _ = try await send(request, priority: .challengeResponse, label: "decline challenge", gameID: nil)
    }

    /// `POST /api/challenge/{username}`: challenge a player (plan §7.1).
    /// Needs the `challenge:write` scope. The game starts through the
    /// ordinary `gameStart` event if they accept.
    func challenge(username: String, request outgoing: LichessBotOutgoingChallenge) async throws -> LichessBotChallenge {
        var request = try makeRequest(path: "/api/challenge/\(try Self.pathComponent(username))", method: "POST")
        var form = URLComponents()
        form.queryItems = [
            URLQueryItem(name: "rated", value: outgoing.rated ? "true" : "false"),
            URLQueryItem(name: "clock.limit", value: String(outgoing.clockLimitSeconds)),
            URLQueryItem(name: "clock.increment", value: String(outgoing.clockIncrementSeconds)),
            URLQueryItem(name: "color", value: outgoing.color.rawValue),
            URLQueryItem(name: "variant", value: "standard"),
        ]
        guard let encoded = form.percentEncodedQuery else {
            throw LichessBotAPIError.invalidURL("challenge form body")
        }
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(encoded.utf8)
        let body = try await send(request, priority: .challengeResponse, label: "challenge \(username)", gameID: nil)
        return try LichessBotOutgoingChallenge.decodeCreated(body)
    }

    /// `POST /api/challenge/{id}/cancel`: withdraw a challenge we sent.
    func cancelChallenge(id: String) async throws {
        let request = try makeRequest(path: "/api/challenge/\(try Self.pathComponent(id))/cancel", method: "POST")
        _ = try await send(request, priority: .challengeResponse, label: "cancel challenge", gameID: nil)
    }

    /// `GET /api/bot/online?nb=`: bots online now, as NDJSON. Needs no
    /// authorization.
    func onlineBots(count: Int) async throws -> [LichessBotUserSummary] {
        var request = try makeRequest(path: "/api/bot/online?nb=\(count)", method: "GET", authorized: false)
        request.setValue("application/x-ndjson", forHTTPHeaderField: "Accept")
        let body = try await send(request, priority: .housekeeping, label: "online bots", gameID: nil)
        var splitter = LichessBotNDJSONSplitter()
        var items = splitter.append(body)
        if splitter.pendingByteCount > 0 {
            items += splitter.append(Data("\n".utf8))
        }
        var users: [LichessBotUserSummary] = []
        for item in items {
            guard case .line(let line) = item else { continue }
            users.append(try decode(LichessBotUserSummary.self, from: line, endpoint: "/api/bot/online"))
        }
        return users
    }

    /// `GET /api/user/{username}`: a player's public profile and ratings.
    func user(username: String) async throws -> LichessBotUserSummary {
        let request = try makeRequest(path: "/api/user/\(try Self.pathComponent(username))", method: "GET")
        return try decode(LichessBotUserSummary.self, from: try await send(request, priority: .housekeeping, label: "user \(username)", gameID: nil), endpoint: "/api/user")
    }

    // MARK: - Game actions

    /// `POST /api/bot/game/{id}/move/{uci}`. `offeringDraw` offers a draw, or
    /// agrees to one the opponent is offering (plan E31).
    func makeMove(gameID: String, uci: String, offeringDraw: Bool) async throws {
        var path = "/api/bot/game/\(try Self.pathComponent(gameID))/move/\(try Self.pathComponent(uci))"
        if offeringDraw {
            path += "?offeringDraw=true"
        }
        let request = try makeRequest(path: path, method: "POST")
        _ = try await send(request, priority: .move, label: "move", gameID: gameID)
    }

    /// `POST /api/bot/game/{id}/draw/{yes|no}`.
    func respondToDraw(gameID: String, accept: Bool) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/draw/\(accept ? "yes" : "no")", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "draw \(accept ? "yes" : "no")", gameID: gameID)
    }

    /// `POST /api/bot/game/{id}/takeback/{yes|no}`.
    func respondToTakeback(gameID: String, accept: Bool) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/takeback/\(accept ? "yes" : "no")", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "takeback \(accept ? "yes" : "no")", gameID: gameID)
    }

    func resign(gameID: String) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/resign", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "resign", gameID: gameID)
    }

    func abort(gameID: String) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/abort", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "abort", gameID: gameID)
    }

    /// Claim victory after the opponent has left (plan §12.4).
    func claimVictory(gameID: String) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/claim-victory", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "claim victory", gameID: gameID)
    }

    /// Claim a draw after the opponent has left. Not a threefold or
    /// fifty-move claim — Lichess ends those automatically for bots (plan §4).
    func claimDraw(gameID: String) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/claim-draw", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "claim draw", gameID: gameID)
    }

    func chat(gameID: String, room: LichessBotChatRoom, text: String) async throws {
        var request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/chat", method: "POST")
        var form = URLComponents()
        form.queryItems = [URLQueryItem(name: "room", value: room.rawValue), URLQueryItem(name: "text", value: text)]
        guard let encoded = form.percentEncodedQuery else {
            throw LichessBotAPIError.invalidURL("chat form body")
        }
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(encoded.utf8)
        _ = try await send(request, priority: .chat, label: "chat", gameID: gameID)
    }

    // MARK: - Records

    /// `GET /game/export/{id}` as JSON, with clocks, opening and the PGN
    /// embedded. The raw body is returned so the record keeps exactly what
    /// Lichess sent.
    func exportGame(gameID: String) async throws -> Data {
        var request = try makeRequest(
            path: "/game/export/\(try Self.pathComponent(gameID))?clocks=true&opening=true&pgnInJson=true",
            method: "GET"
        )
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        // Logged at account level, not into the game's journal: an export is
        // bookkeeping about a finished game, and the journal may already
        // have been filed.
        return try await send(request, priority: .housekeeping, label: "export game \(gameID)", gameID: nil)
    }

    // MARK: - Streams

    /// Open the account's event stream. Opening goes through the gate;
    /// holding it does not.
    func openEventStream() async throws -> LichessBotChunkStream {
        let request = try makeRequest(path: "/api/stream/event", method: "GET")
        return try await openStream(request, label: "open event stream", gameID: nil)
    }

    /// Open one game's stream. Its first line is always `gameFull`.
    func openGameStream(gameID: String) async throws -> LichessBotChunkStream {
        let request = try makeRequest(path: "/api/bot/game/stream/\(try Self.pathComponent(gameID))", method: "GET")
        return try await openStream(request, label: "open game stream", gameID: gameID)
    }

    // MARK: - Plumbing

    private func makeRequest(path: String, method: String, authorized: Bool = true) throws -> URLRequest {
        guard let url = URL(string: path, relativeTo: baseURL)?.absoluteURL else {
            throw LichessBotAPIError.invalidURL(path)
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        if authorized {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// Percent-encode one path segment. Game ids, challenge ids and UCI
    /// moves are ASCII alphanumeric, so this only matters for a malformed
    /// value — which must never be able to alter the path. The allowed set is
    /// ASCII-only on purpose: `CharacterSet.alphanumerics` also admits
    /// non-ASCII letters and digits.
    private static func pathComponent(_ value: String) throws -> String {
        guard !value.isEmpty,
              let encoded = value.addingPercentEncoding(withAllowedCharacters: asciiAlphanumerics) else {
            throw LichessBotAPIError.invalidURL("path segment \"\(value)\"")
        }
        return encoded
    }

    private static let asciiAlphanumerics = CharacterSet(
        charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    /// Send through the gate; return the body of a 2xx response, or throw.
    /// Every attempt is recorded, successful or not.
    private func send(_ request: URLRequest, priority: LichessBotRequestPriority, label: String, gameID: String?, recordBody: Bool = true) async throws -> Data {
        let target = try Self.target(of: request)
        let transport = self.transport
        let timing = RequestTiming()
        let response: LichessBotTransportResponse
        let http: HTTPURLResponse
        do {
            (response, http) = try await gate.perform(priority: priority, label: label) {
                timing.markSent()
                let result = try await transport.data(for: request)
                timing.markResponded()
                return (value: result, response: result.response)
            }
        } catch {
            record(request, target: target, label: label, gameID: gameID, recordBody: recordBody, timing: timing, status: nil, networkProtocol: nil, errorMessage: nil, failure: String(describing: error))
            throw error
        }
        let ok = (200..<300).contains(http.statusCode)
        record(request, target: target, label: label, gameID: gameID, recordBody: recordBody, timing: timing, status: http.statusCode, networkProtocol: response.networkProtocolName, errorMessage: ok ? nil : Self.errorMessage(from: response.body), failure: nil)
        return try Self.checkStatus(http.statusCode, body: response.body)
    }

    private func openStream(_ request: URLRequest, label: String, gameID: String?) async throws -> LichessBotChunkStream {
        let target = try Self.target(of: request)
        let transport = self.transport
        let timing = RequestTiming()
        let chunks: LichessBotChunkStream
        let http: HTTPURLResponse
        do {
            (chunks, http) = try await gate.perform(priority: .streamOpen, label: label) {
                timing.markSent()
                let opened = try await transport.stream(for: request)
                timing.markResponded()
                return (value: opened.chunks, response: opened.response)
            }
        } catch {
            record(request, target: target, label: label, gameID: gameID, recordBody: true, timing: timing, status: nil, networkProtocol: nil, errorMessage: nil, failure: String(describing: error))
            throw error
        }
        if (200..<300).contains(http.statusCode) {
            record(request, target: target, label: label, gameID: gameID, recordBody: true, timing: timing, status: http.statusCode, networkProtocol: nil, errorMessage: nil, failure: nil)
        }
        guard (200..<300).contains(http.statusCode) else {
            // A refused stream carries a short error body; read a bounded
            // amount of it for Lichess's message. Leaving this function drops
            // the stream, and dropping it cancels the connection.
            var body = Data()
            for try await chunk in chunks {
                body.append(chunk)
                if body.count >= Self.refusedStreamBodyLimit {
                    break
                }
            }
            let error = Self.statusError(http.statusCode, body: body)
            record(request, target: target, label: label, gameID: gameID, recordBody: true, timing: timing, status: http.statusCode, networkProtocol: nil, errorMessage: Self.errorMessage(from: body), failure: nil)
            throw error
        }
        return chunks
    }

    /// When a request was queued, sent and answered. Written from the gate's
    /// body closure, read after it returns.
    private final class RequestTiming: Sendable {
        let queuedAt = ContinuousClock.now
        let startedAt = Date()
        private let sentAt = SyncBox<ContinuousClock.Instant?>(nil)
        private let respondedAt = SyncBox<ContinuousClock.Instant?>(nil)

        func markSent() {
            sentAt.value = ContinuousClock.now
        }

        func markResponded() {
            respondedAt.value = ContinuousClock.now
        }

        var queuedMilliseconds: Double {
            LichessBotBackoff.seconds((sentAt.value ?? ContinuousClock.now) - queuedAt) * 1000
        }

        var roundTripMilliseconds: Double? {
            guard let sent = sentAt.value, let responded = respondedAt.value else { return nil }
            return LichessBotBackoff.seconds(responded - sent) * 1000
        }
    }

    /// Method and path-with-query of a request built by `makeRequest`.
    private static func target(of request: URLRequest) throws -> (method: String, path: String) {
        guard let url = request.url, let method = request.httpMethod else {
            throw LichessBotAPIError.invalidURL("a request without a URL or method")
        }
        var path = url.path(percentEncoded: true)
        if let query = url.query(percentEncoded: true) {
            path += "?" + query
        }
        return (method, path)
    }

    private func record(
        _ request: URLRequest,
        target: (method: String, path: String),
        label: String,
        gameID: String?,
        recordBody: Bool,
        timing: RequestTiming,
        status: Int?,
        networkProtocol: String?,
        errorMessage: String?,
        failure: String?
    ) {
        guard let onRequest else { return }
        onRequest(LichessBotRequestRecord(
            startedAt: timing.startedAt,
            gameID: gameID,
            label: label,
            method: target.method,
            path: target.path,
            formFields: recordBody ? Self.formFields(of: request) : [:],
            status: status,
            queuedMilliseconds: timing.queuedMilliseconds,
            roundTripMilliseconds: timing.roundTripMilliseconds,
            networkProtocol: networkProtocol,
            errorMessage: errorMessage,
            failure: failure
        ))
    }

    /// The fields of a form-encoded body; empty for any other body.
    private static func formFields(of request: URLRequest) -> [String: String] {
        guard request.value(forHTTPHeaderField: "Content-Type") == "application/x-www-form-urlencoded",
              let body = request.httpBody,
              let text = String(data: body, encoding: .utf8) else {
            return [:]
        }
        var components = URLComponents()
        components.percentEncodedQuery = text
        var fields: [String: String] = [:]
        for item in components.queryItems ?? [] {
            fields[item.name] = item.value ?? ""
        }
        return fields
    }

    private static let refusedStreamBodyLimit = 4096

    private static func checkStatus(_ status: Int, body: Data) throws -> Data {
        guard (200..<300).contains(status) else {
            throw statusError(status, body: body)
        }
        return body
    }

    private static func statusError(_ status: Int, body: Data) -> LichessBotAPIError {
        switch status {
        case 401, 403:
            return .unauthorized(status: status, message: errorMessage(from: body))
        default:
            return .http(status: status, message: errorMessage(from: body))
        }
    }

    /// Lichess's `{"error": ...}` message, or a short excerpt of a non-JSON
    /// body. Never the whole body: a proxy error page can be large.
    static func errorMessage(from body: Data) -> String? {
        guard !body.isEmpty else { return nil }
        do {
            return try JSONDecoder().decode(LichessBotErrorBody.self, from: body).error
        } catch {
            // Not Lichess's JSON error shape (a proxy's HTML page, plain
            // text): report an excerpt of the raw body instead. The decode
            // error itself carries nothing useful beyond "not that shape".
            let excerpt = String(decoding: body.prefix(errorExcerptLength), as: UTF8.self)
            return excerpt.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private static let errorExcerptLength = 200

    private func decode<T: Decodable>(_ type: T.Type, from body: Data, endpoint: String) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: body)
        } catch {
            throw LichessBotAPIError.undecodableResponse(endpoint: endpoint, detail: String(describing: error))
        }
    }
}
