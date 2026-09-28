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

    init(baseURL: URL, token: String, transport: any LichessBotTransport, gate: LichessBotRequestGate) {
        self.baseURL = baseURL
        self.token = token
        self.transport = transport
        self.gate = gate
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
        let body = try await send(request, priority: .housekeeping, label: "token test")
        let entries: [String: LichessBotTokenInfo?]
        do {
            entries = try JSONDecoder().decode([String: LichessBotTokenInfo?].self, from: body)
        } catch {
            throw LichessBotAPIError.undecodableResponse(endpoint: "/api/token/test", detail: String(describing: error))
        }
        guard let entry = entries[token] else {
            throw LichessBotAPIError.undecodableResponse(endpoint: "/api/token/test", detail: "response has no entry for the submitted token")
        }
        return entry
    }

    /// `GET /api/account`.
    func account() async throws -> LichessBotAccount {
        let request = try makeRequest(path: "/api/account", method: "GET")
        return try decode(LichessBotAccount.self, from: try await send(request, priority: .housekeeping, label: "account"), endpoint: "/api/account")
    }

    /// `POST /api/bot/account/upgrade`. **Irreversible.** Only the guarded
    /// Settings flow calls this (plan §12.2).
    func upgradeToBot() async throws {
        let request = try makeRequest(path: "/api/bot/account/upgrade", method: "POST")
        _ = try await send(request, priority: .housekeeping, label: "upgrade to BOT")
    }

    // MARK: - Challenges

    func acceptChallenge(id: String) async throws {
        let request = try makeRequest(path: "/api/challenge/\(try Self.pathComponent(id))/accept", method: "POST")
        _ = try await send(request, priority: .challengeResponse, label: "accept challenge")
    }

    func declineChallenge(id: String, reason: LichessBotDeclineReason) async throws {
        var request = try makeRequest(path: "/api/challenge/\(try Self.pathComponent(id))/decline", method: "POST")
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("reason=\(reason.rawValue)".utf8)
        _ = try await send(request, priority: .challengeResponse, label: "decline challenge")
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
        _ = try await send(request, priority: .move, label: "move")
    }

    /// `POST /api/bot/game/{id}/draw/{yes|no}`.
    func respondToDraw(gameID: String, accept: Bool) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/draw/\(accept ? "yes" : "no")", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "draw \(accept ? "yes" : "no")")
    }

    /// `POST /api/bot/game/{id}/takeback/{yes|no}`.
    func respondToTakeback(gameID: String, accept: Bool) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/takeback/\(accept ? "yes" : "no")", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "takeback \(accept ? "yes" : "no")")
    }

    func resign(gameID: String) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/resign", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "resign")
    }

    func abort(gameID: String) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/abort", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "abort")
    }

    /// Claim victory after the opponent has left (plan §12.4).
    func claimVictory(gameID: String) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/claim-victory", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "claim victory")
    }

    /// Claim a draw after the opponent has left. Not a threefold or
    /// fifty-move claim — Lichess ends those automatically for bots (plan §4).
    func claimDraw(gameID: String) async throws {
        let request = try makeRequest(path: "/api/bot/game/\(try Self.pathComponent(gameID))/claim-draw", method: "POST")
        _ = try await send(request, priority: .gameCritical, label: "claim draw")
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
        _ = try await send(request, priority: .chat, label: "chat")
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
        return try await send(request, priority: .housekeeping, label: "export game")
    }

    // MARK: - Streams

    /// Open the account's event stream. Opening goes through the gate;
    /// holding it does not.
    func openEventStream() async throws -> LichessBotChunkStream {
        let request = try makeRequest(path: "/api/stream/event", method: "GET")
        return try await openStream(request, label: "open event stream")
    }

    /// Open one game's stream. Its first line is always `gameFull`.
    func openGameStream(gameID: String) async throws -> LichessBotChunkStream {
        let request = try makeRequest(path: "/api/bot/game/stream/\(try Self.pathComponent(gameID))", method: "GET")
        return try await openStream(request, label: "open game stream")
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
    private func send(_ request: URLRequest, priority: LichessBotRequestPriority, label: String) async throws -> Data {
        let transport = self.transport
        let (response, http) = try await gate.perform(priority: priority, label: label) {
            let result = try await transport.data(for: request)
            return (value: result, response: result.response)
        }
        return try Self.checkStatus(http.statusCode, body: response.body)
    }

    private func openStream(_ request: URLRequest, label: String) async throws -> LichessBotChunkStream {
        let transport = self.transport
        let (chunks, http) = try await gate.perform(priority: .streamOpen, label: label) {
            let opened = try await transport.stream(for: request)
            return (value: opened.chunks, response: opened.response)
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
            throw Self.statusError(http.statusCode, body: body)
        }
        return chunks
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
