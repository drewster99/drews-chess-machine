import Foundation

/// A byte stream from a long-lived HTTP response, delivered in chunks.
typealias LichessBotChunkStream = AsyncThrowingStream<Data, Error>

/// The HTTP layer under `LichessBotAPIClient`, abstracted so the client, the
/// gate and the stream handling can be tested against a scripted fake.
protocol LichessBotTransport: Sendable {
    /// A complete request/response. Returns the response for any status;
    /// status handling is the caller's.
    func data(for request: URLRequest) async throws -> LichessBotTransportResponse
    /// Open a long-lived streaming response. Returns once the response
    /// headers arrive; the body keeps arriving on `chunks` until the server
    /// ends it, the connection drops, or the consumer stops iterating (which
    /// cancels the underlying connection).
    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse)
}

struct LichessBotTransportResponse: Sendable {
    let body: Data
    let response: HTTPURLResponse
    /// The negotiated protocol (`"h2"`, `"http/1.1"`, …) when URLSession
    /// reported it. Diagnostic only (plan E32).
    let networkProtocolName: String?
}

enum LichessBotTransportError: LocalizedError {
    case notHTTPResponse

    var errorDescription: String? {
        switch self {
        case .notHTTPResponse:
            return "Lichess returned a non-HTTP response"
        }
    }
}

/// The real transport: two separate `URLSession`s (plan §6, E32, E33).
///
/// - **Requests** use one session with ordinary timeouts.
/// - **Streams** use another, with a very long idle timeout. A game stream
///   can legitimately go quiet while the opponent thinks, and URLSession's
///   request timeout is an idle timeout that would otherwise kill it. Stream
///   liveness is owned by the bot's own turn-aware watchdog instead.
///
/// Separate sessions mean separate connection pools. Under HTTP/1.1 every
/// open stream holds a connection for as long as it stays open, and
/// URLSession caps connections per host. If streams and requests shared a
/// pool, a handful of concurrent games could hold every connection and leave
/// a move POST queued behind them until a clock ran out. HTTP/2 multiplexes
/// and avoids this, but the split makes it impossible either way.
///
/// Both sessions are ephemeral: no cookies, cache or credentials persist to
/// disk. The bearer token travels only in each request's header.
final class LichessBotURLSessionTransport: LichessBotTransport {
    private let requestSession: URLSession
    private let streamSession: URLSession

    init() {
        let requestConfiguration = URLSessionConfiguration.ephemeral
        requestConfiguration.timeoutIntervalForRequest = 30
        requestConfiguration.timeoutIntervalForResource = 120
        requestConfiguration.requestCachePolicy = .reloadIgnoringLocalCacheData
        requestConfiguration.waitsForConnectivity = false
        requestSession = URLSession(configuration: requestConfiguration)

        let streamConfiguration = URLSessionConfiguration.ephemeral
        streamConfiguration.timeoutIntervalForRequest = 24 * 60 * 60
        streamConfiguration.timeoutIntervalForResource = 30 * 24 * 60 * 60
        streamConfiguration.requestCachePolicy = .reloadIgnoringLocalCacheData
        streamConfiguration.httpMaximumConnectionsPerHost = 64
        streamConfiguration.waitsForConnectivity = false
        streamSession = URLSession(configuration: streamConfiguration)
    }

    deinit {
        requestSession.invalidateAndCancel()
        streamSession.invalidateAndCancel()
    }

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        let metrics = LichessBotProtocolNameCollector()
        let (body, response) = try await requestSession.data(for: request, delegate: metrics)
        guard let http = response as? HTTPURLResponse else {
            throw LichessBotTransportError.notHTTPResponse
        }
        return LichessBotTransportResponse(body: body, response: http, networkProtocolName: metrics.protocolName)
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        let (bytes, response) = try await streamSession.bytes(for: request)
        let dataTask = bytes.task
        guard let http = response as? HTTPURLResponse else {
            dataTask.cancel()
            throw LichessBotTransportError.notHTTPResponse
        }
        let chunks = LichessBotChunkStream { continuation in
            let pump = Task {
                do {
                    var chunk = Data()
                    for try await byte in bytes {
                        chunk.append(byte)
                        // Deliver at each line end so a line (and a
                        // keep-alive) reaches the consumer the moment it is
                        // complete, not when some buffer fills.
                        if byte == UInt8(ascii: "\n") {
                            continuation.yield(chunk)
                            chunk.removeAll(keepingCapacity: true)
                        }
                    }
                    if !chunk.isEmpty {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                pump.cancel()
                dataTask.cancel()
            }
        }
        return (chunks, http)
    }
}

/// Collects the negotiated protocol name from URLSession task metrics.
private final class LichessBotProtocolNameCollector: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    // Written once from URLSession's delegate queue, read after the request
    // completes; the lock makes that handoff safe.
    private let storage = SyncBox<String?>(nil)

    var protocolName: String? {
        storage.value
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didFinishCollecting metrics: URLSessionTaskMetrics) {
        storage.value = metrics.transactionMetrics.last?.networkProtocolName
    }
}
