import Foundation
import os

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
    /// A stream's response headers did not arrive in time. Opening a stream
    /// holds the single-flight request gate, so the wait is bounded: a
    /// black-holed connection must never keep move POSTs waiting.
    case streamHeadersTimedOut(Duration)

    var errorDescription: String? {
        switch self {
        case .notHTTPResponse:
            return "Lichess returned a non-HTTP response"
        case .streamHeadersTimedOut(let limit):
            return "Lichess did not answer the stream request within \(limit)"
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
        // Short on purpose: every request holds the single-flight gate, so a
        // stalled one delays every other (plan §5.2). Each request also sets
        // its own, possibly shorter, idle limit for its priority; the
        // session's is the longest of those.
        requestConfiguration.timeoutIntervalForRequest = LichessBotRequestTimeouts.longestIdle
        requestConfiguration.timeoutIntervalForResource = LichessBotRequestTimeouts.resource
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

    /// How long a stream request may wait for its response headers.
    static let streamHeaderDeadline: Duration = .seconds(15)

    /// The most bytes of one unfinished line the transport buffers before
    /// handing them on. A long line is delivered in pieces so the transport
    /// never buffers more of a line than the splitter would keep; the
    /// splitter, not the transport, enforces the line cap.
    static let partialLineDeliveryBytes = 1 << 16

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        let (bytes, response) = try await openBytes(for: request)
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
                        // complete, and in pieces once a long line's buffered
                        // bytes reach the delivery size, so this buffer stays
                        // bounded however long a line gets.
                        if byte == UInt8(ascii: "\n") || chunk.count >= Self.partialLineDeliveryBytes {
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

extension LichessBotURLSessionTransport {
    /// `URLSession.bytes(for:)` raced against `streamHeaderDeadline`. The
    /// losing side is cancelled; a connection that opened just as the
    /// deadline fired is cancelled too, so it can never leak.
    private func openBytes(for request: URLRequest) async throws -> (URLSession.AsyncBytes, URLResponse) {
        let opened = OpenedStreamBox()
        let session = streamSession
        let deadline = Self.streamHeaderDeadline
        let openedInTime = try await withThrowingTaskGroup(of: Bool.self) { group in
            group.addTask {
                let (bytes, response) = try await session.bytes(for: request)
                opened.store(bytes: bytes, response: response)
                return true
            }
            group.addTask {
                try await Task.sleep(for: deadline)
                return false
            }
            defer { group.cancelAll() }
            guard let first = try await group.next() else {
                throw CancellationError()
            }
            return first
        }
        guard openedInTime, let result = opened.take() else {
            opened.take()?.bytes.task.cancel()
            throw LichessBotTransportError.streamHeadersTimedOut(deadline)
        }
        return result
    }
}

/// Carries an opened byte stream out of the task group that raced it
/// against the header deadline. `URLSession.AsyncBytes` is not `Sendable`;
/// the lock makes the single handoff safe.
private final class OpenedStreamBox: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock<(bytes: URLSession.AsyncBytes, response: URLResponse)?>(uncheckedState: nil)

    func store(bytes: URLSession.AsyncBytes, response: URLResponse) {
        lock.withLockUnchecked { $0 = (bytes, response) }
    }

    func take() -> (bytes: URLSession.AsyncBytes, response: URLResponse)? {
        lock.withLockUnchecked { value in
            defer { value = nil }
            return value
        }
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

/// Idle timeouts for non-stream requests. Every such request holds the
/// single-flight gate for as long as it runs, so these bound how long one
/// stalled request can delay every other, moves included (plan §5.2).
enum LichessBotRequestTimeouts {
    /// For requests that must get through: moves, game actions, challenge
    /// traffic. Long enough to ride out a slow response; retrying a move whose
    /// POST timed out is ambiguous (it may have been applied), so these are
    /// not cut short.
    static let urgentIdle: TimeInterval = 10
    /// For requests that can wait or be redone later: chat, account checks,
    /// lists, exports. Short, so one that stalls just as a move comes due
    /// gives the gate back quickly.
    static let deferrableIdle: TimeInterval = 5
    /// Whole-request limit for the request session.
    static let resource: TimeInterval = 30

    /// The request session's own idle limit: the longest any priority uses,
    /// so a request's shorter value is the one that binds whichever of the
    /// two URLSession honors.
    static var longestIdle: TimeInterval {
        max(urgentIdle, deferrableIdle)
    }

    static func idle(for priority: LichessBotRequestPriority) -> TimeInterval {
        switch priority {
        case .move, .gameCritical, .challengeResponse:
            return urgentIdle
        case .chat, .housekeeping:
            return deferrableIdle
        case .streamOpen:
            preconditionFailure("stream opens use the stream session, whose idle timeout must stay long")
        }
    }
}
