import XCTest
@testable import DrewsChessMachine

/// Answers the transport's requests from a script, in order: each entry is
/// an HTTP status to answer with, an error the connection fails with, or a
/// hold (no answer until the test releases it).
/// `URLProtocol` is registered by class, so the script is static; the tests
/// in this file set it under the lock and run one at a time.
private final class LichessBotScriptedURLProtocol: URLProtocol {
    enum Step: Sendable {
        case status(Int)
        case fail(URLError.Code)
        case hold
    }

    /// A held request and the run loop of the thread that started it: a
    /// protocol must talk to its client on that thread.
    private struct Held: @unchecked Sendable {
        let request: LichessBotScriptedURLProtocol
        let runLoop: CFRunLoop
    }

    private static let held = SyncBox<[Held]>([])

    static var heldCount: Int {
        held.value.count
    }

    /// Answers every held request with `status`.
    static func releaseHeld(status: Int) {
        let released = held.mutate { held -> [Held] in
            defer { held = [] }
            return held
        }
        for entry in released {
            CFRunLoopPerformBlock(entry.runLoop, CFRunLoopMode.commonModes.rawValue) {
                entry.request.answer(status: status)
            }
            CFRunLoopWakeUp(entry.runLoop)
        }
    }

    private struct Script: Sendable {
        var steps: [Step] = []
        var served = 0
    }

    private static let script = SyncBox(Script())

    static func load(_ steps: [Step]) {
        script.value = Script(steps: steps)
        held.value = []
    }

    static var served: Int {
        script.value.served
    }

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let step = Self.script.mutate { script -> Step? in
            guard script.served < script.steps.count else { return nil }
            defer { script.served += 1 }
            return script.steps[script.served]
        }
        switch step {
        case .status(let status):
            answer(status: status)
        case .hold:
            let entry = Held(request: self, runLoop: CFRunLoopGetCurrent())
            Self.held.modify { $0.append(entry) }
        case .fail(let code):
            client?.urlProtocol(self, didFailWithError: URLError(code))
        case nil:
            client?.urlProtocol(self, didFailWithError: URLError(.resourceUnavailable))
        }
    }

    override func stopLoading() {}

    private func answer(status: Int) {
        guard let url = request.url,
              let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("{}".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
}

/// The real transport replaces its request session after repeated
/// connection failures (`LICHESS_BOT_REQUEST_FAILURE_RECOVERY_PLAN.md`).
final class LichessBotURLSessionTransportResetTests: XCTestCase {

    private func makeTransport(resets: SyncBox<[LichessBotRequestConnectionHealth.Reset]>) -> LichessBotURLSessionTransport {
        LichessBotURLSessionTransport(
            makeConfiguration: {
                let configuration = URLSessionConfiguration.ephemeral
                configuration.protocolClasses = [LichessBotScriptedURLProtocol.self]
                return configuration
            },
            onRequestConnectionsReset: { reset in resets.modify { $0.append(reset) } }
        )
    }

    private static func request() throws -> URLRequest {
        URLRequest(url: try XCTUnwrap(URL(string: "https://lichess.org/api/account")))
    }

    private static func send(_ transport: LichessBotURLSessionTransport) async throws -> Result<Int, URLError> {
        do {
            return .success(try await transport.data(for: try Self.request()).response.statusCode)
        } catch let error as URLError {
            return .failure(error)
        }
    }

    private func failureCode(_ result: Result<Int, URLError>) -> URLError.Code? {
        guard case .failure(let error) = result else { return nil }
        return error.code
    }

    /// Protects: two requests that got no answer replace the request
    /// session, the reset is reported once, and the next request goes out
    /// (on the new session) and is answered.
    /// Fails if: the transport never acts on connection failures
    /// (2026-10-26: three minutes of timeouts, the session never replaced).
    func testTwoTimeoutsReplaceTheRequestSession() async throws {
        LichessBotScriptedURLProtocol.load([.fail(.timedOut), .fail(.timedOut), .status(200)])
        let resets = SyncBox<[LichessBotRequestConnectionHealth.Reset]>([])
        let transport = makeTransport(resets: resets)
        let first = try await Self.send(transport)
        let second = try await Self.send(transport)
        let third = try await Self.send(transport)
        XCTAssertEqual(failureCode(first), .timedOut)
        XCTAssertEqual(failureCode(second), .timedOut)
        XCTAssertEqual(try third.get(), 200)
        XCTAssertEqual(resets.value.map(\.failures), [2])
        XCTAssertEqual(resets.value.map(\.generation), [1])
        XCTAssertEqual(LichessBotScriptedURLProtocol.served, 3)
    }

    /// Protects: a request still out on the session a reset replaces runs
    /// to its answer; the old session is not cancelled under it.
    /// Fails if: the reset invalidates the replaced session at once
    /// (`invalidateAndCancel`, cancelling a move already sent) instead of
    /// after its last request finishes.
    func testARequestOutOnTheReplacedSessionStillGetsItsAnswer() async throws {
        LichessBotScriptedURLProtocol.load([.hold, .fail(.timedOut), .fail(.timedOut)])
        let resets = SyncBox<[LichessBotRequestConnectionHealth.Reset]>([])
        let transport = makeTransport(resets: resets)
        let heldResult = SyncBox<Result<Int, URLError>?>(nil)
        let answered = expectation(description: "the held request is answered")
        let heldRequest = Task {
            do {
                heldResult.value = try await Self.send(transport)
            } catch {
                XCTFail("the held request threw \(error)")
            }
            answered.fulfill()
        }
        var waits = 0
        while LichessBotScriptedURLProtocol.heldCount == 0, waits < 2000 {
            try await Task.sleep(for: .milliseconds(5))
            waits += 1
        }
        XCTAssertEqual(LichessBotScriptedURLProtocol.heldCount, 1, "the first request is held open")
        _ = try await Self.send(transport)
        _ = try await Self.send(transport)
        XCTAssertEqual(resets.value.map(\.generation), [1])
        LichessBotScriptedURLProtocol.releaseHeld(status: 200)
        await fulfillment(of: [answered], timeout: 10)
        heldRequest.cancel()
        XCTAssertEqual(try XCTUnwrap(heldResult.value).get(), 200)
    }

    /// Protects: a response of any status between failures keeps the
    /// session: Lichess answered, so the connection works.
    /// Fails if: an error status counts as a connection failure, or
    /// failures are counted in total.
    func testAnErrorStatusBetweenTimeoutsKeepsTheSession() async throws {
        LichessBotScriptedURLProtocol.load([.fail(.timedOut), .status(503), .fail(.timedOut), .status(200)])
        let resets = SyncBox<[LichessBotRequestConnectionHealth.Reset]>([])
        let transport = makeTransport(resets: resets)
        for _ in 0..<4 {
            _ = try await Self.send(transport)
        }
        XCTAssertEqual(resets.value, [])
        XCTAssertEqual(LichessBotScriptedURLProtocol.served, 4)
    }
}
