import XCTest
@testable import DrewsChessMachine

/// `LichessBotStreamReader` — splitting a chunk stream and the stall
/// watchdog (Lichess bot plan §6). Uses real time with short limits, except
/// where a test drives the watchdog with `LichessBotManualTime`.
final class LichessBotStreamReaderTests: XCTestCase {

    private func collect(_ items: AsyncThrowingStream<LichessBotStreamItem, Error>) async -> (items: [LichessBotStreamItem], error: Error?) {
        var collected: [LichessBotStreamItem] = []
        do {
            for try await item in items {
                collected.append(item)
            }
            return (collected, nil)
        } catch {
            return (collected, error)
        }
    }

    func testSplitsLinesAcrossChunksAndEndsNormally() async {
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        continuation.yield(Data("{\"a\":".utf8))
        continuation.yield(Data("1}\n\n{\"b\":2}\n".utf8))
        continuation.finish()

        let items = LichessBotStreamReader.items(from: chunks, time: LichessBotSystemTimeSource(), stallTimeout: { nil }, checkInterval: .milliseconds(20))
        let result = await collect(items)
        XCTAssertNil(result.error)
        XCTAssertEqual(result.items, [.line(Data("{\"a\":1}".utf8)), .keepAlive, .line(Data("{\"b\":2}".utf8))])
    }

    func testReportsALineTruncatedByTheEndOfTheStream() async {
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        continuation.yield(Data("{\"a\":1}\n{\"partial".utf8))
        continuation.finish()

        let result = await collect(LichessBotStreamReader.items(from: chunks, time: LichessBotSystemTimeSource(), stallTimeout: { nil }, checkInterval: .milliseconds(20)))
        XCTAssertNil(result.error)
        XCTAssertEqual(result.items.last, .truncatedAtEnd(byteCount: Data("{\"partial".utf8).count))
    }

    func testSilenceBeyondTheLimitIsAStall() async {
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        continuation.yield(Data("\n".utf8))
        // No more bytes, and the stream is never finished.

        let started = ContinuousClock.now
        let result = await collect(LichessBotStreamReader.items(from: chunks, time: LichessBotSystemTimeSource(), stallTimeout: { .milliseconds(150) }, checkInterval: .milliseconds(20)))
        let elapsed = ContinuousClock.now - started
        XCTAssertEqual(result.items, [.keepAlive])
        guard case .stalled(let silence)? = result.error as? LichessBotStreamError else {
            return XCTFail("expected a stall, got \(String(describing: result.error))")
        }
        XCTAssertGreaterThan(silence, .milliseconds(150))
        XCTAssertLessThan(elapsed, .seconds(5), "the watchdog must fire promptly")
        continuation.finish()
    }

    /// Poll `condition` in real time. The manual clock below never moves on
    /// its own, so this only waits for tasks to reach a state; it never
    /// decides the outcome.
    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// Keep-alives spaced well inside the limit hold the stream open across
    /// a total silence far beyond it. The clock is manual: the watchdog is
    /// woken for every check between keep-alives, each only once it is
    /// waiting, so every check runs and sees exactly the silence the test
    /// set up.
    func testKeepAlivesPreventAStall() async throws {
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        let time = LichessBotManualTime()
        let stallLimit: Duration = .milliseconds(150)
        let checkInterval: Duration = .milliseconds(20)
        let checksBetweenKeepAlives = 2
        let keepAliveCount = 8
        let items = LichessBotStreamReader.items(from: chunks, time: time, stallTimeout: { stallLimit }, checkInterval: checkInterval)
        let received = SyncBox<[LichessBotStreamItem]>([])
        let consumer = Task { () -> Error? in
            do {
                for try await item in items {
                    received.modify { $0.append(item) }
                }
                return nil
            } catch {
                return error
            }
        }
        for keepAlive in 1...keepAliveCount {
            continuation.yield(Data("\n".utf8))
            // The reader records the activity before it yields the item.
            try await waitUntil("keep-alive \(keepAlive) is read") { received.value.count == keepAlive }
            for _ in 0..<checksBetweenKeepAlives {
                // The watchdog is the reader's only sleeper; once it waits
                // again, its previous check has run.
                try await waitUntil("the watchdog waits for its next check") { time.waitingSleeperCount == 1 }
                time.advance(by: checkInterval)
            }
        }
        XCTAssertLessThan(checkInterval * checksBetweenKeepAlives, stallLimit, "each gap is inside the limit")
        XCTAssertGreaterThan(checkInterval * (checksBetweenKeepAlives * keepAliveCount), stallLimit, "the gaps together are far beyond it")
        continuation.finish()
        let error = await consumer.value
        XCTAssertNil(error, "keep-alives inside the limit are not a stall")
        XCTAssertEqual(received.value.count, keepAliveCount)
    }

    func testNoLimitMeansNoStall() async {
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        let finisher = Task {
            try await Task.sleep(for: .milliseconds(300))
            continuation.finish()
        }
        let result = await collect(LichessBotStreamReader.items(from: chunks, time: LichessBotSystemTimeSource(), stallTimeout: { nil }, checkInterval: .milliseconds(20)))
        XCTAssertNil(result.error)
        do {
            try await finisher.value
        } catch {
            XCTFail("finisher failed: \(error)")
        }
    }

    func testTransportErrorsEndTheStream() async {
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        continuation.yield(Data("{\"a\":1}\n".utf8))
        continuation.finish(throwing: URLError(.networkConnectionLost))
        let result = await collect(LichessBotStreamReader.items(from: chunks, time: LichessBotSystemTimeSource(), stallTimeout: { nil }, checkInterval: .milliseconds(20)))
        XCTAssertEqual(result.items, [.line(Data("{\"a\":1}".utf8))])
        XCTAssertEqual((result.error as? URLError)?.code, .networkConnectionLost)
    }
}
