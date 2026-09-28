import XCTest
@testable import DrewsChessMachine

/// `LichessBotStreamReader` — splitting a chunk stream and the stall
/// watchdog (Lichess bot plan §6). Uses real time with short limits.
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

    func testKeepAlivesPreventAStall() async {
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        let feeder = Task {
            for _ in 0..<8 {
                continuation.yield(Data("\n".utf8))
                try await Task.sleep(for: .milliseconds(40))
            }
            continuation.finish()
        }
        let result = await collect(LichessBotStreamReader.items(from: chunks, time: LichessBotSystemTimeSource(), stallTimeout: { .milliseconds(150) }, checkInterval: .milliseconds(20)))
        XCTAssertNil(result.error, "keep-alives inside the limit are not a stall")
        XCTAssertEqual(result.items.count, 8)
        do {
            try await feeder.value
        } catch {
            XCTFail("feeder failed: \(error)")
        }
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
