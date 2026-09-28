import XCTest
@testable import DrewsChessMachine

/// `LichessBotNDJSONSplitter` — line splitting across arbitrary network
/// chunk boundaries (Lichess bot plan §6).
final class LichessBotNDJSONTests: XCTestCase {

    private func bytes(_ text: String) -> [UInt8] {
        Array(text.utf8)
    }

    private func lineText(_ item: LichessBotNDJSONItem) -> String? {
        guard case .line(let data) = item else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    func testSplitsSeveralLinesInOneChunk() {
        var splitter = LichessBotNDJSONSplitter()
        let items = splitter.append(bytes("{\"a\":1}\n{\"b\":2}\n"))
        XCTAssertEqual(items.compactMap(lineText), ["{\"a\":1}", "{\"b\":2}"])
        XCTAssertEqual(splitter.pendingByteCount, 0)
    }

    func testBuffersALineSplitAcrossChunks() {
        var splitter = LichessBotNDJSONSplitter()
        XCTAssertEqual(splitter.append(bytes("{\"type\":\"game")), [])
        XCTAssertEqual(splitter.pendingByteCount, 13)
        let items = splitter.append(bytes("State\"}\n"))
        XCTAssertEqual(items.compactMap(lineText), ["{\"type\":\"gameState\"}"])
        XCTAssertEqual(splitter.pendingByteCount, 0)
    }

    func testEveryByteBoundaryGivesTheSameLines() {
        let text = "{\"a\":1}\n\n{\"b\":\"x y\"}\r\n\n{\"c\":3}\n"
        let expected: [LichessBotNDJSONItem] = {
            var whole = LichessBotNDJSONSplitter()
            return whole.append(bytes(text))
        }()
        XCTAssertEqual(expected.count, 5)
        for split in 0...text.utf8.count {
            var splitter = LichessBotNDJSONSplitter()
            let all = bytes(text)
            let items = splitter.append(all[0..<split]) + splitter.append(all[split...])
            XCTAssertEqual(items, expected, "split at \(split)")
        }
    }

    func testEmptyLinesAreKeepAlives() {
        var splitter = LichessBotNDJSONSplitter()
        XCTAssertEqual(splitter.append(bytes("\n\n")), [.keepAlive, .keepAlive])
    }

    func testCRLFIsTolerated() {
        var splitter = LichessBotNDJSONSplitter()
        let items = splitter.append(bytes("{\"a\":1}\r\n\r\n"))
        XCTAssertEqual(items.first.flatMap(lineText), "{\"a\":1}")
        XCTAssertEqual(items.last, .keepAlive)
    }

    func testOversizeLineIsDiscardedAndTheStreamContinues() {
        var splitter = LichessBotNDJSONSplitter(maximumLineLength: 8)
        let items = splitter.append(bytes("0123456789ABCDEF\n{\"ok\":1}\n"))
        XCTAssertEqual(items.count, 2)
        XCTAssertEqual(items.first, .oversizeLineDiscarded(byteCount: 16))
        XCTAssertEqual(items.last.flatMap(lineText), "{\"ok\":1}")
    }
}
