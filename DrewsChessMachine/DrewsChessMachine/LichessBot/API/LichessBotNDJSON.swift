import Foundation

/// One unit a Lichess NDJSON stream produces.
enum LichessBotNDJSONItem: Sendable, Equatable {
    /// A non-empty line: one JSON object, without its line terminator.
    case line(Data)
    /// An empty line. Lichess sends these periodically as keep-alives. They
    /// carry no data but prove the connection is alive, so the stream
    /// watchdog counts them as heartbeat.
    case keepAlive
    /// A line that grew past the length cap before its terminator arrived.
    /// Its bytes are discarded up to the next newline; the stream itself
    /// continues.
    case oversizeLineDiscarded(byteCount: Int)
}

/// Splits a Lichess NDJSON byte stream into lines.
///
/// Network chunks do not respect line boundaries: one chunk can carry
/// several lines, or end mid-line. The splitter buffers the partial tail
/// until its newline arrives. It splits on `\n`, strips one trailing `\r`
/// (tolerating `\r\n`), and reports empty lines as `.keepAlive`.
///
/// A line longer than `maximumLineLength` is not buffered without bound: the
/// splitter reports `.oversizeLineDiscarded` and skips to the next newline.
/// The largest real payload is a `gameFull` with a long move list, and the
/// default cap sits far above it.
///
/// Pure value type with no I/O, so it is tested directly.
struct LichessBotNDJSONSplitter: Sendable {
    static let defaultMaximumLineLength = 1 << 20

    let maximumLineLength: Int
    private var buffer = Data()
    /// True while skipping the remainder of an oversize line.
    private var discarding = false
    private var discardedByteCount = 0

    init(maximumLineLength: Int = LichessBotNDJSONSplitter.defaultMaximumLineLength) {
        self.maximumLineLength = maximumLineLength
    }

    /// Feed a chunk of bytes; returns every item it completes, in order.
    mutating func append<Bytes: Sequence>(_ bytes: Bytes) -> [LichessBotNDJSONItem] where Bytes.Element == UInt8 {
        var items: [LichessBotNDJSONItem] = []
        for byte in bytes {
            if byte == UInt8(ascii: "\n") {
                if discarding {
                    items.append(.oversizeLineDiscarded(byteCount: discardedByteCount))
                    discarding = false
                    discardedByteCount = 0
                } else {
                    items.append(Self.item(for: buffer))
                }
                buffer.removeAll(keepingCapacity: true)
                continue
            }
            if discarding {
                discardedByteCount += 1
                continue
            }
            buffer.append(byte)
            if buffer.count > maximumLineLength {
                discarding = true
                discardedByteCount = buffer.count
                buffer.removeAll(keepingCapacity: true)
            }
        }
        return items
    }

    /// Bytes buffered after the last newline — a line still in flight. When
    /// a stream ends, a non-empty remainder is a truncated line.
    var pendingByteCount: Int {
        discarding ? discardedByteCount : buffer.count
    }

    private static func item(for line: Data) -> LichessBotNDJSONItem {
        var content = line
        if content.last == UInt8(ascii: "\r") {
            content.removeLast()
        }
        let isBlank = content.allSatisfy { $0 == UInt8(ascii: " ") || $0 == UInt8(ascii: "\t") }
        return isBlank ? .keepAlive : .line(content)
    }
}
