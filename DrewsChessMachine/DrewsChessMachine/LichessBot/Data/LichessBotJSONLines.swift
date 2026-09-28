import Foundation

enum LichessBotJSONLinesError: LocalizedError, Equatable {
    /// A complete (newline-terminated) line that does not decode. Only an
    /// unterminated final line is tolerated; anything else is corruption
    /// and is reported, never skipped.
    case undecodableLine(file: String, lineNumber: Int, detail: String)

    var errorDescription: String? {
        switch self {
        case .undecodableLine(let file, let lineNumber, let detail):
            return "\(file) line \(lineNumber) is not a valid entry: \(detail)"
        }
    }
}

/// JSON Lines files: one JSON value per `\n`-terminated line, appended.
enum LichessBotJSONLines {

    /// Decoded lines, plus the byte count of an unterminated final line that
    /// was dropped. An append interrupted by a crash leaves exactly that
    /// shape, so it is dropped and reported rather than treated as
    /// corruption (plan §10.2).
    struct Decoded<Element> {
        let elements: [Element]
        let droppedTrailingByteCount: Int
    }
}

extension LichessBotJSONLines.Decoded: Sendable where Element: Sendable {}

extension LichessBotJSONLines {

    static func decode<Element: Decodable>(_ type: Element.Type, from data: Data, fileName: String) throws -> Decoded<Element> {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        var elements: [Element] = []
        var lineStart = data.startIndex
        var lineNumber = 0
        while let newline = data[lineStart...].firstIndex(of: UInt8(ascii: "\n")) {
            lineNumber += 1
            let line = data[lineStart..<newline]
            lineStart = data.index(after: newline)
            if line.allSatisfy({ $0 == UInt8(ascii: "\r") }) {
                continue
            }
            do {
                elements.append(try decoder.decode(Element.self, from: line))
            } catch {
                throw LichessBotJSONLinesError.undecodableLine(file: fileName, lineNumber: lineNumber, detail: String(describing: error))
            }
        }
        return Decoded(elements: elements, droppedTrailingByteCount: data.distance(from: lineStart, to: data.endIndex))
    }

    /// One encoded line, newline included.
    static func encodeLine<Element: Encodable>(_ element: Element) throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(element)
        data.append(UInt8(ascii: "\n"))
        return data
    }

    /// Append `data` to the file at `url`, creating it if needed, and
    /// optionally force it to disk. Call only on a `LichessBotFileQueue`.
    ///
    /// The file is opened with `O_APPEND`, so every write lands at the file's
    /// end as of that write: an append by another process (every DCM instance
    /// keeps a protocol log) is never overwritten, and a cut by
    /// `cutUnterminatedFinalLine` never leaves a hole of zeros.
    static func append(_ data: Data, to url: URL, synchronize: Bool) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        }
        let descriptor = open(url.path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else {
            let code = errno
            throw CocoaError(.fileWriteUnknown, userInfo: [
                NSFilePathErrorKey: url.path,
                NSUnderlyingErrorKey: NSError(domain: NSPOSIXErrorDomain, code: Int(code)),
            ])
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        do {
            try handle.write(contentsOf: data)
            if synchronize {
                try handle.synchronize()
            }
        } catch {
            closeAfterFailure(handle, url: url)
            throw error
        }
        try handle.close()
    }

    /// How many bytes each backward step of `cutUnterminatedFinalLine` reads
    /// while looking for the file's last newline.
    private static let tailScanChunkByteCount = 64 * 1024

    /// Cut an unterminated final line off the file at `url`, and return the
    /// bytes cut (empty when the file is empty or already ends in a newline).
    ///
    /// `decode` tolerates such a line, because an interrupted append leaves
    /// exactly that shape. An append written straight after it, though, joins
    /// the fragment and the new line into one complete line that doesn't
    /// decode — corruption, which `decode` refuses — so the file could never
    /// be read again. Appenders therefore cut the fragment before appending to
    /// a file whose end they haven't vouched for, and record what they cut:
    /// the bytes can't become an entry, but they stay on record. Truncating is
    /// the only safe repair; ending the fragment with a newline instead would
    /// make it exactly the complete bad line this prevents. Call only on a
    /// `LichessBotFileQueue`.
    static func cutUnterminatedFinalLine(of url: URL) throws -> Data {
        let handle = try FileHandle(forUpdating: url)
        let cut: Data
        do {
            let size = try handle.seekToEnd()
            var scanEnd = size
            var keptLength: UInt64 = 0
            var tail = Data()
            while scanEnd > 0 {
                let scanStart = scanEnd - min(scanEnd, UInt64(tailScanChunkByteCount))
                let wanted = Int(scanEnd - scanStart)
                try handle.seek(toOffset: scanStart)
                guard let chunk = try handle.read(upToCount: wanted), chunk.count == wanted else {
                    throw CocoaError(.fileReadUnknown, userInfo: [
                        NSFilePathErrorKey: url.path,
                        NSLocalizedDescriptionKey: "short read while checking the end of \(url.lastPathComponent)",
                    ])
                }
                if let newline = chunk.lastIndex(of: UInt8(ascii: "\n")) {
                    keptLength = scanStart + UInt64(chunk.distance(from: chunk.startIndex, to: newline)) + 1
                    tail = chunk[chunk.index(after: newline)...] + tail
                    break
                }
                tail = chunk + tail
                scanEnd = scanStart
            }
            if keptLength < size {
                try handle.truncate(atOffset: keptLength)
            }
            cut = tail
        } catch {
            closeAfterFailure(handle, url: url)
            throw error
        }
        try handle.close()
        return cut
    }

    /// Close a handle whose operation already failed. A close failure is
    /// logged, not thrown: the operation's own error is the one to report.
    private static func closeAfterFailure(_ handle: FileHandle, url: URL) {
        do {
            try handle.close()
        } catch {
            SessionLogger.shared.log("[ALARM] LICHESS-BOT closing \(url.lastPathComponent) after a failed operation also failed: \(error.localizedDescription)")
        }
    }
}

extension JSONEncoder.DateEncodingStrategy {
    /// ISO 8601 with milliseconds, so receive times keep sub-second order.
    static let iso8601withFractionalSeconds = JSONEncoder.DateEncodingStrategy.custom { date, encoder in
        var container = encoder.singleValueContainer()
        try container.encode(date.formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true)))
    }
}

extension JSONDecoder.DateDecodingStrategy {
    static let iso8601withFractionalSeconds = JSONDecoder.DateDecodingStrategy.custom { decoder in
        let container = try decoder.singleValueContainer()
        let text = try container.decode(String.self)
        do {
            return try Date.ISO8601FormatStyle(includingFractionalSeconds: true).parse(text)
        } catch {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "not an ISO 8601 timestamp with fractional seconds: \(text)")
        }
    }
}
