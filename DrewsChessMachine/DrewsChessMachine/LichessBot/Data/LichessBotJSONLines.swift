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
    /// optionally force it to disk. Call only on `LichessBotFileQueue`.
    static func append(_ data: Data, to url: URL, synchronize: Bool) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard fm.createFile(atPath: url.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path])
            }
        }
        let handle = try FileHandle(forWritingTo: url)
        do {
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            if synchronize {
                try handle.synchronize()
            }
        } catch {
            do {
                try handle.close()
            } catch let closeError {
                SessionLogger.shared.log("[ALARM] LICHESS-BOT closing \(url.lastPathComponent) after a failed write also failed: \(closeError.localizedDescription)")
            }
            throw error
        }
        try handle.close()
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
