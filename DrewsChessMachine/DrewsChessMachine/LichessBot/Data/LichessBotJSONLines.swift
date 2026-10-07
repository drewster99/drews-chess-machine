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

    /// How `append` forces what it wrote to stable storage.
    enum Synchronization: Sendable, Equatable {
        /// Not forced. Once the write returns the bytes are in the kernel's
        /// cache, so they survive an app crash, but not a kernel panic or a
        /// power loss.
        case none
        /// `fsync`: the kernel hands the bytes to the drive. The drive's own
        /// write cache can still lose them in a power loss.
        case fsync
        /// `F_FULLFSYNC`, past the drive's write cache — and, when this
        /// append created the file, the file's folder too: the new name is
        /// an entry in the folder, which is flushed separately from the
        /// file's contents, so without it a power loss can keep the bytes
        /// and lose the name.
        case fullSync
    }

    /// The calls `append` makes once it holds the file's lock, as values so
    /// a test can see which ran (`F_FULLFSYNC` leaves no trace on disk) and
    /// prove the lock is held while they run. Production passes `system`;
    /// there is no default, so a writer can't pick up a test's calls by
    /// omission.
    struct AppendSystemCalls: Sendable {
        let write: @Sendable (_ data: Data, _ handle: FileHandle) throws -> Void
        let fsync: @Sendable (_ handle: FileHandle) throws -> Void
        let fullSync: @Sendable (_ handle: FileHandle, _ path: String) throws -> Void
        let fullSyncDirectory: @Sendable (_ directory: URL) throws -> Void

        static let system = AppendSystemCalls(
            write: { data, handle in try handle.write(contentsOf: data) },
            fsync: { handle in try handle.synchronize() },
            fullSync: { handle, path in try FileSafety.fullSync(fileDescriptor: handle.fileDescriptor, path: path) },
            fullSyncDirectory: { directory in try FileSafety.fullSync(at: directory) }
        )
    }

    /// What `append` found and did, besides writing.
    struct AppendOutcome: Sendable, Equatable {
        /// The unterminated final line cut off before the write; empty when
        /// the file was empty or ended in a newline.
        let cutTail: Data
        /// True when this append created the file.
        let createdFile: Bool
    }

    /// Append to the JSON Lines file at `url` — the one append path of the
    /// bot's journals and logs. Creates the file (and its folder) when
    /// absent. Call only on a `LichessBotFileQueue`.
    ///
    /// One locked step, all on one descriptor:
    /// 1. `FileSafety.openForAppending`: a symbolic link (even a dangling
    ///    one), folder, FIFO or other non-regular item at `url` is refused
    ///    and left untouched, never written through.
    /// 2. The file's exclusive `flock`, waiting for it. More than one process
    ///    appends to the same files — every DCM instance keeps a protocol
    ///    log, and a challenge withdrawal's result can be logged after
    ///    another instance has taken over the bot — and without the lock one
    ///    process's tail cut could truncate another's append in flight.
    /// 3. The tail check: an unterminated final line (an append that a crash
    ///    interrupted, in this process or another) is cut off
    ///    (`cutUnterminatedFinalLine(of:)` says why) and handed to `compose`,
    ///    which builds the bytes to write — so the caller records what was
    ///    cut ahead of its own lines, in the same write. The check runs on
    ///    every append, not only on the first one this launch makes to a
    ///    file: a launch can vouch for its own appends, but not for another
    ///    process that crashed mid-append since. It costs one `fstat` and a
    ///    one-byte read when the file ends cleanly.
    /// 4. The write. `O_APPEND` puts it at the file's end as of the write, so
    ///    it never lands in front of another writer's line, and a cut never
    ///    leaves a hole of zeros.
    /// 5. `synchronization`.
    /// 6. Close, which releases the lock.
    ///
    /// The lock orders only writers that take it. A build from before it
    /// appends without it, so in the moment between this tail check and its
    /// cut, such a build's new line can be cut along with a crash fragment.
    ///
    /// After a cut the bytes are gone from the file even if the write then
    /// fails, so callers put them in the session log first thing in
    /// `compose`, before anything else can fail.
    @discardableResult
    static func append(
        to url: URL,
        synchronization: Synchronization,
        systemCalls: AppendSystemCalls,
        composing compose: (_ cutTail: Data) throws -> Data
    ) throws -> AppendOutcome {
        if try FileSafety.existingItem(at: url) == nil {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        }
        let opened = try FileSafety.openForAppending(at: url)
        let handle = opened.handle
        let cutTail: Data
        do {
            try FileSafety.waitForExclusiveLock(onOpenFile: handle.fileDescriptor, path: url.path)
            cutTail = try cutUnterminatedFinalLine(ofOpenFile: handle.fileDescriptor, path: url.path)
            let data = try compose(cutTail)
            try systemCalls.write(data, handle)
            switch synchronization {
            case .none:
                break
            case .fsync:
                try systemCalls.fsync(handle)
            case .fullSync:
                try systemCalls.fullSync(handle, url.path)
                if opened.createdByThisCall {
                    try systemCalls.fullSyncDirectory(url.deletingLastPathComponent())
                }
            }
        } catch {
            closeAfterFailure(handle, url: url)
            throw error
        }
        try handle.close()
        return AppendOutcome(cutTail: cutTail, createdFile: opened.createdByThisCall)
    }

    /// How many bytes each backward step of the tail check reads while
    /// looking for the file's last newline.
    private static let tailScanChunkByteCount = 64 * 1024

    /// Cut an unterminated final line off the existing file at `url`, and
    /// return the bytes cut (empty when the file is empty or already ends in
    /// a newline).
    ///
    /// `decode` tolerates such a line, because an interrupted append leaves
    /// exactly that shape. An append written straight after it, though, joins
    /// the fragment and the new line into one complete line that doesn't
    /// decode — corruption, which `decode` refuses — so the file could never
    /// be read again. `append` therefore cuts the fragment first, under the
    /// file's lock, and its caller records what was cut: the bytes can't
    /// become an entry, but they stay on record. Truncating is the only safe
    /// repair; ending the fragment with a newline instead would make it
    /// exactly the complete bad line this prevents.
    ///
    /// The file is opened the way `append` opens it, minus creation
    /// (`FileSafety.openExistingRegularFileForAppending`): a missing file is
    /// an error, and a symbolic link, FIFO or other non-regular item is
    /// refused, so a link's target is never truncated. The cut runs under
    /// the lock `append` takes. Call only on a `LichessBotFileQueue`.
    static func cutUnterminatedFinalLine(of url: URL) throws -> Data {
        let opened = try FileSafety.openExistingRegularFileForAppending(at: url)
        let handle = opened.handle
        let cut: Data
        do {
            try FileSafety.waitForExclusiveLock(onOpenFile: handle.fileDescriptor, path: url.path)
            cut = try cutUnterminatedFinalLine(ofOpenFile: handle.fileDescriptor, path: url.path)
        } catch {
            closeAfterFailure(handle, url: url)
            throw error
        }
        try handle.close()
        return cut
    }

    /// The tail check and cut, on a descriptor the caller opened read-write
    /// and holds the lock of. A file whose last byte is a newline is left
    /// alone after reading just that byte; otherwise it is scanned backwards
    /// for its last newline and truncated just after it (to empty when it
    /// has none). `path` only labels a failure.
    private static func cutUnterminatedFinalLine(ofOpenFile descriptor: Int32, path: String) throws -> Data {
        var info = stat()
        guard fstat(descriptor, &info) == 0 else {
            let code = errno
            throw FileSafetyError.systemCallFailed(path: path, call: "fstat", errnoValue: code)
        }
        let size = Int64(info.st_size)
        guard size > 0 else { return Data() }
        let newline = UInt8(ascii: "\n")
        if try readExactly(descriptor, count: 1, at: size - 1, path: path).first == newline {
            return Data()
        }
        var scanEnd = size
        var keptLength: Int64 = 0
        var tail = Data()
        while scanEnd > 0 {
            let scanStart = scanEnd - min(scanEnd, Int64(tailScanChunkByteCount))
            let chunk = try readExactly(descriptor, count: Int(scanEnd - scanStart), at: scanStart, path: path)
            if let lastNewline = chunk.lastIndex(of: newline) {
                keptLength = scanStart + Int64(chunk.distance(from: chunk.startIndex, to: lastNewline)) + 1
                tail = chunk[chunk.index(after: lastNewline)...] + tail
                break
            }
            tail = chunk + tail
            scanEnd = scanStart
        }
        guard ftruncate(descriptor, off_t(keptLength)) == 0 else {
            let code = errno
            throw FileSafetyError.systemCallFailed(path: path, call: "ftruncate", errnoValue: code)
        }
        return tail
    }

    /// Exactly `count` bytes at `offset`, read with `pread` (which leaves the
    /// descriptor's offset alone). Running out of file first means it shrank
    /// while its lock was held — something that doesn't take the lock — so
    /// it is an error, not a shorter tail.
    private static func readExactly(_ descriptor: Int32, count: Int, at offset: Int64, path: String) throws -> Data {
        var buffer = Data(count: count)
        var filled = 0
        while filled < count {
            let result = buffer.withUnsafeMutableBytes { bytes -> Int in
                guard let base = bytes.baseAddress else { return 0 }
                return pread(descriptor, base + filled, count - filled, off_t(offset) + off_t(filled))
            }
            if result < 0 {
                let code = errno
                if code == EINTR { continue }
                throw FileSafetyError.systemCallFailed(path: path, call: "pread", errnoValue: code)
            }
            guard result > 0 else {
                throw CocoaError(.fileReadUnknown, userInfo: [
                    NSFilePathErrorKey: path,
                    NSLocalizedDescriptionKey: "short read while checking the end of \((path as NSString).lastPathComponent)",
                ])
            }
            filled += result
        }
        return buffer
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
