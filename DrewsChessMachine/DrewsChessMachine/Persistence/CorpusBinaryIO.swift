import Foundation

/// Little-endian byte writer for the game-corpus on-disk format. Kept
/// self-contained (rather than reusing the file-private `Data` helpers in
/// `ModelCheckpointFile`) so the corpus format carries no cross-file coupling.
struct CorpusByteWriter {
    private(set) var data = Data()

    mutating func appendUInt8(_ v: UInt8) { data.append(v) }
    mutating func appendUInt16LE(_ v: UInt16) { appendLE(v) }
    mutating func appendUInt32LE(_ v: UInt32) { appendLE(v) }
    mutating func appendUInt64LE(_ v: UInt64) { appendLE(v) }
    mutating func appendInt64LE(_ v: Int64) { appendLE(UInt64(bitPattern: v)) }
    mutating func appendBytes(_ b: [UInt8]) { data.append(contentsOf: b) }
    mutating func appendData(_ d: Data) { data.append(contentsOf: d) }

    private mutating func appendLE<T: FixedWidthInteger & UnsignedInteger>(_ v: T) {
        var le = v.littleEndian
        // Qualify with `Swift.` so the unqualified `withUnsafeBytes` does not
        // resolve to the `Data` instance method on a captured `self`.
        Swift.withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }
}

/// Bounds-checked little-endian reader. Copies its input to `[UInt8]` once so
/// every read is plain 0-based array indexing (no `Data`-slice index pitfalls).
struct CorpusByteReader {
    private let bytes: [UInt8]
    private(set) var offset: Int

    init(_ data: Data) {
        self.bytes = [UInt8](data)
        self.offset = 0
    }

    init(_ bytes: [UInt8]) {
        self.bytes = bytes
        self.offset = 0
    }

    /// Bytes not yet consumed.
    var remaining: Int { bytes.count - offset }
    /// Bytes consumed so far.
    var consumed: Int { offset }

    mutating func readBytes(_ count: Int) throws -> [UInt8] {
        guard count >= 0, offset + count <= bytes.count else {
            throw GameCorpusError.truncatedRecord
        }
        let slice = Array(bytes[offset..<offset + count])
        offset += count
        return slice
    }

    mutating func readData(_ count: Int) throws -> Data {
        Data(try readBytes(count))
    }

    mutating func readUInt8() throws -> UInt8 {
        let b = try readBytes(1)
        return b[0]
    }

    mutating func readUInt16LE() throws -> UInt16 {
        let b = try readBytes(2)
        return UInt16(b[0]) | (UInt16(b[1]) << 8)
    }

    mutating func readUInt32LE() throws -> UInt32 {
        let b = try readBytes(4)
        return UInt32(b[0])
            | (UInt32(b[1]) << 8)
            | (UInt32(b[2]) << 16)
            | (UInt32(b[3]) << 24)
    }

    mutating func readUInt64LE() throws -> UInt64 {
        let b = try readBytes(8)
        var v: UInt64 = 0
        for i in 0..<8 { v |= UInt64(b[i]) << (8 * i) }
        return v
    }

    mutating func readInt64LE() throws -> Int64 {
        Int64(bitPattern: try readUInt64LE())
    }
}
