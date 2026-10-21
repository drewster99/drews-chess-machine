//
//  SafetensorsFile.swift
//  DrewsChessMachine
//
//  Minimal, dependency-free reader/writer for the HuggingFace `safetensors`
//  format — the native on-disk format for model weights going forward. The
//  whole point is that a file we write is loadable by Python's `safetensors`
//  with no conversion step, so this implementation matches the canonical spec
//  exactly:
//
//      [ 8 bytes  ] u64 little-endian  = header length N
//      [ N bytes  ] UTF-8 JSON header
//      [ rest     ] tensor data region (concatenated, no gaps)
//
//  Header JSON: one key per tensor →
//      { "dtype": "F32", "shape": [...], "data_offsets": [begin, end] }
//  where offsets are relative to the start of the data region. Plus an optional
//  reserved `"__metadata__"` key whose value is a flat string→string map
//  (safetensors metadata is string-only — numbers/JSON are stringified by the
//  caller).
//
//  Scope: we read/write **F32** only (weights are always Float32 on disk; bf16
//  is a compute-only choice). Unknown dtypes on read are a clear error rather
//  than a silent misread. Apple platforms are little-endian, so float buffers
//  memcpy directly to/from the LE on-disk representation.
//
//  Integrity: `encode` injects `__metadata__["content_sha256"]` = SHA-256 over
//  the tensor-data region (not the header, which is self-protecting: a corrupt
//  header fails JSON parse). `decode` recomputes and throws on mismatch. This
//  replaces the trailing-SHA guard of the legacy `.dcmmodel` format.
//

import Foundation
import CryptoKit

// MARK: - Errors

enum SafetensorsError: Error, CustomStringConvertible {
    case truncated(String)
    case headerParseFailed(String)
    case headerNotObject
    case metadataNotStringMap
    case badTensorEntry(String)
    case unsupportedDType(name: String, dtype: String)
    case negativeDimension(name: String)
    case shapeProductOverflow(name: String)
    case shapeCountMismatch(name: String, declared: Int, bytes: Int)
    case shapeDataMismatch(name: String, shapeElements: Int, dataElements: Int)
    case offsetsOutOfRange(name: String)
    case dataRegionNotCovered(expected: Int, covered: Int)
    case contentHashMismatch(expected: String, got: String)

    var description: String {
        switch self {
        case .truncated(let what): return "safetensors: truncated (\(what))"
        case .headerParseFailed(let detail): return "safetensors: header JSON parse failed (\(detail))"
        case .headerNotObject: return "safetensors: header JSON is not an object"
        case .metadataNotStringMap: return "safetensors: __metadata__ is present but is not a string→string map"
        case .badTensorEntry(let name): return "safetensors: malformed entry for tensor '\(name)'"
        case .unsupportedDType(let name, let dtype): return "safetensors: tensor '\(name)' has unsupported dtype '\(dtype)' (only F32 supported)"
        case .negativeDimension(let name): return "safetensors: tensor '\(name)' has a negative shape dimension"
        case .shapeProductOverflow(let name): return "safetensors: tensor '\(name)' shape product overflows Int"
        case .shapeCountMismatch(let name, let declared, let bytes): return "safetensors: tensor '\(name)' shape implies \(declared) floats but byte range holds \(bytes)"
        case .shapeDataMismatch(let name, let shapeElements, let dataElements): return "safetensors: tensor '\(name)' shape implies \(shapeElements) elements but data has \(dataElements)"
        case .offsetsOutOfRange(let name): return "safetensors: tensor '\(name)' data_offsets out of range"
        case .dataRegionNotCovered(let expected, let covered): return "safetensors: data region is \(expected) bytes but tensors cover \(covered) (gaps/overlap not allowed)"
        case .contentHashMismatch(let expected, let got): return "safetensors: content_sha256 mismatch (file \(expected), computed \(got))"
        }
    }
}

// MARK: - Tensor

/// One named F32 tensor: its name, shape, and row-major (C-contiguous) values.
struct SafetensorsTensor: Sendable, Equatable {
    let name: String
    let shape: [Int]
    let data: [Float]
}

// MARK: - Codec

enum SafetensorsFile {

    static let contentHashKey = "content_sha256"

    /// Encode tensors (written in the given order, contiguously) + string
    /// metadata into canonical safetensors bytes. Injects `content_sha256`
    /// over the data region into `__metadata__`.
    static func encode(tensors: [SafetensorsTensor], metadata: [String: String]) throws -> Data {
        // 1. Lay out the data region contiguously, recording offsets.
        var dataRegion = Data()
        var entries: [String: Any] = [:]
        for t in tensors {
            // Catch arch/builder drift at SAVE time rather than only on reload:
            // the declared shape must account for exactly the data we're writing.
            let shapeElements = t.shape.reduce(1, *)
            guard shapeElements == t.data.count else {
                throw SafetensorsError.shapeDataMismatch(
                    name: t.name, shapeElements: shapeElements, dataElements: t.data.count
                )
            }
            let begin = dataRegion.count
            t.data.withUnsafeBufferPointer { buf in
                dataRegion.append(buf)   // host LE == on-disk LE on Apple Silicon
            }
            let end = dataRegion.count
            entries[t.name] = [
                "dtype": "F32",
                "shape": t.shape,
                "data_offsets": [begin, end],
            ] as [String: Any]
        }

        // 2. Integrity hash over the data region; merge into metadata.
        var meta = metadata
        meta[contentHashKey] = sha256Hex(dataRegion)
        entries["__metadata__"] = meta

        // 3. Serialize header (sorted keys → deterministic file bytes).
        let headerData = try JSONSerialization.data(
            withJSONObject: entries,
            options: [.sortedKeys]
        )

        // 4. Assemble: u64 LE header length + header + data.
        var out = Data()
        var headerLen = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { out.append(contentsOf: $0) }
        out.append(headerData)
        out.append(dataRegion)
        return out
    }

    /// Decode safetensors bytes into ordered tensors (by ascending data offset
    /// = write order) + string metadata. Verifies `content_sha256` if present.
    /// The header of a safetensors file: every tensor's name, shape and byte
    /// range in the data region (sorted by offset), and `__metadata__`.
    struct Header: Sendable, Equatable {
        struct Entry: Sendable, Equatable {
            let name: String
            let shape: [Int]
            /// Byte range in the data region (after the header).
            let begin: Int
            let end: Int
            var elementCount: Int { (end - begin) / MemoryLayout<Float>.size }
        }
        let entries: [Entry]
        let metadata: [String: String]
    }

    static func decode(_ data: Data) throws -> (tensors: [SafetensorsTensor], metadata: [String: String]) {
        let (header, dataStart) = try headerAndDataStart(of: data)
        let dataRegion = data.subdata(in: dataStart..<data.count)

        var tensors: [SafetensorsTensor] = []
        tensors.reserveCapacity(header.entries.count)
        for entry in header.entries {
            let slice = dataRegion.subdata(in: entry.begin..<entry.end)
            let floats = slice.withUnsafeBytes { raw -> [Float] in
                Array(raw.bindMemory(to: Float.self))
            }
            tensors.append(SafetensorsTensor(name: entry.name, shape: entry.shape, data: floats))
        }

        // Integrity.
        if let expected = header.metadata[contentHashKey] {
            let got = sha256Hex(dataRegion)
            guard got == expected else {
                throw SafetensorsError.contentHashMismatch(expected: expected, got: got)
            }
        }

        return (tensors, header.metadata)
    }

    /// The header of a whole file's bytes, without reading its tensor data or
    /// checking its content hash: what `readHeader` gives for the file these
    /// bytes were read from. For a caller that already holds the bytes and
    /// must describe exactly the file it loads (`--export-inference-graph`).
    static func header(of data: Data) throws -> Header {
        try headerAndDataStart(of: data).header
    }

    private static func headerAndDataStart(of data: Data) throws -> (header: Header, dataStart: Int) {
        guard data.count >= headerPrefixByteCount else { throw SafetensorsError.truncated("header length prefix") }
        let dataStart = try dataStartOffset(headerLength: data.withUnsafeBytes { raw -> UInt64 in
            UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self))
        }, fileByteCount: data.count)
        let header = try parseHeader(data.subdata(in: headerPrefixByteCount..<dataStart),
                                     dataRegionByteCount: data.count - dataStart)
        return (header, dataStart)
    }

    /// The header of the file at `url`, read without its tensor data: the same
    /// structural checks `decode` applies to the header (`parseHeader`), against
    /// the file's actual size. The content hash is not checked — that needs the
    /// data.
    static func readHeader(at url: URL) throws -> Header {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            do { try handle.close() } catch {
                SessionLogger.shared.log("[SAFETENSORS] closing \(url.lastPathComponent) after a header read failed: \(error)")
            }
        }
        let fileByteCount = try handle.seekToEnd()
        try handle.seek(toOffset: 0)
        guard let prefix = try handle.read(upToCount: headerPrefixByteCount), prefix.count == headerPrefixByteCount else {
            throw SafetensorsError.truncated("header length prefix")
        }
        let headerLength = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard fileByteCount <= UInt64(Int.max) else { throw SafetensorsError.truncated("file larger than Int.max") }
        let dataStart = try dataStartOffset(headerLength: headerLength, fileByteCount: Int(fileByteCount))
        let headerByteCount = dataStart - headerPrefixByteCount
        guard let headerData = try handle.read(upToCount: headerByteCount), headerData.count == headerByteCount else {
            throw SafetensorsError.truncated("header")
        }
        return try parseHeader(headerData, dataRegionByteCount: Int(fileByteCount) - dataStart)
    }

    /// Bytes before the header JSON: its little-endian u64 length.
    private static let headerPrefixByteCount = 8

    /// Where the data region starts, refusing a declared header length the file
    /// can't hold. Checked before any `Int` conversion: a length above
    /// `Int.max` would trap on the cast, and external or hand-edited files reach
    /// this path.
    private static func dataStartOffset(headerLength: UInt64, fileByteCount: Int) throws -> Int {
        guard headerLength <= UInt64(fileByteCount - headerPrefixByteCount) else {
            throw SafetensorsError.truncated("declared header length \(headerLength) exceeds file")
        }
        return headerPrefixByteCount + Int(headerLength)
    }

    /// Parses and checks a header against the size of the data region it
    /// describes: a JSON object; `__metadata__`, when present, a string→string
    /// map; every tensor F32 with a non-negative shape whose element count
    /// matches its byte range; byte ranges inside the data region and covering it
    /// contiguously, without gaps or overlap. The one implementation of these
    /// checks, shared by `decode` and `readHeader`.
    static func parseHeader(_ headerData: Data, dataRegionByteCount: Int) throws -> Header {
        let obj: Any
        do {
            obj = try JSONSerialization.jsonObject(with: headerData)
        } catch {
            throw SafetensorsError.headerParseFailed(error.localizedDescription)
        }
        guard let header = obj as? [String: Any] else { throw SafetensorsError.headerNotObject }

        var metadata: [String: String] = [:]
        if let metaValue = header["__metadata__"] {
            // Must be a string→string map; a non-conforming __metadata__ would
            // otherwise silently drop content_sha256 and skip the integrity check.
            guard let meta = metaValue as? [String: String] else {
                throw SafetensorsError.metadataNotStringMap
            }
            metadata = meta
        }

        var entries: [Header.Entry] = []
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any] else { throw SafetensorsError.badTensorEntry(name) }
            guard let dtype = entry["dtype"] as? String else { throw SafetensorsError.badTensorEntry(name) }
            guard dtype == "F32" else { throw SafetensorsError.unsupportedDType(name: name, dtype: dtype) }
            guard let shapeAny = entry["shape"] as? [Any] else { throw SafetensorsError.badTensorEntry(name) }
            let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeAny.count else { throw SafetensorsError.badTensorEntry(name) }
            guard let offs = entry["data_offsets"] as? [Any], offs.count == 2,
                  let begin = (offs[0] as? NSNumber)?.intValue,
                  let end = (offs[1] as? NSNumber)?.intValue else {
                throw SafetensorsError.badTensorEntry(name)
            }
            guard begin >= 0, end >= begin, end <= dataRegionByteCount else {
                throw SafetensorsError.offsetsOutOfRange(name: name)
            }
            entries.append(Header.Entry(name: name, shape: shape, begin: begin, end: end))
        }
        entries.sort { $0.begin < $1.begin }

        // Coverage check: contiguous, no gaps/overlap.
        var cursor = 0
        for entry in entries {
            guard entry.begin == cursor else {
                throw SafetensorsError.dataRegionNotCovered(expected: dataRegionByteCount, covered: cursor)
            }
            cursor = entry.end
        }
        guard cursor == dataRegionByteCount else {
            throw SafetensorsError.dataRegionNotCovered(expected: dataRegionByteCount, covered: cursor)
        }

        for entry in entries {
            // Overflow-safe shape product: dims are attacker-controlled (e.g.
            // [Int.max, 2]); a plain reduce(*) would trap and crash instead of
            // failing cleanly.
            var declared = 1
            for dim in entry.shape {
                guard dim >= 0 else { throw SafetensorsError.negativeDimension(name: entry.name) }
                let (product, overflow) = declared.multipliedReportingOverflow(by: dim)
                guard !overflow else { throw SafetensorsError.shapeProductOverflow(name: entry.name) }
                declared = product
            }
            let (declaredBytes, byteOverflow) = declared.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
            guard !byteOverflow, entry.end - entry.begin == declaredBytes else {
                throw SafetensorsError.shapeCountMismatch(name: entry.name, declared: declared, bytes: entry.end - entry.begin)
            }
        }
        return Header(entries: entries, metadata: metadata)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
