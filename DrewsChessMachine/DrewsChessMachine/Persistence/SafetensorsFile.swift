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
    case badTensorEntry(String)
    case unsupportedDType(name: String, dtype: String)
    case shapeCountMismatch(name: String, declared: Int, bytes: Int)
    case offsetsOutOfRange(name: String)
    case dataRegionNotCovered(expected: Int, covered: Int)
    case contentHashMismatch(expected: String, got: String)

    var description: String {
        switch self {
        case .truncated(let what): return "safetensors: truncated (\(what))"
        case .headerParseFailed(let detail): return "safetensors: header JSON parse failed (\(detail))"
        case .headerNotObject: return "safetensors: header JSON is not an object"
        case .badTensorEntry(let name): return "safetensors: malformed entry for tensor '\(name)'"
        case .unsupportedDType(let name, let dtype): return "safetensors: tensor '\(name)' has unsupported dtype '\(dtype)' (only F32 supported)"
        case .shapeCountMismatch(let name, let declared, let bytes): return "safetensors: tensor '\(name)' shape implies \(declared) floats but byte range holds \(bytes)"
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
    static func decode(_ data: Data) throws -> (tensors: [SafetensorsTensor], metadata: [String: String]) {
        guard data.count >= 8 else { throw SafetensorsError.truncated("header length prefix") }

        let headerLen = data.withUnsafeBytes { raw -> UInt64 in
            UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self))
        }
        let headerStart = 8
        let dataStart = headerStart + Int(headerLen)
        guard data.count >= dataStart else { throw SafetensorsError.truncated("header") }

        let headerData = data.subdata(in: headerStart..<dataStart)
        let obj: Any
        do {
            obj = try JSONSerialization.jsonObject(with: headerData)
        } catch {
            throw SafetensorsError.headerParseFailed(error.localizedDescription)
        }
        guard let header = obj as? [String: Any] else { throw SafetensorsError.headerNotObject }

        let dataRegion = data.subdata(in: dataStart..<data.count)

        var metadata: [String: String] = [:]
        if let meta = header["__metadata__"] as? [String: String] { metadata = meta }

        struct Parsed { let name: String; let shape: [Int]; let begin: Int; let end: Int }
        var parsed: [Parsed] = []
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
            guard begin >= 0, end >= begin, end <= dataRegion.count else {
                throw SafetensorsError.offsetsOutOfRange(name: name)
            }
            parsed.append(Parsed(name: name, shape: shape, begin: begin, end: end))
        }
        parsed.sort { $0.begin < $1.begin }

        // Coverage check: contiguous, no gaps/overlap.
        var cursor = 0
        for p in parsed {
            guard p.begin == cursor else {
                throw SafetensorsError.dataRegionNotCovered(expected: dataRegion.count, covered: cursor)
            }
            cursor = p.end
        }
        guard cursor == dataRegion.count else {
            throw SafetensorsError.dataRegionNotCovered(expected: dataRegion.count, covered: cursor)
        }

        var tensors: [SafetensorsTensor] = []
        tensors.reserveCapacity(parsed.count)
        for p in parsed {
            let byteCount = p.end - p.begin
            let declared = p.shape.reduce(1, *)
            guard byteCount == declared * MemoryLayout<Float>.size else {
                throw SafetensorsError.shapeCountMismatch(name: p.name, declared: declared, bytes: byteCount)
            }
            let slice = dataRegion.subdata(in: p.begin..<p.end)
            let floats = slice.withUnsafeBytes { raw -> [Float] in
                Array(raw.bindMemory(to: Float.self))
            }
            tensors.append(SafetensorsTensor(name: p.name, shape: p.shape, data: floats))
        }

        // Integrity.
        if let expected = metadata[contentHashKey] {
            let got = sha256Hex(dataRegion)
            guard got == expected else {
                throw SafetensorsError.contentHashMismatch(expected: expected, got: got)
            }
        }

        return (tensors, metadata)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
