import CryptoKit
import Foundation

// MARK: - Errors

enum ModelCheckpointError: LocalizedError {
    case fileTooShort
    case magicMismatch
    case unsupportedVersion(UInt32)
    case archMismatch(expected: UInt32, got: UInt32)
    case tensorCountMismatch(expected: Int, got: Int)
    case tensorIndexMismatch(expected: Int, got: UInt32)
    case elementCountMismatch(tensorIndex: Int, expected: Int, got: Int)
    case implausibleTensorSize(tensorIndex: Int, elementCount: Int, maxAllowed: Int)
    case truncated
    case trailingBytesAfterPayload(remaining: Int)
    case hashMismatch
    case invalidJSON(Error)
    case invalidUTF8
    case encodingFailed(String)

    var errorDescription: String? {
        switch self {
        case .fileTooShort:
            return "File is shorter than the minimum valid .dcmmodel size"
        case .magicMismatch:
            return "Not a .dcmmodel file (magic bytes mismatch)"
        case .unsupportedVersion(let v):
            return "Unsupported .dcmmodel format version \(v)"
        case .archMismatch(let expected, let got):
            return "Architecture mismatch: file was saved with archHash 0x\(String(got, radix: 16)), current build expects 0x\(String(expected, radix: 16))"
        case .tensorCountMismatch(let expected, let got):
            return "Tensor count mismatch: file has \(got) tensors, current build expects \(expected)"
        case .tensorIndexMismatch(let expected, let got):
            return "Tensor index mismatch at position \(expected) (file says \(got))"
        case .elementCountMismatch(let i, let expected, let got):
            return "Tensor \(i) element count mismatch: expected \(expected), got \(got)"
        case .implausibleTensorSize(let i, let elementCount, let maxAllowed):
            return "Tensor \(i) reports \(elementCount) elements, exceeds sanity cap \(maxAllowed) — file is malformed or corrupted"
        case .truncated:
            return "File is truncated — ran out of bytes before the declared payload"
        case .trailingBytesAfterPayload(let remaining):
            return "\(remaining) trailing bytes after payload but before SHA — file is malformed"
        case .hashMismatch:
            return "Integrity check failed: SHA-256 hash at end of file does not match content"
        case .invalidJSON(let err):
            return "Invalid metadata JSON: \(err.localizedDescription)"
        case .invalidUTF8:
            return "Invalid UTF-8 in string field"
        case .encodingFailed(let detail):
            return "Encoding failed: \(detail)"
        }
    }
}

// MARK: - Metadata

/// Descriptive metadata written alongside a model's weights in a
/// `.dcmmodel` file. Extensible — new fields can be added in this
/// JSON blob without breaking the fixed binary header.
struct ModelCheckpointMetadata: Codable, Equatable {
    /// Source of the save: `manual`, `promote`, or `session-autosave`.
    let creator: String
    /// Training step at mint time, if the model came from a live
    /// training session. Nil for standalone builds.
    let trainingStep: Int?
    /// Parent model ID (e.g. the champion that the trainer was forked
    /// from, or the previous champion prior to a promotion). Empty
    /// string if none.
    let parentModelID: String
    /// Short free-form note for the human reading the folder.
    let notes: String
}

// MARK: - Binary format

/// `.dcmmodel` — flat binary checkpoint for one `ChessNetwork` worth
/// of persistent state (trainable weights + BN running stats).
///
/// Format (little-endian throughout):
///
///     [  0 ..  8 ]  magic "DCMMODEL"
///     [  8 .. 12 ]  u32  formatVersion
///     [ 12 .. 16 ]  u32  archHash
///     [ 16 .. 20 ]  u32  numTensors
///     [ 20 .. 28 ]  i64  createdAtUnix
///     [ 28 .. 32 ]  u32  modelIDByteCount
///     [ 32 ..  m ]  utf-8 modelID
///     [  m ..  m+4] u32  metadataJSONByteCount
///     [  m+4..  q ] utf-8 metadataJSON
///     repeated `numTensors` times in declared order:
///         [ .. +4 ]  u32  tensorIndex    (must match its position)
///         [ .. +4 ]  u32  elementCount
///         [ .. .. ]  Float32 × elementCount
///     [ last 32 bytes ] SHA-256 over all preceding bytes
///
/// The trailing SHA guards against disk rot, truncation, and
/// hand-editing. `archHash` hard-refuses to load a file that was
/// minted against a different network shape — there is no
/// migration path.
struct ModelCheckpointFile {
    static let magic: [UInt8] = Array("DCMMODEL".utf8)
    /// Current write version. New encodes always emit this version.
    /// Decode accepts v1 (legacy: trainables + bn only) and v2
    /// (current: trainables + bn + optional optimizer state — used
    /// by trainer.dcmmodel to round-trip momentum velocity buffers
    /// per `ChessTrainer.exportTrainerWeights()`).
    /// TODO(persist-velocity, after 2026-06-04): consider tightening
    /// the decoder to v2-only once any in-flight v1 trainer.dcmmodel
    /// files have been re-saved as v2 in the wild.
    static let formatVersion: UInt32 = 2
    /// Versions the decoder accepts. v1: pre-momentum trainers and
    /// all champion/candidate/probe model files. v2: trainer files
    /// written after the momentum implementation lands; payload may
    /// contain extra tensors past the bn-running-stats slot.
    static let supportedReadVersions: Set<UInt32> = [1, 2]

    /// Hash of the shape constants that determine variable layout.
    /// Any change to `ChessNetwork.channels`, `numBlocks`,
    /// `inputPlanes`, `boardSize`, or `policySize` changes this
    /// value, so stale files refuse to load instead of silently
    /// landing in wrong-sized slots.
    static var currentArchHash: UInt32 {
        var h: UInt32 = 0x811C9DC5 // FNV-1a offset basis
        func mix(_ value: Int) {
            guard let u32 = UInt32(exactly: value) else {
                preconditionFailure("Architecture constant exceeds UInt32: \(value). Widen currentArchHash before using values this large.")
            }
            var v = u32.littleEndian
            withUnsafeBytes(of: &v) { raw in
                for byte in raw {
                    h ^= UInt32(byte)
                    h = h &* 0x01000193
                }
            }
        }
        mix(ChessNetwork.channels)
        mix(ChessNetwork.numBlocks)
        mix(ChessNetwork.inputPlanes)
        mix(ChessNetwork.boardSize)
        mix(ChessNetwork.policySize)
        return h
    }

    /// Smallest legal encoded size — just the fixed header plus a
    /// zero-length modelID, zero-length metadata, and the 32-byte
    /// trailing SHA. Decoding data shorter than this can't produce
    /// a valid file. The tensor count itself is not sanity-checked
    /// here: `ChessNetwork.loadWeights` validates the count
    /// against the live variable list when the decoded weights
    /// are handed to it, so a wrong count is caught there with a
    /// precise error message.
    static let minimumEncodedSize: Int = 8 + 4 + 4 + 4 + 8 + 4 + 4 + 32

    /// Sanity cap on the per-tensor element count read from disk.
    /// Computed from the live arch constants so it auto-tracks any
    /// architecture change (channels, policy width, input planes,
    /// SE width) instead of needing a manual bump.
    ///
    /// Current largest tensors at the post-refresh architecture:
    /// - residual conv weights: `channels × channels × 9 = 147,456`
    /// - stem conv: `inputPlanes × channels × 9 = 23,040`
    /// - policy 1×1 conv: `channels × policyChannels = 9,728`
    /// - SE FC: `channels × (channels / r) = 4,096`
    /// All well below the cap. The 65,536-element slack lets a minor
    /// architectural tweak land without immediately tripping the
    /// implausibleTensorSize guard.
    ///
    /// Paired with the SHA-256 trailer (which already catches corruption
    /// pre-decode) this is defense-in-depth: if the hash ever matches a
    /// malformed element count, we still reject before allocating.
    static var maxTensorElementCount: Int {
        let residualConv = ChessNetwork.channels * ChessNetwork.channels * 9
        let stemConv = ChessNetwork.inputPlanes * ChessNetwork.channels * 9
        let policyConv = ChessNetwork.channels * ChessNetwork.policyChannels
        let seReduced = ChessNetwork.channels / ChessNetwork.seReductionRatio
        let seFC = ChessNetwork.channels * seReduced
        let largest = max(residualConv, stemConv, policyConv, seFC)
        return largest + 65_536
    }

    let modelID: String
    let createdAtUnix: Int64
    let metadata: ModelCheckpointMetadata
    /// One sub-array per persistent tensor.
    /// - For v1 files: order is `trainableVariables + bnRunningStatsVariables`.
    /// - For v2 files: order is `trainableVariables + bnRunningStatsVariables`,
    ///   optionally followed by trainer optimizer-state tensors (currently
    ///   the velocity buffers parallel to `trainableVariables`). The total
    ///   tensor count distinguishes the two payload shapes — the loader
    ///   inspects `weights.count` against the live variable counts to
    ///   decide how to bind them. Element count per sub-array matches the
    ///   tensor's static shape.
    let weights: [[Float]]
    /// Format version actually decoded. New encodes always write
    /// `Self.formatVersion` (currently 2). Decoded value is one of
    /// `Self.supportedReadVersions`. Callers (typically
    /// `ChessTrainer.loadTrainerWeights`) use this to disambiguate
    /// payload layout when the live variable count alone would be
    /// ambiguous.
    let formatVersion: UInt32

    /// Memberwise init with `formatVersion` defaulted to the current
    /// write version, so call sites that build a file for ENCODING
    /// don't need to specify it. The decode path passes the actual
    /// decoded version explicitly.
    init(
        modelID: String,
        createdAtUnix: Int64,
        metadata: ModelCheckpointMetadata,
        weights: [[Float]],
        formatVersion: UInt32 = ModelCheckpointFile.formatVersion
    ) {
        self.modelID = modelID
        self.createdAtUnix = createdAtUnix
        self.metadata = metadata
        self.weights = weights
        self.formatVersion = formatVersion
    }

    // MARK: Encoding

    /// Serialize the checkpoint to a `Data` blob including the
    /// trailing SHA-256. Caller writes the result atomically to
    /// disk.
    func encode() throws -> Data {
        var out = Data()
        out.reserveCapacity(Self.estimateEncodedSize(weights: weights))

        // Fixed header
        out.append(contentsOf: Self.magic)
        out.appendUInt32LE(Self.formatVersion)
        out.appendUInt32LE(Self.currentArchHash)
        out.appendUInt32LE(UInt32(weights.count))
        out.appendInt64LE(createdAtUnix)

        // Model ID (length-prefixed utf-8)
        guard let idBytes = modelID.data(using: .utf8) else {
            throw ModelCheckpointError.invalidUTF8
        }
        out.appendUInt32LE(UInt32(idBytes.count))
        out.append(idBytes)

        // Metadata JSON (length-prefixed utf-8). Sorted keys so the
        // same metadata struct always encodes to byte-identical bytes,
        // which matters for reproducible file hashes.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let metadataBytes: Data
        do {
            metadataBytes = try encoder.encode(metadata)
        } catch {
            throw ModelCheckpointError.invalidJSON(error)
        }
        out.appendUInt32LE(UInt32(metadataBytes.count))
        out.append(metadataBytes)

        // Tensors
        for (i, tensor) in weights.enumerated() {
            out.appendUInt32LE(UInt32(i))
            out.appendUInt32LE(UInt32(tensor.count))
            // Raw Float32 bytes. Apple platforms are all little-endian
            // and IEEE-754 conformant, so withUnsafeBytes on [Float]
            // gives the correct on-disk representation directly.
            tensor.withUnsafeBytes { raw in
                out.append(contentsOf: raw)
            }
        }

        // Trailing SHA-256
        let digest = SHA256.hash(data: out)
        out.append(contentsOf: digest)
        return out
    }

    private static func estimateEncodedSize(weights: [[Float]]) -> Int {
        var total = 8 + 4 + 4 + 4 + 8 // magic + version + arch + numTensors + createdAt
        total += 4 + 64                // modelID length + ~64 bytes
        total += 4 + 512               // metadata length + ~512 bytes
        for t in weights {
            total += 4 + 4 + t.count * 4
        }
        total += 32                    // SHA-256
        return total
    }

    // MARK: Decoding

    static func decode(_ data: Data) throws -> ModelCheckpointFile {
        guard data.count >= minimumEncodedSize else {
            throw ModelCheckpointError.fileTooShort
        }

        // Verify trailing SHA-256 before trusting anything else.
        let contentEnd = data.count - 32
        let content = data.subdata(in: 0..<contentEnd)
        let storedHash = data.subdata(in: contentEnd..<data.count)
        let computedHash = Data(SHA256.hash(data: content))
        guard storedHash == computedHash else {
            throw ModelCheckpointError.hashMismatch
        }

        var reader = DataReader(data: content)

        // Magic
        let magicBytes = try reader.readBytes(8)
        guard magicBytes == Self.magic else {
            throw ModelCheckpointError.magicMismatch
        }

        let version = try reader.readUInt32LE()
        guard Self.supportedReadVersions.contains(version) else {
            throw ModelCheckpointError.unsupportedVersion(version)
        }

        let archHash = try reader.readUInt32LE()
        guard archHash == Self.currentArchHash else {
            throw ModelCheckpointError.archMismatch(
                expected: Self.currentArchHash,
                got: archHash
            )
        }

        let numTensors = Int(try reader.readUInt32LE())

        let createdAtUnix = try reader.readInt64LE()

        // Model ID
        let idLen = Int(try reader.readUInt32LE())
        let idBytes = try reader.readBytes(idLen)
        guard let modelID = String(bytes: idBytes, encoding: .utf8) else {
            throw ModelCheckpointError.invalidUTF8
        }

        // Metadata
        let metadataLen = Int(try reader.readUInt32LE())
        let metadataBytes = try reader.readBytes(metadataLen)
        let metadata: ModelCheckpointMetadata
        do {
            metadata = try JSONDecoder().decode(
                ModelCheckpointMetadata.self,
                from: Data(metadataBytes)
            )
        } catch {
            throw ModelCheckpointError.invalidJSON(error)
        }

        // Tensors
        var weights: [[Float]] = []
        weights.reserveCapacity(numTensors)
        for expectedIndex in 0..<numTensors {
            let actualIndex = try reader.readUInt32LE()
            guard actualIndex == UInt32(expectedIndex) else {
                throw ModelCheckpointError.tensorIndexMismatch(
                    expected: expectedIndex,
                    got: actualIndex
                )
            }
            let elementCount = Int(try reader.readUInt32LE())
            guard elementCount >= 0, elementCount <= Self.maxTensorElementCount else {
                throw ModelCheckpointError.implausibleTensorSize(
                    tensorIndex: expectedIndex,
                    elementCount: elementCount,
                    maxAllowed: Self.maxTensorElementCount
                )
            }
            let (byteCount, overflowed) = elementCount.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
            guard !overflowed else {
                throw ModelCheckpointError.implausibleTensorSize(
                    tensorIndex: expectedIndex,
                    elementCount: elementCount,
                    maxAllowed: Self.maxTensorElementCount
                )
            }
            let raw = try reader.readBytes(byteCount)
            var floats = [Float](repeating: 0, count: elementCount)
            raw.withUnsafeBufferPointer { src in
                floats.withUnsafeMutableBufferPointer { dst in
                    guard let srcBase = src.baseAddress, let dstBase = dst.baseAddress else {
                        return
                    }
                    memcpy(dstBase, srcBase, byteCount)
                }
            }
            weights.append(floats)
        }

        let remaining = reader.remaining
        guard remaining == 0 else {
            throw ModelCheckpointError.trailingBytesAfterPayload(remaining: remaining)
        }

        return ModelCheckpointFile(
            modelID: modelID,
            createdAtUnix: createdAtUnix,
            metadata: metadata,
            weights: weights,
            formatVersion: version
        )
    }
}

// MARK: - Binary Data Reader

/// Minimal forward-only reader for a checkpoint payload. Tracks an
/// offset into a `Data` and throws `ModelCheckpointError.truncated`
/// if a read would go past the end. Not general-purpose — used
/// only by `ModelCheckpointFile.decode`.
private struct DataReader {
    let data: Data
    private var offset: Int = 0

    init(data: Data) {
        self.data = data
    }

    var remaining: Int { data.count - offset }

    mutating func readBytes(_ count: Int) throws -> [UInt8] {
        guard count >= 0, remaining >= count else {
            throw ModelCheckpointError.truncated
        }
        let slice = data.subdata(in: offset..<offset + count)
        offset += count
        return Array(slice)
    }

    mutating func readUInt32LE() throws -> UInt32 {
        let bytes = try readBytes(4)
        var raw: UInt32 = 0
        withUnsafeMutableBytes(of: &raw) { dest in
            bytes.withUnsafeBufferPointer { src in
                guard let srcBase = src.baseAddress, let destBase = dest.baseAddress else {
                    return
                }
                memcpy(destBase, srcBase, 4)
            }
        }
        return UInt32(littleEndian: raw)
    }

    mutating func readInt64LE() throws -> Int64 {
        let bytes = try readBytes(8)
        var raw: Int64 = 0
        withUnsafeMutableBytes(of: &raw) { dest in
            bytes.withUnsafeBufferPointer { src in
                guard let srcBase = src.baseAddress, let destBase = dest.baseAddress else {
                    return
                }
                memcpy(destBase, srcBase, 8)
            }
        }
        return Int64(littleEndian: raw)
    }
}

// MARK: - Data append helpers

private extension Data {
    mutating func appendUInt32LE(_ value: UInt32) {
        var le = value.littleEndian
        // Explicitly qualify with `Swift.` because the unqualified
        // `withUnsafeBytes` inside a `Data` extension resolves to the
        // instance method on `self`, not the global function.
        Swift.withUnsafeBytes(of: &le) { raw in
            append(contentsOf: raw)
        }
    }

    mutating func appendInt64LE(_ value: Int64) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { raw in
            append(contentsOf: raw)
        }
    }
}
