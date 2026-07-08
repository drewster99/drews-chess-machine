import Foundation
import CryptoKit

/// Errors from the game-corpus binary layer.
enum GameCorpusError: LocalizedError {
    case badFrontMagic
    case unsupportedVersion(UInt32)
    case truncatedHeader
    case truncatedRecord
    case recordCRCMismatch
    case corruptMove(promoCode: Int)
    case badTrailerMagic
    case shardSHAMismatch
    case gameCountMismatch(expected: Int, got: Int)
    case corruptMetadata(String)
    case invalidState(String)
    case ioFailed(String)

    var errorDescription: String? {
        switch self {
        case .badFrontMagic: return "Corpus shard front-header magic mismatch"
        case .unsupportedVersion(let v): return "Unsupported corpus shard format version \(v)"
        case .truncatedHeader: return "Corpus shard header truncated"
        case .truncatedRecord: return "Corpus shard record truncated"
        case .recordCRCMismatch: return "Corpus shard record CRC mismatch"
        case .corruptMove(let c): return "Corrupt packed move (promotion code \(c))"
        case .badTrailerMagic: return "Corpus shard trailer magic mismatch"
        case .shardSHAMismatch: return "Corpus shard SHA-256 mismatch"
        case .gameCountMismatch(let e, let g): return "Corpus shard game-count mismatch (trailer \(e), read \(g))"
        case .corruptMetadata(let s): return "Corpus metadata corrupt: \(s)"
        case .invalidState(let s): return "Invalid corpus state: \(s)"
        case .ioFailed(let s): return "Corpus I/O failed: \(s)"
        }
    }
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320) over a record
/// payload. A cheap per-record integrity check used to find the torn tail of
/// an open shard during crash recovery — distinct from the whole-shard
/// SHA-256 that guards a sealed shard.
enum CRC32 {
    private static let table: [UInt32] = {
        (0..<256).map { i -> UInt32 in
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1)
            }
            return c
        }
    }()

    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFFFFFF
        for byte in data {
            let idx = Int((crc ^ UInt32(byte)) & 0xFF)
            crc = table[idx] ^ (crc >> 8)
        }
        return crc ^ 0xFFFFFFFF
    }
}

/// On-disk constants and codecs for a corpus shard file.
///
/// A shard is `front header (256 B, fixed)` + `body` + `trailer (64 B, fixed)`:
/// the front header is written once at create; the body is a stream of framed
/// game records (`len`(u32) | payload | `crc32`(u32)); the trailer is appended
/// at seal and carries the counts and the SHA-256 over everything before it.
/// Seal-time facts live in the trailer rather than backfilled into the front,
/// so the file is pure append.
enum GameCorpusShardFormat {
    static let frontMagic: [UInt8] = Array("DCMGAME1".utf8)    // 8 bytes
    static let trailerMagic: [UInt8] = Array("DCMGSEAL".utf8)  // 8 bytes
    static let version: UInt32 = 1
    static let frontHeaderSize = 256
    static let trailerSize = 64

    struct FrontHeader: Equatable {
        var corpusID: String
        var sourceID: String
        var shardSeq: UInt32
        var createdAtUnix: Int64
    }

    static func encodeFrontHeader(_ h: FrontHeader) throws -> Data {
        var w = CorpusByteWriter()
        w.appendBytes(frontMagic)
        w.appendUInt32LE(version)
        let cid = Data(h.corpusID.utf8)
        let sid = Data(h.sourceID.utf8)
        w.appendUInt16LE(UInt16(cid.count))
        w.appendData(cid)
        w.appendUInt16LE(UInt16(sid.count))
        w.appendData(sid)
        w.appendUInt32LE(h.shardSeq)
        w.appendInt64LE(h.createdAtUnix)
        var data = w.data
        guard data.count <= frontHeaderSize else {
            throw GameCorpusError.corruptMetadata("front header overflow (\(data.count) > \(frontHeaderSize))")
        }
        if data.count < frontHeaderSize {
            data.append(Data(repeating: 0, count: frontHeaderSize - data.count))
        }
        return data
    }

    static func decodeFrontHeader(_ block: Data) throws -> FrontHeader {
        guard block.count >= frontHeaderSize else { throw GameCorpusError.truncatedHeader }
        var r = CorpusByteReader(block.subdata(in: block.startIndex..<(block.startIndex + frontHeaderSize)))
        let magic = try r.readBytes(8)
        guard magic == frontMagic else { throw GameCorpusError.badFrontMagic }
        let v = try r.readUInt32LE()
        guard v == version else { throw GameCorpusError.unsupportedVersion(v) }
        let cidLen = Int(try r.readUInt16LE())
        let cid = try r.readData(cidLen)
        let sidLen = Int(try r.readUInt16LE())
        let sid = try r.readData(sidLen)
        let shardSeq = try r.readUInt32LE()
        let createdAt = try r.readInt64LE()
        guard let corpusID = String(data: cid, encoding: .utf8),
              let sourceID = String(data: sid, encoding: .utf8) else {
            throw GameCorpusError.corruptMetadata("front header id utf8")
        }
        return FrontHeader(corpusID: corpusID, sourceID: sourceID, shardSeq: shardSeq, createdAtUnix: createdAt)
    }

    // MARK: Record codec

    static func encodeRecordPayload(_ game: GameRecord) -> Data {
        var w = CorpusByteWriter()
        var flags: UInt8 = 0
        if game.startFEN != nil { flags |= 0x01 }
        if game.terminationReason != nil { flags |= 0x02 }
        w.appendUInt8(flags)
        w.appendUInt8(game.outcome.rawValue)
        w.appendUInt8(game.terminationReason?.rawValue ?? 0)
        w.appendUInt32LE(UInt32(game.moves.count))
        for m in game.moves { w.appendUInt16LE(PackedMove.pack(m)) }
        if let fen = game.startFEN {
            let fenBytes = Data(fen.utf8)
            w.appendUInt16LE(UInt16(fenBytes.count))
            w.appendData(fenBytes)
        }
        return w.data
    }

    static func decodeRecordPayload(_ r: inout CorpusByteReader) throws -> GameRecord {
        let flags = try r.readUInt8()
        let outcomeRaw = try r.readUInt8()
        let reasonRaw = try r.readUInt8()
        let moveCount = Int(try r.readUInt32LE())
        guard let outcome = GameOutcome(rawValue: outcomeRaw) else {
            throw GameCorpusError.corruptMetadata("outcome \(outcomeRaw)")
        }
        var moves: [ChessMove] = []
        moves.reserveCapacity(moveCount)
        for _ in 0..<moveCount {
            moves.append(try PackedMove.unpack(try r.readUInt16LE()))
        }
        var startFEN: String? = nil
        if flags & 0x01 != 0 {
            let fenLen = Int(try r.readUInt16LE())
            let fenBytes = try r.readData(fenLen)
            guard let fen = String(data: fenBytes, encoding: .utf8) else {
                throw GameCorpusError.corruptMetadata("fen utf8")
            }
            startFEN = fen
        }
        var reason: GameTerminationReason? = nil
        if flags & 0x02 != 0 {
            reason = GameTerminationReason(rawValue: reasonRaw)
        }
        return GameRecord(startFEN: startFEN, moves: moves, outcome: outcome, terminationReason: reason)
    }

    /// One framed record: `len`(u32) | payload | `crc32`(u32 over payload).
    static func encodeFramedRecord(_ game: GameRecord) -> Data {
        let payload = encodeRecordPayload(game)
        var w = CorpusByteWriter()
        w.appendUInt32LE(UInt32(payload.count))
        w.appendData(payload)
        w.appendUInt32LE(CRC32.checksum(payload))
        return w.data
    }

    static func readFramedRecord(_ r: inout CorpusByteReader) throws -> GameRecord {
        let len = Int(try r.readUInt32LE())
        let payload = try r.readData(len)
        let crc = try r.readUInt32LE()
        guard crc == CRC32.checksum(payload) else { throw GameCorpusError.recordCRCMismatch }
        var pr = CorpusByteReader(payload)
        return try decodeRecordPayload(&pr)
    }
}

/// Force the file (or directory) at `url` to durable storage with
/// `F_FULLFSYNC`, falling back to `fsync` on filesystems that lack it. Mirrors
/// `CheckpointManager.fullSyncPath`, kept local so the corpus layer stands
/// alone.
func corpusFullSync(_ url: URL) throws {
    let fd = open(url.path, O_RDONLY)
    guard fd >= 0 else {
        throw GameCorpusError.ioFailed("open for fsync \(url.lastPathComponent): \(String(cString: strerror(errno)))")
    }
    defer { close(fd) }
    if fcntl(fd, F_FULLFSYNC) == -1 {
        if fsync(fd) == -1 {
            throw GameCorpusError.ioFailed("fsync \(url.lastPathComponent): \(String(cString: strerror(errno)))")
        }
    }
}

/// Appends games to a single open shard file and seals it.
///
/// Single-writer: the caller serializes access (recording will sit behind a
/// serial async queue). Maintains a streaming SHA-256 over the bytes written
/// so sealing is a finalize + trailer append with no re-read on the happy
/// path. The streaming hasher is rebuilt by reading the file only on the rare
/// crash-recovery resume path.
final class ShardWriter {
    /// The `….open` URL being appended to.
    let openURL: URL
    private let handle: FileHandle
    private var hasher: SHA256
    private(set) var byteCount: Int
    private(set) var gameCount: Int
    private(set) var plyCount: Int

    /// Create a fresh open shard: writes the 256-byte front header.
    init(creatingAt openURL: URL,
         header: GameCorpusShardFormat.FrontHeader) throws {
        let headerData = try GameCorpusShardFormat.encodeFrontHeader(header)
        do {
            try headerData.write(to: openURL, options: [.atomic])
        } catch {
            throw GameCorpusError.ioFailed("write header \(openURL.lastPathComponent): \(error.localizedDescription)")
        }
        self.openURL = openURL
        self.handle = try FileHandle(forWritingTo: openURL)
        _ = try self.handle.seekToEnd()
        var h = SHA256()
        h.update(data: headerData)
        self.hasher = h
        self.byteCount = headerData.count
        self.gameCount = 0
        self.plyCount = 0
    }

    /// Reopen an existing open shard, truncate it to a recovered valid extent,
    /// and continue appending. Rebuilds the streaming hasher from the truncated
    /// bytes.
    init(resumingAt openURL: URL,
         validByteCount: Int,
         gameCount: Int,
         plyCount: Int) throws {
        let h = try FileHandle(forWritingTo: openURL)
        try h.truncate(atOffset: UInt64(validByteCount))
        let existing = try Data(contentsOf: openURL)
        var hasher = SHA256()
        hasher.update(data: existing)
        _ = try h.seekToEnd()
        self.openURL = openURL
        self.handle = h
        self.hasher = hasher
        self.byteCount = validByteCount
        self.gameCount = gameCount
        self.plyCount = plyCount
    }

    func append(_ game: GameRecord) throws {
        try appendFramed(GameCorpusShardFormat.encodeFramedRecord(game), plyCount: game.moves.count)
    }

    /// Append a record already framed as `len|payload|crc` (e.g. encoded
    /// off-thread by `GameCorpusShardFormat.encodeFramedRecord`). `plyCount` is
    /// the game's move count, carried only for the running ply total. Lets a
    /// parallel producer do the encode/CRC work and leave this writer thin.
    func appendFramed(_ frame: Data, plyCount: Int) throws {
        do {
            try handle.write(contentsOf: frame)
        } catch {
            throw GameCorpusError.ioFailed("append record: \(error.localizedDescription)")
        }
        hasher.update(data: frame)
        byteCount += frame.count
        gameCount += 1
        self.plyCount += plyCount
    }

    /// Finalize the SHA, append the 64-byte trailer, flush once, close, and
    /// atomically rename `….open` → final. Returns the sealed file URL.
    @discardableResult
    func seal(sealUnix: Int64) throws -> URL {
        let digest = Data(hasher.finalize())
        var w = CorpusByteWriter()
        w.appendBytes(GameCorpusShardFormat.trailerMagic)
        w.appendInt64LE(Int64(gameCount))
        w.appendInt64LE(Int64(plyCount))
        w.appendInt64LE(sealUnix)
        w.appendData(digest)
        do {
            try handle.write(contentsOf: w.data)
            try handle.synchronize()
            try handle.close()
        } catch {
            throw GameCorpusError.ioFailed("seal write: \(error.localizedDescription)")
        }
        let finalURL = openURL.deletingPathExtension()
        do {
            try FileManager.default.moveItem(at: openURL, to: finalURL)
        } catch {
            throw GameCorpusError.ioFailed("seal rename: \(error.localizedDescription)")
        }
        return finalURL
    }

    /// Close and delete an open shard that holds no games (header only).
    func discardEmpty() throws {
        try handle.close()
        do { try FileManager.default.removeItem(at: openURL) } catch { /* best-effort cleanup */ }
    }

    /// Flush and close the file handle without sealing, leaving the `….open`
    /// shard on disk for later recovery/resume. The writer is unusable after.
    func closeWithoutSealing() throws {
        try handle.synchronize()
        try handle.close()
    }
}

/// Reading and crash-recovery for shard files.
enum GameCorpusShardIO {
    struct SealedShard {
        var header: GameCorpusShardFormat.FrontHeader
        var gameCount: Int
        var plyCount: Int
        var sealUnix: Int64
        var games: [GameRecord]
    }

    /// Read and fully verify a sealed shard (trailer magic, SHA-256, and every
    /// record's CRC), returning all games in order.
    static func readSealed(at url: URL) throws -> SealedShard {
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch { throw GameCorpusError.ioFailed("read \(url.lastPathComponent): \(error.localizedDescription)") }

        let frontSize = GameCorpusShardFormat.frontHeaderSize
        let trailerSize = GameCorpusShardFormat.trailerSize
        guard data.count >= frontSize + trailerSize else { throw GameCorpusError.truncatedHeader }

        let base = data.startIndex
        let header = try GameCorpusShardFormat.decodeFrontHeader(data.subdata(in: base..<(base + frontSize)))
        let bodyEnd = data.count - trailerSize

        var tr = CorpusByteReader(data.subdata(in: (base + bodyEnd)..<data.endIndex))
        let tmagic = try tr.readBytes(8)
        guard tmagic == GameCorpusShardFormat.trailerMagic else { throw GameCorpusError.badTrailerMagic }
        let gameCount = Int(try tr.readInt64LE())
        let plyCount = Int(try tr.readInt64LE())
        let sealUnix = try tr.readInt64LE()
        let storedSHA = try tr.readData(32)
        let computed = Data(SHA256.hash(data: data.subdata(in: base..<(base + bodyEnd))))
        guard computed == storedSHA else { throw GameCorpusError.shardSHAMismatch }

        var rr = CorpusByteReader(data.subdata(in: (base + frontSize)..<(base + bodyEnd)))
        var games: [GameRecord] = []
        games.reserveCapacity(gameCount)
        while rr.remaining > 0 {
            games.append(try GameCorpusShardFormat.readFramedRecord(&rr))
        }
        guard games.count == gameCount else {
            throw GameCorpusError.gameCountMismatch(expected: gameCount, got: games.count)
        }
        return SealedShard(header: header,
                           gameCount: gameCount,
                           plyCount: plyCount,
                           sealUnix: sealUnix,
                           games: games)
    }

    /// Cheap counts-only read of a sealed shard: seeks straight to the 64-byte
    /// trailer and decodes `(gameCount, plyCount)` without reading or
    /// SHA/CRC-verifying the body. For building a per-shard game-count index
    /// (e.g. `--start-game-index` resolution and the resume `next_game_index`
    /// logging) where reading every full shard would be gratuitous I/O — a
    /// shard is ~64 MB, the trailer is 64 B. Trades the integrity check for
    /// speed; callers that need verified games still use `readSealed`.
    static func readSealedCounts(at url: URL) throws -> (gameCount: Int, plyCount: Int) {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) }
        catch { throw GameCorpusError.ioFailed("open \(url.lastPathComponent): \(error.localizedDescription)") }
        defer { try? handle.close() }

        let trailerSize = GameCorpusShardFormat.trailerSize
        let size = (try? handle.seekToEnd()) ?? 0
        guard size >= UInt64(GameCorpusShardFormat.frontHeaderSize + trailerSize) else {
            throw GameCorpusError.truncatedHeader
        }
        try handle.seek(toOffset: size - UInt64(trailerSize))
        guard let tdata = try handle.read(upToCount: trailerSize), tdata.count == trailerSize else {
            throw GameCorpusError.truncatedHeader
        }
        var tr = CorpusByteReader(tdata)
        let tmagic = try tr.readBytes(8)
        guard tmagic == GameCorpusShardFormat.trailerMagic else { throw GameCorpusError.badTrailerMagic }
        let gameCount = Int(try tr.readInt64LE())
        let plyCount = Int(try tr.readInt64LE())
        return (gameCount, plyCount)
    }

    /// Cheap header+counts read: decode the 256-byte front header and the
    /// 64-byte trailer, without reading or SHA/CRC-verifying the body. Gives the
    /// per-shard `sourceID` (front header) alongside the sealed
    /// `(gameCount, plyCount)` — enough for `CorpusValidator` to rebuild
    /// per-source aggregate counts and check the header's corpus/source identity
    /// in its fast (non-integrity) mode without paying to read shard bodies.
    static func readSealedHeaderAndCounts(at url: URL) throws
        -> (header: GameCorpusShardFormat.FrontHeader, gameCount: Int, plyCount: Int) {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) }
        catch { throw GameCorpusError.ioFailed("open \(url.lastPathComponent): \(error.localizedDescription)") }
        defer { try? handle.close() }

        let frontSize = GameCorpusShardFormat.frontHeaderSize
        let trailerSize = GameCorpusShardFormat.trailerSize
        let size = (try? handle.seekToEnd()) ?? 0
        guard size >= UInt64(frontSize + trailerSize) else { throw GameCorpusError.truncatedHeader }

        try handle.seek(toOffset: 0)
        guard let fdata = try handle.read(upToCount: frontSize), fdata.count == frontSize else {
            throw GameCorpusError.truncatedHeader
        }
        let header = try GameCorpusShardFormat.decodeFrontHeader(fdata)

        try handle.seek(toOffset: size - UInt64(trailerSize))
        guard let tdata = try handle.read(upToCount: trailerSize), tdata.count == trailerSize else {
            throw GameCorpusError.truncatedHeader
        }
        var tr = CorpusByteReader(tdata)
        let tmagic = try tr.readBytes(8)
        guard tmagic == GameCorpusShardFormat.trailerMagic else { throw GameCorpusError.badTrailerMagic }
        let gameCount = Int(try tr.readInt64LE())
        let plyCount = Int(try tr.readInt64LE())
        return (header, gameCount, plyCount)
    }

    struct OpenShardScan {
        var header: GameCorpusShardFormat.FrontHeader
        var validByteCount: Int
        var gameCount: Int
        var plyCount: Int
        var fileSize: Int
    }

    /// Scan an open shard, validating the header then each framed record in
    /// turn, stopping at the first torn/invalid record (CRC mismatch or
    /// truncation). Returns the byte extent of the last complete record so the
    /// caller can truncate the file to it.
    static func scanOpenShard(at url: URL) throws -> OpenShardScan {
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch { throw GameCorpusError.ioFailed("read \(url.lastPathComponent): \(error.localizedDescription)") }

        let frontSize = GameCorpusShardFormat.frontHeaderSize
        guard data.count >= frontSize else { throw GameCorpusError.truncatedHeader }
        let base = data.startIndex
        let header = try GameCorpusShardFormat.decodeFrontHeader(data.subdata(in: base..<(base + frontSize)))

        var validEnd = frontSize
        var gameCount = 0
        var plyCount = 0
        var rr = CorpusByteReader(data.subdata(in: (base + frontSize)..<data.endIndex))
        while rr.remaining > 0 {
            do {
                let game = try GameCorpusShardFormat.readFramedRecord(&rr)
                gameCount += 1
                plyCount += game.moves.count
                validEnd = frontSize + rr.consumed
            } catch {
                break
            }
        }
        return OpenShardScan(header: header,
                             validByteCount: validEnd,
                             gameCount: gameCount,
                             plyCount: plyCount,
                             fileSize: data.count)
    }
}
