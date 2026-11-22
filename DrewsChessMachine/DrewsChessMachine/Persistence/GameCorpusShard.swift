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
    /// A write to the current source's shard failed, so the source stopped
    /// taking games: the shard was left unsealed (`--validate-corpus --fix`
    /// recovers its complete games) and every later append is refused with
    /// this error until `finishSource` records the source as incomplete.
    case sourceStopped(sourceID: String, cause: String)
    /// `corpus.json` is no longer the file that was read before the rewrite
    /// that would have replaced it — something else (a writer finishing its
    /// source) rewrote it in between — so it was left as found.
    case metadataChangedSinceRead(path: String)
    /// A sealed shard that verified when it was opened failed the streamed
    /// read that followed: the file changed in between, or a read failed.
    /// The games already returned were the verified content; the rest of the
    /// shard is not read.
    case streamedReadFailedAfterVerification(shard: String, detail: String)

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
        case let .sourceStopped(sourceID, cause):
            return "Corpus source \(sourceID) stopped recording after a failed write (\(cause)); its unsealed shard "
                + "keeps the games written before it, recoverable with --validate-corpus --fix"
        case .metadataChangedSinceRead(let path):
            return "Corpus metadata \(path) changed after it was read; it was left as found"
        case let .streamedReadFailedAfterVerification(shard, detail):
            return "Corpus shard \(shard) verified at open but failed its streamed read (\(detail))"
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

    // MARK: Trailer codec

    /// The fixed-size trailer a seal appends: the shard's counts, its seal
    /// time, and the SHA-256 over every byte before the trailer.
    struct Trailer: Equatable {
        var gameCount: Int
        var plyCount: Int
        var sealUnix: Int64
        /// 32 bytes.
        var sha256: Data
    }

    /// The trailer's bytes: magic, game count, ply count, seal time (each
    /// little-endian Int64) and the SHA-256 — exactly `trailerSize` bytes.
    /// The one encoder; `decodeTrailer` is its one inverse.
    static func encodeTrailer(_ trailer: Trailer) throws -> Data {
        guard trailer.sha256.count == 32 else {
            throw GameCorpusError.corruptMetadata("trailer SHA-256 is \(trailer.sha256.count) bytes, not 32")
        }
        var w = CorpusByteWriter()
        w.appendBytes(trailerMagic)
        w.appendInt64LE(Int64(trailer.gameCount))
        w.appendInt64LE(Int64(trailer.plyCount))
        w.appendInt64LE(trailer.sealUnix)
        w.appendData(trailer.sha256)
        let data = w.data
        guard data.count == trailerSize else {
            throw GameCorpusError.corruptMetadata("trailer is \(data.count) bytes, not \(trailerSize)")
        }
        return data
    }

    /// Decode exactly one trailer block (`trailerSize` bytes) after checking
    /// its magic. Every reader of a sealed shard decodes its trailer here,
    /// so the cheap trailer reads and the verified full read cannot disagree
    /// on the layout.
    static func decodeTrailer(_ block: Data) throws -> Trailer {
        guard block.count == trailerSize else { throw GameCorpusError.truncatedHeader }
        var r = CorpusByteReader(block)
        let magic = try r.readBytes(8)
        guard magic == trailerMagic else { throw GameCorpusError.badTrailerMagic }
        let gameCount = Int(try r.readInt64LE())
        let plyCount = Int(try r.readInt64LE())
        let sealUnix = try r.readInt64LE()
        let sha256 = try r.readData(32)
        return Trailer(gameCount: gameCount, plyCount: plyCount, sealUnix: sealUnix, sha256: sha256)
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
        return try decodeCheckedPayload(payload, crc: crc)
    }

    /// One framed record's payload, after checking it against the CRC its
    /// frame carries. The one CRC check and decode every reader of a framed
    /// record goes through (the in-memory `readFramedRecord` and the streamed
    /// `SealedShardBodyCursor`).
    static func decodeCheckedPayload(_ payload: Data, crc: UInt32) throws -> GameRecord {
        guard crc == CRC32.checksum(payload) else { throw GameCorpusError.recordCRCMismatch }
        var pr = CorpusByteReader(payload)
        return try decodeRecordPayload(&pr)
    }
}

/// Appends games to a single open shard file and seals it.
///
/// Single-writer: the caller serializes access (recording will sit behind a
/// serial async queue). Maintains a streaming SHA-256 over the bytes written
/// so sealing is a finalize + trailer append with no re-read on the happy
/// path. The streaming hasher is rebuilt by reading the file only on the rare
/// crash-recovery path.
///
/// The writer holds an exclusive lock on its `.open` file from the moment it
/// creates it (or, on recovery, takes it) until the file is sealed, discarded
/// or closed (`FileSafety.createNewFileHoldingExclusiveLock`). The lock is
/// how `GameCorpus.recoverOpenShard` tells a live writer's shard from a crash
/// leftover: the kernel drops it when the writer's process ends, however it
/// ends, so a leftover is never locked and a live shard always is. To keep
/// that true to the end, `seal` renames and `discardEmpty` removes the file
/// *before* closing it — closing first would leave a moment in which the
/// shard is unlocked but still `.open`, and a recovery in that moment would
/// take the trailer for a torn record and race the writer's rename.
final class ShardWriter {
    /// The `….open` URL being appended to.
    let openURL: URL
    private let handle: FileHandle
    /// The open shard's identity, read from the open handle, so
    /// `discardEmpty` removes this very file and never whatever else might
    /// have been put at `openURL` since.
    private let identity: FileSafety.FileIdentity
    private var hasher: SHA256
    private(set) var byteCount: Int
    private(set) var gameCount: Int
    private(set) var plyCount: Int

    /// Create a fresh open shard, holding its lock: writes the front header.
    /// The file is created exclusively — an existing item at `openURL`
    /// (another writer's open shard, debris, anything) is a hard error,
    /// never replaced.
    init(creatingAt openURL: URL,
         header: GameCorpusShardFormat.FrontHeader) throws {
        let headerData = try GameCorpusShardFormat.encodeFrontHeader(header)
        let created: FileSafety.NewFile
        do {
            created = try FileSafety.createNewFileHoldingExclusiveLock(at: openURL)
        } catch {
            throw GameCorpusError.ioFailed("create \(openURL.lastPathComponent): \(error.localizedDescription)")
        }
        do {
            try created.handle.write(contentsOf: headerData)
        } catch {
            // The file is ours (just created exclusively); remove exactly it
            // so a half-written header never looks like a shard — while it is
            // still open and locked, like every removal of a live shard.
            let writeError = error
            do {
                _ = try FileSafety.removeOwnedItem(at: openURL, identity: created.identity)
                try created.handle.close()
            } catch {
                throw GameCorpusError.ioFailed(
                    "write header \(openURL.lastPathComponent): \(writeError.localizedDescription); cleanup failed: \(error.localizedDescription)")
            }
            throw GameCorpusError.ioFailed("write header \(openURL.lastPathComponent): \(writeError.localizedDescription)")
        }
        self.openURL = openURL
        self.handle = created.handle
        self.identity = created.identity
        var h = SHA256()
        h.update(data: headerData)
        self.hasher = h
        self.byteCount = headerData.count
        self.gameCount = 0
        self.plyCount = 0
    }

    /// Take over a leftover open shard for recovery: `handle` is the shard
    /// open read-write with its lock already held
    /// (`FileSafety.openExistingRegularFileWithExclusiveLock`), `contents`
    /// the bytes read through it, and `scan` their scan. The file is cut to
    /// the scan's last complete record on that same locked handle — never
    /// reopened by path — and the streaming hasher is rebuilt from the kept
    /// bytes, so the shard can then be sealed or discarded.
    init(recoveringLockedShardAt openURL: URL,
         handle: FileHandle,
         identity: FileSafety.FileIdentity,
         contents: Data,
         scan: GameCorpusShardIO.OpenShardScan) throws {
        do {
            try handle.truncate(atOffset: UInt64(scan.validByteCount))
            _ = try handle.seekToEnd()
        } catch {
            throw GameCorpusError.ioFailed("truncate \(openURL.lastPathComponent) to its last complete game: \(error.localizedDescription)")
        }
        var hasher = SHA256()
        hasher.update(data: contents.prefix(scan.validByteCount))
        self.openURL = openURL
        self.handle = handle
        self.identity = identity
        self.hasher = hasher
        self.byteCount = scan.validByteCount
        self.gameCount = scan.gameCount
        self.plyCount = scan.plyCount
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

    /// Finalize the SHA, append the 64-byte trailer, flush once, atomically
    /// rename `….open` → final, and only then close (and so unlock) the file.
    /// The rename refuses (and the shard stays `….open`) when anything
    /// already has the final name, so sealing never replaces an existing
    /// sealed shard. The writer is unusable after this call, whether it
    /// returns or throws. Returns the sealed file URL.
    @discardableResult
    func seal(sealUnix: Int64) throws -> URL {
        let digest = Data(hasher.finalize())
        let trailer: Data
        do {
            trailer = try GameCorpusShardFormat.encodeTrailer(GameCorpusShardFormat.Trailer(
                gameCount: gameCount, plyCount: plyCount, sealUnix: sealUnix, sha256: digest))
        } catch let error as GameCorpusError {
            throw closeAfterFailure(error)
        } catch {
            throw closeAfterFailure(GameCorpusError.ioFailed("seal trailer: \(error.localizedDescription)"))
        }
        do {
            try handle.write(contentsOf: trailer)
            try handle.synchronize()
        } catch {
            throw closeAfterFailure(GameCorpusError.ioFailed("seal write: \(error.localizedDescription)"))
        }
        let finalURL = openURL.deletingPathExtension()
        do {
            try FileSafety.renameWithoutReplacing(from: openURL, to: finalURL)
        } catch {
            throw closeAfterFailure(GameCorpusError.ioFailed("seal rename: \(error.localizedDescription)"))
        }
        do {
            try handle.close()
        } catch {
            // The shard is sealed and published; only releasing the
            // descriptor failed. Said as such, so no caller retries a seal.
            throw GameCorpusError.ioFailed(
                "\(openURL.lastPathComponent) was sealed as \(finalURL.lastPathComponent), but closing it failed: \(error.localizedDescription)")
        }
        return finalURL
    }

    /// Delete an open shard that holds no games (header only), then close it.
    /// Removes only the file this writer has open (checked by identity);
    /// anything else now at `openURL`, or nothing at all, is an error rather
    /// than something to delete or ignore. The file is removed while still
    /// open and locked, so it is never an unlocked `.open` shard. The writer
    /// is unusable after this call, whether it returns or throws.
    func discardEmpty() throws {
        let removal: FileSafety.OwnedItemRemoval
        do {
            removal = try FileSafety.removeOwnedItem(at: openURL, identity: identity)
        } catch {
            throw closeAfterFailure(
                GameCorpusError.ioFailed("remove empty shard \(openURL.lastPathComponent): \(error.localizedDescription)"))
        }
        guard removal == .removed else {
            throw closeAfterFailure(
                GameCorpusError.ioFailed("remove empty shard \(openURL.lastPathComponent): it was already gone"))
        }
        do {
            try handle.close()
        } catch {
            throw GameCorpusError.ioFailed(
                "\(openURL.lastPathComponent) was removed, but closing it failed: \(error.localizedDescription)")
        }
    }

    /// Close the handle after a failed seal or discard and return `failure`
    /// to throw — with a failed close added to it, since the original
    /// failure is what the caller must see.
    private func closeAfterFailure(_ failure: GameCorpusError) -> GameCorpusError {
        do {
            try handle.close()
            return failure
        } catch {
            return GameCorpusError.ioFailed(
                "\(failure.localizedDescription); closing \(openURL.lastPathComponent) afterwards also failed: \(error.localizedDescription)")
        }
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
    /// record's CRC), returning all games in order. Holds every decoded game
    /// at once; corpus replay reads through `SealedShardGameStream` instead,
    /// which verifies the same way and then returns the games one at a time.
    static func readSealed(at url: URL) throws -> SealedShard {
        let opened = try OpenedSealedShard(url: url)
        var games: [GameRecord] = []
        try opened.verifyBody { games.append($0) }
        return SealedShard(header: opened.header,
                           gameCount: opened.trailer.gameCount,
                           plyCount: opened.trailer.plyCount,
                           sealUnix: opened.trailer.sealUnix,
                           games: games)
    }

    /// A sealed shard opened for reading: its handle, front header and
    /// trailer. Every verified read of a sealed shard starts here, so the
    /// all-at-once read and the stream check the same things in the same
    /// order. Closes its handle when released.
    final class OpenedSealedShard {
        let url: URL
        let handle: FileHandle
        let header: GameCorpusShardFormat.FrontHeader
        let trailer: GameCorpusShardFormat.Trailer
        /// File offset of the trailer: the end of the bytes the trailer's
        /// SHA-256 covers (front header + records).
        let bodyEnd: UInt64

        /// Open the shard and decode its front header and trailer. The checks
        /// run in the order the all-at-once read always ran them: size, front
        /// header, trailer. The body is not read.
        init(url: URL) throws {
            let handle: FileHandle
            do { handle = try FileHandle(forReadingFrom: url) }
            catch { throw GameCorpusError.ioFailed("read \(url.lastPathComponent): \(error.localizedDescription)") }
            do {
                let trailerBlock = try GameCorpusShardIO.readTrailerBlock(handle: handle, url: url)
                let bodyEnd = try handle.offset() - UInt64(GameCorpusShardFormat.trailerSize)
                try handle.seek(toOffset: 0)
                let frontSize = GameCorpusShardFormat.frontHeaderSize
                guard let front = try handle.read(upToCount: frontSize), front.count == frontSize else {
                    throw GameCorpusError.truncatedHeader
                }
                self.header = try GameCorpusShardFormat.decodeFrontHeader(front)
                self.trailer = try GameCorpusShardFormat.decodeTrailer(trailerBlock)
                self.bodyEnd = bodyEnd
            } catch let corpusError as GameCorpusError {
                GameCorpusShardIO.closeAfterRead(handle, url: url, what: "header")
                throw corpusError
            } catch {
                GameCorpusShardIO.closeAfterRead(handle, url: url, what: "header")
                throw GameCorpusError.ioFailed("read \(url.lastPathComponent): \(error.localizedDescription)")
            }
            self.url = url
            self.handle = handle
        }

        deinit {
            GameCorpusShardIO.closeAfterRead(handle, url: url, what: "shard")
        }

        /// One streamed pass over the whole body that verifies it exactly as
        /// the shard's integrity has always been judged — the SHA-256 first,
        /// then the first bad record, then the game count — handing each
        /// decoded game to `onGame` as it goes. A bad record stops the
        /// decoding but not the hashing, so a shard whose hash also fails
        /// reports the hash, as the all-at-once check (hash, then records)
        /// did. `onGame` may have seen games of a shard that then fails; the
        /// caller discards them with the error.
        func verifyBody(_ onGame: (GameRecord) -> Void) throws {
            do {
                var cursor = try SealedShardBodyCursor(shard: self)
                var decodedCount = 0
                var recordError: Error? = nil
                do {
                    while let game = try cursor.nextGame() {
                        onGame(game)
                        decodedCount += 1
                    }
                } catch let readFailure as SealedShardBodyCursor.ReadFailure {
                    throw readFailure
                } catch {
                    recordError = error
                }
                let digest = try cursor.finishHashing()
                guard digest == trailer.sha256 else { throw GameCorpusError.shardSHAMismatch }
                if let recordError { throw recordError }
                guard decodedCount == trailer.gameCount else {
                    throw GameCorpusError.gameCountMismatch(expected: trailer.gameCount, got: decodedCount)
                }
            } catch let readFailure as SealedShardBodyCursor.ReadFailure {
                throw readFailure.error
            }
        }
    }

    /// A sealed shard's games one at a time, read sequentially through a
    /// bounded window instead of decoded whole — corpus replay's read, so a
    /// shard costs the feed about `SealedShardBodyCursor.readChunkByteCount`
    /// of memory rather than every game it holds (one 64 MiB shard's 368,840
    /// games decoded to 1.36 GB of `[ChessMove]`).
    ///
    /// Opening verifies the whole shard first, exactly as `readSealed` does
    /// (`OpenedSealedShard.verifyBody`), so a shard that fails any check
    /// throws before it returns a single game — the guarantee the replay's
    /// skip-or-refuse handling of an unreadable shard rests on. The games are
    /// then read in a second streamed pass over the same open file, which
    /// checks every record's CRC again and, at the end, the SHA-256 and the
    /// game count (the front header at its start): the file was verified, not
    /// held, so a change made in between (the shard is sealed; nothing the app
    /// does rewrites it) or a failed read is thrown as
    /// `streamedReadFailedAfterVerification` instead of feeding content that
    /// was never verified. The second pass reads from the page cache the
    /// first one filled.
    ///
    /// Single-threaded; owned by one reader.
    final class SealedShardGameStream {
        private let shard: OpenedSealedShard
        private var cursor: SealedShardBodyCursor
        private var failure: GameCorpusError? = nil
        private var finished = false
        /// Games returned (or skipped) so far.
        private(set) var gamesRead = 0

        /// Games the shard holds (verified at open).
        var gameCount: Int { shard.trailer.gameCount }

        /// Open and fully verify the sealed shard at `url`; throws the same
        /// errors as `readSealed` for a shard that fails verification.
        init(verifyingSealedShardAt url: URL) throws {
            let shard = try OpenedSealedShard(url: url)
            try shard.verifyBody { _ in }
            let cursor: SealedShardBodyCursor
            do {
                cursor = try SealedShardBodyCursor(shard: shard)
            } catch {
                throw Self.failedAfterVerification(shard: shard, gamesRead: 0, error)
            }
            guard cursor.header == shard.header else {
                throw Self.failedAfterVerification(
                    shard: shard, gamesRead: 0,
                    GameCorpusError.corruptMetadata("front header differs from the one verified"))
            }
            self.shard = shard
            self.cursor = cursor
        }

        /// The next game in shard order, or nil after the last one (once the
        /// end-of-shard checks pass). After a failure every call rethrows it.
        func nextGame() throws -> GameRecord? {
            if let failure { throw failure }
            if finished { return nil }
            do {
                if let game = try cursor.nextGame() {
                    gamesRead += 1
                    return game
                }
                let digest = try cursor.finishHashing()
                guard digest == shard.trailer.sha256 else {
                    throw GameCorpusError.shardSHAMismatch
                }
                guard gamesRead == shard.trailer.gameCount else {
                    throw GameCorpusError.gameCountMismatch(expected: shard.trailer.gameCount, got: gamesRead)
                }
                finished = true
                return nil
            } catch {
                let failed = Self.failedAfterVerification(shard: shard, gamesRead: gamesRead, error)
                failure = failed
                throw failed
            }
        }

        /// Read past the next `count` games without returning them (a resume
        /// that starts part-way into the shard). Throws if the shard ends
        /// first.
        func skipGames(_ count: Int) throws {
            for skipped in 0..<count {
                guard try nextGame() != nil else {
                    throw GameCorpusError.gameCountMismatch(expected: count, got: skipped)
                }
            }
        }

        private static func failedAfterVerification(
            shard: OpenedSealedShard, gamesRead: Int, _ error: Error
        ) -> GameCorpusError {
            let underlying = (error as? SealedShardBodyCursor.ReadFailure)?.error ?? error
            return .streamedReadFailedAfterVerification(
                shard: shard.url.lastPathComponent,
                detail: "after \(gamesRead) of \(shard.trailer.gameCount) games: \(underlying.localizedDescription)")
        }
    }

    /// A sequential reader over one sealed shard's hashed bytes (front header
    /// and records), from offset 0 to the trailer, through a window of about
    /// `readChunkByteCount` bytes. Every byte is fed to a running SHA-256 as
    /// it is read; `finishHashing` reads and hashes whatever the caller did
    /// not consume, so the digest always covers exactly the bytes the
    /// trailer's hash covers.
    struct SealedShardBodyCursor {
        /// The read size. A record larger than this is read whole (the window
        /// grows to hold it); the bytes left before the trailer bound every
        /// read.
        static let readChunkByteCount = 1 << 20

        /// An I/O failure, as opposed to a content check, kept apart so a
        /// verification pass never reports it as a bad record.
        struct ReadFailure: Error {
            let error: GameCorpusError
        }

        private let handle: FileHandle
        private let url: URL
        private let bodyEnd: UInt64
        /// File offset of the next byte not yet read into `window`.
        private var fileOffset: UInt64 = 0
        private var window: [UInt8] = []
        /// Index in `window` of the next unconsumed byte.
        private var windowStart = 0
        private var hasher = SHA256()
        private var hashFinished = false
        /// The front header as this pass read it.
        private(set) var header: GameCorpusShardFormat.FrontHeader? = nil

        /// Start a pass at the beginning of the file; the front header is
        /// read (and hashed) here.
        init(shard: OpenedSealedShard) throws {
            self.handle = shard.handle
            self.url = shard.url
            self.bodyEnd = shard.bodyEnd
            do { try handle.seek(toOffset: 0) }
            catch { throw ReadFailure(error: .ioFailed("seek \(url.lastPathComponent): \(error.localizedDescription)")) }
            let front = try take(GameCorpusShardFormat.frontHeaderSize)
            self.header = try GameCorpusShardFormat.decodeFrontHeader(front)
        }

        private var buffered: Int { window.count - windowStart }
        private var unreadInFile: UInt64 { bodyEnd - fileOffset }

        /// The next framed record's game, or nil at the end of the body.
        mutating func nextGame() throws -> GameRecord? {
            if buffered == 0 && unreadInFile == 0 { return nil }
            let length = Int(Self.uint32LE(try take(4)))
            let payload = try take(length)
            let crc = Self.uint32LE(try take(4))
            return try GameCorpusShardFormat.decodeCheckedPayload(payload, crc: crc)
        }

        /// Hash the rest of the body without decoding it and return the
        /// digest of every byte from offset 0 to the trailer.
        mutating func finishHashing() throws -> Data {
            precondition(!hashFinished, "SealedShardBodyCursor: finishHashing called twice")
            window = []
            windowStart = 0
            while unreadInFile > 0 {
                _ = try readChunk(atLeast: 1)
            }
            hashFinished = true
            return Data(hasher.finalize())
        }

        /// Consume exactly `count` bytes. Bytes the body does not have are
        /// `truncatedRecord`, as for the in-memory reader.
        private mutating func take(_ count: Int) throws -> Data {
            guard count >= 0, UInt64(count) <= UInt64(buffered) + unreadInFile else {
                throw GameCorpusError.truncatedRecord
            }
            if buffered < count {
                // Drop consumed bytes before growing the window, so it holds
                // at most one chunk plus one record.
                window.removeSubrange(0..<windowStart)
                windowStart = 0
                while buffered < count {
                    window.append(contentsOf: try readChunk(atLeast: count - buffered))
                }
            }
            let bytes = Data(window[windowStart..<(windowStart + count)])
            windowStart += count
            return bytes
        }

        /// Read the next chunk of the body (at least `minimum` bytes, at most
        /// what is left before the trailer) and hash it.
        private mutating func readChunk(atLeast minimum: Int) throws -> Data {
            let count = Int(min(UInt64(max(minimum, Self.readChunkByteCount)), unreadInFile))
            let read: Data?
            do {
                read = try handle.read(upToCount: count)
            } catch {
                throw ReadFailure(error: .ioFailed("read \(url.lastPathComponent): \(error.localizedDescription)"))
            }
            guard let data = read, data.count == count else {
                throw ReadFailure(error: .ioFailed(
                    "read \(url.lastPathComponent): \(read?.count ?? 0) of \(count) bytes at offset \(fileOffset)"))
            }
            hasher.update(data: data)
            fileOffset += UInt64(count)
            return data
        }

        private static func uint32LE(_ bytes: Data) -> UInt32 {
            var value: UInt32 = 0
            for (shift, byte) in bytes.enumerated() { value |= UInt32(byte) << (8 * shift) }
            return value
        }
    }

    /// Seek to the fixed-size trailer of the sealed shard open on `handle`
    /// and return its bytes, after checking the file is large enough to hold
    /// a front header and a trailer. The one seek-and-read the cheap trailer
    /// readers share.
    fileprivate static func readTrailerBlock(handle: FileHandle, url: URL) throws -> Data {
        let trailerSize = GameCorpusShardFormat.trailerSize
        let size = try handle.seekToEnd()
        guard size >= UInt64(GameCorpusShardFormat.frontHeaderSize + trailerSize) else {
            throw GameCorpusError.truncatedHeader
        }
        try handle.seek(toOffset: size - UInt64(trailerSize))
        guard let block = try handle.read(upToCount: trailerSize), block.count == trailerSize else {
            throw GameCorpusError.truncatedHeader
        }
        return block
    }

    /// Close a handle a cheap read opened. Nothing was written through it,
    /// so a failed close loses nothing; it is logged, never thrown over the
    /// read's own result or error.
    fileprivate static func closeAfterRead(_ handle: FileHandle, url: URL, what: String) {
        do {
            try handle.close()
        } catch {
            SessionLogger.shared.log("[CORPUS] closing \(url.lastPathComponent) after its \(what) read failed: \(error.localizedDescription)")
        }
    }

    /// Cheap trailer read: `(gameCount, plyCount)` plus the SHA-256 the shard
    /// was sealed with (hex) — the shard's content identity, without reading
    /// or verifying the body. `readSealed` verifies the body against this
    /// same hash, so a resume that matches each shard's recorded hash here
    /// and then reads it through `readSealed` feeds exactly the recorded
    /// content.
    static func readSealedTrailer(at url: URL) throws -> (gameCount: Int, plyCount: Int, sealSHA256: String) {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) }
        catch { throw GameCorpusError.ioFailed("open \(url.lastPathComponent): \(error.localizedDescription)") }
        defer { closeAfterRead(handle, url: url, what: "trailer") }
        // The seal time is not part of the content identity.
        let trailer = try GameCorpusShardFormat.decodeTrailer(try readTrailerBlock(handle: handle, url: url))
        return (trailer.gameCount, trailer.plyCount, trailer.sha256.map { String(format: "%02x", $0) }.joined())
    }

    /// Cheap header+counts read: decode the fixed-size front header and trailer,
    /// without reading or SHA/CRC-verifying the body. Gives the per-shard
    /// `sourceID` (front header) alongside the sealed `(gameCount, plyCount)` —
    /// enough for `CorpusValidator` to rebuild per-source aggregate counts and
    /// check the header's corpus/source identity in its fast (non-integrity) mode
    /// without paying to read shard bodies.
    static func readSealedHeaderAndCounts(at url: URL) throws
        -> (header: GameCorpusShardFormat.FrontHeader, gameCount: Int, plyCount: Int) {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) }
        catch { throw GameCorpusError.ioFailed("open \(url.lastPathComponent): \(error.localizedDescription)") }
        defer { closeAfterRead(handle, url: url, what: "header and trailer") }

        let trailer = try GameCorpusShardFormat.decodeTrailer(try readTrailerBlock(handle: handle, url: url))
        let frontSize = GameCorpusShardFormat.frontHeaderSize
        try handle.seek(toOffset: 0)
        guard let fdata = try handle.read(upToCount: frontSize), fdata.count == frontSize else {
            throw GameCorpusError.truncatedHeader
        }
        let header = try GameCorpusShardFormat.decodeFrontHeader(fdata)
        return (header, trailer.gameCount, trailer.plyCount)
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
        return try scanOpenShard(contents: data)
    }

    /// `scanOpenShard(at:)` over bytes already read — by recovery, through
    /// the locked handle it will truncate.
    static func scanOpenShard(contents data: Data) throws -> OpenShardScan {
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
