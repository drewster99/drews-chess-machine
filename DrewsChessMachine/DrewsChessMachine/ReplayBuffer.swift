import CryptoKit
import Foundation

/// Thread-safe fixed-capacity ring of labeled self-play positions.
///
/// Self-play workers push whole games via `append(boards:policyIndices:outcome:count:)`
/// once each game ends and outcomes are known. The trainer pulls out
/// minibatches via `sample(count:intoBoards:moves:zs:vBaselines:)`.
/// Both sides run on background tasks; access is serialized by a
/// private serial `DispatchQueue` so the buffer is safe to share
/// across tasks.
///
/// **Storage layout.** Positions are stored in three flat contiguous
/// arrays sized to the full capacity at init — one big allocation per
/// field rather than one `[Float]` per position. This keeps allocator
/// pressure off the hot path (bulk-append is one write-through per
/// game, not one allocation per ply) and lets `sample(...)` copy
/// directly from contiguous source slots into trainer-owned staging
/// buffers.
///
/// Marked `@unchecked Sendable` because the serial queue serializes
/// all mutable state access.
final class ReplayBuffer: @unchecked Sendable {
    /// Number of floats required to hold one encoded board position
    /// (`inputPlanes` × 8 × 8 — currently 20 × 64 = 1280 with the v2
    /// architecture refresh that added two repetition planes).
    static let floatsPerBoard = ChessNetwork.inputPlanes
        * ChessNetwork.boardSize
        * ChessNetwork.boardSize

    /// Maximum number of positions held. Older positions are overwritten
    /// in FIFO order once the buffer is full.
    let capacity: Int

    private let queue = DispatchQueue(label: "drewschess.replaybuffer.serial")

    // MARK: - Ring storage

    /// Flat `[capacity * floatsPerBoard]` raw buffer of encoded boards.
    /// Allocated once at init and never re-sized; owned via a raw
    /// pointer to avoid any Swift Array CoW surprises when the trainer
    /// reads from it through `sample`.
    private let boardStorage: UnsafeMutablePointer<Float>

    /// Flat `[capacity]` move indices in the network's flipped
    /// coordinate system. Same ring index as `boardStorage`.
    private let moveStorage: UnsafeMutablePointer<Int32>

    /// Flat `[capacity]` outcome values (+1 / 0 / -1). Same ring index
    /// as `boardStorage`.
    private let outcomeStorage: UnsafeMutablePointer<Float>

    /// Flat `[capacity]` baseline-value scalars captured at the time
    /// the position was played — the inference network's `v(position)`
    /// output from the forward pass that was already run to pick the
    /// move. Used as the advantage baseline during training:
    /// `policy loss = mean((z − vBaseline) · −log p(a*))`. Detaches
    /// automatically because it enters the training graph through a
    /// placeholder rather than the value-head's live tensor. Same ring
    /// index as `boardStorage`.
    private let vBaselineStorage: UnsafeMutablePointer<Float>

    /// Number of positions currently held, capped at `capacity`.
    private var storedCount: Int = 0
    /// Next write slot in the ring.
    private var writeIndex: Int = 0

    // MARK: - Lifetime

    init(capacity: Int) {
        precondition(capacity > 0, "Replay buffer capacity must be positive")
        self.capacity = capacity

        let boardSlots = capacity * Self.floatsPerBoard
        let boards = UnsafeMutablePointer<Float>.allocate(capacity: boardSlots)
        boards.initialize(repeating: 0, count: boardSlots)
        self.boardStorage = boards

        let moves = UnsafeMutablePointer<Int32>.allocate(capacity: capacity)
        moves.initialize(repeating: 0, count: capacity)
        self.moveStorage = moves

        let outcomes = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        outcomes.initialize(repeating: 0, count: capacity)
        self.outcomeStorage = outcomes

        let vBaselines = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        vBaselines.initialize(repeating: 0, count: capacity)
        self.vBaselineStorage = vBaselines
    }

    deinit {
        let boardSlots = capacity * Self.floatsPerBoard
        boardStorage.deinitialize(count: boardSlots)
        boardStorage.deallocate()

        moveStorage.deinitialize(count: capacity)
        moveStorage.deallocate()

        outcomeStorage.deinitialize(count: capacity)
        outcomeStorage.deallocate()

        vBaselineStorage.deinitialize(count: capacity)
        vBaselineStorage.deallocate()
    }

    // MARK: - Introspection

    /// Current number of positions stored (up to `capacity`).
    var count: Int {
        queue.sync { storedCount }
    }

    /// Monotonically increasing count of all positions ever appended
    /// (not capped at `capacity` — includes positions that have since
    /// been overwritten by the ring). Read by the replay-ratio
    /// controller to compute the 1-minute self-play production rate
    /// without any coupling between self-play workers and the
    /// training worker.
    private var _totalPositionsAdded: Int = 0
    var totalPositionsAdded: Int {
        queue.sync { _totalPositionsAdded }
    }

    /// Per-position storage cost in bytes: board floats + move int32 +
    /// outcome float + vBaseline float. Used by the UI to estimate
    /// buffer RAM usage.
    static let bytesPerPosition: Int = floatsPerBoard * MemoryLayout<Float>.size
        + MemoryLayout<Int32>.size
        + MemoryLayout<Float>.size
        + MemoryLayout<Float>.size

    /// Atomic snapshot of the four persistence-relevant counters.
    /// Read under the lock so the values are mutually consistent
    /// (unlike reading `count` and `totalPositionsAdded` separately).
    struct StateSnapshot: Sendable {
        let storedCount: Int
        let capacity: Int
        let writeIndex: Int
        let totalPositionsAdded: Int
    }

    /// Thread-safe snapshot of the buffer's persistence-relevant
    /// counters. Used by the session-checkpoint path to populate
    /// `hasReplayBuffer*` fields in `SessionCheckpointState`.
    func stateSnapshot() -> StateSnapshot {
        queue.sync {
            StateSnapshot(
                storedCount: storedCount,
                capacity: capacity,
                writeIndex: writeIndex,
                totalPositionsAdded: _totalPositionsAdded
            )
        }
    }

    // MARK: - Append

    /// Append one finished game's positions in bulk. The caller passes
    /// contiguous buffers of `count` board tensors (`count * floatsPerBoard`
    /// floats) and `count` policy indices, plus a single outcome to
    /// broadcast across every row. When the ring is full, new rows
    /// overwrite the oldest in FIFO order with wraparound handled via
    /// two `memcpy` calls at the seam.
    ///
    /// The caller owns the input buffers — they are not retained past
    /// the call, so this method synchronously copies the input bytes
    /// into the ring storage on the serial queue before returning.
    func append(
        boards: UnsafePointer<Float>,
        policyIndices: UnsafePointer<Int32>,
        vBaselines: UnsafePointer<Float>,
        outcome: Float,
        count positionCount: Int
    ) {
        guard positionCount > 0 else { return }
        precondition(positionCount <= capacity,
            "ReplayBuffer.append: positionCount (\(positionCount)) exceeds capacity (\(capacity))")

        queue.sync {
            let floatsPerBoard = Self.floatsPerBoard

            // The incoming positions may straddle the ring's wraparound
            // point. Split the write into at most two contiguous runs:
            // the tail of the ring (from writeIndex to capacity) and then
            // the head (wrapping back to index 0).
            var remaining = positionCount
            var srcOffset = 0  // in positions
            while remaining > 0 {
                let tailSlots = capacity - writeIndex
                let chunk = min(remaining, tailSlots)

                // Boards: chunk * floatsPerBoard floats.
                (boardStorage + writeIndex * floatsPerBoard).update(
                    from: boards + srcOffset * floatsPerBoard,
                    count: chunk * floatsPerBoard
                )
                // Moves: chunk int32s.
                (moveStorage + writeIndex).update(
                    from: policyIndices + srcOffset,
                    count: chunk
                )
                // Outcomes: broadcast — no source buffer, just fill.
                (outcomeStorage + writeIndex).update(
                    repeating: outcome,
                    count: chunk
                )
                // vBaselines: one float per position — the inference-time
                // v(position) captured during move selection. Detaches
                // automatically at train time because it re-enters the
                // graph via a placeholder feed.
                (vBaselineStorage + writeIndex).update(
                    from: vBaselines + srcOffset,
                    count: chunk
                )

                let newWrite = writeIndex + chunk
                writeIndex = newWrite == capacity ? 0 : newWrite
                srcOffset += chunk
                remaining -= chunk
                if storedCount < capacity {
                    storedCount = min(capacity, storedCount + chunk)
                }
                _totalPositionsAdded += chunk
            }
        }
    }

    // MARK: - Sample

    /// Draw `sampleCount` positions uniformly at random (with
    /// replacement) from the positions currently held, writing them
    /// into caller-provided contiguous output buffers. Returns `false`
    /// if the buffer holds fewer than `sampleCount` positions — the
    /// caller should wait for more self-play to land before retrying.
    func sample(
        count sampleCount: Int,
        intoBoards dstBoards: UnsafeMutablePointer<Float>,
        moves dstMoves: UnsafeMutablePointer<Int32>,
        zs dstZs: UnsafeMutablePointer<Float>,
        vBaselines dstVBase: UnsafeMutablePointer<Float>
    ) -> Bool {
        precondition(sampleCount > 0, "Sample count must be positive")
        return queue.sync {
            let held = storedCount
            guard held >= sampleCount else { return false }

            let floatsPerBoard = Self.floatsPerBoard

            for i in 0..<sampleCount {
                let srcIndex = Int.random(in: 0..<held)
                (dstBoards + i * floatsPerBoard).update(
                    from: boardStorage + srcIndex * floatsPerBoard,
                    count: floatsPerBoard
                )
                dstMoves[i] = moveStorage[srcIndex]
                dstZs[i] = outcomeStorage[srcIndex]
                dstVBase[i] = vBaselineStorage[srcIndex]
            }

            return true
        }
    }

    // MARK: - Persistence

    /// Errors thrown by `write(to:)` / `restore(from:)`.
    enum PersistenceError: LocalizedError {
        case badMagic
        case truncatedHeader
        case unsupportedVersion(UInt32)
        case incompatibleBoardSize(expected: Int, got: Int)
        case invalidCounts(capacity: Int, stored: Int, writeIndex: Int)
        case truncatedBody(expected: Int, got: Int)
        case sizeMismatch(expected: Int64, got: Int64)
        case hashMismatch
        case upperBoundExceeded(field: String, value: Int64, max: Int64)
        case writeFailed(Error)
        case readFailed(Error)

        var errorDescription: String? {
            switch self {
            case .badMagic: return "Replay buffer file header magic mismatch"
            case .truncatedHeader: return "Replay buffer file header truncated"
            case .unsupportedVersion(let v): return "Unsupported replay buffer format version \(v)"
            case .incompatibleBoardSize(let exp, let got):
                return "Replay buffer board size mismatch (expected \(exp) floats, file has \(got))"
            case .invalidCounts(let cap, let stored, let wi):
                return "Invalid replay buffer counts (capacity=\(cap) stored=\(stored) writeIndex=\(wi))"
            case .truncatedBody(let exp, let got):
                return "Replay buffer body truncated (expected \(exp) bytes, got \(got))"
            case .sizeMismatch(let exp, let got):
                return "Replay buffer file size mismatch: header predicts \(exp) bytes, file is \(got) bytes"
            case .hashMismatch:
                return "Replay buffer integrity check failed: SHA-256 trailer does not match file contents"
            case .upperBoundExceeded(let field, let value, let max):
                return "Replay buffer header field '\(field)' value \(value) exceeds sanity cap \(max) — file is malformed or corrupted"
            case .writeFailed(let err): return "Replay buffer write failed: \(err)"
            case .readFailed(let err): return "Replay buffer read failed: \(err)"
            }
        }
    }

    /// Binary file magic — 8 ASCII bytes.
    private static let fileMagic: [UInt8] = Array("DCMRPBUF".utf8)
    /// Format version. Bump on any on-disk layout change.
    ///
    /// Current format is v4:
    ///   - Header: 8-byte magic + 4-byte version + 4-byte pad + 5 × Int64
    ///     (floatsPerBoard, capacity, storedCount, writeIndex,
    ///     totalPositionsAdded).
    ///   - Body: four parallel arrays of `storedCount` entries, oldest-first
    ///     — boards (`floatsPerBoard` × Float32), moves (Int32), outcomes
    ///     (Float32), vBaselines (Float32).
    ///   - Trailer: 32-byte SHA-256 digest over every preceding byte
    ///     (header + all four body arrays). Verified before any header
    ///     field is trusted at load time.
    ///
    /// The v4 file-size invariant (checked strictly at load) is:
    /// ```
    /// totalBytes == headerSize + storedCount × (floatsPerBoard × 4 + 12) + 32
    /// ```
    ///
    /// Earlier format versions (v1 without vBaselines, v2 with the
    /// pre-refresh 18-plane board stride, v3 without SHA trailer) are no
    /// longer supported — the reader cleanly rejects them with
    /// `unsupportedVersion`. Per project convention (no migration without
    /// explicit request), older files from previous architecture or
    /// durability iterations are not loadable.
    private static let fileVersion: UInt32 = 4
    /// Header size in bytes: 8 magic + 4 version + 4 pad + 5 × Int64 fields.
    private static let headerSize: Int = 8 + 4 + 4 + 8 * 5
    /// SHA-256 trailer size in bytes.
    private static let trailerSize: Int = 32
    /// Chunk size for raw-buffer writes/reads. Keeps peak Data
    /// allocations bounded even when the ring holds ~1 M positions.
    private static let persistenceChunkBytes: Int = 32 * 1024 * 1024

    /// Sanity caps on header counter fields. Applied before any
    /// allocation or seek arithmetic during load so a corrupted or
    /// hostile header cannot coax the decoder into a massive allocation
    /// or integer overflow. Paired with the SHA-256 trailer (which
    /// catches corruption pre-parse) this is defense-in-depth.
    private static let maxReasonableCapacity: Int64 = 10_000_000
    private static let maxReasonableStoredCount: Int64 = 10_000_000
    private static let maxReasonableFloatsPerBoard: Int64 = 8_192

    /// Write the buffer's current contents to `url` in oldest-first
    /// order. On-disk size is proportional to `storedCount` (not
    /// `capacity`), so partially-filled rings serialize to a smaller
    /// file. Thread-safe — runs on the serial queue for the duration
    /// of the write, which pauses appends and samples until the write
    /// finishes.
    ///
    /// Returns the `StateSnapshot` that was actually serialized.
    /// Post-save verification code that wants to compare the written
    /// file's counters against ground truth must use this return
    /// value, NOT call `stateSnapshot()` separately — concurrent
    /// appends between the write and the follow-up snapshot would
    /// make the two observations diverge and the comparison
    /// spuriously fail. Annotated `@discardableResult` so callers
    /// that just want "save and move on" semantics (tests,
    /// fire-and-forget saves) compile unchanged.
    @discardableResult
    func write(to url: URL) throws -> StateSnapshot {
        try queue.sync {
            try _writeLocked(to: url)
            // Captured under the same lock that serializes the write,
            // so the returned snapshot reflects exactly the state
            // whose bytes just landed in the file.
            return StateSnapshot(
                storedCount: storedCount,
                capacity: capacity,
                writeIndex: writeIndex,
                totalPositionsAdded: _totalPositionsAdded
            )
        }
    }

    private func _writeLocked(to url: URL) throws {
        let stored = storedCount
        let cap = capacity
        let floatsPerBoard = Self.floatsPerBoard
        let wIndex = writeIndex
        let totalAdded = _totalPositionsAdded

        // Header
        var header = Data()
        header.reserveCapacity(Self.headerSize)
        header.append(contentsOf: Self.fileMagic)
        var version = Self.fileVersion
        withUnsafeBytes(of: &version) { header.append(contentsOf: $0) }
        var pad: UInt32 = 0
        withUnsafeBytes(of: &pad) { header.append(contentsOf: $0) }
        var fpb64 = Int64(floatsPerBoard)
        withUnsafeBytes(of: &fpb64) { header.append(contentsOf: $0) }
        var cap64 = Int64(cap)
        withUnsafeBytes(of: &cap64) { header.append(contentsOf: $0) }
        var stc64 = Int64(stored)
        withUnsafeBytes(of: &stc64) { header.append(contentsOf: $0) }
        var wi64 = Int64(wIndex)
        withUnsafeBytes(of: &wi64) { header.append(contentsOf: $0) }
        var ttl64 = Int64(totalAdded)
        withUnsafeBytes(of: &ttl64) { header.append(contentsOf: $0) }

        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            do {
                try fm.removeItem(at: url)
            } catch {
                // Surface the original removal failure directly —
                // hiding it behind `try?` would let the subsequent
                // createFile / FileHandle init throw a secondary
                // "file busy" or "permission denied" error that
                // obscures the root cause.
                throw PersistenceError.writeFailed(error)
            }
        }
        fm.createFile(atPath: url.path, contents: nil)
        let handle: FileHandle
        do {
            handle = try FileHandle(forWritingTo: url)
        } catch {
            throw PersistenceError.writeFailed(error)
        }
        // `try?` on FileHandle.close() in a `defer` is idiomatic:
        // by the time this fires we've either completed the write
        // successfully (in which case a close-time error doesn't
        // invalidate the file we already flushed) or we've already
        // thrown a more meaningful error that we want to propagate
        // to the caller — overwriting it with the close error would
        // mask the real failure.
        defer { try? handle.close() }

        // Streaming SHA-256 hasher. Every byte written to the file
        // (header + all body sections) is fed through this hasher; the
        // finalized 32-byte digest is appended as the trailer. The
        // trailer is itself NOT hashed (same convention as .dcmmodel).
        var hasher = SHA256()
        hasher.update(data: header)

        do {
            try handle.write(contentsOf: header)
        } catch {
            throw PersistenceError.writeFailed(error)
        }

        if stored > 0 {
            // Start position of the oldest stored entry in the ring.
            let startIndex = (stored == cap) ? wIndex : 0

            // Boards — stride in positions, copy in chunks of up to
            // persistenceChunkBytes to bound peak memory on 1 M-position rings.
            let boardStride = floatsPerBoard * MemoryLayout<Float>.size
            let boardChunkPositions = max(1, Self.persistenceChunkBytes / boardStride)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: boardChunkPositions,
                elementBytes: boardStride,
                basePtr: UnsafeRawPointer(boardStorage),
                elementsPerSlot: floatsPerBoard,
                slotSize: boardStride
            )

            // Moves — 4 bytes per slot.
            let moveChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<Int32>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: moveChunk,
                elementBytes: MemoryLayout<Int32>.size,
                basePtr: UnsafeRawPointer(moveStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<Int32>.size
            )

            // Outcomes — 4 bytes per slot.
            let outChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<Float>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: outChunk,
                elementBytes: MemoryLayout<Float>.size,
                basePtr: UnsafeRawPointer(outcomeStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<Float>.size
            )

            // vBaselines — 4 bytes per slot. Present since v2.
            let vBaseChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<Float>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: vBaseChunk,
                elementBytes: MemoryLayout<Float>.size,
                basePtr: UnsafeRawPointer(vBaselineStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<Float>.size
            )
        }

        // Trailer — 32 bytes of SHA-256 over all preceding bytes.
        let digest = Data(hasher.finalize())
        do {
            try handle.write(contentsOf: digest)
        } catch {
            throw PersistenceError.writeFailed(error)
        }

        // Force APFS to flush dirty pages to stable storage before the
        // handle closes. Without this, a crash or power loss after
        // close-returns but before the OS flushes would leave a torn
        // file on disk even though Swift saw the write as successful.
        // Regular `synchronize()` (equivalent to fsync(2)) commits the
        // bytes to the device; for drive-cache-bypass durability,
        // `CheckpointManager.fullSyncPath` uses fcntl(F_FULLFSYNC) on
        // the file after we return.
        do {
            try handle.synchronize()
        } catch {
            throw PersistenceError.writeFailed(error)
        }
    }

    /// Serialize a contiguous logical range of the ring starting at
    /// `start` with length `total`, handling wraparound. The array
    /// is identified by `basePtr` (raw pointer to its element 0) and
    /// `slotSize` bytes per ring slot. Caller must be executing on
    /// the serial `queue` (e.g. from inside `_writeLocked`).
    ///
    /// Every byte written is also fed into `hasher` — the streaming
    /// SHA-256 hasher whose final digest becomes the file's integrity
    /// trailer. Passing the hasher inout (rather than capturing it in
    /// an escaping closure) lets the single hasher object accumulate
    /// across all four section writes from a single `_writeLocked`
    /// call.
    private func writeRange(
        handle: FileHandle,
        hasher: inout SHA256,
        start: Int,
        total: Int,
        capacity: Int,
        chunkPositions: Int,
        elementBytes: Int,
        basePtr: UnsafeRawPointer,
        elementsPerSlot: Int,
        slotSize: Int
    ) throws {
        var remaining = total
        var idx = start
        while remaining > 0 {
            let tailSlots = capacity - idx
            let run = min(remaining, tailSlots)
            var runRemaining = run
            var runIdx = idx
            while runRemaining > 0 {
                let take = min(runRemaining, chunkPositions)
                let byteCount = take * slotSize
                let srcPtr = basePtr.advanced(by: runIdx * slotSize)
                let chunk = Data(bytes: srcPtr, count: byteCount)
                hasher.update(data: chunk)
                do {
                    try handle.write(contentsOf: chunk)
                } catch {
                    throw PersistenceError.writeFailed(error)
                }
                runIdx += take
                runRemaining -= take
            }
            let newIdx = idx + run
            idx = (newIdx == capacity) ? 0 : newIdx
            remaining -= run
        }
    }

    /// Populate this buffer from `url`, replacing any existing
    /// contents. If the file's capacity exceeds this buffer's
    /// capacity, the oldest entries in the file are discarded so
    /// only the newest `capacity` positions are retained. If the
    /// file's capacity is smaller, all file entries are restored
    /// and `writeIndex` continues from the loaded count. The
    /// `totalPositionsAdded` counter is restored verbatim so the
    /// replay-ratio controller's production-rate window stays
    /// continuous across save/resume.
    ///
    /// v4 validation order (each step throws a specific error on
    /// failure and aborts; no field is trusted until all preceding
    /// checks pass):
    ///
    /// 1. File opens and header can be fully read (`truncatedHeader`).
    /// 2. Magic matches "DCMRPBUF" (`badMagic`).
    /// 3. `fileVersion == 4` (`unsupportedVersion`).
    /// 4. `floatsPerBoard` matches the running build's tensor length
    ///    (`incompatibleBoardSize`) — replay-buffer analog of the
    ///    `.dcmmodel` arch-hash check.
    /// 5. Counter upper-bound caps: `capacity`, `storedCount`,
    ///    `floatsPerBoard` each ≤ their `maxReasonable*` threshold
    ///    (`upperBoundExceeded`). Catches corrupt headers before
    ///    any allocation or seek arithmetic.
    /// 6. Counter relationships: non-negative, `storedCount ≤ capacity`,
    ///    `writeIndex` in range (`invalidCounts`).
    /// 7. Actual file size == header-predicted size (`sizeMismatch`).
    ///    Uses strict equality; any deviation is corruption.
    /// 8. SHA-256 over the first `totalBytes - 32` bytes matches the
    ///    32-byte trailer (`hashMismatch`). Full read of the file's
    ///    content bytes before any state is mutated.
    ///
    /// Only after all eight pass does the function mutate any live
    /// state (locking `queue.sync`, resetting counters, re-seeking
    /// to the header end, and reading the four sections into the
    /// ring storage).
    func restore(from url: URL) throws {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw PersistenceError.readFailed(error)
        }
        // `try?` on FileHandle.close() in a `defer` is idiomatic:
        // we've either finished the read successfully (close errors
        // don't invalidate already-consumed data) or thrown with a
        // more meaningful error that we want to propagate — masking
        // it with a close error would obscure the real failure.
        defer { try? handle.close() }

        let headerData: Data
        do {
            guard let hd = try handle.read(upToCount: Self.headerSize),
                  hd.count == Self.headerSize else {
                throw PersistenceError.truncatedHeader
            }
            headerData = hd
        } catch let err as PersistenceError {
            throw err
        } catch {
            throw PersistenceError.readFailed(error)
        }

        let magicMatches = headerData.prefix(8).elementsEqual(Self.fileMagic)
        guard magicMatches else { throw PersistenceError.badMagic }

        let version: UInt32 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
        }
        // v4 is the only currently supported format. Earlier versions
        // (v1 without vBaselines, v2 with the pre-refresh 18-plane
        // board stride, v3 without the SHA trailer) are rejected with
        // `unsupportedVersion`. No migration code per project
        // convention.
        guard version == Self.fileVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }
        let fpbFile: Int64 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 16, as: Int64.self)
        }
        let capFile: Int64 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 24, as: Int64.self)
        }
        let stcFile: Int64 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 32, as: Int64.self)
        }
        let wiFile: Int64 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 40, as: Int64.self)
        }
        let ttlFile: Int64 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 48, as: Int64.self)
        }

        guard Int(fpbFile) == Self.floatsPerBoard else {
            throw PersistenceError.incompatibleBoardSize(
                expected: Self.floatsPerBoard,
                got: Int(fpbFile)
            )
        }

        // Upper-bound caps before any further arithmetic. A corrupted
        // header with `Int64.max` in any field would otherwise survive
        // the non-negative sanity checks below and could either drive
        // a huge allocation or overflow the size computation.
        guard fpbFile <= Self.maxReasonableFloatsPerBoard else {
            throw PersistenceError.upperBoundExceeded(
                field: "floatsPerBoard",
                value: fpbFile,
                max: Self.maxReasonableFloatsPerBoard
            )
        }
        guard capFile <= Self.maxReasonableCapacity else {
            throw PersistenceError.upperBoundExceeded(
                field: "capacity",
                value: capFile,
                max: Self.maxReasonableCapacity
            )
        }
        guard stcFile <= Self.maxReasonableStoredCount else {
            throw PersistenceError.upperBoundExceeded(
                field: "storedCount",
                value: stcFile,
                max: Self.maxReasonableStoredCount
            )
        }

        guard capFile >= 0, stcFile >= 0, stcFile <= capFile,
              wiFile >= 0, wiFile < max(1, capFile) else {
            throw PersistenceError.invalidCounts(
                capacity: Int(capFile),
                stored: Int(stcFile),
                writeIndex: Int(wiFile)
            )
        }

        // Compute expected file size from the header, then require
        // strict byte-for-byte equality against what's on disk. The
        // format is fully deterministic — any deviation is corruption.
        let perSlotBytes: Int64 = fpbFile * Int64(MemoryLayout<Float>.size)
            + Int64(MemoryLayout<Int32>.size)       // moves
            + Int64(MemoryLayout<Float>.size)       // outcomes
            + Int64(MemoryLayout<Float>.size)       // vBaselines
        let expectedBytes: Int64 = Int64(Self.headerSize)
            + stcFile * perSlotBytes
            + Int64(Self.trailerSize)

        let actualBytes: Int64
        do {
            let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
            guard let size = attrs[.size] as? NSNumber else {
                throw PersistenceError.readFailed(
                    NSError(
                        domain: "ReplayBuffer",
                        code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "FileManager returned no size attribute"]
                    )
                )
            }
            actualBytes = size.int64Value
        } catch let err as PersistenceError {
            throw err
        } catch {
            throw PersistenceError.readFailed(error)
        }

        guard actualBytes == expectedBytes else {
            throw PersistenceError.sizeMismatch(
                expected: expectedBytes,
                got: actualBytes
            )
        }

        // SHA-256 verification. Stream-read every byte before the
        // trailer through a fresh hasher, then compare the finalized
        // digest against the last 32 bytes. On match, seek back to
        // the header end so the section reads below can proceed from
        // the correct offset.
        do {
            try handle.seek(toOffset: 0)
            var hasher = SHA256()
            var remaining = actualBytes - Int64(Self.trailerSize)
            while remaining > 0 {
                let take = Int(min(Int64(Self.persistenceChunkBytes), remaining))
                guard let chunk = try handle.read(upToCount: take),
                      chunk.count == take else {
                    throw PersistenceError.readFailed(
                        NSError(
                            domain: "ReplayBuffer",
                            code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "Short read during SHA verify"]
                        )
                    )
                }
                hasher.update(data: chunk)
                remaining -= Int64(take)
            }
            let computed = Data(hasher.finalize())
            guard let storedTrailer = try handle.read(upToCount: Self.trailerSize),
                  storedTrailer.count == Self.trailerSize else {
                throw PersistenceError.readFailed(
                    NSError(
                        domain: "ReplayBuffer",
                        code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "Short read on SHA trailer"]
                    )
                )
            }
            guard computed == storedTrailer else {
                throw PersistenceError.hashMismatch
            }
            // Reposition the handle at the start of the body (just past
            // the 56-byte header) so the section-reads below start at
            // the right offset.
            try handle.seek(toOffset: UInt64(Self.headerSize))
        } catch let err as PersistenceError {
            throw err
        } catch {
            throw PersistenceError.readFailed(error)
        }

        let fileStored = Int(stcFile)
        let target = min(fileStored, capacity)
        let skip = fileStored - target  // oldest-first file entries to discard

        try queue.sync {
            // Reset live state before filling.
            storedCount = 0
            writeIndex = 0
            _totalPositionsAdded = 0

            if fileStored == 0 {
                _totalPositionsAdded = Int(ttlFile)
                return
            }

            let floatsPerBoard = Self.floatsPerBoard
            let boardSlotBytes = floatsPerBoard * MemoryLayout<Float>.size

            // Skip the `skip` oldest board records if capacity shrank.
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * boardSlotBytes)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(boardStorage),
                slotBytes: boardSlotBytes,
                count: target
            )

            // Skip remaining board bytes if we truncated (there's no more
            // board data past the last target slot in the file). Then
            // seek past the skipped-move prefix.
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<Int32>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(moveStorage),
                slotBytes: MemoryLayout<Int32>.size,
                count: target
            )

            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<Float>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(outcomeStorage),
                slotBytes: MemoryLayout<Float>.size,
                count: target
            )

            // vBaselines — present since v2. Skip + read with the same
            // pattern as the other fields.
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<Float>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(vBaselineStorage),
                slotBytes: MemoryLayout<Float>.size,
                count: target
            )

            storedCount = target
            writeIndex = (target == capacity) ? 0 : target
            _totalPositionsAdded = Int(ttlFile)
        }
    }

    private func seekForward(handle: FileHandle, bytes: Int) throws {
        guard bytes > 0 else { return }
        do {
            let current = try handle.offset()
            try handle.seek(toOffset: current + UInt64(bytes))
        } catch {
            throw PersistenceError.readFailed(error)
        }
    }

    private func readContiguous(
        handle: FileHandle,
        into basePtr: UnsafeMutableRawPointer,
        slotBytes: Int,
        count: Int
    ) throws {
        guard count > 0 else { return }
        let chunkSlots = max(1, Self.persistenceChunkBytes / slotBytes)
        var remaining = count
        var offset = 0
        while remaining > 0 {
            let take = min(remaining, chunkSlots)
            let byteCount = take * slotBytes
            let data: Data
            do {
                guard let chunk = try handle.read(upToCount: byteCount),
                      chunk.count == byteCount else {
                    throw PersistenceError.truncatedBody(
                        expected: byteCount,
                        got: 0
                    )
                }
                data = chunk
            } catch let err as PersistenceError {
                throw err
            } catch {
                throw PersistenceError.readFailed(error)
            }
            let dst = basePtr.advanced(by: offset * slotBytes)
            data.withUnsafeBytes { src in
                if let srcBase = src.baseAddress {
                    dst.copyMemory(from: srcBase, byteCount: byteCount)
                }
            }
            offset += take
            remaining -= take
        }
    }
}
