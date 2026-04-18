import Foundation

/// Thread-safe fixed-capacity ring of labeled self-play positions.
///
/// Self-play workers push whole games via `append(boards:policyIndices:outcome:count:)`
/// once each game ends and outcomes are known. The trainer pulls out
/// minibatches via `sample(count:intoBoards:moves:zs:vBaselines:)`.
/// Both sides run on background
/// tasks; access is serialized by an `NSLock` so the buffer is safe
/// to share across tasks.
///
/// **Storage layout.** Positions are stored in three flat contiguous
/// arrays sized to the full capacity at init — one big allocation per
/// field rather than one `[Float]` per position. This keeps allocator
/// pressure off the hot path (bulk-append is one write-through per
/// game, not one allocation per ply) and lets `sample(...)` copy
/// directly from contiguous source slots into trainer-owned staging
/// buffers.
///
/// Marked `@unchecked Sendable` because the `NSLock` serializes all
/// mutable state access.
final class ReplayBuffer: @unchecked Sendable {
    /// Number of floats required to hold one encoded board position
    /// (18 planes × 8 × 8).
    static let floatsPerBoard = ChessNetwork.inputPlanes
        * ChessNetwork.boardSize
        * ChessNetwork.boardSize

    /// Maximum number of positions held. Older positions are overwritten
    /// in FIFO order once the buffer is full.
    let capacity: Int

    private let lock = NSLock()

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
        lock.lock()
        defer { lock.unlock() }
        return storedCount
    }

    /// Monotonically increasing count of all positions ever appended
    /// (not capped at `capacity` — includes positions that have since
    /// been overwritten by the ring). Read by the replay-ratio
    /// controller to compute the 1-minute self-play production rate
    /// without any coupling between self-play workers and the
    /// training worker.
    private var _totalPositionsAdded: Int = 0
    var totalPositionsAdded: Int {
        lock.lock()
        defer { lock.unlock() }
        return _totalPositionsAdded
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
        lock.lock()
        defer { lock.unlock() }
        return StateSnapshot(
            storedCount: storedCount,
            capacity: capacity,
            writeIndex: writeIndex,
            totalPositionsAdded: _totalPositionsAdded
        )
    }

    // MARK: - Append

    /// Append one finished game's positions in bulk. The caller passes
    /// contiguous buffers of `count` board tensors (`count * 1152`
    /// floats) and `count` policy indices, plus a single outcome to
    /// broadcast across every row. When the ring is full, new rows
    /// overwrite the oldest in FIFO order with wraparound handled via
    /// two `memcpy` calls at the seam.
    ///
    /// The caller owns the input buffers — they are not retained past
    /// the call. This method acquires the lock once for the whole game
    /// rather than per-position.
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

        lock.lock()
        defer { lock.unlock() }

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
        lock.lock()
        defer { lock.unlock() }
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

    // MARK: - Persistence

    /// Errors thrown by `write(to:)` / `restore(from:)`.
    enum PersistenceError: LocalizedError {
        case badMagic
        case truncatedHeader
        case unsupportedVersion(UInt32)
        case incompatibleBoardSize(expected: Int, got: Int)
        case invalidCounts(capacity: Int, stored: Int, writeIndex: Int)
        case truncatedBody(expected: Int, got: Int)
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
            case .writeFailed(let err): return "Replay buffer write failed: \(err)"
            case .readFailed(let err): return "Replay buffer read failed: \(err)"
            }
        }
    }

    /// Binary file magic — 8 ASCII bytes.
    private static let fileMagic: [UInt8] = Array("DCMRPBUF".utf8)
    /// Format version. Bump on any on-disk layout change.
    ///
    /// - v1: boards, moves, outcomes
    /// - v2: boards, moves, outcomes, vBaselines (advantage baseline)
    ///
    /// Current writer always writes v2. Reader accepts both; v1 files
    /// load with vBaselines zeroed out, which degrades gracefully to
    /// the pre-advantage formulation (z − 0 = z).
    private static let fileVersion: UInt32 = 2
    /// Header size in bytes: 8 magic + 4 version + 4 pad + 5 × Int64 fields.
    private static let headerSize: Int = 8 + 4 + 4 + 8 * 5
    /// Chunk size for raw-buffer writes/reads. Keeps peak Data
    /// allocations bounded even when the ring holds ~1 M positions.
    private static let persistenceChunkBytes: Int = 32 * 1024 * 1024

    /// Write the buffer's current contents to `url` in oldest-first
    /// order. On-disk size is proportional to `storedCount` (not
    /// `capacity`), so partially-filled rings serialize to a smaller
    /// file. Thread-safe — acquires the lock for the duration of the
    /// write, which pauses appends but not samples in progress
    /// (sample already holds its own critical section).
    func write(to url: URL) throws {
        lock.lock()
        defer { lock.unlock() }

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

        do {
            try handle.write(contentsOf: header)
        } catch {
            throw PersistenceError.writeFailed(error)
        }

        guard stored > 0 else { return }

        // Start position of the oldest stored entry in the ring.
        let startIndex = (stored == cap) ? wIndex : 0

        // Boards — stride in positions, copy in chunks of up to
        // persistenceChunkBytes to bound peak memory on 1 M-position rings.
        let boardStride = floatsPerBoard * MemoryLayout<Float>.size
        let boardChunkPositions = max(1, Self.persistenceChunkBytes / boardStride)
        try writeRange(
            handle: handle,
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
            start: startIndex,
            total: stored,
            capacity: cap,
            chunkPositions: outChunk,
            elementBytes: MemoryLayout<Float>.size,
            basePtr: UnsafeRawPointer(outcomeStorage),
            elementsPerSlot: 1,
            slotSize: MemoryLayout<Float>.size
        )

        // vBaselines — 4 bytes per slot. Present in v2 and later.
        let vBaseChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<Float>.size)
        try writeRange(
            handle: handle,
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

    /// Serialize a contiguous logical range of the ring starting at
    /// `start` with length `total`, handling wraparound. The array
    /// is identified by `basePtr` (raw pointer to its element 0) and
    /// `slotSize` bytes per ring slot. Caller must hold `lock`.
    private func writeRange(
        handle: FileHandle,
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
                do {
                    try handle.write(
                        contentsOf: Data(bytes: srcPtr, count: byteCount)
                    )
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
        // Accept both v1 (pre-advantage, no vBaselines region) and v2
        // (with vBaselines). Reader fills vBaselines with zeros on v1.
        guard version == 1 || version == 2 else {
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
        guard capFile >= 0, stcFile >= 0, stcFile <= capFile,
              wiFile >= 0, wiFile < max(1, capFile) else {
            throw PersistenceError.invalidCounts(
                capacity: Int(capFile),
                stored: Int(stcFile),
                writeIndex: Int(wiFile)
            )
        }

        let fileStored = Int(stcFile)
        let target = min(fileStored, capacity)
        let skip = fileStored - target  // oldest-first file entries to discard

        lock.lock()
        defer { lock.unlock() }

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

        // vBaselines — only present in v2+. On v1 files, leave the
        // whole region zero (pre-filled during the reset above via
        // storedCount=0 → next append will overwrite, but defensive
        // zero-init for sampling against a partially-restored buffer).
        if version >= 2 {
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<Float>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(vBaselineStorage),
                slotBytes: MemoryLayout<Float>.size,
                count: target
            )
        } else {
            // v1 file: zero out the vBaselines region for the slots
            // we just restored so sampling returns a clean 0 baseline.
            (vBaselineStorage).update(repeating: 0, count: target)
        }

        storedCount = target
        writeIndex = (target == capacity) ? 0 : target
        _totalPositionsAdded = Int(ttlFile)
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
