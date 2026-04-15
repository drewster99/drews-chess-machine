import Foundation

/// Thread-safe fixed-capacity ring of labeled self-play positions.
///
/// Self-play workers push whole games via `append(boards:policyIndices:outcome:count:)`
/// once each game ends and outcomes are known. The trainer pulls out
/// minibatches via `sample(count:)`. Both sides run on background
/// tasks; access is serialized by an `NSLock` so the buffer is safe
/// to share across tasks.
///
/// **Storage layout.** Positions are stored in three flat contiguous
/// arrays sized to the full capacity at init — one big allocation per
/// field rather than one `[Float]` per position. This keeps allocator
/// pressure off the hot path (bulk-append is one write-through per
/// game, not one allocation per ply) and gives `sample()` a cache-
/// friendly copy from contiguous source slots into the pre-allocated
/// reusable batch buffers.
///
/// **Batch reuse.** The three per-sample output buffers
/// (`reusableBatchBoards` / `reusableBatchMoves` / `reusableBatchZs`)
/// are allocated lazily on the first call at the trainer's batch size
/// and reused across every subsequent sample. The returned
/// `TrainingBatch` is a non-owning view — its pointers are valid only
/// until the next `sample()` call on this buffer. The training worker
/// is strictly serial (sample → trainStep → repeat, with trainStep
/// synchronous), so there is never more than one batch in flight.
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

    /// Number of positions currently held, capped at `capacity`.
    private var storedCount: Int = 0
    /// Next write slot in the ring.
    private var writeIndex: Int = 0

    // MARK: - Reusable sample batch

    /// Current capacity of the reusable sample buffers (0 until first
    /// `sample()`). Grows to match the largest batch size ever requested.
    private var reusableBatchCapacity: Int = 0
    private var reusableBatchBoards: UnsafeMutablePointer<Float>?
    private var reusableBatchMoves: UnsafeMutablePointer<Int32>?
    private var reusableBatchZs: UnsafeMutablePointer<Float>?

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
    }

    deinit {
        let boardSlots = capacity * Self.floatsPerBoard
        boardStorage.deinitialize(count: boardSlots)
        boardStorage.deallocate()

        moveStorage.deinitialize(count: capacity)
        moveStorage.deallocate()

        outcomeStorage.deinitialize(count: capacity)
        outcomeStorage.deallocate()

        if let ptr = reusableBatchBoards {
            ptr.deinitialize(count: reusableBatchCapacity * Self.floatsPerBoard)
            ptr.deallocate()
        }
        if let ptr = reusableBatchMoves {
            ptr.deinitialize(count: reusableBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = reusableBatchZs {
            ptr.deinitialize(count: reusableBatchCapacity)
            ptr.deallocate()
        }
    }

    // MARK: - Introspection

    /// Current number of positions stored (up to `capacity`).
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return storedCount
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

            let newWrite = writeIndex + chunk
            writeIndex = newWrite == capacity ? 0 : newWrite
            srcOffset += chunk
            remaining -= chunk
            if storedCount < capacity {
                storedCount = min(capacity, storedCount + chunk)
            }
        }
    }

    // MARK: - Sample

    /// Draw `sampleCount` positions uniformly at random (with
    /// replacement) from the positions currently held, packing them
    /// into the buffer's reusable per-sample output arrays and
    /// returning a non-owning `TrainingBatch`. Returns `nil` if the
    /// buffer holds fewer than `sampleCount` positions — the caller
    /// should wait for more self-play to land before retrying.
    ///
    /// **Non-owning contract.** The returned `TrainingBatch`'s pointers
    /// alias this buffer's reusable sample storage. They are valid
    /// only until the next `sample()` call on this `ReplayBuffer`.
    /// The training worker consumes each batch synchronously via
    /// `trainer.trainStep(batch:)` before requesting the next one, so
    /// no overlap is possible in practice.
    func sample(count sampleCount: Int) -> TrainingBatch? {
        precondition(sampleCount > 0, "Sample count must be positive")
        lock.lock()
        defer { lock.unlock() }
        let held = storedCount
        guard held >= sampleCount else { return nil }

        ensureReusableBatchCapacity(sampleCount)

        guard
            let dstBoards = reusableBatchBoards,
            let dstMoves = reusableBatchMoves,
            let dstZs = reusableBatchZs
        else {
            // `ensureReusableBatchCapacity` always populates the three
            // pointers for any positive size, so this branch is
            // unreachable unless allocation failed — surface the
            // failure rather than hand back nil silently.
            return nil
        }

        let floatsPerBoard = Self.floatsPerBoard

        for i in 0..<sampleCount {
            let srcIndex = Int.random(in: 0..<held)
            (dstBoards + i * floatsPerBoard).update(
                from: boardStorage + srcIndex * floatsPerBoard,
                count: floatsPerBoard
            )
            dstMoves[i] = moveStorage[srcIndex]
            dstZs[i] = outcomeStorage[srcIndex]
        }

        return TrainingBatch(
            boards: UnsafePointer(dstBoards),
            moves: UnsafePointer(dstMoves),
            zs: UnsafePointer(dstZs),
            batchSize: sampleCount
        )
    }

    /// Grow the three reusable output buffers to at least the given
    /// capacity, allocating lazily on the first call and re-allocating
    /// only if a larger batch is ever requested. The caller must hold
    /// `lock`.
    private func ensureReusableBatchCapacity(_ needed: Int) {
        guard needed > reusableBatchCapacity else { return }

        if let ptr = reusableBatchBoards {
            ptr.deinitialize(count: reusableBatchCapacity * Self.floatsPerBoard)
            ptr.deallocate()
        }
        if let ptr = reusableBatchMoves {
            ptr.deinitialize(count: reusableBatchCapacity)
            ptr.deallocate()
        }
        if let ptr = reusableBatchZs {
            ptr.deinitialize(count: reusableBatchCapacity)
            ptr.deallocate()
        }

        let boardSlots = needed * Self.floatsPerBoard
        let newBoards = UnsafeMutablePointer<Float>.allocate(capacity: boardSlots)
        newBoards.initialize(repeating: 0, count: boardSlots)
        reusableBatchBoards = newBoards

        let newMoves = UnsafeMutablePointer<Int32>.allocate(capacity: needed)
        newMoves.initialize(repeating: 0, count: needed)
        reusableBatchMoves = newMoves

        let newZs = UnsafeMutablePointer<Float>.allocate(capacity: needed)
        newZs.initialize(repeating: 0, count: needed)
        reusableBatchZs = newZs

        reusableBatchCapacity = needed
    }
}
