import CryptoKit
import Foundation
import os

/// Thread-safe fixed-capacity ring of labeled self-play positions.
///
/// Self-play workers push whole games via `append(boards:policyIndices:outcome:count:)`
/// once each game ends and outcomes are known. The trainer pulls out
/// minibatches via `sample(count:intoBoards:moves:zs:)`.
/// Both sides run on background tasks; access is serialized by a
/// private `OSAllocatedUnfairLock` so the buffer is safe to share
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
/// Marked `@unchecked Sendable` because the lock serializes all
/// mutable state access. The lock is never held across an `await`.
final class ReplayBuffer: @unchecked Sendable {
    /// Number of floats required to hold one encoded board position
    /// (`arch.inputPlanes` × 8 × 8 — e.g. 30 × 64 = 1920 for the basic30
    /// encoding, 20 × 64 = 1280 for basic20). Per-architecture, fixed at
    /// construction: a buffer holds encodings produced by ONE network, so
    /// its stride must match that network's input-plane count. The live
    /// training buffer passes the session network's actual stride
    /// explicitly; a mismatched stride would silently misinterpret every
    /// stored board, which is also why `restore` rejects a file whose
    /// recorded stride differs from this value.
    let floatsPerBoard: Int

    /// Convenience stride for the DEFAULT architecture's input encoding
    /// (`NetworkArchitecture.current.inputEncoding`). Used by tests, the UI
    /// RAM estimate, and tooling that has no specific net in hand. NOT used
    /// by the live training buffer — that passes the session network's actual
    /// `tensorLength(for: network.inputEncoding)` explicitly, and the restore
    /// paths peek the file's recorded stride. Tracks the current preset's
    /// encoding rather than hardcoding basic30, so it stays correct if a
    /// different encoding ever becomes the default.
    static var defaultFloatsPerBoard: Int {
        BoardEncoder.tensorLength(for: NetworkArchitecture.current.inputEncoding)
    }

    /// Maximum number of positions held. Older positions are overwritten
    /// in FIFO order once the buffer is full.
    let capacity: Int

    private let lock = OSAllocatedUnfairLock()

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

    /// Per-position 0-based ply index within its game. Same ring index
    /// as `boardStorage`. Observability; persisted in v5+.
    private let plyIndexStorage: UnsafeMutablePointer<UInt16>

    /// Per-position total game length (plies). Broadcast across every
    /// row of a single appended game. Same ring index as `boardStorage`.
    /// Observability; persisted in v5+.
    private let gameLengthStorage: UnsafeMutablePointer<UInt16>

    /// Per-position temperature (tau) actually used at sampling time.
    /// Same ring index as `boardStorage`. Observability; persisted in v5+.
    private let samplingTauStorage: UnsafeMutablePointer<Float>

    /// Per-position state hash (Swift `Hasher` over the encoded board
    /// bytes). Used as the key for global per-position duplicate
    /// counts. Same ring index as `boardStorage`. Observability;
    /// persisted in v5+ (per-slot hash bytes serialized; the `hashStats`
    /// dict itself is rebuilt on restore).
    private let stateHashStorage: UnsafeMutablePointer<UInt64>

    /// Per-position 32-bit packed identity: high 16 bits = `workerId`,
    /// low 16 bits = `intraWorkerGameIndex` (masked). Broadcast across every
    /// row of a single appended game. Same ring index as `boardStorage`.
    /// Observability; persisted in v5+.
    private let workerGameIdStorage: UnsafeMutablePointer<UInt32>

    /// Per-position non-pawn piece count (range 0–30). Drives the
    /// "phase by material" bucket of the per-batch stats summary,
    /// independent of the ply-based phase bucket. UInt8 is plenty —
    /// chess starts with 16 non-pawn pieces total. Same ring index
    /// as `boardStorage`. Persisted in v6+.
    private let materialCountStorage: UnsafeMutablePointer<UInt8>

    /// Number of positions currently held, capped at `capacity`.
    private var storedCount: Int = 0
    /// Next write slot in the ring.
    private var writeIndex: Int = 0

    /// Scratch dict reused by the fast-path `sample(...)` (no-op
    /// constraints branch) to track per-game position counts within
    /// each batch. Cleared with `removeAll(keepingCapacity: true)` on
    /// each call rather than allocating a fresh `[UInt32: Int]` per
    /// sample. Touched only under `lock`.
    private var fastPerGameScratch: [UInt32: Int] = [:]

    // MARK: - Global hash bookkeeping (observability)

    /// How many ring slots currently hold each unique `state_hash`,
    /// plus per-outcome counts so we can distinguish identical
    /// positions reached by independent rollouts (different outcomes
    /// = legitimate diversity) from pure duplicates (identical
    /// position + identical outcome).
    ///
    /// Updated on every append (incremented per slot written) and on
    /// every eviction (decremented per slot overwritten). Entries
    /// drop to zero are removed so the dict size tracks the unique-
    /// position count rather than the high-water mark.
    public struct BufferedPositionStats: Sendable {
        public var count: UInt32
        public var winSum: UInt32
        public var drawSum: UInt32
        public var lossSum: UInt32
    }
    private var hashStats: [UInt64: BufferedPositionStats] = [:]

    // MARK: - Composition aggregates (observability; rebuilt on restore)
    //
    // Running tallies over the *resident* positions, maintained
    // incrementally in `append`'s insertion/eviction passes and rebuilt
    // in `restore`. They power the O(1) buffer-composition readout
    // (`compositionSnapshot()`) and the length-tilt β solve in
    // `sample(...)`. Lock discipline: mutated only while holding `lock`.
    private var winPositions: Int = 0
    private var drawPositions: Int = 0
    private var lossPositions: Int = 0
    /// Σ over resident positions of that position's game length. Divided
    /// by `storedCount` gives the *position-weighted* mean game length
    /// (= E[L²]/E[L]); `storedCount / distinctResidentGames` gives the
    /// *game-weighted* mean. Both are surfaced in the UI.
    private var sumGameLengthOverResidentPositions: Int = 0
    /// `packedWorkerGameId → resident position count for that game`.
    /// `.count` is the number of distinct resident games. Known wart:
    /// across a *resumed* session the per-worker game index resets, so an
    /// old resident game can collide with a new one and be merged here —
    /// harmless (a slightly low distinct-game count, two games treated as
    /// one for the per-game-cap), not worth tracking a session epoch for.
    private var residentGames: [UInt32: Int] = [:]
    /// Count of resident games whose outcome is decisive (win or loss).
    /// Maintained in lock-step with `residentGames` so the constrained
    /// sampler can size strata against the K-cap-reachable decisive
    /// position pool (`maxPerGame · residentDecisiveGameCount`) without
    /// rescanning the dict every call. Invariant:
    /// `residentDecisiveGameCount ≤ residentGames.count`.
    private var residentDecisiveGameCount: Int = 0
    /// `gameLength → resident position count at that length`. Drives the
    /// length-tilt β root-find in `sample(...)` without re-walking the
    /// ring each call.
    private var residentLengthHistogram: [UInt16: Int] = [:]

    /// Per-batch composition constraints applied by `sample(...)`. Owned
    /// here (rather than passed per call) so the off-main trainer never
    /// needs to read `TrainingParameters.shared`; the main actor pushes
    /// the current values via `setSamplingConstraints(_:)` at session
    /// start and then reactively from `ControlSideEffectsProbe` whenever
    /// one of the three constraint params changes. Defaults to
    /// `.unconstrained` ⇒ `sample` is bit-for-bit the legacy
    /// uniform-with-replacement sampler until the user opts in.
    private var samplingConstraints: SamplingConstraints = .unconstrained

    // MARK: - Material-bucket index (observability + stratified sampling)
    //
    // Per-bucket lists of resident ring-slot indices, maintained in
    // lock-step with `materialCountStorage`. Buckets follow
    // `ReplayBufferAnalyzer.materialBuckets` so any downstream display
    // (UI mini-chart, [BATCH-STATS] `bucket_mix`, CLI recorder) reads
    // the same partition as the offline analyzer. The 5th bucket
    // (23–30 non-pawn pieces) is geometrically unreachable in standard
    // chess but is kept for index alignment with the analyzer.
    //
    // `materialBucketSlots[b]` provides O(1) insert, O(1) remove-by-slot,
    // and O(1) uniform random pick — backed by a contiguous `[Int]`
    // plus a `[Int: Int]` reverse-index. Updated only while holding
    // `lock`. Drives:
    //   - the in-memory bucket distribution surfaced in
    //     `compositionSnapshot()` (the UI's per-batch mini-chart
    //     "buffer" row);
    //   - the bucket-stratified `sample(...)` path enabled when
    //     `samplingConstraints.materialBucketWeights != nil`.
    //
    // No on-disk format change: the index is rebuilt by walking
    // `materialCountStorage` once at the end of `restore(from:)`,
    // matching the same pattern used for `hashStats` and the
    // composition aggregates.
    private var materialBucketSlots: [IndexedSlotSet]

    // MARK: - Lifetime

    /// O(1)-everything container for the per-bucket slot index. Backs
    /// `materialBucketSlots`. NOT thread-safe on its own — must be
    /// touched only while the surrounding `ReplayBuffer.lock` is held.
    ///
    /// `slots` is the contiguous list of resident ring-slot indices in
    /// this bucket. `slotPosition[slot]` is the index of `slot` within
    /// `slots`, so `remove(slot)` does the standard "swap with last,
    /// pop, fix-up reverse index" trick in O(1).
    struct IndexedSlotSet {
        private(set) var slots: [Int] = []
        private var slotPosition: [Int: Int] = [:]

        var count: Int { slots.count }
        var isEmpty: Bool { slots.isEmpty }

        mutating func insert(_ slot: Int) {
            // Guard against double-insert: the bucket index is
            // maintained alongside materialCountStorage, so a slot
            // being inserted twice for the same write would mean the
            // eviction pass was skipped — we want a precondition
            // failure not silent corruption.
            precondition(slotPosition[slot] == nil,
                "IndexedSlotSet.insert: slot \(slot) is already a member")
            slotPosition[slot] = slots.count
            slots.append(slot)
        }

        mutating func remove(_ slot: Int) {
            guard let idx = slotPosition.removeValue(forKey: slot) else { return }
            let last = slots.count - 1
            if idx != last {
                let moved = slots[last]
                slots[idx] = moved
                slotPosition[moved] = idx
            }
            slots.removeLast()
        }

        /// Uniform random pick. Caller must check `!isEmpty` first.
        func randomSlot() -> Int {
            slots[Int.random(in: 0..<slots.count)]
        }

        mutating func removeAll(keepingCapacity: Bool = true) {
            slots.removeAll(keepingCapacity: keepingCapacity)
            slotPosition.removeAll(keepingCapacity: keepingCapacity)
        }
    }

    /// Resolve a `materialCount` (non-pawn piece count) to its bucket
    /// index, matching `ReplayBufferAnalyzer.materialBucketIndex(for:)`.
    /// Single source of truth: read straight from
    /// `ReplayBufferAnalyzer.materialBuckets` so a future redefinition
    /// of the boundaries flows through to the buffer and the analyzer
    /// in lock-step.
    @inline(__always)
    static func materialBucketIndex(for materialCount: UInt8) -> Int {
        ReplayBufferAnalyzer.materialBucketIndex(for: Int(materialCount))
    }

    init(capacity: Int, floatsPerBoard: Int = ReplayBuffer.defaultFloatsPerBoard) {
        precondition(capacity > 0, "Replay buffer capacity must be positive")
        precondition(floatsPerBoard > 0, "Replay buffer floatsPerBoard must be positive")
        self.capacity = capacity
        self.floatsPerBoard = floatsPerBoard

        // Pre-allocate one IndexedSlotSet per analyzer bucket. The 5th
        // (23–30) is structurally unreachable but kept to keep array
        // arithmetic 1:1 with the analyzer's outputs.
        self.materialBucketSlots = Array(
            repeating: IndexedSlotSet(),
            count: ReplayBufferAnalyzer.materialBuckets.count
        )

        let boardSlots = capacity * self.floatsPerBoard
        let boards = UnsafeMutablePointer<Float>.allocate(capacity: boardSlots)
        boards.initialize(repeating: 0, count: boardSlots)
        self.boardStorage = boards

        let moves = UnsafeMutablePointer<Int32>.allocate(capacity: capacity)
        moves.initialize(repeating: 0, count: capacity)
        self.moveStorage = moves

        let outcomes = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        outcomes.initialize(repeating: 0, count: capacity)
        self.outcomeStorage = outcomes

        let plies = UnsafeMutablePointer<UInt16>.allocate(capacity: capacity)
        plies.initialize(repeating: 0, count: capacity)
        self.plyIndexStorage = plies

        let lengths = UnsafeMutablePointer<UInt16>.allocate(capacity: capacity)
        lengths.initialize(repeating: 0, count: capacity)
        self.gameLengthStorage = lengths

        let taus = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        taus.initialize(repeating: 0, count: capacity)
        self.samplingTauStorage = taus

        let hashes = UnsafeMutablePointer<UInt64>.allocate(capacity: capacity)
        hashes.initialize(repeating: 0, count: capacity)
        self.stateHashStorage = hashes

        let workerGames = UnsafeMutablePointer<UInt32>.allocate(capacity: capacity)
        workerGames.initialize(repeating: 0, count: capacity)
        self.workerGameIdStorage = workerGames

        let materials = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
        materials.initialize(repeating: 0, count: capacity)
        self.materialCountStorage = materials
    }

    deinit {
        let boardSlots = capacity * self.floatsPerBoard
        boardStorage.deinitialize(count: boardSlots)
        boardStorage.deallocate()

        moveStorage.deinitialize(count: capacity)
        moveStorage.deallocate()

        outcomeStorage.deinitialize(count: capacity)
        outcomeStorage.deallocate()

        plyIndexStorage.deinitialize(count: capacity)
        plyIndexStorage.deallocate()

        gameLengthStorage.deinitialize(count: capacity)
        gameLengthStorage.deallocate()

        samplingTauStorage.deinitialize(count: capacity)
        samplingTauStorage.deallocate()

        stateHashStorage.deinitialize(count: capacity)
        stateHashStorage.deallocate()

        workerGameIdStorage.deinitialize(count: capacity)
        workerGameIdStorage.deallocate()

        materialCountStorage.deinitialize(count: capacity)
        materialCountStorage.deallocate()
    }

    // MARK: - Introspection

    /// Current number of positions stored (up to `capacity`).
    var count: Int {
        lock.withLock { storedCount }
    }

    /// Monotonically increasing count of all positions ever appended
    /// (not capped at `capacity` — includes positions that have since
    /// been overwritten by the ring). Read by the replay-ratio
    /// controller to compute the 1-minute self-play production rate
    /// without any coupling between self-play workers and the
    /// training worker.
    private var _totalPositionsAdded: Int = 0
    var totalPositionsAdded: Int {
        lock.withLock { _totalPositionsAdded }
    }

    /// Per-position storage cost in bytes: board floats + move int32 +
    /// outcome float + observability fields (ply UInt16 + gameLength
    /// UInt16 + tau Float + hash UInt64 + workerGameId UInt32 +
    /// materialCount UInt8). Used by the UI to estimate buffer RAM usage.
    static func bytesPerPosition(floatsPerBoard: Int) -> Int {
        floatsPerBoard * MemoryLayout<Float>.size
            + MemoryLayout<Int32>.size       // move
            + MemoryLayout<Float>.size       // outcome
            + MemoryLayout<UInt16>.size      // plyIndex
            + MemoryLayout<UInt16>.size      // gameLength
            + MemoryLayout<Float>.size       // samplingTau
            + MemoryLayout<UInt64>.size      // stateHash
            + MemoryLayout<UInt32>.size      // workerGameId
            + MemoryLayout<UInt8>.size       // materialCount
    }
    /// This buffer's per-position storage cost in bytes.
    var bytesPerPosition: Int { Self.bytesPerPosition(floatsPerBoard: floatsPerBoard) }

    /// Typed pointer view of every per-slot column. Lifetime is bounded
    /// to a single `withSlotData(_:)` block: the analyzer reads from
    /// these pointers while holding the buffer's lock, so callers MUST
    /// NOT escape them. `count` is the live `storedCount`; only slots
    /// `0..<count` carry valid data.
    struct SlotPointers: @unchecked Sendable {
        let count: Int
        let capacity: Int
        let boards: UnsafePointer<Float>
        let moves: UnsafePointer<Int32>
        let outcomes: UnsafePointer<Float>
        let plyIndex: UnsafePointer<UInt16>
        let gameLength: UnsafePointer<UInt16>
        let samplingTau: UnsafePointer<Float>
        let stateHash: UnsafePointer<UInt64>
        let workerGameId: UnsafePointer<UInt32>
        let materialCount: UnsafePointer<UInt8>
    }

    /// Hold the buffer's lock and hand `block` a `SlotPointers` view of
    /// every per-slot column. Used by the offline-analyzer paths (CLI
    /// flag + Debug menu item) that need to walk every slot in one pass
    /// without acquiring the lock per slot. Appends and samples are
    /// blocked for the duration of `block` — keep it brief on a live
    /// buffer; on a 1 M-position buffer the analyzer's single pass
    /// completes well under a second.
    ///
    /// `block` must be `@Sendable` and return a `Sendable` value because
    /// `OSAllocatedUnfairLock.withLock` enforces both — and that's
    /// stricter than what the caller actually needs at runtime (the
    /// closure runs synchronously on the calling thread, never crosses
    /// an isolation boundary). To stay within the constraint, the
    /// analyzer is structured so the whole walk + reduction happens
    /// inside this one closure and the closure returns a self-contained
    /// `Sendable` result struct.
    func withSlotData<R: Sendable>(
        _ block: @Sendable (SlotPointers) throws -> R
    ) rethrows -> R {
        try lock.withLock {
            let view = SlotPointers(
                count: storedCount,
                capacity: capacity,
                boards: UnsafePointer(boardStorage),
                moves: UnsafePointer(moveStorage),
                outcomes: UnsafePointer(outcomeStorage),
                plyIndex: UnsafePointer(plyIndexStorage),
                gameLength: UnsafePointer(gameLengthStorage),
                samplingTau: UnsafePointer(samplingTauStorage),
                stateHash: UnsafePointer(stateHashStorage),
                workerGameId: UnsafePointer(workerGameIdStorage),
                materialCount: UnsafePointer(materialCountStorage)
            )
            return try block(view)
        }
    }

    /// Peek at a saved `replay_buffer.bin` header just far enough to
    /// read the `capacity` field — used by the CLI analyzer so it can
    /// allocate a `ReplayBuffer` of the exact size needed to hold every
    /// stored slot, rather than over-allocating against the
    /// `maxReasonableCapacity` ceiling. Validates the magic + version
    /// before returning the parsed value. The full `restore(from:)`
    /// path repeats every validation step, so a header peek that
    /// passes here can still be rejected by `restore` if the body or
    /// trailer is corrupt.
    static func peekCapacity(at url: URL) throws -> Int {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw PersistenceError.readFailed(error)
        }
        // `try?` on FileHandle.close() in a `defer` matches the same
        // idiom used by `write(to:)` and `restore(from:)` elsewhere in
        // this file: a close-time error after a successful read can't
        // invalidate the bytes we already consumed, and on an error
        // path we want the original `throw` to propagate rather than
        // be masked by the close failure.
        defer { try? handle.close() }
        guard let headerData = try handle.read(upToCount: headerSize),
              headerData.count == headerSize else {
            throw PersistenceError.truncatedHeader
        }
        let magicMatches = headerData.prefix(8).elementsEqual(fileMagic)
        guard magicMatches else { throw PersistenceError.badMagic }
        let version: UInt32 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
        }
        guard version == fileVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }
        let cap64: Int64 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 24, as: Int64.self)
        }
        guard cap64 > 0 && cap64 <= maxReasonableCapacity else {
            throw PersistenceError.upperBoundExceeded(
                field: "capacity",
                value: cap64,
                max: maxReasonableCapacity
            )
        }
        return Int(cap64)
    }

    /// Read the per-position board stride (`floatsPerBoard`) a saved file
    /// was written with, without loading it. Needed when constructing a
    /// fresh `ReplayBuffer` to restore INTO: the buffer's stride is fixed at
    /// init and must match the file's, so a restore target for a foreign-
    /// encoding file (e.g. a basic20 buffer = 1280) can't just assume the
    /// default. Mirrors `peekCapacity` (floatsPerBoard is at byte offset 16).
    static func peekFloatsPerBoard(at url: URL) throws -> Int {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw PersistenceError.readFailed(error)
        }
        defer { try? handle.close() }
        guard let headerData = try handle.read(upToCount: headerSize),
              headerData.count == headerSize else {
            throw PersistenceError.truncatedHeader
        }
        let magicMatches = headerData.prefix(8).elementsEqual(fileMagic)
        guard magicMatches else { throw PersistenceError.badMagic }
        let version: UInt32 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
        }
        guard version == fileVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }
        let fpb64: Int64 = headerData.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 16, as: Int64.self)
        }
        guard fpb64 > 0 && fpb64 <= maxReasonableFloatsPerBoard else {
            throw PersistenceError.upperBoundExceeded(
                field: "floatsPerBoard",
                value: fpb64,
                max: maxReasonableFloatsPerBoard
            )
        }
        return Int(fpb64)
    }

    // MARK: - Hash helper (encoded board → stable UInt64 hash)

    /// Hash an encoded `[floatsPerBoard]` board tensor into a single
    /// UInt64. Uses Swift's stdlib `Hasher` (SipHash) over the raw
    /// bytes — process-stable but not persistence-stable. The hash dict
    /// is rebuilt fresh on each session restore so cross-process
    /// stability isn't required.
    @inline(__always)
    static func hashBoard(_ ptr: UnsafePointer<Float>, count: Int) -> UInt64 {
        var hasher = Hasher()
        let raw = UnsafeRawBufferPointer(
            start: UnsafeRawPointer(ptr),
            count: count * MemoryLayout<Float>.size
        )
        hasher.combine(bytes: raw)
        return UInt64(bitPattern: Int64(hasher.finalize()))
    }

    /// Pack a worker_id (0..65_535) and an intra-worker game index
    /// into a single UInt32 for storage in `workerGameIdStorage`.
    /// Top 16 bits = worker, low 16 bits = game. The `gameIndex`
    /// parameter is `UInt32` but is silently masked to 16 bits inside
    /// the pack, so aliasing occurs after 65_536 games per worker
    /// (call sites typically pass an `ActiveGame.intraWorkerGameIndex`
    /// that increments with `&+= 1` in the full UInt32 domain).
    ///
    /// The 16-bit worker domain is large enough that the slot-ID
    /// counter (which monotonically increments across arena
    /// respawns and Stepper resizes — see `BatchedSelfPlayDriver`)
    /// won't wrap inside any realistic session length: at 48 slots
    /// respawning per arena and ~5 min between arenas, 65_536 / 48
    /// ≈ 1366 arenas before the modulo wraps, i.e. ~110 hours.
    @inline(__always)
    static func packWorkerGameId(workerId: UInt16, gameIndex: UInt32) -> UInt32 {
        (UInt32(workerId) << 16) | (gameIndex & 0x0000_FFFF)
    }
    @inline(__always)
    static func unpackWorkerGameId(_ packed: UInt32) -> (workerId: UInt16, gameIndex: UInt32) {
        (UInt16(packed >> 16), packed & 0x0000_FFFF)
    }

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
        lock.withLock {
            StateSnapshot(
                storedCount: storedCount,
                capacity: capacity,
                writeIndex: writeIndex,
                totalPositionsAdded: _totalPositionsAdded
            )
        }
    }

    // MARK: - Composition snapshot + sampling constraints

    /// Per-training-batch composition constraints for `sample(...)`. The
    /// defaults make `sample` behave exactly like the legacy
    /// uniform-with-replacement sampler — see `isNoOp(forBatchSize:)`.
    public struct SamplingConstraints: Sendable, Equatable {
        /// Max positions drawn from any one game within a single batch.
        /// No-op for any value ≥ the batch size.
        public var maxPerGame: Int
        /// Ceiling on the % of sampled positions from drawn games
        /// (`outcome == 0`). True ceiling, not a target:
        /// - if the buffer's *natural* draw fraction (under the
        ///   `maxPerGame` K cap) is **below** this value, the batch
        ///   matches the natural rate — no draw-side padding.
        /// - if the natural rate is **above** this value, the batch
        ///   is downsampled to this percentage.
        /// `100` ⇒ no cap; batch tracks the natural rate exactly
        /// (subject to the other constraints).
        public var maxDrawPercent: Int
        /// Target position-weighted mean game length (plies) of the
        /// sampled batch, enforced by an exponential down-weight on long
        /// games. `0` ⇒ no length tilt.
        public var targetMeanGameLengthPlies: Int
        /// Per-bucket target weights for material-phase stratification.
        ///
        /// `nil` (default) ⇒ no stratification — the sampler uses the
        /// legacy uniform / constrained path with the three knobs above.
        ///
        /// Non-nil ⇒ the sampler draws minibatches with a target
        /// distribution across the 4 active material buckets defined by
        /// `ReplayBufferAnalyzer.materialBuckets` (0–4, 5–8, 9–14,
        /// 15–22 non-pawn pieces; the 5th `23–30` slot is geometrically
        /// unreachable and stays at weight 0). Array length must equal
        /// `ReplayBufferAnalyzer.materialBuckets.count`. Weights are
        /// proportions (not %) and are renormalized internally before
        /// the per-bucket count is rounded — the convention is to pass
        /// `[0.25, 0.25, 0.25, 0.25, 0.0]` for a balanced batch.
        ///
        /// V1 limitation: when this is non-nil the sampler **bypasses**
        /// the W/D/L cap and the per-game K cap (`maxPerGame`,
        /// `maxDrawPercent`). The UI grays those controls out with an
        /// inline explanation banner while stratification is on. Future
        /// work could combine bucket stratification with W/D/L tilt.
        public var materialBucketWeights: [Float]?

        public init(
            maxPerGame: Int,
            maxDrawPercent: Int,
            targetMeanGameLengthPlies: Int,
            materialBucketWeights: [Float]? = nil
        ) {
            self.maxPerGame = maxPerGame
            self.maxDrawPercent = maxDrawPercent
            self.targetMeanGameLengthPlies = targetMeanGameLengthPlies
            self.materialBucketWeights = materialBucketWeights
        }

        /// The legacy uniform-with-replacement sampler.
        public static let unconstrained = SamplingConstraints(
            maxPerGame: Int.max, maxDrawPercent: 100, targetMeanGameLengthPlies: 0,
            materialBucketWeights: nil
        )

        /// Build from the current `TrainingParameters` values (the UI
        /// heartbeat and the session-start path both push this into the
        /// live `ReplayBuffer`).
        @MainActor public static func fromCurrentParameters() -> SamplingConstraints {
            let p = TrainingParameters.shared
            // V1: balanced target across the 4 active buckets when the
            // toggle is on. Custom weights are future work — see the
            // `ReplayBufferStratifyByMaterial` parameter description.
            let bucketWeights: [Float]?
            if p.replayBufferStratifyByMaterial {
                let bucketCount = ReplayBufferAnalyzer.materialBuckets.count
                // The 5th bucket (23–30 non-pawns) is structurally
                // unreachable, so its weight stays 0. The remaining
                // active buckets share the budget equally.
                let activeCount = bucketCount - 1
                var w = [Float](repeating: 0, count: bucketCount)
                let share = activeCount > 0 ? Float(1.0) / Float(activeCount) : 0
                for i in 0..<activeCount { w[i] = share }
                bucketWeights = w
            } else {
                bucketWeights = nil
            }
            return SamplingConstraints(
                maxPerGame: p.maxPliesFromAnyOneGame,
                maxDrawPercent: p.maxDrawPercentPerBatch,
                targetMeanGameLengthPlies: p.targetSampledGameLengthPlies,
                materialBucketWeights: bucketWeights
            )
        }

        /// True when this is equivalent to the legacy uniform sampler for
        /// a batch of `sampleCount` positions (so `sample` can take the
        /// bit-for-bit fast path).
        func isNoOp(forBatchSize sampleCount: Int) -> Bool {
            // Material-bucket stratification is never a no-op: even at
            // balanced weights matching the natural buffer distribution
            // the path differs from uniform-with-replacement (per-bucket
            // draws), so it always takes the dedicated path.
            materialBucketWeights == nil
                && maxPerGame >= sampleCount
                && maxDrawPercent >= 100
                && targetMeanGameLengthPlies <= 0
        }
    }

    /// Achievement report for the most recent `sample(...)` call. Captures
    /// both what the caller asked for (the active `SamplingConstraints`)
    /// and what the buffer was actually able to deliver — the trainer
    /// uses the gap between requested and achieved to emit a `[SAMPLER]`
    /// log line when constraints were degraded, and the `BatchStatsSummary`
    /// uses it to caption the post-constraint histograms.
    ///
    /// Per-batch achievement counters (`achievedDrawCount` and friends,
    /// `distinctGamesInBatch`, `achievedMaxPerGame`, `achievedSumGameLength`)
    /// are populated on **both** the no-op fast path and the constrained
    /// path so the UI's "Last sampled batch" readout works regardless of
    /// constraint state. `wasConstrainedPath == false` means the fast
    /// path was taken (no stratification / no tilt / no per-game cap
    /// enforcement) — the counters describe what came out anyway. The
    /// pre-constraint *requested* fields (`requestedDrawCount`) are
    /// zero on the fast path because there was no request to compare
    /// against. The default value (`.uninitialized`) is what's reported
    /// before any `sample` call has happened against this buffer.
    public struct SamplingResult: Sendable, Equatable {
        /// Mirrors the legacy `sample(...) -> Bool` return: `false` when
        /// the buffer held fewer positions than `sampleCount` and no
        /// emit happened.
        public let didSample: Bool
        /// `false` when `sample` took the bit-for-bit uniform fast path
        /// (constraints at their no-op settings). The achievement counters
        /// below are populated in both paths; this flag only says which
        /// code path produced the batch.
        public let wasConstrainedPath: Bool
        public let constraints: SamplingConstraints
        public let batchSize: Int
        /// `round(maxDrawPercent% · batchSize)` — the draw stratum size
        /// that would have been emitted if both strata had unlimited
        /// resident positions. Compare with `achievedDrawCount` to spot
        /// stratum clamping. `0` on the no-op fast path (no request).
        public let requestedDrawCount: Int
        public let achievedWinCount: Int
        public let achievedDrawCount: Int
        public let achievedLossCount: Int
        public let achievedMaxPerGame: Int
        /// Distinct `workerGameId` values present in the emitted batch.
        /// `batchSize / distinctGamesInBatch` is the avg samples per game.
        public let distinctGamesInBatch: Int
        /// Σ over emitted positions of that position's game length.
        /// `achievedMeanGameLength = achievedSumGameLength / batchSize`.
        public let achievedSumGameLength: Int
        /// `true` when the requested `targetMeanGameLengthPlies` lies at
        /// or below the shortest resident game length in the buffer —
        /// the tilted-mean limit at β→∞ is the shortest length, so the
        /// target is unreachable. β is clamped to a large value and the
        /// trainer surfaces a `[SAMPLER]` line on the next stats step.
        public let lengthTargetInfeasible: Bool
        /// Shortest game length present in the resident histogram at the
        /// time of this `sample` call. `0` when the buffer was empty or
        /// the length tilt was disabled.
        public let shortestResidentLength: Int
        /// `true` when the rejection loop hit `attemptBudget` and fell
        /// back to a uniform fill of the remaining slots. Indicates a
        /// jointly-pathological constraint combination for the current
        /// buffer composition.
        public let attemptBudgetHit: Bool
        /// Per-material-bucket position counts in the emitted batch,
        /// indexed by `ReplayBufferAnalyzer.materialBuckets`. Populated
        /// on all three sampling paths (legacy fast, constrained W/D/L,
        /// bucket-stratified) so the UI's per-batch mini-chart can show
        /// the post-sampling phase distribution regardless of which
        /// path produced the batch. Always `materialBuckets.count`
        /// entries; the trailing slot (23–30) is structurally
        /// unreachable in standard chess and stays at 0.
        public let achievedBucketCounts: [Int]
        /// Per-bucket *requested* counts when bucket stratification is
        /// active (`constraints.materialBucketWeights != nil`). The
        /// renormalized + rounded target distribution that the sampler
        /// tried to hit, BEFORE the per-bucket deficit reallocation
        /// that handles empty buckets. Same shape as
        /// `achievedBucketCounts`. All zeros on the fast / constrained
        /// paths (no per-bucket request).
        public let requestedBucketCounts: [Int]

        public var achievedWinPercent: Double {
            batchSize > 0 ? Double(achievedWinCount) * 100.0 / Double(batchSize) : 0
        }
        public var achievedDrawPercent: Double {
            batchSize > 0 ? Double(achievedDrawCount) * 100.0 / Double(batchSize) : 0
        }
        public var achievedLossPercent: Double {
            batchSize > 0 ? Double(achievedLossCount) * 100.0 / Double(batchSize) : 0
        }
        public var requestedDrawPercent: Double {
            batchSize > 0 ? Double(requestedDrawCount) * 100.0 / Double(batchSize) : 0
        }
        public var achievedMeanGameLength: Double {
            batchSize > 0 ? Double(achievedSumGameLength) / Double(batchSize) : 0
        }
        public var achievedMeanSamplesPerGame: Double {
            distinctGamesInBatch > 0 ? Double(batchSize) / Double(distinctGamesInBatch) : 0
        }
        /// True when the achieved batch deviates from the request in a
        /// way worth surfacing on `[SAMPLER]`: a draw-count gap > 1
        /// position (i.e. the buffer's draw share couldn't support the
        /// requested stratum, in either direction), the length target
        /// was infeasible, or the attempt-budget fallback fired. A
        /// no-op-path or undersampled batch never registers as degraded.
        public var wasDegraded: Bool {
            guard wasConstrainedPath, didSample else { return false }
            if attemptBudgetHit || lengthTargetInfeasible { return true }
            return abs(achievedDrawCount - requestedDrawCount) > max(1, batchSize / 100)
        }

        public static let uninitialized = SamplingResult(
            didSample: false, wasConstrainedPath: false,
            constraints: .unconstrained, batchSize: 0,
            requestedDrawCount: 0,
            achievedWinCount: 0, achievedDrawCount: 0, achievedLossCount: 0,
            achievedMaxPerGame: 0, distinctGamesInBatch: 0,
            achievedSumGameLength: 0,
            lengthTargetInfeasible: false, shortestResidentLength: 0,
            attemptBudgetHit: false,
            achievedBucketCounts: Array(
                repeating: 0,
                count: ReplayBufferAnalyzer.materialBuckets.count
            ),
            requestedBucketCounts: Array(
                repeating: 0,
                count: ReplayBufferAnalyzer.materialBuckets.count
            )
        )
    }

    /// Record of the most-recent `sample(...)` call. Mutated only while
    /// holding `lock`; read by `lastSamplingResult()` and by
    /// `computeBatchStats(...)` so the `BatchStatsSummary` can carry
    /// the constraints that produced its histograms.
    private var _lastSamplingResult: SamplingResult = .uninitialized

    /// Diagnostic report for the most recent `sample(...)` call —
    /// constraints in effect, requested vs achieved draw counts,
    /// per-game cap utilisation, length-target feasibility, and whether
    /// the rejection loop hit its attempt budget. Trainer queue reads
    /// this immediately after `sample` returns to decide whether to
    /// emit a `[SAMPLER]` log line.
    func lastSamplingResult() -> SamplingResult {
        lock.withLock { _lastSamplingResult }
    }

    /// Off-main async variant of `lastSamplingResult()`. The lock
    /// acquire runs on a global executor so the awaiter (the UI
    /// heartbeat on the main actor) never synchronously waits for
    /// `sample(...)` or `append(...)` to release `lock`. Matches
    /// the pattern used by `ChessTrainer.asyncCompletedTrainSteps()`.
    func asyncLastSamplingResult() async -> SamplingResult {
        await withCheckedContinuation { (cont: CheckedContinuation<SamplingResult, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(returning: self.lock.withLock { self._lastSamplingResult })
            }
        }
    }

    /// O(1) snapshot of the resident-set composition, for the UI's
    /// "Replay sampling" readout and the `[STATS]` line.
    public struct CompositionSnapshot: Sendable, Equatable {
        public let storedCount: Int
        public let distinctResidentGames: Int
        /// Subset of `distinctResidentGames` whose outcome is decisive
        /// (win or loss). The constrained sampler uses this to compute
        /// the K-cap-reachable decisive position pool — without it, a
        /// stratum requesting more decisive positions than K · this
        /// could supply would spin the rejection loop until the attempt
        /// budget hit, then silently drop all constraints in the
        /// fallback uniform-fill.
        public let residentDecisiveGameCount: Int
        /// Estimated resident game/segment count derived from the length
        /// histogram as Σ residentPositionsAtLength / gameLength. Unlike
        /// `distinctResidentGames`, this is not fooled by reused
        /// workerGameIds after session resume. It can be fractional only
        /// for the oldest FIFO-front game if that game is partially
        /// truncated in the ring.
        public let gameWeightedResidentGameCount: Double
        public let winPositions: Int
        public let drawPositions: Int
        public let lossPositions: Int
        public let sumGameLengthOverResidentPositions: Int
        /// Per-material-bucket resident position counts, indexed by
        /// `ReplayBufferAnalyzer.materialBuckets`. Drives the "buffer"
        /// row of the popover's bucket mini-chart (the natural skew the
        /// stratifier is correcting). Always `materialBuckets.count`
        /// entries; the 5th (23–30) is structurally unreachable in
        /// standard chess and stays at 0.
        public let residentPerBucket: [Int]

        /// Game-weighted mean game length: simple mean over resident games,
        /// estimated from the resident length histogram as
        /// `storedCount / Σ(count_at_length / length)`. For complete games
        /// this exactly equals `sum(gameLengths) / count(games)`; the only
        /// approximation is a possible fractional contribution from the
        /// oldest FIFO-front game if the ring cuts through that game.
        public var meanGameLengthPerGame: Double {
            gameWeightedResidentGameCount > 0 ? Double(storedCount) / gameWeightedResidentGameCount : 0
        }
        /// Position-weighted mean game length: E[L | sample a position] =
        /// E[L²]/E[L]. ≥ the game-weighted mean whenever lengths vary; the
        /// gap between the two is the game-length dispersion.
        public var meanGameLengthPerSampledPosition: Double {
            storedCount > 0 ? Double(sumGameLengthOverResidentPositions) / Double(storedCount) : 0
        }
        public var winFraction: Double { storedCount > 0 ? Double(winPositions) / Double(storedCount) : 0 }
        public var drawFraction: Double { storedCount > 0 ? Double(drawPositions) / Double(storedCount) : 0 }
        public var lossFraction: Double { storedCount > 0 ? Double(lossPositions) / Double(storedCount) : 0 }
    }

    func compositionSnapshot() -> CompositionSnapshot {
        lock.withLock {
            var estimatedGameCount = 0.0
            for (length, positionCount) in residentLengthHistogram where length > 0 {
                estimatedGameCount += Double(positionCount) / Double(length)
            }
            let perBucket = materialBucketSlots.map { $0.count }
            return CompositionSnapshot(
                storedCount: storedCount,
                distinctResidentGames: residentGames.count,
                residentDecisiveGameCount: residentDecisiveGameCount,
                gameWeightedResidentGameCount: estimatedGameCount,
                winPositions: winPositions,
                drawPositions: drawPositions,
                lossPositions: lossPositions,
                sumGameLengthOverResidentPositions: sumGameLengthOverResidentPositions,
                residentPerBucket: perBucket
            )
        }
    }

    /// Off-main async variant of `compositionSnapshot()`. The lock
    /// acquire + per-length-bucket sum runs on a global executor so
    /// the awaiter (the UI heartbeat on the main actor) never
    /// synchronously waits for `sample(...)` or `append(...)` to
    /// release `lock`. Matches the pattern used by
    /// `ChessTrainer.asyncCompletedTrainSteps()`.
    func asyncCompositionSnapshot() async -> CompositionSnapshot {
        await withCheckedContinuation { (cont: CheckedContinuation<CompositionSnapshot, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(returning: self.compositionSnapshot())
            }
        }
    }

    /// Replace the per-batch composition constraints used by `sample(...)`.
    /// Called from the main actor at session start
    /// (`SessionController+Training.swift`) and reactively whenever one
    /// of the three sampling-constraint params changes
    /// (`ControlSideEffectsProbe`). Takes effect on the next
    /// `sample(...)` call.
    func setSamplingConstraints(_ constraints: SamplingConstraints) {
        lock.withLock { samplingConstraints = constraints }
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
    /// into the ring storage under the buffer's lock before returning.
    func append(
        boards: UnsafePointer<Float>,
        policyIndices: UnsafePointer<Int32>,
        plyIndices: UnsafePointer<UInt16>,
        samplingTaus: UnsafePointer<Float>,
        stateHashes: UnsafePointer<UInt64>,
        materialCounts: UnsafePointer<UInt8>,
        gameLength: UInt16,
        workerId: UInt16,
        intraWorkerGameIndex: UInt32,
        outcome: Float,
        count positionCount: Int
    ) {
        guard positionCount > 0 else { return }
        precondition(positionCount <= capacity,
            "ReplayBuffer.append: positionCount (\(positionCount)) exceeds capacity (\(capacity))")

        let packedId = Self.packWorkerGameId(workerId: workerId, gameIndex: intraWorkerGameIndex)
        // Outcome bucket for the per-hash WLD counters. Drawn games
        // come in as exactly 0; positive = win, negative = loss.
        let isWin: Bool = outcome > 0.5
        let isLoss: Bool = outcome < -0.5

        // `UnsafePointer`/`UnsafeMutablePointer` aren't Sendable, but this
        // closure runs synchronously under the lock and doesn't outlive
        // the call — `withLockUnchecked` is the documented escape hatch
        // for that case (see Apple's `OSAllocatedUnfairLock` reference).
        lock.withLockUnchecked {
            let floatsPerBoard = self.floatsPerBoard

            // The incoming positions may straddle the ring's wraparound
            // point. Split the write into at most two contiguous runs:
            // the tail of the ring (from writeIndex to capacity) and then
            // the head (wrapping back to index 0).
            var remaining = positionCount
            var srcOffset = 0  // in positions
            while remaining > 0 {
                let tailSlots = capacity - writeIndex
                let chunk = min(remaining, tailSlots)

                // EVICTION pass: before overwriting, decrement the
                // per-hash counters for any slot that's currently
                // storing a position (i.e. ring is full and we're
                // about to clobber). Skipped when the ring still has
                // room (storedCount + new positions ≤ capacity).
                if storedCount == capacity {
                    for i in 0..<chunk {
                        let slot = writeIndex + i
                        let oldHash = stateHashStorage[slot]
                        let oldOutcome = outcomeStorage[slot]
                        decrementHashStat(
                            hash: oldHash,
                            isWin: oldOutcome > 0.5,
                            isLoss: oldOutcome < -0.5
                        )
                        decrementCompositionAggregatesForSlot(slot)
                    }
                }

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
                // Observability fields (per-position): ply, hash, tau.
                (plyIndexStorage + writeIndex).update(
                    from: plyIndices + srcOffset,
                    count: chunk
                )
                (samplingTauStorage + writeIndex).update(
                    from: samplingTaus + srcOffset,
                    count: chunk
                )
                (stateHashStorage + writeIndex).update(
                    from: stateHashes + srcOffset,
                    count: chunk
                )
                // Observability fields (broadcast): gameLength, workerGameId.
                (gameLengthStorage + writeIndex).update(
                    repeating: gameLength,
                    count: chunk
                )
                (workerGameIdStorage + writeIndex).update(
                    repeating: packedId,
                    count: chunk
                )
                (materialCountStorage + writeIndex).update(
                    from: materialCounts + srcOffset,
                    count: chunk
                )

                // INSERTION pass for the hash dict — increment counters
                // for the freshly written slots.
                for i in 0..<chunk {
                    incrementHashStat(
                        hash: stateHashes[srcOffset + i],
                        isWin: isWin,
                        isLoss: isLoss
                    )
                }
                incrementCompositionAggregates(
                    gameLength: gameLength, packedId: packedId,
                    isWin: isWin, isLoss: isLoss, count: chunk
                )
                // Material-bucket index insertion. Reads the freshly
                // written `materialCountStorage[slot]` rather than the
                // incoming `materialCounts` pointer at the same offset
                // so the bucket index is provably consistent with the
                // on-ring data (the same data that `restore`'s rebuild
                // loop walks). One bucket per slot; the slot is fresh
                // (the matching eviction pass above just removed any
                // pre-existing tenant), so `insert` won't trip its
                // double-insert precondition.
                for i in 0..<chunk {
                    let slot = writeIndex + i
                    let bucket = Self.materialBucketIndex(for: materialCountStorage[slot])
                    materialBucketSlots[bucket].insert(slot)
                }

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

    // MARK: - Hash dict mutation (must be called while holding `lock`)

    @inline(__always)
    private func incrementHashStat(hash: UInt64, isWin: Bool, isLoss: Bool) {
        var stat = hashStats[hash] ?? BufferedPositionStats(count: 0, winSum: 0, drawSum: 0, lossSum: 0)
        stat.count &+= 1
        if isWin { stat.winSum &+= 1 }
        else if isLoss { stat.lossSum &+= 1 }
        else { stat.drawSum &+= 1 }
        hashStats[hash] = stat
    }

    @inline(__always)
    private func decrementHashStat(hash: UInt64, isWin: Bool, isLoss: Bool) {
        guard var stat = hashStats[hash] else { return }
        if stat.count <= 1 {
            hashStats.removeValue(forKey: hash)
            return
        }
        stat.count -= 1
        if isWin { stat.winSum = stat.winSum > 0 ? stat.winSum - 1 : 0 }
        else if isLoss { stat.lossSum = stat.lossSum > 0 ? stat.lossSum - 1 : 0 }
        else { stat.drawSum = stat.drawSum > 0 ? stat.drawSum - 1 : 0 }
        hashStats[hash] = stat
    }

    // MARK: - Composition aggregate mutation (must be called while holding `lock`)

    /// Add `count` resident positions, all from the same game (so they
    /// share `gameLength`, `packedId`, and outcome class).
    @inline(__always)
    private func incrementCompositionAggregates(
        gameLength: UInt16, packedId: UInt32, isWin: Bool, isLoss: Bool, count: Int
    ) {
        if isWin { winPositions += count }
        else if isLoss { lossPositions += count }
        else { drawPositions += count }
        sumGameLengthOverResidentPositions += Int(gameLength) * count
        // Track decisive-game count transitions: bump only when this
        // packedId crosses from "no resident positions" to "≥1 resident
        // position" *and* the game is decisive. The reverse transition
        // is handled in `decrementCompositionAggregatesForSlot`.
        let wasResident = residentGames[packedId] != nil
        residentGames[packedId, default: 0] += count
        if !wasResident && (isWin || isLoss) {
            residentDecisiveGameCount += 1
        }
        residentLengthHistogram[gameLength, default: 0] += count
    }

    /// Remove one resident position currently stored at ring `slot`,
    /// reading its metadata from the per-position arrays *before* the
    /// caller overwrites them.
    @inline(__always)
    private func decrementCompositionAggregatesForSlot(_ slot: Int) {
        let outcome = outcomeStorage[slot]
        let isDecisive = (outcome > 0.5) || (outcome < -0.5)
        if outcome > 0.5 { winPositions = winPositions > 0 ? winPositions - 1 : 0 }
        else if outcome < -0.5 { lossPositions = lossPositions > 0 ? lossPositions - 1 : 0 }
        else { drawPositions = drawPositions > 0 ? drawPositions - 1 : 0 }
        let len = gameLengthStorage[slot]
        sumGameLengthOverResidentPositions -= Int(len)
        if sumGameLengthOverResidentPositions < 0 { sumGameLengthOverResidentPositions = 0 }
        let gid = workerGameIdStorage[slot]
        if let c = residentGames[gid] {
            if c <= 1 {
                residentGames.removeValue(forKey: gid)
                // Last resident position for this game just left the ring.
                // Mirrors the symmetric bump in incrementCompositionAggregates.
                if isDecisive && residentDecisiveGameCount > 0 {
                    residentDecisiveGameCount -= 1
                }
            } else {
                residentGames[gid] = c - 1
            }
        }
        if let c = residentLengthHistogram[len] {
            if c <= 1 { residentLengthHistogram.removeValue(forKey: len) } else { residentLengthHistogram[len] = c - 1 }
        }
        // Material-bucket index removal. Reads from the still-resident
        // `materialCountStorage[slot]` BEFORE the caller's bulk write
        // overwrites it — same ordering invariant the outcome/length/gid
        // reads above rely on.
        let oldBucket = Self.materialBucketIndex(for: materialCountStorage[slot])
        materialBucketSlots[oldBucket].remove(slot)
    }

    /// Drop all composition aggregates (used by `restore` before refilling).
    private func resetCompositionAggregates() {
        winPositions = 0
        drawPositions = 0
        lossPositions = 0
        sumGameLengthOverResidentPositions = 0
        residentGames.removeAll(keepingCapacity: true)
        residentDecisiveGameCount = 0
        residentLengthHistogram.removeAll(keepingCapacity: true)
        for i in materialBucketSlots.indices {
            materialBucketSlots[i].removeAll(keepingCapacity: true)
        }
    }

    // MARK: - Hash dict introspection

    /// Number of distinct positions (by `state_hash`) currently held
    /// in the buffer. Read under the lock so it's atomic with respect
    /// to in-flight appends. Cheap (just dict.count).
    var uniquePositionCount: Int {
        lock.withLock { hashStats.count }
    }

    /// Look up the WLD counts for a specific hash. Returns nil if no
    /// slot in the buffer currently stores that hash. Used by the
    /// per-batch summarizer to distinguish pure dups from
    /// outcome-diverse dups.
    func bufferedPositionStats(forHash hash: UInt64) -> BufferedPositionStats? {
        lock.withLock { hashStats[hash] }
    }

    // MARK: - Sample

    /// Draw `sampleCount` positions from the positions currently held,
    /// writing them into caller-provided contiguous output buffers.
    /// Returns `false` if the buffer holds fewer than `sampleCount`
    /// positions — the caller should wait for more self-play to land
    /// before retrying.
    ///
    /// When the current `samplingConstraints` are at their no-op defaults
    /// (the ship default) this is exactly the legacy uniform-with-
    /// replacement sampler. Otherwise it applies, per batch:
    ///   • a hard ceiling on the % of positions from drawn games
    ///     (`maxDrawPercent`), enforced by outcome stratification;
    ///   • a hard cap on positions from any one game (`maxPerGame`),
    ///     enforced by a per-batch tally;
    ///   • a soft target on the position-weighted mean game length
    ///     (`targetMeanGameLengthPlies`), enforced by an exponential
    ///     down-weight on long games (β solved from the resident length
    ///     histogram). The draw/per-game caps are hard; the length target
    ///     yields first if the combination is jointly tight.
    /// All of this is done by rejection sampling over the same uniform
    /// draw the legacy path uses — O(sampleCount · small constant + #distinct
    /// resident lengths), with the unchanged O(sampleCount · floatsPerBoard)
    /// board memcpy still dominating.
    ///
    /// The training-required outputs (`dstBoards`, `dstMoves`, `dstZs`)
    /// are mandatory. The observability metadata outputs
    /// (`dstPlies`, `dstGameLengths`, `dstTaus`, `dstHashes`,
    /// `dstWorkerGameIds`) are optional — pass `nil` when the caller
    /// only needs the training payload (most batches), pass non-nil
    /// when it's a stats-collection batch.
    func sample(
        count sampleCount: Int,
        intoBoards dstBoards: UnsafeMutablePointer<Float>,
        moves dstMoves: UnsafeMutablePointer<Int32>,
        zs dstZs: UnsafeMutablePointer<Float>,
        plies dstPlies: UnsafeMutablePointer<UInt16>? = nil,
        gameLengths dstGameLengths: UnsafeMutablePointer<UInt16>? = nil,
        taus dstTaus: UnsafeMutablePointer<Float>? = nil,
        hashes dstHashes: UnsafeMutablePointer<UInt64>? = nil,
        workerGameIds dstWorkerGameIds: UnsafeMutablePointer<UInt32>? = nil,
        materialCounts dstMaterialCounts: UnsafeMutablePointer<UInt8>? = nil
    ) -> Bool {
        precondition(sampleCount > 0, "Sample count must be positive")
        // `UnsafeMutablePointer` isn't Sendable; the closure runs
        // synchronously under the lock and doesn't outlive the call.
        return lock.withLockUnchecked {
            let held = storedCount
            guard held >= sampleCount else {
                _lastSamplingResult = .uninitialized
                return false
            }

            let floatsPerBoard = self.floatsPerBoard

            @inline(__always)
            func emit(_ i: Int, _ srcIndex: Int) {
                (dstBoards + i * floatsPerBoard).update(
                    from: boardStorage + srcIndex * floatsPerBoard,
                    count: floatsPerBoard
                )
                dstMoves[i] = moveStorage[srcIndex]
                dstZs[i] = outcomeStorage[srcIndex]
                if let dstPlies { dstPlies[i] = plyIndexStorage[srcIndex] }
                if let dstGameLengths { dstGameLengths[i] = gameLengthStorage[srcIndex] }
                if let dstTaus { dstTaus[i] = samplingTauStorage[srcIndex] }
                if let dstHashes { dstHashes[i] = stateHashStorage[srcIndex] }
                if let dstWorkerGameIds { dstWorkerGameIds[i] = workerGameIdStorage[srcIndex] }
                if let dstMaterialCounts { dstMaterialCounts[i] = materialCountStorage[srcIndex] }
            }

            let constraints = samplingConstraints
            let bucketCount = ReplayBufferAnalyzer.materialBuckets.count
            let zeroBuckets = Array(repeating: 0, count: bucketCount)

            @inline(__always)
            func bucketOf(_ srcIndex: Int) -> Int {
                Self.materialBucketIndex(for: materialCountStorage[srcIndex])
            }

            // ---- Material-bucket stratified path ----
            //
            // Active when `constraints.materialBucketWeights != nil`. The
            // sampler draws `target[b]` positions from each bucket `b`,
            // where `target` is derived from the user-specified weights
            // and then deficit-reallocated across non-clamped buckets so
            // an empty bucket doesn't reduce the batch size.
            //
            // V1 limitation: this path **bypasses** the W/D/L cap
            // (`maxDrawPercent`) and the per-game K cap (`maxPerGame`).
            // The UI grays those controls out with an explanation banner
            // while this is on. Future work could add a bucket × W/D/L
            // joint stratification; the existing two-stratum scaffolding
            // in the constrained path below would be a natural starting
            // point.
            if let weights = constraints.materialBucketWeights, weights.count == bucketCount {
                // Renormalize and round to integer per-bucket counts.
                var totalWeight: Float = 0
                for w in weights where w > 0 { totalWeight += w }
                // Active buckets — those the user wants AND that have
                // resident positions. Both gates matter: a 0-weight
                // bucket should never be sampled (intended behaviour);
                // a 0-residency bucket can't be sampled (empty).
                var activeBuckets: [Int] = []
                activeBuckets.reserveCapacity(bucketCount)
                if totalWeight > 0 {
                    for i in 0..<bucketCount where weights[i] > 0 && !materialBucketSlots[i].isEmpty {
                        activeBuckets.append(i)
                    }
                }
                // Degenerate: all weights zero OR no resident buckets
                // overlap the user's weight vector. Fall back to uniform
                // fill of the full buffer rather than emit an undersized
                // batch — and flag `attemptBudgetHit` so the [SAMPLER]
                // line surfaces the degraded path.
                if activeBuckets.isEmpty {
                    var degWin = 0, degDraw = 0, degLoss = 0
                    var degSumLen = 0
                    var degBucketCounts = [Int](repeating: 0, count: bucketCount)
                    var degPerGame: [UInt32: Int] = [:]
                    degPerGame.reserveCapacity(min(sampleCount, residentGames.count) + 1)
                    for i in 0..<sampleCount {
                        let srcIndex = Int.random(in: 0..<held)
                        emit(i, srcIndex)
                        let z = outcomeStorage[srcIndex]
                        if z > 0 { degWin += 1 }
                        else if z < 0 { degLoss += 1 }
                        else { degDraw += 1 }
                        degSumLen += Int(gameLengthStorage[srcIndex])
                        degPerGame[workerGameIdStorage[srcIndex], default: 0] += 1
                        degBucketCounts[bucketOf(srcIndex)] += 1
                    }
                    var degMaxPerGame = 0
                    for (_, c) in degPerGame where c > degMaxPerGame { degMaxPerGame = c }
                    _lastSamplingResult = SamplingResult(
                        didSample: true, wasConstrainedPath: true,
                        constraints: constraints, batchSize: sampleCount,
                        requestedDrawCount: 0,
                        achievedWinCount: degWin,
                        achievedDrawCount: degDraw,
                        achievedLossCount: degLoss,
                        achievedMaxPerGame: degMaxPerGame,
                        distinctGamesInBatch: degPerGame.count,
                        achievedSumGameLength: degSumLen,
                        lengthTargetInfeasible: false, shortestResidentLength: 0,
                        attemptBudgetHit: true,
                        achievedBucketCounts: degBucketCounts,
                        requestedBucketCounts: zeroBuckets
                    )
                    return true
                }

                // Initial per-bucket target: round(sampleCount · w / Σw).
                var targets = [Int](repeating: 0, count: bucketCount)
                var requestedTargets = [Int](repeating: 0, count: bucketCount)
                var sumTargets = 0
                for i in 0..<bucketCount where weights[i] > 0 {
                    let raw = Double(sampleCount) * Double(weights[i]) / Double(totalWeight)
                    let t = Int(raw.rounded())
                    targets[i] = t
                    requestedTargets[i] = t
                    sumTargets += t
                }
                // Rounding correction — push +/-1 into the largest target
                // so Σ targets == sampleCount exactly.
                if sumTargets != sampleCount {
                    var maxIdx = 0
                    for i in 0..<bucketCount where targets[i] > targets[maxIdx] { maxIdx = i }
                    targets[maxIdx] = max(0, targets[maxIdx] + (sampleCount - sumTargets))
                    requestedTargets[maxIdx] = targets[maxIdx]
                }
                // Clamp by residency, accumulate deficit. A bucket whose
                // weight is 0 but residency is large is still eligible to
                // absorb deficit *as a last resort* — kept out of the
                // primary `activeBuckets` candidate list so it only
                // engages after every weighted bucket is at residency
                // cap. In V1 (balanced target with the structurally-
                // empty bucket-4 at weight 0) this never engages because
                // bucket 4 has 0 residency.
                var deficit = 0
                var clamped = [Bool](repeating: false, count: bucketCount)
                for i in 0..<bucketCount {
                    let resident = materialBucketSlots[i].count
                    if targets[i] >= resident {
                        deficit += targets[i] - resident
                        targets[i] = resident
                        clamped[i] = true
                    }
                }
                // Phase 1: reallocate among weighted active buckets that
                // still have room. Round-robin one slot at a time keeps
                // the distribution balanced even when buckets fill at
                // different rates.
                while deficit > 0 {
                    var progressed = false
                    for i in activeBuckets where !clamped[i] {
                        let cap = materialBucketSlots[i].count - targets[i]
                        if cap > 0 {
                            targets[i] += 1
                            deficit -= 1
                            progressed = true
                            if materialBucketSlots[i].count == targets[i] { clamped[i] = true }
                            if deficit == 0 { break }
                        } else {
                            clamped[i] = true
                        }
                    }
                    if !progressed { break }
                }
                // Phase 2 (last resort): if every weighted active bucket
                // is now at residency cap, draw the remaining deficit
                // from zero-weight buckets that still have residency.
                // This is the only way to fully fill the batch when the
                // user's weight vector concentrates mass on buckets that
                // can't supply enough positions; it preserves the
                // function's "always fill the dst buffers" contract.
                if deficit > 0 {
                    for i in 0..<bucketCount where !clamped[i] {
                        let cap = materialBucketSlots[i].count - targets[i]
                        if cap > 0 {
                            let take = min(deficit, cap)
                            targets[i] += take
                            deficit -= take
                            if deficit == 0 { break }
                        }
                    }
                }
                // After both phases the residual deficit should be
                // zero — we already verified `held >= sampleCount`, and
                // Σ residency across all buckets equals `held`. The
                // `attemptBudgetHit` flag below catches any remaining
                // deficit as a defensive guard.

                var stratWin = 0, stratDraw = 0, stratLoss = 0
                var stratSumLen = 0
                var stratPerGame: [UInt32: Int] = [:]
                stratPerGame.reserveCapacity(min(sampleCount, residentGames.count) + 1)
                var emitted = 0
                var achievedBucketCounts = [Int](repeating: 0, count: bucketCount)
                for b in 0..<bucketCount {
                    let target = targets[b]
                    if target == 0 || materialBucketSlots[b].isEmpty { continue }
                    for _ in 0..<target {
                        let srcIndex = materialBucketSlots[b].randomSlot()
                        emit(emitted, srcIndex)
                        let z = outcomeStorage[srcIndex]
                        if z > 0 { stratWin += 1 }
                        else if z < 0 { stratLoss += 1 }
                        else { stratDraw += 1 }
                        stratSumLen += Int(gameLengthStorage[srcIndex])
                        stratPerGame[workerGameIdStorage[srcIndex], default: 0] += 1
                        achievedBucketCounts[b] += 1
                        emitted += 1
                    }
                }
                // Defensive under-fill guard. Cannot trigger when
                // `held >= sampleCount` and the residency arithmetic
                // above is correct; kept so a future refactor can't
                // silently leak uninitialised dst slots.
                var stratAttemptBudgetHit = false
                if emitted < sampleCount {
                    stratAttemptBudgetHit = true
                    while emitted < sampleCount {
                        let srcIndex = Int.random(in: 0..<held)
                        emit(emitted, srcIndex)
                        let z = outcomeStorage[srcIndex]
                        if z > 0 { stratWin += 1 }
                        else if z < 0 { stratLoss += 1 }
                        else { stratDraw += 1 }
                        stratSumLen += Int(gameLengthStorage[srcIndex])
                        stratPerGame[workerGameIdStorage[srcIndex], default: 0] += 1
                        achievedBucketCounts[bucketOf(srcIndex)] += 1
                        emitted += 1
                    }
                }
                var stratMaxPerGame = 0
                for (_, c) in stratPerGame where c > stratMaxPerGame { stratMaxPerGame = c }
                _lastSamplingResult = SamplingResult(
                    didSample: true, wasConstrainedPath: true,
                    constraints: constraints, batchSize: sampleCount,
                    requestedDrawCount: 0,
                    achievedWinCount: stratWin,
                    achievedDrawCount: stratDraw,
                    achievedLossCount: stratLoss,
                    achievedMaxPerGame: stratMaxPerGame,
                    distinctGamesInBatch: stratPerGame.count,
                    achievedSumGameLength: stratSumLen,
                    lengthTargetInfeasible: false, shortestResidentLength: 0,
                    attemptBudgetHit: stratAttemptBudgetHit,
                    achievedBucketCounts: achievedBucketCounts,
                    requestedBucketCounts: requestedTargets
                )
                return true
            }

            // Fast path — bit-for-bit the legacy uniform-with-replacement
            // sampler. The emit loop is identical to the legacy
            // implementation; the extra book-keeping below is the per-batch
            // composition tally for the UI's "Last sampled batch" readout
            // (W/D/L counts, distinct games, per-game max, Σ game length).
            // O(batchSize) dict/array ops; negligible vs the per-position
            // board memcpy in `emit`. Same fields are populated on the
            // constrained path further down, so the UI doesn't have to
            // branch on which path produced the batch.
            if constraints.isNoOp(forBatchSize: sampleCount) {
                var fastWin = 0, fastDraw = 0, fastLoss = 0
                var fastSumLen = 0
                var fastBucketCounts = [Int](repeating: 0, count: bucketCount)
                // Reuse the buffer-owned dict scratch across calls.
                fastPerGameScratch.removeAll(keepingCapacity: true)
                fastPerGameScratch.reserveCapacity(min(sampleCount, residentGames.count) + 1)
                for i in 0..<sampleCount {
                    let srcIndex = Int.random(in: 0..<held)
                    emit(i, srcIndex)
                    let z = outcomeStorage[srcIndex]
                    if z > 0 { fastWin += 1 }
                    else if z < 0 { fastLoss += 1 }
                    else { fastDraw += 1 }
                    fastSumLen += Int(gameLengthStorage[srcIndex])
                    fastPerGameScratch[workerGameIdStorage[srcIndex], default: 0] += 1
                    fastBucketCounts[bucketOf(srcIndex)] += 1
                }
                var fastMaxPerGame = 0
                for (_, c) in fastPerGameScratch where c > fastMaxPerGame { fastMaxPerGame = c }
                _lastSamplingResult = SamplingResult(
                    didSample: true, wasConstrainedPath: false,
                    constraints: constraints, batchSize: sampleCount,
                    requestedDrawCount: 0,
                    achievedWinCount: fastWin,
                    achievedDrawCount: fastDraw,
                    achievedLossCount: fastLoss,
                    achievedMaxPerGame: fastMaxPerGame,
                    distinctGamesInBatch: fastPerGameScratch.count,
                    achievedSumGameLength: fastSumLen,
                    lengthTargetInfeasible: false, shortestResidentLength: 0,
                    attemptBudgetHit: false,
                    achievedBucketCounts: fastBucketCounts,
                    requestedBucketCounts: zeroBuckets
                )
                return true
            }

            // ---- Constrained sampling ----
            let drawCount = drawPositions
            let decisiveCount = held - drawCount
            let cap = max(constraints.maxPerGame, 1)

            // K-aware stratum ceilings. Under `maxPerGame = K`, the batch
            // can pull at most `K · <resident decisive game count>`
            // decisive positions (and symmetrically for draws). Whichever
            // is lower — the position pool or the K·games product — is
            // the true ceiling for that stratum. Folding both into the
            // sizing math BEFORE the rejection loop means a stratum we
            // couldn't fill never gets attempted. Without this, an
            // oversized decisive stratum (e.g. requesting more decisive
            // positions than `cap · residentDecisiveGameCount` can
            // supply) spun the rejection loop until the attempt budget
            // hit, at which point the fallback uniform-fill below
            // dropped *all* constraints (the K cap, the draw cap, the
            // length tilt) silently. That was a bug, not a feature —
            // the fix is to clamp here so the loop never enters the
            // pathological regime.
            //
            // Subtraction in `residentDrawGameCount` is safe:
            // `residentDecisiveGameCount` is maintained in lock-step with
            // `residentGames`, so the invariant
            // `residentDecisiveGameCount ≤ residentGames.count` holds.
            let residentDrawGameCount = residentGames.count - residentDecisiveGameCount
            // Saturate on overflow: an effectively-unbounded `cap` (e.g.
            // `Int.max` from the `.unconstrained` preset, which the
            // tests use to exercise the draw-cap and length-tilt code
            // paths in isolation) makes `cap * residentXGameCount`
            // overflow. Semantically the K-cap is inactive there, so
            // the stratum's reachable ceiling is just the position
            // count — `min(positions, ∞) = positions`. Use
            // `multipliedReportingOverflow` to detect and short-circuit.
            let (decProduct, decOverflow) = cap.multipliedReportingOverflow(by: residentDecisiveGameCount)
            let decisiveReachable = decOverflow ? decisiveCount : min(decisiveCount, decProduct)
            let (drawProduct, drawOverflow) = cap.multipliedReportingOverflow(by: residentDrawGameCount)
            let drawReachable = drawOverflow ? drawCount : min(drawCount, drawProduct)

            // Stratum sizes. `maxDrawPercent` is a true CEILING (not a
            // fill-to-target): the requested draw count is
            // `min(pct% · sampleCount, natural% · sampleCount)`, where
            // "natural %" is the draw fraction we'd see under uniform-
            // with-K-cap sampling (drawReachable / (drawReachable +
            // decisiveReachable)). Three cases:
            //   * natural < cap → take natural (cap doesn't bind)
            //   * natural > cap → take cap (cap downsamples draws)
            //   * pct >= 100 → take natural (no cap at all)
            // The historical implementation was a fill-to-target
            // (always emit pct% draws even when the buffer was
            // naturally below the cap), which made it impossible to
            // ever sample at the buffer's natural composition without
            // also relaxing the per-game K cap. Confirmed broken via
            // [BATCH-STATS] showing batch D=50.0% on a buffer with
            // D=58.9% — the cap was binding correctly there, but on
            // the symmetric case (buffer below cap) it was inflating.
            //
            // If the decisive pool is short of (sampleCount - bDraw),
            // the slack still flows back to draws so the batch fills
            // (existing behavior). In that case `achievedDrawCount`
            // exceeds `requestedDrawCount` and the trainer surfaces a
            // `[SAMPLER]` line via `wasDegraded` — the cap got
            // exceeded because the alternative was an underfilled
            // batch.
            //
            // `requestedDrawCount` is the EFFECTIVE intent (post-min
            // with natural). The trainer compares it against `bDraw`
            // after clamping to decide whether to emit a `[SAMPLER]`
            // line; under the new semantics a batch at the natural
            // rate is NOT degraded even if the cap was set higher.
            let pct = min(max(constraints.maxDrawPercent, 0), 100)
            let reachableTotal = drawReachable + decisiveReachable
            let naturalDrawCount: Int
            if reachableTotal > 0 {
                naturalDrawCount = Int(
                    (Double(drawReachable) / Double(reachableTotal) * Double(sampleCount)).rounded()
                )
            } else {
                naturalDrawCount = 0
            }
            let capAllowedDrawCount = Int((Double(pct) / 100.0 * Double(sampleCount)).rounded())
            let requestedDrawCount: Int
            if pct >= 100 {
                requestedDrawCount = naturalDrawCount
            } else {
                requestedDrawCount = min(capAllowedDrawCount, naturalDrawCount)
            }
            var bDraw = min(requestedDrawCount, drawReachable)
            var bDec = sampleCount - bDraw
            if bDec > decisiveReachable {
                let deficit = bDec - decisiveReachable
                bDec = decisiveReachable
                bDraw = min(bDraw + deficit, drawReachable)
            }
            // `bDraw + bDec == sampleCount` whenever at least one stratum
            // has slack to absorb the other's deficit — the common case
            // even under heavy buffer skew, because the abundant stratum
            // has plenty of reachable slack. The jointly-pathological
            // case (both reachable ceilings clamp, with too few resident
            // games on both sides to fill the requested batch under K)
            // leaves `bDraw + bDec < sampleCount`; the post-loop under-
            // fill guard further down catches that and the fallback
            // fills the remainder uniformly. Under-fill of the strata
            // in that case is unavoidable — there literally isn't
            // enough material under the requested K cap.

            // Length-tilt β over the resident length histogram (0 ⇒ no tilt).
            // The solver also reports whether the request is infeasible
            // (target ≤ shortest resident game length) and the shortest
            // resident length itself, so the trainer can surface both on
            // the `[SAMPLER]` line.
            //
            // When the target is infeasible the *effective* target used
            // by `tiltAccepts` is the shortest resident length: with
            // β = 1e18 the tilt then rejects every length > shortest
            // (factor ≈ 0) and accepts every length == shortest (the
            // `len > effectiveTarget` early-return). Without this
            // redirection the original target would force the tilt to
            // reject *every* length (since all lens > target on the
            // infeasible branch), and the loop would hit the attempt
            // budget on every batch.
            let target = constraints.targetMeanGameLengthPlies
            let tiltSolve: (beta: Double, infeasible: Bool, shortestResidentLength: Int)
            if target > 0 {
                tiltSolve = solveLengthTiltBeta(target: target)
            } else {
                tiltSolve = (0.0, false, 0)
            }
            let beta = tiltSolve.beta
            let effectiveTarget: Int = tiltSolve.infeasible ? tiltSolve.shortestResidentLength : target

            // Per-game tally for the K cap, scoped to this batch.
            // `cap` was computed above for the K-aware stratum sizing;
            // reused here as the enforcement threshold inside the loop.
            var perGameCount: [UInt32: Int] = [:]
            perGameCount.reserveCapacity(min(sampleCount, residentGames.count) + 1)

            // Bound total attempts so a jointly-pathological combination
            // can't spin forever — fall back to a uniform fill of whatever
            // remains rather than hang or return a short batch. Generous:
            // each rejected attempt is two array reads + an RNG call (a few
            // ns), so even the cap is sub-10 ms, but it's high enough that
            // the hard caps stay honoured down to a stratum holding ~0.4% of
            // the buffer — far below anything realistic.
            let attemptBudget = 256 * sampleCount + 8192

            @inline(__always)
            func tiltAccepts(_ srcIndex: Int) -> Bool {
                guard beta > 0 else { return true }
                let len = Int(gameLengthStorage[srcIndex])
                guard len > effectiveTarget else { return true }
                return Double.random(in: 0..<1) < exp(-beta * Double(len - effectiveTarget))
            }

            // Achievement tallies — accumulate as we emit, then publish
            // to `_lastSamplingResult` before returning. The attempt-budget
            // fallback also tallies perGameCount (so distinctGamesInBatch
            // and achievedMaxPerGame remain meaningful even on the
            // pathological-combination path).
            var achievedWinCount = 0
            var achievedDrawCount = 0
            var achievedLossCount = 0
            var achievedMaxPerGame = 0
            var achievedSumGameLength = 0
            var attemptBudgetHit = false
            var achievedBucketCounts = [Int](repeating: 0, count: bucketCount)

            @inline(__always)
            func tallyOutcome(_ srcIndex: Int) {
                let z = outcomeStorage[srcIndex]
                if z > 0 { achievedWinCount += 1 }
                else if z < 0 { achievedLossCount += 1 }
                else { achievedDrawCount += 1 }
                achievedBucketCounts[bucketOf(srcIndex)] += 1
            }

            var emitted = 0
            var attempts = 0
            // Decisive stratum first (it's usually the scarcer one), then draws.
            for (wantDecisive, quota) in [(true, bDec), (false, bDraw)] {
                var filled = 0
                while filled < quota {
                    attempts += 1
                    if attempts > attemptBudget {
                        attemptBudgetHit = true
                        while emitted < sampleCount {
                            let srcIndex = Int.random(in: 0..<held)
                            emit(emitted, srcIndex)
                            tallyOutcome(srcIndex)
                            achievedSumGameLength += Int(gameLengthStorage[srcIndex])
                            perGameCount[workerGameIdStorage[srcIndex], default: 0] += 1
                            emitted += 1
                        }
                        break
                    }
                    let srcIndex = Int.random(in: 0..<held)
                    let isDraw = outcomeStorage[srcIndex] == 0.0
                    if isDraw == wantDecisive { continue }   // wrong stratum
                    if !tiltAccepts(srcIndex) { continue }
                    let gid = workerGameIdStorage[srcIndex]
                    let c = perGameCount[gid] ?? 0
                    if c >= cap { continue }
                    perGameCount[gid] = c + 1
                    emit(emitted, srcIndex)
                    tallyOutcome(srcIndex)
                    achievedSumGameLength += Int(gameLengthStorage[srcIndex])
                    emitted += 1
                    filled += 1
                }
                if attemptBudgetHit { break }
            }
            // Under-fill guard. After the K-aware pre-clamp at the top
            // of this block, `bDec + bDraw == sampleCount` whenever at
            // least one stratum has slack to absorb the other's
            // deficit (the common case). The jointly-pathological
            // combo — *both* reachable ceilings clamping at once —
            // leaves `bDec + bDraw < sampleCount`, the strata fill
            // cleanly to their (small) quotas, and emitted ends up
            // below sampleCount. The function's contract is to fill
            // the caller's dst buffers in full; we cannot return with
            // uninitialised slots. Drop the K cap on the spillover
            // (same semantics as the budget-exhaustion fallback inside
            // the rejection loop) and mark `attemptBudgetHit = true`
            // so the [SAMPLER] log and the popover banner both fire —
            // the operator must see that the caps couldn't be honoured
            // for the full batch. Unreachable at production buffer
            // sizes; reached only in stress tests and pathological
            // hand-tuned configurations (e.g. very low K with very few
            // resident games and a large batch).
            if emitted < sampleCount {
                attemptBudgetHit = true
                while emitted < sampleCount {
                    let srcIndex = Int.random(in: 0..<held)
                    emit(emitted, srcIndex)
                    tallyOutcome(srcIndex)
                    achievedSumGameLength += Int(gameLengthStorage[srcIndex])
                    perGameCount[workerGameIdStorage[srcIndex], default: 0] += 1
                    emitted += 1
                }
            }
            for (_, c) in perGameCount where c > achievedMaxPerGame { achievedMaxPerGame = c }
            _lastSamplingResult = SamplingResult(
                didSample: true, wasConstrainedPath: true,
                constraints: constraints, batchSize: sampleCount,
                requestedDrawCount: requestedDrawCount,
                achievedWinCount: achievedWinCount,
                achievedDrawCount: achievedDrawCount,
                achievedLossCount: achievedLossCount,
                achievedMaxPerGame: achievedMaxPerGame,
                distinctGamesInBatch: perGameCount.count,
                achievedSumGameLength: achievedSumGameLength,
                lengthTargetInfeasible: tiltSolve.infeasible,
                shortestResidentLength: tiltSolve.shortestResidentLength,
                attemptBudgetHit: attemptBudgetHit,
                achievedBucketCounts: achievedBucketCounts,
                requestedBucketCounts: zeroBuckets
            )
            return true
        }
    }

    /// Solve for β ≥ 0 such that the position-weighted mean resident game
    /// length, exponentially down-weighted by `exp(-β·max(0, L−T))`,
    /// equals `target` plies. The tilted mean is monotone-decreasing in
    /// β, and its limit at β→∞ is the shortest resident game length
    /// (all weight collapses onto the shortest games). So:
    ///
    ///   - target ≥ untilted mean ⇒ β = 0, no tilt needed.
    ///   - shortest_resident < target < untilted mean ⇒ bracket and bisect.
    ///   - target ≤ shortest_resident (with any longer games present)
    ///     ⇒ infeasible: the tilted mean cannot fall below the shortest
    ///     game length. Clamp β to a large value (effectively reject
    ///     everything but the shortest games) and report
    ///     `infeasible: true` so the trainer can surface a `[SAMPLER]`
    ///     line. Avoids the prior bracket-grow loop that silently
    ///     runaway-clamped on this case.
    ///
    /// Must be called while holding `lock`.
    private func solveLengthTiltBeta(target: Int) -> (beta: Double, infeasible: Bool, shortestResidentLength: Int) {
        if residentLengthHistogram.isEmpty { return (0, false, 0) }
        let t = Double(target)
        // Snapshot histogram into arrays for the tight inner loop.
        let lens = residentLengthHistogram.keys.map { Double($0) }
        let wts = residentLengthHistogram.keys.map { Double(residentLengthHistogram[$0] ?? 0) }
        let shortestLen = Int(lens.min() ?? 0)
        func tiltedMean(_ beta: Double) -> Double {
            var num = 0.0, den = 0.0
            for k in lens.indices {
                let len = lens[k]
                let factor = len > t ? exp(-beta * (len - t)) : 1.0
                num += wts[k] * factor * len
                den += wts[k] * factor
            }
            return den > 0 ? num / den : 0
        }
        if tiltedMean(0) <= t { return (0, false, shortestLen) }   // already short enough
        // Achievability check: the position-weighted mean at β→∞
        // converges to the shortest resident length. If the target is
        // at or below that limit (and there exist longer games), no β
        // can bring the mean down to the target. Reaching equality
        // exactly requires β = ∞; treat that as infeasible too so the
        // operator sees a clear message.
        if Double(shortestLen) >= t {
            return (1.0e18, true, shortestLen)
        }
        // Bracket: grow β until the tilted mean drops to/below target.
        // With the infeasibility guard above, this terminates in
        // O(log(target / minMargin)) steps in practice — no runaway.
        var hi = 1.0e-3
        var grows = 0
        while tiltedMean(hi) > t {
            hi *= 4
            grows += 1
            if grows > 40 { return (hi, false, shortestLen) }   // safety clamp
        }
        var lo = 0.0
        for _ in 0..<60 {   // bisection
            let mid = 0.5 * (lo + hi)
            if tiltedMean(mid) > t { lo = mid } else { hi = mid }
        }
        return (0.5 * (lo + hi), false, shortestLen)
    }

    // MARK: - Per-batch stats summarizer

    /// Compact summary of one sampled minibatch — the per-batch
    /// observability output produced for the `[BATCH-STATS]` log line
    /// every `batch_stats_interval` SGD steps.
    ///
    /// All histograms are dictionaries keyed by a short label
    /// (`"op"`, `"early"`, `"mid"`, `"late"`, `"end"` for ply phase;
    /// `"short"`, `"med"`, `"long"` for game length; `"W"`, `"D"`,
    /// `"L"` for outcome; etc.). Counts sum to `batchSize` for any
    /// partition-style histogram.
    ///
    /// The histograms describe a *post-sampling-constraints* batch when
    /// `samplingConstraintsApplied == true` — i.e. the per-batch
    /// `maxPerGame` / `maxDrawPercent` / `targetMeanGameLengthPlies`
    /// were active during the sample. Consumers reading the JSON for
    /// post-run analysis should branch on `sampling_constraints.applied`
    /// before comparing histograms across runs.
    public struct BatchStatsSummary: Sendable {
        public let step: Int
        public let batchSize: Int
        public let uniqueCount: Int
        public let uniquePct: Double
        public let dupMax: Int
        /// `dup_distribution[k]` = number of distinct hashes in this
        /// batch that occurred exactly `k` times. `Σ k · dup_dist[k] = batchSize`,
        /// `Σ dup_dist[k] = uniqueCount`. This shape (count of distinct
        /// hashes per multiplicity, not weighted by occurrences) matches
        /// the original observability spec.
        public let dupDistribution: [Int: Int]
        public let phaseByPlyHistogram: [String: Int]
        public let phaseByMaterialHistogram: [String: Int]
        public let gameLengthHistogram: [String: Int]
        public let samplingTauHistogram: [String: Int]
        public let workerIdHistogram: [String: Int]
        public let outcomeHistogram: [String: Int]
        public let phaseByPlyXOutcomeHistogram: [String: Int]
        public let bufferUniquePositions: Int     // global, dict.count
        public let bufferStoredCount: Int
        /// Composition constraints in effect when this batch was
        /// sampled. Captioning so a reader of the JSON line (or of
        /// `result.json`) can tell whether the histograms reflect a
        /// constrained-path batch.
        public let samplingConstraints: SamplingConstraints
        /// Mirrors `lastSamplingResult().wasConstrainedPath` — `false`
        /// when the sample call took the bit-for-bit uniform fast path
        /// (constraints at their no-op settings), `true` when the
        /// composition controls actively shaped the batch.
        public let samplingConstraintsApplied: Bool
        /// Per-bucket counts in the batch this summary describes,
        /// indexed by `ReplayBufferAnalyzer.materialBuckets`. Sourced
        /// from the matching `SamplingResult.achievedBucketCounts`
        /// captured under the buffer's lock alongside the other
        /// constraint fields. Surfaced on the `[BATCH-STATS]` JSON
        /// line as `bucket_mix` / `bucket_mix_pct` so post-run
        /// analysis can confirm phase-stratification took effect (when
        /// the toggle was on) or audit the natural per-phase mix
        /// (when off).
        public let materialBucketCounts: [Int]

        /// Render to a single-line JSON string for the `[BATCH-STATS]`
        /// log entry. Counts AND fractions of every histogram are
        /// included so post-run analysis from log files (rather than
        /// result.json) doesn't lose information. Stable key ordering
        /// for grep/parse.
        public func jsonLine() -> String {
            func encodeIntDict(_ d: [String: Int]) -> String {
                let pairs = d.sorted { $0.key < $1.key }.map { "\"\($0.key)\":\($0.value)" }
                return "{" + pairs.joined(separator: ",") + "}"
            }
            func encodePctDict(_ d: [String: Int], denom: Double) -> String {
                guard denom > 0 else { return "{}" }
                let pairs = d.sorted { $0.key < $1.key }.map {
                    String(format: "\"%@\":%.4f", $0.key as NSString, Double($0.value) / denom)
                }
                return "{" + pairs.joined(separator: ",") + "}"
            }
            func encodeIntKeyDict(_ d: [Int: Int]) -> String {
                let pairs = d.sorted { $0.key < $1.key }.map { "\"\($0.key)\":\($0.value)" }
                return "{" + pairs.joined(separator: ",") + "}"
            }
            func encodeIntKeyPctDict(_ d: [Int: Int], denom: Double) -> String {
                guard denom > 0 else { return "{}" }
                let pairs = d.sorted { $0.key < $1.key }.map {
                    String(format: "\"%d\":%.4f", $0.key, Double($0.value) / denom)
                }
                return "{" + pairs.joined(separator: ",") + "}"
            }
            let bs = Double(batchSize)
            let uniq = Double(uniqueCount)
            let stored = Double(bufferStoredCount)
            var out = "{"
            out += "\"step\":\(step),"
            out += "\"batch_size\":\(batchSize),"
            // Caption: the constraints that produced this batch. Placed
            // early so a grep on `[BATCH-STATS]` lines can branch on
            // `applied` before reading the histograms.
            out += "\"sampling_constraints\":{"
            out += "\"applied\":\(samplingConstraintsApplied ? "true" : "false"),"
            out += "\"max_per_game\":\(samplingConstraints.maxPerGame),"
            out += "\"max_draw_pct\":\(samplingConstraints.maxDrawPercent),"
            out += "\"target_length\":\(samplingConstraints.targetMeanGameLengthPlies),"
            // Material-bucket stratification flag: `true` means the
            // batch was drawn with a per-bucket target distribution
            // (V1: balanced across 0–4 / 5–8 / 9–14 / 15–22 non-pawn
            // pieces), bypassing the W/D/L cap + per-game K cap. When
            // both `applied:true` and `stratify_by_material:true`, the
            // `bucket_mix` field below reflects the bucket-stratified
            // batch; when `false`, it reflects the natural mix of the
            // (uniform or W/D/L-tilted) batch.
            out += "\"stratify_by_material\":\(samplingConstraints.materialBucketWeights != nil ? "true" : "false")"
            out += "},"
            out += "\"unique_count\":\(uniqueCount),"
            out += String(format: "\"unique_pct\":%.4f,", uniquePct)
            out += "\"dup_max\":\(dupMax),"
            out += "\"dup_distribution\":\(encodeIntKeyDict(dupDistribution)),"
            out += "\"dup_distribution_pct\":\(encodeIntKeyPctDict(dupDistribution, denom: uniq)),"
            out += "\"phase_by_ply\":\(encodeIntDict(phaseByPlyHistogram)),"
            out += "\"phase_by_ply_pct\":\(encodePctDict(phaseByPlyHistogram, denom: bs)),"
            out += "\"phase_by_material\":\(encodeIntDict(phaseByMaterialHistogram)),"
            out += "\"phase_by_material_pct\":\(encodePctDict(phaseByMaterialHistogram, denom: bs)),"
            // Per-bucket position counts in the analyzer's bucket
            // partition (0–4 / 5–8 / 9–14 / 15–22 / 23–30 non-pawn
            // pieces). Keyed by the bucket label string so downstream
            // analysis joins straight against the analyzer's output
            // dictionaries. Distinct from `phase_by_material` above,
            // which uses a coarser open/early/mid/late/end partition
            // — the analyzer-aligned buckets are the unit the
            // stratifier targets.
            do {
                var bucketCountsDict: [String: Int] = [:]
                for (i, def) in ReplayBufferAnalyzer.materialBuckets.enumerated()
                    where i < materialBucketCounts.count {
                    bucketCountsDict[def.label] = materialBucketCounts[i]
                }
                out += "\"bucket_mix\":\(encodeIntDict(bucketCountsDict)),"
                out += "\"bucket_mix_pct\":\(encodePctDict(bucketCountsDict, denom: bs)),"
            }
            out += "\"game_length\":\(encodeIntDict(gameLengthHistogram)),"
            out += "\"game_length_pct\":\(encodePctDict(gameLengthHistogram, denom: bs)),"
            out += "\"sampling_tau\":\(encodeIntDict(samplingTauHistogram)),"
            out += "\"sampling_tau_pct\":\(encodePctDict(samplingTauHistogram, denom: bs)),"
            out += "\"worker_id\":\(encodeIntDict(workerIdHistogram)),"
            out += "\"worker_id_pct\":\(encodePctDict(workerIdHistogram, denom: bs)),"
            out += "\"outcome\":\(encodeIntDict(outcomeHistogram)),"
            out += "\"outcome_pct\":\(encodePctDict(outcomeHistogram, denom: bs)),"
            out += "\"phase_by_ply_x_outcome\":\(encodeIntDict(phaseByPlyXOutcomeHistogram)),"
            out += "\"phase_by_ply_x_outcome_pct\":\(encodePctDict(phaseByPlyXOutcomeHistogram, denom: bs)),"
            out += "\"buffer_unique\":\(bufferUniquePositions),"
            out += "\"buffer_stored\":\(bufferStoredCount),"
            out += String(format: "\"buffer_unique_pct\":%.4f", stored > 0 ? Double(bufferUniquePositions) / stored : 0)
            out += "}"
            return out
        }
    }

    /// Compute a `BatchStatsSummary` from a freshly-sampled minibatch.
    /// Caller must pass the same buffers it just received from
    /// `sample(...)`, with the metadata buffers populated (i.e. the
    /// non-nil overload).
    ///
    /// O(N) where N = `batchSize`; runs on the trainer thread, ~1ms
    /// at batch=4096. Snapshot the buffer's `uniquePositionCount` and
    /// `storedCount` once under the lock so the summary's "buffer
    /// state" is a single consistent view.
    func computeBatchStats(
        step: Int,
        batchSize: Int,
        plies: UnsafePointer<UInt16>,
        gameLengths: UnsafePointer<UInt16>,
        taus: UnsafePointer<Float>,
        hashes: UnsafePointer<UInt64>,
        workerGameIds: UnsafePointer<UInt32>,
        materialCounts: UnsafePointer<UInt8>,
        zs: UnsafePointer<Float>
    ) -> BatchStatsSummary {
        // Per-batch dup count by hash.
        var perHashCount: [UInt64: Int] = [:]
        perHashCount.reserveCapacity(batchSize)
        for i in 0..<batchSize {
            perHashCount[hashes[i], default: 0] += 1
        }
        let uniqueCount = perHashCount.count
        let uniquePct = batchSize > 0 ? Double(uniqueCount) / Double(batchSize) : 0
        // dup_distribution[k] = number of DISTINCT hashes that
        // appeared k times in the batch. Σ k * dup_dist[k] = batchSize;
        // Σ dup_dist[k] = uniqueCount. Different from "weighted by
        // occurrences" — this shape is what the spec calls for and
        // is more directly interpretable: "1 hash appeared 12 times"
        // rather than "12 batch slots are part of a 12-count group."
        var dupMax = 0
        var dupDist: [Int: Int] = [:]
        for (_, c) in perHashCount {
            if c > dupMax { dupMax = c }
            dupDist[c, default: 0] += 1
        }

        // Phase by ply. Buckets sized so each holds a meaningful
        // share of typical self-play batches. Inclusive on upper
        // bound:
        //   open: ≤ 15    early: 16–35    mid: 36–75
        //   late: 76–125    end: 126+
        var phaseByPly: [String: Int] = [
            "open": 0, "early": 0, "mid": 0, "late": 0, "end": 0,
        ]
        // Phase by material (non-pawn piece count). open ≥ 14
        // (full-board, both queens still on), early 12–13, mid 8–11,
        // late 4–7, end ≤ 3 (king-and-pawn endgame territory). Cells
        // populated with zero so absent buckets show explicitly in
        // the JSON rather than being missing keys.
        var phaseByMaterial: [String: Int] = ["open": 0, "early": 0, "mid": 0, "late": 0, "end": 0]
        // Game length buckets:
        //   short: ≤ 50    medium: 51–150    long: 151–300
        //   very_long: 301+
        var gameLenHist: [String: Int] = [
            "short": 0, "medium": 0, "long": 0, "very_long": 0,
        ]
        var samplingTauHist: [String: Int] = [:]
        var workerIdHist: [String: Int] = [:]
        var outcomeHist: [String: Int] = ["W": 0, "D": 0, "L": 0]
        var phaseByPlyXOutcome: [String: Int] = [:]

        for i in 0..<batchSize {
            let ply = Int(plies[i])
            let phasePlyLabel: String
            if ply <= 15 { phasePlyLabel = "open" }
            else if ply <= 35 { phasePlyLabel = "early" }
            else if ply <= 75 { phasePlyLabel = "mid" }
            else if ply <= 125 { phasePlyLabel = "late" }
            else { phasePlyLabel = "end" }
            phaseByPly[phasePlyLabel, default: 0] += 1

            let mat = Int(materialCounts[i])
            let phaseMatLabel: String
            if mat >= 14 { phaseMatLabel = "open" }
            else if mat >= 12 { phaseMatLabel = "early" }
            else if mat >= 8 { phaseMatLabel = "mid" }
            else if mat >= 4 { phaseMatLabel = "late" }
            else { phaseMatLabel = "end" }
            phaseByMaterial[phaseMatLabel, default: 0] += 1

            let gameLen = Int(gameLengths[i])
            let lenLabel: String
            if gameLen <= 50 { lenLabel = "short" }
            else if gameLen <= 150 { lenLabel = "medium" }
            else if gameLen <= 300 { lenLabel = "long" }
            else { lenLabel = "very_long" }
            gameLenHist[lenLabel, default: 0] += 1

            let tauKey = String(format: "%.1f", taus[i])
            samplingTauHist[tauKey, default: 0] += 1

            let (workerId, _) = Self.unpackWorkerGameId(workerGameIds[i])
            workerIdHist["\(workerId)", default: 0] += 1

            let z = zs[i]
            let outcomeLabel: String
            if z > 0.5 { outcomeLabel = "W" }
            else if z < -0.5 { outcomeLabel = "L" }
            else { outcomeLabel = "D" }
            outcomeHist[outcomeLabel, default: 0] += 1

            phaseByPlyXOutcome["\(phasePlyLabel)_\(outcomeLabel)", default: 0] += 1
        }

        // Snapshot buffer-global counters AND the constraints/path
        // taken by the most-recent `sample(...)` under the lock so the
        // summary reflects a single consistent view. The caller is
        // expected to have just called `sample(...)` on the trainer
        // queue, so `_lastSamplingResult` describes the batch whose
        // histograms we're summarising.
        let (uniqBuf, storedBuf, lastResult) = lock.withLock {
            (hashStats.count, storedCount, _lastSamplingResult)
        }

        return BatchStatsSummary(
            step: step,
            batchSize: batchSize,
            uniqueCount: uniqueCount,
            uniquePct: uniquePct,
            dupMax: dupMax,
            dupDistribution: dupDist,
            phaseByPlyHistogram: phaseByPly,
            phaseByMaterialHistogram: phaseByMaterial,
            gameLengthHistogram: gameLenHist,
            samplingTauHistogram: samplingTauHist,
            workerIdHistogram: workerIdHist,
            outcomeHistogram: outcomeHist,
            phaseByPlyXOutcomeHistogram: phaseByPlyXOutcome,
            bufferUniquePositions: uniqBuf,
            bufferStoredCount: storedBuf,
            samplingConstraints: lastResult.constraints,
            samplingConstraintsApplied: lastResult.wasConstrainedPath,
            materialBucketCounts: lastResult.achievedBucketCounts
        )
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
    /// Current format is v7:
    ///   - Header: 8-byte magic + 4-byte version + 4-byte pad + 5 × Int64
    ///     (floatsPerBoard, capacity, storedCount, writeIndex,
    ///     totalPositionsAdded).
    ///   - Body (oldest-first, `storedCount` entries each):
    ///     1. boards (`floatsPerBoard` × Float32)
    ///     2. moves (Int32)
    ///     3. outcomes (Float32)
    ///     4. plyIndices (UInt16) ← v5
    ///     5. gameLengths (UInt16) ← v5
    ///     6. samplingTaus (Float32) ← v5
    ///     7. stateHashes (UInt64) ← v5
    ///     8. workerGameIds (UInt32) ← v5
    ///     9. materialCounts (UInt8) ← v6 new
    ///   - Trailer: 32-byte SHA-256 digest over every preceding byte.
    ///
    /// v7 dropped the per-slot `vBaselines` (Float32) column that used
    /// to sit between `outcomes` and `plyIndices`: the W/D/L value-head
    /// rewrite made the play-time-frozen baseline dead — the trainer now
    /// recomputes the policy-gradient baseline from a fresh forward pass
    /// every step, so there is nothing left to persist.
    ///
    /// Older replay-buffer versions are rejected rather than loaded
    /// with synthesized metadata. Session resume should either restore
    /// the exact saved state or fail loudly.
    private static let fileVersion: UInt32 = 7
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
    /// file. Thread-safe — holds the buffer's lock for the duration
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
        try lock.withLock {
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
        let floatsPerBoard = self.floatsPerBoard
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

            // v5 metadata sections — present since v5. Each is one
            // slot wide (broadcast fields are stored per-slot for
            // simplicity; the file shape remains a flat per-slot
            // record).
            let plyChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<UInt16>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: plyChunk,
                elementBytes: MemoryLayout<UInt16>.size,
                basePtr: UnsafeRawPointer(plyIndexStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<UInt16>.size
            )
            let lenChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<UInt16>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: lenChunk,
                elementBytes: MemoryLayout<UInt16>.size,
                basePtr: UnsafeRawPointer(gameLengthStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<UInt16>.size
            )
            let tauChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<Float>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: tauChunk,
                elementBytes: MemoryLayout<Float>.size,
                basePtr: UnsafeRawPointer(samplingTauStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<Float>.size
            )
            let hashChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<UInt64>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: hashChunk,
                elementBytes: MemoryLayout<UInt64>.size,
                basePtr: UnsafeRawPointer(stateHashStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<UInt64>.size
            )
            let widChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<UInt32>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: widChunk,
                elementBytes: MemoryLayout<UInt32>.size,
                basePtr: UnsafeRawPointer(workerGameIdStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<UInt32>.size
            )
            let matChunk = max(1, Self.persistenceChunkBytes / MemoryLayout<UInt8>.size)
            try writeRange(
                handle: handle,
                hasher: &hasher,
                start: startIndex,
                total: stored,
                capacity: cap,
                chunkPositions: matChunk,
                elementBytes: MemoryLayout<UInt8>.size,
                basePtr: UnsafeRawPointer(materialCountStorage),
                elementsPerSlot: 1,
                slotSize: MemoryLayout<UInt8>.size
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
    /// `slotSize` bytes per ring slot. Caller must already hold the
    /// buffer's `lock` (e.g. from inside `_writeLocked`).
    ///
    /// Every byte written is also fed into `hasher` — the streaming
    /// SHA-256 hasher whose final digest becomes the file's integrity
    /// trailer. Passing the hasher inout (rather than capturing it in
    /// an escaping closure) lets the single hasher object accumulate
    /// across all column-section writes from a single `_writeLocked`
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
    /// Validation order (each step throws a specific error on
    /// failure and aborts; no field is trusted until all preceding
    /// checks pass):
    ///
    /// 1. File opens and header can be fully read (`truncatedHeader`).
    /// 2. Magic matches "DCMRPBUF" (`badMagic`).
    /// 3. `fileVersion` matches the current format (`unsupportedVersion`).
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
    /// state (taking the buffer's `lock`, resetting counters,
    /// re-seeking to the header end, and reading the nine column
    /// sections into the ring storage).
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

        guard Int(fpbFile) == self.floatsPerBoard else {
            throw PersistenceError.incompatibleBoardSize(
                expected: self.floatsPerBoard,
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
        let basePerSlotBytes: Int64 = fpbFile * Int64(MemoryLayout<Float>.size)
            + Int64(MemoryLayout<Int32>.size)       // moves
            + Int64(MemoryLayout<Float>.size)       // outcomes
        let metadataPerSlotBytes: Int64 = Int64(MemoryLayout<UInt16>.size)   // plyIndex
            + Int64(MemoryLayout<UInt16>.size)      // gameLength
            + Int64(MemoryLayout<Float>.size)       // samplingTau
            + Int64(MemoryLayout<UInt64>.size)      // stateHash
            + Int64(MemoryLayout<UInt32>.size)      // workerGameId
            + Int64(MemoryLayout<UInt8>.size)       // materialCount
        let perSlotBytes = basePerSlotBytes + metadataPerSlotBytes
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

        try lock.withLock {
            // Reset live state before filling.
            storedCount = 0
            writeIndex = 0
            _totalPositionsAdded = 0
            hashStats.removeAll(keepingCapacity: true)
            resetCompositionAggregates()

            if fileStored == 0 {
                _totalPositionsAdded = Int(ttlFile)
                return
            }

            let floatsPerBoard = self.floatsPerBoard
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

            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<UInt16>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(plyIndexStorage),
                slotBytes: MemoryLayout<UInt16>.size,
                count: target
            )
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<UInt16>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(gameLengthStorage),
                slotBytes: MemoryLayout<UInt16>.size,
                count: target
            )
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<Float>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(samplingTauStorage),
                slotBytes: MemoryLayout<Float>.size,
                count: target
            )
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<UInt64>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(stateHashStorage),
                slotBytes: MemoryLayout<UInt64>.size,
                count: target
            )
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<UInt32>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(workerGameIdStorage),
                slotBytes: MemoryLayout<UInt32>.size,
                count: target
            )
            if skip > 0 {
                try seekForward(handle: handle, bytes: skip * MemoryLayout<UInt8>.size)
            }
            try readContiguous(
                handle: handle,
                into: UnsafeMutableRawPointer(materialCountStorage),
                slotBytes: MemoryLayout<UInt8>.size,
                count: target
            )

            storedCount = target
            writeIndex = (target == capacity) ? 0 : target
            _totalPositionsAdded = Int(ttlFile)

            // Rebuild the hash dict + composition aggregates + material
            // bucket index from the restored columns. The bucket index
            // is purely derived from `materialCountStorage`, which v6+
            // .dcmsession files already carry — so no migration shim is
            // needed for sessions saved before this feature landed.
            for slot in 0..<target {
                let h = stateHashStorage[slot]
                let outcome = outcomeStorage[slot]
                let isWin = outcome > 0.5
                let isLoss = outcome < -0.5
                incrementHashStat(hash: h, isWin: isWin, isLoss: isLoss)
                incrementCompositionAggregates(
                    gameLength: gameLengthStorage[slot],
                    packedId: workerGameIdStorage[slot],
                    isWin: isWin, isLoss: isLoss, count: 1
                )
                let bucket = Self.materialBucketIndex(for: materialCountStorage[slot])
                materialBucketSlots[bucket].insert(slot)
            }
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
