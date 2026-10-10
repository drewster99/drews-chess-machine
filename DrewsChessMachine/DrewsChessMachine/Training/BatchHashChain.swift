import CryptoKit
import Foundation

/// A SHA-256 of every training batch's exact bytes, chained within windows of
/// `windowSteps` trainer steps, so a resumed run can prove it trained on the
/// same batches as the original instead of assuming it (GPU fault forensics
/// plan, Part B; owner request 2026-10-09).
///
/// What is hashed: the batch the training step consumes, in batch order —
/// boards, played-move indices, and outcomes after the draw-penalty rewrite —
/// the bytes the GPU trains on, apart from the value baseline (which depends
/// on the weights, not the batch).
///
/// Cost: about 12 ms of one CPU core per 4,096-position batch (~30 MB),
/// measured on this Mac beside three training runs. It runs on a utility
/// queue of its own, from copies the step already makes, so the training step
/// never waits for it; `entry(forTrainerStep:)` waits only when a log line
/// asks for a hash that isn't done yet.
///
/// The chain: `chain = SHA-256(previous chain ‖ batch hash)`, restarting at
/// the first step of every window from `SHA-256("dcm-batch-chain-v1" ‖ window
/// start)`. Checkpoints land on window boundaries (multiples of 1,000), so an
/// exact resume computes the same chain values as the original run without
/// storing anything: a matching chain at any step proves every batch of that
/// window up to the step was byte-identical. A chain that didn't see every
/// step of its window since the window began — a segment that started
/// mid-window, or a GUI promotion that rewound the trainer's clock — is
/// `partial` until the next window starts.
///
/// What a match means per path: in corpus replay the feed is deterministic,
/// so a resume that matches the original's chain trained on exactly the same
/// batches. In GUI self-play and train-vs-UCI a resumed buffer refills from
/// new games, so the hashes differ by design; within one process they still
/// identify each batch (a crash dump names its batch by hash).
final class BatchHashChain: @unchecked Sendable {
    /// Chain window length in trainer steps; equal to the checkpoint
    /// interval so every exact resume starts on a window boundary.
    static let windowSteps = 1_000
    /// How many recent entries are kept (for crash dumps).
    static let retainedEntries = 1_000

    struct Entry: Sendable, Equatable {
        let trainerStep: Int
        /// SHA-256 of the batch, 64 hex digits.
        let batchHash: String
        /// The window's chain through this step, 64 hex digits; nil when the
        /// chain is partial (didn't see every step of the window).
        let chain: String?

        /// `[BATCH-HASH] trainerStep=N batchHash=<16 hex> batchChain=<16 hex | partial>`
        var logLine: String {
            "[BATCH-HASH] trainerStep=\(trainerStep) batchHash=\(batchHash.prefix(16)) "
                + "batchChain=\(chain.map { String($0.prefix(16)) } ?? "partial")"
        }
    }

    /// At most this many batches wait to be hashed. Each holds a copy of its
    /// boards (~30 MB at batch 4,096), so a starved hash queue must not grow
    /// without bound; at the limit `submit` waits for the oldest, which slows
    /// the trainer's queue rather than the machine's memory.
    static let maximumBacklog = 8

    private let queue = DispatchQueue(label: "drewschessmachine.batch-hash", qos: .utility)
    private let backlog = DispatchSemaphore(value: BatchHashChain.maximumBacklog)
    // Queue-confined.
    private var chainState: Data?
    private var chainStep: Int?
    private var recent: [Entry] = []

    init() {}

    /// Queues the hash of the batch trained at `trainerStep` (the step's
    /// number once it completes). Returns at once.
    func submit(trainerStep: Int, boards: [Float], moves: [Int32], outcomes: [Float]) {
        backlog.wait()
        queue.async {
            let batchHash = Self.batchHash(boards: boards, moves: moves, outcomes: outcomes)
            self.fold(trainerStep: trainerStep, batchHash: batchHash)
            self.backlog.signal()
        }
    }

    /// Ends the current chain: the next step's chain is `partial` unless it
    /// starts a window. Called when the trainer's weights and clock are
    /// replaced (a load or a resume), so a batch after the replacement never
    /// extends a chain of batches trained on other weights.
    func reset() {
        queue.async {
            self.chainState = nil
            self.chainStep = nil
        }
    }

    /// The entry for `trainerStep`, waiting for its hash if it is queued;
    /// nil when that step's batch was never submitted (or is no longer kept).
    func entry(forTrainerStep trainerStep: Int) async -> Entry? {
        await withCheckedContinuation { (continuation: CheckedContinuation<Entry?, Never>) in
            queue.async {
                continuation.resume(returning: self.recent.last { $0.trainerStep == trainerStep })
            }
        }
    }

    /// The kept entries, oldest first, once every queued hash is done.
    func recentEntries() async -> [Entry] {
        await withCheckedContinuation { (continuation: CheckedContinuation<[Entry], Never>) in
            queue.async {
                continuation.resume(returning: self.recent)
            }
        }
    }

    /// Whether `trainerStep` gets a `[BATCH-HASH]` line: every 100 trainer
    /// steps, so two runs' lines always overlap whatever their time-based
    /// step lines do.
    static func logsLine(atTrainerStep trainerStep: Int) -> Bool {
        trainerStep > 0 && trainerStep % 100 == 0
    }

    // MARK: - Pure parts (tested directly)

    /// SHA-256 over a version tag, then the boards', moves' and outcomes'
    /// bytes, as 64 hex digits.
    static func batchHash(boards: [Float], moves: [Int32], outcomes: [Float]) -> String {
        var hasher = SHA256()
        hasher.update(data: Data("dcm-batch-v1".utf8))
        boards.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        moves.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        outcomes.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        return hex(hasher.finalize())
    }

    /// The chain value a window starts from.
    static func windowSeed(windowStart: Int) -> Data {
        var hasher = SHA256()
        hasher.update(data: Data("dcm-batch-chain-v1".utf8))
        withUnsafeBytes(of: Int64(windowStart).littleEndian) { hasher.update(bufferPointer: $0) }
        return Data(hasher.finalize())
    }

    /// `SHA-256(previous ‖ batch hash bytes)`.
    static func extend(_ previous: Data, with batchHash: String) -> Data {
        var hasher = SHA256()
        hasher.update(data: previous)
        hasher.update(data: Data(batchHash.utf8))
        return Data(hasher.finalize())
    }

    // MARK: - Private (on `queue`)

    private func fold(trainerStep: Int, batchHash: String) {
        let windowStart = ((trainerStep - 1) / Self.windowSteps) * Self.windowSteps
        let chain: Data?
        if trainerStep - 1 == windowStart {
            chain = Self.extend(Self.windowSeed(windowStart: windowStart), with: batchHash)
        } else if let chainState, chainStep == trainerStep - 1 {
            chain = Self.extend(chainState, with: batchHash)
        } else {
            chain = nil
        }
        chainState = chain
        chainStep = trainerStep
        recent.append(Entry(trainerStep: trainerStep, batchHash: batchHash, chain: chain.map { Self.hex($0) }))
        if recent.count > Self.retainedEntries {
            recent.removeFirst(recent.count - Self.retainedEntries)
        }
    }

    private static func hex<D: Sequence>(_ bytes: D) -> String where D.Element == UInt8 {
        bytes.map { String(format: "%02x", $0) }.joined()
    }
}
