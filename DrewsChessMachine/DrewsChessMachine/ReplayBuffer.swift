import Foundation

/// Thread-safe fixed-capacity ring of labeled self-play positions.
///
/// Self-play workers push whole games via `append(contentsOf:)` once each
/// game ends and outcomes have been labeled. The trainer pulls out minibatches
/// via `sample(count:)`. Both sides run on background tasks; access is
/// serialized by an `NSLock` so the buffer is safe to share across tasks.
///
/// Marked `@unchecked Sendable` for the same reason as `GameWatcher` and
/// `CancelBox`: the lock makes concurrent access safe even though the stored
/// elements contain non-Sendable value-type arrays.
final class ReplayBuffer: @unchecked Sendable {
    /// Maximum number of positions held. Older positions are overwritten
    /// in FIFO order once the buffer is full.
    let capacity: Int

    private let lock = NSLock()
    /// Backing ring. Grown by `append` up to `capacity`, then overwritten
    /// in place at `writeIndex`. Non-optional so `sample(count:)` never has
    /// to handle a "slot is empty" case — once an index is valid for read,
    /// the slot is guaranteed to hold a real position.
    private var storage: [TrainingPosition] = []
    private var writeIndex = 0

    init(capacity: Int) {
        precondition(capacity > 0, "Replay buffer capacity must be positive")
        self.capacity = capacity
        // One-shot reserve so the transition from "growing" to "overwriting"
        // doesn't trigger a final resize mid-self-play.
        storage.reserveCapacity(capacity)
    }

    /// Current number of positions stored (up to `capacity`).
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return storage.count
    }

    /// Append a batch of positions from one finished game. Outcomes must
    /// already be labeled — the buffer has no visibility into game results.
    /// When the buffer is full, new positions overwrite the oldest in
    /// FIFO order.
    func append(contentsOf positions: [TrainingPosition]) {
        guard !positions.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        for position in positions {
            if storage.count < capacity {
                storage.append(position)
            } else {
                storage[writeIndex] = position
            }
            writeIndex = (writeIndex + 1) % capacity
        }
    }

    /// Draw `sampleCount` positions uniformly at random (with replacement)
    /// from the positions currently held, packing them into a flat
    /// `TrainingBatch` ready for the trainer to upload. Returns nil if the
    /// buffer holds fewer than `sampleCount` positions — the caller should
    /// wait for more self-play to land before retrying.
    func sample(count sampleCount: Int) -> TrainingBatch? {
        precondition(sampleCount > 0, "Sample count must be positive")
        lock.lock()
        defer { lock.unlock() }
        let held = storage.count
        guard held >= sampleCount else { return nil }

        let floatsPerBoard = ChessNetwork.inputPlanes
            * ChessNetwork.boardSize
            * ChessNetwork.boardSize
        var boards = [Float](repeating: 0, count: sampleCount * floatsPerBoard)
        var moves = [Int32](repeating: 0, count: sampleCount)
        var zs = [Float](repeating: 0, count: sampleCount)

        // Pack into flat buffers under the lock so the trainer can upload
        // straight into MPSGraphTensorData without any per-position gather
        // on its side.
        boards.withUnsafeMutableBufferPointer { dstBuf in
            guard let dstBase = dstBuf.baseAddress else { return }
            for i in 0..<sampleCount {
                let idx = Int.random(in: 0..<held)
                let position = storage[idx]
                position.inputTensor.withUnsafeBufferPointer { srcBuf in
                    guard let srcBase = srcBuf.baseAddress else { return }
                    (dstBase + i * floatsPerBoard).update(
                        from: srcBase,
                        count: floatsPerBoard
                    )
                }
                moves[i] = Int32(position.policyIndex)
                zs[i] = position.outcome
            }
        }

        return TrainingBatch(
            boards: boards,
            moves: moves,
            zs: zs,
            batchSize: sampleCount
        )
    }
}
