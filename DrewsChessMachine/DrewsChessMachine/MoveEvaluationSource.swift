import Foundation

/// Source of (policy, value) predictions for a single encoded board.
///
/// `MPSChessPlayer` used to own a `ChessMPSNetwork` directly and call
/// `network.evaluate(board:)` inline on every move. That hard-wired the
/// player to exactly one single-position forward pass per ply, with no
/// room for the self-play batcher to coalesce N games' submissions into
/// one batched `graph.run`. This protocol breaks that coupling:
///
/// - `DirectMoveEvaluationSource` wraps a `ChessMPSNetwork` and calls
///   `evaluate(board:)` per request. Used by arena (sequential single-
///   game tournaments) and the Play Game screen — their per-move latency
///   is fine on the single-position path.
/// - `BatchedMoveEvaluationSource` is the self-play barrier batcher: N
///   slot tasks park at their per-ply `evaluate` call; the N-th
///   submission fires one batched `graph.run`; all N resume with their
///   own `(policy, value)` slice.
///
/// `encodedBoard` is 1152 floats (18 × 8 × 8 NCHW) produced by
/// `BoardEncoder.encode`, wrapped in a plain `[Float]` so the value
/// can cross the actor boundary into `BatchedMoveEvaluationSource`
/// (raw `UnsafeBufferPointer` is not `Sendable`). `MPSChessPlayer`
/// takes one `Array(...)` copy at the call site, which is equivalent
/// to the copy the batcher used to take internally — total copies are
/// unchanged, the copy just moves one actor boundary earlier.
///
/// Returned `policy` is a fresh Swift `[Float]` of 4096 raw logits —
/// the caller owns the bytes, so batchers can safely reuse their
/// readback scratch on the next batch without invalidating any
/// outstanding policy buffer. `value` is a single scalar in [-1, +1].
protocol MoveEvaluationSource: AnyObject, Sendable {
    func evaluate(
        encodedBoard: [Float]
    ) async throws -> (policy: [Float], value: Float)
}

/// Passes every request straight through to a single `ChessMPSNetwork`
/// on the calling thread. No queueing, no batching. Suitable for paths
/// that already serialize their own calls into the network — arena
/// tournaments and the Play Game screen both play one game at a time,
/// and `ChessMachine.runGameLoop` serializes the two players within a
/// game. Self-play no longer uses this path; it routes through
/// `BatchedMoveEvaluationSource`.
///
/// The underlying `ChessMPSNetwork`'s policy readback is non-reentrant,
/// so this source copies the 4096 policy logits into a fresh `[Float]`
/// before returning. That copy matches the `BatchedMoveEvaluationSource`
/// contract (caller owns the bytes) and costs one 16 KB allocation per
/// move — negligible at non-self-play cadence.
final class DirectMoveEvaluationSource: MoveEvaluationSource, @unchecked Sendable {
    let network: ChessMPSNetwork

    init(network: ChessMPSNetwork) {
        self.network = network
    }

    func evaluate(
        encodedBoard: [Float]
    ) async throws -> (policy: [Float], value: Float) {
        let (policyBuf, value) = try encodedBoard.withUnsafeBufferPointer { buf in
            try network.evaluate(board: buf)
        }
        let policy = Array(policyBuf)
        return (policy: policy, value: value)
    }
}
