import Foundation

/// `@unchecked Sendable` wrapper around an
/// `UnsafeMutableBufferPointer<Float>` so the destination buffer can
/// cross actor boundaries (`MPSChessPlayer` → `BatchedMoveEvaluationSource`)
/// and be captured into `@Sendable` consume closures dispatched onto
/// `ChessNetwork.executionQueue`. The Swift stdlib does not expose a
/// blanket `Sendable` conformance for `UnsafeMutableBufferPointer`
/// because mutating from multiple threads is generally unsafe.
///
/// In our use this wrapper IS safe because the
/// `MoveEvaluationSource.evaluate(encodedBoard:intoPolicy:)` contract
/// states:
/// - the pointer aliases caller-owned scratch (per-player, e.g.
///   `MPSChessPlayer.policyScratchPtr`),
/// - the underlying memory must outlive the await (the awaiting task
///   is suspended — its stored properties remain alive),
/// - the destination must not be aliased by any other concurrent
///   `evaluate` call.
///
/// Under those preconditions there is exactly one writer (the source's
/// internal evaluation pipeline) and exactly one subsequent reader
/// (the awaiting caller, after the continuation resume publishes the
/// writes via Swift's happens-before edge across resumption).
struct PolicyDestination: @unchecked Sendable {
    /// Non-optional pointer extracted from the wrapped buffer at init.
    /// `UnsafeMutableBufferPointer.baseAddress` is `Optional` (it may
    /// be nil when the buffer is the zero-element empty buffer); we
    /// require non-nil here so call sites don't need to unwrap on
    /// every per-slot copy.
    let pointer: UnsafeMutablePointer<Float>
    let count: Int

    init(_ buffer: UnsafeMutableBufferPointer<Float>) {
        guard let base = buffer.baseAddress else {
            preconditionFailure(
                "PolicyDestination requires a non-nil baseAddress; "
                + "got an empty UnsafeMutableBufferPointer with count=\(buffer.count). "
                + "Caller must allocate ChessNetwork.policySize floats."
            )
        }
        self.pointer = base
        self.count = buffer.count
    }
}

/// Source of (policy, value) predictions for a single encoded board.
///
/// `MPSChessPlayer` used to own a `ChessMPSNetwork` directly and call
/// `network.evaluate(board:)` inline on every move. That hard-wired the
/// player to exactly one single-position forward pass per ply, with no
/// room for the self-play batcher to coalesce N games' submissions into
/// one batched `graph.run`. This protocol breaks that coupling:
///
/// - `DirectMoveEvaluationSource` wraps a `ChessMPSNetwork` and delegates
///   to `network.evaluate(board:consume:)` per request. Used by arena
///   (sequential single-game tournaments) and the Play Game screen —
///   their per-move latency is fine on the single-position path.
/// - `BatchedMoveEvaluationSource` is the self-play barrier batcher: N
///   slot tasks park at their per-ply `evaluate` call; the N-th
///   submission fires one batched `graph.run`; all N resume after their
///   policy bytes have been written into their per-slot destinations.
///
/// `encodedBoard` is `BoardEncoder.tensorLength` floats (currently
/// `inputPlanes` × 8 × 8 = 1280 floats in NCHW layout) produced by
/// `BoardEncoder.encode`, wrapped in a plain `[Float]` so the value
/// can cross the actor boundary into `BatchedMoveEvaluationSource`
/// (raw `UnsafeBufferPointer` is not `Sendable`). `MPSChessPlayer`
/// takes one `Array(...)` copy at the call site.
///
/// Caller-owned destination contract: every `evaluate` call writes
/// exactly `ChessNetwork.policySize` floats into `intoPolicy` (provided
/// by the caller) and returns the scalar value head output. The
/// destination buffer:
///
/// - must have `count == ChessNetwork.policySize` (preconditioned),
/// - must alias memory the caller keeps alive until this `await` returns,
/// - must not be aliased by any other concurrent `evaluate` call.
///
/// Players (the canonical caller, `MPSChessPlayer`) pre-allocate a
/// single scratch for the player's lifetime and pass the same buffer
/// every ply; one ply runs at a time per player, so no aliasing is
/// possible. This replaces the previous `[Float]`-returning contract,
/// which paid a fresh policy allocation on every move.
protocol MoveEvaluationSource: AnyObject, Sendable {
    func evaluate(
        encodedBoard: [Float],
        intoPolicy: PolicyDestination
    ) async throws -> Float
}

/// Passes every request straight through to a single `ChessMPSNetwork`
/// on the calling thread. No queueing, no batching. Suitable for paths
/// that already serialize their own calls into the network — arena
/// tournaments and the Play Game screen both play one game at a time,
/// and `ChessMachine.runGameLoop` serializes the two players within a
/// game. Self-play no longer uses this path; it routes through
/// `BatchedMoveEvaluationSource`.
///
/// The underlying `ChessMPSNetwork`'s policy readback is non-reentrant.
/// We hand it a closure that copies the policy bytes directly from the
/// network's scratch into the caller-owned `intoPolicy` destination,
/// so the caller owns the bytes at every synchronization boundary —
/// the network's scratch can be reused on the next call without
/// invalidating any outstanding policy buffer.
final class DirectMoveEvaluationSource: MoveEvaluationSource, @unchecked Sendable {
    let network: ChessMPSNetwork

    init(network: ChessMPSNetwork) {
        self.network = network
    }

    func evaluate(
        encodedBoard: [Float],
        intoPolicy: PolicyDestination
    ) async throws -> Float {
        precondition(intoPolicy.count == ChessNetwork.policySize,
            "DirectMoveEvaluationSource.evaluate: intoPolicy.count \(intoPolicy.count) "
            + "must equal ChessNetwork.policySize \(ChessNetwork.policySize)")
        // `intoPolicy` is `Sendable` via the wrapper. The `var`
        // mutation from inside the `@Sendable` consume closure
        // requires `nonisolated(unsafe)`; safe because the await
        // suspends the calling task for the full closure window and
        // Swift's continuation resume publishes the write to the
        // post-await read.
        nonisolated(unsafe) var capturedValue: Float = 0
        try await network.evaluate(board: encodedBoard) { policyBuf, value in
            // Runs on `ChessNetwork.executionQueue` inside
            // `internalEvaluate`'s `autoreleasepool`. The destination
            // is owned by the caller and is alive for the duration of
            // this closure call (the awaiting task is suspended at
            // `try await network.evaluate(board:consume:)`, so its
            // stored properties — including the destination's
            // backing — stay valid).
            //
            // `policyBuf.baseAddress` is non-nil by construction:
            // `internalEvaluate` builds the buffer as
            // `UnsafeBufferPointer(start: inferencePolicyScratchPtr, count: policySize)`
            // where the scratch pointer is the result of a
            // `UnsafeMutablePointer.allocate` and is always non-nil.
            // A nil here would indicate an internal invariant
            // violation in the network, not a recoverable condition.
            guard let policyBase = policyBuf.baseAddress else {
                preconditionFailure(
                    "ChessNetwork single-board consume buffer has nil baseAddress"
                )
            }
            intoPolicy.pointer.update(
                from: policyBase,
                count: policyBuf.count
            )
            capturedValue = value
        }
        return capturedValue
    }
}
