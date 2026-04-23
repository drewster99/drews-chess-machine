import Foundation

// MARK: - Errors

enum BatchedMoveEvaluationSourceError: LocalizedError {
    case batchResultShapeMismatch(expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .batchResultShapeMismatch(let expected, let got):
            return "BatchedMoveEvaluationSource: batched evaluate returned \(got) values, expected \(expected)"
        }
    }
}

// MARK: - Barrier Batcher

/// Barrier-style batcher for self-play inference.
///
/// N `ChessMachine.runGameLoop` tasks (one per self-play slot) all call
/// `evaluate(encodedBoard:)` exactly once per ply. Each call parks in a
/// `CheckedContinuation`; when the N-th submission arrives (where N =
/// `expectedSlotCount`), the barrier fires a single batched
/// `network.evaluate(batchBoards:count:)`, slices the N policy vectors
/// and N value scalars out of the returned buffers, and resumes all N
/// continuations with their own `(policy, value)`. The cycle repeats
/// on the next ply.
///
/// Because each live slot submits exactly once per barrier cycle, the
/// invariant `pending.count == expectedSlotCount on fire` holds by
/// construction — no deadline, no timeout, no stragglers.
///
/// Slot lifecycle coordination is handled by the driver:
/// - Grow: call `setExpectedSlotCount(newN)` *before* spawning the new
///   slot task, so its first submission is counted toward the new
///   barrier.
/// - Shrink: cancel the slot task and await its exit (it finishes its
///   current game first), then call `setExpectedSlotCount(newN)`.
/// - Pause (arena): cancel all slot tasks, await their exit, call
///   `setExpectedSlotCount(0)`. On resume, call
///   `setExpectedSlotCount(liveCount)` before re-spawning.
///
/// All state — pending queue, expected count, network access, weight
/// load/export — is actor-isolated, so barrier bookkeeping and the
/// `graph.run` call are serialized with respect to everything else.
actor BatchedMoveEvaluationSource: MoveEvaluationSource {
    // MARK: - Configuration

    private static let boardFloats = BoardEncoder.tensorLength
    private static let policySize = ChessNetwork.policySize

    // MARK: - State

    let network: ChessMPSNetwork

    /// Number of live slots the driver expects to submit per barrier
    /// cycle. The barrier fires as soon as `pending.count` reaches this
    /// value. Set to 0 while the driver has no active slots (during
    /// arena pauses or before the first slot is spawned) — any late
    /// submission under those conditions still parks but won't fire
    /// the batch until the count is raised back up.
    private var expectedSlotCount: Int = 0

    /// Optional replay-ratio controller. When set, the batcher reports
    /// a `recordSelfPlayBarrierTick` event after each successful
    /// `graph.run` while in normal (non-drain) mode. The controller
    /// computes aggregate self-play ms-per-move from the
    /// (positions_produced, elapsed) pair — far finer-grained than
    /// the previous once-per-game event cadence.
    private var replayRatioController: ReplayRatioController? = nil

    /// CFAbsoluteTime (seconds, monotonic-ish) of the previous barrier
    /// fire, used to compute inter-tick elapsed for the ratio
    /// controller. Nil before the first fire — skipped because there
    /// is no prior to subtract from.
    private var lastBarrierFireAt: CFAbsoluteTime? = nil

    /// Parked submissions awaiting the next batch fire.
    private var pending: [Pending] = []
    private var batchFireScheduled = false

    private struct Pending {
        /// Copy of the caller's encoded board — the caller's
        /// `UnsafeBufferPointer` is only valid for the duration of
        /// `evaluate(encodedBoard:)`, so we snapshot into owned storage
        /// before parking.
        let boardCopy: [Float]
        let continuation: CheckedContinuation<(policy: [Float], value: Float), Error>
    }

    /// Reusable scratch for packing `count × BoardEncoder.tensorLength`
    /// floats into one contiguous buffer before the batched `graph.run`.
    /// Resized to *exactly* `totalFloats` each batch so the buffer can
    /// be handed straight to `network.evaluate(batchBoards:count:)`
    /// without a trimming copy — that API validates
    /// `batchBoards.count == count * tensorLength`. Steady-state (stable
    /// slot count) the resize is a no-op and the same COW storage is
    /// reused indefinitely; a slot-count change or a reentrant
    /// `fireBatch` overlap triggers at most one COW uniquify (still far
    /// cheaper than the per-batch full-buffer copy the previous
    /// implementation paid unconditionally to trim an oversized scratch).
    /// Swift `[Float]` so storage management is automatic and the
    /// actor doesn't need a manual `deinit` that would have to touch
    /// non-Sendable raw pointers from a nonisolated deinit.
    private var packBuffer: [Float] = []

    /// One-shot debug flag for the batched-output correctness check.
    /// First time `fireBatch` runs with a `count >= 2` batch that
    /// contains at least two distinct board encodings, we compare
    /// those two slots' returned policy slices. If the slices are
    /// bit-identical despite the inputs differing, the batch is
    /// collapsing outputs (e.g. shape-broadcast bug, readback
    /// aliasing, slicing bug) — which would explain "all slots pick
    /// the same move" in self-play. On pass we log and flip the flag
    /// so the check runs once per session.
    #if DEBUG
    private var batchCorrectnessCheckDone = false
    #endif

    // MARK: - Init

    init(network: ChessMPSNetwork) {
        self.network = network
    }

    // MARK: - Slot-Count Coordination

    /// Update the barrier threshold.
    ///
    /// - Raising: takes effect on the next submission that would have
    ///   fired the old threshold.
    /// - Lowering to `n > 0` below the current pending count: fires
    ///   immediately to flush the queue.
    /// - Setting to 0 (drain mode): fires immediately with whatever is
    ///   queued, so parked callers wake up; subsequent submissions in
    ///   drain mode fire as single-element (or whatever-is-pending)
    ///   batches — see `evaluate(encodedBoard:)`. Drain mode is used by
    ///   `BatchedSelfPlayDriver.stopAll` during arena pauses and
    ///   session shutdown to guarantee in-flight slots can always make
    ///   progress even while they're being asked to exit.
    func setExpectedSlotCount(_ n: Int) {
        precondition(n >= 0, "expectedSlotCount must be >= 0")
        expectedSlotCount = n
        // Entering drain mode means the steady-state production
        // measurement series is interrupted — slots are being torn
        // down, fires from here until re-engagement are either
        // stragglers or nothing at all. Clear the last-fire timestamp
        // so the first post-resume fire is SKIPPED (no spurious
        // multi-minute elapsed across the pause window) rather than
        // producing a garbage ms/move sample that would swing the
        // controller hard.
        if n == 0 {
            lastBarrierFireAt = nil
        }
        if !pending.isEmpty && (n == 0 || pending.count >= n) {
            scheduleBatchFireIfNeeded()
        }
    }

    /// Install the replay-ratio controller. Called once at session
    /// start, after the controller is built. A nil assignment
    /// disables tick reporting — the batcher otherwise has no
    /// observable effect whether or not the controller is attached.
    /// `lastBarrierFireAt` is cleared alongside so a re-attach after
    /// a session break doesn't report an absurd multi-minute elapsed
    /// on the first post-reattach tick.
    func setReplayRatioController(_ controller: ReplayRatioController?) {
        replayRatioController = controller
        lastBarrierFireAt = nil
    }

    // MARK: - Inference

    func evaluate(
        encodedBoard: [Float]
    ) async throws -> (policy: [Float], value: Float) {
        return try await withCheckedThrowingContinuation { continuation in
            pending.append(Pending(boardCopy: encodedBoard, continuation: continuation))
            // Fire when:
            // - Normal barrier mode (`expectedSlotCount > 0`) and the
            //   barrier threshold is met.
            // - Drain mode (`expectedSlotCount == 0`) — the driver has
            //   asked every slot to wind down, so we can't wait for a
            //   larger group to assemble. Process each submission as
            //   its own micro-batch so in-flight slots can finish
            //   their current games and exit.
            if expectedSlotCount == 0 || pending.count >= expectedSlotCount {
                scheduleBatchFireIfNeeded()
            }
        }
    }

    private func scheduleBatchFireIfNeeded() {
        guard !batchFireScheduled else { return }
        batchFireScheduled = true
        Task {
            await self.fireBatch()
        }
    }

    private func fireBatch() async {
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        batchFireScheduled = false
        // fireBatch is only invoked from paths that have already
        // checked `pending.count >= expectedSlotCount > 0`, so an
        // empty batch here would indicate a bookkeeping bug rather
        // than a condition to silently tolerate.
        precondition(!batch.isEmpty, "BatchedMoveEvaluationSource.fireBatch invoked with empty pending queue")

        let count = batch.count
        let totalFloats = count * Self.boardFloats
        // Size the pack scratch to exactly what the batch needs. In the
        // steady state (stable slot count) this is a no-op, the same
        // COW storage is reused batch after batch, and the downstream
        // `network.evaluate(batchBoards:count:)` call passes `packBuffer`
        // straight through without any intermediate trimming copy.
        // Reassignment (rather than `removeLast` / shrink-in-place)
        // keeps the code path symmetric for grow and shrink and avoids
        // accidentally carrying stale floats across a size change.
        if packBuffer.count != totalFloats {
            packBuffer = [Float](repeating: 0, count: totalFloats)
        }

        packBuffer.withUnsafeMutableBufferPointer { packBuf in
            guard let packBase = packBuf.baseAddress else { return }
            for (i, item) in batch.enumerated() {
                let dst = packBase.advanced(by: i * Self.boardFloats)
                item.boardCopy.withUnsafeBufferPointer { src in
                    if let srcBase = src.baseAddress {
                        dst.update(from: srcBase, count: Self.boardFloats)
                    }
                }
            }
        }

        do {
            // Pass `packBuffer` (an owned Swift `[Float]`) directly
            // across the `await`. Swift's COW guarantees the buffer
            // stays intact for the duration of the evaluation even if
            // a reentrant `fireBatch` runs on this actor at a
            // suspension point and resizes/rewrites our `packBuffer`
            // stored property — the first mutating access there would
            // uniquify a fresh buffer, leaving the one captured by the
            // in-flight `network.evaluate` call untouched.
            let (policy, values) = try await network.evaluate(
                batchBoards: packBuffer,
                count: count
            )
            guard policy.count == count * Self.policySize else {
                throw BatchedMoveEvaluationSourceError.batchResultShapeMismatch(
                    expected: count * Self.policySize, got: policy.count
                )
            }
            guard values.count == count else {
                throw BatchedMoveEvaluationSourceError.batchResultShapeMismatch(
                    expected: count, got: values.count
                )
            }

            #if DEBUG
            runBatchCorrectnessCheckIfNeeded(batch: batch, policy: policy, values: values)
            #endif

            // Report this tick to the replay-ratio controller. One
            // barrier fire = one "measurement event" on the self-play
            // side; the (count, elapsed) pair is an aggregate
            // ms-per-move sample by construction. Drain-mode fires
            // (`expectedSlotCount == 0`) are skipped — those are
            // shutdown / arena-pause stragglers, not representative
            // of steady-state production rate. First fire after a
            // reattach is also skipped because `lastBarrierFireAt`
            // is nil and the elapsed would be meaningless.
            if let controller = replayRatioController, expectedSlotCount > 0 {
                let nowFire = CFAbsoluteTimeGetCurrent()
                if let last = lastBarrierFireAt {
                    let elapsedMs = (nowFire - last) * 1000.0
                    if elapsedMs > 0 {
                        controller.recordSelfPlayBarrierTick(
                            positionsProduced: count,
                            elapsedMs: elapsedMs,
                            workerCount: expectedSlotCount
                        )
                    }
                }
                lastBarrierFireAt = nowFire
            }

            for (i, item) in batch.enumerated() {
                let start = i * Self.policySize
                let end = start + Self.policySize
                let policySlice = Array(policy[start..<end])
                let value = values[i]
                item.continuation.resume(returning: (policy: policySlice, value: value))
            }
        } catch {
            for item in batch {
                item.continuation.resume(throwing: error)
            }
        }

        if !pending.isEmpty && (expectedSlotCount == 0 || pending.count >= expectedSlotCount) {
            scheduleBatchFireIfNeeded()
        }
    }

    // MARK: - Debug Correctness Check

    #if DEBUG
    /// One-shot check: if this batch has at least one pair of slots
    /// whose encoded boards differ, verify their returned policy
    /// slices also differ. Identical outputs for distinct inputs
    /// would be a batching bug (shape broadcast, readback aliasing,
    /// slicing bug). On the first batch that provides a pair we can
    /// actually compare, either log success + flip the flag (so this
    /// is O(1) across the session) or `preconditionFailure` on
    /// mismatch.
    private func runBatchCorrectnessCheckIfNeeded(
        batch: [Pending],
        policy: [Float],
        values: [Float]
    ) {
        if batchCorrectnessCheckDone { return }
        let count = batch.count
        guard count >= 2 else { return }
        let base = batch[0].boardCopy
        for i in 1..<count where batch[i].boardCopy != base {
            let slot0Policy = Array(policy[0..<Self.policySize])
            let otherStart = i * Self.policySize
            let otherPolicy = Array(policy[otherStart..<otherStart + Self.policySize])
            let slot0Value = values[0]
            let otherValue = values[i]
            if slot0Policy == otherPolicy && slot0Value == otherValue {
                preconditionFailure(
                    "BatchedMoveEvaluationSource correctness check FAILED: "
                    + "slots 0 and \(i) returned bit-identical policy+value "
                    + "despite distinct inputs. Batched graph.run is "
                    + "collapsing outputs across the batch dimension."
                )
            }
            SessionLogger.shared.log(
                "[BATCHER] Correctness check passed: batch size \(count), "
                + "slot 0 vs slot \(i) produced distinct policies "
                + "(value0=\(slot0Value) value\(i)=\(otherValue))."
            )
            batchCorrectnessCheckDone = true
            return
        }
        // All boards in this batch are identical (e.g. every slot at
        // ply 0 of a new game). Wait for a later batch that actually
        // has distinct inputs before running the check.
    }
    #endif

}
