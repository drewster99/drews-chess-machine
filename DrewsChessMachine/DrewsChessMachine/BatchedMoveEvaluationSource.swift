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

/// Snapshot of batch-size telemetry collected by a
/// `BatchedMoveEvaluationSource` since it was constructed. Used by the
/// arena callsite to log, on tournament end, how much GPU batching the
/// run actually achieved across both the candidate and the champion
/// networks. The histogram is keyed by per-fire batch size and counts
/// how many fires landed at each size — a quick way to see whether the
/// coalescing-window timer was actually finding multi-slot batches or
/// the run was effectively serial.
struct BatchSizeStats: Sendable {
    let totalBatches: Int
    let totalPositions: Int
    let minBatch: Int
    let maxBatch: Int
    let mean: Double
    let histogram: [Int: Int]

    static let empty = BatchSizeStats(
        totalBatches: 0, totalPositions: 0,
        minBatch: 0, maxBatch: 0, mean: 0, histogram: [:]
    )

    /// Render the histogram + summary into a single human-readable line
    /// suitable for direct emission to the session log.
    func formatLogLine() -> String {
        guard totalBatches > 0 else {
            return "no batches fired"
        }
        let histStr = histogram.keys.sorted()
            .map { "\($0):\(histogram[$0] ?? 0)" }
            .joined(separator: ",")
        return String(
            format: "mean=%.2f min=%d max=%d batches=%d positions=%d hist={%@}",
            mean, minBatch, maxBatch, totalBatches, totalPositions, histStr
        )
    }
}

/// Barrier-style batcher for chess inference.
///
/// **Self-play mode** (the original use): N `ChessMachine.runGameLoop`
/// tasks (one per self-play slot) all call `evaluate(encodedBoard:)`
/// exactly once per ply. Each call parks in a `CheckedContinuation`;
/// when the N-th submission arrives (where N = `expectedSlotCount`),
/// the barrier fires a single batched
/// `network.evaluate(batchBoards:count:)`, slices the N policy vectors
/// and N value scalars out of the returned buffers, and resumes all N
/// continuations with their own `(policy, value)`. The cycle repeats
/// on the next ply.
///
/// Because each live slot submits exactly once per barrier cycle, the
/// invariant `pending.count == expectedSlotCount on fire` holds by
/// construction — no deadline, no timeout, no stragglers. This is the
/// happy case and the construction parameter `maxBatchWaitMs` defaults
/// to 0 so self-play behavior is unchanged.
///
/// **Arena mode** (added for concurrent arena games): two batchers are
/// constructed — one wrapping the candidate inference network, the
/// other wrapping the arena champion. Each arena game alternates
/// candidate and champion moves, so at any instant only ~K/2 of the K
/// concurrent games are on a given side. The strict count barrier
/// would never fire either batcher in steady state. To handle this,
/// when `maxBatchWaitMs > 0` the batcher schedules a coalescing
/// timer that fires the next batch after that many ms have elapsed
/// since `pending` last went from empty to non-empty. The barrier
/// fires on whichever happens first (count-met OR timer-expired), so
/// in the early game when all K games are synchronized the count
/// barrier still fires immediately, while the desynchronized
/// steady-state fires partial batches at the end of the wait window.
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
/// - Arena game-ends: as games finish during a parallel arena run,
///   `decrementExpectedSlotCount()` is called per-exit so the count
///   barrier keeps tracking live game count rather than initial K.
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

    /// Coalescing-window upper bound. When > 0 and the count barrier
    /// has not been met, the batcher fires whatever is pending after
    /// this many ms have elapsed since `pending` first became
    /// non-empty in the current cycle. 0 disables the timer and
    /// preserves the strict count-only barrier (the self-play mode).
    /// See the class doc comment for why arena needs this and
    /// self-play does not.
    let maxBatchWaitMs: Double

    /// Number of live slots the driver expects to submit per barrier
    /// cycle. The barrier fires as soon as `pending.count` reaches this
    /// value. Set to 0 while the driver has no active slots (during
    /// arena pauses or before the first slot is spawned) — any late
    /// submission under those conditions still parks but won't fire
    /// the batch until the count is raised back up.
    private var expectedSlotCount: Int = 0

    /// Generation counter incremented each time a fire is scheduled or
    /// completed. Coalescing-timer tasks capture the generation they
    /// were scheduled for and no-op on wake-up if the generation has
    /// moved on (because the count barrier or an earlier timer task
    /// already fired the queue). Without this, a late timer expiry
    /// after a count-fire would attempt to fire an already-empty queue
    /// or, worse, fire prematurely and cut a fresh window short.
    private var batchWaitGeneration: Int = 0

    /// Per-batch-size histogram. Keyed by `count` at fire time;
    /// values are how many fires landed at that count. Read out via
    /// `snapshotBatchSizeStats` at tournament end. Reset is not
    /// supported — each batcher is short-lived (one arena's worth of
    /// games) so accumulating monotonically through the run is the
    /// natural model.
    private var batchSizeCounts: [Int: Int] = [:]
    private var batchSizeTotalPositions: Int = 0
    private var batchSizeTotalBatches: Int = 0
    private var batchSizeMin: Int = .max
    private var batchSizeMax: Int = 0

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
        /// Per-call identifier so the cancellation handler can find
        /// and remove this specific entry without disturbing other
        /// pending submissions. UUID is overkill on coverage but
        /// trivial in cost (16 bytes + a single allocation per call,
        /// negligible against the ~1 ms ply we're already paying).
        let token: UUID
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

    init(network: ChessMPSNetwork, maxBatchWaitMs: Double = 0) {
        self.network = network
        self.maxBatchWaitMs = max(0, maxBatchWaitMs)
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

    /// Decrement `expectedSlotCount` by one, clamped at 0. Used by
    /// the parallel-arena driver to track live game count as games
    /// finish, so the count barrier remains achievable rather than
    /// pegged at the initial K. Calling this is functionally
    /// equivalent to `setExpectedSlotCount(currentCount - 1)`; we
    /// expose it as a one-shot to avoid the read-then-write race
    /// callers would otherwise have to coordinate themselves.
    func decrementExpectedSlotCount() {
        setExpectedSlotCount(max(0, expectedSlotCount - 1))
    }

    /// Snapshot the cumulative batch-size telemetry. Safe to call at
    /// any time; arena code reads it once after the tournament has
    /// completed (and the batchers have drained) to log how much
    /// concurrent batching the run actually achieved.
    func snapshotBatchSizeStats() -> BatchSizeStats {
        guard batchSizeTotalBatches > 0 else {
            return .empty
        }
        let mean = Double(batchSizeTotalPositions) / Double(batchSizeTotalBatches)
        return BatchSizeStats(
            totalBatches: batchSizeTotalBatches,
            totalPositions: batchSizeTotalPositions,
            minBatch: batchSizeMin == .max ? 0 : batchSizeMin,
            maxBatch: batchSizeMax,
            mean: mean,
            histogram: batchSizeCounts
        )
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
        // Per-call token used by the cancellation handler below to
        // find and remove this specific parked submission so a
        // cancelled awaiter unblocks instead of leaking forever.
        // Without this, the concurrent arena's `withThrowingTaskGroup`
        // auto-cancel-on-throw deadlocks: when one slot throws the
        // group cancels its siblings, but Swift does NOT auto-resume
        // continuations parked in `withCheckedThrowingContinuation`,
        // so the siblings hang and the group's auto-await-on-exit
        // blocks the parent forever. The cancellation handler is
        // also a no-op for self-play because that driver drains
        // *before* cancelling slots — by the time `task.cancel()`
        // arrives, no slot is parked at the batcher.
        let token = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let wasEmpty = pending.isEmpty
                pending.append(Pending(
                    token: token,
                    boardCopy: encodedBoard,
                    continuation: continuation
                ))
                // Fire when:
                // - Normal barrier mode (`expectedSlotCount > 0`)
                //   and the barrier threshold is met.
                // - Drain mode (`expectedSlotCount == 0`) — the
                //   driver has asked every slot to wind down, so we
                //   can't wait for a larger group to assemble.
                //   Process each submission as its own micro-batch
                //   so in-flight slots can finish their current
                //   games and exit.
                if expectedSlotCount == 0 || pending.count >= expectedSlotCount {
                    scheduleBatchFireIfNeeded()
                } else if wasEmpty && maxBatchWaitMs > 0 && expectedSlotCount > 0 {
                    // Coalescing-window timer (arena mode). Schedule
                    // a backstop fire for the current generation;
                    // whichever path (count-met above OR timer below)
                    // wins the race bumps the generation, so the
                    // loser is a no-op. Only armed on the 0→1
                    // transition so we don't stack multiple sleeping
                    // tasks per cycle.
                    scheduleBatchWaitTimer(forGeneration: batchWaitGeneration)
                }
            }
        } onCancel: {
            // Runs synchronously when the awaiting task is cancelled.
            // We can't touch actor state from here directly, so hop
            // back to the actor to remove this submission's pending
            // entry and resume its continuation with `CancellationError`.
            // Race-safe vs. `fireBatch`: both `cancelPending` and
            // `fireBatch` are actor-isolated, so they serialize. If
            // `fireBatch` resumed the continuation first, our
            // cancelPending finds nothing and no-ops; if cancelPending
            // ran first, fireBatch's snapshot of `pending` won't
            // include this entry. No double-resume in either order.
            Task {
                await self.cancelPending(token: token)
            }
        }
    }

    /// Remove the pending submission identified by `token` (if still
    /// queued) and resume its continuation with `CancellationError`.
    /// Called from the cancellation handler in `evaluate`. A no-op if
    /// the entry is not found — that just means `fireBatch` already
    /// claimed and resumed it before this handler reached the actor.
    private func cancelPending(token: UUID) {
        guard let idx = pending.firstIndex(where: { $0.token == token }) else {
            return
        }
        let item = pending.remove(at: idx)
        item.continuation.resume(throwing: CancellationError())
    }

    private func scheduleBatchFireIfNeeded() {
        guard !batchFireScheduled else { return }
        batchFireScheduled = true
        // Bump the generation here as well so any in-flight timer
        // task for this cycle wakes up and sees its generation is
        // stale. Without the bump, the count path firing would still
        // leave a sleeping timer that, on wake-up, would try to fire
        // an empty (or freshly partial) queue.
        batchWaitGeneration &+= 1
        Task {
            await self.fireBatch()
        }
    }

    /// Arm a coalescing-window timer for the given generation. After
    /// `maxBatchWaitMs` the timer task wakes up; if `pending` is still
    /// non-empty AND the generation hasn't moved on (i.e. neither the
    /// count barrier nor an earlier timer fire has consumed the
    /// queue), the timer schedules the fire itself. Called only when
    /// `maxBatchWaitMs > 0` and only on the 0→1 pending transition.
    private func scheduleBatchWaitTimer(forGeneration generation: Int) {
        let waitNs = UInt64((maxBatchWaitMs * 1_000_000.0).rounded())
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: waitNs)
            guard let self else { return }
            await self.fireIfStillCurrent(generation: generation)
        }
    }

    private func fireIfStillCurrent(generation: Int) {
        guard generation == batchWaitGeneration else { return }
        guard !pending.isEmpty else { return }
        scheduleBatchFireIfNeeded()
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

            // Record this fire into the batch-size histogram. Done
            // after the network call succeeds so failed fires (which
            // resume continuations with `error`) aren't counted as
            // measured throughput.
            batchSizeCounts[count, default: 0] += 1
            batchSizeTotalBatches += 1
            batchSizeTotalPositions += count
            if count < batchSizeMin { batchSizeMin = count }
            if count > batchSizeMax { batchSizeMax = count }

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
