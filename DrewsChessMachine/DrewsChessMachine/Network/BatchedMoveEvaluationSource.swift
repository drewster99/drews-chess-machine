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

/// Why a barrier fire was scheduled. Tagged at every
/// `scheduleBatchFireIfNeeded` call site so the end-of-arena summary
/// can answer "is the coalescing-window timer actually in play?"
/// directly, rather than having to infer it from the size histogram.
///
/// - `full`: a submission landed and pushed `pending.count` to or above
///   `expectedSlotCount`. The healthy steady-state case at sane wait
///   configurations.
/// - `timer`: the coalescing-window timer expired with `pending` still
///   non-empty and the count barrier not met. Indicates the wait knob
///   is actually doing something — or that one or more slots have
///   stalled in their per-ply CPU work and aren't keeping up.
/// - `drain`: `expectedSlotCount == 0` (drain mode). Used during arena
///   pauses and session shutdown so parked callers always make
///   progress.
/// - `threshold`: `setExpectedSlotCount` lowered the bar below the
///   currently-pending count. Same effect as `full`, different cause —
///   tracking separately makes "we shrank the pool mid-arena" visible.
/// - `refill`: at the end of `fireBatch`, `pending` was non-empty and
///   met the (possibly-lowered) threshold, so we scheduled the next
///   fire immediately. Indicates back-to-back fires with no idle gap.
enum FireReason: String, Sendable, CaseIterable {
    case full
    case timer
    case drain
    case threshold
    case refill
}

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
    /// Per-fire-reason counters. Total across keys equals the number
    /// of *scheduled* fires (which is `>= totalBatches` because a
    /// scheduled fire can land in `fireBatch` with an empty batch when
    /// the only pending submission was cancelled — see the
    /// "Cancellation race" guard in `fireBatch`).
    let fireReasonCounts: [FireReason: Int]
    /// Number of fires where `expectedSlotCount` changed during the
    /// `await network.evaluate(...)` GPU window. Non-zero is normal —
    /// games end during fires — but asymmetry across cand/champ or a
    /// large `expectedDriftMaxDelta` would point at coordination
    /// problems between the harvest loop and the slot-count actor.
    let expectedDriftCount: Int
    let expectedDriftMaxDelta: Int

    static let empty = BatchSizeStats(
        totalBatches: 0, totalPositions: 0,
        minBatch: 0, maxBatch: 0, mean: 0, histogram: [:],
        fireReasonCounts: [:],
        expectedDriftCount: 0, expectedDriftMaxDelta: 0
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

    /// Render the fire-reason counters into a single human-readable
    /// line. Always emits all five reasons in a fixed order so two
    /// lines (cand vs champ) can be eyeballed for asymmetry without
    /// having to mentally line up the keys. Missing reasons render
    /// as `=0` for the same reason.
    func formatFireReasonsLine() -> String {
        let order: [FireReason] = [.full, .timer, .drain, .threshold, .refill]
        let parts = order.map { r in
            "\(r.rawValue)=\(fireReasonCounts[r] ?? 0)"
        }
        return parts.joined(separator: " ")
    }

    /// Render the `expectedSlotCount`-drift counters. `drifts` is
    /// the number of fires whose GPU-await window saw a change in
    /// `expectedSlotCount`; `maxDelta` is the largest absolute change
    /// seen across all drifting fires. Steady-state non-zero `drifts`
    /// is healthy (slot retirement during a fire is expected); cross-
    /// side asymmetry or `maxDelta > ~5` is the interesting signal.
    func formatExpectedDriftLine() -> String {
        return "drifts=\(expectedDriftCount) maxDelta=\(expectedDriftMaxDelta) (out of batches=\(totalBatches))"
    }
}

/// Wall-clock timing accumulators for one `BatchedMoveEvaluationSource`,
/// snapshotted at end-of-arena. `wait` is the cumulative time the batcher
/// spent in coalescing windows — measured from the moment `pending` first
/// became non-empty in a cycle to the moment `fireBatch` actually started
/// processing it. `run` is the cumulative time inside the underlying
/// `network.evaluate(batchBoards:count:)` call. Anything outside those
/// two windows but inside the arena's wall time is "idle" — the batcher
/// had nothing pending (e.g. the OTHER batcher was running, or slots
/// were doing CPU-side legality + sampling work).
///
/// The arena callsite computes `idle = max(0, wallNs - waitNs - runNs)`
/// when emitting the end-of-arena timing summary; it isn't stored here
/// because the wall-time anchor is owned by the caller, not the batcher.
struct BatchTimingStats: Sendable {
    let totalBatches: Int
    let totalWaitNanos: UInt64
    let totalRunNanos: UInt64

    static let empty = BatchTimingStats(
        totalBatches: 0, totalWaitNanos: 0, totalRunNanos: 0
    )

    var totalWaitMs: Double { Double(totalWaitNanos) / 1_000_000.0 }
    var totalRunMs: Double { Double(totalRunNanos) / 1_000_000.0 }
    var meanWaitMs: Double {
        totalBatches > 0 ? totalWaitMs / Double(totalBatches) : 0
    }
    var meanRunMs: Double {
        totalBatches > 0 ? totalRunMs / Double(totalBatches) : 0
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

    /// Human-readable label used in per-batch debug log lines (e.g.
    /// "candidate" / "champion" so the arena's log stream can be
    /// disambiguated by network). Empty string suppresses the name
    /// from the log; ignored entirely when `logBatchTimings` is false.
    let name: String

    /// When `true`, every successful `fireBatch` emits a single
    /// `[ARENA-BATCH]` line carrying batch size, this-batch wait/run
    /// ms, and running averages. False for self-play (the line cadence
    /// would dominate the log file at 8+ workers × ~ms-per-ply).
    ///
    /// Layered with the `DCM_LOG_PER_BATCH_TIMINGS` compile-time flag:
    /// the runtime gate selects WHICH batchers should log when the
    /// feature is enabled, but the compile flag determines whether
    /// the emission code is even present in the binary. With the flag
    /// undefined (the default), per-batch emission is fully elided
    /// regardless of this value — only the end-of-arena summary
    /// remains. To turn on, add `-D DCM_LOG_PER_BATCH_TIMINGS` to the
    /// project's `OTHER_SWIFT_FLAGS` and rebuild.
    let logBatchTimings: Bool

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

    /// Per-fire-reason counter. Incremented at every successful
    /// `scheduleBatchFireIfNeeded` call (i.e. ones that pass the
    /// `batchFireScheduled` guard). A scheduled fire that ultimately
    /// lands in `fireBatch` with an empty batch (cancellation race)
    /// still counts here — the count is "scheduled fires" not
    /// "successful evaluates" — because the scheduling intent is
    /// itself the diagnostic signal.
    private var fireReasonCounts: [FireReason: Int] = [:]

    /// Drift detection: number of fires whose `await network.evaluate`
    /// window observed a change in `expectedSlotCount` between entry
    /// and resume, plus the max absolute change seen. Mirrors the
    /// `expected-drift` log line emitted at end of arena. The shape
    /// of the signal we're watching for is asymmetry across cand/
    /// champ or sustained `maxDelta > ~5`; small drift counts are
    /// the norm because games end during fires.
    private var expectedDriftCount: Int = 0
    private var expectedDriftMaxDelta: Int = 0

    /// Wall-clock timing accumulators, paired with the size histogram
    /// above. `batchPendingStartedNanos` records the `DispatchTime`
    /// at which `pending` last transitioned 0 → 1 in the current
    /// cycle; `fireBatch` reads it on entry to compute that cycle's
    /// wait window, then resets it. 0 means "no current cycle" (the
    /// pending queue is empty or a fire just consumed it). The wait
    /// window includes both the deliberate coalescing wait (if any)
    /// AND the small async-hop latency between `scheduleBatchFireIfNeeded`
    /// returning and the dispatched `Task` actually entering `fireBatch`,
    /// which is the right thing to measure from a slot's perspective.
    private var batchTotalWaitNanos: UInt64 = 0
    private var batchTotalRunNanos: UInt64 = 0
    private var batchPendingStartedNanos: UInt64 = 0

    /// Optional replay-ratio controller. When set, the batcher reports
    /// a `recordSelfPlayBarrierTick` event after each successful
    /// `graph.run` while in normal (non-drain) mode. The controller
    /// computes aggregate self-play ms-per-move from the
    /// (positions_produced, elapsed) pair — far finer-grained than
    /// the previous once-per-game event cadence.
    private var replayRatioController: ReplayRatioController? = nil

    /// Optional joint-GPU timer shared across the two arena batchers.
    /// Each barrier fire reports `fireStarted` / `fireEnded` around
    /// the `await network.evaluate(...)` call so the arena callsite
    /// can emit a single `[ARENA] timing joint` line covering wall-
    /// clock during which AT LEAST ONE side was on the GPU. nil for
    /// self-play (only one batcher exists) and for any path that
    /// doesn't care about cross-batcher GPU saturation.
    private let gpuTimer: ArenaGpuTimer?

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

    init(
        network: ChessMPSNetwork,
        maxBatchWaitMs: Double = 0,
        name: String = "",
        logBatchTimings: Bool = false,
        gpuTimer: ArenaGpuTimer? = nil
    ) {
        self.network = network
        self.maxBatchWaitMs = max(0, maxBatchWaitMs)
        self.name = name
        self.logBatchTimings = logBatchTimings
        self.gpuTimer = gpuTimer
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
        // stragglers or nothing at all. Tell the controller to
        // clear its self-play wall-clock stamp so the first
        // post-resume fire is treated as a fresh first-tick (no
        // garbage multi-minute ms/move from the pause window).
        if n == 0 {
            replayRatioController?.resetSelfPlayClock()
        }
        if !pending.isEmpty && (n == 0 || pending.count >= n) {
            scheduleBatchFireIfNeeded(reason: n == 0 ? .drain : .threshold)
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
            // Even on a no-batches path we still want fire-reason
            // counters preserved — `.empty` zeros them, but if a
            // tournament was cancelled before any batch completed
            // there could still be schedule attempts worth surfacing.
            return BatchSizeStats(
                totalBatches: 0,
                totalPositions: 0,
                minBatch: 0,
                maxBatch: 0,
                mean: 0,
                histogram: [:],
                fireReasonCounts: fireReasonCounts,
                expectedDriftCount: expectedDriftCount,
                expectedDriftMaxDelta: expectedDriftMaxDelta
            )
        }
        let mean = Double(batchSizeTotalPositions) / Double(batchSizeTotalBatches)
        return BatchSizeStats(
            totalBatches: batchSizeTotalBatches,
            totalPositions: batchSizeTotalPositions,
            minBatch: batchSizeMin == .max ? 0 : batchSizeMin,
            maxBatch: batchSizeMax,
            mean: mean,
            histogram: batchSizeCounts,
            fireReasonCounts: fireReasonCounts,
            expectedDriftCount: expectedDriftCount,
            expectedDriftMaxDelta: expectedDriftMaxDelta
        )
    }

    /// Snapshot the cumulative wait/run timing accumulators. Companion
    /// to `snapshotBatchSizeStats`; arena code reads both at end-of-
    /// run to log batch-size distribution AND wall-clock attribution.
    func snapshotBatchTimingStats() -> BatchTimingStats {
        return BatchTimingStats(
            totalBatches: batchSizeTotalBatches,
            totalWaitNanos: batchTotalWaitNanos,
            totalRunNanos: batchTotalRunNanos
        )
    }

    /// Install the replay-ratio controller. Called once at session
    /// start, after the controller is built. A nil assignment
    /// disables tick reporting — the batcher otherwise has no
    /// observable effect whether or not the controller is attached.
    /// On every set we ask the freshly-attached (or freshly-detached
    /// previous) controller to reset its self-play wall-clock stamp,
    /// so a re-attach after a session break doesn't produce an
    /// absurd multi-minute elapsed sample on the first tick.
    func setReplayRatioController(_ controller: ReplayRatioController?) {
        replayRatioController?.resetSelfPlayClock()
        replayRatioController = controller
        controller?.resetSelfPlayClock()
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
                // Mark the start of a wait window on the 0→1 pending
                // transition. `fireBatch` reads this and computes
                // `now - startedNanos` as the cycle's wait time. Set
                // even when count-only fires (no timer arm) — we
                // still want to attribute the small async-hop latency
                // between `scheduleBatchFireIfNeeded` returning and
                // the dispatched `Task` actually entering `fireBatch`.
                if wasEmpty {
                    batchPendingStartedNanos = DispatchTime.now().uptimeNanoseconds
                }
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
                    scheduleBatchFireIfNeeded(
                        reason: expectedSlotCount == 0 ? .drain : .full
                    )
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
            Task.detached { [weak self] in
                await self?.cancelPending(token: token)
            }
        }
    }

    /// Remove the pending submission identified by `token` (if still
    /// queued) and resume its continuation with `CancellationError`.
    /// Called from the cancellation handler in `evaluate`. A no-op if
    /// the entry is not found — that just means `fireBatch` already
    /// claimed and resumed it before this handler reached the actor.
    ///
    /// If this cancellation empties the pending queue, bump the
    /// coalescing-timer generation and clear the wait-window stamp
    /// so the now-defunct cycle's state can't pollute the next one:
    /// a sleeping timer for the prior generation will no-op on wake,
    /// and the next 0→1 transition records a fresh wait start. This
    /// also prevents a `fireBatch` Task that was already dispatched
    /// (count-met or timer-met just before the cancel) from spending
    /// the cycle's accumulated wait on what will be an empty batch.
    private func cancelPending(token: UUID) {
        guard let idx = pending.firstIndex(where: { $0.token == token }) else {
            return
        }
        let item = pending.remove(at: idx)
        item.continuation.resume(throwing: CancellationError())

        if pending.isEmpty {
            batchWaitGeneration &+= 1
            batchPendingStartedNanos = 0
        }
    }

    private func scheduleBatchFireIfNeeded(reason: FireReason) {
        guard !batchFireScheduled else { return }
        batchFireScheduled = true
        // Bump the generation here as well so any in-flight timer
        // task for this cycle wakes up and sees its generation is
        // stale. Without the bump, the count path firing would still
        // leave a sleeping timer that, on wake-up, would try to fire
        // an empty (or freshly partial) queue.
        batchWaitGeneration &+= 1
        // Tag this scheduled fire with the reason that originated it.
        // We count at the schedule site rather than at fire time so
        // the `batchFireScheduled` guard above is the natural
        // deduplicator — `fireBatch` itself runs once per scheduled
        // fire (modulo the cancellation race that lands an empty
        // batch, which we still want to count as a schedule).
        fireReasonCounts[reason, default: 0] += 1
        Task.detached { [weak self] in
            await self?.fireBatch()
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
        Task.detached { [weak self] in
            do {
                try await Task.sleep(nanoseconds: waitNs)
            } catch {
                // `Task.sleep` only throws on cancellation. If the
                // timer task is cancelled (e.g. enclosing arena run
                // is winding down) we just bail — the actor's drain
                // path on `setExpectedSlotCount(0)` flushes any
                // leftover pending entries, so there's nothing for
                // a stale timer to do.
                return
            }
            guard let self else { return }
            await self.fireIfStillCurrent(generation: generation)
        }
    }

    private func fireIfStillCurrent(generation: Int) {
        guard generation == batchWaitGeneration else { return }
        guard !pending.isEmpty else { return }
        scheduleBatchFireIfNeeded(reason: .timer)
    }

    private func fireBatch() async {
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        batchFireScheduled = false

        // Cancellation race: between `scheduleBatchFireIfNeeded`
        // setting `batchFireScheduled = true` (and dispatching this
        // Task) and this Task actually entering the actor,
        // `cancelPending` may have removed the only pending entry.
        // We land here with an empty batch — not a bug, just a race
        // we have to tolerate. The cancelled slot has already been
        // resumed-with-error by `cancelPending`, and `cancelPending`
        // also bumped `batchWaitGeneration` and cleared
        // `batchPendingStartedNanos`, so there's nothing to do here:
        // no continuations to resume, no telemetry to record (we
        // shouldn't attribute wait time to a cycle that produced no
        // batch), no fire to schedule. Just return.
        guard !batch.isEmpty else {
            return
        }

        // Close the wait window and accumulate. We measure wait at
        // fireBatch entry rather than after the await so the wait
        // window terminates the moment the batcher starts owning the
        // batch — even before the GPU call begins. Done after the
        // empty-batch guard so we never attribute wait time to a
        // cycle that didn't produce a batch.
        let fireStartedNanos = DispatchTime.now().uptimeNanoseconds
        var thisWaitNanos: UInt64 = 0
        if batchPendingStartedNanos > 0, fireStartedNanos > batchPendingStartedNanos {
            thisWaitNanos = fireStartedNanos - batchPendingStartedNanos
            batchTotalWaitNanos &+= thisWaitNanos
        }
        batchPendingStartedNanos = 0

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

        // Snapshot `expectedSlotCount` before the await so we can
        // detect drift across the GPU window. Any change here is
        // typically driven by `decrementExpectedSlotCount` calls from
        // the harvest loop as games end during the fire. Symmetric
        // drift across cand/champ is benign; a large `maxDelta` or
        // asymmetric drift would point at the coordination paths
        // between the harvest loop and the batcher actor (cf. the
        // 991282b "expectedSlotCount drift" commit).
        let expectedBeforeRun = expectedSlotCount
        // `gpuTimerEnded` lets us guarantee `fireEnded()` is called
        // exactly once whether `network.evaluate` returns normally OR
        // throws. Declared outside the do-block so the catch path can
        // observe it. The shape-mismatch `throw`s further down also
        // route through the same catch but happen *after*
        // `fireEnded()` has already run, so the catch path no-ops in
        // that case.
        gpuTimer?.fireStarted()
        var gpuTimerEnded = false

        do {
            // Pass `packBuffer` (an owned Swift `[Float]`) directly
            // across the `await`. Swift's COW guarantees the buffer
            // stays intact for the duration of the evaluation even if
            // a reentrant `fireBatch` runs on this actor at a
            // suspension point and resizes/rewrites our `packBuffer`
            // stored property — the first mutating access there would
            // uniquify a fresh buffer, leaving the one captured by the
            // in-flight `network.evaluate` call untouched.
            let runStartedNanos = DispatchTime.now().uptimeNanoseconds
            let (policy, values) = try await network.evaluate(
                batchBoards: packBuffer,
                count: count
            )
            let runFinishedNanos = DispatchTime.now().uptimeNanoseconds
            gpuTimer?.fireEnded()
            gpuTimerEnded = true
            let thisRunNanos: UInt64 = runFinishedNanos > runStartedNanos
                ? runFinishedNanos - runStartedNanos
                : 0
            batchTotalRunNanos &+= thisRunNanos
            if expectedSlotCount != expectedBeforeRun {
                expectedDriftCount &+= 1
                let delta = abs(expectedSlotCount - expectedBeforeRun)
                if delta > expectedDriftMaxDelta {
                    expectedDriftMaxDelta = delta
                }
            }

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

            // Per-batch debug line for arena (double-gated). Compile-
            // time `DCM_LOG_PER_BATCH_TIMINGS` elides this entire
            // block from the binary by default — the per-arena log
            // volume isn't worth carrying day-to-day. To turn on,
            // add `-D DCM_LOG_PER_BATCH_TIMINGS` to OTHER_SWIFT_FLAGS
            // and rebuild. The runtime `logBatchTimings` gate then
            // selects WHICH batchers emit (arena yes, self-play no)
            // when the compile flag is on. The wait/run accumulators
            // above ALWAYS update so the end-of-arena summary line
            // (`[ARENA] timing ...`) keeps working with the flag off.
            #if DCM_LOG_PER_BATCH_TIMINGS
            if logBatchTimings {
                let thisRunMs = Double(thisRunNanos) / 1_000_000.0
                let thisWaitMs = Double(thisWaitNanos) / 1_000_000.0
                let avgRunMs = batchSizeTotalBatches > 0
                    ? Double(batchTotalRunNanos) / 1_000_000.0 / Double(batchSizeTotalBatches)
                    : 0
                let avgWaitMs = batchSizeTotalBatches > 0
                    ? Double(batchTotalWaitNanos) / 1_000_000.0 / Double(batchSizeTotalBatches)
                    : 0
                let label = name.isEmpty ? "" : " \(name)"
                SessionLogger.shared.log(String(
                    format: "[ARENA-BATCH]%@ #%d size=%d wait=%.1fms run=%.1fms avgWait=%.1fms avgRun=%.1fms",
                    label,
                    batchSizeTotalBatches,
                    count,
                    thisWaitMs,
                    thisRunMs,
                    avgWaitMs,
                    avgRunMs
                ))
            }
            #endif

            #if DEBUG
            runBatchCorrectnessCheckIfNeeded(batch: batch, policy: policy, values: values)
            #endif

            // Report this tick to the replay-ratio controller. One
            // barrier fire = one "measurement event" on the self-play
            // side. Drain-mode fires (`expectedSlotCount == 0`) are
            // skipped — those are shutdown / arena-pause stragglers,
            // not representative of steady-state production rate.
            // The controller owns its own wall-clock stamp internally
            // and treats its first call after a reset as
            // measurement-less (just stamps the time), so a re-attach
            // after a session break never produces a garbage sample.
            if let controller = replayRatioController, expectedSlotCount > 0 {
                // Controller now owns the inter-tick wall clock
                // directly. We just report the current per-game
                // spDelay setting (what the driver is sleeping
                // between games), and the controller subtracts it
                // from its own measured wall.
                let spDelayMs = Double(controller.computedSelfPlayDelayMs)
                controller.recordSelfPlayBarrierTick(
                    positionsProduced: count,
                    currentDelaySettingMs: spDelayMs,
                    workerCount: expectedSlotCount
                )
            }

            for (i, item) in batch.enumerated() {
                let start = i * Self.policySize
                let end = start + Self.policySize
                let policySlice = Array(policy[start..<end])
                let value = values[i]
                item.continuation.resume(returning: (policy: policySlice, value: value))
            }
        } catch {
            // Two error sources route here: (a) `network.evaluate`
            // itself threw — `fireEnded()` was NOT called yet — and
            // (b) the post-evaluate shape-mismatch `throw`s above —
            // `fireEnded()` already ran. The flag disambiguates so the
            // joint timer's `inFlight` count never gets stuck.
            if !gpuTimerEnded {
                gpuTimer?.fireEnded()
            }
            for item in batch {
                item.continuation.resume(throwing: error)
            }
        }

        if !pending.isEmpty && (expectedSlotCount == 0 || pending.count >= expectedSlotCount) {
            scheduleBatchFireIfNeeded(
                reason: expectedSlotCount == 0 ? .drain : .refill
            )
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
