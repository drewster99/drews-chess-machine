import Foundation

// MARK: - Batched Self-Play Driver

/// Drives N concurrent `ChessMachine` self-play games against one shared
/// `BatchedMoveEvaluationSource` (the barrier batcher).
///
/// Each slot is an unstructured `Task` that loops forever: allocate a
/// fresh `ChessMachine`, run one game to termination against white/black
/// `MPSChessPlayer`s pointed at the shared batcher, record stats +
/// diversity, repeat. Every per-ply `evaluate` call on every slot parks
/// in the batcher; when the N-th submission arrives the batcher fires a
/// single `network.evaluate(batchBoards:count:)` and resumes all N slots
/// with their own `(policy, value)`. Batch size is exactly N every time.
///
/// The outer `run()` loop reconciles slot count to `countBox.count` (the
/// Stepper-driven live worker count) and handles arena pause requests
/// via `pauseGate`:
///
/// - Grow: bump `batcher.setExpectedSlotCount(newN)` *before* spawning
///   a new slot Task so its first submission is counted.
/// - Shrink: cancel the last `K` slot Tasks (each exits at its next
///   game boundary, because `ChessMachine.runGameLoop` doesn't check
///   `Task.isCancelled` mid-game), await their exits, then bump
///   `setExpectedSlotCount(newN)` down. During the in-flight window
///   the remaining live slots keep firing the barrier at the old
///   count — slot exit happens only after the last game's final move.
/// - Pause (arena): cancel every slot, await exits, set expected count
///   to 0, `markWaiting()` on the gate, spin-wait until `resume()`.
///   On resume, respawn to the current target count.
///
/// Slot-task bodies mirror today's worker body in `ContentView`
/// (`realTrainingTask` / `withTaskGroup`), just with the
/// `MoveEvaluationSource` in place of per-worker networks and without
/// per-worker pause gates.
final class BatchedSelfPlayDriver: @unchecked Sendable {
    // MARK: - Dependencies

    let batcher: BatchedMoveEvaluationSource
    let buffer: ReplayBuffer
    let statsBox: ParallelWorkerStatsBox
    let diversityTracker: GameDiversityTracker
    let countBox: WorkerCountBox
    let pauseGate: WorkerPauseGate
    let gameWatcher: GameWatcher?
    /// Live self-play schedule source. Read at each new-game boundary
    /// inside `slotLoop` so UI edits take effect on the next game a
    /// slot starts, without killing the long-lived slot task.
    let scheduleBox: SamplingScheduleBox
    /// Optional source of the self-play per-game sleep. When the
    /// replay-ratio auto-adjuster decides training is the bottleneck
    /// (GPU overhead exceeds the target cycle even at zero training
    /// delay), it flips the sign of its signed delay state and asks
    /// self-play to slow down instead. The slot loop reads
    /// `controller.computedSelfPlayDelayMs` at the bottom of each
    /// game and sleeps for that many ms before starting the next
    /// game. `nil` (and zero) mean no extra delay — preserves the
    /// original "as fast as the batcher allows" behavior.
    let replayRatioController: ReplayRatioController?
    private let liveSlots = LiveSlotSet()
    private var nextSlotID: Int = 0

    // MARK: - Init

    init(
        batcher: BatchedMoveEvaluationSource,
        buffer: ReplayBuffer,
        statsBox: ParallelWorkerStatsBox,
        diversityTracker: GameDiversityTracker,
        countBox: WorkerCountBox,
        pauseGate: WorkerPauseGate,
        gameWatcher: GameWatcher?,
        scheduleBox: SamplingScheduleBox,
        replayRatioController: ReplayRatioController? = nil
    ) {
        self.batcher = batcher
        self.buffer = buffer
        self.statsBox = statsBox
        self.diversityTracker = diversityTracker
        self.countBox = countBox
        self.pauseGate = pauseGate
        self.gameWatcher = gameWatcher
        self.scheduleBox = scheduleBox
        self.replayRatioController = replayRatioController
    }

    // MARK: - Driver Loop

    func run() async {
        var slots: [(id: Int, task: Task<Void, Never>)] = []

        while !Task.isCancelled {
            slots.removeAll { !liveSlots.contains($0.id) }

            // Arena pause: stop all slots, park at the gate.
            if pauseGate.isRequestedToPause {
                await stopAll(slots: &slots)
                pauseGate.markWaiting()
                while pauseGate.isRequestedToPause && !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(5))
                }
                pauseGate.markRunning()
                if Task.isCancelled { break }
                // Fall through; the next iteration will reconcile slot count.
                continue
            }

            // Reconcile slot count to the live target.
            let target = countBox.count
            if slots.count < target {
                // Grow: expand barrier count first, then spawn. The new
                // slot's first submission is counted toward the new
                // barrier threshold on arrival.
                await batcher.setExpectedSlotCount(target)
                while slots.count < target {
                    let slotID = nextSlotID
                    nextSlotID += 1
                    liveSlots.insert(slotID)
                    // Strong self capture: the driver is owned by the
                    // parent `withTaskGroup`'s closure for the
                    // duration of this child task, so it will not be
                    // deallocated under the slot. Bind through a
                    // local so the capture is unambiguously strong.
                    let driverSelf: BatchedSelfPlayDriver = self
                    let slot = Task(priority: .userInitiated) {
                        await driverSelf.slotLoop(id: slotID)
                    }
                    slots.append((id: slotID, task: slot))
                }
            } else if slots.count > target {
                // Shrink: lower the barrier threshold FIRST. If we
                // cancelled slots and then waited, the first cancelled
                // slot to exit its game would remove a barrier
                // contributor while `expectedSlotCount` was still the
                // old count — the remaining slots would then stall
                // forever waiting for a submission that never comes.
                // Lowering the threshold first means each subsequent
                // submission during the shrink transition fires its
                // own small batch immediately (batch size 1..=target
                // depending on actor mailbox timing), which is fine —
                // we trade a little GPU efficiency for liveness.
                await batcher.setExpectedSlotCount(target)
                let toRemove = slots.count - target
                var cancelled: [Task<Void, Never>] = []
                cancelled.reserveCapacity(toRemove)
                for _ in 0..<toRemove {
                    let last = slots.removeLast()
                    last.task.cancel()
                    cancelled.append(last.task)
                }
                for t in cancelled { _ = await t.value }
            }

            // Idle tick when there's nothing to do. If target is 0 the
            // session is essentially paused; poll at 50 ms so a Stepper
            // bump to 1+ picks up promptly. Otherwise steady-state slots
            // are running and the driver just needs to re-check pause /
            // count on the next tick.
            let sleepMs = slots.isEmpty ? 50 : 100
            try? await Task.sleep(for: .milliseconds(sleepMs))
        }

        // Session cancelled — cancel every slot, wait for them to finish
        // their current games, and drain the barrier.
        await stopAll(slots: &slots)
    }

    private func stopAll(slots: inout [(id: Int, task: Task<Void, Never>)]) async {
        let snapshot = slots
        slots.removeAll()
        // Drop the barrier threshold to 0 (drain mode) BEFORE waiting
        // on the cancelled slot tasks. Otherwise the first cancelled
        // slot to finish its current game and exit would stop
        // contributing submissions at the old threshold, and the
        // remaining slots would stall at the barrier — their games
        // frozen mid-ply because the batcher can never reach the old
        // count again. Drain mode lets each in-flight submission fire
        // as its own micro-batch, so every slot's current game runs
        // to completion and the slot can exit cleanly.
        await batcher.setExpectedSlotCount(0)
        for s in snapshot { s.task.cancel() }
        for s in snapshot { _ = await s.task.value }
    }

    // MARK: - Slot Body

    private func slotLoop(id: Int) async {
        defer { liveSlots.remove(id) }
        // Reusable players per slot — `ChessMachine.beginNewGame` calls
        // `onNewGame` on each before starting, resetting per-game scratch
        // without reallocating the board / policy / value buffers. The
        // `schedule` is refreshed at the top of each game from
        // `scheduleBox` so UI edits to tau start / floor / decay take
        // effect on the next game a slot starts, without re-allocating
        // the per-player scratch.
        let white = MPSChessPlayer(
            name: "White",
            source: batcher,
            replayBuffer: buffer,
            schedule: scheduleBox.selfPlay
        )
        let black = MPSChessPlayer(
            name: "Black",
            source: batcher,
            replayBuffer: buffer,
            schedule: scheduleBox.selfPlay
        )
        // Stamp the slot id onto each player so per-position
        // observability metadata can attribute samples back to which
        // self-play slot produced them. UInt8 caps at 255; clamp.
        let stampedWorkerId = UInt8(min(id, Int(UInt8.max)))
        white.workerId = stampedWorkerId
        black.workerId = stampedWorkerId

        while !Task.isCancelled {
            // Refresh schedule between games so live UI edits propagate
            // without having to restart the session. Safe here because
            // the player is not yet inside a `beginNewGame` call.
            let liveSchedule = scheduleBox.selfPlay
            white.schedule = liveSchedule
            black.schedule = liveSchedule

            // Live-display gating matches today's worker logic: when
            // exactly one slot is active, that slot animates the board.
            // Evaluated per game (not at spawn) so a Stepper toggle
            // between 1 and >1 kicks in on the next game rather than
            // being latched at spawn time.
            let liveDisplay = countBox.count == 1

            if liveDisplay {
                gameWatcher?.resetCurrentGame()
                gameWatcher?.markPlaying(true)
            }

            let machine = ChessMachine()
            if liveDisplay {
                machine.delegate = gameWatcher
            }

            let gameStart = CFAbsoluteTimeGetCurrent()
            let result: GameResult
            do {
                result = try await machine.beginNewGame(white: white, black: black)
            } catch is CancellationError {
                // Slot task was cancelled (driver shrink, arena pause,
                // session stop). `beginNewGame` threw before emitting
                // `onGameEnded`, so `MPSChessPlayer`'s partial-game
                // scratch was not flushed to the replay buffer — no
                // fake-stalemate pollution. Skip stats recording for
                // the incomplete game and exit the slot.
                if liveDisplay {
                    gameWatcher?.markPlaying(false)
                }
                break
            } catch {
                if liveDisplay {
                    gameWatcher?.markPlaying(false)
                }
                break
            }
            let gameDurationMs = (CFAbsoluteTimeGetCurrent() - gameStart) * 1000

            // Aggregate game stats — every slot contributes identically.
            // Move count = plies recorded by both players, read before
            // the next `beginNewGame` resets them.
            let positions = white.recordedPliesCount + black.recordedPliesCount
            statsBox.recordCompletedGame(
                moves: positions,
                durationMs: gameDurationMs,
                result: result
            )
            diversityTracker.recordGame(moves: machine.moveHistory)

            // Metadata-only feed to the controller: every completed
            // game refreshes `positionsPerGame` (used only to convert
            // the sp-slowdown signed delay into a per-worker per-game
            // sleep). The rate measurement ITSELF comes from the
            // batcher's per-barrier-tick hook — much finer
            // granularity and uncontaminated by applied sleeps.
            replayRatioController?.recordSelfPlayGameLength(positions)

            // Replay-ratio self-play throttle. Applied after a game
            // has fully flushed (recordCompletedGame + recordGame) and
            // *before* the next `beginNewGame`, so no partial-game
            // state ever sits suspended across the sleep.
            //
            // Coupling note: while this slot sleeps between games,
            // it is not submitting to the batcher. The other N-1
            // slots will continue hitting the batcher's barrier
            // (which still expects N submissions because expected
            // slot count is unchanged), so they park at the barrier
            // until this slot resumes and makes its first submission
            // of the next game. Net effect: one slot's between-game
            // sleep becomes a brief all-N-slot pause at the barrier,
            // amplifying the production-rate reduction. This is
            // benign — the controller's feedback loop converges on
            // whatever sleep value actually produces the target
            // ratio, regardless of coupling — and it is not a
            // deadlock risk: cancellation interrupts `Task.sleep`
            // immediately, and arena-pause / session-stop both drop
            // `expectedSlotCount` to 0 via `stopAll` before awaiting
            // slot exits, which releases any parked barrier slots.
            //
            // A zero value (controller in the training-slowdown
            // regime, or no controller supplied) skips the sleep
            // entirely so this adds no overhead in the common case.
            if let selfPlayDelayMs = replayRatioController?.computedSelfPlayDelayMs,
               selfPlayDelayMs > 0 {
                try? await Task.sleep(for: .milliseconds(selfPlayDelayMs))
            }
        }
    }
}

private final class LiveSlotSet: @unchecked Sendable {
    private let lock = NSLock()
    private var ids: Set<Int> = []

    func insert(_ id: Int) {
        lock.lock()
        ids.insert(id)
        lock.unlock()
    }

    func remove(_ id: Int) {
        lock.lock()
        ids.remove(id)
        lock.unlock()
    }

    func contains(_ id: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ids.contains(id)
    }
}
