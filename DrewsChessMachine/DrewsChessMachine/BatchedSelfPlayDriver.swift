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

    // MARK: - Init

    init(
        batcher: BatchedMoveEvaluationSource,
        buffer: ReplayBuffer,
        statsBox: ParallelWorkerStatsBox,
        diversityTracker: GameDiversityTracker,
        countBox: WorkerCountBox,
        pauseGate: WorkerPauseGate,
        gameWatcher: GameWatcher?
    ) {
        self.batcher = batcher
        self.buffer = buffer
        self.statsBox = statsBox
        self.diversityTracker = diversityTracker
        self.countBox = countBox
        self.pauseGate = pauseGate
        self.gameWatcher = gameWatcher
    }

    // MARK: - Driver Loop

    func run() async {
        var slots: [Task<Void, Never>] = []

        while !Task.isCancelled {
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
                    let idx = slots.count
                    // Strong self capture: the driver is owned by the
                    // parent `withTaskGroup`'s closure for the
                    // duration of this child task, so it will not be
                    // deallocated under the slot. Bind through a
                    // local so the capture is unambiguously strong.
                    let driverSelf: BatchedSelfPlayDriver = self
                    let slot = Task(priority: .userInitiated) {
                        await driverSelf.slotLoop(index: idx)
                    }
                    slots.append(slot)
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
                    last.cancel()
                    cancelled.append(last)
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

    private func stopAll(slots: inout [Task<Void, Never>]) async {
        let snapshot = slots
        slots.removeAll()
        for s in snapshot { s.cancel() }
        for s in snapshot { _ = await s.value }
        await batcher.setExpectedSlotCount(0)
    }

    // MARK: - Slot Body

    private func slotLoop(index: Int) async {
        // Reusable players per slot — `ChessMachine.beginNewGame` calls
        // `onNewGame` on each before starting, resetting per-game scratch
        // without reallocating the board / policy / value buffers.
        let white = MPSChessPlayer(
            name: "White",
            source: batcher,
            replayBuffer: buffer,
            schedule: .selfPlay
        )
        let black = MPSChessPlayer(
            name: "Black",
            source: batcher,
            replayBuffer: buffer,
            schedule: .selfPlay
        )

        while !Task.isCancelled {
            // Live-display gating matches today's worker logic: slot 0
            // animates the board if and only if exactly one slot is
            // active. Evaluated per game (not at spawn) so a Stepper
            // toggle between 1 and >1 kicks in on the next game rather
            // than being latched at spawn time.
            let liveDisplay = index == 0 && countBox.count == 1

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
                let task = try machine.beginNewGame(white: white, black: black)
                result = await task.value
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
        }
    }
}
