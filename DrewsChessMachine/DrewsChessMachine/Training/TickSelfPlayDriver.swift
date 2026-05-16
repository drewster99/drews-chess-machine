import Foundation
import os

// MARK: - Tick-Based Self-Play Driver

/// Single-task self-play driver that advances K concurrent games in
/// lockstep, one ply per "tick". Replaces the `BatchedSelfPlayDriver`'s
/// "one unstructured `Task` per concurrent game + actor barrier
/// batcher" topology with a much cheaper "one driver task + vector of
/// game-state structs" model.
///
/// **Why.** The legacy driver allocates `~7.5 MB` of per-`MPSChessPlayer`
/// `gameBoardScratchPtr` per slot (2 players × 3.75 MB at the 512-ply
/// pre-allocation cap), parks K Swift tasks at an actor barrier per
/// ply, and runs a delicate `setExpectedSlotCount` deadlock-avoidance
/// dance on grow/shrink. At K=4096 that's ~30 GB just for board-history
/// scratch and 4096 long-lived Tasks. The tick driver collapses every
/// game's state into an `ActiveGame` (~1.2 MB at the typical
/// `maxPliesPerGame=150` cap) and uses zero actor coordination — the
/// driver task is the only caller of the network.
///
/// **The tick loop.** One iteration of `runOneTick`:
///
/// 1. Parallel-encode the current position of each of K games into
///    the driver-owned `tickScratch` buffer, distributed across P
///    worker tasks via `withTaskGroup`. P defaults to
///    `ProcessInfo.processInfo.activeProcessorCount`.
/// 2. Issue one batched `network.evaluateBatched(...)` call. The
///    consume closure memcpy's the full K-position policy slice into
///    the driver's `policyResultScratch` and the K value scalars into
///    `valueResultScratch`, then returns. (Doing the sample inside
///    `consume` would serialize the K samples on the network's
///    executionQueue; copying out and resuming the driver task lets
///    the next step parallelize.)
/// 3. Parallel sample + apply: P worker tasks each handle a strided
///    slice of K games. For each game: call
///    `MoveSampler.sampleMove(...)` on its policy slice (with its
///    own slice of `samplerProbsScratch` / `samplerEtaScratch`),
///    record the ply on the `ActiveGame`, apply the move via
///    `engine.applyMoveAndAdvance(_:)`. K samples done concurrently.
/// 4. Game-end pass: serial walk over `games[0..<K]`. For each
///    game that terminated (engine.result != nil) or hit max-plies:
///    record stats, apply draw-keep filter, flush to the replay
///    buffer, reset the slot with a freshly-read
///    `TrainingParameters.shared.maxPliesPerGame` cap and the
///    current `scheduleBox.selfPlay`.
/// 5. Optional replay-ratio throttle (proportional to how many
///    games finished this tick — preserves the integrated rate of
///    the legacy per-game sleep).
/// 6. Inform the controller of the tick: `recordSelfPlayBarrierTick(...)`.
///
/// **Live-display path.** When K == 1, after sampling each move the
/// driver also calls `gameWatcher?.onMoveApplied(...)` and at game-end
/// `gameWatcher?.onGameEnded(...)` — same shape the legacy delegate
/// path emitted, just without the `ChessMachine` indirection. K > 1
/// suppresses these calls (matches the legacy driver's
/// `liveDisplay = countBox.count == 1` check).
///
/// **Live K change.** `games.count` is the live worker count. Grow:
/// `append` new `ActiveGame`s (and grow scratch if K exceeds the
/// current capacity). Shrink: `removeLast` the tail games (in-flight
/// partial plies are dropped — matches the legacy driver's
/// cancellation-mid-game behavior). No `expectedSlotCount` to
/// coordinate, no Task cancellations, no deadlock window.
///
/// **Arena pause.** Polled at top of every loop iteration. On request:
/// drop all in-flight games (no flush), `pauseGate.markWaiting()`,
/// spin until `!pauseGate.isRequestedToPause`, `pauseGate.markRunning()`.
/// Mirrors the legacy driver's pause semantics exactly.
///
/// **Self-play vs arena.** This phase ships only the self-play wiring:
/// all `ActiveGame`s get both their `whiteNetwork` and `blackNetwork`
/// set to the same champion `network`. The arena port (Phase 7) will
/// pass different networks per side, and `runOneTick` will partition
/// the K games into sub-batches keyed by `game.currentNetwork` and
/// fire one `evaluateBatched` call per unique network. For now there's
/// exactly one unique network so the partitioning collapses to a
/// single batched call covering all K positions.
final class TickSelfPlayDriver: SelfPlayDriverProtocol, @unchecked Sendable {

    // MARK: - Dependencies

    let network: ChessMPSNetwork
    let buffer: ReplayBuffer
    let statsBox: ParallelWorkerStatsBox
    let diversityTracker: GameDiversityTracker
    let countBox: WorkerCountBox
    let pauseGate: WorkerPauseGate
    let gameWatcher: GameWatcher?
    let scheduleBox: SamplingScheduleBox
    let replayRatioController: ReplayRatioController?

    // MARK: - Private state (driver-task-owned, no lock needed)

    private var games: [ActiveGame] = []
    private var nextWorkerId: UInt16 = 0

    /// Pre-allocated per-tick board-encoding scratch. Sized to
    /// `tickScratchCapK × BoardEncoder.tensorLength` floats. Grows on
    /// the first tick that needs more capacity; never shrinks within
    /// a session (avoid thrash on Stepper toggles).
    private var tickScratch: UnsafeMutablePointer<Float>?
    private var tickScratchCapK: Int = 0

    /// Per-tick policy readback: `tickScratchCapK × ChessNetwork.policySize`
    /// floats. The network's consume closure memcpy's into here so the
    /// subsequent parallel-sample pass can read without crossing the
    /// network's executionQueue. Same growth policy as `tickScratch`.
    private var policyResultScratch: UnsafeMutablePointer<Float>?

    /// Per-tick value readback: `tickScratchCapK` floats. Currently
    /// unused by the sampler (the W/D/L value head's scalar is for the
    /// trainer's policy-gradient baseline, computed off the replay
    /// buffer — not consumed at sample time). Sized symmetrically so a
    /// future use can lift directly out of it without re-plumbing.
    private var valueResultScratch: UnsafeMutablePointer<Float>?

    /// Per-tick sampler scratches (one slice per game per pass).
    /// Sized `tickScratchCapK × MoveSampler.scratchCapacity` floats
    /// each. Game `i`'s sample uses bytes `i*256..<(i+1)*256` —
    /// distinct per game so parallel sampling has no aliasing.
    private var samplerProbsScratch: UnsafeMutablePointer<Float>?
    private var samplerEtaScratch: UnsafeMutablePointer<Float>?

    // MARK: - Init

    init(
        network: ChessMPSNetwork,
        buffer: ReplayBuffer,
        statsBox: ParallelWorkerStatsBox,
        diversityTracker: GameDiversityTracker,
        countBox: WorkerCountBox,
        pauseGate: WorkerPauseGate,
        gameWatcher: GameWatcher?,
        scheduleBox: SamplingScheduleBox,
        replayRatioController: ReplayRatioController? = nil
    ) {
        self.network = network
        self.buffer = buffer
        self.statsBox = statsBox
        self.diversityTracker = diversityTracker
        self.countBox = countBox
        self.pauseGate = pauseGate
        self.gameWatcher = gameWatcher
        self.scheduleBox = scheduleBox
        self.replayRatioController = replayRatioController
    }

    deinit {
        let boardFloats = BoardEncoder.tensorLength
        if let p = tickScratch {
            p.deinitialize(count: tickScratchCapK * boardFloats); p.deallocate()
        }
        if let p = policyResultScratch {
            p.deinitialize(count: tickScratchCapK * ChessNetwork.policySize); p.deallocate()
        }
        if let p = valueResultScratch {
            p.deinitialize(count: tickScratchCapK); p.deallocate()
        }
        if let p = samplerProbsScratch {
            p.deinitialize(count: tickScratchCapK * MoveSampler.scratchCapacity); p.deallocate()
        }
        if let p = samplerEtaScratch {
            p.deinitialize(count: tickScratchCapK * MoveSampler.scratchCapacity); p.deallocate()
        }
    }

    // MARK: - Driver Loop

    func run() async {
        let P = max(1, ProcessInfo.processInfo.activeProcessorCount)
        SessionLogger.shared.log("[SP-TICK] driver starting, P=\(P)")

        while !Task.isCancelled {
            // 1. Arena pause check.
            if pauseGate.isRequestedToPause {
                // Drop in-flight games (no flush — matches legacy
                // behavior on arena pause). The next iteration after
                // resume will re-create games to match countBox.count.
                games.removeAll(keepingCapacity: true)
                pauseGate.markWaiting()
                while pauseGate.isRequestedToPause && !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(5))
                }
                pauseGate.markRunning()
                continue
            }

            // 2. Reconcile K to the live target.
            let targetK = countBox.count
            if games.count < targetK {
                ensureScratchCapacity(targetK)
                let cap = await MainActor.run { TrainingParameters.shared.selfPlayMaxPliesPerGame }
                let liveSchedule = scheduleBox.selfPlay
                while games.count < targetK {
                    let g = ActiveGame(
                        workerId: nextWorkerId,
                        whiteNetwork: network,
                        blackNetwork: network,
                        capPlies: cap,
                        schedule: liveSchedule
                    )
                    nextWorkerId &+= 1
                    g.resetForNewGame(maxPliesCap: cap, schedule: liveSchedule)
                    games.append(g)
                }
            } else if games.count > targetK {
                games.removeLast(games.count - targetK)
            }

            // 3. Idle when target is zero.
            if games.isEmpty {
                try? await Task.sleep(for: .milliseconds(50))
                continue
            }

            // 4. The tick.
            await runOneTick(P: P)
        }

        SessionLogger.shared.log("[SP-TICK] driver exiting")
        games.removeAll(keepingCapacity: true)
    }

    // MARK: - Per-tick body

    private func runOneTick(P: Int) async {
        let K = games.count
        let boardFloats = BoardEncoder.tensorLength
        guard let scratch = tickScratch,
              let policyOut = policyResultScratch,
              let valueOut = valueResultScratch,
              let probsBase = samplerProbsScratch,
              let etaBase = samplerEtaScratch
        else {
            // `ensureScratchCapacity` runs whenever games.count grows,
            // so this can only fire if K became > 0 without a grow
            // pass — defensive: skip the tick.
            return
        }

        // (a) Parallel encode. Each strided-slice task writes K_local
        //     boards into its slice of `scratch`. Pointers and
        //     `[ActiveGame]` are wrapped/snapshotted into Sendable
        //     locals so the closure captures Sendable-only state and
        //     the compiler doesn't have to reason about `self`.
        let scratchCarrier = MutablePointerCarrier(pointer: scratch)
        let gameRefs = games
        await withTaskGroup(of: Void.self) { group in
            for p in 0..<P {
                group.addTask {
                    var i = p
                    while i < K {
                        let g = gameRefs[i]
                        let dst = scratchCarrier.pointer + i * boardFloats
                        BoardEncoder.encode(
                            g.engine.state,
                            into: UnsafeMutableBufferPointer(start: dst, count: boardFloats)
                        )
                        i += P
                    }
                }
            }
        }

        // (b) One batched GPU forward. consume copies the full K-position
        //     policy + values into our scratch and returns. The pointer
        //     targets are wrapped in `MutablePointerCarrier` (an
        //     `@unchecked Sendable` shim, mirroring `PolicyDestination`)
        //     so they can cross into the `@Sendable` consume closure
        //     without the compiler refusing to capture
        //     `UnsafeMutablePointer<Float>` directly.
        let floatCount = K * boardFloats
        let policyTarget = MutablePointerCarrier(pointer: policyOut)
        let valueTarget = MutablePointerCarrier(pointer: valueOut)
        let policySize = ChessNetwork.policySize
        do {
            try await network.evaluateBatched(
                batchBoardsPointer: UnsafePointer(scratch),
                floatCount: floatCount,
                count: K
            ) { policyBuf, valueBuf in
                guard let pBase = policyBuf.baseAddress, let vBase = valueBuf.baseAddress else {
                    return
                }
                policyTarget.pointer.update(from: pBase, count: K * policySize)
                valueTarget.pointer.update(from: vBase, count: K)
            }
        } catch {
            SessionLogger.shared.log("[SP-TICK] network error: \(error); skipping tick")
            return
        }

        // (c) Parallel sample + apply. Per game `i`:
        //       - slice policyOut at [i*policySize ..< (i+1)*policySize]
        //       - call MoveSampler.sampleMove with that game's slice
        //         of probs/eta scratches
        //       - capture the side-to-move BEFORE applying so recordPly
        //         attributes the ply to the correct color
        //       - apply the move on the engine
        //       - if K == 1, capture the move to emit to GameWatcher
        //         after this pass (single-game live display path)
        //     Sampling state mutations (game.recordPly, game.engine
        //     advance, game.randomishCount) are confined to slot `i`
        //     across all tasks — no aliasing between tasks because the
        //     stride is `P` and each slot's `ActiveGame` is touched by
        //     exactly one task.
        let policyCarrier = MutablePointerCarrier(pointer: policyOut)
        let probsCarrier = MutablePointerCarrier(pointer: probsBase)
        let etaCarrier = MutablePointerCarrier(pointer: etaBase)
        // scratchCarrier already declared above (encode pass) — reuse.
        await withTaskGroup(of: Void.self) { group in
            for p in 0..<P {
                group.addTask {
                    var i = p
                    while i < K {
                        let g = gameRefs[i]
                        // The game may already be in a terminal state
                        // if a previous tick ended it; the game-end
                        // pass clears them but processes them on the
                        // SAME tick (after this one). Skip to avoid
                        // applyMoveAndAdvance throwing gameAlreadyOver.
                        if g.engine.result != nil {
                            i += P
                            continue
                        }
                        let legalMoves = g.engine.currentLegalMoves
                        if legalMoves.isEmpty {
                            i += P
                            continue
                        }
                        let sideToMove = g.engine.state.currentPlayer
                        let policySliceStart = policyCarrier.pointer.advanced(by: i * policySize)
                        let policySliceBuf = UnsafeBufferPointer<Float>(
                            start: policySliceStart, count: policySize
                        )
                        let probsSliceStart = probsCarrier.pointer.advanced(by: i * MoveSampler.scratchCapacity)
                        let probsSliceBuf = UnsafeMutableBufferPointer<Float>(
                            start: probsSliceStart, count: MoveSampler.scratchCapacity
                        )
                        let etaSliceStart = etaCarrier.pointer.advanced(by: i * MoveSampler.scratchCapacity)
                        let etaSliceBuf = UnsafeMutableBufferPointer<Float>(
                            start: etaSliceStart, count: MoveSampler.scratchCapacity
                        )

                        // Per-side ply count for tau + Dirichlet ply
                        // limit. Matches what MPSChessPlayer passes
                        // today (per-player gamePliesRecorded).
                        let sidePlyIndex: Int
                        switch sideToMove {
                        case .white: sidePlyIndex = g.whitePliesRecorded
                        case .black: sidePlyIndex = g.blackPliesRecorded
                        }

                        let result = MoveSampler.sampleMove(
                            logits: policySliceBuf,
                            legalMoves: legalMoves,
                            currentPlayer: sideToMove,
                            ply: sidePlyIndex,
                            schedule: g.schedule,
                            probsScratch: probsSliceBuf,
                            etaScratch: etaSliceBuf
                        )

                        // Non-pawn piece count for the material-phase
                        // bucket — same per-ply calculation MPSChessPlayer
                        // does today.
                        var matCount: Int = 0
                        for sq in g.engine.state.board {
                            if let piece = sq, piece.type != .pawn {
                                matCount += 1
                            }
                        }
                        let materialCount = UInt8(min(matCount, Int(UInt8.max)))

                        let plyTau = g.schedule.tau(forPly: sidePlyIndex)
                        g.recordPly(
                            side: sideToMove,
                            encodedBoardSrc: UnsafePointer(scratchCarrier.pointer + i * boardFloats),
                            policyIndex: result.policyIndex,
                            samplingTau: plyTau,
                            materialCount: materialCount
                        )
                        if result.randomish {
                            g.randomishCount += 1
                        }

                        do {
                            try g.engine.applyMoveAndAdvance(result.move)
                        } catch {
                            // Should not fire: result.move came from
                            // engine.currentLegalMoves above. Log
                            // defensively in case a future refactor
                            // breaks the invariant.
                            SessionLogger.shared.log(
                                "[SP-TICK] applyMoveAndAdvance threw on slot \(i): \(error)"
                            )
                        }
                        i += P
                    }
                }
            }
        }

        // (c.5) Live-display emit: when K == 1 the single game's just-
        //       applied move + new state goes to the watcher. We do
        //       this here (post-apply) rather than inside the parallel
        //       loop because GameWatcher takes its own lock and we
        //       want to keep the parallel pass lock-free.
        if K == 1, let g = games.first, let lastMove = g.engine.moveHistory.last {
            gameWatcher?.onMoveApplied(move: lastMove, newState: g.engine.state)
        }

        // (d) Game-end pass.
        let finishedThisTick = await handleGameEnds()

        // (e) Replay-ratio throttle (proportional scaling — see class doc).
        if let delayMs = replayRatioController?.computedSelfPlayDelayMs,
           delayMs > 0,
           finishedThisTick > 0,
           K > 0 {
            // Per-tick analog of the legacy per-game sleep. In steady
            // state with avg-length games, `finishedThisTick / K ≈
            // K · tickRate / (K · ticksPerGame) ≈ 1 / ticksPerGame`,
            // so the integrated delay matches the legacy
            // delayMs-per-game cadence. Min 1 ms so a single finished
            // game still yields control briefly.
            let scaled = max(1, Int(Double(delayMs) * Double(finishedThisTick) / Double(K)))
            try? await Task.sleep(for: .milliseconds(scaled))
        }

        // (f) Tell the controller about this tick.
        replayRatioController?.recordSelfPlayBarrierTick(
            positionsProduced: K,
            currentDelaySettingMs: Double(replayRatioController?.computedSelfPlayDelayMs ?? 0),
            workerCount: K
        )
    }

    // MARK: - Game-end pass

    /// Returns the number of games that completed (naturally OR via
    /// max-plies drop) this tick. Driven by the replay-ratio throttle
    /// scaling in `runOneTick`.
    private func handleGameEnds() async -> Int {
        let K = games.count
        if K == 0 { return 0 }

        // Read main-actor params once per pass — same cadence as the
        // legacy driver (which reads them per finished game). At the
        // few-finished-games-per-tick rate this is unmeasurably cheap.
        let drawKeepFraction: Double
            = await MainActor.run { TrainingParameters.shared.selfPlayDrawKeepFraction }
        let nextMaxPlies: Int
            = await MainActor.run { TrainingParameters.shared.selfPlayMaxPliesPerGame }

        var finished = 0
        for i in 0..<K {
            let g = games[i]
            let natural = g.engine.result
            let droppedMaxPlies = (natural == nil) && (g.totalPliesPlayed >= g.maxPliesCap)
            guard natural != nil || droppedMaxPlies else { continue }
            finished += 1

            let gameDurationMs = (CFAbsoluteTimeGetCurrent() - g.gameStartedAt) * 1000
            let positions = g.totalPliesPlayed

            if let result = natural {
                statsBox.recordCompletedGame(
                    moves: positions, durationMs: gameDurationMs, result: result
                )
                diversityTracker.recordGame(moves: g.engine.moveHistory)
                replayRatioController?.recordSelfPlayGameLength(positions)

                // Live-display final-game emit.
                if K == 1 {
                    let stats = GameStats(
                        totalMoves: positions,
                        whiteMoves: g.whitePliesRecorded,
                        blackMoves: g.blackPliesRecorded,
                        whiteThinkingTimeMs: 0,
                        blackThinkingTimeMs: 0,
                        totalGameTimeMs: gameDurationMs
                    )
                    gameWatcher?.onGameEnded(result: result, finalState: g.engine.state, stats: stats)
                }

                let isDraw: Bool
                switch result {
                case .checkmate: isDraw = false
                case .stalemate,
                     .drawByFiftyMoveRule,
                     .drawByInsufficientMaterial,
                     .drawByThreefoldRepetition:
                    isDraw = true
                }
                let kept: Bool
                if isDraw && drawKeepFraction < 1.0 {
                    kept = Double.random(in: 0..<1) < drawKeepFraction
                } else {
                    kept = true
                }
                if kept {
                    if let flushed = g.flush(buffer: buffer, result: result), flushed.positions > 0 {
                        statsBox.recordEmittedGame(result: result, flushed: flushed)
                        replayRatioController?.recordSelfPlayEmittedGame(positions: flushed.positions)
                    }
                }
                // If !kept: skip flush. The next resetForNewGame zeroes
                // the per-side fill counters and the recorded scratch
                // is dropped on the floor (matches legacy behavior).
            } else {
                // max-plies dropped: stats, diversity, length feed, but
                // no flush and no emitted-game count.
                statsBox.recordDroppedGame(
                    moves: positions, durationMs: gameDurationMs
                )
                diversityTracker.recordGame(moves: g.engine.moveHistory)
                replayRatioController?.recordSelfPlayGameLength(positions)
            }

            // Reset slot for the next game using the latest schedule
            // and the freshly-read cap. Schedule is read once per
            // game-end (per slot) so live UI edits propagate at the
            // same cadence as the legacy driver.
            g.resetForNewGame(maxPliesCap: nextMaxPlies, schedule: scheduleBox.selfPlay)

            // Reset live-display game-start marker for the K==1 path
            // so the next game in the single watched slot is fresh.
            if K == 1 {
                gameWatcher?.resetCurrentGame()
                gameWatcher?.markPlaying(true)
            }
        }
        return finished
    }

    // MARK: - Scratch capacity

    /// Grow all per-tick scratches if K exceeds the current capacity.
    /// Doubling growth so amortized cost stays O(K) over a session;
    /// never shrinks.
    private func ensureScratchCapacity(_ K: Int) {
        if K <= tickScratchCapK { return }
        let newCap = max(K, tickScratchCapK * 2)
        let boardFloats = BoardEncoder.tensorLength
        let policySize = ChessNetwork.policySize
        let scratchCap = MoveSampler.scratchCapacity

        let newTickFloats = newCap * boardFloats
        let newPolicyFloats = newCap * policySize
        let newSamplerFloats = newCap * scratchCap

        let newTick = UnsafeMutablePointer<Float>.allocate(capacity: newTickFloats)
        newTick.initialize(repeating: 0, count: newTickFloats)
        if let old = tickScratch {
            old.deinitialize(count: tickScratchCapK * boardFloats); old.deallocate()
        }
        tickScratch = newTick

        let newPolicy = UnsafeMutablePointer<Float>.allocate(capacity: newPolicyFloats)
        newPolicy.initialize(repeating: 0, count: newPolicyFloats)
        if let old = policyResultScratch {
            old.deinitialize(count: tickScratchCapK * policySize); old.deallocate()
        }
        policyResultScratch = newPolicy

        let newValue = UnsafeMutablePointer<Float>.allocate(capacity: newCap)
        newValue.initialize(repeating: 0, count: newCap)
        if let old = valueResultScratch {
            old.deinitialize(count: tickScratchCapK); old.deallocate()
        }
        valueResultScratch = newValue

        let newProbs = UnsafeMutablePointer<Float>.allocate(capacity: newSamplerFloats)
        newProbs.initialize(repeating: 0, count: newSamplerFloats)
        if let old = samplerProbsScratch {
            old.deinitialize(count: tickScratchCapK * scratchCap); old.deallocate()
        }
        samplerProbsScratch = newProbs

        let newEta = UnsafeMutablePointer<Float>.allocate(capacity: newSamplerFloats)
        newEta.initialize(repeating: 0, count: newSamplerFloats)
        if let old = samplerEtaScratch {
            old.deinitialize(count: tickScratchCapK * scratchCap); old.deallocate()
        }
        samplerEtaScratch = newEta

        tickScratchCapK = newCap
    }
}

// MARK: - SelfPlayDriverProtocol

/// Tiny existential shared by both `BatchedSelfPlayDriver` (legacy
/// task-per-game) and `TickSelfPlayDriver` (new tick-based). The
/// wiring site in `SessionController+Training.swift` picks one based
/// on `TrainingParameters.shared.selfPlayUseTickDriver` and feeds it
/// into the same `withTaskGroup.addTask { await driver.run() }`
/// shape.
protocol SelfPlayDriverProtocol: Sendable {
    func run() async
}

extension BatchedSelfPlayDriver: SelfPlayDriverProtocol {}

// MARK: - Sendable carrier for the consume-closure pointer copy

/// `UnsafeMutablePointer<T>` is not Sendable but we need to copy bytes
/// into driver-owned scratch from inside the `@Sendable` consume
/// closure of `evaluateBatched`. This is the standard
/// `PolicyDestination`-shaped escape hatch: the driver task owns the
/// underlying buffer, the consume closure runs synchronously on the
/// network's executionQueue and completes before the awaiting
/// `evaluateBatched` call returns, so the pointer is valid for the
/// closure's entire lifetime.
private struct MutablePointerCarrier: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
}
