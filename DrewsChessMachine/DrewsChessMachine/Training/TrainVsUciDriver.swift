import Foundation

// MARK: - Train-vs-UCI Driver

/// Tick-based driver that plays the **live trainer** network against a
/// fixed pool of external UCI engines (`UCIArbiter`) and feeds the
/// finished games into the replay buffer — the game-production engine
/// for `--train-vs-uci`.
///
/// **Relationship to the other drivers.** This is a sibling of
/// `BatchedSelfPlayDriver` and the arena's `TickTournamentDriver`, and
/// deliberately a *separate* class rather than a mutation of either —
/// the self-play driver must keep working untouched. It reuses the same
/// building blocks: `ActiveGame` per slot, `BoardEncoder` +
/// `network.evaluateBatched` for the trainer's move, `MoveSampler` for
/// sampling, and `ActiveGame.flush` → `ReplayBuffer` for ingest.
///
/// **Topology.** One process per opponent instance = one game slot (UCI
/// is serial per process). K is fixed at the opponent count — it does
/// not grow/shrink like self-play's `WorkerCountBox`. Each tick:
///
/// 1. Drain any opponent moves that resolved since the last tick and
///    apply them (advancing those slots to the trainer's turn).
/// 2. **Batch the trainer's move** across every slot where it is the
///    trainer's turn and the slot is ready — one `evaluateBatched` call
///    over that compacted sub-batch. This is the GPU win and it batches
///    "however many are ready", exactly like the existing drivers; it
///    does not pad K.
/// 3. Flush finished games and start the next game on their slots.
/// 4. Dispatch the opponent's move for every slot now on the opponent's
///    turn — as an **async Task off the tick barrier** so a slow engine
///    never stalls the trainer batch. Results come back via a `SyncBox`.
///
/// **Whole-game recording (corpus-replay style).** BOTH sides' plies are
/// recorded into the replay buffer: the trainer's sampled moves and the
/// opponent's (Stockfish/Sloppy) moves alike, each with the mover's move
/// as the policy target and the terminal outcome signed by the mover's
/// colour via the standard two-sided `ActiveGame.flush`. Opponent rows
/// are advantage-weighted imitation — exactly the mechanism
/// `CorpusReplayFeeder` uses with human corpus games, and deliberately
/// so: against a stronger opponent the imitation signal (learning the
/// moves that beat you) is the richest data in the game. Whole-game
/// recording also keeps the buffer's adjacent-ply history reconstruction
/// valid, so history input encodings work normally. No forward pass is
/// spent on opponent moves at play time; like corpus rows, they are
/// evaluated by the trainer's forward pass at training time.
///
/// **Live weights.** The `network` handed in is the live trainer network
/// — it is being updated concurrently by the trainer loop. That is the
/// intended design (mildly off-policy drift within a game, identical to
/// what self-play/corpus-replay already accept). This driver never
/// mutates the network; it only calls `evaluateBatched`.
///
/// **Serial contract.** K is small (a handful of engines), so the per-
/// tick CPU work (encode / sample / apply) is serial — no `withTaskGroup`
/// pointer gymnastics. The single batched GPU forward is the only place
/// concurrency matters, and the opponent moves are the only cross-task
/// work (delivered through `eventBox`).
final class TrainVsUciDriver: @unchecked Sendable {

    // MARK: - Public config types

    /// One external opponent instance: its (not-yet-launched) arbiter,
    /// a `kind` for per-kind stats aggregation (e.g. `"stockfish"`), and
    /// a unique `instanceLabel` (e.g. `"stockfish#2"`).
    struct Opponent: Sendable {
        let arbiter: UCIArbiter
        let kind: String
        let instanceLabel: String
    }

    /// Per-slot running tallies, snapshotted for the periodic stats line.
    struct SlotStats: Sendable {
        let kind: String
        let instanceLabel: String
        var gamesCompleted: Int = 0
        var pliesPlayed: Int = 0
        var trainerWins: Int = 0
        var trainerLosses: Int = 0
        var draws: Int = 0
        /// Games abandoned without a flush (opponent crash / illegal move).
        var aborted: Int = 0
        /// Games dropped for exceeding the ply cap without terminating
        /// (unknown outcome — not flushed).
        var capDropped: Int = 0
    }

    // MARK: - Dependencies

    private let network: ChessMPSNetwork
    private let buffer: ReplayBuffer
    private let opponents: [Opponent]
    private let schedule: SamplingSchedule
    private let maxPliesPerGame: Int

    // MARK: - Cross-task delivery

    /// Opponent-move / lifecycle results pushed by async Tasks and
    /// drained on the driver task each tick. The only shared state
    /// between the driver task and its child Tasks.
    private let eventBox = SyncBox<[SlotEvent]>([])

    /// Per-slot stats, updated on the driver task at game-end and read
    /// (via `statsSnapshot()`) by the periodic-stats emitter on another
    /// task — hence lock-protected.
    private let statsBox: SyncBox<[SlotStats]>

    // MARK: - Driver-task-owned state

    private var slots: [Slot] = []
    private var scratches: Scratches?

    // MARK: - Init

    init(
        network: ChessMPSNetwork,
        buffer: ReplayBuffer,
        opponents: [Opponent],
        schedule: SamplingSchedule,
        maxPliesPerGame: Int
    ) {
        self.network = network
        self.buffer = buffer
        self.opponents = opponents
        self.schedule = schedule
        self.maxPliesPerGame = max(1, maxPliesPerGame)
        self.statsBox = SyncBox(opponents.map {
            SlotStats(kind: $0.kind, instanceLabel: $0.instanceLabel)
        })
    }

    /// Snapshot of per-slot stats for the periodic `[VS-UCI-STATS]` line.
    func statsSnapshot() -> [SlotStats] {
        statsBox.value
    }

    // MARK: - Run loop

    /// Launch + handshake every opponent, then pump games until the task
    /// is cancelled. Shuts every engine down on exit.
    func run() async {
        SessionLogger.shared.log("[VS-UCI] driver starting, opponents=\(opponents.count)")

        // 1. Launch + handshake all engines concurrently. A slot whose
        //    engine fails to come up is dropped (its stats stay zeroed).
        var readySlots: [Slot] = []
        await withTaskGroup(of: (Int, Bool).self) { group in
            for (i, opp) in opponents.enumerated() {
                group.addTask {
                    do {
                        try await opp.arbiter.launch()
                        try await opp.arbiter.handshake()
                        return (i, true)
                    } catch {
                        SessionLogger.shared.log(
                            "[VS-UCI] \(opp.instanceLabel) failed to start: \(error.localizedDescription)")
                        // launch() may have succeeded before handshake()
                        // threw (e.g. a process that starts but never sends
                        // uciok) — tear it down so no engine process / reader
                        // task is orphaned. shutdown() is a safe no-op if the
                        // process never launched.
                        await opp.arbiter.shutdown()
                        return (i, false)
                    }
                }
            }
            var upFlags = [Bool](repeating: false, count: opponents.count)
            for await (i, ok) in group { upFlags[i] = ok }
            for (i, ok) in upFlags.enumerated() where ok {
                readySlots.append(Slot(
                    index: i,
                    opponent: opponents[i],
                    network: network,
                    capPlies: maxPliesPerGame,
                    schedule: schedule
                ))
            }
        }

        guard !readySlots.isEmpty else {
            SessionLogger.shared.log("[VS-UCI] no engines started; driver exiting")
            return
        }
        slots = readySlots

        // Scratches sized to the live slot count.
        let boardFloats = BoardEncoder.tensorLength(for: network.inputEncoding)
        let sc = Scratches(capK: slots.count, boardFloats: boardFloats)
        scratches = sc
        defer {
            sc.deallocate()
            scratches = nil
        }

        // 2. Begin game #1 on each ready slot. Handshake already put the
        //    engine in a fresh state, so no `ucinewgame` needed for the
        //    first game — the slot starts `.idle`.
        for slot in slots {
            beginFirstGame(slot)
        }

        // 3. Tick until cancelled.
        while !Task.isCancelled {
            let didWork = await runOneTick(boardFloats: boardFloats)
            if !didWork {
                // Everything is awaiting an opponent / preparing — yield
                // briefly instead of busy-spinning.
                try? await Task.sleep(for: .milliseconds(2))
            }
        }

        SessionLogger.shared.log("[VS-UCI] driver exiting; shutting down engines")
        for slot in slots {
            await slot.opponent.arbiter.shutdown()
        }
    }

    // MARK: - One tick

    /// Returns true if the tick did any productive work (drained an
    /// opponent event or applied at least one trainer move).
    private func runOneTick(boardFloats: Int) async -> Bool {
        // (1) Drain opponent / lifecycle events.
        let events = eventBox.mutate { pending -> [SlotEvent] in
            let drained = pending
            pending.removeAll(keepingCapacity: true)
            return drained
        }
        for event in events {
            applyEvent(event)
        }

        // (2) Trainer-move batch: every slot where it is the trainer's
        //     turn and the slot is idle & non-terminal & has legal moves.
        var evalSlots: [Int] = []
        for (idx, slot) in slots.enumerated() {
            guard slot.phase == .idle, slot.game.engine.result == nil else { continue }
            guard slot.game.engine.state.currentPlayer == slot.trainerColor else { continue }
            guard !slot.game.engine.currentLegalMoves.isEmpty else { continue }
            // Stop at the game-length cap: no further plies are recorded
            // (the per-side scratch is sized `(cap+1)/2`) and the game-end
            // pass drops the game once this trips.
            guard slot.game.engine.moveHistory.count < slot.game.maxPliesCap else { continue }
            evalSlots.append(idx)
        }
        if !evalSlots.isEmpty {
            await evaluateAndApplyTrainerMoves(evalSlots, boardFloats: boardFloats)
        }

        // (3) Game-end pass: flush naturally-finished games; drop games
        //     that hit the length cap (their true outcome is unknown, so
        //     we do NOT inject a fake-draw label — same stance as
        //     self-play). Recycle both onto the next game.
        for slot in slots {
            guard slot.phase == .idle else { continue }
            if let result = slot.game.engine.result {
                finishGame(slot, result: result)
            } else if slot.game.engine.moveHistory.count >= slot.game.maxPliesCap {
                dropCappedGame(slot)
            }
        }

        // (4) Opponent-dispatch pass: every idle, non-terminal slot where
        //     it is now the opponent's turn gets an async `bestMove`.
        for slot in slots {
            guard slot.phase == .idle, slot.game.engine.result == nil else { continue }
            guard slot.game.engine.state.currentPlayer == slot.trainerColor.opposite else { continue }
            dispatchOpponentMove(slot)
        }

        return !events.isEmpty || !evalSlots.isEmpty
    }

    /// Encode the eval-set boards into a compacted batch, run one GPU
    /// forward, then sample + apply + record the trainer's move for each.
    private func evaluateAndApplyTrainerMoves(_ evalSlots: [Int], boardFloats: Int) async {
        guard let sc = scratches else { return }
        let encoding = network.inputEncoding
        let policySize = ChessNetwork.policySize
        let count = evalSlots.count

        // Encode into the compacted tick scratch.
        for (compact, idx) in evalSlots.enumerated() {
            let g = slots[idx].game
            let dst = sc.tickScratch + compact * boardFloats
            BoardEncoder.encode(
                g.engine.state,
                history: g.engine.recentStates,
                into: UnsafeMutableBufferPointer(start: dst, count: boardFloats),
                encoding: encoding
            )
        }

        // One batched GPU forward over the compacted sub-batch. Only the
        // policy readback is needed for sampling; value / WDL are ignored.
        let policyTarget = PointerCarrier(pointer: sc.policyScratch)
        do {
            try await network.evaluateBatched(
                batchBoardsPointer: UnsafePointer(sc.tickScratch),
                floatCount: count * boardFloats,
                count: count
            ) { policyBuf, _, _ in
                guard let pBase = policyBuf.baseAddress else { return }
                policyTarget.pointer.update(from: pBase, count: count * policySize)
            }
        } catch {
            SessionLogger.shared.log("[VS-UCI] trainer network error: \(error); skipping tick batch")
            return
        }

        // Sample + apply + record each trainer move (serial — K is small).
        for (compact, idx) in evalSlots.enumerated() {
            let slot = slots[idx]
            let g = slot.game
            let legal = g.engine.currentLegalMoves
            if legal.isEmpty { continue }
            let sideToMove = g.engine.state.currentPlayer

            let policySlice = UnsafeBufferPointer<Float>(
                start: sc.policyScratch + compact * policySize, count: policySize)
            let probsSlice = UnsafeMutableBufferPointer<Float>(
                start: sc.samplerProbsScratch + compact * MoveSampler.scratchCapacity,
                count: MoveSampler.scratchCapacity)
            let etaSlice = UnsafeMutableBufferPointer<Float>(
                start: sc.samplerEtaScratch + compact * MoveSampler.scratchCapacity,
                count: MoveSampler.scratchCapacity)

            // Game-total half-move index of the position being played.
            // Must be the true move count (both sides), NOT
            // `totalPliesPlayed` — that counts only the trainer's
            // recorded plies here, so it would feed the tau schedule and
            // Dirichlet gate a half-speed / colour-shifted ply (trainer-
            // as-Black would sample its first move as ply 0, not ply 1).
            let currentHalfMove = g.engine.moveHistory.count
            let result = MoveSampler.sampleMove(
                logits: policySlice,
                legalMoves: legal,
                currentPlayer: sideToMove,
                ply: currentHalfMove,
                schedule: g.schedule,
                probsScratch: probsSlice,
                etaScratch: etaSlice
            )

            var matCount = 0
            for sq in g.engine.state.board {
                if let piece = sq, piece.type != .pawn { matCount += 1 }
            }
            let materialCount = UInt8(min(matCount, Int(UInt8.max)))
            let plyTau = g.schedule.tau(forPly: currentHalfMove)

            // Record ONLY the trainer's ply (on-policy, outcome-only).
            g.recordPly(
                side: sideToMove,
                encodedBoardSrc: UnsafePointer(sc.tickScratch + compact * boardFloats),
                policyIndex: result.policyIndex,
                samplingTau: plyTau,
                materialCount: materialCount
            )
            do {
                try g.engine.applyMoveAndAdvance(result.move)
            } catch {
                SessionLogger.shared.log("[VS-UCI] trainer applyMove threw on slot \(slot.index): \(error)")
            }
        }
    }

    /// Record the opponent's about-to-be-played ply: encode the current
    /// position, map the move into the mover-relative policy frame, and
    /// stage it on the slot's `ActiveGame` — the same per-ply recipe
    /// `CorpusReplayFeeder` uses for corpus moves. CPU-only; no forward
    /// pass (like corpus rows, opponent rows are evaluated by the
    /// trainer's forward pass at training time). Serial on the driver
    /// task, so one shared encode scratch suffices.
    private func recordOpponentPly(_ slot: Slot, move: ChessMove) {
        guard let sc = scratches else { return }
        let g = slot.game
        // Cap guard mirrors the trainer-side eval gate: never record past
        // the per-side staging capacity. (Unreachable in practice — a
        // capped game is dropped before its next opponent dispatch.)
        guard g.engine.moveHistory.count < g.maxPliesCap else { return }

        BoardEncoder.encode(
            g.engine.state,
            history: g.engine.recentStates,
            into: UnsafeMutableBufferPointer(start: sc.opponentEncodeScratch, count: sc.boardFloats),
            encoding: network.inputEncoding
        )
        let sideToMove = g.engine.state.currentPlayer
        let halfMove = g.engine.moveHistory.count
        var matCount = 0
        for sq in g.engine.state.board {
            if let piece = sq, piece.type != .pawn { matCount += 1 }
        }
        g.recordPly(
            side: sideToMove,
            encodedBoardSrc: UnsafePointer(sc.opponentEncodeScratch),
            policyIndex: PolicyEncoding.policyIndex(move, currentPlayer: sideToMove),
            samplingTau: g.schedule.tau(forPly: halfMove),
            materialCount: UInt8(min(matCount, Int(UInt8.max)))
        )
    }

    // MARK: - Opponent dispatch

    /// Fire the opponent's `bestMove` on a background Task. The Task only
    /// touches the (Sendable) arbiter and a `[String]` move snapshot, and
    /// reports back via `eventBox` — it never touches driver/game state.
    private func dispatchOpponentMove(_ slot: Slot) {
        slot.phase = .awaitingOpponent
        let idx = slot.index
        let arbiter = slot.opponent.arbiter
        let moves = slot.game.engine.moveHistory.map { $0.uci }
        let box = eventBox
        Task {
            do {
                let best = try await arbiter.bestMove(startFEN: nil, moves: moves)
                box.modify { $0.append(SlotEvent(slotIndex: idx, kind: .opponentMove(best))) }
            } catch {
                box.modify { $0.append(SlotEvent(slotIndex: idx, kind: .opponentFailed(error.localizedDescription))) }
            }
        }
    }

    // MARK: - Event application (driver task)

    private func applyEvent(_ event: SlotEvent) {
        let slot = slots[slotArrayIndex(for: event.slotIndex)]
        switch event.kind {
        case .opponentMove(let best):
            slot.phase = .idle
            switch best {
            case .move(let token):
                guard let move = ChessMove.parseUCI(token, legal: slot.game.engine.currentLegalMoves) else {
                    SessionLogger.shared.log(
                        "[VS-UCI] \(slot.opponent.instanceLabel) returned illegal/unparseable move '\(token)'; aborting game")
                    abortGame(slot)
                    return
                }
                // Record the opponent's ply (whole-game recording — the
                // opponent's move becomes an advantage-weighted imitation
                // target, exactly like a corpus game's moves), then apply.
                recordOpponentPly(slot, move: move)
                do {
                    try slot.game.engine.applyMoveAndAdvance(move)
                } catch {
                    SessionLogger.shared.log("[VS-UCI] opponent applyMove threw on slot \(slot.index): \(error)")
                    abortGame(slot)
                }
            case .null:
                // The engine says it has no move. If our engine already
                // sees a terminal result, the game-end pass handles it;
                // otherwise the two disagree — abort defensively.
                if slot.game.engine.result == nil {
                    SessionLogger.shared.log(
                        "[VS-UCI] \(slot.opponent.instanceLabel) returned no move in a non-terminal position; aborting game")
                    abortGame(slot)
                }
            }
        case .opponentFailed(let message):
            SessionLogger.shared.log("[VS-UCI] \(slot.opponent.instanceLabel) move failed: \(message); recovering engine")
            recordAbort(slot)
            recoverEngine(slot)
        case .prepared:
            slot.phase = .idle
        case .prepareFailed(let message):
            SessionLogger.shared.log("[VS-UCI] \(slot.opponent.instanceLabel) prepare failed: \(message); retrying")
            recoverEngine(slot)
        }
    }

    // MARK: - Game lifecycle

    private func beginFirstGame(_ slot: Slot) {
        slot.gameIndex = 0
        slot.trainerColor = trainerColor(forGameIndex: 0)
        slot.game.resetForNewGame(maxPliesCap: maxPliesPerGame, schedule: schedule)
        slot.phase = .idle
    }

    /// Flush a naturally-finished game (both sides' plies, outcomes
    /// signed by mover colour — the standard two-sided path), tally
    /// stats, then recycle the slot.
    private func finishGame(_ slot: Slot, result: GameResult) {
        let totalPlies = slot.game.engine.moveHistory.count
        _ = slot.game.flush(buffer: buffer, result: result)

        let trainerScore = Self.trainerOutcome(result: result, trainerColor: slot.trainerColor)
        let stableIndex = slot.index
        statsBox.modify { all in
            all[stableIndex].gamesCompleted += 1
            all[stableIndex].pliesPlayed += totalPlies
            if trainerScore > 0 { all[stableIndex].trainerWins += 1 }
            else if trainerScore < 0 { all[stableIndex].trainerLosses += 1 }
            else { all[stableIndex].draws += 1 }
        }

        recycleSlot(slot)
    }

    /// Abandon the in-progress game WITHOUT flushing (illegal move / bad
    /// engine state), tally it as aborted, and recycle the slot.
    private func abortGame(_ slot: Slot) {
        recordAbort(slot)
        recycleSlot(slot)
    }

    /// The game ran past the length cap without terminating. Drop it
    /// without flushing (unknown outcome), tally it, and recycle.
    private func dropCappedGame(_ slot: Slot) {
        let stableIndex = slot.index
        statsBox.modify { $0[stableIndex].capDropped += 1 }
        recycleSlot(slot)
    }

    private func recordAbort(_ slot: Slot) {
        let stableIndex = slot.index
        statsBox.modify { $0[stableIndex].aborted += 1 }
    }

    /// Start the next game on a slot whose engine is alive: reset the
    /// game, alternate the trainer's colour, and send `ucinewgame`
    /// (async) before the game plays — so the slot waits in `.preparing`.
    private func recycleSlot(_ slot: Slot) {
        slot.gameIndex += 1
        slot.trainerColor = trainerColor(forGameIndex: slot.gameIndex)
        slot.game.resetForNewGame(maxPliesCap: maxPliesPerGame, schedule: schedule)
        slot.phase = .preparing
        let idx = slot.index
        let arbiter = slot.opponent.arbiter
        let box = eventBox
        Task {
            do {
                try await arbiter.startNewGame()
                box.modify { $0.append(SlotEvent(slotIndex: idx, kind: .prepared)) }
            } catch {
                box.modify { $0.append(SlotEvent(slotIndex: idx, kind: .prepareFailed(error.localizedDescription))) }
            }
        }
    }

    /// Recover a crashed/wedged engine: relaunch + re-handshake +
    /// `ucinewgame` on a background Task, then resume with a fresh game.
    /// The game state was already reset by the caller path
    /// (`recycleSlot` on prepareFailed, or a fresh reset here otherwise).
    private func recoverEngine(_ slot: Slot) {
        slot.gameIndex += 1
        slot.trainerColor = trainerColor(forGameIndex: slot.gameIndex)
        slot.game.resetForNewGame(maxPliesCap: maxPliesPerGame, schedule: schedule)
        slot.phase = .preparing
        let idx = slot.index
        let arbiter = slot.opponent.arbiter
        let box = eventBox
        Task {
            await arbiter.shutdown()
            // Backoff so a permanently-broken engine (e.g. a bad path)
            // can't hot-loop relaunch/fail on this slot.
            try? await Task.sleep(for: .seconds(1))
            do {
                try await arbiter.launch()
                try await arbiter.handshake()
                try await arbiter.startNewGame()
                box.modify { $0.append(SlotEvent(slotIndex: idx, kind: .prepared)) }
            } catch {
                box.modify { $0.append(SlotEvent(slotIndex: idx, kind: .prepareFailed(error.localizedDescription))) }
            }
        }
    }

    // MARK: - Helpers

    /// Even game index → trainer plays White; odd → Black. Alternating
    /// colour per game keeps the outcome labels unbiased.
    private func trainerColor(forGameIndex gameIndex: Int) -> PieceColor {
        gameIndex % 2 == 0 ? .white : .black
    }

    /// Trainer's game result as +1 (win) / -1 (loss) / 0 (draw).
    static func trainerOutcome(result: GameResult, trainerColor: PieceColor) -> Int {
        switch result {
        case .checkmate(let winner):
            return winner == trainerColor ? 1 : -1
        case .stalemate, .drawByFiftyMoveRule, .drawByInsufficientMaterial, .drawByThreefoldRepetition:
            return 0
        }
    }

    /// Map a stable slot `index` (== the opponent's position in the
    /// original `opponents` array) to its position in `slots` (which may
    /// be shorter if some engines failed to launch).
    private func slotArrayIndex(for stableIndex: Int) -> Int {
        // slots is tiny; a linear scan is cheaper than a dictionary and
        // avoids extra state. slots preserve ascending `index` order.
        for (arrayIdx, slot) in slots.enumerated() where slot.index == stableIndex {
            return arrayIdx
        }
        // Unreachable: every event carries a live slot's index.
        return 0
    }
}

// MARK: - Slot

/// Driver-task-owned per-slot state. A `final class` so the driver can
/// mutate `phase` / `trainerColor` in place across the tick passes.
private final class Slot {
    let index: Int
    let opponent: TrainVsUciDriver.Opponent
    let game: ActiveGame
    var phase: SlotPhase = .idle
    var trainerColor: PieceColor = .white
    var gameIndex: Int = 0

    init(
        index: Int,
        opponent: TrainVsUciDriver.Opponent,
        network: ChessMPSNetwork,
        capPlies: Int,
        schedule: SamplingSchedule
    ) {
        self.index = index
        self.opponent = opponent
        // Both network refs are the trainer network; the driver evaluates
        // via its own `network` and never reads these, but ActiveGame
        // requires them for its scratch sizing (encoding width).
        self.game = ActiveGame(
            workerId: UInt16(truncatingIfNeeded: index),
            whiteNetwork: network,
            blackNetwork: network,
            capPlies: capPlies,
            schedule: schedule
        )
    }
}

/// Where a slot is in its per-ply lifecycle.
private enum SlotPhase {
    /// No async op pending; the tick decides what to do from the engine's
    /// side-to-move and result.
    case idle
    /// An opponent `bestMove` Task is in flight.
    case awaitingOpponent
    /// A `ucinewgame` / relaunch Task is in flight; the slot is not
    /// playable until it reports `.prepared`.
    case preparing
}

// MARK: - Cross-task event

/// Reported by an async Task back to the driver task via `eventBox`.
private struct SlotEvent: Sendable {
    let slotIndex: Int
    let kind: Kind
    enum Kind: Sendable {
        case opponentMove(UCIBestMove)
        case opponentFailed(String)
        case prepared
        case prepareFailed(String)
    }
}

// MARK: - Scratches

/// Heap scratch for the trainer sub-batch, sized to the slot count.
/// Single-network (unlike the arena's two-network scratches): there is
/// only the trainer to batch — the opponent's move comes from a process.
private final class Scratches: @unchecked Sendable {
    let capK: Int
    let boardFloats: Int
    let tickScratch: UnsafeMutablePointer<Float>        // capK * boardFloats
    let policyScratch: UnsafeMutablePointer<Float>      // capK * policySize
    let samplerProbsScratch: UnsafeMutablePointer<Float> // capK * scratchCapacity
    let samplerEtaScratch: UnsafeMutablePointer<Float>   // capK * scratchCapacity
    /// One full-width encode target for `recordOpponentPly` — serial use
    /// on the driver task, so a single buffer covers every slot.
    let opponentEncodeScratch: UnsafeMutablePointer<Float> // boardFloats

    init(capK: Int, boardFloats: Int) {
        precondition(capK >= 1, "Scratches.init: capK must be >= 1")
        self.capK = capK
        self.boardFloats = boardFloats
        let policySize = ChessNetwork.policySize
        let scratchCap = MoveSampler.scratchCapacity
        self.tickScratch = Self.alloc(capK * boardFloats)
        self.policyScratch = Self.alloc(capK * policySize)
        self.samplerProbsScratch = Self.alloc(capK * scratchCap)
        self.samplerEtaScratch = Self.alloc(capK * scratchCap)
        self.opponentEncodeScratch = Self.alloc(boardFloats)
    }

    private static func alloc(_ count: Int) -> UnsafeMutablePointer<Float> {
        let p = UnsafeMutablePointer<Float>.allocate(capacity: count)
        p.initialize(repeating: 0, count: count)
        return p
    }

    func deallocate() {
        let policySize = ChessNetwork.policySize
        let scratchCap = MoveSampler.scratchCapacity
        tickScratch.deinitialize(count: capK * boardFloats); tickScratch.deallocate()
        policyScratch.deinitialize(count: capK * policySize); policyScratch.deallocate()
        samplerProbsScratch.deinitialize(count: capK * scratchCap); samplerProbsScratch.deallocate()
        samplerEtaScratch.deinitialize(count: capK * scratchCap); samplerEtaScratch.deallocate()
        opponentEncodeScratch.deinitialize(count: boardFloats); opponentEncodeScratch.deallocate()
    }
}

/// `@unchecked Sendable` shim so a driver-owned `UnsafeMutablePointer`
/// can be captured by the `@Sendable` `evaluateBatched` consume closure
/// (which runs synchronously and completes before the await returns).
/// Mirrors the same shim in the other tick drivers.
private struct PointerCarrier: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
}
