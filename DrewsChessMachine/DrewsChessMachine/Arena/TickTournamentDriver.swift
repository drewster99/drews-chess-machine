import Foundation

// MARK: - Tick-Based Tournament Driver

/// Tournament driver for arena play, using the same tick-based
/// topology as `BatchedSelfPlayDriver`: one driver Task pumps K
/// active games in lockstep with parallel encode + parallel sample
/// across P CPU workers, and two batched `network.evaluateBatched`
/// calls per tick — one per network (candidate / champion) —
/// instead of K Swift tasks each parking in an actor barrier.
///
/// **Public contract** mirrors `TournamentDriver.run` so the call
/// site in `SessionController+Arena.swift` can branch on the
/// `arenaUseTickDriver` flag and feed either driver into the same
/// downstream `TournamentStats` consumer. The signature differs in
/// taking `(candidateNetwork:, championNetwork:)` directly rather
/// than `(playerA:, playerB:)` factories — the tick driver doesn't
/// allocate per-game player objects; it owns the per-game state on
/// an `ActiveGame` and calls `MoveSampler` inline.
///
/// **K behavior.** Initial K = `min(concurrency, games)`. As games
/// finish, the slot is recycled for the next gameIndex (alternating
/// candidate-color: even gameIndex → candidate is white, odd → black)
/// until all `games` games have been spawned. Past that point the
/// slot count shrinks as the remaining in-flight games complete.
/// Driver exits when K reaches 0.
///
/// **Cancellation.** Checked at top of each tick body — same gate as
/// the legacy driver's `Task.isCancelled || isCancelled?() == true`.
/// On cancel, the driver finishes the current tick (so we don't
/// abandon in-progress GPU work mid-call) and exits. In-flight games'
/// `TournamentGameRecord`s are NOT emitted for unfinished games —
/// `gamesPlayed` will be less than the requested total.
///
/// **No replay-buffer ingest.** Arena games don't feed the trainer.
/// Each completed game's `TournamentGameRecord` is emitted via the
/// `onGameRecorded` callback for the post-arena validity sweep, and
/// outcomes are tallied locally into `TournamentStats`.
///
/// **Per-side outcome attribution.** Player A is the candidate;
/// player B is the champion. Tallies are kept from A's perspective.
/// For each game we know `aIsWhite` (== `gameIndex % 2 == 0`), so we
/// can decode `result` into A-won / A-lost / draw and bucket by side.
///
/// **GameWatcher live display is not wired here.** Arena's UX
/// surfaces the per-game progress through `onGameCompleted` (chip
/// + countdown), not a watcher board. Matches the legacy driver.
final class TickTournamentDriver: @unchecked Sendable {

    // MARK: - Public API

    func run(
        candidateNetwork: ChessMPSNetwork,
        championNetwork: ChessMPSNetwork,
        arenaSchedule: SamplingSchedule,
        games totalGames: Int,
        concurrency: Int = 1,
        diversityTracker: GameDiversityTracker? = nil,
        isCancelled: (@Sendable () -> Bool)? = nil,
        onGameCompleted: (@Sendable (Int, Int, Int, Int) -> Void)? = nil,
        onGameRecorded: (@Sendable (TournamentGameRecord) -> Void)? = nil
    ) async throws -> TournamentStats {
        guard totalGames > 0 else {
            return TournamentStats(
                gamesPlayed: 0,
                playerAWins: 0, playerBWins: 0, draws: 0,
                playerAWinsAsWhite: 0, playerAWinsAsBlack: 0,
                playerALossesAsWhite: 0, playerALossesAsBlack: 0,
                playerADrawsAsWhite: 0, playerADrawsAsBlack: 0
            )
        }
        let initialK = max(1, min(concurrency, totalGames))
        let P = max(1, ProcessInfo.processInfo.activeProcessorCount)
        SessionLogger.shared.log(
            "[ARENA-TICK] driver starting, totalGames=\(totalGames) initialK=\(initialK) P=\(P)"
        )

        // Allocate scratches sized to initialK (arena K never grows;
        // it only shrinks as the pool retires). Two encode + two
        // policy scratches because the partition by network produces
        // up to two disjoint sub-batches per tick.
        let scratches = TickArenaScratches(capK: initialK)
        defer { scratches.deallocate() }

        // Per-tick partition working storage. Sized to capK so the
        // partition pass needs no per-tick allocation.
        var candIndices = [Int]()
        var champIndices = [Int]()
        candIndices.reserveCapacity(initialK)
        champIndices.reserveCapacity(initialK)
        // Inverse map: globalIdx → compactIdx within its network's
        // sub-batch this tick. Sized to capK.
        var compactIdx = [Int](repeating: 0, count: initialK)

        // The K active games. Each holds its (whiteNetwork, blackNetwork)
        // assignment for the entire game (set when the slot is reset
        // to a new gameIndex). `gameIndex` and `aIsWhite` (true when
        // candidate is white this game) are tracked separately so
        // outcome attribution survives slot recycling.
        var games: [ActiveGame] = []
        var gameIndices: [Int] = []      // current gameIndex per slot
        var aIsWhiteForSlot: [Bool] = [] // candidate-is-white flag per slot
        games.reserveCapacity(initialK)
        gameIndices.reserveCapacity(initialK)
        aIsWhiteForSlot.reserveCapacity(initialK)

        // Tally accumulators (driver-task-owned; no locking).
        var aWins = 0
        var bWins = 0
        var draws = 0
        var aWinsAsWhite = 0
        var aWinsAsBlack = 0
        var aLossesAsWhite = 0
        var aLossesAsBlack = 0
        var aDrawsAsWhite = 0
        var aDrawsAsBlack = 0
        var completed = 0
        var nextGameIndexToSpawn = 0

        // Initial slot fan-out. Each slot's ActiveGame is initialized
        // with the right (whiteNetwork, blackNetwork) assignment for
        // its gameIndex.
        //
        // `arenaCapPlies` does double duty: (1) sizes the per-side
        // staging scratches (whiteBoardScratch etc., capPlies/2 + 1
        // plies each); (2) serves as the "max plies, otherwise treat
        // as draw" cap. Realistic chess games end well under 300
        // plies via 50-move-rule / 3-fold-repetition / normal
        // termination, so 1024 is effectively "no cap" while keeping
        // the arithmetic well clear of `Int.max` overflow in
        // `ActiveGame.resetForNewGame`'s `(newCap + 1) / 2` step.
        // Per-game staging at 1024 plies: ~3.95 MB per side, ~7.9 MB
        // total per ActiveGame. At K=400 arena concurrency: ~3.2 GB
        // peak. Acceptable.
        let arenaCapPlies = 1024
        for i in 0..<initialK {
            let candIsWhite = (i % 2 == 0)
            let (wNet, bNet) = candIsWhite
                ? (candidateNetwork, championNetwork)
                : (championNetwork, candidateNetwork)
            let g = ActiveGame(
                workerId: UInt16(truncatingIfNeeded: i),
                whiteNetwork: wNet,
                blackNetwork: bNet,
                capPlies: arenaCapPlies,
                schedule: arenaSchedule
            )
            g.resetForNewGame(maxPliesCap: arenaCapPlies, schedule: arenaSchedule)
            games.append(g)
            gameIndices.append(i)
            aIsWhiteForSlot.append(candIsWhite)
            nextGameIndexToSpawn += 1
        }

        // Main loop: tick until all games complete or cancellation.
        // No pause/resume in arena (arenas are atomic from the
        // outer scheduler's POV).
        while !games.isEmpty {
            if Task.isCancelled || (isCancelled?() == true) {
                SessionLogger.shared.log(
                    "[ARENA-TICK] cancellation: dropping \(games.count) in-flight games"
                )
                break
            }

            try await runOneTick(
                games: games,
                P: P,
                candidateNetwork: candidateNetwork,
                championNetwork: championNetwork,
                scratches: scratches,
                candIndices: &candIndices,
                champIndices: &champIndices,
                compactIdx: &compactIdx
            )

            // Game-end pass: serially walk slots; on completion,
            // either recycle the slot for the next gameIndex or
            // remove it from the active vector.
            //
            // Two completion paths:
            //   - natural: `engine.result != nil` (checkmate / stalemate / etc.)
            //   - max-plies-drop: `totalPliesPlayed >= maxPliesCap` and no
            //     natural result. Treat as a draw by `.stalemate` (the
            //     least-loaded synthetic outcome — most chess engines
            //     score "ran too long without termination" as a draw).
            //     Without this branch a stuck game would occupy its slot
            //     forever, looping through ticks with no progress.
            var i = 0
            while i < games.count {
                let g = games[i]
                let naturalResult = g.engine.result
                let hitMaxPlies = (naturalResult == nil)
                    && (g.totalPliesPlayed >= g.maxPliesCap)
                guard let result = naturalResult ?? (hitMaxPlies ? .stalemate : nil) else {
                    i += 1
                    continue
                }
                if hitMaxPlies {
                    SessionLogger.shared.log(
                        "[ARENA-TICK] slot \(i) gameIndex \(gameIndices[i]) hit max-plies cap \(g.maxPliesCap) — scoring as draw"
                    )
                }

                let gameIndex = gameIndices[i]
                let aIsWhite = aIsWhiteForSlot[i]
                let record = TournamentGameRecord(
                    gameIndex: gameIndex,
                    aIsWhite: aIsWhite,
                    result: result,
                    moveHistory: g.engine.moveHistory
                )

                completed += 1
                switch result {
                case .checkmate(let winner):
                    let aWon = (winner == .white && aIsWhite)
                        || (winner == .black && !aIsWhite)
                    if aWon {
                        aWins += 1
                        if aIsWhite { aWinsAsWhite += 1 } else { aWinsAsBlack += 1 }
                    } else {
                        bWins += 1
                        if aIsWhite { aLossesAsWhite += 1 } else { aLossesAsBlack += 1 }
                    }
                case .stalemate,
                     .drawByFiftyMoveRule,
                     .drawByInsufficientMaterial,
                     .drawByThreefoldRepetition:
                    draws += 1
                    if aIsWhite { aDrawsAsWhite += 1 } else { aDrawsAsBlack += 1 }
                }

                diversityTracker?.recordGame(moves: record.moveHistory)
                onGameRecorded?(record)
                onGameCompleted?(completed, aWins, bWins, draws)

                // Recycle slot for next game, or retire it.
                if nextGameIndexToSpawn < totalGames {
                    let nextIdx = nextGameIndexToSpawn
                    let candIsWhiteNext = (nextIdx % 2 == 0)
                    // Slot's (whiteNetwork, blackNetwork) pair is
                    // immutable on ActiveGame (let-bound). Allocate
                    // a fresh ActiveGame with the right pairing for
                    // the next game and drop the old one. The old
                    // ActiveGame's allocations are freed on its
                    // dealloc — net allocation churn is one
                    // ActiveGame per game (≈1.2 MB at capPlies=512;
                    // not great, but matches the legacy `MPSChessPlayer`
                    // churn pattern which also re-allocated per
                    // game via the playerA/playerB factories).
                    let (wNet, bNet) = candIsWhiteNext
                        ? (candidateNetwork, championNetwork)
                        : (championNetwork, candidateNetwork)
                    let newG = ActiveGame(
                        workerId: UInt16(truncatingIfNeeded: nextIdx),
                        whiteNetwork: wNet,
                        blackNetwork: bNet,
                        capPlies: arenaCapPlies,
                        schedule: arenaSchedule
                    )
                    newG.resetForNewGame(maxPliesCap: arenaCapPlies, schedule: arenaSchedule)
                    games[i] = newG
                    gameIndices[i] = nextIdx
                    aIsWhiteForSlot[i] = candIsWhiteNext
                    nextGameIndexToSpawn += 1
                    i += 1
                } else {
                    // No more games to spawn — this slot retires.
                    games.remove(at: i)
                    gameIndices.remove(at: i)
                    aIsWhiteForSlot.remove(at: i)
                    // Do NOT advance i — we just shifted the next
                    // slot into position i.
                }
            }
        }

        SessionLogger.shared.log(
            "[ARENA-TICK] driver done, completed=\(completed)/\(totalGames)"
        )

        return TournamentStats(
            gamesPlayed: aWins + bWins + draws,
            playerAWins: aWins,
            playerBWins: bWins,
            draws: draws,
            playerAWinsAsWhite: aWinsAsWhite,
            playerAWinsAsBlack: aWinsAsBlack,
            playerALossesAsWhite: aLossesAsWhite,
            playerALossesAsBlack: aLossesAsBlack,
            playerADrawsAsWhite: aDrawsAsWhite,
            playerADrawsAsBlack: aDrawsAsBlack
        )
    }

    // MARK: - Tick body

    private func runOneTick(
        games: [ActiveGame],
        P: Int,
        candidateNetwork: ChessMPSNetwork,
        championNetwork: ChessMPSNetwork,
        scratches: TickArenaScratches,
        candIndices: inout [Int],
        champIndices: inout [Int],
        compactIdx: inout [Int]
    ) async throws {
        let K = games.count
        let boardFloats = BoardEncoder.tensorLength
        let policySize = ChessNetwork.policySize

        // Partition K games by current-side network. Compute the
        // inverse compactIdx map so the parallel encode pass can
        // write game `i`'s board into its compact slot in the
        // appropriate per-network scratch.
        //
        // We compare ChessMPSNetwork by reference identity (`===`).
        // ChessMPSNetwork is a class; the two networks passed in are
        // distinct instances for arena. Self-play uses one instance
        // for both sides, but TickTournamentDriver is only used for
        // arena — there the two are always distinct.
        candIndices.removeAll(keepingCapacity: true)
        champIndices.removeAll(keepingCapacity: true)
        for i in 0..<K {
            let net = games[i].currentNetwork
            if net === candidateNetwork {
                compactIdx[i] = candIndices.count
                candIndices.append(i)
            } else {
                compactIdx[i] = champIndices.count
                champIndices.append(i)
            }
        }

        // (a) Parallel encode. Per game i: write encoded board into
        //     `(candScratch or champScratch) + compactIdx[i] * boardFloats`
        //     depending on which network owns this ply.
        let gameRefs = games
        let compactIdxSnap = compactIdx
        let candCarrier = ArenaPointerCarrier(pointer: scratches.candTickScratch)
        let champCarrier = ArenaPointerCarrier(pointer: scratches.champTickScratch)
        let candNetRef = NetworkRefCarrier(network: candidateNetwork)
        await withTaskGroup(of: Void.self) { group in
            for p in 0..<P {
                group.addTask {
                    var i = p
                    while i < K {
                        let g = gameRefs[i]
                        let net = g.currentNetwork
                        let dst = (net === candNetRef.network ? candCarrier.pointer : champCarrier.pointer)
                            + compactIdxSnap[i] * boardFloats
                        BoardEncoder.encode(
                            g.engine.state,
                            into: UnsafeMutableBufferPointer(start: dst, count: boardFloats)
                        )
                        i += P
                    }
                }
            }
        }

        // (b) Two batched GPU forwards — one per non-empty sub-batch.
        //     Each consume closure memcpy's the policy + value
        //     readback into the corresponding per-network policy /
        //     value scratch. Sequential calls (the network's
        //     `executionQueue` would serialize them anyway). For very
        //     small sub-batches the GPU pays a per-call overhead;
        //     skip empty calls.
        let candCount = candIndices.count
        if candCount > 0 {
            let policyTarget = ArenaPointerCarrier(pointer: scratches.candPolicyScratch)
            let valueTarget = ArenaPointerCarrier(pointer: scratches.candValueScratch)
            try await candidateNetwork.evaluateBatched(
                batchBoardsPointer: UnsafePointer(scratches.candTickScratch),
                floatCount: candCount * boardFloats,
                count: candCount
            ) { policyBuf, valueBuf in
                guard let pBase = policyBuf.baseAddress, let vBase = valueBuf.baseAddress else {
                    return
                }
                policyTarget.pointer.update(from: pBase, count: candCount * policySize)
                valueTarget.pointer.update(from: vBase, count: candCount)
            }
        }
        let champCount = champIndices.count
        if champCount > 0 {
            let policyTarget = ArenaPointerCarrier(pointer: scratches.champPolicyScratch)
            let valueTarget = ArenaPointerCarrier(pointer: scratches.champValueScratch)
            try await championNetwork.evaluateBatched(
                batchBoardsPointer: UnsafePointer(scratches.champTickScratch),
                floatCount: champCount * boardFloats,
                count: champCount
            ) { policyBuf, valueBuf in
                guard let pBase = policyBuf.baseAddress, let vBase = valueBuf.baseAddress else {
                    return
                }
                policyTarget.pointer.update(from: pBase, count: champCount * policySize)
                valueTarget.pointer.update(from: vBase, count: champCount)
            }
        }

        // (c) Parallel sample + apply. Per game i: pick its policy
        //     slice from the right per-network scratch (using
        //     compactIdx[i]), sample, apply, record.
        let candPolicyCarrier = ArenaPointerCarrier(pointer: scratches.candPolicyScratch)
        let champPolicyCarrier = ArenaPointerCarrier(pointer: scratches.champPolicyScratch)
        let probsCarrier = ArenaPointerCarrier(pointer: scratches.samplerProbsScratch)
        let etaCarrier = ArenaPointerCarrier(pointer: scratches.samplerEtaScratch)
        // Capture for closure
        let scratchCandCarrier = candCarrier
        let scratchChampCarrier = champCarrier
        await withTaskGroup(of: Void.self) { group in
            for p in 0..<P {
                group.addTask {
                    var i = p
                    while i < K {
                        let g = gameRefs[i]
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
                        let net = g.currentNetwork
                        let isCand = (net === candNetRef.network)
                        let compact = compactIdxSnap[i]
                        let policySliceStart = (isCand ? candPolicyCarrier.pointer : champPolicyCarrier.pointer)
                            .advanced(by: compact * policySize)
                        let policySliceBuf = UnsafeBufferPointer<Float>(
                            start: policySliceStart, count: policySize
                        )
                        let probsSliceStart = probsCarrier.pointer
                            .advanced(by: i * MoveSampler.scratchCapacity)
                        let probsSliceBuf = UnsafeMutableBufferPointer<Float>(
                            start: probsSliceStart, count: MoveSampler.scratchCapacity
                        )
                        let etaSliceStart = etaCarrier.pointer
                            .advanced(by: i * MoveSampler.scratchCapacity)
                        let etaSliceBuf = UnsafeMutableBufferPointer<Float>(
                            start: etaSliceStart, count: MoveSampler.scratchCapacity
                        )

                        let gameTotalPly = g.totalPliesPlayed
                        let result = MoveSampler.sampleMove(
                            logits: policySliceBuf,
                            legalMoves: legalMoves,
                            currentPlayer: sideToMove,
                            ply: gameTotalPly,
                            schedule: g.schedule,
                            probsScratch: probsSliceBuf,
                            etaScratch: etaSliceBuf
                        )

                        // Material-count loop kept inline; same
                        // cheap 64-square iteration as self-play.
                        var matCount: Int = 0
                        for sq in g.engine.state.board {
                            if let piece = sq, piece.type != .pawn {
                                matCount += 1
                            }
                        }
                        let materialCount = UInt8(min(matCount, Int(UInt8.max)))

                        let plyTau = g.schedule.tau(forPly: gameTotalPly)
                        let encodedSrc = (isCand ? scratchCandCarrier.pointer : scratchChampCarrier.pointer)
                            + compact * boardFloats
                        g.recordPly(
                            side: sideToMove,
                            encodedBoardSrc: UnsafePointer(encodedSrc),
                            policyIndex: result.policyIndex,
                            samplingTau: plyTau,
                            materialCount: materialCount
                        )

                        do {
                            try g.engine.applyMoveAndAdvance(result.move)
                        } catch {
                            SessionLogger.shared.log(
                                "[ARENA-TICK] applyMoveAndAdvance threw on slot \(i): \(error)"
                            )
                        }
                        i += P
                    }
                }
            }
        }
    }
}

// MARK: - Per-tick scratches (per-network)

/// Heap-allocated scratch buffers for one arena tournament's worth
/// of ticks. Sized to capK (the initial slot count); arena K only
/// ever shrinks, so this allocation is one-shot per arena run.
/// Mirrors `BatchedSelfPlayDriver`'s tick scratches but duplicated
/// per network (candidate + champion) so the partition pass can
/// write into the right contiguous sub-batch.
private final class TickArenaScratches: @unchecked Sendable {
    let capK: Int
    let candTickScratch: UnsafeMutablePointer<Float>     // capK * boardFloats
    let champTickScratch: UnsafeMutablePointer<Float>    // capK * boardFloats
    let candPolicyScratch: UnsafeMutablePointer<Float>   // capK * policySize
    let champPolicyScratch: UnsafeMutablePointer<Float>  // capK * policySize
    let candValueScratch: UnsafeMutablePointer<Float>    // capK
    let champValueScratch: UnsafeMutablePointer<Float>   // capK
    let samplerProbsScratch: UnsafeMutablePointer<Float> // capK * MoveSampler.scratchCapacity
    let samplerEtaScratch: UnsafeMutablePointer<Float>   // capK * MoveSampler.scratchCapacity

    init(capK: Int) {
        precondition(capK >= 1, "TickArenaScratches.init: capK must be >= 1")
        self.capK = capK
        let boardFloats = BoardEncoder.tensorLength
        let policySize = ChessNetwork.policySize
        let scratchCap = MoveSampler.scratchCapacity

        self.candTickScratch    = Self.alloc(capK * boardFloats)
        self.champTickScratch   = Self.alloc(capK * boardFloats)
        self.candPolicyScratch  = Self.alloc(capK * policySize)
        self.champPolicyScratch = Self.alloc(capK * policySize)
        self.candValueScratch   = Self.alloc(capK)
        self.champValueScratch  = Self.alloc(capK)
        self.samplerProbsScratch = Self.alloc(capK * scratchCap)
        self.samplerEtaScratch   = Self.alloc(capK * scratchCap)
    }

    private static func alloc(_ count: Int) -> UnsafeMutablePointer<Float> {
        let p = UnsafeMutablePointer<Float>.allocate(capacity: count)
        p.initialize(repeating: 0, count: count)
        return p
    }

    func deallocate() {
        let boardFloats = BoardEncoder.tensorLength
        let policySize = ChessNetwork.policySize
        let scratchCap = MoveSampler.scratchCapacity
        candTickScratch.deinitialize(count: capK * boardFloats);   candTickScratch.deallocate()
        champTickScratch.deinitialize(count: capK * boardFloats);  champTickScratch.deallocate()
        candPolicyScratch.deinitialize(count: capK * policySize);  candPolicyScratch.deallocate()
        champPolicyScratch.deinitialize(count: capK * policySize); champPolicyScratch.deallocate()
        candValueScratch.deinitialize(count: capK);  candValueScratch.deallocate()
        champValueScratch.deinitialize(count: capK); champValueScratch.deallocate()
        samplerProbsScratch.deinitialize(count: capK * scratchCap); samplerProbsScratch.deallocate()
        samplerEtaScratch.deinitialize(count: capK * scratchCap);   samplerEtaScratch.deallocate()
    }
}

// MARK: - Sendable carrier for the network reference

/// `ChessMPSNetwork` is `@unchecked Sendable`, so a bare reference
/// crosses task boundaries fine. The carrier exists only so the
/// closures inside `runOneTick`'s task groups have a single
/// captured value to refer to (without having to repeat
/// `candidateNetwork === net` everywhere — clearer than a captured
/// raw reference, and matches the `ArenaPointerCarrier` carrier
/// pattern used for pointer state).
private struct NetworkRefCarrier: @unchecked Sendable {
    let network: ChessMPSNetwork
}

// MARK: - Sendable carrier for pointer state (file-private duplicate
// of the same shim in `BatchedSelfPlayDriver.swift`. Used inside the
// `runOneTick` task groups to satisfy Swift 6 strict-concurrency
// without exposing the carrier across files.)
private struct ArenaPointerCarrier: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
}
